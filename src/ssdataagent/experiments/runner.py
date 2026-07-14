from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ssdataagent.agent.context import Condition
from ssdataagent.agent.llm_client import build_client
from ssdataagent.config import REPO_ROOT, load_llm_config, results_root
from ssdataagent.data.event_timing import conditional_joint_repair, event_timing_variables
from ssdataagent.data.loader import load_disjoint_train, load_real_data
from ssdataagent.data.schema import load_schema
from ssdataagent.data.splitter import split_train_eval
from ssdataagent.data.transfer import TRANSFER_PAIRS, compute_crosswalk, load_source_wave
from ssdataagent.evaluation.overdetermination import overdetermination
from ssdataagent.evaluation.runner import PassRates, by_domain, run_evaluation
from ssdataagent.experiments.conditions import get_condition
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.registry import get_strategy


@dataclass
class ExperimentConfig:
    name: str
    datasets: list[str]
    conditions: list[str]
    max_iterations: int
    sandbox_timeout: int
    n_rows: int
    # Fraction of the benchmark sample kept for fitting, the rest becoming the
    # eval reference. Ignored — and unnecessary — when disjoint_train_sample is
    # set, since a disjoint pool means we no longer have to cut the reference in
    # half to obtain held-out training rows.
    train_eval_split: float = 0.5
    results_root: Path = field(default_factory=results_root)
    unseen_variables: dict[str, list[str]] = field(default_factory=dict)
    # Per-experiment knobs (all optional). Defaults preserve historical
    # behavior: prompt_variant=baseline + LLM picked from .env / llm.yaml.
    # Setting llm_* here lets the batch runner cycle different models in one
    # process without rewriting the env between experiments.
    prompt_variant: str = "baseline"
    llm_model: str | None = None
    llm_provider: str | None = None
    llm_base_url: str | None = None
    # Random seed for the train/eval split and event-timing repair. Vary it
    # across otherwise-identical experiments to get multi-seed runs (T4/T5 are
    # high-variance, so those benchmarks should be read as multi-seed means).
    seed: int = 42
    # Draw this many rows from the dataset's full source instead of the fixed
    # paper sample (longitudinal datasets only) — needed for the sparse
    # life-event subset to stabilize T4/T5. None = keep the paper sample.
    full_source_sample: int | None = None
    # Replace the life-event-timing columns with a joint donor resample matched
    # on covariates (fixes T4 event-order, preserves T5). Only fires for
    # conditions that expose fit microdata and datasets with a T4 event config.
    event_timing_repair: bool = False
    event_timing_condition_cols: dict[str, list[str]] = field(default_factory=dict)
    # Train on this many rows drawn from the full source EXCLUDING the benchmark
    # rows, and score against the benchmark sample whole. Supersedes
    # full_source_sample / train_eval_split. Requires the dataset to declare a
    # `full_source_key` so the exclusion can be proven rather than assumed.
    disjoint_train_sample: int | None = None
    # block_donor strategy. "domain" = one block per schema domain (maximal
    # conditioning, best T3); "mega" = demography then everything else as one block
    # (maximal joint fidelity, best T2).
    donor_granularity: str = "domain"
    donor_min_cell: int = 25


def _run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _write_common(
    *,
    run_dir: Path,
    meta: dict,
    generated,
    dataset: str,
    run_id: str,
    eval_df,
) -> PassRates:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    generated.to_csv(run_dir / "generated.csv", index=False)
    rates = run_evaluation(
        dataset_name=dataset, run_id=run_id, generated=generated, sampled=eval_df,
    )
    od = _safe_overdetermination(generated=generated, eval_df=eval_df, dataset=dataset)
    (run_dir / "eval.json").write_text(_serialize_rates(rates, dataset, overdetermination=od))
    return rates


def _safe_overdetermination(*, generated, eval_df, dataset) -> dict:
    """Compute the over-determination block; never break the scoring tail."""
    try:
        return overdetermination(
            real=eval_df, sim=generated, schema=load_schema(dataset),
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _serialize_rates(r: PassRates, dataset_name: str | None = None,
                     *, overdetermination: dict | None = None) -> str:
    payload: dict = {
        "by_type": r.by_type,
        "by_variable": r.by_variable,
        "by_pair": r.by_pair,
        "overall_average": r.overall_average,
    }
    if dataset_name is not None:
        try:
            payload["by_domain"] = by_domain(r, load_schema(dataset_name))
        except Exception:
            pass
    if overdetermination is not None:
        payload["overdetermination"] = overdetermination
    return json.dumps(payload, indent=2)


def _load_existing(run_dir: Path) -> PassRates | None:
    if not run_dir.exists():
        return None
    completed = sorted(run_dir.glob("*/eval.json"))
    if not completed:
        return None
    blob = json.loads(completed[-1].read_text())
    return PassRates(
        by_type=blob.get("by_type", {}),
        by_variable=blob.get("by_variable", {}),
        by_pair=blob.get("by_pair", {}),
        overall_average=blob.get("overall_average"),
    )


def run_experiment(
    cfg: ExperimentConfig,
    *,
    resume: bool = False,
) -> dict[tuple[str, str], PassRates]:
    overrides: dict[str, str] = {}
    if cfg.llm_model:
        overrides["model"] = cfg.llm_model
    if cfg.llm_provider:
        overrides["provider"] = cfg.llm_provider
    if cfg.llm_base_url:
        overrides["base_url"] = cfg.llm_base_url
    llm_cfg = load_llm_config(overrides=overrides or None)
    client = build_client(llm_cfg)
    results: dict[tuple[str, str], PassRates] = {}

    for dataset in cfg.datasets:
        if cfg.disjoint_train_sample is not None:
            # Score against the paper's benchmark sample WHOLE, and train on rows
            # the benchmark has never seen. Without a full source these two demands
            # conflict and we have to halve the reference to manufacture training
            # rows — which both starves the model and scores us at half the paper's
            # N. load_disjoint_train proves the exclusion on the source's row key,
            # or refuses.
            eval_df = load_real_data(dataset)
            train = load_disjoint_train(
                dataset, n_sample=cfg.disjoint_train_sample, seed=cfg.seed
            )
        else:
            df = load_real_data(dataset, n_sample=cfg.full_source_sample, seed=cfg.seed)
            train, eval_df = split_train_eval(df, ratio=cfg.train_eval_split, seed=cfg.seed)
        for cond_name in cfg.conditions:
            spec = get_condition(cond_name)
            cond_dir = cfg.results_root / cfg.name / cond_name / dataset

            if resume:
                existing = _load_existing(cond_dir)
                if existing is not None:
                    results[(cond_name, dataset)] = existing
                    continue

            run_id = _run_id()
            run_dir = cond_dir / run_id
            workspace = run_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            try:
                rates = _run_one_condition(
                    spec=spec, dataset=dataset, run_id=run_id,
                    run_dir=run_dir, workspace=workspace,
                    train=train, eval_df=eval_df, cfg=cfg,
                    client=client, llm_cfg=llm_cfg,
                )
                results[(cond_name, dataset)] = rates
            except Exception as e:
                # Log the failure but continue with the next condition.
                err_path = run_dir / "error.txt"
                err_path.write_text(f"{type(e).__name__}: {e}\n")
                print(f"[runner] {cond_name} on {dataset} FAILED: {type(e).__name__}: {e}")
                results[(cond_name, dataset)] = PassRates()
    return results


_DEFAULT_EVENT_COND_COLS = ["gender", "race"]


def _maybe_repair_event_timing(generated, *, gate, dataset, cfg):
    """Optionally replace the life-event-timing columns with a covariate-matched
    joint donor resample (see ssdataagent.data.event_timing). Returns
    ``(frame, meta)``; a no-op returns the frame unchanged with empty meta.

    Only fires when the experiment opts in, the condition exposes fit microdata
    (a real donor pool — FULL/NO_SEMANTIC/UNSEEN), and the dataset has a T4
    event config. NO_DATA/aggregate/DIRECT have no donor and are left alone.
    """
    import numpy as np

    if not cfg.event_timing_repair:
        return generated, {}
    donor = gate.fit_microdata()
    if donor is None:
        return generated, {}
    event_vars = event_timing_variables(dataset)
    if not event_vars:
        return generated, {}
    cond_cols = cfg.event_timing_condition_cols.get(dataset, _DEFAULT_EVENT_COND_COLS)
    repaired = conditional_joint_repair(
        generated, donor, event_vars,
        condition_cols=cond_cols, rng=np.random.default_rng(cfg.seed),
    )
    return repaired, {"event_timing_repair": {"event_vars": event_vars,
                                              "condition_cols": cond_cols}}


def _run_one_condition(
    *, spec, dataset, run_id, run_dir, workspace, train, eval_df, cfg, client, llm_cfg,
) -> PassRates:
    if spec.context_condition is Condition.TRANSFER and dataset in TRANSFER_PAIRS:
        source_name = TRANSFER_PAIRS[dataset]
        source_df = load_source_wave(source_name)
        crosswalk = compute_crosswalk(
            load_schema(dataset), load_schema(source_name), source_df, eval_df,
        )
        gate = InfoGate(
            condition=spec.context_condition, dataset_name=dataset, workspace=workspace,
            client=client, train=train, eval_rows=eval_df,
            unseen_variables=tuple(cfg.unseen_variables.get(dataset, [])),
            source=source_df, source_name=source_name, crosswalk=tuple(crosswalk),
        )
    else:
        gate = InfoGate(
            condition=spec.context_condition,
            dataset_name=dataset,
            workspace=workspace,
            client=client,
            train=train,
            eval_rows=eval_df,
            unseen_variables=tuple(cfg.unseen_variables.get(dataset, [])),
        )
    strategy = get_strategy(spec.strategy)
    result = strategy.generate(gate, run_dir, cfg)
    generated, repair_meta = _maybe_repair_event_timing(
        result.generated, gate=gate, dataset=dataset, cfg=cfg,
    )
    meta = {
        "experiment": cfg.name,
        "dataset": dataset,
        "condition": spec.name,
        "run_id": run_id,
        "git_sha": _git_sha(),
        "model": llm_cfg.model,
        "provider": llm_cfg.provider,
    }
    meta.update(result.meta_extras)
    meta.update(repair_meta)
    return _write_common(
        run_dir=run_dir, meta=meta, generated=generated,
        dataset=dataset, run_id=run_id, eval_df=eval_df,
    )
