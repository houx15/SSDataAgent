from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ssdataagent.agent.llm_client import build_client
from ssdataagent.config import REPO_ROOT, load_llm_config, results_root
from ssdataagent.data.loader import load_real_data
from ssdataagent.data.splitter import split_train_eval
from ssdataagent.data.schema import load_schema
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
    train_eval_split: float
    n_rows: int
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
    (run_dir / "eval.json").write_text(_serialize_rates(rates, dataset))
    return rates


def _serialize_rates(r: PassRates, dataset_name: str | None = None) -> str:
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
        df = load_real_data(dataset)
        train, eval_df = split_train_eval(df, ratio=cfg.train_eval_split, seed=42)
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


def _run_one_condition(
    *, spec, dataset, run_id, run_dir, workspace, train, eval_df, cfg, client, llm_cfg,
) -> PassRates:
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
    return _write_common(
        run_dir=run_dir, meta=meta, generated=result.generated,
        dataset=dataset, run_id=run_id, eval_df=eval_df,
    )
