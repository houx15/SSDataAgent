from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ssdataagent.agent.context import Condition, build_context
from ssdataagent.agent.llm_client import build_client
from ssdataagent.agent.orchestrator import Orchestrator
from ssdataagent.config import REPO_ROOT, load_llm_config
from ssdataagent.data.loader import load_real_data
from ssdataagent.data.splitter import split_train_eval
from ssdataagent.data.schema import load_schema
from ssdataagent.evaluation.runner import PassRates, by_domain, run_evaluation
from ssdataagent.experiments.conditions import get_condition
from ssdataagent.experiments.logger import log_run


@dataclass
class ExperimentConfig:
    name: str
    datasets: list[str]
    conditions: list[str]
    max_iterations: int
    sandbox_timeout: int
    train_eval_split: float
    n_rows: int
    results_root: Path = REPO_ROOT / "results"
    unseen_variables: dict[str, list[str]] = field(default_factory=dict)


def _run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


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
    llm_cfg = load_llm_config()
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
    if not spec.is_agent:
        from ssdataagent.experiments.direct_generation import generate_direct
        transcript: list[dict] = []
        generated = generate_direct(
            client=client,
            sampled=eval_df,
            dataset_name=dataset,
            transcript_out=transcript,
        )
        meta = {
            "experiment": cfg.name,
            "dataset": dataset,
            "condition": spec.name,
            "run_id": run_id,
            "git_sha": _git_sha(),
            "model": llm_cfg.model,
            "provider": llm_cfg.provider,
            "n_individuals": len(eval_df),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        prompts_lines = [
            json.dumps({"row": e["row"], "role": "user", "content": e["prompt"]})
            for e in transcript
        ]
        responses_lines = [
            json.dumps({"row": e["row"], "role": "assistant", "content": e["response"]})
            for e in transcript
        ]
        (run_dir / "prompts.jsonl").write_text(
            "\n".join(prompts_lines) + ("\n" if prompts_lines else "")
        )
        (run_dir / "responses.jsonl").write_text(
            "\n".join(responses_lines) + ("\n" if responses_lines else "")
        )
        generated.to_csv(run_dir / "generated.csv", index=False)
        rates = run_evaluation(
            dataset_name=dataset,
            run_id=run_id,
            generated=generated,
            sampled=eval_df,
        )
        (run_dir / "eval.json").write_text(_serialize_rates(rates, dataset))
        return rates

    unseen = cfg.unseen_variables.get(dataset, [])
    ctx = build_context(
        condition=spec.context_condition,
        dataset_name=dataset,
        train_df=train,
        workspace=workspace,
        unseen_variables=unseen if spec.context_condition is Condition.UNSEEN else None,
    )
    orch = Orchestrator(
        client=client,
        n_rows=cfg.n_rows,
        max_validation_iters=cfg.max_iterations,
        sandbox_timeout=cfg.sandbox_timeout,
    )
    result = orch.run(
        condition=spec.context_condition,
        dataset_name=dataset,
        workspace=workspace,
        has_data=ctx.has_data,
        has_descriptions=ctx.has_descriptions,
    )
    log_run(
        result,
        run_dir=run_dir,
        meta={
            "experiment": cfg.name,
            "dataset": dataset,
            "condition": spec.name,
            "run_id": run_id,
            "git_sha": _git_sha(),
            "model": llm_cfg.model,
            "provider": llm_cfg.provider,
            "unseen_variables": unseen,
        },
    )

    rates = run_evaluation(
        dataset_name=dataset,
        run_id=run_id,
        generated=result.generated,
        sampled=eval_df,
    )
    (run_dir / "eval.json").write_text(_serialize_rates(rates))
    return rates
