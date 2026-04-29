from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ssdataagent.config import REPO_ROOT
from ssdataagent.data.schema import load_schema
from ssdataagent.generation.formatter import format_generated, write_simulated


@dataclass(frozen=True)
class PassRates:
    by_type: dict[str, float] = field(default_factory=dict)
    by_variable: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_average: float | None = None


def parse_pass_rates(eval_dir: Path) -> PassRates:
    """Read SSDataBench's evaluation outputs from *eval_dir* into PassRates.

    Format reference (observed in smoke run):
      - overall_insignificant_summary.csv:
            type,avg_insignificant_rate,summary_path
            type1,0.96,...
            ...
            Overall Average,0.95,
      - summary_type<N>.csv:
            variable,type,insignificant_rate,key
            gender,categorical,0.93,
            ...
            ,,0.96,avg
    """
    by_type: dict[str, float] = {}
    by_variable: dict[str, dict[str, float]] = {}
    overall_average: float | None = None

    overall = eval_dir / "overall_insignificant_summary.csv"
    if overall.exists():
        df = pd.read_csv(overall)
        for _, row in df.iterrows():
            name = str(row["type"]).strip()
            rate = row["avg_insignificant_rate"]
            if name == "Overall Average":
                overall_average = float(rate)
            elif pd.notna(rate):
                by_type[name] = float(rate)

    for f in sorted(eval_dir.glob("summary_type*.csv")):
        # Only single-type summaries; skip _strength variants for now (those have
        # different schemas — pair-wise stats aren't tracked at variable level).
        if "strength" in f.stem:
            continue
        type_name = f.stem.replace("summary_", "")
        df = pd.read_csv(f)
        per: dict[str, float] = {}
        for _, row in df.iterrows():
            var = row.get("variable")
            key = row.get("key", "")
            if pd.isna(var) or str(key).strip() == "avg":
                continue
            rate = row.get("insignificant_rate")
            if pd.notna(rate):
                per[str(var)] = float(rate)
        if per:
            by_variable[type_name] = per

    return PassRates(by_type=by_type, by_variable=by_variable, overall_average=overall_average)


def run_evaluation(
    *,
    dataset_name: str,
    run_id: str,
    generated: pd.DataFrame,
    sampled: pd.DataFrame,
    ssdatabench_root: Path | None = None,
    config_path: str | None = None,
) -> PassRates:
    """Format the agent's output, copy it into ssdatabench/simulated_data/...,
    invoke the dataset's evaluation script in --single mode, and parse the
    resulting pass rates.
    """
    ssdatabench_root = ssdatabench_root or (REPO_ROOT / "ssdatabench")
    schema = load_schema(dataset_name)
    formatted = format_generated(generated, dataset_name)
    sim_csv = write_simulated(
        formatted,
        dataset_name=dataset_name,
        run_id=run_id,
        ssdatabench_root=ssdatabench_root,
        sampled_df=sampled,
    )
    sim_root = sim_csv.parent
    output_base = (
        ssdatabench_root / "evaluation_results"
        / schema.ssdatabench_sim_subdir / f"agent_{run_id}"
    )
    output_base.mkdir(parents=True, exist_ok=True)

    if config_path is None:
        config_path = (
            f"evaluation/config/{schema.ssdatabench_sim_subdir}_subset/"
            "evaluation_master.yaml"
        )

    cmd = [
        "python", schema.evaluation_script,
        "--single",
        "--sim-root", str(sim_root.relative_to(ssdatabench_root)),
        "--output-base", str(output_base.relative_to(ssdatabench_root)),
        "--config", config_path,
    ]
    subprocess.run(cmd, cwd=ssdatabench_root, check=False)
    return parse_pass_rates(output_base)


def split_by_seen_unseen(
    rates: PassRates, unseen_vars: list[str]
) -> tuple[PassRates, PassRates]:
    """Partition by_variable into seen vs unseen pass-rate views."""
    seen: dict[str, dict[str, float]] = {}
    unseen: dict[str, dict[str, float]] = {}
    for type_name, vars_ in rates.by_variable.items():
        seen[type_name] = {k: v for k, v in vars_.items() if k not in unseen_vars}
        unseen[type_name] = {k: v for k, v in vars_.items() if k in unseen_vars}
    return (
        PassRates(by_type=rates.by_type, by_variable=seen, overall_average=rates.overall_average),
        PassRates(by_type=rates.by_type, by_variable=unseen, overall_average=rates.overall_average),
    )
