from __future__ import annotations

import subprocess
import sys
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
    # Type 2 is pairwise (var1, var2). Stored as nested dict so we don't lose the
    # pair structure (rolling up to per-variable means via _per_variable_from_pairs).
    by_pair: dict[str, list[dict]] = field(default_factory=dict)


def _per_variable_from_pairs(pair_rows: list[dict]) -> dict[str, float]:
    """Average pair rates per variable (each pair contributes to both vars)."""
    sums: dict[str, list[float]] = {}
    for row in pair_rows:
        rate = row.get("insignificant_rate")
        if rate is None:
            continue
        for k in ("var1", "var2"):
            v = row.get(k)
            if v:
                sums.setdefault(str(v), []).append(float(rate))
    return {v: sum(rs) / len(rs) for v, rs in sums.items() if rs}


def parse_pass_rates(eval_dir: Path) -> PassRates:
    """Parse SSDataBench evaluation outputs into a PassRates view.

    Schemas across types:
      - type1: ``variable, type, insignificant_rate, key`` — variable-level
      - type2: ``var1, var2, type1, type2, mode, insignificant_rate, key`` — pair-level
      - type3: ``response, model_type, mode, insignificant_rate, ..., key`` — response-level
      - type4: ``combo, insignificant_rate, ...`` — chronological-event-order
        keyed by the ``combo`` string (e.g., "age_finished_education→age_at_first_marriage→age_at_first_child")
      - type5: ``combo, predictor, type, insignificant_rate, mode, ...`` — event-order
        × covariate; rolled up per-predictor (mean across combos)
    The trailing ``avg``/``avg_strength``/``avg_all`` rows are dropped.
    """
    by_type: dict[str, float] = {}
    by_variable: dict[str, dict[str, float]] = {}
    by_pair: dict[str, list[dict]] = {}
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
        if "strength" in f.stem:
            continue
        type_name = f.stem.replace("summary_", "")
        df = pd.read_csv(f)

        cols = set(df.columns)
        if type_name == "type5" and {"combo", "predictor"}.issubset(cols):
            # Type 5 — event_order × covariate. Each row is one combo×predictor
            # pair; roll up to per-predictor means and keep the raw pairs.
            pair_rows: list[dict] = []
            for _, row in df.iterrows():
                key = row.get("key", "")
                if str(key).strip().startswith("avg"):
                    continue
                rate = row.get("insignificant_rate")
                combo = row.get("combo")
                pred = row.get("predictor")
                if pd.isna(rate) or pd.isna(combo) or pd.isna(pred):
                    continue
                pair_rows.append({
                    "combo": str(combo), "predictor": str(pred),
                    "insignificant_rate": float(rate),
                    "mode": str(row.get("mode", "")),
                })
            if pair_rows:
                by_pair[type_name] = pair_rows
                bucket: dict[str, list[float]] = {}
                for r in pair_rows:
                    bucket.setdefault(r["predictor"], []).append(r["insignificant_rate"])
                by_variable[type_name] = {
                    p: sum(rs) / len(rs) for p, rs in bucket.items()
                }
            continue

        if type_name == "type4" and "combo" in cols and "var1" not in cols:
            # Type 4 — combo-keyed (no var1/var2). Use combo as the variable.
            per: dict[str, float] = {}
            for _, row in df.iterrows():
                key = row.get("key", row.get("combo", ""))
                if str(row.get("combo", "")).strip() == "avg":
                    continue
                rate = row.get("insignificant_rate")
                combo = row.get("combo")
                if pd.isna(rate) or pd.isna(combo):
                    continue
                per[str(combo)] = float(rate)
            if per:
                by_variable[type_name] = per
            continue

        if {"var1", "var2"}.issubset(cols):
            # Type 2 — pairwise. Keep the raw pair rows and roll up to per-variable.
            pair_rows: list[dict] = []
            for _, row in df.iterrows():
                key = row.get("key", "")
                if str(key).strip().startswith("avg"):
                    continue
                rate = row.get("insignificant_rate")
                if pd.isna(rate) or pd.isna(row.get("var1")) or pd.isna(row.get("var2")):
                    continue
                pair_rows.append({
                    "var1": str(row["var1"]), "var2": str(row["var2"]),
                    "insignificant_rate": float(rate),
                    "mode": str(row.get("mode", "")),
                })
            if pair_rows:
                by_pair[type_name] = pair_rows
                by_variable[type_name] = _per_variable_from_pairs(pair_rows)
            continue

        # Type 1 uses 'variable'; type 3+ use 'response' (the modeled outcome).
        var_col = "variable" if "variable" in cols else ("response" if "response" in cols else None)
        if var_col is None:
            continue
        per: dict[str, float] = {}
        for _, row in df.iterrows():
            key = row.get("key", "")
            if str(key).strip().startswith("avg"):
                continue
            var = row.get(var_col)
            if pd.isna(var):
                continue
            rate = row.get("insignificant_rate")
            if pd.notna(rate):
                # If the same response appears multiple times (e.g., overall + strength),
                # keep the last one — type3 emits 'strength' and 'overall' rows per response.
                per[str(var)] = float(rate)
        if per:
            by_variable[type_name] = per

    return PassRates(
        by_type=by_type,
        by_variable=by_variable,
        overall_average=overall_average,
        by_pair=by_pair,
    )


def by_domain(rates: PassRates, schema) -> dict[str, dict[str, float]]:
    """Aggregate per-variable rates into per-domain rates using the schema's
    ``domains`` map. Returns ``{type: {domain: mean_rate}}``. Variables without
    a declared domain are bucketed under ``Other``. Type 4 keys are
    ``combo`` strings (e.g., ``E→C→M``); domains aren't meaningful so they all
    fall under ``Sequence``.
    """
    out: dict[str, dict[str, float]] = {}
    for type_name, vars_ in rates.by_variable.items():
        bucket: dict[str, list[float]] = {}
        for var, rate in vars_.items():
            if type_name == "type4":
                dom = "Sequence"
            else:
                dom = schema.domains.get(var, "Other")
            bucket.setdefault(dom, []).append(float(rate))
        out[type_name] = {d: sum(rs) / len(rs) for d, rs in bucket.items() if rs}
    return out


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
        subdir = schema.ssdatabench_sim_subdir
        subset_master = ssdatabench_root / "evaluation" / "config" / f"{subdir}_subset" / "evaluation_master.yaml"
        if subset_master.exists():
            config_path = f"evaluation/config/{subdir}_subset/evaluation_master.yaml"
        else:
            # Longitudinal datasets typically don't ship a `_subset` variant —
            # fall back to the upstream master config (full T1-T5 suite).
            config_path = f"evaluation/config/{subdir}/evaluation_master.yaml"

    # Use the same interpreter as the parent so the eval picks up venv-only
    # deps (autograd is needed for type2's Cramer's V; system python often
    # doesn't have it). Also prepend the venv's bin to PATH so the eval's
    # nested subprocess (`python evaluation/run_all_types.py` from
    # batch_eval.py) picks up the same interpreter.
    cmd = [
        sys.executable, schema.evaluation_script,
        "--single",
        "--sim-root", str(sim_root.relative_to(ssdatabench_root)),
        "--output-base", str(output_base.relative_to(ssdatabench_root)),
        "--config", config_path,
    ]
    import os
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    subprocess.run(cmd, cwd=ssdatabench_root, check=False, env=env)
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
