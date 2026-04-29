from __future__ import annotations

from pathlib import Path

import pandas as pd

from ssdataagent.data.schema import load_schema


def format_generated(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Coerce a generated DataFrame to schema-conformant values.

    Categoricals outside the allowed set become NaN; numerics outside the
    declared range are clipped to the range.
    """
    schema = load_schema(dataset_name)
    out = df.copy()
    for var, allowed in schema.allowed_values.items():
        if var in out.columns:
            out.loc[~out[var].isin(allowed), var] = pd.NA
    for var, (lo, hi) in schema.numeric_ranges.items():
        if var in out.columns:
            out[var] = pd.to_numeric(out[var], errors="coerce").clip(lower=lo, upper=hi)
    if "profile_id" not in out.columns:
        out["profile_id"] = range(len(out))
    return out


def write_simulated(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    run_id: str,
    ssdatabench_root: Path,
    sampled_df: pd.DataFrame | None = None,
) -> Path:
    """Write the simulated CSV (and optional paired sampled CSV) into
    SSDataBench's expected ``simulated_data/<subdir>/agent_<run_id>/``
    folder. Returns the path to the simulated CSV.
    """
    schema = load_schema(dataset_name)
    sim_dir = (
        ssdatabench_root / "simulated_data"
        / schema.ssdatabench_sim_subdir / f"agent_{run_id}"
    )
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim_path = sim_dir / f"sim_profiles_{run_id}.csv"
    df.to_csv(sim_path, index=False)
    if sampled_df is not None:
        sampled_path = sim_dir / f"sampled_{run_id}.csv"
        sampled_df.to_csv(sampled_path, index=False)
    return sim_path
