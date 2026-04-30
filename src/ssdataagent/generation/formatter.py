from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import load_schema


def format_generated(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Coerce a generated DataFrame to schema-conformant values.

    Categoricals outside the allowed set become NaN; numerics outside the
    declared range are clipped to the range.

    Schema-declared variables the agent didn't produce (or produced as
    all-NaN, common in unseen-variable runs) are filled with a random-uniform
    baseline drawn from the schema range/allowed values. This represents
    "agent made no informed prediction" — the bootstrap eval will score the
    variable low because the marginal won't match real, which is the correct
    meaningful negative result. Fixed seed for reproducibility.
    """
    schema = load_schema(dataset_name)
    out = df.copy()
    n = len(out)
    rng = np.random.default_rng(20260430)
    expected_vars = set(schema.background_variables) | set(schema.target_variables)
    for var in expected_vars:
        if var not in out.columns:
            out[var] = pd.NA
    for var, allowed in schema.allowed_values.items():
        if var in out.columns:
            out.loc[~out[var].isin(allowed), var] = pd.NA
    for var, (lo, hi) in schema.numeric_ranges.items():
        if var in out.columns:
            out[var] = pd.to_numeric(out[var], errors="coerce").clip(lower=lo, upper=hi)
    # Fill any all-NaN expected column with a uniform-random baseline
    # so SSDataBench's bootstrap can sample (and will score it at near-zero,
    # the correct outcome for "agent didn't predict this variable").
    for var in expected_vars:
        if var in out.columns and out[var].isna().all():
            if var in schema.allowed_values:
                out[var] = rng.choice(schema.allowed_values[var], size=n)
            elif var in schema.numeric_ranges:
                lo, hi = schema.numeric_ranges[var]
                out[var] = rng.uniform(lo, hi, size=n)
            # if neither (no schema info), leave as NaN — eval may still skip
    # profile_id must exist AND be non-NaN — SSDataBench's bootstrap does
    # dropna() on a [profile_id, var] frame, so any all-NaN profile_id wipes
    # every row before the test even starts.
    if "profile_id" not in out.columns or out["profile_id"].isna().all():
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
