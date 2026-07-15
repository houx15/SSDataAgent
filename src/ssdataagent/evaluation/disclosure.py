"""Disclosure risk — the axis SSDataBench never measures.

The benchmark scores only *fidelity* (do the synthetic marginals/joints match?).
By that measure, whole-row resampling is unbeatable — but it republishes actual
respondents verbatim, which is exactly what synthetic data exists to avoid. A
fidelity-only benchmark therefore rewards a method that could never be deployed.

These metrics put a number on that. Given a synthetic frame and the real pool it
was built from, we ask: how many synthetic individuals ARE a real respondent?

  copy_rate            fraction of synthetic rows identical, on the modelled
                       columns, to some real pool row. Whole-row resampling -> ~1.0
                       (every synthetic person IS a real one); a recombining method
                       (block-donor) -> near 0.
  unique_copy_rate     fraction identical to a real row that is UNIQUE in the pool.
                       These are the individually re-identifiable copies — the
                       highest-risk disclosures, not just coincidental collisions.
  copy_rate_excess     copy_rate minus the baseline collision rate a *fresh real
                       sample* shows against the pool. Isolates memorisation from
                       the coincidence floor (sparse rows collide by chance).

Higher is worse. All three are in [0, 1].
"""
from __future__ import annotations

import hashlib

import pandas as pd


def _row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """A stable hash per row over `cols`. StringDtype NA survives astype(str) as
    <NA>; map(str) forces every cell to a real str so NA==NA rather than raising."""
    flat = df[cols].apply(lambda c: c.map(str))
    joined = flat.agg("\x1f".join, axis=1)
    return joined.map(lambda v: hashlib.blake2b(v.encode(), digest_size=8).hexdigest())


def disclosure_metrics(
    synthetic: pd.DataFrame,
    real_pool: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    baseline: pd.DataFrame | None = None,
) -> dict:
    """Re-identification risk of `synthetic` against the `real_pool` it drew from.

    Args:
        synthetic: generated rows.
        real_pool: the real microdata being protected (e.g. the training donors).
        columns: which columns define an identity; defaults to the columns common
            to both frames.
        baseline: an independent real sample (rows NOT in real_pool) used to
            measure the coincidence floor for `copy_rate_excess`. Optional.

    Returns a dict of the three rates plus the supporting counts.
    """
    if columns is None:
        columns = [c for c in synthetic.columns if c in real_pool.columns]
    if not columns:
        raise ValueError("no shared columns between synthetic and real_pool")

    pool_h = _row_hashes(real_pool, columns)
    pool_counts = pool_h.value_counts()
    pool_set = set(pool_counts.index)
    unique_pool = {h for h, n in pool_counts.items() if n == 1}

    syn_h = _row_hashes(synthetic, columns)
    is_copy = syn_h.isin(pool_set)
    is_unique_copy = syn_h.isin(unique_pool)

    copy_rate = float(is_copy.mean())
    out = {
        "copy_rate": copy_rate,
        "unique_copy_rate": float(is_unique_copy.mean()),
        "n_synthetic": int(len(synthetic)),
        "n_columns": len(columns),
    }
    if baseline is not None:
        base_h = _row_hashes(baseline, columns)
        floor = float(base_h.isin(pool_set).mean())
        out["coincidence_floor"] = floor
        out["copy_rate_excess"] = max(0.0, copy_rate - floor)
    return out
