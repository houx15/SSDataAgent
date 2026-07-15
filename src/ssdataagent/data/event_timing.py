"""Conditional joint resampling of life-course event-timing columns.

SSDataBench's T4 (event-order consistency) and T5 (event-order x covariate)
benchmarks score whether generated data reproduces the *joint ordering* of a
handful of life-event age columns (e.g. age_started_work < age_at_first_sex <
age_at_first_marriage). Every strategy scores ~0 on T4 because it fills those
age columns roughly independently, smearing the ordering distribution.

This module replaces the event-timing block of a generated frame with age
tuples resampled *jointly* from donor (real) rows, matched on a few covariates
so the order<->covariate association T5 checks is preserved too. Missingness is
carried over verbatim (donor tuples may contain NaN), so the marginal
missingness of these columns is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from ssdataagent.data.schema import _resolve_ssdatabench_path, load_schema


def event_timing_variables(dataset_name: str) -> list[str]:
    """The event-timing columns SSDataBench's T4 config lists for this dataset,
    restricted to those the schema actually knows about. Empty for
    cross-sectional datasets (no type4 config)."""
    schema = load_schema(dataset_name)
    cfg_path = _resolve_ssdatabench_path(
        f"ssdatabench/evaluation/config/{schema.ssdatabench_sim_subdir}/type4.yaml"
    )
    if not cfg_path.exists():
        return []
    with cfg_path.open() as f:
        spec = yaml.safe_load(f) or {}
    ev = spec.get("event_variables") or {}
    known = set(schema.background_variables) | set(schema.target_variables)
    return [v for v in ev if v in known]


def _coarsen_key_columns(
    donor: pd.DataFrame, generated: pd.DataFrame, cols: list[str], n_bins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Binned copies of the condition columns for both frames, using the donor's
    own quantile edges so the two sides land in the same bins.

    A continuous covariate (birth_year, an age) almost never matches on its exact
    value, so an unbinned key collapses straight to the unconditional fallback and
    the covariate association is lost. Quantile-binning restores it — the same
    trick block_donor uses on its keys. Missing is its own stratum (-1)."""
    dk, gk = {}, {}
    for c in cols:
        ds = pd.to_numeric(donor[c], errors="coerce")
        if ds.notna().mean() > 0.5 and ds.nunique() > n_bins * 2:
            qs = np.linspace(0, 1, n_bins + 1)[1:-1]
            edges = np.unique(np.nanquantile(ds.to_numpy(dtype=float), qs))

            def _bin(s):
                v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
                b = np.digitize(v, edges).astype(float)
                b[np.isnan(v)] = -1.0
                return b

            dk[c], gk[c] = _bin(donor[c]), _bin(generated[c])
        else:
            dk[c] = donor[c].astype(str).to_numpy()
            gk[c] = generated[c].astype(str).to_numpy()
    return (pd.DataFrame(dk, index=donor.index),
            pd.DataFrame(gk, index=generated.index))


def conditional_joint_repair(
    generated: pd.DataFrame,
    donor: pd.DataFrame,
    event_vars: list[str],
    *,
    condition_cols: list[str],
    rng: np.random.Generator,
    min_cell: int = 20,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Return a copy of ``generated`` with ``event_vars`` replaced by age tuples
    resampled jointly from ``donor``.

    For each generated row, a donor row is drawn from those sharing the row's
    values on ``condition_cols`` (so the event-order/covariate association is
    preserved); the whole ``event_vars`` tuple is copied over at once (so their
    joint ordering matches real). Continuous condition columns are quantile-binned
    (``n_bins``) so they actually match instead of collapsing to the unconditional
    fallback. When a covariate cell holds fewer than ``min_cell`` donors, the match
    falls back to progressively coarser keys (drop the last condition column, ... ,
    finally unconditional). Rows that still can't be matched keep their values.
    """
    event_vars = [v for v in event_vars if v in generated.columns and v in donor.columns]
    if not event_vars or len(donor) == 0:
        return generated.copy()

    cond = [c for c in condition_cols if c in generated.columns and c in donor.columns]
    donor = donor.reset_index(drop=True)
    out = generated.reset_index(drop=True).copy()
    n = len(out)
    # positional index into `donor` chosen for each generated row; -1 = unmatched
    chosen = np.full(n, -1, dtype=np.int64)
    donor_all = donor.index.to_numpy()
    donor_keys, gen_keys_frame = _coarsen_key_columns(donor, out, cond, n_bins)

    # Progressive fallback over shrinking condition keys, then unconditional.
    # Keys are built as tuples on both sides so matching is independent of any
    # pandas groupby-key quirks (scalar-vs-tuple for single-column keys).
    for depth in range(len(cond), -1, -1):
        pending = np.flatnonzero(chosen < 0)
        if pending.size == 0:
            break
        key = cond[:depth]
        if not key:  # unconditional: any donor
            chosen[pending] = rng.choice(donor_all, size=pending.size, replace=True)
            break
        groups: dict[tuple, list[int]] = {}
        for i, k in enumerate(donor_keys[key].itertuples(index=False, name=None)):
            groups.setdefault(k, []).append(i)
        pools = {k: np.array(v) for k, v in groups.items() if len(v) >= min_cell}
        gen_keys = gen_keys_frame.loc[pending, key].itertuples(index=False, name=None)
        for row_pos, gk in zip(pending, gen_keys):
            pool = pools.get(gk)
            if pool is not None:
                chosen[row_pos] = rng.choice(pool)

    matched = chosen >= 0
    if matched.any():
        picks = chosen[matched]
        for v in event_vars:
            # object array so donor NaN / string codes copy over without the
            # int-cast corruption that plain numeric assignment would cause.
            vals = out[v].to_numpy(dtype=object, copy=True)
            vals[matched] = donor[v].to_numpy(dtype=object)[picks]
            out[v] = pd.Series(vals, index=out.index).infer_objects()
    return out
