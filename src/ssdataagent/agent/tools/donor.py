"""Block-donor step family.

Every pre-existing step family sacrifices something the benchmark measures:

  ``fit_marginal``       drops NaN at fit time and never re-emits it, so a column
                         that is 60% missing in real comes out 0% missing in sim.
                         The eval turns missingness into *selection* (T1/T3/T4 all
                         coerce-to-numeric then dropna), so this silently swaps the
                         scored subpopulation.
  ``linear_regression``  draws ``yhat + N(0, sigma)``. The conditional mean is right
                         but the marginal is smeared into a Gaussian, so T1 fails
                         even when the model is good.
  ``empirical_lookup``   keys on the *exact* tuple of ``given`` values. With a
                         continuous column in ``given``, 60-95% of sample-time keys
                         were never seen at fit time and fall back to the global
                         marginal — a marginal wearing a conditional's clothes.
  any per-column draw    fills each event-age column independently, smearing the
                         joint life-course ordering that T4/T5 score.

A block-donor step fixes all four at once. Columns are declared as a *block*; for
each simulated row one real donor is matched on a **coarsened** key over
already-generated columns (with progressive fallback to shorter keys, then
unconditional), and that donor's whole block is copied verbatim.

Consequences, all by construction rather than by tuning:
  * values are real -> the marginal is exact (T1)
  * NaN and censoring sentinels ('never married') copy across -> missingness and
    occurrence rates are reproduced, so the scored subpopulation is right
  * the block is copied as a unit -> within-block joint structure, chronology, and
    hard logical constraints ("no marriage age if never married") hold exactly
  * the key is binned, not exact -> conditional structure without the fallback cliff
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ssdataagent.agent.tools.fit import Step
from ssdataagent.agent.tools.state import RuntimeState, data_withheld_error


# A cell must hold at least this many donors before we condition on it; thinner
# cells fall back to a shorter key. Guards against copying one specific person.
DEFAULT_MIN_CELL = 25
# Quantile bins per continuous condition column. More bins = sharper conditioning
# but thinner cells; 5 keeps cells populated at the training sizes we run at.
DEFAULT_N_BINS = 5


def _coarsen_spec(train: pd.DataFrame, cols: list[str], n_bins: int) -> dict:
    """Per-column coarsening rule. Numeric columns become quantile bins; anything
    else is matched on its literal value."""
    spec: dict[str, tuple[str, Any]] = {}
    for c in cols:
        s = pd.to_numeric(train[c], errors="coerce")
        if s.notna().mean() > 0.5 and s.nunique() > n_bins * 2:
            qs = np.linspace(0, 1, n_bins + 1)[1:-1]
            edges = np.unique(np.nanquantile(s.to_numpy(dtype=float), qs))
            spec[c] = ("numeric", edges)
        else:
            spec[c] = ("categorical", None)
    return spec


def _key_frame(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply the coarsening. Missing gets its own stratum (-1) rather than being
    lumped with the lowest bin — 'we don't know' is a real, predictive state."""
    out = {}
    for c, (kind, edges) in spec.items():
        if kind == "numeric":
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            b = np.digitize(v, edges).astype(float)
            b[np.isnan(v)] = -1.0
            out[c] = b
        else:
            out[c] = df[c].astype(str).to_numpy()
    return pd.DataFrame(out, index=df.index)


@dataclass
class BlockDonorStep(Step):
    """Copies `block_cols` verbatim from a covariate-matched real donor."""
    kind: str = field(default="block_donor", init=False)
    family: str = field(default="block_donor", init=False)
    block_cols: list[str] = field(default_factory=list)
    given: list[str] = field(default_factory=list)
    min_cell: int = DEFAULT_MIN_CELL
    spec: dict = field(default_factory=dict)
    donors: pd.DataFrame | None = None
    # deepest-key-first list of (key_columns, {key_tuple: donor row positions})
    pools: list = field(default_factory=list)

    def fit(self, train: pd.DataFrame, n_bins: int = DEFAULT_N_BINS) -> "BlockDonorStep":
        self.donors = train.reset_index(drop=True)
        self.spec = _coarsen_spec(self.donors, self.given, n_bins) if self.given else {}
        self.pools = []
        if not self.given:
            return self
        keys = _key_frame(self.donors, self.spec)
        for depth in range(len(self.given), 0, -1):
            sub = self.given[:depth]
            groups: dict[tuple, list[int]] = {}
            for i, k in enumerate(keys[sub].itertuples(index=False, name=None)):
                groups.setdefault(k, []).append(i)
            self.pools.append(
                (sub, {k: np.asarray(v) for k, v in groups.items() if len(v) >= self.min_cell})
            )
        return self

    def donor_index(self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame) -> np.ndarray:
        """One donor row position per simulated row. Deepest key that has a
        populated cell wins; anything still unmatched draws unconditionally."""
        chosen = np.full(n_rows, -1, dtype=np.int64)
        if self.given:
            gk = _key_frame(partial[self.given], self.spec)
            for sub, table in self.pools:
                pending = np.flatnonzero(chosen < 0)
                if pending.size == 0:
                    break
                rows = gk.loc[pending, sub].itertuples(index=False, name=None)
                for pos, k in zip(pending, rows):
                    pool = table.get(k)
                    if pool is not None:
                        chosen[pos] = pool[rng.integers(len(pool))]
        pending = np.flatnonzero(chosen < 0)
        if pending.size:
            chosen[pending] = rng.integers(0, len(self.donors), size=pending.size)
        return chosen

    def sample_block(self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame) -> pd.DataFrame:
        picks = self.donor_index(rng, n_rows, partial)
        return self.donors.loc[picks, self.block_cols].reset_index(drop=True)

    def sample(self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame) -> np.ndarray:
        # Single-column use, and the fallback path if a caller ignores block_cols.
        return self.sample_block(rng, n_rows, partial)[self.col].to_numpy()

    def to_meta(self) -> dict:
        return {
            "col": self.col, "kind": self.kind, "family": self.family,
            "block_cols": list(self.block_cols), "given": list(self.given),
            "min_cell": self.min_cell,
        }


def fit_block_donor(
    state: RuntimeState,
    cols: list[str],
    given: list[str] | None = None,
    min_cell: int = DEFAULT_MIN_CELL,
) -> dict:
    """Register `cols` as one block, drawn together from a donor matched on `given`.

    Put columns in the same block when they are mutually constrained — life-event
    ages that must stay in chronological order, or a flag and the value it gates
    ("never married" / age at first marriage). One real person then supplies the
    whole block, so those constraints cannot be violated.
    """
    if not state.has_data:
        return data_withheld_error()
    if not isinstance(cols, list) or not cols:
        return {"error": "bad_arguments", "details": "cols must be a non-empty list"}
    given = list(given or [])

    train = state.train_fit
    unknown = [c for c in cols + given if c not in train.columns]
    if unknown:
        return {"error": "unknown_column", "details": f"not in train: {unknown}"}

    order = state.chain.generation_order
    missing_from_order = [c for c in cols if c not in order]
    if missing_from_order:
        return {
            "error": "col_not_in_order",
            "details": f"call set_generation_order including {missing_from_order} first",
        }
    first_pos = min(order.index(c) for c in cols)
    late = [g for g in given if g not in order or order.index(g) >= first_pos]
    if late:
        return {
            "error": "given_after_col",
            "details": f"{late} must appear before the block's first column in generation_order",
        }

    step = BlockDonorStep(col=cols[0], block_cols=list(cols), given=given, min_cell=min_cell)
    step.fit(train)

    # A block step is registered under every column it produces, so Chain.sample
    # finds it wherever it reaches the block in the generation order.
    for c in cols:
        state.chain.add_alias(c, step)

    cond_frac = None
    if given:
        probe = step.donor_index(np.random.default_rng(0), len(train), train)
        keys = _key_frame(train, step.spec)
        deepest = step.pools[0][1] if step.pools else {}
        hits = sum(1 for k in keys[step.given].itertuples(index=False, name=None) if k in deepest)
        cond_frac = round(hits / len(train), 3)
        del probe

    return {
        "block": list(cols), "given": given, "family": "block_donor",
        "n_donors": int(len(train)), "min_cell": min_cell,
        # How often the *full* key found a populated cell. Low means the block is
        # conditioning on little more than the fallback — drop a condition column.
        "full_key_match_rate": cond_frac,
        "registered": True,
    }
