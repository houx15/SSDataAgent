"""Public X-margins: admit B's census-standard demographic margins (age/gender/race) while
keeping the copula and Y-marginals from A / the description. See
docs/superpowers/specs/2026-07-28-public-x-margins-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PUBLIC_X = ("age", "gender", "race")


def with_public_x(base_marg: pd.DataFrame, b_pool: pd.DataFrame, x_cols, *,
                  seed: int = 0) -> pd.DataFrame:
    """Copy of ``base_marg`` with each column in ``x_cols`` that is present in ``b_pool``
    replaced by a length-preserving resample of ``b_pool``'s column (carrying B's marginal,
    including its missingness rate). Every other column is byte-identical to ``base_marg``;
    a column absent from ``b_pool`` (or an empty ``b_pool`` column) is left unchanged."""
    rng = np.random.default_rng(seed)
    out = base_marg.copy()
    n = len(out)
    for c in x_cols:
        if c not in b_pool.columns:
            continue
        src = b_pool[c].to_numpy()
        if len(src) == 0:
            continue
        out[c] = src[rng.integers(0, len(src), n)]
    return out
