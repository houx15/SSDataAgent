"""Empirical-copula transfer: source A's full joint (shared-row resample) + target marginals.
Fixes the categorical dependence the one-factor glat mechanism loses. See
docs/superpowers/specs/2026-07-29-empirical-copula-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.generate import _is_numeric, _marginal_map


def empirical_transfer(source: pd.DataFrame, marg: pd.DataFrame, cols: list[str],
                       n: int, seed: int) -> pd.DataFrame:
    """A's empirical joint + ``marg``'s marginals. A single shared row-resample (``base``)
    supplies the copula: numeric columns keep the row's rank (as ``transfer_build``);
    categorical columns take the resampled row's ACTUAL category interval (joint-preserving)
    instead of the one-factor ``glat`` ordering. ``marg`` supplies the inverse-CDF value map
    and each column's missingness RATE. With ``marg is source`` this reproduces a direct
    row-resample of A's joint."""
    rng = np.random.default_rng(seed)
    m = len(source)
    num = {c: _is_numeric(source[c]) for c in cols}
    base = rng.integers(0, m, n)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        if num[c]:
            uf = (pd.to_numeric(source[c], errors="coerce")
                  .rank(pct=True, method="first").to_numpy(dtype=float)).copy()
            nan = np.isnan(uf)
            uf[nan] = rng.random(int(nan.sum()))
            u = uf[base]
        else:
            sv = source[c].to_numpy()[base]                          # resampled ACTUAL categories
            freq = source[c].dropna().astype(str).value_counts(normalize=True)  # frequency desc
            edges = np.concatenate([[0.0], np.cumsum(freq.to_numpy())])
            lo = {k: edges[i] for i, k in enumerate(freq.index)}
            wd = {k: edges[i + 1] - edges[i] for i, k in enumerate(freq.index)}
            key = pd.Series(sv).astype(str)                          # NaN -> "nan" (not a key)
            lo_a = key.map(lo).to_numpy(dtype=float)
            wd_a = key.map(wd).to_numpy(dtype=float)
            u = lo_a + rng.random(n) * wd_a
            bad = ~np.isfinite(u)                                    # NaN rows / unseen categories
            u[bad] = rng.random(int(bad.sum()))
        u = np.clip(u, 1e-6, 1 - 1e-6)
        em = _marginal_map(marg[c], u, num[c])
        miss = float(marg[c].isna().mean())
        if miss > 0:
            want = int(round(miss * n))
            mask = source[c].isna().to_numpy()[base].copy()
            have = int(mask.sum())
            if have > want:
                mask[rng.choice(np.flatnonzero(mask), have - want, replace=False)] = False
            elif have < want:
                free = np.flatnonzero(~mask)
                mask[rng.choice(free, min(want - have, len(free)), replace=False)] = True
            em[mask] = np.nan
        out[c] = em
    return pd.DataFrame(out)
