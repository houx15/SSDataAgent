"""Transfer characterization study: measure how heterogeneous contexts are and why.

Analyst-side (reads A and B freely) -- see
docs/superpowers/specs/2026-07-27-transfer-characterization-study-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ssdataagent.transfer.decompose import _is_num

_EPS = 1e-9


def marginal_distance(a_col: pd.Series, b_col: pd.Series) -> tuple[float, str]:
    """Distance between the marginal of ``a_col`` and ``b_col``. Both numeric ->
    standardized 1-Wasserstein (divide by pooled SD of the non-missing values);
    otherwise -> total variation (0.5 * sum|p-q|) with NaN bucketed as its own category."""
    if _is_num(a_col) and _is_num(b_col):
        av = pd.to_numeric(a_col, errors="coerce").dropna().to_numpy(dtype=float)
        bv = pd.to_numeric(b_col, errors="coerce").dropna().to_numpy(dtype=float)
        if len(av) == 0 or len(bv) == 0:
            return np.nan, "wasserstein"
        sd = float(np.std(np.concatenate([av, bv]))) or 1.0
        return float(wasserstein_distance(av / sd, bv / sd)), "wasserstein"
    pa = a_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    pb = b_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    tv = 0.5 * float((pa.reindex(idx, fill_value=0.0)
                      - pb.reindex(idx, fill_value=0.0)).abs().sum())
    return tv, "tv"


def shape_level_split(a: pd.DataFrame, b: pd.DataFrame, response: str, focal: str,
                      *, bins: int = 10) -> dict:
    """Split the A->B conditional-mean gap of a NUMERIC ``response`` over bins of ``focal``
    into a level term (mean gap) and a shape term (rms residual). g(x) = E_B[Y|x] - E_A[Y|x]
    over shared quantile bins of ``focal``; level = mean_x g(x); shape = rms_x(g(x)-level);
    shape_ratio = shape/(|level|+shape+eps). ~0 => pure level shift; ~1 => shape change.
    Bins with no data in either context are skipped."""
    fa = pd.to_numeric(a[focal], errors="coerce")
    fb = pd.to_numeric(b[focal], errors="coerce")
    ya = pd.to_numeric(a[response], errors="coerce")
    yb = pd.to_numeric(b[response], errors="coerce")
    pooled = pd.concat([fa, fb]).dropna()
    empty = {"response": response, "focal": focal, "level": np.nan,
             "shape": np.nan, "shape_ratio": np.nan, "n_bins": 0}
    if len(pooled) == 0:
        return empty
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(pooled, qs))
    ca = np.digitize(fa.to_numpy(dtype=float), edges)
    cb = np.digitize(fb.to_numpy(dtype=float), edges)
    fa_ok, fb_ok = fa.notna().to_numpy(), fb.notna().to_numpy()
    ya_ok, yb_ok = ya.notna().to_numpy(), yb.notna().to_numpy()
    yav, ybv = ya.to_numpy(dtype=float), yb.to_numpy(dtype=float)
    gaps = []
    for k in np.unique(np.concatenate([ca, cb])):
        ma = (ca == k) & fa_ok & ya_ok
        mb = (cb == k) & fb_ok & yb_ok
        if ma.sum() == 0 or mb.sum() == 0:
            continue
        gaps.append(float(ybv[mb].mean() - yav[ma].mean()))
    if not gaps:
        return empty
    g = np.array(gaps, dtype=float)
    level = float(g.mean())
    shape = float(np.sqrt(np.mean((g - level) ** 2)))
    return {"response": response, "focal": focal, "level": level, "shape": shape,
            "shape_ratio": float(shape / (abs(level) + shape + _EPS)), "n_bins": len(g)}
