# src/ssdataagent/transfer/decompose.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ssdataagent.data.conditional_variance import _dummy_design

_EPS = 1e-9


def _is_num(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _edges(a_col: pd.Series, b_col: pd.Series, bins: int) -> np.ndarray:
    pooled = pd.to_numeric(pd.concat([a_col, b_col]), errors="coerce").dropna()
    if len(pooled) == 0:
        return np.array([])
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    return np.unique(np.quantile(pooled, qs))


def _codes(s: pd.Series, num: bool, edges: np.ndarray | None) -> np.ndarray:
    if num:
        v = pd.to_numeric(s, errors="coerce")
        c = np.digitize(v.to_numpy(dtype=float), edges) if edges is not None and len(edges) \
            else np.zeros(len(v), dtype=int)
        c = c.astype(object)
        c[v.isna().to_numpy()] = "__nan__"
        return c.astype(str)
    return s.astype("string").fillna("__nan__").to_numpy().astype(str)


def raking_weights(a: pd.DataFrame, b: pd.DataFrame, covariates: list[str],
                   *, bins: int = 10, iters: int = 30) -> np.ndarray:
    """Per-row weights on A so its weighted covariate marginals match B's (IPF/raking)."""
    n = len(a)
    w = np.ones(n, dtype=float)
    specs = []
    for c in covariates:
        num = _is_num(a[c])
        edges = _edges(a[c], b[c], bins) if num else None
        a_codes = _codes(a[c], num, edges)
        b_codes = _codes(b[c], num, edges)
        target = pd.Series(b_codes).value_counts(normalize=True)
        specs.append((a_codes, target))
    for _ in range(iters):
        for a_codes, target in specs:
            cur = pd.Series(w).groupby(a_codes).sum()
            cur = cur / cur.sum()
            factor = {k: target.get(k, 1e-12) / max(cur.get(k, 1e-12), 1e-12)
                      for k in np.unique(a_codes)}
            w = w * np.array([factor[k] for k in a_codes])
            w = w * (n / w.sum())
    return w


def _weighted_props(vals: np.ndarray, w: np.ndarray) -> pd.Series:
    key = pd.Series(vals).astype("string").fillna("__nan__").to_numpy()
    s = pd.Series(w).groupby(key).sum()
    return s / s.sum()


def _tv(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return 0.5 * float((p.reindex(idx, fill_value=0.0) - q.reindex(idx, fill_value=0.0)).abs().sum())


def kob_decompose(a: pd.DataFrame, b: pd.DataFrame, response: str,
                  covariates: list[str]) -> dict:
    """DFL reweighting decomposition of the A->B gap in ``response``.

    composition_share = (gap_raw - gap_residual) / gap_raw, where gap_residual is the gap
    remaining after raking A's covariates to B's. Numeric response uses standardized
    1-Wasserstein; categorical uses total-variation distance.
    """
    num = _is_num(a[response]) and _is_num(b[response])
    w = raking_weights(a, b, covariates)
    if num:
        av = pd.to_numeric(a[response], errors="coerce")
        bv = pd.to_numeric(b[response], errors="coerce")
        oka, okb = av.notna().to_numpy(), bv.notna().to_numpy()
        avv, wv = av[oka].to_numpy(dtype=float), w[oka]
        bvv = bv[okb].to_numpy(dtype=float)
        if len(avv) == 0 or len(bvv) == 0:
            gap_raw = gap_res = np.nan
        else:
            sd = float(np.std(np.concatenate([avv, bvv]))) or 1.0
            gap_raw = wasserstein_distance(avv / sd, bvv / sd)
            gap_res = wasserstein_distance(avv / sd, bvv / sd, u_weights=wv)
    else:
        pa_raw = pd.Series(a[response]).astype("string").fillna("__nan__").value_counts(normalize=True)
        pb = pd.Series(b[response]).astype("string").fillna("__nan__").value_counts(normalize=True)
        pa_w = _weighted_props(a[response].to_numpy(), w)
        gap_raw = _tv(pa_raw, pb)
        gap_res = _tv(pa_w, pb)
    if not np.isfinite(gap_raw) or gap_raw < _EPS:
        share = np.nan
        label = "aligned"
    else:
        share = float(np.clip((gap_raw - gap_res) / gap_raw, 0.0, 1.0))
        label = "composition-dominated" if share >= 0.5 else "mechanism-shifted"
    return {
        "response": response,
        "composition_share": share,
        "mechanism_share": (1.0 - share) if np.isfinite(share) else np.nan,
        "gap_raw": gap_raw, "gap_residual": gap_res,
        "label": label, "method": "dfl",
    }


def oaxaca_blinder(a: pd.DataFrame, b: pd.DataFrame, response: str,
                   covariates: list[str], *,
                   numeric_predictors: frozenset[str] = frozenset()) -> dict:
    """Twofold Oaxaca-Blinder for a NUMERIC response (cross-check for kob_decompose).

    Builds a SHARED dummy design on A∪B so coefficient vectors are aligned, fits OLS
    separately on each, and splits the mean gap into endowment (composition) and
    coefficient (mechanism) terms with A as the reference.
    """
    both = pd.concat([a[covariates], b[covariates]], ignore_index=True)
    design, ok = _dummy_design(both, covariates, numeric_predictors)
    na = len(a)
    ya = pd.to_numeric(a[response], errors="coerce").to_numpy(dtype=float)
    yb = pd.to_numeric(b[response], errors="coerce").to_numpy(dtype=float)
    da = design.iloc[:na]
    db = design.iloc[na:]
    oka = ok[:na] & np.isfinite(ya)
    okb = ok[na:] & np.isfinite(yb)

    def _fit(d, y, m):
        X = np.column_stack([np.ones(int(m.sum())), d.loc[m].to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        return beta, X.mean(axis=0)

    beta_a, xbar_a = _fit(da, ya, oka)
    beta_b, xbar_b = _fit(db, yb, okb)
    endowment = float((xbar_b - xbar_a) @ beta_a)
    coefficient = float(xbar_b @ (beta_b - beta_a))
    denom = abs(endowment) + abs(coefficient)
    share = abs(endowment) / denom if denom > _EPS else np.nan
    return {"response": response, "endowment": endowment, "coefficient": coefficient,
            "composition_share_ob": share, "method": "oaxaca-blinder"}
