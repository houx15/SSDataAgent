from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kendalltau


def _is_num(s: pd.Series) -> bool:
    s = s.dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if ct.to_numpy().sum() == 0 or min(ct.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    r, k = ct.shape
    denom = n * max(1, min(r - 1, k - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def pair_association(frame: pd.DataFrame, v1: str, v2: str) -> tuple[float, str]:
    """Copula probe for a variable pair. Both numeric/ordinal -> Kendall's tau
    (rank-based, marginal-invariant). Any nominal member -> Cramer's V on binned data."""
    s1, s2 = frame[v1], frame[v2]
    num1, num2 = _is_num(s1), _is_num(s2)
    if num1 and num2:
        a = pd.to_numeric(s1, errors="coerce")
        c = pd.to_numeric(s2, errors="coerce")
        ok = a.notna() & c.notna()
        if int(ok.sum()) < 10:
            return np.nan, "kendall"
        tau, _ = kendalltau(a[ok], c[ok])
        return (float(tau) if tau == tau else np.nan), "kendall"

    def _cat(s: pd.Series, num: bool) -> pd.Series:
        if num:
            v = pd.to_numeric(s, errors="coerce")
            try:
                return pd.qcut(v, 5, duplicates="drop").astype("string")
            except (ValueError, IndexError):
                return v.rank(pct=True).round(1).astype("string")
        return s.astype("string")

    c1, c2 = _cat(s1, num1), _cat(s2, num2)
    ok = c1.notna() & c2.notna()
    if int(ok.sum()) < 10:
        return np.nan, "cramers_v"
    return _cramers_v(c1[ok], c2[ok]), "cramers_v"


def copula_stability(a: pd.DataFrame, b: pd.DataFrame, cols: list[str],
                     *, threshold: float = 0.10) -> pd.DataFrame:
    """Per unordered variable pair: association in A vs B, and |delta| stability label."""
    rows = []
    for v1, v2 in itertools.combinations(cols, 2):
        assoc_a, method_a = pair_association(a, v1, v2)
        assoc_b, method_b = pair_association(b, v1, v2)
        method = method_a
        if method_a != method_b:
            # e.g. differential missingness pushed the pair below the numeric threshold in
            # one context only -> Kendall tau vs Cramer's V are not comparable. Do not diff
            # incomparable metrics; mark undefined.
            delta, label, method = np.nan, "undefined", f"{method_a}/{method_b}"
        elif np.isfinite(assoc_a) and np.isfinite(assoc_b):
            delta = abs(assoc_a - assoc_b)
            label = "stable" if delta < threshold else "shifted"
        else:
            delta, label = np.nan, "undefined"
        rows.append({"v1": v1, "v2": v2, "method": method,
                     "assoc_a": assoc_a, "assoc_b": assoc_b,
                     "abs_delta": delta, "label": label})
    return pd.DataFrame(rows, columns=["v1", "v2", "method", "assoc_a", "assoc_b",
                                       "abs_delta", "label"])
