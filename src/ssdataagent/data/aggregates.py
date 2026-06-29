from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ssdataagent.data.schema import DatasetSchema


def _is_numeric(schema: DatasetSchema, col: str) -> bool:
    return col in schema.numeric_ranges


def marginals(df: pd.DataFrame, variables, schema: DatasetSchema, *, n_bins: int = 10) -> dict:
    """Univariate marginal per variable.
    Categorical -> {"kind": "categorical", "probs": {value: prob}} over allowed_values
      (missing categories at 0.0; normalized over non-null rows).
    Numeric -> {"kind": "numeric", "quantiles": {q: value}, "mean": float, "std": float}."""
    out: dict[str, dict] = {}
    for v in variables:
        if v not in df.columns:
            continue
        col = df[v].dropna()
        if _is_numeric(schema, v):
            x = pd.to_numeric(col, errors="coerce").dropna()
            if len(x) == 0:
                out[v] = {"kind": "numeric", "quantiles": {}, "mean": None, "std": None}
                continue
            qs = {round(float(q), 3): float(np.quantile(x, q))
                  for q in np.linspace(0, 1, n_bins + 1)}
            out[v] = {"kind": "numeric", "quantiles": qs,
                      "mean": float(x.mean()), "std": float(x.std(ddof=0))}
        else:
            cats = schema.allowed_values.get(v) or sorted(col.unique().tolist())
            counts = col.value_counts()
            total = float(counts.sum()) or 1.0
            out[v] = {"kind": "categorical",
                      "probs": {str(c): float(counts.get(c, 0)) / total for c in cats}}
    return out


def _cramers_v(a: pd.Series, b: pd.Series):
    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = float(tab.to_numpy().sum())
    denom = n * (min(tab.shape) - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(chi2 / denom))


def _corr_ratio(categories: pd.Series, values: pd.Series):
    vals = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"c": categories.to_numpy(), "x": vals.to_numpy()}).dropna()
    if frame["c"].nunique() < 2 or len(frame) < 2:
        return None
    grand = frame["x"].mean()
    ss_total = float(((frame["x"] - grand) ** 2).sum())
    if ss_total <= 0:
        return None
    ss_between = float(sum(len(g) * (g["x"].mean() - grand) ** 2
                          for _, g in frame.groupby("c")))
    return float(np.sqrt(ss_between / ss_total))


def associations(df: pd.DataFrame, target_variables, schema: DatasetSchema) -> dict:
    """Symmetric pairwise association among target variables in [0,1]:
    cat x cat -> Cramer's V; num x num -> |Pearson r|; cat x num -> correlation ratio eta.
    Degenerate/uncomputable pairs are omitted (never raises)."""
    out: dict[str, dict[str, float]] = {}
    tv = [v for v in target_variables if v in df.columns]
    for i, a in enumerate(tv):
        for b in tv[i + 1:]:
            sub = df[[a, b]].dropna()
            an, bn = _is_numeric(schema, a), _is_numeric(schema, b)
            if len(sub) < 2:
                val = None
            elif an and bn:
                x = pd.to_numeric(sub[a], errors="coerce")
                y = pd.to_numeric(sub[b], errors="coerce")
                val = None if x.std(ddof=0) == 0 or y.std(ddof=0) == 0 \
                    else float(abs(np.corrcoef(x, y)[0, 1]))
            elif not an and not bn:
                val = _cramers_v(sub[a], sub[b])
            else:
                cat, num = (a, b) if not an else (b, a)
                val = _corr_ratio(sub[cat], sub[num])
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            out.setdefault(a, {})[b] = val
            out.setdefault(b, {})[a] = val
    return out
