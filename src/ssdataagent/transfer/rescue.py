"""B5 -- learned R^2 rescue (Phase 3, slice 2). A closed-form, numpy-only
empirical-Bayes model that predicts a target context's per-outcome covariate-R^2
by shrinking B4's same-instrument retrieval estimate toward a cross-context
pooled prior, weighted by retrieval reliability (ESS). LLM-free.

See docs/superpowers/specs/2026-07-23-b5-learned-r2-rescue-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.generate import _is_numeric

FEATURE_NAMES: tuple[str, ...] = ("entropy", "n_predictors", "is_numeric")


def _normalized_entropy(series: pd.Series, numeric: bool) -> float:
    """Shannon entropy of the univariate marginal, normalized to [0, 1]. Numeric
    columns are decile-binned first so a single 'spread/diversity' feature is
    comparable across numeric and categorical outcomes. Reads only the marginal."""
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    if numeric:
        v = pd.to_numeric(s, errors="coerce").dropna()
        if v.nunique() <= 1:
            return 0.0
        binned = pd.qcut(v, min(10, v.nunique()), duplicates="drop")
        counts = binned.value_counts()
    else:
        counts = s.value_counts()
    p = (counts / counts.sum()).to_numpy()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def outcome_features(pool: pd.DataFrame, outcome: str, predictors: list[str],
                     *, numeric_predictors: frozenset[str] = frozenset()) -> dict[str, float]:
    """Firewall-clean structural features of one outcome, from public marginals +
    crosswalk structure only. Never reads the joint. Keys: entropy (normalized
    marginal diversity), n_predictors (usable predictor count), is_numeric."""
    numeric = _is_numeric(pool[outcome])
    preds = [c for c in predictors if c in pool.columns]
    return {
        "entropy": _normalized_entropy(pool[outcome], numeric),
        "n_predictors": float(len(preds)),
        "is_numeric": 1.0 if numeric else 0.0,
    }
