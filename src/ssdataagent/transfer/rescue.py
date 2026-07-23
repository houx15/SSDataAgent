"""B5 -- learned R^2 rescue (Phase 3, slice 2). A closed-form, numpy-only
empirical-Bayes model that predicts a target context's per-outcome covariate-R^2
by shrinking B4's same-instrument retrieval estimate toward a cross-context
pooled prior, weighted by retrieval reliability (ESS). LLM-free.

See docs/superpowers/specs/2026-07-23-b5-learned-r2-rescue-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PriorFit:
    """Cross-context pooled prior: mu(features) = intercept + standardized-feature
    slopes, with tau2 the between-context residual variance (empirical-Bayes)."""
    feature_names: tuple[str, ...]
    coef: np.ndarray        # length 1 + n_features (intercept first)
    mean: np.ndarray        # feature means (centering)
    scale: np.ndarray       # feature stds (scaling; zeros replaced by 1)
    tau2: float

    def predict(self, feats: dict) -> float:
        x = np.array([feats[n] for n in self.feature_names], dtype=float)
        xs = (x - self.mean) / self.scale
        return float(self.coef[0] + xs @ self.coef[1:])


@dataclass(frozen=True)
class NoiseFit:
    """Retrieval-noise curve sigma2(ess) = max(a + b/ess, floor). Fit where the
    per-sibling spread is measurable (cps pseudo-targets); extrapolated to gss."""
    a: float
    b: float
    floor: float = 1e-4

    def sigma2(self, ess: float) -> float:
        return max(self.a + self.b / max(ess, 1e-6), self.floor)


def fit_prior(rows: list[dict], *, tau2_floor: float = 1e-4) -> PriorFit:
    """GLS/OLS of true_r2 on standardized structural features across all training
    (context, outcome) rows. tau2 = residual variance (dof-corrected), floored."""
    F = np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["true_r2"] for r in rows], dtype=float)
    mean = F.mean(axis=0)
    scale = F.std(axis=0)
    scale[scale == 0] = 1.0
    Fs = (F - mean) / scale
    X = np.column_stack([np.ones(len(y)), Fs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    tau2 = max(float(resid @ resid) / dof, tau2_floor)
    return PriorFit(FEATURE_NAMES, coef, mean, scale, tau2)


def fit_noise(points: list[tuple[float, float]], *, floor: float = 1e-4) -> NoiseFit:
    """Fit sigma2(ess) = a + b/ess from (ess, squared_error) calibration points by
    OLS, clamping a, b >= 0. With a single point the fit is underdetermined, so all
    noise is attributed to the 1/ess term (a=0) -- the conservative choice that
    makes sigma2 grow as ess shrinks."""
    if not points:
        return NoiseFit(a=floor, b=0.0, floor=floor)
    if len(points) == 1:
        e0, se0 = points[0]
        return NoiseFit(a=0.0, b=max(se0 * max(e0, 1e-6), 0.0), floor=floor)
    E = np.array([[1.0, 1.0 / max(e, 1e-6)] for e, _ in points], dtype=float)
    y = np.array([se for _, se in points], dtype=float)
    ab, *_ = np.linalg.lstsq(E, y, rcond=None)
    return NoiseFit(a=max(float(ab[0]), 0.0), b=max(float(ab[1]), 0.0), floor=floor)


def predict_r2(x_co: float | None, ess: float, feats: dict,
               prior: PriorFit, noise: NoiseFit, *,
               clip: tuple[float, float] = (0.0, 1.0)) -> float:
    """Empirical-Bayes posterior R^2: precision-weighted blend of the retrieval
    estimate x_co (precision 1/sigma2(ess)) and the pooled prior mu (precision
    1/tau2). x_co None -> pure prior. Clipped to the unit interval."""
    mu = prior.predict(feats)
    if x_co is None:
        post = mu
    else:
        s2 = noise.sigma2(ess)
        t2 = prior.tau2
        post = (x_co / s2 + mu / t2) / (1.0 / s2 + 1.0 / t2)
    return float(np.clip(post, clip[0], clip[1]))
