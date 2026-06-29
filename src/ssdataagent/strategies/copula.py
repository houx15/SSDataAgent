from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-6


def build_cuts(train, cols, schema) -> dict:
    """Per-column inversion data. numeric -> sorted train values;
    categorical -> (categories, cumulative upper edges)."""
    cuts: dict[str, dict] = {}
    for c in cols:
        if c in schema.numeric_ranges:
            vals = pd.to_numeric(train[c], errors="coerce").dropna().to_numpy()
            cuts[c] = {"kind": "num", "sorted": np.sort(vals)}
        else:
            cats = schema.allowed_values.get(c) or sorted(train[c].dropna().unique().tolist())
            counts = train[c].value_counts()
            probs = np.array([max(counts.get(v, 0), 0) for v in cats], dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(cats), 1.0 / len(cats))
            cuts[c] = {"kind": "cat", "cats": list(cats), "cum": np.cumsum(probs)}
    return cuts


def latent_value(col_cut, value) -> float:
    if col_cut["kind"] == "num":
        s = col_cut["sorted"]
        if len(s) == 0 or pd.isna(value):
            return 0.0
        pos = int(np.searchsorted(s, float(value), side="right"))
        u = min(max((pos - 0.5) / len(s), EPS), 1 - EPS)
        return float(norm.ppf(u))
    cats, cum = col_cut["cats"], col_cut["cum"]
    if value not in cats:
        return 0.0
    i = cats.index(value)
    lo = cum[i - 1] if i > 0 else 0.0
    u = min(max((lo + cum[i]) / 2.0, EPS), 1 - EPS)
    return float(norm.ppf(u))


def latent_matrix(df, cols, cuts) -> np.ndarray:
    out = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        out[:, j] = [latent_value(cuts[c], v) for v in df[c].tolist()]
    return out


def invert(z_array, col_cut) -> list:
    u = np.clip(norm.cdf(z_array), EPS, 1 - EPS)
    if col_cut["kind"] == "num":
        s = col_cut["sorted"]
        return list(np.quantile(s, u)) if len(s) else [0.0] * len(u)
    cats, cum = col_cut["cats"], col_cut["cum"]
    idx = np.searchsorted(cum, u, side="left")
    idx = np.clip(idx, 0, len(cats) - 1)
    return [cats[i] for i in idx]


def make_pd(M, reg) -> np.ndarray:
    M = (M + M.T) / 2.0
    M = M + reg * np.eye(M.shape[0])
    w, V = np.linalg.eigh(M)
    w = np.clip(w, reg, None)
    return (V * w) @ V.T


def correlated_normal(Sigma, n_samples, rng) -> np.ndarray:
    """Draw n_samples rows from N(0, Sigma) using the Cholesky factor."""
    d = Sigma.shape[0]
    if d == 0:
        return np.zeros((n_samples, 0))
    L = np.linalg.cholesky(make_pd(Sigma, 1e-9))
    return rng.standard_normal((n_samples, d)) @ L.T
