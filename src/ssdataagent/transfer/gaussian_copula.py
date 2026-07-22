# src/ssdataagent/transfer/gaussian_copula.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ssdataagent.transfer.generate import _is_numeric, _latent, _marginal_map

_EPS = 1e-6


def fit_latent_correlation(pool: pd.DataFrame, cols: list[str], *,
                           seed: int = 0) -> tuple[np.ndarray, dict[str, bool]]:
    """Latent Gaussian-copula correlation of ``cols`` in ``pool``.

    Each column is put through the same latent transform the generator uses
    (``_latent``: rank-copula for numeric, category-ordering for nominal), mapped to
    normal scores, and Pearson-correlated. This IS the source copula the generator draws
    from, so Step A's edits are on the same scale it will sample."""
    rng = np.random.default_rng(seed)
    num = {c: _is_numeric(pool[c]) for c in cols}
    znum = {c: pd.to_numeric(pool[c], errors="coerce").rank(pct=True)
            for c in cols if num[c]}
    glat = (pd.DataFrame(znum).mean(axis=1).fillna(0.5).to_numpy() if znum
            else np.full(len(pool), 0.5))
    z = np.column_stack([
        norm.ppf(np.clip(_latent(pool, c, num[c], glat, rng), _EPS, 1 - _EPS))
        for c in cols
    ])
    R = np.corrcoef(z, rowvar=False)
    if R.ndim == 0:                       # single column
        R = np.array([[1.0]])
    return np.nan_to_num(R, nan=0.0) * (1 - np.eye(len(cols))) + np.eye(len(cols)), num


def nearest_correlation(R: np.ndarray, *, eps: float = _EPS) -> np.ndarray:
    """Nearest PSD correlation matrix: symmetrize, clip eigenvalues to ``eps``, renormalize."""
    A = (R + R.T) / 2.0
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    A = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(A), eps, None))
    A = A / np.outer(d, d)
    np.fill_diagonal(A, 1.0)
    return A


def draw_copula(R: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` uniform draws from the Gaussian copula with correlation ``R``."""
    R = nearest_correlation(R)
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, R.shape[0])) @ L.T
    return np.clip(norm.cdf(z), _EPS, 1 - _EPS)


def copula_to_frame(u: np.ndarray, marg: pd.DataFrame, cols: list[str],
                    num: dict[str, bool], rng: np.random.Generator) -> pd.DataFrame:
    """Inverse-CDF each uniform column onto ``marg``'s marginal, applying its missingness rate."""
    n = u.shape[0]
    out: dict[str, np.ndarray] = {}
    for j, c in enumerate(cols):
        em = _marginal_map(marg[c], u[:, j], num[c])
        miss = float(marg[c].isna().mean())
        if miss > 0:
            mask = rng.random(n) < miss
            em[mask] = np.nan
        out[c] = em
    return pd.DataFrame(out)
