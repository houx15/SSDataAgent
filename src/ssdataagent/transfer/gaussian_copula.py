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


def nearest_correlation(R: np.ndarray, *, eps: float = _EPS, max_iter: int = 10) -> np.ndarray:
    """Nearest PSD correlation matrix: symmetrize, clip eigenvalues to ``eps``, renormalize.

    Contract: the returned matrix is symmetric, has a unit diagonal, and its
    minimum eigenvalue is >= ``eps``.

    A single clip -> reconstruct -> renormalize-diagonal pass is NOT enough:
    renormalizing the diagonal via ``A / outer(d, d)`` is a congruence
    transform (``A' = D A D`` for diagonal ``D``), which preserves
    positive-*definiteness* in sign but does not preserve the eigenvalue
    *floor* -- it can pull small eigenvalues back below ``eps``. So we
    iterate the whole pass, re-checking the floor each round, which recovers
    it for the vast majority of inputs within a couple of iterations.

    If ``max_iter`` rounds are not enough (pathological/ill-conditioned
    inputs), fall back to shrinking toward the identity matrix: for any
    symmetric ``A`` with unit diagonal and ``alpha`` in [0, 1],
    ``A' = (1 - alpha) * A + alpha * I`` shares A's eigenvectors (I is
    invariant under any orthogonal change of basis), so its eigenvalues are
    exactly ``(1 - alpha) * w_i + alpha``. Solving for the ``alpha`` that
    maps the current minimum eigenvalue to ``eps`` gives a closed-form,
    guaranteed fix -- no more iteration needed -- while also leaving the
    diagonal exactly 1 (since ``(1 - alpha) * 1 + alpha * 1 == 1``).
    """
    A = (R + R.T) / 2.0
    np.fill_diagonal(A, 1.0)
    n = A.shape[0]

    for _ in range(max_iter):
        w, V = np.linalg.eigh(A)
        w = np.clip(w, eps, None)
        A = (V * w) @ V.T
        d = np.sqrt(np.clip(np.diag(A), eps, None))
        A = A / np.outer(d, d)
        np.fill_diagonal(A, 1.0)
        A = (A + A.T) / 2.0
        if np.linalg.eigvalsh(A).min() >= eps:
            return A

    # Fallback: guaranteed shrinkage toward the identity (see docstring).
    w_min = np.linalg.eigvalsh(A).min()
    if w_min < eps:
        # w_min < 1 always holds here (trace(A) == n with unit diagonal, so
        # the minimum of n eigenvalues summing to n can't exceed 1), so this
        # division is safe. Tiny slack guards against floating-point rounding
        # landing exactly on the floor.
        alpha = (eps - w_min) / (1.0 - w_min)
        alpha = min(max(alpha, 0.0), 1.0) + 1e-12
        alpha = min(alpha, 1.0)
        A = (1.0 - alpha) * A + alpha * np.eye(n)
        np.fill_diagonal(A, 1.0)
        A = (A + A.T) / 2.0
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
