from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ssdataagent.strategies import copula


def rake(cell_vectors, cell_weights, known_vec, *, max_iter: int = 50, tol: float = 1e-6):
    """IPF: scale each cell's prob vector so the cell-weighted mixture matches
    known_vec, preserving relative cross-cell differences. Per-cell vectors
    stay normalized."""
    known = np.asarray(known_vec, float)
    cells = list(cell_vectors)
    P = {c: np.asarray(cell_vectors[c], float).copy() for c in cells}
    total_w = sum(cell_weights[c] for c in cells) or 1.0
    w = {c: cell_weights[c] / total_w for c in cells}
    for _ in range(max_iter):
        mix = sum(w[c] * P[c] for c in cells)
        if np.max(np.abs(mix - known)) < tol:
            break
        ratio = np.divide(known, mix, out=np.ones_like(known), where=mix > 0)
        for c in cells:
            v = P[c] * ratio
            s = v.sum()
            if s > 0:
                P[c] = v / s
    return P


def build_target_copula(ref, targets, schema, *, reg: float = 1e-6) -> np.ndarray:
    """T×T copula correlation over targets: signed empirical correlation from
    `ref` microdata when available (A/B); identity (independence) when ref is
    None (C) or fewer than 2 targets."""
    t = len(targets)
    if ref is None or t < 2 or len(ref) < 2:
        return np.eye(t)
    cuts = copula.build_cuts(ref, list(targets), schema)
    Z = copula.latent_matrix(ref, list(targets), cuts)
    corr = np.corrcoef(Z, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return np.eye(t)
    return copula.make_pd(corr, reg)


def sample_targets(eval_cell_keys, calibrated, supports, Sigma, targets, *, seed: int = 42):
    """Draw a correlated latent per row, map each component through the row's
    cell's calibrated marginal. Categorical -> support member; numeric -> uniform
    within the chosen even-width bin."""
    rng = np.random.default_rng(seed)
    n, t = len(eval_cell_keys), len(targets)
    Z = copula.correlated_normal(Sigma, n, rng) if t else np.zeros((n, 0))
    U = np.clip(norm.cdf(Z), copula.EPS, 1 - copula.EPS)
    cums = {c: {tt: np.cumsum(calibrated[c][tt]) for tt in targets} for c in calibrated}
    out: dict[str, list] = {tt: [None] * n for tt in targets}
    for i in range(n):
        c = eval_cell_keys[i]
        for j, tt in enumerate(targets):
            cum = cums[c][tt]
            idx = int(np.searchsorted(cum, U[i, j], side="left"))
            idx = min(max(idx, 0), len(cum) - 1)
            sup = supports[tt]
            if sup["kind"] == "cat":
                out[tt][i] = sup["support"][idx]
            else:
                lo, hi = float(sup["edges"][idx]), float(sup["edges"][idx + 1])
                out[tt][i] = float(lo + rng.random() * (hi - lo))
    return out
