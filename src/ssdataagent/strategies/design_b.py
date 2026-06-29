from __future__ import annotations

import numpy as np


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
