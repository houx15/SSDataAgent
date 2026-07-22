# src/ssdataagent/transfer/recalibrate.py
from __future__ import annotations

import numpy as np

from ssdataagent.transfer.gaussian_copula import nearest_correlation


def tau_to_r(tau: float) -> float:
    """Kendall's tau -> Gaussian latent correlation (the copula's closed form)."""
    return float(np.sin(np.pi * tau / 2.0))


def _target_entry(r_src: float, s: float, t: float, method: str) -> float:
    """New latent entry matching target magnitude ``t`` while keeping ``r_src``'s sign."""
    sign = 1.0 if r_src >= 0 else -1.0
    if method == "kendall":
        mag = abs(tau_to_r(t))
    else:  # cramers_v is unsigned: scale the current magnitude by the target/source ratio
        mag = abs(r_src) * (t / s) if s > 1e-9 else abs(r_src)
    return float(np.clip(sign * mag, -0.999, 0.999))


def recalibrate_matrix(R_source: np.ndarray, cols: list[str], a_src: dict,
                       a_tgt: dict, methods: dict, *,
                       threshold: float = 0.10) -> np.ndarray:
    """Edit unstable pairs of ``R_source`` toward the target association, then PSD-project.

    Stable (|a_tgt-a_src| < threshold), ``undefined``, and missing pairs keep the source
    entry. Unstable pairs move to the target magnitude, source sign preserved."""
    idx = {c: i for i, c in enumerate(cols)}
    R = np.array(R_source, dtype=float).copy()
    for key, method in methods.items():
        v1, v2 = key
        if v1 not in idx or v2 not in idx:
            continue
        s, t = a_src.get(key), a_tgt.get(key)
        if (method == "undefined" or s is None or t is None
                or not np.isfinite(s) or not np.isfinite(t)):
            continue
        if abs(t - s) < threshold:
            continue
        i, j = idx[v1], idx[v2]
        new = _target_entry(R[i, j], float(s), float(t), method)
        R[i, j] = R[j, i] = new
    return nearest_correlation(R)
