# src/ssdataagent/transfer/recalibrate.py
from __future__ import annotations

import numpy as np

from ssdataagent.transfer.gaussian_copula import nearest_correlation


def tau_to_r(tau: float) -> float:
    """Kendall's tau -> Gaussian latent correlation (the copula's closed form)."""
    return float(np.sin(np.pi * tau / 2.0))


def _target_entry(r_src: float, s: float, t: float, method: str) -> float:
    """New latent entry matching target magnitude ``t`` while keeping ``r_src``'s sign.

    Caller (``recalibrate_matrix``) guarantees ``r_src != 0.0`` (sign is undefined at
    zero, see below) and, for ``method == "cramers_v"``, ``s >= 0.05`` (source too weak
    to scale by otherwise). Both guards live in the caller so they can ``continue``
    (keep the untouched source entry) rather than return a value here.
    """
    sign = 1.0 if r_src >= 0 else -1.0
    if method == "kendall":
        mag = abs(tau_to_r(t))
    else:  # cramers_v is unsigned: scale the current magnitude by the target/source ratio
        mag = abs(r_src) * (t / s)
        # The ratio can blow up (e.g. s=0.02, t=0.5 -> 25x) far past anything the
        # target association actually supports; cap it before the shared clip below
        # so a moderately-small `s` doesn't silently saturate at the clip bound.
        mag = min(mag, 0.95)
    return float(np.clip(sign * mag, -0.999, 0.999))


def recalibrate_matrix(R_source: np.ndarray, cols: list[str], a_src: dict,
                       a_tgt: dict, methods: dict, *,
                       threshold: float = 0.10) -> np.ndarray:
    """Edit unstable pairs of ``R_source`` toward the target association, then PSD-project.

    Stable (|a_tgt-a_src| < threshold) and missing pairs keep the source entry.
    Unstable pairs move to the target magnitude, source sign preserved.

    ``methods`` expects the single-method strings produced by
    ``copula_stability.pair_association`` (``"kendall"`` / ``"cramers_v"``). Any other
    value is treated as undefined and the pair is left alone -- this includes, but is
    not limited to, the literal ``"undefined"`` value and also
    ``copula_stability.copula_stability``'s concatenated ``"a/b"`` encoding of a
    source/target method mismatch (e.g. ``"kendall/cramers_v"``), which must never be
    diffed as if it were a single compatible method.
    """
    idx = {c: i for i, c in enumerate(cols)}
    R = np.array(R_source, dtype=float).copy()
    for key, method in methods.items():
        v1, v2 = key
        if v1 not in idx or v2 not in idx:
            continue
        if method not in ("kendall", "cramers_v"):
            # Defensive guard: only these two single-method strings are recognized.
            # Anything else (missing/"undefined"/a mismatch encoding like
            # "kendall/cramers_v") means we don't know how to safely compare source
            # and target, so keep the source entry untouched.
            continue
        s, t = a_src.get(key), a_tgt.get(key)
        if s is None or t is None or not np.isfinite(s) or not np.isfinite(t):
            continue
        if abs(t - s) < threshold:
            continue
        i, j = idx[v1], idx[v2]
        r_src = R[i, j]
        if r_src == 0.0:
            # A zero latent entry (e.g. from gaussian_copula.fit_latent_correlation's
            # nan_to_num(nan=0.0) for degenerate/zero-variance columns) carries no sign
            # information. "Sign is source-owned" -- with no source sign to preserve,
            # adopting the target's sign would violate that rule, so skip and keep the
            # (zero) source entry rather than defaulting to positive.
            continue
        if method == "cramers_v" and s < 0.05:
            # Source association below this floor is too weak to form a reliable
            # target/source scaling ratio (a small `s` in the denominator can blow the
            # ratio up arbitrarily) -- keep the source entry instead of scaling by it.
            continue
        new = _target_entry(r_src, float(s), float(t), method)
        R[i, j] = R[j, i] = new
    return nearest_correlation(R)
