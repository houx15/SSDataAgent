"""Blind face-swap (Approach A): transfer source A's copula, get the target's marginals
from an LLM that reads only the target's textual description. See
docs/superpowers/specs/2026-07-26-blind-faceswap-design.md.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ssdataagent.transfer.generate import _is_numeric

_logger = logging.getLogger(__name__)


def _synth_numeric(quantiles: list[float], L: int) -> np.ndarray:
    """Length-L numeric column whose empirical distribution matches ``quantiles`` (values at
    evenly spaced probabilities 0..1). Inverse-CDF interpolation on a regular p-grid."""
    q = np.sort(np.asarray(quantiles, dtype=float))
    ps = np.linspace(0.0, 1.0, len(q))
    grid = (np.arange(L) + 0.5) / L
    return np.interp(grid, ps, q)


def _synth_categorical(probs: dict, L: int) -> np.ndarray:
    """Length-L object column whose value_counts(normalize) match ``probs`` (largest-remainder
    rounding, so the length is exactly L and the result is deterministic)."""
    cats = np.array(list(probs.keys()), dtype=object)
    p = np.asarray([probs[c] for c in probs.keys()], dtype=float)
    p = p / p.sum()
    exact = p * L
    counts = np.floor(exact).astype(int)
    rem = int(L - counts.sum())
    if rem > 0:
        counts[np.argsort(-(exact - counts))[:rem]] += 1
    return np.repeat(cats, counts)


def build_marg_frame(elicited: dict, source_a: pd.DataFrame, cols: list[str], *,
                     L: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthesize the ``marg`` frame for transfer_build from LLM-elicited distributions.
    Numeric-ness, the category universe, and each column's missingness RATE come from
    ``source_a`` (transferred structure); the distribution SHAPE comes from ``elicited``.
    A column absent/malformed in ``elicited`` falls back to A's own marginal (carry-over)."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        num = _is_numeric(source_a[c])
        dist = elicited.get(c)
        try:
            if dist is None:
                raise ValueError("no elicited distribution")
            col = (_synth_numeric(dist["quantiles"], L) if num
                   else _synth_categorical(dist["probs"], L)).astype(object)
        except (KeyError, ValueError, TypeError) as e:
            _logger.warning("blind: column %r falls back to A's marginal (%s)", c, e)
            vals = source_a[c].dropna().to_numpy()
            col = (vals[rng.integers(0, len(vals), L)].astype(object) if len(vals)
                   else np.full(L, np.nan, dtype=object))
        miss = float(source_a[c].isna().mean())          # carry A's missingness rate
        if miss > 0:
            k = int(round(miss * L))
            if k > 0:
                col = col.copy()
                col[rng.choice(L, min(k, L), replace=False)] = np.nan
        out[c] = col
    return pd.DataFrame(out)
