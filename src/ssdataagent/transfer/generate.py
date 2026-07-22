# src/ssdataagent/transfer/generate.py
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


def _is_numeric(s: pd.Series) -> bool:
    s = s.dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _latent(pool: pd.DataFrame, c: str, num: bool, glat: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    """Per-row uniform latent for column c, correlated across columns via glat.
    Ported verbatim from nodonor_bracket._latent so carryover mode matches build()."""
    m = len(pool)
    if num:
        u = np.array(pd.to_numeric(pool[c], errors="coerce")
                     .rank(pct=True, method="first").to_numpy(dtype=float))
        nan = np.isnan(u)
        u[nan] = rng.random(int(nan.sum()))
        return u
    s = pool[c].astype(str)
    order = (pd.Series(glat, index=range(m)).groupby(s.to_numpy()).mean()
             .sort_values().index.tolist())
    pos = {v: i for i, v in enumerate(order)}
    b = np.array(s.map(pos).to_numpy(dtype=float), dtype=float)
    nn = np.isnan(b)
    b[nn] = rng.integers(0, max(1, len(order)), int(nn.sum()))
    return pd.Series(b + rng.random(m)).rank(pct=True, method="first").to_numpy()


def _marginal_map(marg_col: pd.Series, u: np.ndarray, num: bool) -> np.ndarray:
    """Inverse-CDF: map uniforms u onto marg_col's empirical marginal (object array)."""
    if num:
        vals = np.sort(pd.to_numeric(marg_col, errors="coerce").dropna().to_numpy())
        if len(vals) == 0:
            return np.full(len(u), np.nan, dtype=object)
        return vals[np.clip((u * len(vals)).astype(int), 0, len(vals) - 1)].astype(object)
    vc = marg_col.dropna().astype(str).value_counts(normalize=True)
    if len(vc) == 0:
        return np.full(len(u), np.nan, dtype=object)
    cats, edges = vc.index.to_numpy(), np.cumsum(vc.to_numpy())
    return cats[np.searchsorted(edges, u, side="right").clip(0, len(cats) - 1)].astype(object)


def transfer_build(struct: pd.DataFrame, marg: pd.DataFrame, cols: list[str],
                   n: int, seed: int, mode: str) -> pd.DataFrame:
    """Copula from ``struct``, marginals from ``marg``.

    mode "carryover"     -- struct == marg (== source A). Equals
                            nodonor_bracket.build(A, ..., "copula-fixed").
    mode "marginal-swap" -- struct = A, marg = B. A's dependence, B's marginals.

    Numeric detection, the shared latent index (``base``), and each column's missingness
    PATTERN come from ``struct``; the inverse-CDF value map and the missingness RATE come
    from ``marg``. In carryover the two frames are identical, so the rng draw order matches
    build() exactly (asserted by test).
    """
    if mode not in ("carryover", "marginal-swap"):
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed)
    m = len(struct)
    num = {c: _is_numeric(struct[c]) for c in cols}
    znum = {c: pd.to_numeric(struct[c], errors="coerce").rank(pct=True)
            for c in cols if num[c]}
    glat = (pd.DataFrame(znum).mean(axis=1).fillna(0.5).to_numpy() if znum
            else np.full(m, 0.5))
    base = rng.integers(0, m, n)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        u = np.clip(_latent(struct, c, num[c], glat, rng)[base], 1e-6, 1 - 1e-6)
        em = _marginal_map(marg[c], u, num[c])
        miss = float(marg[c].isna().mean())
        if miss > 0:
            want = int(round(miss * n))
            mask = struct[c].isna().to_numpy()[base].copy()
            have = int(mask.sum())
            if have > want:
                mask[rng.choice(np.flatnonzero(mask), have - want, replace=False)] = False
            elif have < want:
                free = np.flatnonzero(~mask)
                mask[rng.choice(free, min(want - have, len(free)), replace=False)] = True
            em[mask] = np.nan
        out[c] = em
    return pd.DataFrame(out)


def transfer_build_b2(source_pool: pd.DataFrame, target_pool: pd.DataFrame,
                      cols: list[str], covariates: list[str], outcomes: list[str],
                      n: int, seed: int) -> pd.DataFrame:
    """B2 — source's shared-latent construction (``transfer_build``'s vehicle),
    recalibrated to the target's published aggregates (Amendment 1).

    Step A: fit a per-column coherence rate ``alpha_c`` (``fit_coherence_alphas``) so
    each pair's output association scales as ``alpha_c * alpha_d * a_src(c,d)`` toward
    the target pool's associations, then generate via ``transfer_build``'s shared
    ``base`` index -- kept for column ``c`` on a Bernoulli(``alpha_c``) fraction of
    rows, independently redrawn otherwise (the same mechanism
    ``conditional_variance.sample_variance_repaired`` uses per outcome). This
    preserves nominal structure the way the retired Gaussian-copula vehicle did not
    (every column still co-samples from the same source row, just less often).
    ``alpha <= 1`` so this can only weaken dependence toward the target, never
    manufacture dependence the source lacks -- see ``fit_coherence_alphas``.

    Step B (unchanged): nudge each numeric outcome's covariate-R^2 onto the target
    pool's R^2. Reads from the target only low-order aggregates of ``target_pool``
    (never its joint or a test sample).

    With every fitted ``alpha_c == 1.0`` and no outcomes to blend, this reproduces
    ``transfer_build(source_pool, target_pool, cols, n, seed, "marginal-swap")``
    exactly -- same values, same RNG consumption order (asserted by test): B2 is B1
    plus a knob.
    """
    from ssdataagent.transfer.copula_stability import pairwise_associations
    from ssdataagent.transfer.recalibrate import (
        bidirectional_r2_blend, fit_coherence_alphas,
    )
    from ssdataagent.transfer.target_aggregates import target_aggregates

    agg = target_aggregates(target_pool, cols, covariates, outcomes)
    src_assoc = pairwise_associations(source_pool, cols)
    a_src = {k: v[0] for k, v in src_assoc.items()}
    a_tgt = agg["pairwise_assoc"]
    # a pair is comparable only if source and target used the same association method
    methods = {
        k: (src_assoc[k][1] if src_assoc[k][1] == agg["pairwise_method"].get(k)
            else "undefined")
        for k in src_assoc
    }
    alphas, n_pairs_understrength = fit_coherence_alphas(cols, a_src, a_tgt, methods)
    _logger.info(
        "transfer_build_b2: fitted coherence alphas %s (n_pairs_understrength=%d)",
        {c: round(alphas[c], 4) for c in cols}, n_pairs_understrength,
    )

    # The shared-latent construction, mirroring transfer_build's marginal-swap path
    # exactly (same rng object, same call order) so that alpha_c == 1.0 for every
    # column reproduces it bit-for-bit. Numeric detection / the shared latent index /
    # each column's missingness PATTERN come from the SOURCE (as in transfer_build's
    # ``struct``); the inverse-CDF value map and missingness RATE come from the TARGET
    # (as in transfer_build's ``marg``) -- using the target's own numeric verdict for
    # target-side operations, since a shared column can legitimately flip numeric-ness
    # across pools and using the source's verdict there would corrupt the target
    # marginal (see _marginal_map: the numeric branch drops non-numeric rows first).
    source_num = {c: _is_numeric(source_pool[c]) for c in cols}
    target_num = {c: _is_numeric(target_pool[c]) for c in cols}
    for c in cols:
        if source_num.get(c) != target_num.get(c):
            _logger.warning(
                "transfer_build_b2: numeric-ness of column %r disagrees between "
                "source (%s) and target (%s); using the target's verdict for "
                "target-side operations (marginal map, R^2-blend predictor design)",
                c, source_num.get(c), target_num.get(c),
            )

    rng = np.random.default_rng(seed)
    m = len(source_pool)
    znum = {c: pd.to_numeric(source_pool[c], errors="coerce").rank(pct=True)
            for c in cols if source_num[c]}
    glat = (pd.DataFrame(znum).mean(axis=1).fillna(0.5).to_numpy() if znum
            else np.full(m, 0.5))
    base = rng.integers(0, m, n)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        u_full = _latent(source_pool, c, source_num[c], glat, rng)
        a = float(np.clip(alphas.get(c, 1.0), 0.0, 1.0))
        if a >= 1.0:
            # No extra rng draw when fully coherent -- this is what makes alpha_c ==
            # 1.0 for every column reproduce transfer_build's rng consumption exactly.
            idx = base
        else:
            keep = rng.random(n) < a
            idx = np.where(keep, base, rng.integers(0, m, n))
        u = np.clip(u_full[idx], 1e-6, 1 - 1e-6)
        em = _marginal_map(target_pool[c], u, target_num[c])
        miss = float(target_pool[c].isna().mean())
        if miss > 0:
            want = int(round(miss * n))
            mask = source_pool[c].isna().to_numpy()[idx].copy()
            have = int(mask.sum())
            if have > want:
                mask[rng.choice(np.flatnonzero(mask), have - want, replace=False)] = False
            elif have < want:
                free = np.flatnonzero(~mask)
                mask[rng.choice(free, min(want - have, len(free)), replace=False)] = True
            em[mask] = np.nan
        out[c] = em
    frame = pd.DataFrame(out)

    num_pred = frozenset(c for c in covariates if target_num.get(c, False))
    return bidirectional_r2_blend(frame, outcomes, covariates, agg["outcome_r2"],
                                  numeric_predictors=num_pred, rng=rng)
