# src/ssdataagent/transfer/generate.py
from __future__ import annotations

import numpy as np
import pandas as pd


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
    """B2 — source copula recalibrated to the target's published aggregates.

    Step A: fit the source latent correlation, edit unstable pairs toward the target
    pool's associations (bidirectional, sign preserved), draw onto target marginals.
    Step B: nudge each numeric outcome's covariate-R^2 onto the target pool's R^2.
    Reads from the target only low-order aggregates of ``target_pool`` (never its joint
    or a test sample)."""
    from ssdataagent.transfer.copula_stability import pairwise_associations
    from ssdataagent.transfer.gaussian_copula import (
        copula_to_frame, draw_copula, fit_latent_correlation,
    )
    from ssdataagent.transfer.recalibrate import (
        bidirectional_r2_blend, recalibrate_matrix,
    )
    from ssdataagent.transfer.target_aggregates import target_aggregates

    R_source, num = fit_latent_correlation(source_pool, cols)
    src_assoc = pairwise_associations(source_pool, cols)
    agg = target_aggregates(target_pool, cols, covariates, outcomes)

    a_src = {k: v[0] for k, v in src_assoc.items()}
    a_tgt = agg["pairwise_assoc"]
    # a pair is comparable only if source and target used the same association method
    methods = {
        k: (src_assoc[k][1] if src_assoc[k][1] == agg["pairwise_method"].get(k)
            else "undefined")
        for k in src_assoc
    }
    R_prime = recalibrate_matrix(R_source, cols, a_src, a_tgt, methods)

    u = draw_copula(R_prime, n, seed)
    rng = np.random.default_rng(seed)
    frame = copula_to_frame(u, target_pool, cols, num, rng)

    num_pred = frozenset(c for c in covariates if num.get(c, False))
    return bidirectional_r2_blend(frame, outcomes, covariates, agg["outcome_r2"],
                                  numeric_predictors=num_pred, rng=rng)
