# src/ssdataagent/transfer/recalibrate.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.gaussian_copula import nearest_correlation
from ssdataagent.transfer.generate import _is_numeric, _marginal_map

_EPS = 1e-9


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


def _rank_pct(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True, method="first").to_numpy()


def _ols_prediction(frame: pd.DataFrame, y: str, predictors: list[str],
                    num_pred: frozenset[str]) -> np.ndarray | None:
    """Fitted E[y|X] over a one-hot / numeric design of its own -- NOT the same design
    as ``conditional_variance._dummy_design``: this mean-imputes numeric predictor NaNs
    and one-hot encodes with ``dummy_na=True`` (keeping every row), whereas
    ``_dummy_design`` drops any row with a missing predictor and uses ``dummy_na=False``.

    That's fine here: this only orders the "strengthen" pole (the fitted E[y|X] used to
    rank rows before blending), it does not measure R^2 itself. The bisection in
    ``bidirectional_r2_blend`` re-measures the true ``covariate_r2`` (via the real
    ``_dummy_design``) every iteration and is self-correcting, so a slightly different
    design here only perturbs the pole's row ordering, never the target R^2 achieved.
    """
    yv = pd.to_numeric(frame[y], errors="coerce")
    blocks = []
    for p in predictors:
        if p not in frame.columns:
            continue
        if p in num_pred or _is_numeric(frame[p]):
            v = pd.to_numeric(frame[p], errors="coerce")
            blocks.append(v.fillna(v.mean()).to_numpy().reshape(-1, 1))
        else:
            d = pd.get_dummies(frame[p].astype("string"), dummy_na=True, dtype=float)
            blocks.append(d.to_numpy())
    ok = yv.notna().to_numpy()
    if not blocks or int(ok.sum()) < 20:
        return None
    design = np.column_stack(blocks)
    X = np.column_stack([np.ones(len(frame)), design])
    beta, *_ = np.linalg.lstsq(X[ok], yv.to_numpy()[ok], rcond=None)
    return X @ beta


def bidirectional_r2_blend(frame: pd.DataFrame, outcomes: list[str],
                           predictors: list[str], r2_target: dict,
                           *, numeric_predictors: frozenset[str],
                           rng: np.random.Generator, iters: int = 16) -> pd.DataFrame:
    """Nudge each numeric outcome's covariate-R^2 onto its target by blending toward an
    independent pole (weaken) or the conditional-mean pole (strengthen), preserving the
    outcome's marginal via inverse-CDF back onto its own column."""
    frame = frame.copy()
    num_pred = frozenset(numeric_predictors)
    preds = [p for p in predictors if p in frame.columns]
    for y in outcomes:
        if y not in frame.columns or not _is_numeric(frame[y]):
            continue
        target = r2_target.get(y)
        own = covariate_r2(frame, y, preds, numeric_predictors=num_pred)
        # covariate_r2 returns nan (not None) for a constant-valued outcome (zero
        # variance -> sst == 0). A bare `own <= _EPS` guard never fires on nan (every
        # nan comparison is False), so a constant column would fall through into the
        # blend below; the marginal is degenerate so the VALUES wouldn't move, but the
        # dtype would still flip (e.g. float64 -> object) via the object-typed
        # `_marginal_map` output, breaking the "left completely unchanged" contract.
        # Guard against non-finite `own`/`target` explicitly so it never gets that far.
        if (target is None or own is None or not np.isfinite(own)
                or not np.isfinite(target) or own <= _EPS or abs(target - own) < 1e-3):
            continue
        # Restrict the whole blend (rank, pole, threshold draw, remap) to the rows
        # where the outcome is observed, so originally-missing rows are never touched:
        # they must come back exactly as NaN, not filled from the pole or pinned to
        # the marginal's minimum by an undefined-index cast.
        full = pd.to_numeric(frame[y], errors="coerce").to_numpy(dtype=float)
        obs_idx = np.flatnonzero(~np.isnan(full))
        if len(obs_idx) == 0:
            continue
        yv = full[obs_idx]
        u_cur = _rank_pct(yv)
        strengthen = target > own
        if strengthen:
            yhat = _ols_prediction(frame, y, preds, num_pred)
            if yhat is None:
                continue
            pole = _rank_pct(yhat[obs_idx])
        else:
            pole = _rank_pct(rng.permutation(len(yv)).astype(float))
        r = rng.random(len(yv))                         # fixed thresholds -> monotone in g
        marg = frame[y]
        lo, hi, best = 0.0, 1.0, full.copy()
        for _ in range(iters):
            g = (lo + hi) / 2.0
            u = np.where(r < g, pole, u_cur)
            remapped_obs = _marginal_map(marg, np.clip(u, 1e-6, 1 - 1e-6), True)
            candidate = full.copy()
            candidate[obs_idx] = remapped_obs
            tmp = frame.copy()
            tmp[y] = candidate
            r2 = covariate_r2(tmp, y, preds, numeric_predictors=num_pred)
            if r2 is None:
                break
            best = candidate
            need_more_pole = (r2 < target) if strengthen else (r2 > target)
            if need_more_pole:
                lo = g
            else:
                hi = g
        frame[y] = best
    return frame
