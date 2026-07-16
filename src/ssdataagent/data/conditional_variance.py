"""Conditional-variance repair — the T2/T3 fix, sibling of event_order_knowledge.

A one-shot LLM generator emits, for each respondent, its *best single* value for
every trait: ``E[Y | covariates]``, the conditional mean. Do that for a whole
population and the outcomes become near-deterministic functions of the covariates:
residual (conditional) variance vanishes, so pairwise associations tighten (hurts
T2) and regression coefficients / R^2 inflate (hurts T3). This *mean-collapse* is a
single bug seen through two tests, and it is strictly *one-directional* — the joint
is always too strong, never too weak.

The repair keeps the LLM's coherent joint via a shared person index (a copula),
then, per outcome, blends toward an independent draw at a rate ``alpha`` chosen so
the outcome's covariate-R^2 lands on a target::

    alpha = clip(sqrt(R2_target / R2_own), 0, 1)

A per-row blend scales the predictor->outcome covariance by ``alpha`` and restores
residual variance in the same move (``beta ~ alpha * beta_coherent``). ``alpha`` is
per outcome, not global: genuinely predictable outcomes (cognition) keep ``alpha~1``
while fabricated-structure outcomes (attitudes) are crushed to ``alpha~0.2``.

Honesty ladder for the target R^2 (mirrors the event-order module):
  * ``R2_own``    — measured on our OWN generation, never the test. Always allowed.
  * elicited      — the LLM reasons as a demographer about effect sizes
                    (`elicit_r2_targets`); non-circular, it never reads pool/test.
  * pool          — the disjoint pool's covariate-R^2; a supplied low-order aggregate
                    (same footing as handing over marginals), a labeled ceiling.

Predictor columns are held fully coherent so real demographic structure is
preserved; only outcome columns are repaired.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_EPS = 1e-6


# --------------------------------------------------------------- covariate R^2 (T3)

def _dummy_design(frame: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Dummy-encoded predictor matrix and a row mask for complete predictor rows."""
    parts, ok = [], np.ones(len(frame), dtype=bool)
    for p in predictors:
        s = frame[p].astype("string")
        ok &= s.notna().to_numpy()
        parts.append(pd.get_dummies(s, prefix=p, dummy_na=False))
    design = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)
    return design, ok


def covariate_r2(
    frame: pd.DataFrame,
    response: str,
    predictors: list[str],
    *,
    log_vars: frozenset[str] = frozenset(),
    min_rows: int = 20,
) -> float | None:
    """R^2 of an OLS of ``response`` on the dummy-encoded ``predictors`` — the exact
    statistic the T3 benchmark compares. Returns None if too few complete rows."""
    if response not in frame.columns:
        return None
    y = pd.to_numeric(frame[response], errors="coerce")
    if response in log_vars:
        y = np.log(y.where(y > 0))
    design, ok = _dummy_design(frame, predictors)
    ok &= y.notna().to_numpy()
    if int(ok.sum()) < min_rows or design.shape[1] == 0:
        return None
    X = np.column_stack([np.ones(int(ok.sum())), design.loc[ok].to_numpy(dtype=float)])
    yv = y.to_numpy(dtype=float)[ok]
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    sst = float(((yv - yv.mean()) ** 2).sum())
    return 1.0 - float(resid @ resid) / sst if sst > 0 else np.nan


def repair_alpha(own_r2: float | None, target_r2: float | None) -> float:
    """Mixing rate that deflates a covariate-R^2 from ``own_r2`` to ``target_r2``:
    ``clip(sqrt(target/own), 0, 1)``. 1.0 (no repair) when own is ~0 or unknown, or
    when the target is not below own (the failure is one-directional)."""
    if own_r2 is None or target_r2 is None or own_r2 <= _EPS:
        return 1.0
    return float(np.clip(np.sqrt(max(target_r2, 0.0) / own_r2), 0.0, 1.0))


def variance_repair_alphas(
    raw: pd.DataFrame,
    predictors: list[str],
    targets: dict[str, float | None],
    *,
    log_vars: frozenset[str] = frozenset(),
) -> dict[str, float]:
    """Per-outcome ``alpha``: ``R2_own`` measured on ``raw`` (our own generation,
    never the test), deflated toward each outcome's ``targets`` value."""
    return {
        c: repair_alpha(covariate_r2(raw, c, predictors, log_vars=log_vars), t)
        for c, t in targets.items()
    }


# --------------------------------------------------------------- calibration + repair

def _is_numeric(series: pd.Series) -> bool:
    s = series.dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _column_latent(
    raw: pd.DataFrame,
    col: str,
    is_num: bool,
    global_latent: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """A uniform-in-[0,1] latent per raw respondent for ``col`` that preserves the
    raw generation's rank structure. Numeric: percentile rank of the value.
    Categorical: percentile rank of a per-category mean of ``global_latent`` plus
    jitter, so within-category ties break uniformly and the marginal maps exactly."""
    m = len(raw)
    if is_num:
        u = np.array(pd.to_numeric(raw[col], errors="coerce")
                     .rank(pct=True, method="first").to_numpy(), dtype=float)
        nan = np.isnan(u)
        u[nan] = rng.random(int(nan.sum()))
        return u
    s = raw[col].astype(str)
    order = (pd.Series(global_latent, index=range(m)).groupby(s.to_numpy()).mean()
             .sort_values().index.tolist())
    pos = {v: i for i, v in enumerate(order)}
    base = np.array(s.map(pos).to_numpy(dtype=float), dtype=float)
    nan = np.isnan(base)
    base[nan] = rng.integers(0, max(1, len(order)), int(nan.sum()))
    score = base + rng.random(m)
    return pd.Series(score).rank(pct=True, method="first").to_numpy()


def _map_to_marginal(
    pool_col: pd.Series,
    u: np.ndarray,
    is_num: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Map a uniform latent ``u`` onto ``pool_col``'s marginal (numeric: sorted-value
    inverse CDF; categorical: cumulative value_counts), reinjecting the column's
    missingness rate. This is the T1-locking calibration step."""
    miss = float(pool_col.isna().mean())
    n = len(u)
    if is_num:
        vals = np.sort(pd.to_numeric(pool_col, errors="coerce").dropna().to_numpy())
        emitted = vals[np.clip((u * len(vals)).astype(int), 0, len(vals) - 1)].astype(object)
    else:
        vc = pool_col.dropna().astype(str).value_counts(normalize=True)
        cats, edges = vc.index.to_numpy(), np.cumsum(vc.to_numpy())
        emitted = cats[np.searchsorted(edges, u, side="right").clip(0, len(cats) - 1)].astype(object)
    if miss > 0:
        k = int(round(miss * n))
        if k:
            emitted[rng.choice(n, k, replace=False)] = np.nan
    return emitted


def sample_variance_repaired(
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    columns: list[str],
    predictors: list[str],
    n_rows: int,
    rng: np.random.Generator,
    *,
    alpha: dict[str, float] | None = None,
    default_alpha: float = 0.5,
) -> pd.DataFrame:
    """Sample ``n_rows`` calibrated respondents from the raw LLM generation with
    per-outcome conditional-variance repair.

    Every column is mapped onto ``pool``'s marginal (locks T1). Columns are linked
    through a shared person index so the joint is preserved (a copula); each
    non-predictor column then keeps that link only on a per-row Bernoulli(alpha_c)
    fraction and is otherwise redrawn from an independent respondent, which scales
    its covariate association by ``alpha_c`` while restoring residual variance.
    Predictors stay fully coherent. ``alpha`` maps a column to its rate; columns
    absent from it use ``default_alpha`` (predictors are always 1.0)."""
    alpha = dict(alpha or {})
    coherent = set(predictors)
    columns = [c for c in columns if c in pool.columns]
    is_num = {c: _is_numeric(pool[c]) for c in columns}
    num_cols = [c for c in columns if is_num[c]]
    m = len(raw)

    znum = {c: pd.to_numeric(raw[c], errors="coerce").rank(pct=True) for c in num_cols
            if c in raw.columns}
    global_latent = (pd.DataFrame(znum).mean(axis=1).fillna(0.5).to_numpy()
                     if znum else np.full(m, 0.5))

    base_idx = rng.integers(0, m, n_rows)
    out: dict[str, np.ndarray] = {}
    for c in columns:
        u_full = (_column_latent(raw, c, is_num[c], global_latent, rng)
                  if c in raw.columns else rng.random(m))
        if c in coherent:
            idx = base_idx
        else:
            a = alpha.get(c, default_alpha)
            if a >= 1.0:
                idx = base_idx
            else:
                keep = rng.random(n_rows) < a
                idx = np.where(keep, base_idx, rng.integers(0, m, n_rows))
        u_r = np.clip(u_full[idx], _EPS, 1 - _EPS)
        out[c] = _map_to_marginal(pool[c], u_r, is_num[c], rng)
    return pd.DataFrame(out)


# --------------------------------------------------------------- LLM R^2 elicitation

_ELICIT_PROMPT = """You are a quantitative sociologist analysing a general-population
survey. For EACH outcome below, estimate R^2: the fraction of that outcome's variance
explained by an OLS regression on just these demographic predictors: {predictors}.
Nothing else is in the model.

Reason briefly, per outcome, from published effect sizes and general knowledge. Anchor
yourself: attitudes / personality / mindsets / mental-health self-reports are barely
predicted by demographics (R^2 ~ 0.00-0.10); fertility / sibling counts are modest
(~0.05-0.15); income and years of schooling are moderate (~0.2-0.4); cognitive test
scores can be high (~0.3-0.8, education is a strong correlate).

Outcomes:
{items}

End your reply with ONLY a JSON object mapping each outcome name to its R^2 in [0, 1]:
{{"outcome": r2, ...}}"""


def _extract_last_json(text: str) -> dict:
    matches = re.findall(r"\{[^{}]*\}", text)
    return json.loads(matches[-1]) if matches else json.loads(text)


def elicit_r2_targets(
    responses: list[str],
    predictors: list[str],
    client,
    model: str,
    *,
    glosses: dict[str, str] | None = None,
    cache_path: str | None = None,
) -> dict[str, float]:
    """Elicit a per-outcome covariate-R^2 envelope from the LLM reasoning as a
    demographer — the non-circular target for the repair. It never reads the pool or
    the test; only outcome names and optional short ``glosses`` are sent. Raw reply
    cached to ``cache_path`` so re-scoring needs no LLM call."""
    if cache_path and Path(cache_path).exists():
        raw = Path(cache_path).read_text()
    else:
        items = "\n".join(f"- {c}: {glosses.get(c, c)}" if glosses else f"- {c}"
                          for c in responses)
        resp = client.chat.completions.create(
            model=model, max_tokens=3000, temperature=0.2,
            messages=[{"role": "user", "content": _ELICIT_PROMPT.format(
                predictors=", ".join(predictors), items=items)}])
        raw = resp.choices[0].message.content or ""
        if cache_path:
            Path(cache_path).write_text(raw)
    return {k: float(v) for k, v in _extract_last_json(raw).items()}
