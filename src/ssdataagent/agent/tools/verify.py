"""Verify-family tools — score the in-progress chain against held_out.

These are local proxies for SSDataBench's T1/T2/T4 metrics:
- T1 (univariate distributions) → KS for numeric, TV-distance for categorical
- T2 (pairwise associations) → |Δ pearson| numeric, |Δ Cramér's V| categorical
- T4 (event chronology) → compliance rate for an ordered tuple of age columns

Pass thresholds match EXP-002's planned hard thresholds:
- T1 numeric: KS ≤ 0.10
- T1 categorical: TV ≤ 0.10
- T2: |Δr| or |ΔV| ≤ 0.15
- T4: compliance ≥ 0.80

These are heuristics; the real scoring runs in SSDataBench after commit.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ssdataagent.agent.tools.inspect import _to_jsonable, describe_column
from ssdataagent.agent.tools.state import RuntimeState, data_withheld_error


T1_NUMERIC_THRESHOLD = 0.10        # KS statistic ≤ this = pass
T1_CATEGORICAL_THRESHOLD = 0.10    # TV distance ≤ this = pass
T2_THRESHOLD = 0.15                # |Δr| or |ΔV| ≤ this = pass
T4_THRESHOLD = 0.80                # compliance rate ≥ this = pass
# T3 compares the R² of `col ~ given` between real and sim via a delta-method
# z-test, and scores the fraction of bootstrap iterations that fail to reject.
# n=500 is the eval's bootstrap_sample_n. Reproduced here so the agent can see
# the thing it is actually graded on instead of inferring it.
T3_BOOTSTRAP_N = 500
T3_ALPHA = 0.05


# ===================== sampling helper =====================


def _preview_sample(state: RuntimeState, n_rows: int) -> tuple[pd.DataFrame | None, dict | None]:
    """Sample n_rows from the chain. If any col in generation_order isn't
    fit yet, fill it with empirical from train_fit so the preview can run.
    Returns (df, error_dict). Exactly one is None."""
    chain = state.chain
    if not chain.generation_order:
        return None, {
            "error": "empty_chain",
            "details": "no generation_order yet; call set_generation_order + fit_marginal at least once",
        }
    # Fill any unfit columns with a temporary empirical Step so sample() runs.
    from ssdataagent.agent.tools.fit import MarginalStep
    saved_steps = dict(chain.steps)
    try:
        for col in chain.generation_order:
            if col not in chain.steps:
                if col not in state.train_fit.columns:
                    return None, {
                        "error": "unknown_column",
                        "details": f"generation_order references unknown column {col!r}",
                    }
                nn = state.train_fit[col].dropna()
                if len(nn) == 0:
                    return None, {
                        "error": "all_missing",
                        "details": f"column {col!r} is entirely NaN; can't preview",
                    }
                stub = MarginalStep(col=col, family="empirical")
                stub.fit_values = nn.to_numpy()
                chain.add(stub)
        df = chain.sample(state.rng, n_rows)
    finally:
        chain.steps = saved_steps
    return df, None


def _refusal(state: RuntimeState) -> dict | None:
    return data_withheld_error() if not state.has_data else None


# ===================== sample_preview =====================


def sample_preview(state: RuntimeState, n: int = 100) -> dict:
    """Sample n rows from the in-progress chain and return per-column summaries.

    Output shape matches `describe_column` per col so the agent can directly
    compare sim vs real.
    """
    if (refusal := _refusal(state)) is not None:
        return refusal
    if n < 10 or n > 1000:
        return {"error": "bad_n", "details": "n must be between 10 and 1000"}
    sim, err = _preview_sample(state, n)
    if err is not None:
        return err
    summaries: dict[str, dict] = {}
    for col in sim.columns:
        s = sim[col]
        if pd.api.types.is_numeric_dtype(s):
            nn = pd.to_numeric(s, errors="coerce").dropna()
            summaries[col] = {
                "kind": "numeric",
                "n_missing": int(s.isna().sum()),
                "mean": _to_jsonable(nn.mean()) if len(nn) else None,
                "std": _to_jsonable(nn.std()) if len(nn) else None,
                "min": _to_jsonable(nn.min()) if len(nn) else None,
                "max": _to_jsonable(nn.max()) if len(nn) else None,
            }
        else:
            counts = s.value_counts(dropna=True).head(10)
            summaries[col] = {
                "kind": "categorical",
                "n_missing": int(s.isna().sum()),
                "top_values": [{"value": _to_jsonable(v), "count": int(c)} for v, c in counts.items()],
            }
    return {"n": int(len(sim)), "columns": summaries}


# ===================== score_marginal =====================


def _ks_score(real: pd.Series, sim: pd.Series) -> dict:
    real_nn = pd.to_numeric(real, errors="coerce").dropna()
    sim_nn = pd.to_numeric(sim, errors="coerce").dropna()
    if len(real_nn) < 5 or len(sim_nn) < 5:
        return {"error": "too_few_obs", "details": f"need ≥5 each; have real={len(real_nn)} sim={len(sim_nn)}"}
    stat, pvalue = ks_2samp(real_nn.to_numpy(), sim_nn.to_numpy())
    return {
        "metric": "ks",
        "ks": _to_jsonable(stat),
        "p_value": _to_jsonable(pvalue),
        "pass": bool(stat <= T1_NUMERIC_THRESHOLD),
        "threshold": T1_NUMERIC_THRESHOLD,
        "n_real": int(len(real_nn)),
        "n_sim": int(len(sim_nn)),
    }


def _tv_score(real: pd.Series, sim: pd.Series) -> dict:
    real_counts = real.dropna().value_counts(normalize=True)
    sim_counts = sim.dropna().value_counts(normalize=True)
    cats = sorted(set(real_counts.index) | set(sim_counts.index), key=str)
    real_p = np.array([real_counts.get(c, 0.0) for c in cats])
    sim_p = np.array([sim_counts.get(c, 0.0) for c in cats])
    tv = 0.5 * float(np.abs(real_p - sim_p).sum())
    return {
        "metric": "tv_distance",
        "tv_distance": _to_jsonable(tv),
        "pass": bool(tv <= T1_CATEGORICAL_THRESHOLD),
        "threshold": T1_CATEGORICAL_THRESHOLD,
        "n_categories": int(len(cats)),
        "n_real": int(real.dropna().size),
        "n_sim": int(sim.dropna().size),
    }


def score_marginal(state: RuntimeState, col: str) -> dict:
    """KS or TV-distance between real (held_out) and sim (chain preview)."""
    if (refusal := _refusal(state)) is not None:
        return refusal
    if col not in state.held_out.columns:
        return {"error": "unknown_column", "details": f"col {col!r} not in held_out"}
    sim, err = _preview_sample(state, n_rows=max(200, len(state.held_out)))
    if err is not None:
        return err
    if col not in sim.columns:
        return {"error": "col_not_in_chain", "details": f"chain doesn't generate {col!r}; sim has {list(sim.columns)}"}
    real_s = state.held_out[col]
    sim_s = sim[col]
    if pd.api.types.is_numeric_dtype(real_s):
        out = _ks_score(real_s, sim_s)
    else:
        out = _tv_score(real_s, sim_s)
    out["col"] = col
    return out


# ===================== score_pair =====================


def _cramers_v(s1: pd.Series, s2: pd.Series) -> float:
    """Cramér's V on two categorical series."""
    ct = pd.crosstab(s1, s2)
    if ct.size == 0:
        return float("nan")
    chi2 = float(((ct - ct.values.sum() * (ct.sum(axis=1).values[:, None] / ct.values.sum())
                       * (ct.sum(axis=0).values[None, :] / ct.values.sum())) ** 2
                      / (ct.sum(axis=1).values[:, None] / ct.values.sum() * ct.sum(axis=0).values[None, :]).clip(min=1e-12)
                      ).sum().sum())
    n = float(ct.values.sum())
    r, k = ct.shape
    denom = n * max(1, min(r - 1, k - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def score_pair(state: RuntimeState, col1: str, col2: str) -> dict:
    """|Δ correlation| (numeric pair) or |Δ Cramér's V| (categorical pair)."""
    if (refusal := _refusal(state)) is not None:
        return refusal
    for c in (col1, col2):
        if c not in state.held_out.columns:
            return {"error": "unknown_column", "details": f"col {c!r} not in held_out"}
    sim, err = _preview_sample(state, n_rows=max(200, len(state.held_out)))
    if err is not None:
        return err
    real_pair = state.held_out[[col1, col2]].dropna()
    sim_pair = sim[[col1, col2]].dropna()
    if len(real_pair) < 5 or len(sim_pair) < 5:
        return {"error": "too_few_obs", "details": f"real={len(real_pair)} sim={len(sim_pair)}"}

    col1_num = pd.api.types.is_numeric_dtype(real_pair[col1])
    col2_num = pd.api.types.is_numeric_dtype(real_pair[col2])
    if col1_num and col2_num:
        r_real = real_pair[col1].corr(real_pair[col2])
        r_sim = sim_pair[col1].corr(sim_pair[col2])
        delta = abs(float(r_real) - float(r_sim))
        return {
            "col1": col1, "col2": col2, "metric": "abs_delta_pearson",
            "real": _to_jsonable(r_real), "sim": _to_jsonable(r_sim),
            "abs_delta": _to_jsonable(delta),
            "pass": bool(delta <= T2_THRESHOLD), "threshold": T2_THRESHOLD,
        }
    if not col1_num and not col2_num:
        v_real = _cramers_v(real_pair[col1], real_pair[col2])
        v_sim = _cramers_v(sim_pair[col1], sim_pair[col2])
        delta = abs(v_real - v_sim) if not (np.isnan(v_real) or np.isnan(v_sim)) else float("nan")
        return {
            "col1": col1, "col2": col2, "metric": "abs_delta_cramers_v",
            "real": _to_jsonable(v_real), "sim": _to_jsonable(v_sim),
            "abs_delta": _to_jsonable(delta) if not np.isnan(delta) else None,
            "pass": bool(not np.isnan(delta) and delta <= T2_THRESHOLD),
            "threshold": T2_THRESHOLD,
        }
    return {
        "error": "mixed_dtypes",
        "details": "score_pair currently supports only numeric–numeric or categorical–categorical",
    }


# ===================== score_event_order =====================


def score_event_order(state: RuntimeState, events: list[str]) -> dict:
    """For an ordered list of age-event columns (e.g. age_started_work,
    age_at_first_marriage, age_at_first_child), compute the fraction of sim
    rows where the values are non-decreasing in the supplied order. Pass
    threshold matches the paper's strong bar."""
    if (refusal := _refusal(state)) is not None:
        return refusal
    if not isinstance(events, list) or len(events) < 2:
        return {"error": "bad_arguments", "details": "events must be a list of ≥2 column names"}
    sim, err = _preview_sample(state, n_rows=max(200, len(state.held_out)))
    if err is not None:
        return err
    for c in events:
        if c not in sim.columns:
            return {"error": "unknown_column", "details": f"event col {c!r} not in chain"}
    # Real survey data (addhealth, cfps) stores literal strings like
    # 'never married' / 'never sex' in event-age columns for people who
    # never had the event. Coerce non-numerics to NaN and drop those rows
    # so the order check operates only on participants who actually had
    # both events. (EXP-006f addhealth follow-up — was crashing here with
    # `ValueError: could not convert string to float`.)
    raw = sim[events]
    coerced = raw.apply(pd.to_numeric, errors="coerce")
    sub = coerced.dropna()
    n_excluded = int(len(raw) - len(sub))
    if len(sub) == 0:
        return {
            "error": "no_complete_rows",
            "details": (
                "no sim rows have all event cols numeric+non-NA; "
                f"{n_excluded} rows were excluded as missing or as non-numeric "
                "sentinels (e.g. 'never married'). Either the chain isn't "
                "producing event ages for enough participants, or all sampled "
                "rows are non-event sentinels."
            ),
        }
    arr = sub.to_numpy(dtype=float)
    compliant = np.all(np.diff(arr, axis=1) >= 0, axis=1)
    rate = float(compliant.mean())
    # Audit: a successful event-order check is what unblocks commit_generator
    # for longitudinal chains. Errors above (bad_arguments / unknown_column /
    # no_complete_rows) returned early and do not reach this point.
    state.event_order_calls.append(tuple(events))
    return {
        "events": list(events),
        "metric": "compliance_rate",
        "compliance_rate": _to_jsonable(rate),
        "pass": bool(rate >= T4_THRESHOLD),
        "threshold": T4_THRESHOLD,
        "n_complete": int(len(sub)),
        "n_excluded_non_numeric": n_excluded,
    }


# ===================== score_conditional (T3) =====================


def _r2(df: pd.DataFrame, col: str, given: list[str]) -> tuple[float | None, int]:
    """R² of an OLS of `col` on `given`, using the eval's own preparation:
    coerce the response to numeric (so censoring sentinels like 'never married'
    become NaN), then drop rows missing the response or any predictor. That
    dropna is why missingness matters so much — it decides which subpopulation
    the regression is even fit on."""
    import statsmodels.formula.api as smf

    d = df[[col] + given].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    for g in given:
        if not pd.api.types.is_numeric_dtype(d[g]):
            d[g] = d[g].astype(str).replace({"nan": np.nan}).astype("category")
    d = d.dropna()
    if len(d) < max(20, 2 * len(given) + 2) or d[col].nunique() < 3:
        return None, len(d)
    formula = f"Q('{col}') ~ " + " + ".join(f"Q('{g}')" for g in given)
    try:
        return float(smf.ols(formula, data=d).fit().rsquared), len(d)
    except Exception:
        return None, len(d)


def score_conditional(state: RuntimeState, col: str, given: list[str]) -> dict:
    """Score `col ~ given` the way SSDataBench's T3 does — by comparing the
    regression's R², not its coefficients.

    This is the benchmark the chain was flying blind on. A column drawn from a
    marginal has R² ≈ 0 against a real 0.2-0.5, which fails every bootstrap
    iteration and scores a flat 0.000. Equally, `n_sim` far exceeding `n_real`
    means the two regressions are fit on different subpopulations — usually
    because the sim lost the column's missingness.
    """
    from scipy.stats import norm

    if (refusal := _refusal(state)) is not None:
        return refusal
    if not isinstance(given, list) or not given:
        return {"error": "bad_arguments", "details": "given must be a non-empty list"}

    sim, err = _preview_sample(state, n_rows=max(1000, len(state.held_out)))
    if err is not None:
        return err
    for c in [col] + given:
        if c not in sim.columns:
            return {"error": "unknown_column", "details": f"{c!r} not in the chain"}
        if c not in state.held_out.columns:
            return {"error": "unknown_column", "details": f"{c!r} not in held_out"}

    r2_real, n_real = _r2(state.held_out, col, given)
    r2_sim, n_sim = _r2(sim, col, given)
    if r2_real is None:
        return {
            "error": "real_model_unfittable",
            "details": f"cannot fit {col!r} ~ {given} on held_out (n={n_real}); "
                       "the eval will skip this response too",
        }
    if r2_sim is None:
        return {
            "col": col, "given": given, "metric": "r2_match",
            "r2_real": _to_jsonable(r2_real), "r2_sim": None,
            "n_real": n_real, "n_sim": n_sim, "pass": False,
            "details": "the simulated rows can't support this regression at all",
        }

    n = T3_BOOTSTRAP_N
    se_real = np.sqrt(4 * r2_real * (1 - r2_real) ** 2 / n)
    se_sim = np.sqrt(4 * r2_sim * (1 - r2_sim) ** 2 / n)
    se = np.sqrt(se_real ** 2 + se_sim ** 2)
    z = (r2_real - r2_sim) / se if se > 0 else np.inf
    p = float(2 * norm.sf(abs(z)))

    # What matters is the FRACTION of rows that survive into the regression, not
    # the raw counts — held_out and the chain sample are different sizes, so raw
    # counts would always look mismatched.
    keep_real = n_real / max(len(state.held_out), 1)
    keep_sim = n_sim / max(len(sim), 1)

    out = {
        "col": col, "given": given, "metric": "r2_match",
        "r2_real": _to_jsonable(r2_real), "r2_sim": _to_jsonable(r2_sim),
        "abs_delta": _to_jsonable(abs(r2_real - r2_sim)),
        "p_value": _to_jsonable(p),
        "pass": bool(p > T3_ALPHA),
        "n_real": int(n_real), "n_sim": int(n_sim),
        "scored_fraction_real": _to_jsonable(keep_real),
        "scored_fraction_sim": _to_jsonable(keep_sim),
    }
    if r2_real > 0.02 and r2_sim < 0.25 * r2_real:
        out["diagnosis"] = (
            f"sim R²={r2_sim:.3f} vs real {r2_real:.3f} — {col!r} is close to "
            "independent of its predictors. A marginal step, or an empirical_lookup "
            "that is falling back to the global pool, would do exactly this."
        )
    elif keep_real > 0 and (keep_sim > 1.25 * keep_real or keep_sim < 0.8 * keep_real):
        out["diagnosis"] = (
            f"{keep_sim:.0%} of simulated rows enter this regression vs {keep_real:.0%} "
            f"of real ones — it is being fit on a different subpopulation. Usually "
            f"{col!r} lost its missingness, so rows that should have dropped out of the "
            "test are still in it."
        )
    return out


# ===================== score_overall =====================


# Cap the predictor set: T3's own configs use a handful of demographic /
# attainment covariates, and a long `given` thins the regression's rows for no gain.
_MAX_T3_PREDICTORS = 3


def _conditional_targets(state: RuntimeState) -> list[tuple[str, list[str]]]:
    """(response, predictors) pairs the composite score always evaluates.

    Derived from the DATA, not from the chain: any numeric column whose R² on the
    leading columns of the generation order is meaningfully above zero has real
    structure that a good chain owes us. Reading targets off the chain instead
    would let the agent delete a conditional and thereby delete its own penalty.
    """
    order = state.chain.generation_order
    held = state.held_out
    predictors = [c for c in order[:6] if c in held.columns and held[c].nunique() > 1]
    targets: list[tuple[str, list[str]]] = []
    for col in order:
        if col not in held.columns:
            continue
        num = pd.to_numeric(held[col], errors="coerce")
        if num.notna().sum() < 30 or num.nunique() < 5:
            continue  # T3 only regresses numeric responses with spread
        given = [p for p in predictors if p != col][:_MAX_T3_PREDICTORS]
        if not given:
            continue
        r2, _ = _r2(held, col, given)
        if r2 is not None and r2 > 0.02:   # there IS structure here to reproduce
            targets.append((col, given))
    return targets


def score_overall(state: RuntimeState) -> dict:
    """Composite health check for the in-progress chain: marginals (T1) AND the
    conditional structure (T3) that the benchmark also grades.

    It used to report marginal pass-rate ALONE, and that was actively harmful.
    Replacing a conditional step with `fit_marginal` makes a column's marginal
    perfect *by construction* — so under a marginal-only score, deleting the
    model is always the winning move. On both acs and cfps the agent did exactly
    that, wiping out the conditional structure for precisely the T3 response
    variables and scoring 0.000. It was not being careless; it was maximizing the
    number we gave it. So the number now includes what it costs.
    """
    if (refusal := _refusal(state)) is not None:
        return refusal
    if not state.chain.generation_order:
        return {"error": "empty_chain", "details": "no generation_order yet"}
    rows = []
    n_pass = 0
    for col in state.chain.generation_order:
        out = score_marginal(state, col)
        rows.append({"col": col, **{k: v for k, v in out.items() if k != "col"}})
        if out.get("pass"):
            n_pass += 1
    marginal_rate = n_pass / len(rows) if rows else None

    # T3 side. The response set is FIXED — every numeric column that has real
    # conditional structure — and is NOT read off the current chain. If we only
    # scored steps that still have parents, then deleting a conditional would
    # delete its own penalty, and the Goodhart loophole would simply reopen: a
    # chain of pure marginals would score a perfect conditional rate on the empty
    # set. A column with structure in the data owes an explanation either way.
    cond_rows, cond_pass = [], 0
    for col, given in _conditional_targets(state):
        c = score_conditional(state, col, given)
        if c.get("error"):
            continue
        cond_rows.append({k: c.get(k) for k in
                          ("col", "given", "r2_real", "r2_sim", "pass", "diagnosis")
                          if c.get(k) is not None})
        cond_pass += bool(c.get("pass"))
    conditional_rate = cond_pass / len(cond_rows) if cond_rows else None

    parts = [r for r in (marginal_rate, conditional_rate) if r is not None]
    result: dict[str, Any] = {
        "composite_score": _to_jsonable(sum(parts) / len(parts)) if parts else None,
        "marginal_pass_rate": _to_jsonable(marginal_rate),
        "conditional_pass_rate": _to_jsonable(conditional_rate),
        "n_columns": len(rows),
        "n_pass": n_pass,
        "pass_rate": _to_jsonable(marginal_rate),  # kept for backwards compatibility
        "rows": rows,
        "conditional_rows": cond_rows,
        "note": (
            "composite_score averages the marginal (T1) and conditional (T3) pass "
            "rates. Swapping a conditional step for fit_marginal raises the first "
            "and destroys the second — the trade is visible here, so make it "
            "deliberately."
        ),
    }
    # Hint for longitudinal chains — only when there's actually chronology to check.
    from ssdataagent.agent.tools.commit import _event_age_columns
    event_cols = _event_age_columns(state.chain.generation_order)
    if len(event_cols) >= 2:
        result["event_age_columns"] = sorted(event_cols)
        result["chronology_hint"] = (
            "score_overall covers per-column marginals only (T1). This chain has "
            f"{len(event_cols)} event-age columns ({sorted(event_cols)}); call "
            "score_event_order with them in chronological order to check T4 "
            "(event ordering) before commit_generator."
        )
    return result
