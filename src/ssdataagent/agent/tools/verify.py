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


# ===================== score_overall =====================


def score_overall(state: RuntimeState) -> dict:
    """Run score_marginal on every column in the chain. Returns a summary
    table. Cheap proxy for "how good is the in-progress chain right now".

    Note: this only covers per-column marginals (T1). When the chain has
    multiple event-age columns, we surface a `chronology_hint` so the agent
    doesn't treat this score as the final criterion — T4 (event ordering)
    requires `score_event_order`.
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
    result: dict[str, Any] = {
        "n_columns": len(rows),
        "n_pass": n_pass,
        "pass_rate": _to_jsonable(n_pass / len(rows)) if rows else None,
        "rows": rows,
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
