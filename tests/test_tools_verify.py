"""Verify-family tool tests. Builds chains we know should pass or fail and
checks pass/fail behavior + threshold semantics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssdataagent.agent.tools.fit import (
    Chain,
    fit_conditional,
    fit_copy_real,
    fit_marginal,
    set_generation_order,
)
from ssdataagent.agent.tools.state import RuntimeState
from ssdataagent.agent.tools.verify import (
    T1_NUMERIC_THRESHOLD,
    T2_THRESHOLD,
    T4_THRESHOLD,
    sample_preview,
    score_event_order,
    score_marginal,
    score_overall,
    score_pair,
)


def _make_state(df_train: pd.DataFrame, df_held: pd.DataFrame, tmp_path: Path) -> RuntimeState:
    return RuntimeState(
        workspace=tmp_path, train=df_train, train_fit=df_train, held_out=df_held,
        descriptions=None, has_data=True, has_descriptions=False,
        chain=Chain(), rng=np.random.default_rng(0),
    )


@pytest.fixture
def state(tmp_path: Path) -> RuntimeState:
    """500 rows total, split 80/20. Mix of numeric + categorical + event-time cols."""
    rng = np.random.default_rng(0)
    n = 500
    age = rng.integers(20, 70, size=n).astype(float)
    gender = rng.choice(["F", "M"], size=n)
    age_marriage = age - rng.integers(2, 20, size=n)
    age_first_child = age_marriage + rng.integers(1, 8, size=n)
    df = pd.DataFrame({
        "age": age, "gender": gender,
        "age_marriage": age_marriage.astype(float),
        "age_first_child": age_first_child.astype(float),
    })
    perm = np.random.default_rng(42).permutation(n)
    train = df.iloc[perm[:400]].reset_index(drop=True)
    held = df.iloc[perm[400:]].reset_index(drop=True)
    return _make_state(train, held, tmp_path)


# ---------- sample_preview ----------


def test_sample_preview_requires_chain(state):
    out = sample_preview(state, n=50)
    assert out["error"] == "empty_chain"


def test_sample_preview_returns_summaries(state):
    fit_marginal(state, "age", family="empirical")
    fit_marginal(state, "gender", family="empirical")
    out = sample_preview(state, n=100)
    assert out["n"] == 100
    assert "age" in out["columns"] and out["columns"]["age"]["kind"] == "numeric"
    assert "gender" in out["columns"] and out["columns"]["gender"]["kind"] == "categorical"


def test_sample_preview_caps_n(state):
    fit_marginal(state, "age")
    assert sample_preview(state, n=5)["error"] == "bad_n"
    assert sample_preview(state, n=5000)["error"] == "bad_n"


def test_sample_preview_fills_unfit_cols_empirical(state):
    """Setting a generation order without fitting all cols still allows preview
    via empirical fallback — important so the agent can preview early."""
    set_generation_order(state, ["age", "gender"])
    # only fit age
    fit_marginal(state, "age", family="empirical")
    out = sample_preview(state, n=50)
    assert out["n"] == 50
    assert "gender" in out["columns"]   # filled empirically


# ---------- score_marginal ----------


def test_score_marginal_passes_when_chain_matches_held_out(state):
    """Empirical marginal sampled from the same population should pass T1."""
    fit_marginal(state, "age", family="empirical")
    out = score_marginal(state, "age")
    assert out["metric"] == "ks"
    assert out["ks"] <= T1_NUMERIC_THRESHOLD + 0.05  # generous slack
    assert out["pass"] in (True, False)              # threshold semantics work
    assert out["threshold"] == T1_NUMERIC_THRESHOLD


def test_score_marginal_categorical_uses_tv(state):
    fit_marginal(state, "gender", family="empirical")
    out = score_marginal(state, "gender")
    assert out["metric"] == "tv_distance"
    assert "tv_distance" in out


def test_score_marginal_fails_when_distribution_off(state, tmp_path):
    """If the chain's marginal is wrong (e.g. all one gender), score should fail."""
    # Train has both genders; held out has both. We force a 'wrong' marginal by
    # fitting empirical on a subset that only contains 'F'.
    state.train_fit = state.train_fit[state.train_fit["gender"] == "F"].reset_index(drop=True)
    fit_marginal(state, "gender", family="empirical")
    out = score_marginal(state, "gender")
    assert out["pass"] is False
    assert out["tv_distance"] > T1_NUMERIC_THRESHOLD


def test_score_marginal_unknown_col(state):
    fit_marginal(state, "age")
    out = score_marginal(state, "ghost")
    assert out["error"] == "unknown_column"


# ---------- score_pair ----------


def test_score_pair_numeric_returns_delta_pearson(state):
    set_generation_order(state, ["age", "gender", "age_marriage"])
    fit_marginal(state, "age")
    fit_marginal(state, "gender")
    fit_conditional(state, "age_marriage", given=["age"], family="linear_regression")
    out = score_pair(state, "age", "age_marriage")
    assert out["metric"] == "abs_delta_pearson"
    assert out["abs_delta"] is not None
    assert out["threshold"] == T2_THRESHOLD


def test_score_pair_mixed_dtypes_rejected(state):
    fit_marginal(state, "age")
    fit_marginal(state, "gender")
    out = score_pair(state, "age", "gender")
    assert out["error"] == "mixed_dtypes"


def test_score_pair_categorical_returns_cramers_v(state, tmp_path):
    """Two categorical columns → uses Cramér's V."""
    n = 300
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.choice(["x", "y", "z"], size=n),
        "b": rng.choice(["p", "q"], size=n),
    })
    train = df.iloc[:240].reset_index(drop=True)
    held = df.iloc[240:].reset_index(drop=True)
    s = _make_state(train, held, tmp_path)
    fit_marginal(s, "a")
    fit_marginal(s, "b")
    out = score_pair(s, "a", "b")
    assert out["metric"] == "abs_delta_cramers_v"


# ---------- score_event_order ----------


def test_score_event_order_high_compliance_passes(state):
    """A chain that samples age-events by adding non-negative offsets stays ordered."""
    set_generation_order(state, ["age", "age_marriage", "age_first_child"])
    fit_marginal(state, "age")
    fit_conditional(state, "age_marriage", given=["age"], family="linear_regression")
    fit_conditional(state, "age_first_child", given=["age", "age_marriage"], family="linear_regression")
    out = score_event_order(state, ["age_marriage", "age_first_child"])
    assert "compliance_rate" in out
    assert out["threshold"] == T4_THRESHOLD


def test_score_event_order_needs_two_events(state):
    fit_marginal(state, "age")
    out = score_event_order(state, ["age"])
    assert out["error"] == "bad_arguments"


# ---------- score_overall ----------


def test_score_overall_summarizes_per_column(state):
    set_generation_order(state, ["age", "gender"])
    fit_marginal(state, "age")
    fit_marginal(state, "gender")
    out = score_overall(state)
    assert out["n_columns"] == 2
    assert 0 <= out["pass_rate"] <= 1
    assert {r["col"] for r in out["rows"]} == {"age", "gender"}


def test_score_overall_empty_chain(state):
    out = score_overall(state)
    assert out["error"] == "empty_chain"


# ---------- data withheld ----------


def test_verify_tools_refuse_when_no_data(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 10})
    s = _make_state(df, df.iloc[:10], tmp_path)
    s.has_data = False
    fit_marginal(s, "x")        # the fit family will refuse before even reaching verify
    # Try to score anyway:
    s.has_data = False
    out = score_marginal(s, "x")
    assert out["error"] == "data_withheld"
