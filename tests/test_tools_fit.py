"""Fit-family tool tests. Synthetic data with known structure so fits +
sampling can be checked against ground truth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssdataagent.agent.tools.fit import (
    Chain,
    ConditionalStep,
    CopyRealStep,
    MarginalStep,
    fit_conditional,
    fit_copy_real,
    fit_marginal,
    replace_step,
    set_generation_order,
)
from ssdataagent.agent.tools.state import RuntimeState


@pytest.fixture
def state(tmp_path: Path) -> RuntimeState:
    """Synthetic mini survey: 200 rows, gender→child_number→age_first_childbirth structure."""
    rng = np.random.default_rng(0)
    n = 200
    gender = rng.choice(["F", "M"], size=n, p=[0.55, 0.45])
    age = rng.integers(20, 70, size=n)
    child_number = rng.integers(0, 4, size=n)
    age_first = np.where(child_number > 0, age - rng.integers(2, 25, size=n), np.nan)
    df = pd.DataFrame({
        "gender": gender,
        "age": age.astype(float),
        "child_number": child_number.astype(float),
        "age_first_childbirth": age_first,
    })
    return RuntimeState(
        workspace=tmp_path,
        train=df, train_fit=df, held_out=df.iloc[:0],
        descriptions=None, has_data=True, has_descriptions=False,
        chain=Chain(),
        rng=np.random.default_rng(0),
    )


# ---------- set_generation_order ----------


def test_set_generation_order_basic(state):
    out = set_generation_order(state, ["gender", "age", "child_number", "age_first_childbirth"])
    assert out["set"] is True
    assert state.chain.generation_order == ["gender", "age", "child_number", "age_first_childbirth"]


def test_set_generation_order_unknown_col(state):
    out = set_generation_order(state, ["gender", "ghost"])
    assert out["error"] == "unknown_column"


def test_set_generation_order_duplicate(state):
    out = set_generation_order(state, ["gender", "gender"])
    assert out["error"] == "duplicate_columns"


def test_set_generation_order_replaces(state):
    set_generation_order(state, ["gender", "age"])
    set_generation_order(state, ["age", "gender"])
    assert state.chain.generation_order == ["age", "gender"]


# ---------- fit_marginal ----------


def test_fit_marginal_empirical_registers(state):
    out = fit_marginal(state, "gender", family="empirical")
    assert out["registered"] is True
    assert state.chain.has("gender")
    assert "gender" in state.chain.generation_order


def test_fit_marginal_kde_numeric(state):
    out = fit_marginal(state, "age", family="kde")
    assert out["registered"] is True
    step = state.chain.steps["age"]
    assert isinstance(step, MarginalStep)
    samples = step.sample(np.random.default_rng(0), 100, pd.DataFrame(index=range(100)))
    assert samples.shape == (100,)
    assert samples.mean() == pytest.approx(state.train_fit["age"].mean(), abs=10)


def test_fit_marginal_kde_rejects_categorical(state):
    out = fit_marginal(state, "gender", family="kde")
    assert out["error"] == "non_numeric_for_kde"


def test_fit_marginal_normal_numeric(state):
    out = fit_marginal(state, "age", family="normal")
    step = state.chain.steps["age"]
    assert step.normal_params is not None
    mu, _ = step.normal_params
    assert mu == pytest.approx(state.train_fit["age"].mean())


def test_fit_marginal_unknown_family(state):
    out = fit_marginal(state, "age", family="bogus")
    assert out["error"] == "unknown_family"


def test_fit_marginal_unknown_col(state):
    out = fit_marginal(state, "nope", family="empirical")
    assert out["error"] == "unknown_column"


# ---------- fit_conditional ----------


def test_fit_conditional_requires_set_generation_order_for_target(state):
    """`col` must be in the generation_order before fit_conditional."""
    fit_marginal(state, "gender")
    fit_marginal(state, "age")          # both auto-extend the order with themselves
    out = fit_conditional(state, "child_number", given=["gender"], family="empirical_lookup")
    assert out["error"] == "col_not_in_order"


def test_fit_conditional_rejects_given_after_col(state):
    set_generation_order(state, ["child_number", "gender"])  # gender LATER than child_number
    fit_marginal(state, "child_number")
    fit_marginal(state, "gender")
    out = fit_conditional(state, "child_number", given=["gender"], family="empirical_lookup")
    assert out["error"] == "given_after_col"


def test_fit_conditional_empirical_lookup_runs_and_samples(state):
    set_generation_order(state, ["gender", "child_number"])
    fit_marginal(state, "gender")
    out = fit_conditional(state, "child_number", given=["gender"], family="empirical_lookup")
    assert out["registered"] is True
    step = state.chain.steps["child_number"]
    assert isinstance(step, ConditionalStep)
    partial = pd.DataFrame({"gender": np.array(["F"] * 50 + ["M"] * 50)})
    values = step.sample(np.random.default_rng(0), 100, partial)
    assert len(values) == 100
    # Every sample comes from the lookup or fallback — should be a real child_number value.
    real_vals = set(state.train_fit["child_number"].dropna())
    assert set(values).issubset(real_vals)


def test_fit_conditional_linear_regression_numeric(state):
    """linear_regression on a numeric col fits successfully and reports R²."""
    set_generation_order(state, ["age", "child_number", "age_first_childbirth"])
    fit_marginal(state, "age")
    fit_marginal(state, "child_number")
    out = fit_conditional(
        state, "age_first_childbirth", given=["age", "child_number"],
        family="linear_regression",
    )
    assert out["registered"] is True
    assert out["score"] is not None
    assert -1.0 <= out["score"] <= 1.0
    # Sanity: the linear model was trained on the obs subset (no NaNs in age_first_childbirth).
    assert out["n_obs"] == int(state.train_fit["age_first_childbirth"].notna().sum())


def test_fit_conditional_linear_regression_rejects_categorical(state):
    set_generation_order(state, ["age", "gender"])
    fit_marginal(state, "age")
    out = fit_conditional(state, "gender", given=["age"], family="linear_regression")
    assert out["error"] == "family_dtype_mismatch"


def test_fit_conditional_logistic_regression_categorical(state):
    """logistic_regression on a categorical col fits and reports accuracy."""
    set_generation_order(state, ["age", "child_number", "gender"])
    fit_marginal(state, "age")
    fit_marginal(state, "child_number")
    out = fit_conditional(state, "gender", given=["age", "child_number"], family="logistic_regression")
    assert out["registered"] is True
    assert 0.0 <= out["score"] <= 1.0


def test_fit_conditional_allow_missing_builds_na_model(state):
    """When allow_missing=True and col has NAs, an na_model gets attached."""
    set_generation_order(state, ["child_number", "age_first_childbirth"])
    fit_marginal(state, "child_number")
    out = fit_conditional(
        state, "age_first_childbirth", given=["child_number"],
        family="empirical_lookup", allow_missing=True,
    )
    assert out["allow_missing"] is True
    step = state.chain.steps["age_first_childbirth"]
    assert step.na_model is not None
    # Sample and check NA mask actually injects None for child_number=0 rows.
    partial = pd.DataFrame({"child_number": np.array([0.0] * 50 + [2.0] * 50)})
    values = step.sample(np.random.default_rng(0), 100, partial)
    n_na = sum(1 for v in values if v is None)
    # First 50 rows have child_number=0 → most should be NA in train,
    # so the NA model should predict NA for many of them.
    assert n_na > 10


def test_fit_conditional_unknown_given(state):
    set_generation_order(state, ["gender", "age"])
    fit_marginal(state, "gender")
    out = fit_conditional(state, "age", given=["nope"], family="empirical_lookup")
    assert out["error"] == "unknown_column"


# ---------- fit_copy_real ----------


def test_fit_copy_real_registers(state):
    out = fit_copy_real(state, "gender")
    assert out["registered"] is True
    assert isinstance(state.chain.steps["gender"], CopyRealStep)


# ---------- replace_step ----------


def test_replace_step_drops_existing(state):
    fit_marginal(state, "age")
    out = replace_step(state, "age")
    assert out["was_present"] is True
    assert not state.chain.has("age")


def test_replace_step_idempotent(state):
    out = replace_step(state, "age")
    assert out["was_present"] is False


# ---------- end-to-end Chain.sample ----------


def test_chain_sample_full_pipeline(state):
    """Build a complete chain and draw N rows that look like the real data."""
    set_generation_order(state, ["gender", "age", "child_number", "age_first_childbirth"])
    fit_marginal(state, "gender", family="empirical")
    fit_marginal(state, "age", family="kde")
    fit_conditional(state, "child_number", given=["gender", "age"], family="empirical_lookup")
    fit_conditional(
        state, "age_first_childbirth", given=["age", "child_number"],
        family="linear_regression", allow_missing=True,
    )
    out = state.chain.sample(np.random.default_rng(123), 100)
    assert list(out.columns) == ["gender", "age", "child_number", "age_first_childbirth"]
    assert len(out) == 100
    # Marginals approximately right.
    assert set(out["gender"].unique()).issubset({"F", "M"})
    # Some NAs in age_first_childbirth (allow_missing should have produced them).
    n_na = out["age_first_childbirth"].isna().sum()
    assert n_na > 0
