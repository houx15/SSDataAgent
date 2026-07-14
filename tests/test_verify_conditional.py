"""The instrument.

`score_overall` used to report the marginal pass-rate alone. Under that number,
replacing a conditional step with `fit_marginal` is always a winning move — an iid
draw from the real column reproduces its marginal *perfectly*. On both acs and cfps
the agent duly deleted the conditional models for exactly the T3 response variables
and scored 0.000. It was maximizing the number we gave it.

These tests pin the fix: the composite score must make that trade *visible*, and
`score_conditional` must expose the two failure signatures (structure lost,
subpopulation changed) that the old toolset could not see at all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssdataagent.agent.tools import dispatch
from ssdataagent.agent.tools.state import build_runtime_state


def _train(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """`income` genuinely depends on education and gender (R² well above zero),
    and is structurally missing for a third of rows."""
    rng = np.random.default_rng(seed)
    edu = rng.choice(["low", "mid", "high"], n, p=[0.4, 0.4, 0.2])
    gender = rng.choice(["Male", "Female"], n)
    base = pd.Series(edu).map({"low": 20.0, "mid": 35.0, "high": 60.0}).to_numpy()
    income = base + np.where(gender == "Male", 6.0, 0.0) + rng.normal(0, 6, n)
    income[rng.random(n) < 0.33] = np.nan  # structural missingness
    return pd.DataFrame({"gender": gender, "education": edu, "income": income})


@pytest.fixture()
def state(tmp_path: Path):
    st = build_runtime_state(
        workspace=tmp_path, train=_train(), descriptions=None,
        has_data=True, has_descriptions=True, seed=3,
    )
    dispatch(st, "set_generation_order", {"cols": ["gender", "education", "income"]})
    dispatch(st, "fit_marginal", {"col": "gender"})
    dispatch(st, "fit_marginal", {"col": "education"})
    return st


def _conditional(state):
    return dispatch(state, "fit_conditional", {
        "col": "income", "given": ["education", "gender"],
        "family": "linear_regression", "allow_missing": True,
    })


def test_score_conditional_reports_r2_of_both_sides(state):
    """T3 grades R², not coefficients — so the tool has to surface R²."""
    _conditional(state)
    out = dispatch(state, "score_conditional",
                   {"col": "income", "given": ["education", "gender"]})
    assert out.get("error") is None, out
    assert out["metric"] == "r2_match"
    assert out["r2_real"] > 0.5, "fixture must have real conditional structure"
    assert out["r2_sim"] > 0.3, f"a fitted conditional should carry it: {out}"


def test_score_conditional_catches_a_marginal_masquerading_as_a_model(state):
    """The exact bug: a marginal column has R² ~ 0 against a real R² ~ 0.8, which
    fails every bootstrap iteration and scores a flat 0.000. The old toolset had no
    way to see this."""
    dispatch(state, "fit_marginal", {"col": "income"})
    out = dispatch(state, "score_conditional",
                   {"col": "income", "given": ["education", "gender"]})
    assert out["pass"] is False
    assert out["r2_sim"] < 0.1, f"a marginal must show near-zero R²: {out}"
    assert "independent of its predictors" in out.get("diagnosis", "")


def test_score_conditional_flags_a_changed_subpopulation(state):
    """Missingness IS selection: the eval drops missing rows before regressing, so a
    column that loses its NaNs puts a different, larger population into the test.

    Fitted WITH the conditional (allow_missing=False) so the structure is intact and
    the only defect left is the lost missingness — otherwise a marginal would trip
    both signatures at once and we couldn't tell which one fired."""
    dispatch(state, "fit_conditional", {
        "col": "income", "given": ["education", "gender"],
        "family": "linear_regression", "allow_missing": False,
    })
    out = dispatch(state, "score_conditional",
                   {"col": "income", "given": ["education", "gender"]})
    assert out["r2_sim"] > 0.3, "structure must be intact, so only selection is wrong"
    assert out["scored_fraction_sim"] > 0.95, "the sim emits no NaN at all"
    assert out["scored_fraction_real"] < 0.75, "the fixture is ~33% missing"
    assert "subpopulation" in out.get("diagnosis", ""), out


def test_composite_score_falls_when_a_conditional_is_swapped_for_a_marginal(state):
    """The regression test for the Goodhart collapse.

    Under the OLD marginal-only score this swap looked like an improvement, which is
    why the agent kept making it. The composite must move the other way."""
    _conditional(state)
    before = dispatch(state, "score_overall", {})

    dispatch(state, "replace_step", {"col": "income"})
    dispatch(state, "fit_marginal", {"col": "income"})
    after = dispatch(state, "score_overall", {})

    assert before["conditional_pass_rate"] > after["conditional_pass_rate"], (
        "the conditional rate must register the destroyed structure"
    )
    assert after["composite_score"] < before["composite_score"], (
        f"composite must PENALIZE the swap: {before['composite_score']} -> "
        f"{after['composite_score']}. This is exactly the move that drove T3 to zero."
    )


def test_composite_still_reports_the_marginal_rate(state):
    """Backwards compatibility — the marginal view is still there, just no longer
    the only thing the agent can see."""
    _conditional(state)
    out = dispatch(state, "score_overall", {})
    assert out["marginal_pass_rate"] == out["pass_rate"]
    assert 0.0 <= out["composite_score"] <= 1.0
    assert out["conditional_rows"], "income has parents, so it must appear here"
