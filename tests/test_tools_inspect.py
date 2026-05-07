"""Inspect-family tool tests. Hand-rolled DataFrames so each tool's
contract is exercised against known-truth values."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssdataagent.agent.tools.inspect import (
    correlation,
    cross_tab,
    describe_column,
    groupby_stat,
    head_rows,
    list_columns,
    missing_pattern,
)
from ssdataagent.agent.tools.state import RuntimeState


@pytest.fixture
def state(tmp_path: Path) -> RuntimeState:
    """A small but realistic survey-shaped frame: numeric, categorical,
    structural NA, and an unrelated column for cross-tab tests."""
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 25, 35, np.nan, 45, 55],
        "gender": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
        "child_number": [0, 2, 1, 0, 3, 0, 1, 2, 0, 3],
        # Structural NA: age_first_childbirth is NA exactly when child_number == 0
        "age_first_childbirth": [np.nan, 28, 33, np.nan, 35, np.nan, 30, 27, np.nan, 32],
        "all_nan": [np.nan] * 10,
    })
    return RuntimeState(
        workspace=tmp_path,
        train=df, train_fit=df, held_out=df.iloc[:0],
        descriptions=None, has_data=True, has_descriptions=False,
    )


def test_list_columns_reports_all_columns(state):
    out = list_columns(state)
    names = [r["name"] for r in out["columns"]]
    assert names == ["age", "gender", "child_number", "age_first_childbirth", "all_nan"]
    assert out["n_rows"] == 10
    age = next(r for r in out["columns"] if r["name"] == "age")
    assert age["n_missing"] == 1
    assert age["missing_rate"] == pytest.approx(0.1)


def test_describe_column_numeric(state):
    out = describe_column(state, "age")
    assert out["kind"] == "numeric"
    assert out["mean"] == pytest.approx((20+30+40+50+60+25+35+45+55)/9)
    assert out["min"] == 20
    assert out["max"] == 60
    assert out["n_missing"] == 1


def test_describe_column_categorical(state):
    out = describe_column(state, "gender")
    assert out["kind"] == "categorical"
    assert out["n_categories"] == 2
    counts = {r["value"]: r["count"] for r in out["value_counts"]}
    assert counts == {"F": 5, "M": 5}


def test_describe_column_all_nan(state):
    out = describe_column(state, "all_nan")
    # all_nan column is float64 with all NaN — currently treated as numeric.
    assert out.get("all_missing") is True
    assert out["n_missing"] == 10


def test_describe_column_unknown_returns_error(state):
    out = describe_column(state, "no_such_col")
    assert out["error"] == "unknown_column"
    assert "no_such_col" in out["details"]


def test_cross_tab_shape(state):
    out = cross_tab(state, "gender", "child_number")
    assert out["col1"] == "gender"
    assert out["col2"] == "child_number"
    assert set(out["rows"]) == {"F", "M"}
    # Every row x col cell should be a number.
    for row in out["values"]:
        for v in row:
            assert isinstance(v, (int, float))


def test_cross_tab_normalize(state):
    out = cross_tab(state, "gender", "child_number", normalize=True)
    total = sum(sum(row) for row in out["values"])
    assert total == pytest.approx(1.0, rel=1e-6)
    assert out["normalized"] is True


def test_missing_pattern_detects_structural_na(state):
    """child_number=0 ↔ age_first_childbirth NA. The pattern '10' (child_number
    present, age_first_childbirth NA) should appear with frequency = #zeros / n."""
    out = missing_pattern(state, ["child_number", "age_first_childbirth"])
    patterns = {p["pattern"]: p for p in out["patterns"]}
    assert "10" in patterns         # has child_number, missing age_first_childbirth
    assert "11" in patterns         # both present
    n_zero_kids = (state.train_fit["child_number"] == 0).sum()
    assert patterns["10"]["count"] == n_zero_kids


def test_missing_pattern_unknown_col(state):
    out = missing_pattern(state, ["age", "ghost"])
    assert out["error"] == "unknown_column"


def test_correlation_pearson(state):
    out = correlation(state, "age", "child_number", method="pearson")
    assert out["method"] == "pearson"
    assert out["n"] == 9                          # one age is NaN
    assert -1.0 <= out["coef"] <= 1.0


def test_correlation_non_numeric(state):
    out = correlation(state, "gender", "age")
    assert out["error"] == "non_numeric"


def test_correlation_unknown_method(state):
    out = correlation(state, "age", "child_number", method="kendall")
    assert out["error"] == "unknown_method"


def test_groupby_stat_mean(state):
    out = groupby_stat(state, "gender", "age", stat="mean")
    by_grp = {r["group_value"]: r["stat_value"] for r in out["groups"]}
    # Female ages: 20, 40, 60, 35, 45 (8th row was male NaN).
    assert by_grp["F"] == pytest.approx(np.mean([20, 40, 60, 35, 45]))


def test_groupby_stat_count_works_on_categorical_value(state):
    # count is the only stat that doesn't require numeric value_col.
    out = groupby_stat(state, "child_number", "gender", stat="count")
    assert out["stat"] == "count"
    counts = {r["group_value"]: r["stat_value"] for r in out["groups"]}
    assert sum(counts.values()) == len(state.train_fit)


def test_groupby_stat_non_numeric_value_for_mean(state):
    out = groupby_stat(state, "gender", "child_number", stat="mean")  # OK — child_number is numeric
    assert "groups" in out
    out2 = groupby_stat(state, "child_number", "gender", stat="mean")
    assert out2["error"] == "non_numeric_value"


def test_head_rows_basic(state):
    out = head_rows(state, n=3)
    assert out["n"] == 3
    assert len(out["rows"]) == 3
    assert "age" in out["columns"]
    # NaN should serialize to None.
    assert out["rows"][0]["age_first_childbirth"] is None


def test_head_rows_caps_n(state):
    assert head_rows(state, n=0)["error"] == "bad_n"
    assert head_rows(state, n=999)["error"] == "bad_n"


def test_data_withheld_when_has_data_false(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    state = RuntimeState(
        workspace=tmp_path,
        train=df, train_fit=df, held_out=df.iloc[:0],
        descriptions={"x": "irrelevant"},
        has_data=False,        # NO_DATA condition
        has_descriptions=True,
    )
    for fn in [list_columns, lambda s: describe_column(s, "x"),
               lambda s: head_rows(s, 2)]:
        out = fn(state)
        assert out.get("error") == "data_withheld"
