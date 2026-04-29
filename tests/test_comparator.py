import pytest

from ssdataagent.evaluation.comparator import summary_pivot, to_long_table
from ssdataagent.evaluation.runner import PassRates, split_by_seen_unseen


def _rates(t1=0.7, t2=0.5):
    return PassRates(by_type={"type1": t1, "type2": t2})


def test_long_table_shape():
    inputs = {
        ("full_agent", "gss"): _rates(0.7, 0.5),
        ("direct_generation", "gss"): _rates(0.3, 0.2),
    }
    df = to_long_table(inputs)
    assert list(df.columns) == ["condition", "dataset", "type", "pass_rate"]
    assert len(df) == 4


def test_summary_pivot():
    inputs = {
        ("full_agent", "gss"): _rates(0.8, 0.6),
        ("direct_generation", "gss"): _rates(0.4, 0.2),
    }
    pivot = summary_pivot(to_long_table(inputs))
    assert pivot.loc["full_agent", "gss"] == pytest.approx(0.7)
    assert pivot.loc["direct_generation", "gss"] == pytest.approx(0.3)


def test_split_by_seen_unseen():
    rates = PassRates(by_variable={
        "type1": {"gender": 0.8, "income": 0.3},
        "type2": {"age": 0.7, "income": 0.2},
    })
    seen, unseen = split_by_seen_unseen(rates, unseen_vars=["income"])
    assert seen.by_variable["type1"]["gender"] == 0.8
    assert "income" not in seen.by_variable["type1"]
    assert unseen.by_variable["type1"]["income"] == 0.3
    assert unseen.by_variable["type2"]["income"] == 0.2
