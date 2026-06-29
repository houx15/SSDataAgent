import pytest

from ssdataagent.agent.context import Condition
from ssdataagent.experiments.conditions import (
    CONDITIONS,
    ConditionSpec,
    get_condition,
)


def test_all_four_main_conditions_registered():
    expected = {"full_agent", "agent_no_semantic", "agent_no_data", "direct_generation"}
    assert expected.issubset(set(CONDITIONS))


def test_full_agent_unseen_registered():
    assert "full_agent_unseen" in CONDITIONS


def test_condition_to_context_mapping():
    spec = get_condition("full_agent")
    assert isinstance(spec, ConditionSpec)
    assert spec.context_condition is Condition.FULL
    assert spec.strategy == "agent"


def test_direct_generation_is_not_agent():
    assert get_condition("direct_generation").strategy == "direct"


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        get_condition("nope")


def test_baseline_conditions_registered():
    for name in ("hotdeck", "cart", "copula"):
        spec = get_condition(name)
        assert spec.context_condition is Condition.FULL
        assert spec.strategy == name


def test_design_b_conditions_registered():
    assert get_condition("design_b_full").context_condition is Condition.FULL
    assert get_condition("design_b_aggregate").context_condition is Condition.NO_DATA
    assert get_condition("design_b_transfer").context_condition is Condition.TRANSFER
    for n in ("design_b_full", "design_b_aggregate", "design_b_transfer"):
        assert get_condition(n).strategy == "design_b"
