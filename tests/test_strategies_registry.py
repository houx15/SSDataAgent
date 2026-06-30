import pytest

from ssdataagent.strategies.agent_strategy import AgentStrategy
from ssdataagent.strategies.baselines import (
    CartStrategy,
    CopulaStrategy,
    HotDeckStrategy,
)
from ssdataagent.strategies.direct_strategy import DirectGenerationStrategy
from ssdataagent.strategies.registry import get_strategy


def test_get_strategy_returns_agent():
    assert isinstance(get_strategy("agent"), AgentStrategy)


def test_get_strategy_returns_direct():
    assert isinstance(get_strategy("direct"), DirectGenerationStrategy)


def test_get_strategy_unknown_raises():
    with pytest.raises(KeyError):
        get_strategy("nope")


def test_get_strategy_returns_baselines():
    assert isinstance(get_strategy("hotdeck"), HotDeckStrategy)
    assert isinstance(get_strategy("cart"), CartStrategy)
    assert isinstance(get_strategy("copula"), CopulaStrategy)


def test_get_strategy_returns_design_b():
    from ssdataagent.strategies.design_b import DesignBStrategy
    assert isinstance(get_strategy("design_b"), DesignBStrategy)


def test_design_c_registered():
    from ssdataagent.strategies.registry import get_strategy
    s = get_strategy("design_c")
    assert s.name == "design_c"
