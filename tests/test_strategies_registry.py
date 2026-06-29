import pytest

from ssdataagent.strategies.agent_strategy import AgentStrategy
from ssdataagent.strategies.direct_strategy import DirectGenerationStrategy
from ssdataagent.strategies.registry import get_strategy


def test_get_strategy_returns_agent():
    assert isinstance(get_strategy("agent"), AgentStrategy)


def test_get_strategy_returns_direct():
    assert isinstance(get_strategy("direct"), DirectGenerationStrategy)


def test_get_strategy_unknown_raises():
    with pytest.raises(KeyError):
        get_strategy("nope")
