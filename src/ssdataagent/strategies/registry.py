from __future__ import annotations

from ssdataagent.strategies.agent_strategy import AgentStrategy
from ssdataagent.strategies.base import Strategy
from ssdataagent.strategies.baselines import (
    CartStrategy,
    CopulaStrategy,
    HotDeckStrategy,
)
from ssdataagent.strategies.design_b import DesignBStrategy
from ssdataagent.strategies.design_a import DesignAStrategy
from ssdataagent.strategies.design_c import DesignCStrategy
from ssdataagent.strategies.direct_strategy import DirectGenerationStrategy

STRATEGIES: dict[str, type] = {
    "agent": AgentStrategy,
    "direct": DirectGenerationStrategy,
    "hotdeck": HotDeckStrategy,
    "cart": CartStrategy,
    "copula": CopulaStrategy,
    "design_b": DesignBStrategy,
    "design_c": DesignCStrategy,
    "design_a": DesignAStrategy,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; known: {list(STRATEGIES)}")
    return STRATEGIES[name]()
