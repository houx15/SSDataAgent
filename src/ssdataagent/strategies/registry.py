from __future__ import annotations

from ssdataagent.strategies.agent_strategy import AgentStrategy
from ssdataagent.strategies.base import Strategy
from ssdataagent.strategies.baselines import (
    CartStrategy,
    CopulaStrategy,
    HotDeckStrategy,
)
from ssdataagent.strategies.block_donor import BlockDonorStrategy
from ssdataagent.strategies.design_b import DesignBStrategy
from ssdataagent.strategies.design_a import DesignAStrategy
from ssdataagent.strategies.design_c import DesignCStrategy
from ssdataagent.strategies.direct_strategy import DirectGenerationStrategy
from ssdataagent.strategies.s1 import S1PersonasStrategy, S1RakedStrategy, S1RawStrategy

STRATEGIES: dict[str, type] = {
    "agent": AgentStrategy,
    "direct": DirectGenerationStrategy,
    "hotdeck": HotDeckStrategy,
    "block_donor": BlockDonorStrategy,
    "cart": CartStrategy,
    "copula": CopulaStrategy,
    "design_b": DesignBStrategy,
    "design_c": DesignCStrategy,
    "design_a": DesignAStrategy,
    "s1_raw": S1RawStrategy,
    "s1_raked": S1RakedStrategy,
    "s1_personas": S1PersonasStrategy,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; known: {list(STRATEGIES)}")
    return STRATEGIES[name]()
