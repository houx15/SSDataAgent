from __future__ import annotations

from dataclasses import dataclass

from ssdataagent.agent.context import Condition


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    context_condition: Condition
    is_agent: bool


CONDITIONS: dict[str, ConditionSpec] = {
    "full_agent": ConditionSpec("full_agent", Condition.FULL, is_agent=True),
    "agent_no_semantic": ConditionSpec(
        "agent_no_semantic", Condition.NO_SEMANTIC, is_agent=True
    ),
    "agent_no_data": ConditionSpec("agent_no_data", Condition.NO_DATA, is_agent=True),
    "full_agent_unseen": ConditionSpec(
        "full_agent_unseen", Condition.UNSEEN, is_agent=True
    ),
    "direct_generation": ConditionSpec(
        "direct_generation", Condition.DIRECT, is_agent=False
    ),
}


def get_condition(name: str) -> ConditionSpec:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; known: {list(CONDITIONS)}")
    return CONDITIONS[name]
