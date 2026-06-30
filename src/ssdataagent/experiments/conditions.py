from __future__ import annotations

from dataclasses import dataclass

from ssdataagent.agent.context import Condition


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    context_condition: Condition
    strategy: str


CONDITIONS: dict[str, ConditionSpec] = {
    "full_agent": ConditionSpec("full_agent", Condition.FULL, strategy="agent"),
    "agent_no_semantic": ConditionSpec(
        "agent_no_semantic", Condition.NO_SEMANTIC, strategy="agent"
    ),
    "agent_no_data": ConditionSpec("agent_no_data", Condition.NO_DATA, strategy="agent"),
    "full_agent_unseen": ConditionSpec(
        "full_agent_unseen", Condition.UNSEEN, strategy="agent"
    ),
    "direct_generation": ConditionSpec(
        "direct_generation", Condition.DIRECT, strategy="direct"
    ),
    "hotdeck": ConditionSpec("hotdeck", Condition.FULL, strategy="hotdeck"),
    "cart": ConditionSpec("cart", Condition.FULL, strategy="cart"),
    "copula": ConditionSpec("copula", Condition.FULL, strategy="copula"),
    "design_b_full": ConditionSpec("design_b_full", Condition.FULL, strategy="design_b"),
    "design_b_aggregate": ConditionSpec("design_b_aggregate", Condition.NO_DATA, strategy="design_b"),
    "design_b_transfer": ConditionSpec("design_b_transfer", Condition.TRANSFER, strategy="design_b"),
    "design_c_full": ConditionSpec("design_c_full", Condition.FULL, strategy="design_c"),
    "design_c_aggregate": ConditionSpec("design_c_aggregate", Condition.NO_DATA, strategy="design_c"),
    "design_c_transfer": ConditionSpec("design_c_transfer", Condition.TRANSFER, strategy="design_c"),
    "design_a_full": ConditionSpec("design_a_full", Condition.FULL, strategy="design_a"),
    "design_a_aggregate": ConditionSpec("design_a_aggregate", Condition.NO_DATA, strategy="design_a"),
    "design_a_transfer": ConditionSpec("design_a_transfer", Condition.TRANSFER, strategy="design_a"),
}


def get_condition(name: str) -> ConditionSpec:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; known: {list(CONDITIONS)}")
    return CONDITIONS[name]
