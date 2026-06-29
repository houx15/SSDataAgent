from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from ssdataagent.data.schema import load_schema


class Condition(str, Enum):
    FULL = "full_agent"
    NO_SEMANTIC = "agent_no_semantic"
    NO_DATA = "agent_no_data"
    UNSEEN = "full_agent_unseen"
    DIRECT = "direct_generation"
    TRANSFER = "transfer"


@dataclass(frozen=True)
class AgentContext:
    condition: Condition
    dataset_name: str
    workspace: Path
    has_data: bool
    has_descriptions: bool
    unseen_variables: tuple[str, ...] = ()


def build_context(
    *,
    condition: Condition,
    dataset_name: str,
    train_df: pd.DataFrame,
    workspace: Path,
    unseen_variables: list[str] | None = None,
) -> AgentContext:
    schema = load_schema(dataset_name)
    workspace.mkdir(parents=True, exist_ok=True)
    unseen = tuple(unseen_variables or ())

    has_data = condition in (Condition.FULL, Condition.NO_SEMANTIC, Condition.UNSEEN)
    has_descriptions = condition in (Condition.FULL, Condition.NO_DATA, Condition.UNSEEN)

    if has_data:
        # Filter to schema-relevant columns only. Longitudinal datasets ship
        # per-age sequential columns (income_14..65, education_14..65, etc.)
        # that the schema's static-only loader excludes; copying them through
        # to the agent makes the prompt unwieldy and tempts it to fit
        # 500-column joint distributions that have nothing to do with eval.
        keep = set(schema.background_variables) | set(schema.target_variables)
        keep.add("profile_id")
        cols = [c for c in train_df.columns if c in keep]
        df = train_df[cols].copy()
        if condition is Condition.UNSEEN:
            df = df.drop(columns=[c for c in unseen if c in df.columns])
        df.to_csv(workspace / "train.csv", index=False)

    if has_descriptions:
        payload = {
            "context": schema.population_context,
            "descriptions": schema.descriptions,
            "allowed_values": schema.allowed_values,
            "numeric_ranges": {k: list(v) for k, v in schema.numeric_ranges.items()},
            "background_variables": schema.background_variables,
            "target_variables": schema.target_variables,
        }
        (workspace / "descriptions.json").write_text(json.dumps(payload, indent=2))

    return AgentContext(
        condition=condition,
        dataset_name=dataset_name,
        workspace=workspace,
        has_data=has_data,
        has_descriptions=has_descriptions,
        unseen_variables=unseen,
    )
