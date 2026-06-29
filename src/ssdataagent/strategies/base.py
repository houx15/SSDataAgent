from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

from ssdataagent.agent.context import Condition

if TYPE_CHECKING:
    from ssdataagent.agent.llm_client import LLMClient
    from ssdataagent.experiments.runner import ExperimentConfig


@dataclass
class StrategyResult:
    generated: pd.DataFrame
    meta_extras: dict = field(default_factory=dict)


@dataclass
class InfoGate:
    condition: Condition
    dataset_name: str
    workspace: Path
    client: "LLMClient"
    train: pd.DataFrame
    eval_rows: pd.DataFrame
    unseen_variables: tuple[str, ...] = ()

    def background(self) -> pd.DataFrame:
        """Test/eval rows — always allowed."""
        return self.eval_rows

    def fit_microdata(self) -> pd.DataFrame | None:
        """Train split when the condition permits microdata; None otherwise.

        Mirrors agent.context.build_context's has_data gating exactly:
        FULL / NO_SEMANTIC / UNSEEN expose data; NO_DATA / DIRECT do not.
        """
        if self.condition in (Condition.FULL, Condition.NO_SEMANTIC, Condition.UNSEEN):
            return self.train
        return None


@runtime_checkable
class Strategy(Protocol):
    name: str

    def generate(self, gate: InfoGate, run_dir: Path, cfg: "ExperimentConfig") -> StrategyResult:
        """Fill all target vars for each background row, writing the
        strategy's own method-specific artifacts into run_dir. Returns the
        generated frame plus strategy-specific meta.json fields."""
        ...
