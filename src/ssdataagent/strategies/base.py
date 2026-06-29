from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.aggregates import associations, marginals
from ssdataagent.data.schema import load_schema

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
    source: pd.DataFrame | None = None
    source_name: str | None = None
    crosswalk: tuple[str, ...] = ()

    def background(self) -> pd.DataFrame:
        """Test/eval rows — always allowed."""
        return self.eval_rows

    def fit_microdata(self) -> pd.DataFrame | None:
        """Microdata a strategy may fit on. Source (crosswalk cols) under
        TRANSFER; train under FULL/NO_SEMANTIC/UNSEEN; None under NO_DATA/DIRECT."""
        if self.condition is Condition.TRANSFER:
            return None if self.source is None else self.source[list(self.crosswalk)]
        if self.condition in (Condition.FULL, Condition.NO_SEMANTIC, Condition.UNSEEN):
            return self.train
        return None

    def _reference_microdata(self) -> pd.DataFrame | None:
        """Frame the aggregates are computed from: source for TRANSFER, train for
        FULL/NO_SEMANTIC/UNSEEN/NO_DATA, None for DIRECT."""
        if self.condition is Condition.DIRECT:
            return None
        if self.condition is Condition.TRANSFER:
            return None if self.source is None else self.source[list(self.crosswalk)]
        return self.train

    def known_marginals(self) -> dict | None:
        ref = self._reference_microdata()
        if ref is None:
            return None
        schema = load_schema(self.dataset_name)
        targets = [t for t in schema.target_variables if t in ref.columns]
        return marginals(ref, targets, schema)

    def known_associations(self) -> dict | None:
        ref = self._reference_microdata()
        if ref is None:
            return None
        schema = load_schema(self.dataset_name)
        targets = [t for t in schema.target_variables if t in ref.columns]
        return associations(ref, targets, schema)


@runtime_checkable
class Strategy(Protocol):
    name: str

    def generate(self, gate: InfoGate, run_dir: Path, cfg: "ExperimentConfig") -> StrategyResult:
        """Fill all target vars for each background row, writing the
        strategy's own method-specific artifacts into run_dir. Returns the
        generated frame plus strategy-specific meta.json fields."""
        ...
