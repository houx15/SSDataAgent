from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# Held-out fraction used by the verify family. 20% means 200 rows on the
# default n=1000 train sample, big enough for KS / χ² to be meaningful but
# small enough that the agent still gets to fit on most of the data.
HELDOUT_FRACTION = 0.20
HELDOUT_RANDOM_SEED = 42


@dataclass
class RuntimeState:
    """Owned by the orchestrator across one run; passed to every tool.

    Tools never mutate `train` or `descriptions`. Only the in-progress
    `chain` and `progress_log` change across turns.
    """
    workspace: Path
    train: pd.DataFrame                  # full available training sample
    train_fit: pd.DataFrame              # 80% slice for fitting
    held_out: pd.DataFrame               # 20% slice for verify-family scoring
    descriptions: Optional[dict[str, Any]]  # None when condition strips semantic info
    has_data: bool                       # NO_DATA condition makes data tools refuse
    has_descriptions: bool               # NO_SEMANTIC condition strips descriptions
    chain: Any = None                    # tools.fit.Chain — populated lazily to avoid import cycle
    progress_log: list[str] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    committed: bool = False              # set by commit_generator; orchestrator reads it to exit the loop
    # Audit trail of every successful score_event_order call (events tuple).
    # commit_generator reads this to gate longitudinal commits — see EXP-006e.
    event_order_calls: list[tuple[str, ...]] = field(default_factory=list)


def split_train_heldout(
    df: pd.DataFrame,
    fraction: float = HELDOUT_FRACTION,
    seed: int = HELDOUT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic shuffle-and-split. The held-out slice is what the
    verify family scores against. Done once at the top of a run so every
    score_* call sees the same target distribution."""
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0,1), got {fraction}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_heldout = max(1, int(round(len(df) * fraction)))
    held = df.iloc[idx[:n_heldout]].reset_index(drop=True)
    fit = df.iloc[idx[n_heldout:]].reset_index(drop=True)
    return fit, held


def build_runtime_state(
    *,
    workspace: Path,
    train: pd.DataFrame,
    descriptions: Optional[dict[str, Any]],
    has_data: bool,
    has_descriptions: bool,
    seed: int = 0,
) -> RuntimeState:
    """Construct a RuntimeState. Imports Chain lazily to avoid a cycle."""
    from ssdataagent.agent.tools.fit import Chain
    train_fit, held_out = split_train_heldout(train)
    state = RuntimeState(
        workspace=workspace,
        train=train,
        train_fit=train_fit,
        held_out=held_out,
        descriptions=descriptions if has_descriptions else None,
        has_data=has_data,
        has_descriptions=has_descriptions,
        rng=np.random.default_rng(seed),
    )
    state.chain = Chain()
    return state


def data_withheld_error() -> dict[str, str]:
    """The standard refusal returned by data-reading tools when the
    NO_DATA condition is active. Same shape so the agent's error
    handling is uniform."""
    return {
        "error": "data_withheld",
        "details": (
            "This run is in the no_data condition. Inspect-family tools that "
            "read train.csv are disabled. Use descriptions / allowed_values "
            "from the system prompt to plan, then fit_marginal/fit_conditional "
            "directly."
        ),
    }
