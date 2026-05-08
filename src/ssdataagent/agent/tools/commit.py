"""Commit-family tools.

`commit_generator` is the "I'm done modeling, please sample N rows" signal.
The actual sampling happens in the orchestrator after the loop exits — the
tool just validates the chain and flips `state.committed = True`.

When the chain has multiple event-age columns (longitudinal datasets), the
commit is gated on `score_event_order` having been called — see EXP-006e
retro for why this gate exists.

`report_progress` is a no-op narrative hook the agent can use to journal
its decisions; the orchestrator persists progress_log alongside transcript.json.
"""
from __future__ import annotations

import re
from typing import Iterable

from ssdataagent.agent.tools.state import RuntimeState


# Column-name patterns SSDataBench uses for event-age fields.
# Matches: age_at_first_marriage, age_at_first_child, age_at_first_divorce,
# age_finished_education, age_started_work, age_at_birth_of_first_child, ...
# Does NOT match: age (current age), birth_year, age_30_40 (age window).
_EVENT_AGE_PATTERN = re.compile(r"^age_(at_|finished_|started_|began_|completed_)")


def _event_age_columns(cols: Iterable[str]) -> list[str]:
    """Return the subset of `cols` whose name signals an event-age field.

    The heuristic is intentionally conservative — false negatives push the
    agent off the gate (we miss a chronology check), but false positives
    would block commits on cross-sectional chains that just happen to have
    `age`-prefixed columns. Better to under-trigger than over-trigger.
    """
    return [c for c in cols if _EVENT_AGE_PATTERN.match(c)]


def commit_generator(state: RuntimeState) -> dict:
    """Validate the chain is ready and signal end-of-modeling. Auto-fills any
    column in generation_order that's still unfit with empirical from train_fit
    so the orchestrator's later sample() never raises.

    For longitudinal chains (≥2 event-age columns), refuses unless
    `score_event_order` has already covered ≥2 of those columns.
    """
    chain = state.chain
    if not chain.generation_order:
        return {
            "error": "empty_chain",
            "details": "no generation_order yet; call set_generation_order + fit_marginal at least once",
        }

    # Chronology gate (EXP-006e). Skip on chains with <2 event-age cols —
    # nothing to check for ordering.
    event_cols = _event_age_columns(chain.generation_order)
    if len(event_cols) >= 2:
        event_set = set(event_cols)
        scored_pair = any(
            len(event_set & set(call)) >= 2
            for call in state.event_order_calls
        )
        if not scored_pair:
            return {
                "error": "missing_event_order_check",
                "event_age_columns": sorted(event_cols),
                "details": (
                    f"This chain has {len(event_cols)} event-age columns "
                    f"({sorted(event_cols)}); commit requires score_event_order "
                    "to have been called with at least two of them in chronological "
                    "order so T4 (event ordering) is verified, not assumed. "
                    "Call score_event_order(events=[...]) and then retry commit_generator."
                ),
            }

    # Auto-fill any unfit cols with empirical so sample() doesn't raise.
    from ssdataagent.agent.tools.fit import MarginalStep
    auto_filled = []
    for col in chain.generation_order:
        if not chain.has(col):
            if col not in state.train_fit.columns:
                return {
                    "error": "unknown_column",
                    "details": f"generation_order references unknown column {col!r}",
                }
            nn = state.train_fit[col].dropna()
            if len(nn) == 0:
                return {
                    "error": "all_missing",
                    "details": f"column {col!r} is entirely NaN; can't auto-fill at commit",
                }
            stub = MarginalStep(col=col, family="empirical")
            stub.fit_values = nn.to_numpy()
            chain.add(stub)
            auto_filled.append(col)

    state.committed = True
    return {
        "committed": True,
        "n_steps": len(chain.steps),
        "generation_order": list(chain.generation_order),
        "auto_filled_with_empirical": auto_filled,
    }


def report_progress(state: RuntimeState, message: str) -> dict:
    """Append a free-form note to state.progress_log. No effect on chain."""
    if not isinstance(message, str):
        return {"error": "bad_arguments", "details": "message must be a string"}
    state.progress_log.append(message)
    return {"logged": True, "n_entries": len(state.progress_log)}
