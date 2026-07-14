"""Commit-family tools.

`commit_generator` is the "I'm done modeling, please sample N rows" signal.
The actual sampling happens in the orchestrator after the loop exits — the
tool just validates the chain and flips `state.committed = True`.

Gates here are ADVISORY: they warn and record, they never refuse. The chronology
check used to *block* commit until `score_event_order` had run — but that tool
crashed on datasets whose event-age columns carry string sentinels ('never
married'), which made the gate unsatisfiable. On cfps the agent then spent half
its turn budget on six failed commits, progressively destroying the very event
columns the gate was meant to protect, and still shipped a T4 score of 0.000. A
gate that can become unsatisfiable is a trap, and the agent cannot tell a broken
tool from a broken model. So the forcing function now lives in the *score*, where
a bad choice is visible, rather than in a refusal, where it is punitive.

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
    # The orchestrator may set state.t4_unverified=True at max_turns to bypass
    # this gate (EXP-006f) — that path keeps the run alive at the cost of a
    # T4 penalty, instead of crashing with the gate as the only blocker.
    event_cols = _event_age_columns(chain.generation_order)
    chronology_unverified = False
    if len(event_cols) >= 2:
        event_set = set(event_cols)
        chronology_unverified = not any(
            len(event_set & set(call)) >= 2 for call in state.event_order_calls
        )

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

    # Final consistency guard (EXP-006f addhealth follow-up): even after
    # auto-fill, a conditional Step may still reference a `given` column
    # that's no longer in generation_order — typically because the agent
    # called set_generation_order again with a list that dropped some cols.
    # Sample() would then raise KeyError. Catch it here instead.
    pos = {c: i for i, c in enumerate(chain.generation_order)}
    for col in chain.generation_order:
        step = chain.steps.get(col)
        given = getattr(step, "given", None) if step is not None else None
        if not given:
            continue
        missing_or_late = [g for g in given if g not in pos or pos[g] >= pos[col]]
        if missing_or_late:
            return {
                "error": "inconsistent_chain",
                "details": (
                    f"step for {col!r} has given={given} but {missing_or_late} "
                    f"are not in generation_order before {col!r}; reorder or "
                    "call replace_step on this column before committing"
                ),
            }

    state.committed = True
    result = {
        "committed": True,
        "n_steps": len(chain.steps),
        "generation_order": list(chain.generation_order),
        "auto_filled_with_empirical": auto_filled,
    }
    if chronology_unverified:
        state.t4_unverified = True
        result["t4_unverified"] = True
        result["warning"] = (
            "committed without score_event_order coverage of the event-age columns "
            f"({sorted(event_cols)}); T4 (event ordering) is unverified and may score 0"
        )
    return result


def report_progress(state: RuntimeState, message: str) -> dict:
    """Append a free-form note to state.progress_log. No effect on chain."""
    if not isinstance(message, str):
        return {"error": "bad_arguments", "details": "message must be a string"}
    state.progress_log.append(message)
    return {"logged": True, "n_entries": len(state.progress_log)}
