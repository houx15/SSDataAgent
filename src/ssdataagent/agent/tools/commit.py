"""Commit-family tools.

`commit_generator` is the "I'm done modeling, please sample N rows" signal.
The actual sampling happens in the orchestrator after the loop exits — the
tool just validates the chain and flips `state.committed = True`.

`report_progress` is a no-op narrative hook the agent can use to journal
its decisions; the orchestrator persists progress_log alongside transcript.json.
"""
from __future__ import annotations

from ssdataagent.agent.tools.state import RuntimeState


def commit_generator(state: RuntimeState) -> dict:
    """Validate the chain is ready and signal end-of-modeling. Auto-fills any
    column in generation_order that's still unfit with empirical from train_fit
    so the orchestrator's later sample() never raises."""
    chain = state.chain
    if not chain.generation_order:
        return {
            "error": "empty_chain",
            "details": "no generation_order yet; call set_generation_order + fit_marginal at least once",
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
