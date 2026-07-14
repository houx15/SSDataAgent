"""Tool registry — maps tool name (as it appears in OpenAI function calls)
to a callable `(state, **args) -> dict`.

The orchestrator dispatches each `tool_call` from the LLM through this
registry. Tools never raise; on bad input they return {"error": ...}.
"""
from __future__ import annotations

import traceback
from typing import Callable

from ssdataagent.agent.tools import commit as _commit
from ssdataagent.agent.tools import donor as _donor
from ssdataagent.agent.tools import fit as _fit
from ssdataagent.agent.tools import inspect as _inspect
from ssdataagent.agent.tools import verify as _verify
from ssdataagent.agent.tools.schemas import all_tool_schemas
from ssdataagent.agent.tools.state import RuntimeState, build_runtime_state


# Registry — populated below. Type signature is (state, **kwargs) -> dict.
TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    # Inspect family
    "list_columns": _inspect.list_columns,
    "describe_column": _inspect.describe_column,
    "cross_tab": _inspect.cross_tab,
    "missing_pattern": _inspect.missing_pattern,
    "correlation": _inspect.correlation,
    "groupby_stat": _inspect.groupby_stat,
    "head_rows": _inspect.head_rows,
    # Fit family
    "set_generation_order": _fit.set_generation_order,
    "fit_marginal": _fit.fit_marginal,
    "fit_conditional": _fit.fit_conditional,
    "fit_copy_real": _fit.fit_copy_real,
    "fit_block_donor": _donor.fit_block_donor,
    "replace_step": _fit.replace_step,
    # Verify family
    "sample_preview": _verify.sample_preview,
    "score_marginal": _verify.score_marginal,
    "score_conditional": _verify.score_conditional,
    "score_pair": _verify.score_pair,
    "score_event_order": _verify.score_event_order,
    "score_overall": _verify.score_overall,
    # Commit family
    "commit_generator": _commit.commit_generator,
    "report_progress": _commit.report_progress,
}


def dispatch(state: RuntimeState, name: str, arguments: dict) -> dict:
    """Look up `name` in the registry and call it with `arguments`. Catches
    every exception type and returns it as a tool-result dict so the LLM
    can self-correct without crashing the orchestrator loop. The contract
    is: tools never propagate exceptions to the caller — bad input or
    internal errors all become {"error": ..., "details": ...} dicts."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {
            "error": "unknown_tool",
            "details": f"no tool named {name!r}; available: {sorted(TOOL_REGISTRY)}",
        }
    try:
        return fn(state, **arguments)
    except TypeError as e:
        return {"error": "bad_arguments", "details": f"{e}"}
    except Exception as e:
        # Genuine internal error inside a tool — usually a sklearn / pandas
        # surprise from an upstream fit. Surface it as a tool result so the
        # agent can call replace_step / use a different family. Include a
        # truncated traceback so post-mortem from tool_calls.json doesn't
        # require re-running with a debugger (EXP-006f addhealth follow-up).
        tb_text = traceback.format_exc()
        # Keep the last ~12 lines — the frames closest to the actual raise
        # are the diagnostic ones, and the LLM doesn't benefit from the
        # full dispatch chain.
        tb_short = "\n".join(tb_text.splitlines()[-12:])
        return {
            "error": "tool_internal_error",
            "details": f"{type(e).__name__}: {e}",
            "traceback": tb_short,
        }


__all__ = [
    "TOOL_REGISTRY",
    "RuntimeState",
    "build_runtime_state",
    "dispatch",
    "all_tool_schemas",
]
