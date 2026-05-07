"""Tests for the tool-using orchestrator path. The LLM is mocked and
returns a scripted sequence of tool_calls — we verify the orchestrator
correctly dispatches each one, mutates RuntimeState, persists artefacts,
and produces a generated DataFrame."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ssdataagent.agent.context import Condition
from ssdataagent.agent.llm_client import ToolCallRequest, ToolUsingResponse
from ssdataagent.agent.orchestrator import Orchestrator, RunResult


def _make_response(content: str = "", *, calls: list[tuple[str, dict]] | None = None) -> ToolUsingResponse:
    """Build a ToolUsingResponse from a list of (tool_name, args) tuples."""
    tool_calls = []
    serialized = []
    for i, (name, args) in enumerate(calls or []):
        cid = f"call_{i:03d}"
        tool_calls.append(ToolCallRequest(id=cid, name=name, arguments=args))
        serialized.append({
            "id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    msg = {"role": "assistant", "content": content}
    if serialized:
        msg["tool_calls"] = serialized
    return ToolUsingResponse(content=content, tool_calls=tool_calls, assistant_message=msg)


def _stub_client(responses: list[ToolUsingResponse]):
    """A mock LLMClient whose chat_with_tools returns each response in turn."""
    client = MagicMock()
    client.chat_with_tools.side_effect = responses
    return client


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace prepared as `experiments/runner.py` would: train.csv +
    descriptions.json on disk."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "age": rng.integers(20, 70, size=n).astype(float),
        "gender": rng.choice(["F", "M"], size=n),
    })
    ws = tmp_path / "workspace"
    ws.mkdir()
    df.to_csv(ws / "train.csv", index=False)
    (ws / "descriptions.json").write_text(json.dumps({
        "context": "Synthetic test fixture",
        "background_variables": ["age", "gender"],
        "target_variables": [],
        "descriptions": {"age": "Age in years", "gender": "F or M"},
    }))
    return ws


def test_tool_using_loop_inspect_fit_commit(workspace: Path):
    """Happy path: agent calls list_columns, fits two marginals, commits.
    Orchestrator should produce generated.csv with the right shape and
    persist transcript.json + chain.json + tool_calls.json."""
    responses = [
        _make_response(calls=[("list_columns", {})]),
        _make_response(calls=[("fit_marginal", {"col": "age", "family": "kde"})]),
        _make_response(calls=[("fit_marginal", {"col": "gender", "family": "empirical"})]),
        _make_response(calls=[("commit_generator", {})]),
    ]
    client = _stub_client(responses)

    orch = Orchestrator(
        client=client, n_rows=50,
        prompt_variant="rubric_tools",
        max_turns=10,
    )
    result = orch.run(
        condition=Condition.FULL, dataset_name="test_ds",
        workspace=workspace, has_data=True, has_descriptions=True,
    )

    assert isinstance(result, RunResult)
    assert len(result.generated) == 50
    assert set(result.generated.columns) == {"age", "gender"}
    # Only 4 LLM calls because the agent committed cleanly.
    assert client.chat_with_tools.call_count == 4
    # Tool call log captures every dispatched call.
    tool_names = [e["tool"] for e in result.tool_call_log]
    assert tool_names == ["list_columns", "fit_marginal", "fit_marginal", "commit_generator"]
    # chain.json + transcript.json + tool_calls.json + generated.csv on disk.
    for fname in ("chain.json", "transcript.json", "tool_calls.json", "generated.csv"):
        assert (workspace / fname).exists(), f"missing {fname}"
    chain_meta = json.loads((workspace / "chain.json").read_text())
    assert chain_meta["generation_order"] == ["age", "gender"]


def test_tool_using_loop_handles_tool_errors_and_self_corrects(workspace: Path):
    """Agent calls describe_column on a non-existent col; tool returns an
    error; agent recovers by calling the right one. The orchestrator must
    keep going (not raise)."""
    responses = [
        _make_response(calls=[("describe_column", {"col": "nope"})]),  # bad col → error
        _make_response(calls=[("fit_marginal", {"col": "age"})]),
        _make_response(calls=[("commit_generator", {})]),
    ]
    client = _stub_client(responses)
    orch = Orchestrator(client=client, n_rows=20, prompt_variant="rubric_tools", max_turns=10)
    result = orch.run(
        condition=Condition.FULL, dataset_name="test_ds",
        workspace=workspace, has_data=True, has_descriptions=True,
    )
    assert len(result.generated) == 20
    # First tool call result was an error.
    first = result.tool_call_log[0]
    assert first["tool"] == "describe_column"
    assert "error" in first["result"]


def test_tool_using_loop_max_turns_force_commits(workspace: Path):
    """If the agent never commits, the orchestrator force-commits at
    max_turns. The chain auto-fills any unfit cols with empirical so
    generation still produces output."""
    # Agent only fits one of two cols, then keeps calling list_columns
    # forever. With max_turns=4, after one fit_marginal we run out of turns.
    responses = [
        _make_response(calls=[("set_generation_order", {"cols": ["age", "gender"]})]),
        _make_response(calls=[("fit_marginal", {"col": "age"})]),
        _make_response(calls=[("list_columns", {})]),
        _make_response(calls=[("list_columns", {})]),  # 4th turn — limit
    ]
    client = _stub_client(responses)
    orch = Orchestrator(client=client, n_rows=15, prompt_variant="rubric_tools", max_turns=4)
    result = orch.run(
        condition=Condition.FULL, dataset_name="test_ds",
        workspace=workspace, has_data=True, has_descriptions=True,
    )
    assert len(result.generated) == 15
    # Forced commit_generator entry shows up in the tool log.
    forced = [e for e in result.tool_call_log if e["tool"] == "commit_generator (forced)"]
    assert len(forced) == 1
    assert forced[0]["result"]["committed"] is True
    assert "gender" in forced[0]["result"]["auto_filled_with_empirical"]


def test_tool_using_loop_nudges_when_no_tool_call(workspace: Path):
    """If the agent emits text without a tool call, the orchestrator
    nudges it to keep going. Loop should resume normally."""
    responses = [
        _make_response(content="Hmm, let me think about this."),  # no tool_call → nudge
        _make_response(calls=[("fit_marginal", {"col": "age"})]),
        _make_response(calls=[("fit_marginal", {"col": "gender"})]),
        _make_response(calls=[("commit_generator", {})]),
    ]
    client = _stub_client(responses)
    orch = Orchestrator(client=client, n_rows=10, prompt_variant="rubric_tools", max_turns=10)
    result = orch.run(
        condition=Condition.FULL, dataset_name="test_ds",
        workspace=workspace, has_data=True, has_descriptions=True,
    )
    assert len(result.generated) == 10
    # The transcript should contain a "nudge" user message.
    nudge_entries = [e for e in result.transcript if e.role == "user" and "no tool call" in e.content]
    assert len(nudge_entries) == 1


def test_tool_using_loop_respects_no_data_condition(workspace: Path):
    """In NO_DATA condition, inspect/fit tools refuse with data_withheld.
    The orchestrator still runs the loop; the agent must commit using
    descriptions only."""
    # Start with no train.csv on disk to simulate has_data=False.
    (workspace / "train.csv").unlink()
    responses = [
        # The agent tries list_columns; gets data_withheld error.
        _make_response(calls=[("list_columns", {})]),
        # Tries fit_marginal anyway; also withheld.
        _make_response(calls=[("fit_marginal", {"col": "age"})]),
        # Gives up and commits — empty chain → error.
        _make_response(calls=[("commit_generator", {})]),
    ]
    client = _stub_client(responses)
    orch = Orchestrator(client=client, n_rows=5, prompt_variant="rubric_tools", max_turns=3)
    with pytest.raises(RuntimeError, match="forced commit failed"):
        # commit_generator returns an error (empty chain) → orchestrator
        # tries to force-commit at max_turns boundary, which also fails.
        # We expect this to surface, not silently succeed.
        result = orch.run(
            condition=Condition.NO_DATA, dataset_name="test_ds",
            workspace=workspace, has_data=False, has_descriptions=True,
        )
    # All three tool results should be data_withheld or empty_chain errors.
    # (we can't access result here; instead look at the workspace files.)
