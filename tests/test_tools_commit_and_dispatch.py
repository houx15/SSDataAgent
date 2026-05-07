"""Commit-family + end-to-end dispatch through the registry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssdataagent.agent.tools import all_tool_schemas, dispatch
from ssdataagent.agent.tools.fit import Chain
from ssdataagent.agent.tools.state import RuntimeState


@pytest.fixture
def state(tmp_path: Path) -> RuntimeState:
    df = pd.DataFrame({
        "age": np.arange(20.0, 80.0).repeat(2),
        "gender": ["F", "M"] * 60,
    })
    perm = np.random.default_rng(0).permutation(len(df))
    train = df.iloc[perm[:96]].reset_index(drop=True)
    held = df.iloc[perm[96:]].reset_index(drop=True)
    return RuntimeState(
        workspace=tmp_path,
        train=train, train_fit=train, held_out=held,
        descriptions=None, has_data=True, has_descriptions=False,
        chain=Chain(), rng=np.random.default_rng(0),
    )


def test_all_tool_schemas_collects_every_family():
    """Schema registry has at least one tool from each family."""
    schemas = all_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    # One sentinel per family — confirms each family wired itself in.
    assert "list_columns" in names         # inspect
    assert "fit_marginal" in names         # fit
    assert "score_marginal" in names       # verify
    assert "commit_generator" in names     # commit
    assert "report_progress" in names


def test_dispatch_unknown_tool_returns_error(state):
    out = dispatch(state, "nope", {})
    assert out["error"] == "unknown_tool"
    assert "available" in out["details"]


def test_dispatch_bad_arguments_returns_error(state):
    out = dispatch(state, "describe_column", {"col": "age", "extra": "kw"})
    assert out["error"] == "bad_arguments"


def test_full_dispatch_workflow(state):
    """Drive the agent's whole loop through dispatch — inspect → fit → verify
    → commit. End-to-end smoke test of the registry."""
    # 1. Inspect
    out = dispatch(state, "list_columns", {})
    assert out["n_rows"] == 96

    # 2. Fit
    out = dispatch(state, "set_generation_order", {"cols": ["age", "gender"]})
    assert out["set"] is True

    out = dispatch(state, "fit_marginal", {"col": "age", "family": "kde"})
    assert out["registered"] is True

    out = dispatch(state, "fit_marginal", {"col": "gender", "family": "empirical"})
    assert out["registered"] is True

    # 3. Verify
    out = dispatch(state, "score_overall", {})
    assert out["n_columns"] == 2

    # 4. Commit
    assert state.committed is False
    out = dispatch(state, "commit_generator", {})
    assert out["committed"] is True
    assert state.committed is True
    assert out["n_steps"] == 2


def test_commit_auto_fills_unfit_cols(state):
    """If generation_order has cols that aren't fit, commit_generator stubs
    them with empirical so the orchestrator's sample() doesn't crash."""
    dispatch(state, "set_generation_order", {"cols": ["age", "gender"]})
    dispatch(state, "fit_marginal", {"col": "age"})
    out = dispatch(state, "commit_generator", {})
    assert out["committed"] is True
    assert "gender" in out["auto_filled_with_empirical"]


def test_commit_empty_chain_rejected(state):
    out = dispatch(state, "commit_generator", {})
    assert out["error"] == "empty_chain"
    assert state.committed is False


def test_report_progress_appends_to_log(state):
    out = dispatch(state, "report_progress", {"message": "fitting age as kde"})
    assert out["logged"] is True
    assert state.progress_log == ["fitting age as kde"]
