# tests/test_strategy_agent.py
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.agent.orchestrator import RunResult, TranscriptEntry
from ssdataagent.agent.sandbox import SandboxResult
from ssdataagent.strategies.agent_strategy import AgentStrategy
from ssdataagent.strategies.base import InfoGate


def _run_result():
    return RunResult(
        generated=pd.DataFrame({"profile_id": [0], "gender": ["Male"]}),
        transcript=[TranscriptEntry("user", "hi", "EXPLORATION")],
        code_steps=["print(1)"],
        sandbox_results=[SandboxResult(stdout="o", stderr="", exit_code=0,
                                       duration_s=0.0, timed_out=False)],
    )


def _cfg():
    return SimpleNamespace(n_rows=10, max_iterations=1, sandbox_timeout=5,
                           prompt_variant="baseline")


def _gate(workspace):
    return InfoGate(
        condition=Condition.FULL, dataset_name="gss", workspace=workspace,
        client=object(), train=pd.DataFrame({"profile_id": [1], "age": [40]}),
        eval_rows=pd.DataFrame({"profile_id": [0], "age": [30]}),
    )


@patch("ssdataagent.strategies.agent_strategy.log_run")
@patch("ssdataagent.strategies.agent_strategy.Orchestrator")
@patch("ssdataagent.strategies.agent_strategy.build_context")
def test_agent_strategy_runs_orchestrator_and_logs(MockBuild, MockOrch, mock_log, tmp_path):
    MockBuild.return_value = MagicMock(has_data=True, has_descriptions=True)
    MockOrch.return_value.run.return_value = _run_result()
    result = AgentStrategy().generate(_gate(tmp_path), tmp_path, _cfg())

    assert result.generated.equals(pd.DataFrame({"profile_id": [0], "gender": ["Male"]}))
    assert result.meta_extras == {"unseen_variables": []}
    MockOrch.return_value.run.assert_called_once()
    mock_log.assert_called_once()
    # log_run is called with the orchestrator result and the run_dir, no meta:
    _, kwargs = mock_log.call_args
    assert kwargs["run_dir"] == tmp_path
    # FULL is not UNSEEN, so build_context must be told unseen_variables=None:
    _, build_kwargs = MockBuild.call_args
    assert build_kwargs["unseen_variables"] is None


@patch("ssdataagent.strategies.agent_strategy.log_run")
@patch("ssdataagent.strategies.agent_strategy.Orchestrator")
@patch("ssdataagent.strategies.agent_strategy.build_context")
def test_agent_strategy_passes_unseen_only_for_unseen_condition(MockBuild, MockOrch, _l, tmp_path):
    MockBuild.return_value = MagicMock(has_data=True, has_descriptions=True)
    MockOrch.return_value.run.return_value = _run_result()
    gate = InfoGate(
        condition=Condition.UNSEEN, dataset_name="gss", workspace=tmp_path,
        client=object(), train=pd.DataFrame({"profile_id": [1]}),
        eval_rows=pd.DataFrame({"profile_id": [0]}), unseen_variables=("income",),
    )
    result = AgentStrategy().generate(gate, tmp_path, _cfg())
    assert result.meta_extras == {"unseen_variables": ["income"]}
    _, build_kwargs = MockBuild.call_args
    assert build_kwargs["unseen_variables"] == ["income"]
