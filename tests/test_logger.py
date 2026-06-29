import json

import pandas as pd

from ssdataagent.agent.orchestrator import RunResult, TranscriptEntry
from ssdataagent.agent.sandbox import SandboxResult
from ssdataagent.experiments.logger import log_run


def _fake_run_result():
    return RunResult(
        generated=pd.DataFrame({"x": [1, 2, 3]}),
        transcript=[
            TranscriptEntry("user", "hi", "EXPLORATION"),
            TranscriptEntry("assistant", "ok", "EXPLORATION"),
            TranscriptEntry("tool", "exit=0", "EXPLORATION"),
        ],
        code_steps=["print('hi')"],
        sandbox_results=[
            SandboxResult(
                stdout="hi", stderr="", exit_code=0,
                duration_s=0.1, timed_out=False,
            )
        ],
    )


def test_log_run_writes_expected_files(tmp_path):
    run_dir = tmp_path / "experiments" / "exp1" / "full_agent" / "gss" / "20260429-120000"
    log_run(_fake_run_result(), run_dir=run_dir)
    assert (run_dir / "prompts.jsonl").exists()
    assert (run_dir / "responses.jsonl").exists()
    assert (run_dir / "code" / "step_001.py").exists()
    assert (run_dir / "code" / "step_001.stdout").exists()
    # meta.json and generated.csv are now written by the runner, not log_run:
    assert not (run_dir / "meta.json").exists()
    assert not (run_dir / "generated.csv").exists()


def test_log_run_handles_empty_results(tmp_path):
    run_dir = tmp_path / "empty"
    empty = RunResult(generated=pd.DataFrame(), transcript=[], code_steps=[], sandbox_results=[])
    log_run(empty, run_dir=run_dir)
    assert run_dir.exists()
