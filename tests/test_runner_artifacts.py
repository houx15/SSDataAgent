import json
from unittest.mock import MagicMock, patch

import pandas as pd

from ssdataagent.agent.orchestrator import RunResult, TranscriptEntry
from ssdataagent.agent.sandbox import SandboxResult
from ssdataagent.evaluation.runner import PassRates
from ssdataagent.experiments.runner import ExperimentConfig, run_experiment


def _agent_run_result():
    return RunResult(
        generated=pd.DataFrame({"profile_id": [0, 1], "gender": ["Male", "Female"]}),
        transcript=[
            TranscriptEntry("user", "hi", "EXPLORATION"),
            TranscriptEntry("assistant", "ok", "EXPLORATION"),
        ],
        code_steps=["print('hi')"],
        sandbox_results=[
            SandboxResult(stdout="out", stderr="err", exit_code=0,
                          duration_s=0.1, timed_out=False)
        ],
    )


def _read(run_dir, name):
    return (run_dir / name).read_text()


def _only_run_dir(cond_dir):
    runs = [p for p in cond_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1, runs
    return runs[0]


@patch("ssdataagent.experiments.runner._git_sha", return_value="testsha")
@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.strategies.agent_strategy.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_agent_artifacts_are_stable(_cfg, _client, MockOrch, _eval, _sha, tmp_path):
    _cfg.return_value = MagicMock(model="m1", provider="p1")
    MockOrch.return_value.run.return_value = _agent_run_result()
    cfg = ExperimentConfig(
        name="charexp", datasets=["gss"], conditions=["full_agent"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    run_dir = _only_run_dir(tmp_path / "charexp" / "full_agent" / "gss")

    meta = json.loads(_read(run_dir, "meta.json"))
    assert meta == {
        "experiment": "charexp", "dataset": "gss", "condition": "full_agent",
        "run_id": run_dir.name, "git_sha": "testsha", "model": "m1",
        "provider": "p1", "unseen_variables": [],
    }
    assert _read(run_dir, "generated.csv") == "profile_id,gender\n0,Male\n1,Female\n"
    assert _read(run_dir, "prompts.jsonl") == \
        json.dumps({"stage": "EXPLORATION", "role": "user", "content": "hi"}) + "\n"
    assert _read(run_dir, "responses.jsonl") == \
        json.dumps({"stage": "EXPLORATION", "role": "assistant", "content": "ok"}) + "\n"
    assert _read(run_dir, "code/step_001.py") == "print('hi')"
    assert _read(run_dir, "code/step_001.stdout") == "out"
    assert _read(run_dir, "code/step_001.exit") == "0"


def _fake_direct(*, client, sampled, dataset_name, transcript_out=None):
    if transcript_out is not None:
        transcript_out.append({"row": 0, "prompt": "P", "response": "R"})
    return pd.DataFrame({"profile_id": [0], "gender": ["Male"]})


@patch("ssdataagent.strategies.direct_strategy.generate_direct", side_effect=_fake_direct)
@patch("ssdataagent.experiments.runner._git_sha", return_value="testsha")
@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.strategies.agent_strategy.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_direct_artifacts_are_stable(_cfg, _client, MockOrch, _eval, _sha, _direct, tmp_path):
    _cfg.return_value = MagicMock(model="m1", provider="p1")
    cfg = ExperimentConfig(
        name="charexp", datasets=["gss"], conditions=["direct_generation"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    run_dir = _only_run_dir(tmp_path / "charexp" / "direct_generation" / "gss")

    meta = json.loads(_read(run_dir, "meta.json"))
    assert meta["condition"] == "direct_generation"
    assert meta["git_sha"] == "testsha"
    assert "n_individuals" in meta
    assert _read(run_dir, "generated.csv") == "profile_id,gender\n0,Male\n"
    assert _read(run_dir, "prompts.jsonl") == \
        json.dumps({"row": 0, "role": "user", "content": "P"}) + "\n"
    assert _read(run_dir, "responses.jsonl") == \
        json.dumps({"row": 0, "role": "assistant", "content": "R"}) + "\n"


def _fake_overdet(*, real, sim, schema, **kw):
    return {"cell_based": {"headline_gap": 0.42, "coverage": 1.0, "n_cells": 1,
                           "per_target": {}}, "model_based": {"headline_gap": None,
                           "per_target": {}}}


@patch("ssdataagent.experiments.runner.overdetermination", side_effect=_fake_overdet)
@patch("ssdataagent.experiments.runner._git_sha", return_value="testsha")
@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.strategies.agent_strategy.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_eval_json_has_overdetermination(_cfg, _client, MockOrch, _eval, _sha, _od, tmp_path):
    _cfg.return_value = MagicMock(model="m1", provider="p1")
    MockOrch.return_value.run.return_value = _agent_run_result()
    cfg = ExperimentConfig(
        name="charexp", datasets=["gss"], conditions=["full_agent"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    run_dir = _only_run_dir(tmp_path / "charexp" / "full_agent" / "gss")
    blob = json.loads(_read(run_dir, "eval.json"))
    assert blob["overdetermination"]["cell_based"]["headline_gap"] == 0.42


def _fake_design_b_generate(self, gate, run_dir, cfg):
    # assert the runner built a transfer gate, then emit a trivial frame
    from ssdataagent.strategies.base import StrategyResult
    import pandas as pd
    assert gate.source is not None and len(gate.crosswalk) > 0
    out = pd.DataFrame({"profile_id": range(len(gate.background()))})
    return StrategyResult(generated=out, meta_extras={"backend": "design_b"})


@patch("ssdataagent.strategies.design_b.DesignBStrategy.generate", _fake_design_b_generate)
@patch("ssdataagent.experiments.runner._git_sha", return_value="testsha")
@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_transfer_condition_builds_source_gate(_cfg, _client, _eval, _sha, tmp_path):
    _cfg.return_value = MagicMock(model="m1", provider="p1")
    cfg = ExperimentConfig(
        name="dbexp", datasets=["gss"], conditions=["design_b_transfer"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    run_dir = _only_run_dir(tmp_path / "dbexp" / "design_b_transfer" / "gss")
    meta = json.loads(_read(run_dir, "meta.json"))
    assert meta["backend"] == "design_b"
    assert json.loads(_read(run_dir, "eval.json"))  # eval.json written
