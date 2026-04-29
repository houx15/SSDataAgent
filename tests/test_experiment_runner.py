from unittest.mock import MagicMock, patch

import pandas as pd

from ssdataagent.evaluation.runner import PassRates
from ssdataagent.experiments.runner import ExperimentConfig, run_experiment


def _fake_run_result():
    rr = MagicMock()
    rr.generated = pd.DataFrame({"profile_id": [0, 1], "gender": ["Male", "Female"]})
    rr.transcript = []
    rr.code_steps = []
    rr.sandbox_results = []
    return rr


@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.experiments.runner.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_run_experiment_executes_each_pair(
    _cfg, _client, MockOrch, _eval, tmp_path
):
    MockOrch.return_value.run.return_value = _fake_run_result()
    cfg = ExperimentConfig(
        name="t1",
        datasets=["gss"],
        conditions=["full_agent", "agent_no_semantic"],
        max_iterations=1,
        sandbox_timeout=10,
        train_eval_split=0.5,
        n_rows=10,
        results_root=tmp_path,
    )
    results = run_experiment(cfg)
    assert len(results) == 2
    assert ("full_agent", "gss") in results
    assert ("agent_no_semantic", "gss") in results


@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.experiments.runner.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_run_experiment_resume_skips_done(
    _cfg, _client, MockOrch, _eval, tmp_path
):
    MockOrch.return_value.run.return_value = _fake_run_result()
    cfg = ExperimentConfig(
        name="t2",
        datasets=["gss"],
        conditions=["full_agent"],
        max_iterations=1,
        sandbox_timeout=10,
        train_eval_split=0.5,
        n_rows=10,
        results_root=tmp_path,
    )
    run_experiment(cfg)
    first_count = MockOrch.return_value.run.call_count
    run_experiment(cfg, resume=True)
    assert MockOrch.return_value.run.call_count == first_count


@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.experiments.runner.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_direct_generation_skipped_in_phase5(
    _cfg, _client, MockOrch, _eval, tmp_path
):
    """Phase 5: direct_generation produces an empty PassRates and does not
    call the orchestrator. Phase 7 wires in the real direct generator."""
    MockOrch.return_value.run.return_value = _fake_run_result()
    cfg = ExperimentConfig(
        name="t3",
        datasets=["gss"],
        conditions=["direct_generation"],
        max_iterations=1,
        sandbox_timeout=10,
        train_eval_split=0.5,
        n_rows=10,
        results_root=tmp_path,
    )
    results = run_experiment(cfg)
    assert ("direct_generation", "gss") in results
    assert MockOrch.return_value.run.call_count == 0
