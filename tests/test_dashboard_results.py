from pathlib import Path

import pytest

from ssdataagent.dashboard.results import ExperimentResults, load_results

FIXTURES = Path(__file__).parent / "fixtures" / "dashboard"


def test_load_results_reads_summary_and_done_flag():
    r = load_results(FIXTURES / "results" / "exp_demo_a")
    assert r.exp_name == "exp_demo_a"
    assert r.prompt_variant == "rubric_tools_v1"
    assert r.llm_model == "gpt-5.4-2026-03-05"
    assert r.llm_provider == "openai"
    assert r.finished_at == "2026-05-10T11:00:00"


def test_load_results_scores_indexed_by_condition_dataset_type():
    r = load_results(FIXTURES / "results" / "exp_demo_a")
    assert r.scores[("full_agent", "demo", "type1")] == pytest.approx(0.5)
    assert r.scores[("full_agent", "demo", "type2")] == pytest.approx(0.6)
    assert r.scores[("full_agent", "demo", "type3")] == pytest.approx(0.16)


def test_load_results_handles_missing_t4_t5_for_cross_sectional():
    r = load_results(FIXTURES / "results" / "exp_demo_a")
    assert ("full_agent", "demo", "type4") not in r.scores
    assert ("full_agent", "demo", "type5") not in r.scores


def test_load_results_handles_multi_condition():
    r = load_results(FIXTURES / "results" / "exp_demo_b_cross")
    assert ("full_agent", "demo_x", "type1") in r.scores
    assert ("agent_no_semantic", "demo_x", "type1") in r.scores


def test_load_results_missing_done_flag_returns_none():
    assert load_results(FIXTURES / "results" / "nonexistent") is None
