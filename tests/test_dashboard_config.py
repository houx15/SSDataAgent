from pathlib import Path

from ssdataagent.dashboard.config import ExperimentConfig, load_configs

FIXTURES = Path(__file__).parent / "fixtures" / "dashboard"


def test_load_configs_returns_dict_keyed_by_exp_name():
    configs = load_configs(FIXTURES / "experiments.yaml")
    assert "exp_demo_a" in configs
    assert "exp_demo_b_cross" in configs
    assert "pilot_demo" in configs


def test_load_configs_pulls_per_exp_fields():
    configs = load_configs(FIXTURES / "experiments.yaml")
    a = configs["exp_demo_a"]
    assert a.datasets == ["demo"]
    assert a.conditions == ["full_agent"]
    assert a.n_rows == 100
    assert a.prompt_variant == "rubric_tools_v1"
    assert a.llm_model == "gpt-5.4-2026-03-05"


def test_load_configs_tolerates_missing_keys():
    configs = load_configs(FIXTURES / "experiments.yaml")
    pilot = configs["pilot_demo"]
    assert pilot.sandbox_timeout is None
    assert pilot.train_eval_split is None
