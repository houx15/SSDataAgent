from pathlib import Path

import yaml
import pytest

from ssdataagent.console import forking


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "experiments.yaml"
    p.write_text(yaml.safe_dump({"experiments": {
        "base": {"datasets": ["gss"], "conditions": ["full_agent"],
                 "max_iterations": 3, "sandbox_timeout": 60,
                 "train_eval_split": 0.5, "n_rows": 1000},
        "other": {"datasets": ["cps"], "conditions": ["hotdeck"],
                  "max_iterations": 1, "sandbox_timeout": 60,
                  "train_eval_split": 0.5, "n_rows": 100},
    }}))
    return p


def test_fork_creates_entry_and_preserves_siblings(yaml_path: Path):
    entry = forking.fork_experiment(
        yaml_path, "base", "base_fork", {"n_rows": 500, "prompt_variant": "rubric"})
    assert entry["n_rows"] == 500
    assert entry["prompt_variant"] == "rubric"
    assert entry["datasets"] == ["gss"]            # inherited from base
    data = yaml.safe_load(yaml_path.read_text())["experiments"]
    assert set(data) == {"base", "base_fork", "other"}   # siblings preserved
    assert data["base"]["n_rows"] == 1000          # base untouched


def test_fork_unknown_base_raises(yaml_path: Path):
    with pytest.raises(KeyError):
        forking.fork_experiment(yaml_path, "nope", "x", {})


def test_fork_duplicate_name_raises(yaml_path: Path):
    with pytest.raises(ValueError):
        forking.fork_experiment(yaml_path, "base", "other", {})
