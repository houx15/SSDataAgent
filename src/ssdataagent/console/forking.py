"""Fork a config/experiments.yaml entry, preserving siblings."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_experiments(yaml_path: Path) -> dict:
    data = yaml.safe_load(Path(yaml_path).read_text()) or {}
    return data.get("experiments", {})


def fork_experiment(yaml_path: Path, base: str, new_name: str,
                    overrides: dict) -> dict:
    yaml_path = Path(yaml_path)
    data = yaml.safe_load(yaml_path.read_text()) or {}
    experiments = data.setdefault("experiments", {})
    if base not in experiments:
        raise KeyError(f"unknown base experiment {base!r}")
    if new_name in experiments:
        raise ValueError(f"experiment {new_name!r} already exists")
    entry = copy.deepcopy(experiments[base])
    entry.update(overrides)
    experiments[new_name] = entry
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return entry
