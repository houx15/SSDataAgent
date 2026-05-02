from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ssdataagent.config import REPO_ROOT


DATASETS_YAML = REPO_ROOT / "config" / "datasets.yaml"


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    real_data_path: Path
    background_variables: list[str]
    target_variables: list[str]
    descriptions: dict[str, str]
    allowed_values: dict[str, list[Any]]
    numeric_ranges: dict[str, tuple[float, float]]
    population_context: str
    ssdatabench_sim_subdir: str
    evaluation_script: str
    domains: dict[str, str] = field(default_factory=dict)


def _registry() -> dict[str, dict]:
    with DATASETS_YAML.open() as f:
        return yaml.safe_load(f)["datasets"]


def load_schema(name: str) -> DatasetSchema:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown dataset {name!r}; known: {list(reg)}")
    entry = reg[name]
    yaml_path = REPO_ROOT / entry["ssdatabench_yaml"]
    with yaml_path.open() as f:
        spec = yaml.safe_load(f)

    background = list((spec.get("input_variables") or {}).keys())
    targets = list((spec.get("output_variables") or {}).keys())

    descriptions: dict[str, str] = {}
    allowed: dict[str, list[Any]] = {}
    numeric: dict[str, tuple[float, float]] = {}
    domains: dict[str, str] = {}

    combined = {**(spec.get("input_variables") or {}), **(spec.get("output_variables") or {})}
    for var, meta in combined.items():
        meta = meta or {}
        descriptions[var] = meta.get("description", "")
        if meta.get("domain"):
            domains[var] = str(meta["domain"])
        a = meta.get("allowed")
        if isinstance(a, list):
            allowed[var] = a
        elif isinstance(a, dict) and a.get("type") == "numeric":
            numeric[var] = (float(a["min"]), float(a["max"]))

    pop_ctx = spec.get("context") or ""

    return DatasetSchema(
        name=name,
        real_data_path=REPO_ROOT / entry["real_data_path"],
        background_variables=background,
        target_variables=targets,
        descriptions=descriptions,
        allowed_values=allowed,
        numeric_ranges=numeric,
        population_context=str(pop_ctx).strip(),
        ssdatabench_sim_subdir=entry["ssdatabench_sim_subdir"],
        evaluation_script=entry["evaluation_script"],
    )
