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

    # Longitudinal datasets list `type: sequential` variables (income_14..65,
    # education_14..65, etc.) that expand to dozens of per-age columns. The
    # SSDataBench T1-T5 evals only reference the *static* aggregates (e.g.,
    # mean_income_30_40, age_finished_education) — generating sequential cols
    # would balloon the agent's prompt without adding eval coverage.
    def _is_static(meta: dict | None) -> bool:
        return (meta or {}).get("type", "static") != "sequential"
    inputs = spec.get("input_variables") or {}
    outputs = spec.get("output_variables") or {}
    background = [v for v, m in inputs.items() if _is_static(m)]
    targets = [v for v, m in outputs.items() if _is_static(m)]

    descriptions: dict[str, str] = {}
    allowed: dict[str, list[Any]] = {}
    numeric: dict[str, tuple[float, float]] = {}
    domains: dict[str, str] = {}

    combined = {**inputs, **outputs}
    for var, meta in combined.items():
        meta = meta or {}
        if not _is_static(meta):
            continue
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
        domains=domains,
    )
