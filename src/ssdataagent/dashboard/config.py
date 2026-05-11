"""Load config/experiments.yaml into per-experiment dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    exp_name: str
    datasets: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    max_iterations: int | None = None
    sandbox_timeout: int | None = None
    train_eval_split: float | None = None
    n_rows: int | None = None
    prompt_variant: str = ""
    llm_model: str = ""
    llm_provider: str = ""
    llm_base_url: str = ""


def load_configs(path: Path) -> dict[str, ExperimentConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    experiments = raw.get("experiments", {}) or {}
    out: dict[str, ExperimentConfig] = {}
    for name, body in experiments.items():
        body = body or {}
        out[name] = ExperimentConfig(
            exp_name=name,
            datasets=list(body.get("datasets", [])),
            conditions=list(body.get("conditions", [])),
            max_iterations=body.get("max_iterations"),
            sandbox_timeout=body.get("sandbox_timeout"),
            train_eval_split=body.get("train_eval_split"),
            n_rows=body.get("n_rows"),
            prompt_variant=body.get("prompt_variant", ""),
            llm_model=body.get("llm_model", ""),
            llm_provider=body.get("llm_provider", ""),
            llm_base_url=body.get("llm_base_url", ""),
        )
    return out
