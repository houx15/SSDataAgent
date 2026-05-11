"""Load per-experiment results: summary.csv and done.flag."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


ScoreKey = tuple[str, str, str]  # (condition, dataset, type)


@dataclass
class ExperimentResults:
    exp_name: str
    prompt_variant: str
    llm_model: str
    llm_provider: str
    finished_at: str
    n_dataset_condition_cells: int
    scores: dict[ScoreKey, float] = field(default_factory=dict)


def load_results(exp_dir: Path) -> ExperimentResults | None:
    exp_dir = Path(exp_dir)
    done_flag = exp_dir / "done.flag"
    summary = exp_dir / "summary.csv"
    if not done_flag.exists():
        return None
    flag = json.loads(done_flag.read_text(encoding="utf-8"))
    scores = _load_summary(summary) if summary.exists() else {}
    return ExperimentResults(
        exp_name=flag.get("experiment", exp_dir.name),
        prompt_variant=flag.get("prompt_variant", ""),
        llm_model=flag.get("llm_model", ""),
        llm_provider=flag.get("llm_provider", ""),
        finished_at=flag.get("finished_at", ""),
        n_dataset_condition_cells=flag.get("n_dataset_condition_cells", 0),
        scores=scores,
    )


def _load_summary(path: Path) -> dict[ScoreKey, float]:
    scores: dict[ScoreKey, float] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["condition"], row["dataset"], row["type"])
                scores[key] = float(row["pass_rate"])
            except (KeyError, ValueError):
                continue
    return scores
