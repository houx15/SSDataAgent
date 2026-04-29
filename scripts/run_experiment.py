"""CLI: python scripts/run_experiment.py --experiment pilot_gss [--resume]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ssdataagent.config import REPO_ROOT
from ssdataagent.evaluation.comparator import summary_pivot, to_long_table
from ssdataagent.experiments.runner import ExperimentConfig, run_experiment

# Ensure the build_eval_subset helper runs before the experiment.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_eval_subset import build_subset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "experiments.yaml"))
    args = p.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())["experiments"][args.experiment]
    cfg = ExperimentConfig(name=args.experiment, **spec)

    for dataset in cfg.datasets:
        build_subset(dataset)

    rates = run_experiment(cfg, resume=args.resume)
    long = to_long_table(rates)
    pivot = summary_pivot(long)
    print("\n=== Summary (mean pass rate by condition x dataset) ===")
    print(pivot)
    out = REPO_ROOT / "results" / cfg.name / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
