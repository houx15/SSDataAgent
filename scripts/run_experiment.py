"""CLI: python scripts/run_experiment.py --experiment pilot_gss [--resume]

On success writes `results/<exp>/done.flag` with a JSON summary; on failure
writes `failed.flag` with the traceback. The batch runner uses these flags
to decide what to skip when resumed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import yaml

# Bootstrap so this script runs without `pip install -e .` (matches the
# pattern in run_batch.py / status.py / generate_exp_report.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ssdataagent.config import REPO_ROOT, results_root  # noqa: E402
from ssdataagent.evaluation.comparator import summary_pivot, to_long_table  # noqa: E402
from ssdataagent.experiments.runner import ExperimentConfig, run_experiment  # noqa: E402

# Ensure the build_eval_subset helper runs before the experiment.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_eval_subset import build_subset  # noqa: E402


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "experiments.yaml"))
    args = p.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())["experiments"][args.experiment]
    cfg = ExperimentConfig(name=args.experiment, **spec)
    exp_root = results_root() / cfg.name
    exp_root.mkdir(parents=True, exist_ok=True)
    done_flag = exp_root / "done.flag"
    failed_flag = exp_root / "failed.flag"

    if args.resume and done_flag.exists():
        print(f"[run_experiment] {cfg.name}: done.flag exists, nothing to do")
        return
    # Clear stale failure marker before re-attempting.
    failed_flag.unlink(missing_ok=True)

    try:
        for dataset in cfg.datasets:
            build_subset(dataset)
        rates = run_experiment(cfg, resume=args.resume)
        long = to_long_table(rates)
        pivot = summary_pivot(long)
        print("\n=== Summary (mean pass rate by condition x dataset) ===")
        print(pivot)
        out = exp_root / "summary.csv"
        long.to_csv(out, index=False)
        print(f"\nWrote {out}")

        # Stamp completion AFTER summary.csv is on disk so a later reader can
        # trust that done.flag implies summary.csv exists.
        done_flag.write_text(json.dumps({
            "experiment": cfg.name,
            "finished_at": _now_iso(),
            "n_dataset_condition_cells": len(rates),
            "prompt_variant": cfg.prompt_variant,
            "llm_model": cfg.llm_model,
            "llm_provider": cfg.llm_provider,
        }, indent=2))
        print(f"Wrote {done_flag}")
    except Exception as e:
        failed_flag.write_text(json.dumps({
            "experiment": cfg.name,
            "failed_at": _now_iso(),
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
        }, indent=2))
        print(f"[run_experiment] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"Wrote {failed_flag}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
