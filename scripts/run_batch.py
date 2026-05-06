"""Run a list of experiments back-to-back. Resumable.

Usage (typically inside tmux on a cloud box):

    python scripts/run_batch.py exp001_rubric exp002_thresholds exp003_preserve_miss

Each experiment runs as a subprocess invocation of `scripts/run_experiment.py`
with stdout+stderr captured to `results/<exp>/run.log`. Sequential by design —
parallel API budgets are usually tight, sequential resume logic is trivial.

Resume semantics:
    - if `results/<exp>/done.flag` exists, skip silently
    - if `results/<exp>/failed.flag` exists, retry (the run script clears it
      on next attempt)
    - one experiment failing doesn't block the rest of the batch

After every experiment, `results/_batch_status.json` is rewritten so a `cat`
over SSH gives at-a-glance state without ls'ing through directories.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ssdataagent.config import results_root  # noqa: E402

RESULTS_ROOT = results_root()
EXPERIMENTS_YAML = REPO_ROOT / "config" / "experiments.yaml"
BATCH_STATUS = RESULTS_ROOT / "_batch_status.json"


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _exp_state(exp: str) -> str:
    """Source of truth: the flag files. Anything else can drift."""
    exp_dir = RESULTS_ROOT / exp
    if (exp_dir / "done.flag").exists():
        return "done"
    if (exp_dir / "failed.flag").exists():
        return "failed"
    if (exp_dir / "run.log").exists():
        return "running_or_interrupted"
    return "pending"


def _write_status(batch: list[str], current: str | None) -> None:
    BATCH_STATUS.parent.mkdir(parents=True, exist_ok=True)
    BATCH_STATUS.write_text(json.dumps({
        "updated_at": _now_iso(),
        "current": current,
        "experiments": {e: _exp_state(e) for e in batch},
    }, indent=2))


def _validate_exp_names(names: list[str]) -> None:
    spec = yaml.safe_load(EXPERIMENTS_YAML.read_text())["experiments"]
    unknown = [n for n in names if n not in spec]
    if unknown:
        raise SystemExit(
            f"unknown experiment(s): {unknown}\n"
            f"known: {sorted(spec)}"
        )


def _run_one(exp: str, *, resume: bool, force: bool) -> int:
    exp_dir = RESULTS_ROOT / exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    if not force and (exp_dir / "done.flag").exists():
        print(f"[batch {_now_iso()}] {exp}: SKIP (done.flag exists)")
        return 0

    log_path = exp_dir / "run.log"
    print(f"[batch {_now_iso()}] {exp}: START -> {log_path}")
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"),
        "--experiment", exp,
    ]
    if resume:
        cmd.append("--resume")
    with log_path.open("w") as logf:
        logf.write(f"[batch] cmd: {' '.join(cmd)}\n")
        logf.write(f"[batch] started_at: {_now_iso()}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            cwd=REPO_ROOT, env=os.environ,
        )
        logf.write(f"\n[batch] finished_at: {_now_iso()} exit={proc.returncode}\n")
    state = _exp_state(exp)
    print(f"[batch {_now_iso()}] {exp}: DONE exit={proc.returncode} state={state}")
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiments", nargs="+",
                   help="experiment names from config/experiments.yaml")
    p.add_argument("--no-resume", action="store_true",
                   help="don't pass --resume to run_experiment "
                        "(default: pass it, so per-condition retries work)")
    p.add_argument("--force", action="store_true",
                   help="re-run experiments even if done.flag exists")
    args = p.parse_args()

    _validate_exp_names(args.experiments)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_status(args.experiments, current=None)

    failures: list[str] = []
    for exp in args.experiments:
        _write_status(args.experiments, current=exp)
        rc = _run_one(exp, resume=not args.no_resume, force=args.force)
        if rc != 0:
            failures.append(exp)
        _write_status(args.experiments, current=None)

    print(f"\n[batch {_now_iso()}] BATCH COMPLETE — {len(args.experiments)} experiments, "
          f"{len(failures)} failure(s)")
    if failures:
        print(f"failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
