#!/usr/bin/env python3
"""Scaffold a new experiment markdown file under docs/experiments/.

    python scripts/new_experiment.py <exp_name> \
        --hypothesis "rubric block in SYSTEM_PROMPT lifts overall by 0.03-0.08" \
        --baseline pilot_paper_agents_gpt54

Creates `docs/experiments/<YYYY-MM-DD>-<exp_name>.md` from `_template.md`,
stamped with the current git sha and active LLM model. Also inserts a
_pending_ row into LEDGER.md so the experiment is tracked the moment it's
planned, not just after it finishes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "docs" / "experiments"
TEMPLATE = EXPERIMENTS_DIR / "_template.md"
LEDGER = EXPERIMENTS_DIR / "LEDGER.md"


def git_sha() -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        .decode()
        .strip()
    )


def active_model() -> str:
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("LLM_MODEL="):
                return line.split("=", 1)[1].split("#", 1)[0].strip()
    cfg = REPO_ROOT / "config" / "llm.yaml"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("model:"):
                return stripped.split(":", 1)[1].strip()
    return "unknown"


def insert_ledger_row(row: str) -> None:
    text = LEDGER.read_text()
    lines = text.splitlines(keepends=True)
    # Insert directly under the table separator row (`|---|---|...`).
    insert_at = next(
        (i + 1 for i, ln in enumerate(lines) if ln.lstrip().startswith("|---")),
        None,
    )
    if insert_at is None:
        raise RuntimeError(f"could not locate table separator in {LEDGER}")
    lines.insert(insert_at, row)
    LEDGER.write_text("".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_name", help="must match a key in config/experiments.yaml")
    p.add_argument("--hypothesis", default="<TODO>")
    p.add_argument(
        "--baseline",
        default="none",
        help="prior exp_name being compared against, or 'none'",
    )
    args = p.parse_args()

    today = dt.date.today().isoformat()
    sha = git_sha()
    model = active_model()
    out_path = EXPERIMENTS_DIR / f"{today}-{args.exp_name}.md"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")

    body = (
        TEMPLATE.read_text()
        .replace("{{exp_name}}", args.exp_name)
        .replace("{{date}}", today)
        .replace("{{model}}", model)
        .replace("{{git_sha}}", sha)
        .replace("{{hypothesis}}", args.hypothesis)
        .replace("{{baseline_exp}}", args.baseline)
    )
    out_path.write_text(body)

    retro_link = f"[retro]({out_path.name})"
    row = (
        f"| {today} | `{args.exp_name}` | {model} | {sha} | "
        f"{args.hypothesis} | _pending_ | {retro_link} |\n"
    )
    insert_ledger_row(row)

    print(out_path)


if __name__ == "__main__":
    main()
