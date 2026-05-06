"""Print at-a-glance state of every experiment under results/.

Source of truth is the flag files (results/<exp>/done.flag, failed.flag,
run.log) — no separate db, no drift. Run from anywhere over SSH:

    python scripts/status.py
    python scripts/status.py exp001 exp002        # filter to specific names
    python scripts/status.py --watch              # refresh every 5s

Exit code is 0 if every experiment scanned is `done`, 1 otherwise — useful
for a final `&& echo DONE | mail you@example.com` in tmux.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
BATCH_STATUS = RESULTS_ROOT / "_batch_status.json"

GLYPH = {
    "done": "✓",
    "running_or_interrupted": "⟳",
    "failed": "✗",
    "pending": "·",
}


def _state(exp_dir: Path) -> str:
    if (exp_dir / "done.flag").exists():
        return "done"
    if (exp_dir / "failed.flag").exists():
        return "failed"
    if (exp_dir / "run.log").exists():
        return "running_or_interrupted"
    return "pending"


def _failure_reason(exp_dir: Path) -> str:
    flag = exp_dir / "failed.flag"
    try:
        blob = json.loads(flag.read_text())
    except Exception:
        return "(failed.flag unreadable)"
    return f"{blob.get('error_type', '?')}: {blob.get('error_msg', '?')[:80]}"


def _done_summary(exp_dir: Path) -> str:
    flag = exp_dir / "done.flag"
    try:
        blob = json.loads(flag.read_text())
    except Exception:
        return ""
    bits = []
    if blob.get("prompt_variant"):
        bits.append(f"variant={blob['prompt_variant']}")
    if blob.get("llm_model"):
        bits.append(f"model={blob['llm_model']}")
    if blob.get("n_dataset_condition_cells") is not None:
        bits.append(f"cells={blob['n_dataset_condition_cells']}")
    return " ".join(bits)


def _scan(filter_names: list[str] | None) -> list[tuple[str, str, str]]:
    if not RESULTS_ROOT.exists():
        return []
    rows: list[tuple[str, str, str]] = []
    for child in sorted(RESULTS_ROOT.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if filter_names and child.name not in filter_names:
            continue
        st = _state(child)
        if st == "failed":
            extra = _failure_reason(child)
        elif st == "done":
            extra = _done_summary(child)
        else:
            extra = ""
        rows.append((st, child.name, extra))
    return rows


def _print(rows: list[tuple[str, str, str]]) -> None:
    if not rows:
        print("(no experiments found under results/)")
        return
    width = max(len(name) for _, name, _ in rows)
    by_state = {"done": 0, "running_or_interrupted": 0, "failed": 0, "pending": 0}
    for st, name, extra in rows:
        by_state[st] = by_state.get(st, 0) + 1
        glyph = GLYPH.get(st, "?")
        line = f"  {glyph} {name:<{width}}  [{st}]"
        if extra:
            line += f"  {extra}"
        print(line)
    print()
    summary = "  ".join(f"{GLYPH[s]} {by_state.get(s, 0)} {s}"
                        for s in ["done", "running_or_interrupted", "failed", "pending"])
    print(summary)
    if BATCH_STATUS.exists():
        try:
            blob = json.loads(BATCH_STATUS.read_text())
            print(f"\nbatch updated_at={blob.get('updated_at')} current={blob.get('current')}")
        except Exception:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="*", help="filter to specific experiment names")
    p.add_argument("--watch", action="store_true",
                   help="refresh every 5s (Ctrl-C to exit)")
    args = p.parse_args()

    while True:
        rows = _scan(args.names or None)
        if args.watch:
            print("\033[2J\033[H", end="")  # clear screen
        _print(rows)
        if not args.watch:
            return 0 if rows and all(s == "done" for s, _, _ in rows) else 1
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
