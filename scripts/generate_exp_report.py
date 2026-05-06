"""Build the per-experiment markdown report.

    python scripts/generate_exp_report.py exp001_rubric_in_system_prompt
    python scripts/generate_exp_report.py exp001_rubric_in_system_prompt \\
        --baseline pilot_paper_agents_gpt54        # show A/B vs prior run

Reads:
    config/experiments.yaml             — the experiment's settings
    results/<exp>/full_agent/<ds>/<run>/eval.json  — per-cell T1-T5 scores
    config/paper_baselines.json         — paper-best per (dataset, T-type)
    docs/experiments/STRATEGY.md        — backlog item if it exists

Writes:
    docs/experiments/<YYYY-MM-DD>-<exp>-report.md

with three sections:
    Strategy   — hypothesis + variant deltas (yaml + STRATEGY)
    Results    — per-(dataset, T-type) table
    vs Paper   — same shape, side-by-side with paper-best, with Δ column
    [optional] vs Baseline — A/B vs --baseline experiment

Designed to be re-runnable: existing report is overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENTS_YAML = REPO_ROOT / "config" / "experiments.yaml"
PAPER_BASELINES = REPO_ROOT / "config" / "paper_baselines.json"
STRATEGY_MD = REPO_ROOT / "docs" / "experiments" / "STRATEGY.md"
REPORTS_DIR = REPO_ROOT / "docs" / "experiments"

T_KEYS = ["type1", "type2", "type3", "type4", "type5"]
T_LABELS = {"type1": "T1", "type2": "T2", "type3": "T3", "type4": "T4", "type5": "T5"}


def _exp_spec(exp: str) -> dict:
    spec = yaml.safe_load(EXPERIMENTS_YAML.read_text())["experiments"]
    if exp not in spec:
        raise SystemExit(f"unknown experiment {exp!r}")
    return spec[exp]


def _latest_eval(exp: str, condition: str, dataset: str) -> dict | None:
    cond_dir = RESULTS_ROOT / exp / condition / dataset
    if not cond_dir.exists():
        return None
    candidates = sorted(cond_dir.glob("*/eval.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def _strategy_blurb(exp: str) -> str:
    """Pull the matching backlog line out of STRATEGY.md, if present."""
    if not STRATEGY_MD.exists():
        return ""
    text = STRATEGY_MD.read_text()
    # Backlog lines look like:  "- [ ] **EXP-001** — Add T1-T5 rubric block ..."
    # Match by exp name OR by an "EXP-" tag heuristically extracted from the name.
    for line in text.splitlines():
        if exp in line:
            return line.strip()
    m = re.match(r"exp\s*[-_]?(\d+)", exp.lower())
    if m:
        tag = f"EXP-{int(m.group(1)):03d}"
        for line in text.splitlines():
            if tag in line:
                return line.strip()
    return ""


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---:" if i > 0 else "---" for i in range(len(headers))]) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return "NaN" if v != v else f"{v:.3f}"
    return str(v)


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None or (isinstance(a, float) and a != a) or (isinstance(b, float) and b != b):
        return "—"
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment")
    p.add_argument("--condition", default="full_agent",
                   help="condition to report on (default: full_agent)")
    p.add_argument("--baseline", default=None,
                   help="prior experiment to A/B against (optional)")
    p.add_argument("--out", default=None, help="output path (default: docs/experiments/...)")
    args = p.parse_args()

    spec = _exp_spec(args.experiment)
    paper = json.loads(PAPER_BASELINES.read_text())
    by_dataset_paper = paper["by_dataset_by_type"]
    overall_paper = paper["by_dataset_overall"]
    strategy_line = _strategy_blurb(args.experiment)

    datasets: list[str] = spec["datasets"]
    cells: dict[str, dict | None] = {
        ds: _latest_eval(args.experiment, args.condition, ds) for ds in datasets
    }
    baseline_cells: dict[str, dict | None] = {}
    if args.baseline:
        baseline_cells = {
            ds: _latest_eval(args.baseline, args.condition, ds) for ds in datasets
        }

    # Section 1: Strategy
    bits: list[str] = []
    today = dt.date.today().isoformat()
    bits.append(f"# {args.experiment} — report ({today})\n")
    bits.append("## Strategy\n")
    if strategy_line:
        bits.append(f"From STRATEGY.md backlog:\n\n> {strategy_line}\n")
    bits.append(f"- **prompt_variant:** `{spec.get('prompt_variant', 'baseline')}`")
    bits.append(f"- **llm_model:** `{spec.get('llm_model') or '(from .env)'}`")
    bits.append(f"- **datasets:** {', '.join(datasets)}")
    bits.append(f"- **conditions:** {', '.join(spec['conditions'])}")
    bits.append(f"- **n_rows / dataset:** {spec.get('n_rows')}")
    bits.append(f"- **max_iterations:** {spec.get('max_iterations')}")
    if args.baseline:
        bits.append(f"- **A/B baseline experiment:** `{args.baseline}`")
    bits.append("")

    # Section 2: Results — per (dataset, T-type)
    bits.append(f"## Results — `{args.condition}`\n")
    headers = ["Dataset"] + [T_LABELS[k] for k in T_KEYS] + ["overall"]
    rows = []
    for ds in datasets:
        cell = cells[ds]
        row: list[str] = [ds]
        if cell is None:
            row += ["—"] * len(T_KEYS) + ["(no eval)"]
        else:
            by_type = cell.get("by_type", {})
            for k in T_KEYS:
                row.append(_fmt(by_type.get(k)))
            row.append(_fmt(cell.get("overall_average")))
        rows.append(row)
    bits.append(_md_table(headers, rows))
    bits.append("")

    # Section 3: vs paper
    bits.append("## vs Paper-best (best of 15 LLMs in SSDataBench)\n")
    bits.append("Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.\n")
    headers = ["Dataset", "T-type", "ours", "paper-best", "Δ"]
    rows = []
    for ds in datasets:
        cell = cells[ds]
        ds_paper = by_dataset_paper.get(ds, {})
        if cell is None:
            rows.append([ds, "—", "(no eval)", "—", "—"])
            continue
        by_type = cell.get("by_type", {})
        for k in T_KEYS:
            ours = by_type.get(k)
            pb = ds_paper.get(T_LABELS[k])
            if ours is None and pb is None:
                continue  # T4/T5 don't apply to cross-sectional, skip blank row
            rows.append([ds, T_LABELS[k], _fmt(ours), _fmt(pb), _delta(ours, pb)])
        # Overall
        rows.append([ds, "**overall**",
                     _fmt(cell.get("overall_average")),
                     _fmt(overall_paper.get(ds)),
                     _delta(cell.get("overall_average"), overall_paper.get(ds))])
    bits.append(_md_table(headers, rows))
    bits.append("")

    # Section 4: vs baseline (only if --baseline supplied)
    if args.baseline:
        bits.append(f"## vs Baseline experiment `{args.baseline}`\n")
        bits.append("Δ = this experiment − baseline. Positive = improvement.\n")
        headers = ["Dataset", "T-type", "this", "baseline", "Δ"]
        rows = []
        for ds in datasets:
            cell = cells[ds]
            bcell = baseline_cells.get(ds)
            if cell is None or bcell is None:
                rows.append([ds, "—", _fmt(cell.get("overall_average") if cell else None),
                             _fmt(bcell.get("overall_average") if bcell else None), "—"])
                continue
            by_type = cell.get("by_type", {})
            b_by_type = bcell.get("by_type", {})
            for k in T_KEYS:
                if by_type.get(k) is None and b_by_type.get(k) is None:
                    continue
                rows.append([ds, T_LABELS[k], _fmt(by_type.get(k)),
                             _fmt(b_by_type.get(k)),
                             _delta(by_type.get(k), b_by_type.get(k))])
            rows.append([ds, "**overall**",
                         _fmt(cell.get("overall_average")),
                         _fmt(bcell.get("overall_average")),
                         _delta(cell.get("overall_average"), bcell.get("overall_average"))])
        bits.append(_md_table(headers, rows))
        bits.append("")

    # Retro placeholder so the report doubles as the working document.
    bits.append("## Retro\n")
    bits.append("- **What worked:**")
    bits.append("- **What didn't:**")
    bits.append("- **Surprises:**")
    bits.append("- **Lesson worth preserving:**")
    bits.append("- **Next experiment:**\n")

    out_path = Path(args.out) if args.out else REPORTS_DIR / f"{today}-{args.experiment}-report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(bits))
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
