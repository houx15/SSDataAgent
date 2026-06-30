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
sys.path.insert(0, str(REPO_ROOT / "src"))
from ssdataagent.config import results_root  # noqa: E402
from ssdataagent import reports  # noqa: E402

RESULTS_ROOT = results_root()
EXPERIMENTS_YAML = REPO_ROOT / "config" / "experiments.yaml"
PAPER_BASELINES = REPO_ROOT / "config" / "paper_baselines.json"
STRATEGY_MD = REPO_ROOT / "docs" / "experiments" / "STRATEGY.md"
REPORTS_DIR = REPO_ROOT / "docs" / "experiments"

T_KEYS = reports.T_KEYS
T_LABELS = reports.T_LABELS


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


def _fmt(v) -> str:
    return reports._fmt(v)


def _delta(a, b) -> str:
    return reports._delta(a, b)


def _md_table(headers, rows) -> str:
    return reports._md_table(headers, rows)


def _overdetermination_section(cells: dict, datasets: list) -> str:
    return reports._overdetermination_section(cells, datasets)


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
    strategy_line = _strategy_blurb(args.experiment)

    datasets: list[str] = spec["datasets"]
    baseline_cells: dict[str, dict | None] = {}
    if args.baseline:
        baseline_cells = {
            ds: _latest_eval(args.baseline, args.condition, ds) for ds in datasets
        }

    # Build the shared report body (Strategy + Results + vs Paper)
    body = reports.render_markdown_report(
        args.experiment,
        condition=args.condition,
        baseline=args.baseline,
        results_root=RESULTS_ROOT,
        experiments_yaml=EXPERIMENTS_YAML,
        paper_baselines=PAPER_BASELINES,
    )

    # The shared body uses a generic header; we want the script's richer header
    # (date, llm_model, max_iterations, strategy_line, A/B baseline note).
    # Replace the first heading line and Strategy section.
    today = dt.date.today().isoformat()

    strategy_section_lines: list[str] = []
    strategy_section_lines.append(f"# {args.experiment} — report ({today})\n")
    strategy_section_lines.append("## Strategy\n")
    if strategy_line:
        strategy_section_lines.append(f"From STRATEGY.md backlog:\n\n> {strategy_line}\n")
    strategy_section_lines.append(f"- **prompt_variant:** `{spec.get('prompt_variant', 'baseline')}`")
    strategy_section_lines.append(f"- **llm_model:** `{spec.get('llm_model') or '(from .env)'}`")
    strategy_section_lines.append(f"- **datasets:** {', '.join(datasets)}")
    strategy_section_lines.append(f"- **conditions:** {', '.join(spec['conditions'])}")
    strategy_section_lines.append(f"- **n_rows / dataset:** {spec.get('n_rows')}")
    strategy_section_lines.append(f"- **max_iterations:** {spec.get('max_iterations')}")
    if args.baseline:
        strategy_section_lines.append(f"- **A/B baseline experiment:** `{args.baseline}`")
    strategy_section_lines.append("")

    # Strip the generic header and Strategy block from the shared body, then
    # prepend the richer script-specific header.
    body_lines = body.splitlines()
    # Find where "## Results" begins in the shared body to splice after Strategy
    results_start = 0
    for i, line in enumerate(body_lines):
        if line.startswith("## Results"):
            results_start = i
            break

    bits: list[str] = strategy_section_lines + body_lines[results_start:]

    # Section 4: vs baseline (only if --baseline supplied)
    if args.baseline:
        paper = json.loads(PAPER_BASELINES.read_text())
        cells: dict[str, dict | None] = {
            ds: _latest_eval(args.experiment, args.condition, ds) for ds in datasets
        }
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
