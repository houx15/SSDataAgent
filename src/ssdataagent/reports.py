"""Report core, shared by scripts/generate_exp_report.py and the console.

Pure string-building over flat-file artifacts (eval.json / experiments.yaml /
paper_baselines.json). No I/O beyond reading those paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

T_KEYS = ["type1", "type2", "type3", "type4", "type5"]
T_LABELS = {"type1": "T1", "type2": "T2", "type3": "T3", "type4": "T4", "type5": "T5"}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return "NaN" if v != v else f"{v:.3f}"
    return str(v)


def _delta(a, b) -> str:
    if a is None or b is None or (isinstance(a, float) and a != a) or (isinstance(b, float) and b != b):
        return "—"
    d = a - b
    return f"{'+' if d >= 0 else ''}{d:.3f}"


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---:" if i > 0 else "---" for i in range(len(headers))]) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _latest_eval(results_root: Path, exp: str, condition: str, dataset: str) -> dict | None:
    cond_dir = results_root / exp / condition / dataset
    cands = sorted(cond_dir.glob("*/eval.json"))
    if not cands:
        return None
    return json.loads(cands[-1].read_text())


def _overdetermination_section(cells: dict, datasets: list) -> str:
    headers = ["Dataset", "gap (cell)", "coverage", "n_cells", "gap (model)"]
    rows = []
    for ds in datasets:
        cell = cells.get(ds)
        od = (cell or {}).get("overdetermination") if cell else None
        if not od:
            rows.append([ds, "—", "—", "—", "—"])
            continue
        cb = od.get("cell_based", {}) or {}
        mb = od.get("model_based", {}) or {}
        rows.append([ds, _fmt(cb.get("headline_gap")), _fmt(cb.get("coverage")),
                     _fmt(cb.get("n_cells")), _fmt(mb.get("headline_gap"))])
    return ("## Over-determination gap — `H_real − H_sim` (bits, higher = sim more collapsed)\n\n"
            + _md_table(headers, rows))


def render_markdown_report(experiment: str, *, condition: str = "full_agent",
                           baseline: str | None = None, results_root: Path,
                           experiments_yaml: Path, paper_baselines: Path) -> str:
    spec = yaml.safe_load(Path(experiments_yaml).read_text())["experiments"][experiment]
    paper = json.loads(Path(paper_baselines).read_text())
    by_dataset_paper = paper["by_dataset_by_type"]
    overall_paper = paper["by_dataset_overall"]
    datasets = spec["datasets"]
    cells = {ds: _latest_eval(results_root, experiment, condition, ds) for ds in datasets}

    bits: list[str] = [f"# {experiment} — report\n", "## Strategy\n",
                       f"- **prompt_variant:** `{spec.get('prompt_variant', 'baseline')}`",
                       f"- **datasets:** {', '.join(datasets)}",
                       f"- **conditions:** {', '.join(spec['conditions'])}",
                       f"- **n_rows / dataset:** {spec.get('n_rows')}", ""]

    bits.append(f"## Results — `{condition}`\n")
    headers = ["Dataset"] + [T_LABELS[k] for k in T_KEYS] + ["overall"]
    rows = []
    for ds in datasets:
        cell = cells[ds]
        row = [ds]
        if cell is None:
            row += ["—"] * len(T_KEYS) + ["(no scores)"]
        else:
            by_type = cell.get("by_type", {})
            row += [_fmt(by_type.get(k)) for k in T_KEYS]
            row.append(_fmt(cell.get("overall_average")))
        rows.append(row)
    bits += [_md_table(headers, rows), "", _overdetermination_section(cells, datasets), ""]

    bits.append("## vs Paper-best\n")
    headers = ["Dataset", "T-type", "ours", "paper-best", "Δ"]
    rows = []
    for ds in datasets:
        cell = cells[ds]
        ds_paper = by_dataset_paper.get(ds, {})
        if cell is None:
            rows.append([ds, "—", "(no scores)", "—", "—"]); continue
        by_type = cell.get("by_type", {})
        for k in T_KEYS:
            ours, pb = by_type.get(k), ds_paper.get(T_LABELS[k])
            if ours is None and pb is None:
                continue
            rows.append([ds, T_LABELS[k], _fmt(ours), _fmt(pb), _delta(ours, pb)])
        rows.append([ds, "**overall**", _fmt(cell.get("overall_average")),
                     _fmt(overall_paper.get(ds)),
                     _delta(cell.get("overall_average"), overall_paper.get(ds))])
    bits += [_md_table(headers, rows), ""]
    return "\n".join(bits)


def render_html_report(markdown_text: str) -> str:
    body = MarkdownIt("commonmark", {"html": False}).render(markdown_text)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>SSDataAgent report</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>"
            f"</head><body>{body}</body></html>")
