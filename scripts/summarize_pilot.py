"""Summarize a pilot's eval.json files into the SSDataBench reporting style.

Reads every eval.json under results/<experiment>/<condition>/<dataset>/<run>/
and prints:
  - Headline pivot: condition × dataset → overall pass rate
  - Per-type table: condition × dataset × type
  - Per-domain table: condition × dataset × domain (rolled up from by_variable)

Usage:
    python scripts/summarize_pilot.py pilot_paper_agents
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ssdataagent.config import results_root  # noqa: E402
from ssdataagent.data.schema import load_schema  # noqa: E402
from ssdataagent.evaluation.runner import PassRates, by_domain  # noqa: E402


def _load_eval(p: Path, results_dir: Path) -> dict:
    blob = json.loads(p.read_text())
    parts = p.relative_to(results_dir).parts
    # parts = (experiment, condition, dataset, run_id, "eval.json")
    return {
        "experiment": parts[0],
        "condition": parts[1],
        "dataset": parts[2],
        "run_id": parts[3],
        "blob": blob,
    }


def _domain_view(blob: dict, dataset: str) -> dict[str, dict[str, float]]:
    if "by_domain" in blob:
        return blob["by_domain"]
    # Backfill from by_variable using the schema (older runs lacked by_domain).
    rates = PassRates(by_variable=blob.get("by_variable", {}))
    try:
        return by_domain(rates, load_schema(dataset))
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment")
    p.add_argument("--results-root", default=str(results_root()))
    args = p.parse_args()

    root = Path(args.results_root) / args.experiment
    if not root.exists():
        sys.exit(f"no such experiment dir: {root}")

    results_dir = Path(args.results_root)
    rows = [_load_eval(p, results_dir) for p in sorted(root.rglob("eval.json"))]
    if not rows:
        sys.exit(f"no eval.json files under {root}")

    # 1) Headline overall_average
    head = pd.DataFrame([
        {
            "condition": r["condition"],
            "dataset": r["dataset"],
            "overall": r["blob"].get("overall_average"),
        }
        for r in rows
    ]).pivot_table(index="condition", columns="dataset", values="overall", aggfunc="mean")

    # 2) by_type
    type_rows = []
    for r in rows:
        for t, rate in (r["blob"].get("by_type") or {}).items():
            type_rows.append({
                "condition": r["condition"], "dataset": r["dataset"],
                "type": t, "rate": rate,
            })
    types = pd.DataFrame(type_rows).pivot_table(
        index=["condition", "type"], columns="dataset", values="rate", aggfunc="mean"
    ) if type_rows else pd.DataFrame()

    # 3) by_domain (with backfill)
    dom_rows = []
    for r in rows:
        dom = _domain_view(r["blob"], r["dataset"])
        # Aggregate across types within domain (mean of all type-domain cells)
        per_dom: dict[str, list[float]] = {}
        for type_name, dmap in dom.items():
            for d, rate in dmap.items():
                per_dom.setdefault(d, []).append(float(rate))
        for d, rs in per_dom.items():
            dom_rows.append({
                "condition": r["condition"], "dataset": r["dataset"],
                "domain": d, "rate": sum(rs) / len(rs),
            })
    domains = pd.DataFrame(dom_rows).pivot_table(
        index=["condition", "domain"], columns="dataset", values="rate", aggfunc="mean"
    ) if dom_rows else pd.DataFrame()

    print("\n=== Headline (overall_average) ===")
    print(head.round(3))
    print("\n=== Per-type ===")
    print(types.round(3))
    print("\n=== Per-domain (mean over types within domain) ===")
    print(domains.round(3))

    out = root / "summary_full.csv"
    flat = pd.DataFrame(rows)[["condition", "dataset", "run_id"]].assign(
        overall=[r["blob"].get("overall_average") for r in rows],
    )
    flat.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
