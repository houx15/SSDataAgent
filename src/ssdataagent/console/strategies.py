"""Strategy-centric aggregation for the console's main board.

Pure functions over the same run records the leaderboard uses, plus the PNAS
paper baselines. Groups runs by *method family*, finds each family's best run
(highest overall_average, pilots excluded), and benchmarks every T-type against
the paper's reported best. No I/O.
"""
from __future__ import annotations

import json
from typing import Any

T_KEYS = ["type1", "type2", "type3", "type4", "type5"]
T_LABELS = {"type1": "T1", "type2": "T2", "type3": "T3", "type4": "T4", "type5": "T5"}

# condition name -> (method family, human "how the data was used" label)
CONDITION_INFO: dict[str, tuple[str, str]] = {
    "full_agent": ("Agent", "full context (train microdata + semantics)"),
    "agent_no_semantic": ("Agent", "microdata, variable names hidden"),
    "agent_no_data": ("Agent", "no data access (knowledge only)"),
    "full_agent_unseen": ("Agent", "microdata, one variable held out"),
    "direct_generation": ("Direct", "no data (direct prompt per record)"),
    "hotdeck": ("Hotdeck", "full training microdata"),
    "cart": ("CART", "full training microdata"),
    "copula": ("Copula", "full training microdata"),
    "design_a_full": ("Design A", "full training microdata"),
    "design_a_aggregate": ("Design A", "published marginals only"),
    "design_a_transfer": ("Design A", "earlier survey wave (transfer)"),
    "design_b_full": ("Design B", "full training microdata"),
    "design_b_aggregate": ("Design B", "published marginals only"),
    "design_b_transfer": ("Design B", "earlier survey wave (transfer)"),
    "design_c_full": ("Design C", "full training microdata"),
    "design_c_aggregate": ("Design C", "published marginals only"),
    "design_c_transfer": ("Design C", "earlier survey wave (transfer)"),
    "s1_raw": ("S1", "marginals only (raw LLM distributions)"),
    "s1_raked_full": ("S1", "full microdata (raked)"),
    "s1_raked_transfer": ("S1", "earlier wave, raked (transfer)"),
    "s1_raked_aggregate": ("S1", "published marginals (raked)"),
    "s1_personas": ("S1", "marginals only (persona mixture)"),
}

# how each method works, shown on the detail page
STRATEGY_BLURB: dict[str, str] = {
    "Agent": (
        "An LLM data-analyst agent that iteratively writes and runs Python in a "
        "sandbox to study the training sample and model its joint distribution, "
        "then generates synthetic respondents. Ablation conditions hide the "
        "variable semantics or withhold the microdata entirely."
    ),
    "Direct": (
        "The LLM writes each synthetic respondent directly from a natural-language "
        "prompt, one record at a time, with no code execution or fitted model."
    ),
    "Hotdeck": (
        "Nonparametric donor imputation: each synthetic person copies the target "
        "values of a demographically-matched real training record. Preserves the "
        "real joint distribution exactly but cannot extrapolate beyond it."
    ),
    "CART": (
        "Sequential classification/regression trees: each target is modelled given "
        "the backgrounds and previously-drawn targets, and sampled from the matching "
        "tree leaf."
    ),
    "Copula": (
        "A Gaussian copula fit on the training microdata: correlated latent normals "
        "are sampled and inverted through each target's empirical marginal, "
        "reproducing pairwise associations."
    ),
    "Design A": (
        "The LLM proposes a constrained causal DAG over the targets (backgrounds "
        "exogenous); each node is an in-house Bayesian GLM (BayesianRidge / logistic). "
        "We walk the DAG in topological order and sample each target from its full "
        "predictive distribution conditioned on its parents — the full-predictive draw "
        "avoids the over-determination that point predictions cause."
    ),
    "Design B": (
        "The LLM elicits a target distribution for each demographic cell; these are "
        "IPF-raked to the known marginals, coupled across targets by a data-grounded "
        "Gaussian copula, and sampled. Marginals from the LLM, associations from data."
    ),
    "Design C": (
        "Data-first retrieval and repair: k-NN retrieves real donors by background, "
        "hot-decks candidate target vectors (preserving the real joint), then rakes the "
        "donor weights to a goal marginal so each row emits a real donor value with "
        "corrected margins."
    ),
    "S1": (
        "A distribution-as-output diagnostic (not a contender): the LLM emits a "
        "conditional distribution per demographic cell and targets are sampled "
        "independently (no copula). Variants isolate sampling collapse (raw vs raked) "
        "and within-group diversity (persona mixture)."
    ),
}

# display order (roughly: agent/baselines, then the three designs, then diagnostic)
FAMILY_ORDER = ["Agent", "Direct", "Hotdeck", "CART", "Copula",
                "Design A", "Design B", "Design C", "S1"]


def classify(condition: str) -> tuple[str, str]:
    """(family, data-usage label) for a condition; unknown -> (condition, '')."""
    return CONDITION_INFO.get(condition, (condition, ""))


def _by_type(rec: dict) -> dict[str, Any]:
    try:
        return json.loads(rec.get("by_type_json") or "{}")
    except (TypeError, ValueError):
        return {}


def _pnas_types(by_type: dict, paper_types: dict) -> list[dict]:
    """Per-T-type comparison rows (only T-types present in ours or the paper)."""
    rows = []
    for k in T_KEYS:
        ours = by_type.get(k)
        pb = paper_types.get(T_LABELS[k])
        if ours is None and pb is None:
            continue
        delta = (ours - pb) if (ours is not None and pb is not None) else None
        rows.append({"t": T_LABELS[k], "ours": ours, "paper": pb, "delta": delta})
    return rows


def _run_record(rec: dict) -> dict:
    fam, mode = classify(rec["condition"])
    return {
        "family": fam,
        "data_mode": mode,
        "experiment": rec["experiment"],
        "condition": rec["condition"],
        "dataset": rec["dataset"],
        "run_id": rec.get("run_id"),
        "model": rec.get("model"),
        "overall_average": rec.get("overall_average"),
        "by_type": _by_type(rec),
        "is_pilot": str(rec["experiment"]).startswith("pilot_"),
    }


def build_board(records: list[dict], paper: dict) -> list[dict]:
    """One row per family = its best (highest-overall, non-pilot) run, benchmarked
    against the PNAS paper best for that run's dataset. Sorted by overall desc."""
    paper_overall = paper.get("by_dataset_overall", {})
    paper_by_type = paper.get("by_dataset_by_type", {})
    by_fam: dict[str, list[dict]] = {}
    for rec in records:
        r = _run_record(rec)
        if r["is_pilot"] or r["overall_average"] is None:
            continue
        by_fam.setdefault(r["family"], []).append(r)

    board = []
    for fam, runs in by_fam.items():
        best = max(runs, key=lambda r: r["overall_average"])
        ds = best["dataset"]
        pb_overall = paper_overall.get(ds)
        overall = best["overall_average"]
        board.append({
            "family": fam,
            "blurb": STRATEGY_BLURB.get(fam, ""),
            "best_dataset": ds,
            "data_mode": best["data_mode"],
            "condition": best["condition"],
            "experiment": best["experiment"],
            "run_id": best["run_id"],
            "model": best["model"],
            "overall_average": overall,
            "paper_overall": pb_overall,
            "delta_overall": (overall - pb_overall) if pb_overall is not None else None,
            "types": _pnas_types(best["by_type"], paper_by_type.get(ds, {})),
            "n_runs": len(runs),
        })
    order = {f: i for i, f in enumerate(FAMILY_ORDER)}
    board.sort(key=lambda b: (-b["overall_average"], order.get(b["family"], 99)))
    return board


def build_detail(family: str, records: list[dict], paper: dict) -> dict:
    """All runs for a family grouped by (dataset, condition), each benchmarked
    against the paper, plus the how-it-works blurb, models, and datasets seen."""
    paper_overall = paper.get("by_dataset_overall", {})
    paper_by_type = paper.get("by_dataset_by_type", {})
    runs = [_run_record(r) for r in records]
    runs = [r for r in runs if r["family"] == family and not r["is_pilot"]
            and r["overall_average"] is not None]

    rows = []
    for r in runs:
        ds = r["dataset"]
        pb_overall = paper_overall.get(ds)
        rows.append({
            "dataset": ds,
            "condition": r["condition"],
            "data_mode": r["data_mode"],
            "model": r["model"],
            "run_id": r["run_id"],
            "experiment": r["experiment"],
            "overall_average": r["overall_average"],
            "paper_overall": pb_overall,
            "delta_overall": (r["overall_average"] - pb_overall) if pb_overall is not None else None,
            "types": _pnas_types(r["by_type"], paper_by_type.get(ds, {})),
        })
    rows.sort(key=lambda x: (x["dataset"], x["condition"]))
    return {
        "family": family,
        "blurb": STRATEGY_BLURB.get(family, ""),
        "models": sorted({r["model"] for r in runs if r["model"]}),
        "datasets": sorted({r["dataset"] for r in runs}),
        "data_modes": sorted({r["data_mode"] for r in runs if r["data_mode"]}),
        "n_runs": len(runs),
        "rows": rows,
    }
