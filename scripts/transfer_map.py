#!/usr/bin/env python
"""Build the transfer map: Layer-1 diagnostics (composition-vs-mechanism + copula
stability) over all pairs, and Layer-2 firewalled B0/B1 baselines over the scored pairs.

See docs/superpowers/specs/2026-07-22-transfer-map-design.md. No LLM, no API key.

    .venv/bin/python scripts/transfer_map.py
    .venv/bin/python scripts/transfer_map.py --pairs cps_1970_1980 --seeds 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.transfer.copula_stability import copula_stability  # noqa: E402
from ssdataagent.transfer.decompose import _is_num, kob_decompose, oaxaca_blinder  # noqa: E402
from ssdataagent.transfer.generate import transfer_build, transfer_build_b2  # noqa: E402
from ssdataagent.transfer.pairs import (  # noqa: E402
    PAIRS, covariates_outcomes, load_pair,
)
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

OUT = REPO / "results" / "transfer_map"

# Layer-2 ladder order — single source of truth, checked against `run_layer2`'s
# `configs` dict so the two cannot drift apart (see the assert in `run_layer2`).
LAYER2_CONFIG_NAMES = [
    "B0_carryover", "B1_marginal_swap", "B2_recalibrated",
    "within_B_floor", "within_B_ceiling",
]

# Composition axes for the KOB reweighting: the demographic core the benchmark itself hands
# systems as `input: true`. Raking on these (small, low-cardinality, always-supported) gives
# a stable composition estimate; the full background set (parental occupation, immigrant
# status) adds high-cardinality margins that concentrate the weights and add noise.
CORE_DEMOGRAPHICS = ("age", "gender", "race")


def composition_covariates(covariates: list[str]) -> list[str]:
    """The demographic core present in this pair's covariates; fall back to all covariates
    if none of the core axes survived the crosswalk."""
    core = [c for c in covariates if c in CORE_DEMOGRAPHICS]
    return core or covariates



def run_layer1(a: pd.DataFrame, b: pd.DataFrame, cols: list[str],
               covariates: list[str], outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostic map (reads both contexts' microdata — the answer key, not firewalled)."""
    comp_cov = composition_covariates(covariates)
    rows = []
    for y in outcomes:
        d = kob_decompose(a, b, y, comp_cov)
        # Oaxaca-Blinder is a numeric-outcome cross-check only; skip categorical outcomes
        # (an all-NaN coerced design produces meaningless terms + numpy warnings).
        if _is_num(a[y]) and _is_num(b[y]):
            try:
                ob = oaxaca_blinder(a, b, y, comp_cov, numeric_predictors=frozenset(
                    c for c in comp_cov if _is_num(a[c])))
                d["composition_share_ob"] = ob["composition_share_ob"]
            except Exception:
                d["composition_share_ob"] = float("nan")
        else:
            d["composition_share_ob"] = float("nan")
        rows.append(d)
    kob = pd.DataFrame(rows)
    cop = copula_stability(a, b, cols)
    return kob, cop


def run_layer2(pair, *, seeds: int, n: int, bootstrap_B: int) -> pd.DataFrame:
    """Firewalled B0/B1 baselines scored against the target's benchmark reference.

    Scored on the crosswalk variables only (config restricted to what the source can carry).
    """
    import tempfile

    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema

    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    schema = load_schema(pair.target_dataset)
    ref = nb._drop_unnamed(pd.read_csv(schema.real_data_path, low_memory=False))
    b_pool, guarantee = nb.carve_pool(pair.target_dataset)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    types = nb.TYPES.get(pair.target_dataset, (1, 2, 3))

    out = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(schema.ssdatabench_sim_subdir, set(cols), types,
                                      Path(cfg_td))

        def _score_many(builder):
            recs = [nb.score(builder(s), pair.target_dataset, ref, types,
                             seed=1000 + s, bootstrap_B=bootstrap_B, config_dir=cfg_dir)
                    for s in range(1, seeds + 1)]
            return mean_scores(pd.DataFrame(recs))

        covs, outs = covariates_outcomes(pair.schema_name, cols)

        configs = {
            "B0_carryover": lambda s: transfer_build(a, a, cols, n, s, "carryover"),
            "B1_marginal_swap": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
            "B2_recalibrated": lambda s: transfer_build_b2(a, b_pool, cols, covs, outs, n, s),
            "within_B_floor": lambda s: nb.build(b_pool, cols, n, s, "independence"),
            "within_B_ceiling": lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
        }
        assert list(configs.keys()) == LAYER2_CONFIG_NAMES  # single source of truth for order
        for name, builder in configs.items():
            rec = {"pair": pair.id, "config": name, "guarantee": guarantee}
            rec.update(_score_many(builder))
            out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=None, help="pair ids (default: all)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--no-scoring", action="store_true",
                    help="Layer 1 only (skip the slow Layer-2 firewalled baseline scoring)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in PAIRS if a.pairs is None or p.id in a.pairs]

    all_map, all_cop, all_base = [], [], []
    for p in pairs:
        src, tgt, cols = load_pair(p)
        cov, outc = covariates_outcomes(p.schema_name, cols)
        kob, cop = run_layer1(src, tgt, cols, cov, outc)
        kob.insert(0, "pair", p.id); cop.insert(0, "pair", p.id)
        kob.to_csv(OUT / f"map_{p.id}.csv", index=False)
        all_map.append(kob); all_cop.append(cop)
        print(f"\n=== {p.id} (Layer 1) ===")
        print(kob[["response", "composition_share", "mechanism_share", "label"]].to_string(index=False))
        stable = (cop["label"] == "stable").mean() if len(cop) else float("nan")
        print(f"copula stable fraction: {stable:.2f}  ({len(cop)} pairs)")
        if p.scored and not a.no_scoring:
            base = run_layer2(p, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B)
            base.to_csv(OUT / f"baselines_{p.id}.csv", index=False)
            all_base.append(base)
            print(f"--- {p.id} (Layer 2, firewalled baselines) ---")
            print(base.to_string(index=False))

    if all_map:
        pd.concat(all_map, ignore_index=True).to_csv(OUT / "map.csv", index=False)
        pd.concat(all_cop, ignore_index=True).to_csv(OUT / "copula.csv", index=False)
    if all_base:
        pd.concat(all_base, ignore_index=True).to_csv(OUT / "baselines.csv", index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
