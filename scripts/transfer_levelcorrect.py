#!/usr/bin/env python
"""Level-correction channel -- location-shifted marginals for cross-context transfer.
Keeps source A's marginal SHAPE + copula; corrects only each numeric outcome's LOCATION for
the target from oracle / LLM-description / sibling-pooled / ESS-gated-hybrid estimates. See
docs/superpowers/specs/2026-07-27-level-correction-design.md.

    export OPENROUTER_API_KEY=...     # first run only; level elicitation is cached
    .venv/bin/python scripts/transfer_levelcorrect.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.data.schema import load_schema  # noqa: E402
from ssdataagent.transfer.generate import transfer_build  # noqa: E402
from ssdataagent.transfer.levelcorrect import (  # noqa: E402
    apply_affine_shift, apply_level_shift, assemble_shifts, numeric_outcomes,
    oracle_affine, oracle_shifts, pooled_shifts,
)
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes, load_pair  # noqa: E402
from ssdataagent.transfer.retrieval import reweighted_pool, sibling_csvs  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

RAKE_CANDIDATES = ("age", "gender", "race")
POOL_N = 50000          # weighted resample size for a stable pooled-mean estimate
OUT = REPO / "results" / "transfer_map"


def _pool(pair, a, b_pool, cols, covs, *, seed=0):
    """Sibling pool raked to B's public X-margins -> (sib_rew, ess, n_siblings)."""
    import nodonor_bracket as nb
    sib_paths = sibling_csvs(pair)
    sibs = [nb._drop_unnamed(pd.read_csv(p, low_memory=False)) for p in sib_paths]
    rake_cols = [c for c in RAKE_CANDIDATES if c in covs]
    rng = np.random.default_rng(seed)
    sib_rew, ess = reweighted_pool(sibs, b_pool, cols, rake_cols, POOL_N, rng)
    return sib_rew, ess, len(sibs)


def run_levelcorrect(pair, *, seeds, n, bootstrap_B, dry_run=False, configs_filter=None):
    """Score LC_none(==B0), LC_oracle, LC_llm, LC_pooled, LC_hybrid, ref_oracle_comp(==B1),
    ref_floor, ref_ceiling on the crosswalk cols -- identical protocol to B0-B6/face-swap.
    Only ref_oracle_comp + LC_oracle + the scorer touch B's pool; LC_llm reads B's
    description; LC_pooled/LC_hybrid read B's public X-margins."""
    import nodonor_bracket as nb

    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    schema = load_schema(ds)
    ref = nb._drop_unnamed(pd.read_csv(schema.real_data_path, low_memory=False))
    b_pool, guarantee = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    covs, outs = covariates_outcomes(pair.schema_name, cols)
    types = nb.TYPES.get(ds, (1, 2, 3))

    ys = numeric_outcomes(a, b_pool, outs)
    sib_rew, ess, n_sib = _pool(pair, a, b_pool, cols, covs)

    if dry_run:
        # Real-data wiring check without the LLM or the scorer (deterministic arms only).
        print(f"{pair.id}: numeric outcomes={ys}", flush=True)
        print(f"  siblings={n_sib} ess={ess:.3f} rake_cols={[c for c in RAKE_CANDIDATES if c in covs]}",
              flush=True)
        print(f"  oracle Δ={ {y: round(v, 3) for y, v in oracle_shifts(a, b_pool, ys).items()} }",
              flush=True)
        print(f"  pooled Δ={ {y: round(v, 3) for y, v in pooled_shifts(a, sib_rew, ys).items()} }",
              flush=True)
        return None

    affine_oracle = oracle_affine(a, b_pool, ys)
    # Only the additive/estimated arms need the LLM+pooled assembly; skip it (and the LLM call)
    # when the selected configs don't include them -- e.g. the cheap LC_oracle_affine test.
    est_arms = ("LC_llm", "LC_pooled", "LC_hybrid", "LC_oracle")
    need_est = configs_filter is None or any(c in configs_filter for c in est_arms)
    shifts = (assemble_shifts(a, b_pool, sib_rew, ds, ys, n_sib, ess) if need_est
              else {"oracle": oracle_shifts(a, b_pool, ys), "llm": {}, "pooled": {}, "hybrid": {}})
    print(f"{pair.id}: numeric outcomes {ys}; siblings {n_sib} ess {ess:.3f}", flush=True)

    configs = {
        "LC_none":         lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "LC_oracle":       lambda s: transfer_build(a, apply_level_shift(a, shifts["oracle"]),
                                                    cols, n, s, "marginal-swap"),
        "LC_llm":          lambda s: transfer_build(a, apply_level_shift(a, shifts["llm"]),
                                                    cols, n, s, "marginal-swap"),
        "LC_pooled":       lambda s: transfer_build(a, apply_level_shift(a, shifts["pooled"]),
                                                    cols, n, s, "marginal-swap"),
        "LC_hybrid":       lambda s: transfer_build(a, apply_level_shift(a, shifts["hybrid"]),
                                                    cols, n, s, "marginal-swap"),
        "LC_oracle_affine": lambda s: transfer_build(a, apply_affine_shift(a, affine_oracle),
                                                     cols, n, s, "marginal-swap"),
        "ref_oracle_comp": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
        "ref_floor":       lambda s: nb.build(b_pool, cols, n, s, "independence"),
        "ref_ceiling":     lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
    }

    if configs_filter:
        configs = {k: v for k, v in configs.items() if k in configs_filter}

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(schema.ssdatabench_sim_subdir, set(cols), types,
                                      Path(cfg_td))
        for name, builder in configs.items():
            recs = [nb.score(builder(s), ds, ref, types, seed=1000 + s,
                             bootstrap_B=bootstrap_B, config_dir=cfg_dir)
                    for s in range(1, seeds + 1)]
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "n_numeric": len(ys), "ess": round(ess, 3), "n_siblings": n_sib}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if not configs_filter else "_partial"   # don't clobber the full ladder CSV
    df.to_csv(OUT / f"levelcorrect_{pair.id}{suffix}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_id")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print shifts on real data (no LLM, no scoring)")
    ap.add_argument("--configs", default=None,
                    help="comma-separated config names to score (subset); writes *_partial.csv")
    args = ap.parse_args()
    pair = next(p for p in PAIRS if p.id == args.pair_id)
    cf = set(args.configs.split(",")) if args.configs else None
    run_levelcorrect(pair, seeds=args.seeds, n=args.n, bootstrap_B=args.bootstrap_B,
                     dry_run=args.dry_run, configs_filter=cf)


if __name__ == "__main__":
    main()
