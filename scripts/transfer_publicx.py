#!/usr/bin/env python
"""Public X-margins -- admit B's census-standard demographics (age/gender/race) into the
marginal frame; keep the copula from A and Y-marginals from A / the LLM description. Scores
PX_carry / PX_llm alongside FS_* and the ladder references. See
docs/superpowers/specs/2026-07-28-public-x-margins-design.md.

    export OPENROUTER_API_KEY=...     # first run only; Y elicitation is cached (blind cache)
    .venv/bin/python scripts/transfer_publicx.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.data.schema import load_schema  # noqa: E402
from ssdataagent.transfer.blind import build_marg_frame, elicit_marginals  # noqa: E402
from ssdataagent.transfer.generate import transfer_build  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes, load_pair  # noqa: E402
from ssdataagent.transfer.publicx import PUBLIC_X, with_public_x  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def _marginal_tv(a_col, b_col):
    pa = a_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    pb = b_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return 0.5 * float((pa.reindex(idx, fill_value=0.0) - pb.reindex(idx, fill_value=0.0)).abs().sum())


def run_publicx(pair, *, seeds, n, bootstrap_B, dry_run=False):
    """Score FS_carryover(==B0), PX_carry, PX_llm, FS_llm(blind), ref_oracle_comp(==B1),
    ref_floor, ref_ceiling. PX_* admit B's public age/gender/race margins only."""
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
    x_cols = [c for c in PUBLIC_X if c in covs]

    if dry_run:
        # Show the composition gap being fixed and confirm the swap, no LLM / no scorer.
        print(f"{pair.id}: PUBLIC_X in play = {x_cols}", flush=True)
        for c in x_cols:
            gap = _marginal_tv(a[c], b_pool[c])
            swapped = with_public_x(a, b_pool, [c], seed=0)
            gap_after = _marginal_tv(swapped[c], b_pool[c])
            print(f"  {c}: TV(A,B)={gap:.3f}  ->  TV(swapped,B)={gap_after:.3f}", flush=True)
        return None

    elicited = elicit_marginals(ds, a, cols)          # cached; reads source A + B's description
    llm_marg = build_marg_frame(elicited, a, cols)
    print(f"{pair.id}: x_cols={x_cols}; elicited {sum(1 for c in cols if c in elicited)}/{len(cols)}",
          flush=True)

    configs = {
        "FS_carryover":    lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "PX_carry":        lambda s: transfer_build(a, with_public_x(a, b_pool, x_cols, seed=s),
                                                    cols, n, s, "marginal-swap"),
        "PX_llm":          lambda s: transfer_build(a, with_public_x(llm_marg, b_pool, x_cols, seed=s),
                                                    cols, n, s, "marginal-swap"),
        "FS_llm":          lambda s: transfer_build(a, llm_marg, cols, n, s, "marginal-swap"),
        "ref_oracle_comp": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
        "ref_floor":       lambda s: nb.build(b_pool, cols, n, s, "independence"),
        "ref_ceiling":     lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
    }

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(schema.ssdatabench_sim_subdir, set(cols), types, Path(cfg_td))
        for name, builder in configs.items():
            recs = [nb.score(builder(s), ds, ref, types, seed=1000 + s,
                             bootstrap_B=bootstrap_B, config_dir=cfg_dir)
                    for s in range(1, seeds + 1)]
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "x_cols": "|".join(x_cols)}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"publicx_{pair.id}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_id")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the demographic composition gap being fixed (no LLM, no scoring)")
    args = ap.parse_args()
    pair = next(p for p in PAIRS if p.id == args.pair_id)
    run_publicx(pair, seeds=args.seeds, n=args.n, bootstrap_B=args.bootstrap_B, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
