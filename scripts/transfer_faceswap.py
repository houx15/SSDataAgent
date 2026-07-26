#!/usr/bin/env python
"""Blind face-swap (Approach A) -- description-only cross-context generation. Transfer
source A's copula; get the target's marginals from an LLM reading only the target's
audited description. Scores FS_llm alongside deterministic references. See
docs/superpowers/specs/2026-07-26-blind-faceswap-design.md.

    export OPENROUTER_API_KEY=...        # first run only; elicitation is cached
    .venv/bin/python scripts/transfer_faceswap.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
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
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def run_faceswap(pair, *, seeds, n, bootstrap_B):
    """Score FS_carryover (==B0), FS_llm (blind face-swap), ref_oracle_comp (==B1, reads B),
    ref_floor, ref_ceiling -- all on the crosswalk cols, identical protocol to B0-B6.
    The FS_* paths read only source A + B's description; only ref_oracle_comp and the
    scorer touch B's pool."""
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

    # Blind elicitation (cached) -> synthetic marginal frame. Reads source A + description.
    elicited = elicit_marginals(ds, a, cols)
    n_elicited = sum(1 for c in cols if c in elicited)
    llm_marg = build_marg_frame(elicited, a, cols)
    print(f"{pair.id}: elicited {n_elicited}/{len(cols)} marginals; "
          f"{len(cols) - n_elicited} fell back to source A", flush=True)

    configs = {
        "FS_carryover":    lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "FS_llm":          lambda s: transfer_build(a, llm_marg, cols, n, s, "marginal-swap"),
        "ref_oracle_comp": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
        "ref_floor":       lambda s: nb.build(b_pool, cols, n, s, "independence"),
        "ref_ceiling":     lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
    }

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(schema.ssdatabench_sim_subdir, set(cols), types,
                                      Path(cfg_td))
        for name, builder in configs.items():
            recs = [nb.score(builder(s), ds, ref, types, seed=1000 + s,
                             bootstrap_B=bootstrap_B, config_dir=cfg_dir)
                    for s in range(1, seeds + 1)]
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "n_elicited": n_elicited if name == "FS_llm" else ""}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"faceswap_{pair.id}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", choices=[p.id for p in PAIRS if p.scored])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    a = ap.parse_args()
    pair = [p for p in PAIRS if p.id == a.pair][0]
    df = run_faceswap(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'faceswap_{pair.id}.csv'}")
    print("REGIME: blind. FS_* read only source A + the target's audited text description; "
          "ref_oracle_comp/scorer read B's pool (labeled upper bound / yardstick).")


if __name__ == "__main__":
    main()
