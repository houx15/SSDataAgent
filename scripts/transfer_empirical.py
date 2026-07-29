#!/usr/bin/env python
"""Empirical-copula transfer -- a faithful structure model. Scores EC_carry (A's joint + A's
marginals, reads no B) and EC_oracle (A's joint + B's true marginals) against the current
engine and the ladder references. See docs/superpowers/specs/2026-07-29-empirical-copula-design.md.

    .venv/bin/python scripts/transfer_empirical.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
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
from ssdataagent.transfer.empirical_copula import empirical_transfer  # noqa: E402
from ssdataagent.transfer.generate import transfer_build  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes, load_pair  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def run_empirical(pair, *, seeds, n, bootstrap_B):
    """Score carryover(current engine), EC_carry, EC_oracle, ref_oracle_comp(==B1),
    ref_floor, ref_ceiling -- identical protocol to B0-B6. Only EC_oracle + ref_oracle_comp +
    the scorer touch B's pool; EC_carry reads no B."""
    import nodonor_bracket as nb

    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    schema = load_schema(ds)
    ref = nb._drop_unnamed(pd.read_csv(schema.real_data_path, low_memory=False))
    b_pool, guarantee = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    covariates_outcomes(pair.schema_name, cols)   # (kept for parity with the harness)
    types = nb.TYPES.get(ds, (1, 2, 3))

    configs = {
        "carryover":       lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "EC_carry":        lambda s: empirical_transfer(a, a, cols, n, s),
        "EC_oracle":       lambda s: empirical_transfer(a, b_pool, cols, n, s),
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
            row = {"pair": pair.id, "config": name, "guarantee": guarantee}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"empirical_{pair.id}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_id")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    args = ap.parse_args()
    pair = next(p for p in PAIRS if p.id == args.pair_id)
    run_empirical(pair, seeds=args.seeds, n=args.n, bootstrap_B=args.bootstrap_B)


if __name__ == "__main__":
    main()
