#!/usr/bin/env python
"""B6 -- ESS-gated hybrid generator (Phase 4 integration). One autonomous config
that selects B5's retrieval-blended R^2 vs. the pooled-prior R^2 from firewall-clean
signals (n_siblings, ESS), provenance-tagged. LLM-free.

    .venv/bin/python scripts/transfer_b6.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
    .venv/bin/python scripts/transfer_b6.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
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
from ssdataagent.transfer.generate import transfer_build_b2  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS  # noqa: E402
from ssdataagent.transfer.rescue import hybrid_r2_map, select_r2_source  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402
from transfer_b4 import b4_columns  # noqa: E402
from transfer_b5 import _load, predict_target_r2  # noqa: E402

OUT = REPO / "results" / "transfer_map"

TAU = 0.3


def run_b6(pair, *, seeds, n, bootstrap_B, tau: float = TAU):
    """Score the single B6_hybrid config: gate (n_siblings, ess) picks B5's learned
    or prior_only R^2 map, then score through the IDENTICAL B4/B5 sib_rew vehicle via
    the r2_target seam. Writes CSV with source + n_siblings provenance columns."""
    import nodonor_bracket as nb
    ds, cols, covs, outs = b4_columns(pair)
    target_pool, guarantee = nb.carve_pool(ds)
    ref = _load(load_schema(ds).real_data_path)
    types = nb.TYPES.get(ds, (1, 2, 3))

    learned, prior_only, ess, sib_rew, n_siblings = predict_target_r2(pair)
    use_retrieval = select_r2_source(n_siblings, ess, tau=tau)
    r2_map, provenance = hybrid_r2_map(learned, prior_only, use_retrieval)
    source = "retrieval" if use_retrieval else "prior"
    print(f"{pair.id}: n_siblings {n_siblings}, ess {ess:.3f}, tau {tau} -> "
          f"source={source}")
    print(f"{pair.id}: provenance {provenance}")

    recs = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(load_schema(ds).ssdatabench_sim_subdir,
                                      set(cols), types, Path(cfg_td))
        for s in range(1, seeds + 1):
            sim = transfer_build_b2(sib_rew, target_pool, cols, covs, outs,
                                    n, s, r2_target=r2_map)
            recs.append(nb.score(sim, ds, ref, types, seed=1000 + s,
                                 bootstrap_B=bootstrap_B, config_dir=cfg_dir))

    row = {"pair": pair.id, "config": "B6_hybrid", "guarantee": guarantee,
           "ess_ratio": round(ess, 4), "n_siblings": n_siblings, "source": source}
    row.update(mean_scores(pd.DataFrame(recs)))
    df = pd.DataFrame([row])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"b6_{pair.id}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", choices=[p.id for p in PAIRS if p.scored])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--tau", type=float, default=TAU)
    a = ap.parse_args()
    pair = [p for p in PAIRS if p.id == a.pair][0]
    df = run_b6(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B, tau=a.tau)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'b6_{pair.id}.csv'}")
    print("REGIME: no-donor + hybrid. Target supplies marginals + X-margins + public"
          " outcome features only; conditional strength is auto-selected between the"
          " retrieval blend and the cross-context prior by an ESS gate reading no score.")


if __name__ == "__main__":
    main()
