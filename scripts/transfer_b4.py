#!/usr/bin/env python
"""B4 -- retrieval + KOB transport (Phase 3, slice 1). Transport the target's joint
Y-structure (T2 associations + T3 covariate-R^2) from leave-one-context-out same-instrument
siblings, raked to the target's public X-marginals, through the existing B1/B2 shared-latent
machinery. Reads NO target Y-side joint aggregate. Two configs:

  B4_retrieval          -- R^2 target transported from the reweighted siblings (fully
                           firewalled: reads no target covariate-R^2).
  B4_retrieval_targetR2 -- R^2 target kept from the target pool (B2's source); isolates the
                           retrieval/reweighting effect. Diagnostic.

See docs/superpowers/specs/2026-07-23-b4-retrieval-kob-transport-design.md. LLM-free.

    .venv/bin/python scripts/transfer_b4.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
    .venv/bin/python scripts/transfer_b4.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
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

from ssdataagent.transfer.generate import transfer_build_b2  # noqa: E402
from ssdataagent.transfer.pairs import (  # noqa: E402
    PAIRS, covariates_outcomes, load_pair,
)
from ssdataagent.transfer.retrieval import reweighted_pool, sibling_csvs  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402
from ssdataagent.transfer.target_aggregates import target_aggregates  # noqa: E402
from transfer_map import composition_covariates  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def b4_columns(pair):
    """(ds, cols, covs, outs). ``cols`` is derived IDENTICALLY to run_layer2 -- source `a`
    ∩ target pool ∩ reference -- so B4 is scored on the same variables as B0-B3."""
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema
    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    ref = nb._drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    b_pool, _ = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    covs, outs = covariates_outcomes(pair.schema_name, cols)
    return ds, cols, covs, outs


def _load_siblings(pair, cols):
    """LOCO sibling frames restricted to ``cols``; keep only siblings containing every
    crosswalk column (a sibling missing a col cannot carry that variable's structure).
    Returns (frames_by_wave, dropped) -- both dicts wave_name -> frame / reason."""
    import nodonor_bracket as nb
    frames, dropped = {}, {}
    for csv in sibling_csvs(pair):
        f = nb._drop_unnamed(pd.read_csv(csv, low_memory=False))
        missing = [c for c in cols if c not in f.columns]
        if missing:
            dropped[csv.stem] = f"missing {missing}"
        else:
            frames[csv.stem] = f[cols]
    return frames, dropped


def reweighted_pool_for(pair, cols, target_pool, rng, n_rew=None):
    """Build the raked sibling pseudo-population for ``pair``. n_rew defaults to the stack
    size (preserve scale). Returns (sib_rew, ess_ratio, used_waves, dropped)."""
    # composition_covariates(cols) returns {age,gender,race} ∩ cols (the raking axes),
    # falling back to all cols only if none of the demographic core survived the crosswalk.
    # Firewall-safe either way: raking (raking_weights) reads only per-column UNIVARIATE
    # target marginals, never a joint -- so even the fallback stays within "public
    # marginals". For both scored pairs the demographic core survives, so the fallback is
    # never hit today.
    rake_cols = composition_covariates(cols)
    frames, dropped = _load_siblings(pair, cols)
    if not frames:
        raise RuntimeError(f"{pair.id}: no usable siblings (dropped: {dropped})")
    sib_list = list(frames.values())
    stack_size = sum(len(f) for f in sib_list)
    sib_rew, ess = reweighted_pool(sib_list, target_pool, cols, rake_cols,
                                   n_rew or stack_size, rng)
    return sib_rew, ess, list(frames), dropped


def per_sibling_r2(pair, cols, covs, outs, target_pool, rng):
    """Each sibling's individually-raked outcome-R^2 (diagnostic spread). A wide spread
    across waves is itself the finding: mechanism drifts -> the learned model is needed."""
    rake_cols = composition_covariates(cols)
    frames, _ = _load_siblings(pair, cols)
    out = {}
    for wave, f in frames.items():
        sr, _ = reweighted_pool([f], target_pool, cols, rake_cols, len(f), rng)
        out[wave] = target_aggregates(sr, cols, covs, outs)["outcome_r2"]
    return out


def run_b4(pair, *, seeds, n, bootstrap_B, n_rew=None):
    """Build sib_rew once -> score B4_retrieval (r2_pool=sib_rew) and B4_retrieval_targetR2
    (r2_pool=None, i.e. target pool) over seeds -> write results CSV. Firewalled: reads the
    target's marginals + CORE_DEMOGRAPHICS margins only; the joint Y-structure is
    transported from siblings."""
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema

    ds, cols, covs, outs = b4_columns(pair)
    target_pool, guarantee = nb.carve_pool(ds)
    ref = nb._drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    types = nb.TYPES.get(ds, (1, 2, 3))

    sib_rew, ess, used, dropped = reweighted_pool_for(
        pair, cols, target_pool, np.random.default_rng(0), n_rew)
    print(f"{pair.id}: siblings used {used}; dropped {dropped}; "
          f"sib_rew {len(sib_rew)} rows; ess_ratio {ess:.3f}")
    spread = per_sibling_r2(pair, cols, covs, outs, target_pool, np.random.default_rng(0))
    print(f"{pair.id}: per-sibling outcome_r2 (spread diagnostic): {spread}")

    configs = {
        "B4_retrieval": dict(r2_pool=sib_rew),        # transported R^2 -- fully firewalled
        "B4_retrieval_targetR2": dict(r2_pool=None),  # keep B2's target R^2 -- diagnostic
    }
    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(load_schema(ds).ssdatabench_sim_subdir,
                                      set(cols), types, Path(cfg_td))
        for name, kw in configs.items():
            recs = []
            for s in range(1, seeds + 1):
                sim = transfer_build_b2(sib_rew, target_pool, cols, covs, outs, n, s, **kw)
                recs.append(nb.score(sim, ds, ref, types, seed=1000 + s,
                                     bootstrap_B=bootstrap_B, config_dir=cfg_dir))
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "ess_ratio": round(ess, 4)}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"b4_{pair.id}.csv", index=False)
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
    df = run_b4(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'b4_{pair.id}.csv'}")
    print("REGIME: no-donor + stricter. Target supplies marginals + X-margins (raking) only;"
          " joint Y-structure transported from LOCO siblings.")


if __name__ == "__main__":
    main()
