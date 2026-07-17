#!/usr/bin/env python
"""How noisy is the SSDataBench scorer itself, on FIXED input?

Motivation: two runs of `nodonor_fullmethod.py cps` with identical seeds and identical
cached input disagreed (raw T3 0.047 vs 0.007 vs 0.027). The generator was not the
cause. The scorer is:

  * `rng = np.random.default_rng()` is UNSEEDED (type1.py:41, type2/3/4/5 likewise), so
    every scoring run draws different bootstrap samples. Nothing is reproducible.
  * `bootstrap_B: 1` for T1 in EVERY dataset config -- one 500-row bootstrap draw
    decides each variable's pass/fail, so T1 is quantized coin-flipping (k/7 on cps).
    T2-T5 use B=10, which is better but still small.

This script scores ONE fixed simulated frame R times. All spread in the output is the
benchmark's own measurement noise -- a floor on what any comparison can resolve.

    sigma_per_call            the noise in a single score
    SE(R=5)  = sigma/sqrt(5)  the noise in a 5-seed mean, our usual protocol
    min_resolvable ~ 2.8 * SE the smallest difference two 5-seed means can distinguish
                              (a two-sided 0.05 test on a difference of means)

Usage:
    .venv/bin/python scripts/bench_noise.py cps [--reps 30] [--n 5000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import ssdataagent.data.conditional_variance as cv  # noqa: E402
from nodonor_bracket import TYPES, _drop_unnamed, build, carve_pool, score  # noqa: E402
from ssdataagent.data.schema import load_schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()

    ds = a.dataset
    ref = _drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    pool, guarantee = carve_pool(ds)
    schema = load_schema(ds)
    modeled = [c for c in ref.columns
               if (c in schema.domains or c in schema.background_variables)]
    cols = [c for c in modeled if c in pool.columns and c != "profile_id"]
    types = TYPES.get(ds, (1, 2, 3, 4, 5))

    # One frame, built once, never rebuilt. Any spread below is the SCORER.
    sim = build(pool, cols, a.n, seed=1, mode="rowresample")
    print(f"{ds}: one FIXED rowresample frame (n={a.n}), scored {a.reps}x")
    print("all spread below is the benchmark's own noise -- the input never changes\n")

    rows = [score(sim, ds, ref, types) for _ in range(a.reps)]
    d = pd.DataFrame(rows)
    tcols = [f"T{t}" for t in types] + ["overall"]

    print(f"{'':<18}" + "".join(f"{c:>10}" for c in tcols))
    print("-" * (18 + 10 * len(tcols)))
    for label, fn in (("mean", lambda s: s.mean()), ("sigma_per_call", lambda s: s.std()),
                      ("min", lambda s: s.min()), ("max", lambda s: s.max())):
        print(f"{label:<18}" + "".join(
            f"{fn(d[c]):>10.3f}" if d[c].notna().any() else f"{'--':>10}" for c in tcols))
    print(f"{'range(max-min)':<18}" + "".join(
        f"{(d[c].max()-d[c].min()):>10.3f}" if d[c].notna().any() else f"{'--':>10}"
        for c in tcols))
    print()
    print(f"{'SE(R=5)':<18}" + "".join(
        f"{d[c].std()/np.sqrt(5):>10.3f}" if d[c].notna().any() else f"{'--':>10}"
        for c in tcols))
    print(f"{'min_resolvable':<18}" + "".join(
        f"{2.8*d[c].std()/np.sqrt(5):>10.3f}" if d[c].notna().any() else f"{'--':>10}"
        for c in tcols))
    print("\nmin_resolvable = the smallest gap two 5-seed means can distinguish at "
          "p<0.05.\nAny reported difference smaller than that is noise, not a result.")


if __name__ == "__main__":
    main()
