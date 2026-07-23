#!/usr/bin/env python
"""B3 -- the no-donor LLM full method (nodonor_fullmethod stages) pointed at the TARGET
context, restricted to the crosswalk columns and scored on the identical footing as
B0/B1/B2. Reports three rungs: B3_raw (no repair), B3_elicited (LLM-elicited R^2), and
B3_pool_R2 (pool covariate-R^2, same aggregate footing as B2).

See docs/superpowers/specs/2026-07-23-b3-llm-prior-pointed-at-target-design.md.

    export OPENROUTER_API_KEY=...            # first run only (gss); cps is cached
    .venv/bin/python scripts/transfer_b3.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
    .venv/bin/python scripts/transfer_b3.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
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
from ssdataagent.transfer.b3_specs import SPECS  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, load_pair  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402, F401

OUT = REPO / "results" / "transfer_map"


def b3_columns(pair):
    """(ds, cols, spec, predictors, downstream). `cols` is the crosswalk restricted to the
    source, target pool, and reference columns -- IDENTICAL to run_layer2's `cols` so B3 is
    scored on the same variables as B0/B1/B2."""
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema
    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    ref = nb._drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    b_pool, _ = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    spec = SPECS[ds]
    predictors = [c for c in spec.predictors if c in cols]
    downstream = [c for c in cols if c not in spec.seeds and c not in spec.derived]
    return ds, cols, spec, predictors, downstream


def rung_alphas(raw, spec, elicited, pool, predictors):
    """The three rungs' alpha dicts. raw = all 1.0 (no repair); elicited = alphas from the
    LLM-elicited R^2 targets; pool_R2 = alphas from the pool's covariate-R^2 (a low-order
    aggregate, same footing as B2). Numeric-only outcomes are repaired; cv skips the rest."""
    outcomes = [c for c in raw.columns if c not in predictors]
    elic_targets = {c: elicited[c] for c in outcomes
                    if c not in spec.predictors and elicited.get(c) is not None}
    pool_targets = {c: cv.covariate_r2(pool, c, predictors,
                                       numeric_predictors=spec.numeric_predictors,
                                       log_vars=spec.log_vars)
                    for c in elic_targets}
    elic_alpha = cv.variance_repair_alphas(
        raw, predictors, elic_targets,
        numeric_predictors=spec.numeric_predictors, log_vars=spec.log_vars)
    pool_alpha = cv.variance_repair_alphas(
        raw, predictors, {c: t for c, t in pool_targets.items() if t is not None},
        numeric_predictors=spec.numeric_predictors, log_vars=spec.log_vars)
    return {
        "B3_raw": {c: 1.0 for c in elic_alpha},
        "B3_elicited": elic_alpha,
        "B3_pool_R2": pool_alpha,
    }
