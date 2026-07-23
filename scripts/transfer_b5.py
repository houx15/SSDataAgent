#!/usr/bin/env python
"""B5 -- learned R^2 rescue (Phase 3, slice 2). Empirical-Bayes shrinkage of B4's
retrieval R^2 toward a cross-context pooled prior, ESS-gated. LLM-free.

    .venv/bin/python scripts/transfer_b5.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
    .venv/bin/python scripts/transfer_b5.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
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

from ssdataagent.config import data_root  # noqa: E402
from ssdataagent.data.conditional_variance import covariate_r2  # noqa: E402
from ssdataagent.data.schema import load_schema  # noqa: E402
from ssdataagent.transfer.generate import _is_numeric, transfer_build_b2  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes  # noqa: E402
from ssdataagent.transfer.retrieval import reweighted_pool  # noqa: E402
from ssdataagent.transfer.rescue import (  # noqa: E402
    fit_noise, fit_prior, outcome_features, predict_r2,
)
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402
from ssdataagent.transfer.target_aggregates import target_aggregates  # noqa: E402
from transfer_b4 import b4_columns, reweighted_pool_for  # noqa: E402
from transfer_map import composition_covariates  # noqa: E402

OUT = REPO / "results" / "transfer_map"

# Single-wave, schema-backed datasets contributing prior rows only (no siblings).
_SINGLE_WAVE = ("acs", "nlsy", "addhealth", "cfps", "us")


def _load(csv: Path) -> pd.DataFrame:
    import nodonor_bracket as nb
    return nb._drop_unnamed(pd.read_csv(csv, low_memory=False))


def corpus_contexts() -> list[tuple[str, Path, str]]:
    """(context_id, csv, schema_name) for every schema-backed context on disk. cps
    and gss enumerate all waves (same-instrument siblings); single-wave datasets use
    their schema real_data_path. Skips datasets whose CSV is absent."""
    out: list[tuple[str, Path, str]] = []
    for csv in sorted((data_root() / "cps").glob("cps-asec*.csv")):
        out.append((f"cps_{csv.stem[-4:]}", csv, "cps"))
    for csv in sorted((data_root() / "gss").glob("gss*.csv")):
        out.append((csv.stem, csv, "gss"))
    for name in _SINGLE_WAVE:
        try:
            p = load_schema(name).real_data_path
        except KeyError:
            continue
        if p.exists():
            out.append((name, p, name))
    return out


def context_records(context_id: str, csv: Path, schema_name: str) -> list[dict]:
    """True covariate-R^2 + structural features for every resolvable native outcome
    of one context. Outcomes whose R^2 is None (too few rows / categorical) are
    skipped with no row. This is the ground truth the prior is fit on."""
    df = _load(csv)
    sch = load_schema(schema_name)
    covs = [c for c in sch.background_variables if c in df.columns]
    outs = [c for c in sch.target_variables if c in df.columns]
    num_pred = frozenset(c for c in covs if _is_numeric(df[c]))
    rows: list[dict] = []
    for o in outs:
        r2 = covariate_r2(df, o, covs, numeric_predictors=num_pred)
        if r2 is None or r2 != r2:                       # None or NaN -> unusable
            continue
        feats = outcome_features(df, o, covs, numeric_predictors=num_pred)
        rows.append({"context": context_id, "outcome": o,
                     "true_r2": float(np.clip(r2, 0.0, 1.0)), **feats})
    return rows


def training_rows(exclude_context_ids: set[str]) -> list[dict]:
    """All context_records across the corpus, minus the held-out context(s)."""
    rows: list[dict] = []
    for cid, csv, sch in corpus_contexts():
        if cid in exclude_context_ids:
            continue
        rows.extend(context_records(cid, csv, sch))
    return rows


def _cps_wave_csvs() -> list[Path]:
    return sorted((data_root() / "cps").glob("cps-asec*.csv"))


def noise_points(exclude_csv: Path) -> list[tuple[float, float]]:
    """(ess, squared_error) calibration points from cps waves as pseudo-targets: for
    each cps wave w (!= exclude_csv), rake the OTHER cps waves to w's margins, and
    compare the transported R^2 against w's TRUE R^2 per shared outcome. cps always
    has >=2 remaining siblings, so every ESS point is well-supported. gss is never a
    pseudo-target (its only sibling is the real target -> would leak)."""
    exclude = exclude_csv.resolve()
    waves = _cps_wave_csvs()
    pts: list[tuple[float, float]] = []
    for w in waves:
        if w.resolve() == exclude:
            continue
        sibs = [s for s in waves if s.resolve() != w.resolve()]
        if len(sibs) < 1:
            continue
        wpool = _load(w)
        sch = load_schema("cps")
        cols = [c for c in (list(sch.background_variables) + list(sch.target_variables))
                if c in wpool.columns and all(c in _load(s).columns for s in sibs)]
        covs, outs = covariates_outcomes("cps", cols)
        rake = composition_covariates(cols)
        sib_frames = [_load(s)[cols] for s in sibs]
        stack_n = sum(len(f) for f in sib_frames)
        sib_rew, ess = reweighted_pool(sib_frames, wpool, cols, rake, stack_n,
                                       np.random.default_rng(0))
        num_pred = frozenset(c for c in covs if _is_numeric(wpool[c]))
        x = target_aggregates(sib_rew, cols, covs, outs)["outcome_r2"]
        for o in outs:
            true = covariate_r2(wpool, o, covs, numeric_predictors=num_pred)
            if true is None or x.get(o) is None:
                continue
            pts.append((ess, float((x[o] - true) ** 2)))
    return pts
