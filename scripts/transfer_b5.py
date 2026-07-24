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


def _cps_pseudo_targets(exclude_csv: Path) -> list[tuple[Path, list[Path]]]:
    """(pseudo_target_wave, sibling_waves) for cps calibration, LOCO-clean: the real
    target ``exclude_csv`` is dropped as a pseudo-target AND from every sibling pool,
    so no calibration point ever reads the held-out target's microdata. Path-level
    only (no I/O), so it is cheap to unit-test."""
    exclude = exclude_csv.resolve()
    waves = [w for w in _cps_wave_csvs() if w.resolve() != exclude]
    return [(w, [s for s in waves if s.resolve() != w.resolve()]) for w in waves]


def noise_points(exclude_csv: Path) -> list[tuple[float, float]]:
    """(ess, squared_error) calibration points from cps waves as pseudo-targets: for
    each cps wave w (!= exclude_csv), rake the OTHER cps waves to w's margins, and
    compare the transported R^2 against w's TRUE R^2 per shared outcome. cps always
    has >=2 remaining siblings, so every ESS point is well-supported. gss is never a
    pseudo-target (its only sibling is the real target -> would leak)."""
    pts: list[tuple[float, float]] = []
    for w, sibs in _cps_pseudo_targets(exclude_csv):
        if not sibs:
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


def _target_context_id(pair) -> str:
    """The corpus context_id of the pair's TARGET wave, to hold it out under LOCO."""
    stem = pair.target_csv.stem
    if pair.schema_name == "cps":
        return f"cps_{stem[-4:]}"
    return stem                                       # gss1994 / gss2018


def predict_target_r2(pair):
    """Fit the EB model LOCO (target wave excluded), then predict the scored pair's
    crosswalk-outcome R^2 two ways: full posterior (learned) and prior-only. Also
    returns B4's raked sibling pool (the structure vehicle) so the scorer draws
    through the IDENTICAL vehicle B4 used -- B5 vs B4 then differs ONLY in the R^2
    target. Returns (learned_r2, prior_only_r2, ess, sib_rew)."""
    import nodonor_bracket as nb
    ds, cols, covs, outs = b4_columns(pair)
    target_pool, _ = nb.carve_pool(ds)

    # Retrieval data point x_co + ESS, reusing B4's raked sibling pool (default_rng(0)
    # -> byte-identical to B4's sib_rew).
    sib_rew, ess, _, _ = reweighted_pool_for(pair, cols, target_pool,
                                             np.random.default_rng(0))
    x_co = target_aggregates(sib_rew, cols, covs, outs)["outcome_r2"]

    # Fit prior on all contexts except the held-out target wave; noise on cps waves
    # (excluding the target if it is a cps wave).
    prior = fit_prior(training_rows({_target_context_id(pair)}))
    noise = fit_noise(noise_points(pair.target_csv))
    print(f"{pair.id}: prior tau2 {prior.tau2:.4f} coef {np.round(prior.coef, 3)}; "
          f"noise a {noise.a:.4f} b {noise.b:.4f}; target ess {ess:.3f}")

    num_pred = frozenset(c for c in covs if _is_numeric(target_pool[c]))
    learned, prior_only = {}, {}
    for o in outs:
        feats = outcome_features(target_pool, o, covs, numeric_predictors=num_pred)
        learned[o] = predict_r2(x_co.get(o), ess, feats, prior, noise)
        prior_only[o] = predict_r2(None, ess, feats, prior, noise)
    return learned, prior_only, ess, sib_rew


def run_b5(pair, *, seeds, n, bootstrap_B):
    """Score B5_learned (EB posterior R^2) and B5_prior_only (prior-only R^2) through
    the B2 machinery via the r2_target seam, identically to B0-B4. The structure
    vehicle is B4's raked sibling pool ``sib_rew`` (source_pool), so B5 differs from
    B4_retrieval ONLY in where the per-outcome R^2 comes from. Writes CSV."""
    import nodonor_bracket as nb
    ds, cols, covs, outs = b4_columns(pair)
    target_pool, guarantee = nb.carve_pool(ds)
    ref = _load(load_schema(ds).real_data_path)
    types = nb.TYPES.get(ds, (1, 2, 3))

    learned, prior_only, ess, sib_rew = predict_target_r2(pair)
    configs = {"B5_learned": learned, "B5_prior_only": prior_only}

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(load_schema(ds).ssdatabench_sim_subdir,
                                      set(cols), types, Path(cfg_td))
        for name, r2_map in configs.items():
            recs = []
            for s in range(1, seeds + 1):
                sim = transfer_build_b2(sib_rew, target_pool, cols, covs, outs,
                                        n, s, r2_target=r2_map)
                recs.append(nb.score(sim, ds, ref, types, seed=1000 + s,
                                     bootstrap_B=bootstrap_B, config_dir=cfg_dir))
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "ess_ratio": round(ess, 4)}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"b5_{pair.id}.csv", index=False)
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
    df = run_b5(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'b5_{pair.id}.csv'}")
    print("REGIME: no-donor + learned. Target supplies marginals + X-margins + public"
          " outcome features only; conditional strength is an EB blend of retrieval"
          " and a cross-context prior.")


if __name__ == "__main__":
    main()
