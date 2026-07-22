#!/usr/bin/env python
"""Reproduce the no-donor / microdata bracket for a dataset. No LLM, no API key.

This is the replication path for `docs/report/2026-07-17-cross-dataset-corrections.md`.
It exists in the repo (not a session scratchpad) because the scratchpad copies of these
measurements were destroyed by a /tmp wipe on 2026-07-16, taking the reports'
reproducibility with them.

Four reference configurations, all scored against the paper's fixed benchmark sample:

  independence   the no-donor FLOOR -- every column drawn independently from the
                 disjoint pool's marginal. "We know the aggregates, nothing else."
  copula-old     shared-index copula with missingness scattered at RANDOM (the bug
                 fixed in 77c02e9). Kept so the correction stays auditable.
  copula-fixed   same, but each sampled respondent keeps their OWN missingness and we
                 only top up / trim to the pool's rate.
  rowresample    whole real rows from the disjoint pool = the MICRODATA CEILING.

Regimes are not comparable: `independence` is no-donor (marginals only), while
`copula-*` and `rowresample` use the pool's joint and are therefore microdata methods.
Never quote a copula/rowresample number against a no-donor one.

Usage:
    .venv/bin/python scripts/nodonor_bracket.py cps
    .venv/bin/python scripts/nodonor_bracket.py cfps --seeds 5 --n 5000
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from ssdataagent.data.schema import load_schema  # noqa: E402

CONFIG_DIR = REPO / "ssdatabench" / "evaluation" / "config"

# Datasets whose full source ships a unique row key -> a PROVABLY disjoint pool.
# cps has none: see `carve_pool` for the weaker guarantee we can offer there.
KEYED = {"cfps": "pid", "addhealth": "AID"}
# Same-year full sources not declared in config/datasets.yaml. Each is the FULL survey
# wave the benchmark's 1000-row reference was sampled FROM, so a row-disjoint carve yields
# a genuine held-out pool. gss uses the 2018 wave (the same-year source): it carries
# `wealth` and `mental_health`, the columns the older gss1994 transfer wave lacked, which
# was the standing blocker. acs has NO such source here -- `acs_clean.csv` is the same 1000
# rows as the reference, not a larger population, so acs stays blocked pending real ACS
# microdata (public via IPUMS).
FULL_SOURCE = {"cps": "real_data/cps/cps-asec1980.csv",
               "gss": "real_data/gss/gss2018.csv"}
# Cross-sectional datasets have no T4/T5 (their event_variables are commented out).
TYPES = {"cps": (1, 2, 3), "gss": (1, 2, 3), "acs": (1, 2, 3)}


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]


def _rowkey(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """A row-identity string. NaN-SAFE: `Series.str.cat` PROPAGATES NaN, so keying with
    it silently nulls every row that has a missing value -- which made an earlier carve
    delete all 92k incomplete cps rows and destroy the marginals (bogus T1=0.14). Map
    each column to str first so no NaN ever reaches the join."""
    s = df[cols[0]].map(str)
    for c in cols[1:]:
        s = s.str.cat(df[c].map(str), sep="|")
    return s


def carve_pool(ds: str) -> tuple[pd.DataFrame, str]:
    """(pool, guarantee) -- the benchmark's population minus the benchmark rows.

    Two guarantee levels, and the difference is not cosmetic:
      "person-disjoint"  the full source has a unique person key (cfps `pid`), so we can
                         prove the pool excludes the benchmark respondents.
      "row-disjoint"     no key exists (cps). We remove ONE occurrence of each benchmark
                         row, which excludes the exact rows scored against, but a
                         different person with identical recorded attributes remains
                         (cps: ~36k such rows). Weaker. Label it when reporting.
    """
    schema = load_schema(ds)
    ref = _drop_unnamed(pd.read_csv(schema.real_data_path, low_memory=False))
    src = schema.full_source_path or (REPO / FULL_SOURCE[ds] if ds in FULL_SOURCE else None)
    if src is None:
        raise SystemExit(f"{ds}: no full source; cannot build a disjoint pool")
    full = _drop_unnamed(pd.read_csv(src, low_memory=False))

    key = KEYED.get(ds)
    if key and key in full.columns and key in ref.columns:
        pool = full[~full[key].isin(set(ref[key]))].reset_index(drop=True)
        return pool, "person-disjoint"

    cols = [c for c in ref.columns if c in full.columns and c != "profile_id"]
    fk, bk = _rowkey(full, cols), _rowkey(ref, cols)
    need = bk.value_counts().to_dict()
    drop: list[int] = []
    for k, idx in fk.groupby(fk).groups.items():
        c = need.get(k, 0)
        if c:
            drop.extend(list(idx)[:c])
    return full.drop(index=drop).reset_index(drop=True), "row-disjoint"


def _is_numeric(s: pd.Series) -> bool:
    s = s.dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _latent(pool: pd.DataFrame, c: str, num: bool, glat: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    m = len(pool)
    if num:
        u = np.array(pd.to_numeric(pool[c], errors="coerce")
                     .rank(pct=True, method="first").to_numpy(dtype=float))
        nan = np.isnan(u)
        u[nan] = rng.random(int(nan.sum()))
        return u
    s = pool[c].astype(str)
    order = (pd.Series(glat, index=range(m)).groupby(s.to_numpy()).mean()
             .sort_values().index.tolist())
    pos = {v: i for i, v in enumerate(order)}
    b = np.array(s.map(pos).to_numpy(dtype=float), dtype=float)
    nn = np.isnan(b)
    b[nn] = rng.integers(0, max(1, len(order)), int(nn.sum()))
    return pd.Series(b + rng.random(m)).rank(pct=True, method="first").to_numpy()


def build(pool: pd.DataFrame, cols: list[str], n: int, seed: int, mode: str) -> pd.DataFrame:
    """mode: independence | copula-old | copula-fixed | rowresample"""
    if mode == "rowresample":
        return pool[cols].sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    m = len(pool)
    num = {c: _is_numeric(pool[c]) for c in cols}
    znum = {c: pd.to_numeric(pool[c], errors="coerce").rank(pct=True) for c in cols if num[c]}
    glat = (pd.DataFrame(znum).mean(axis=1).fillna(0.5).to_numpy() if znum
            else np.full(m, 0.5))
    base = rng.integers(0, m, n)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        s = pool[c]
        idx = rng.integers(0, m, n) if mode == "independence" else base
        u = np.clip(_latent(pool, c, num[c], glat, rng)[idx], 1e-6, 1 - 1e-6)
        if num[c]:
            vals = np.sort(pd.to_numeric(s, errors="coerce").dropna().to_numpy())
            em = vals[np.clip((u * len(vals)).astype(int), 0, len(vals) - 1)].astype(object)
        else:
            vc = s.dropna().astype(str).value_counts(normalize=True)
            cats, edges = vc.index.to_numpy(), np.cumsum(vc.to_numpy())
            em = cats[np.searchsorted(edges, u, side="right").clip(0, len(cats) - 1)].astype(object)
        miss = float(s.isna().mean())
        if miss > 0:
            want = int(round(miss * n))
            if mode == "copula-fixed":
                mask = s.isna().to_numpy()[idx].copy()
                have = int(mask.sum())
                if have > want:
                    mask[rng.choice(np.flatnonzero(mask), have - want, replace=False)] = False
                elif have < want:
                    free = np.flatnonzero(~mask)
                    mask[rng.choice(free, min(want - have, len(free)), replace=False)] = True
                em[mask] = np.nan
            elif want:
                em[rng.choice(n, want, replace=False)] = np.nan
        out[c] = em
    return pd.DataFrame(out)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["profile_id"] = range(len(out))  # the matched bootstrap needs overlapping ids
    return out


@contextlib.contextmanager
def _seeded_rng(seed: int | None):
    """Make the benchmark's bootstrap reproducible.

    Every type module does ``rng = np.random.default_rng()`` with NO seed, so two scoring
    runs of the SAME frame return different numbers (measured on cps: overall 0.498 /
    0.453 / 0.446 / 0.313 on identical input). Patching the factory for the duration of
    the call makes a run repeatable; it does not reduce the noise, it only pins it.
    """
    if seed is None:
        yield
        return
    orig = np.random.default_rng
    counter = itertools.count()

    def factory(s=None, *a, **k):
        return orig(s, *a, **k) if s is not None else orig(seed * 100_003 + next(counter))

    np.random.default_rng = factory
    try:
        yield
    finally:
        np.random.default_rng = orig


def _cfg_with_B(cfg: Path, td: Path, t: int, bootstrap_B: int | None) -> str:
    """Return a config path, optionally overriding ``bootstrap_B``.

    ``insignificant_rate`` is ``not_sig / B`` -- the FRACTION of B bootstrap iterations
    in which the test came out insignificant. B is a Monte Carlo replicate count, not
    part of the estimand, so raising it estimates the SAME quantity more precisely. The
    shipped configs set B=1 for T1 (one 500-row draw decides each variable: sigma~0.10,
    quantized to k/n_vars) and B=10 elsewhere; the library's own default is B=1000, so
    the small values look like a speed hack rather than a definition.

    Published numbers were produced at the shipped B and therefore carry that noise. Any
    run that raises B must say so -- it is a better estimate of the same statistic, but
    its error bars are not comparable to a published run's.
    """
    if bootstrap_B is None:
        return str(cfg)
    import yaml
    spec = yaml.safe_load(cfg.read_text())
    spec["bootstrap_B"] = int(bootstrap_B)
    out = td / f"type{t}_B{bootstrap_B}.yaml"
    out.write_text(yaml.safe_dump(spec))
    return str(out)


def score(sim: pd.DataFrame, ds: str, ref: pd.DataFrame, types: tuple[int, ...],
          *, seed: int | None = None, bootstrap_B: int | None = None,
          config_dir: Path | None = None) -> dict:
    """Score ``sim`` against ``ref``.

    ``seed`` pins the benchmark's unseeded bootstrap so a run is reproducible.
    ``bootstrap_B`` overrides the config's Monte Carlo replicate count (see
    ``_cfg_with_B``); leave it None to reproduce a published-comparable run.
    ``config_dir`` overrides the type-config root (default ``CONFIG_DIR``); the
    transfer runner passes a restricted-to-crosswalk config here so a sim that can only
    carry the source's variables is scored on exactly those. Omit for the stock configs.
    """
    schema = load_schema(ds)
    base_cfg = config_dir or CONFIG_DIR
    from ssdatabench.evaluation.code_by_type import type1, type2, type3, type4, type5
    runners = {1: type1.run_type1_eval, 2: type2.run_type2_eval, 3: type3.run_type3_eval,
               4: type4.run_type4_eval, 5: type5.run_type5_eval}
    out: dict = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        real_csv, sim_csv = td / "real.csv", td / "sim.csv"
        _prep(ref).to_csv(real_csv, index=False)
        _prep(sim).to_csv(sim_csv, index=False)
        for t in types:
            cfg = base_cfg / schema.ssdatabench_sim_subdir / f"type{t}.yaml"
            if not cfg.exists():
                continue
            odir, buf = td / f"out{t}", io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
                        _seeded_rng(None if seed is None else seed + t):
                    runners[t](_cfg_with_B(cfg, td, t, bootstrap_B),
                               real_csv=str(real_csv), sim_csv=str(sim_csv),
                               out_dir=str(odir), verbose=False)
            except Exception as e:
                out[f"T{t}"] = None
                out[f"T{t}_error"] = f"{type(e).__name__}: {e}"
                continue
            out[f"T{t}"] = _rate(odir, t)
    vals = [v for k, v in out.items() if k.startswith("T") and "_" not in k and v is not None]
    out["overall"] = float(np.mean(vals)) if vals else None
    return out


def _rate(odir: Path, t: int) -> float | None:
    hits = sorted(odir.glob(f"summary_type{t}*.csv"))
    hits = [h for h in hits if "strength" not in h.stem] or hits
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    if "insignificant_rate" not in df.columns:
        return None
    k = next((c for c in ("variable", "response", "combo", "var1") if c in df.columns), None)
    if k is not None:  # drop the trailing avg rows
        df = df[~df[k].astype(str).str.lower().str.startswith("avg")]
    v = pd.to_numeric(df["insignificant_rate"], errors="coerce")
    return float(v.fillna(0.0).mean()) if len(v) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="cfps | cps | addhealth")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=5000, help="simulated rows per config")
    ap.add_argument("--bootstrap-B", type=int, default=200,
                    help="scorer Monte Carlo replicates. Shipped configs use B=1 (T1) / "
                         "B=10, so a single score swings ~0.10 on T1. B is a replicate "
                         "count, not the estimand -- raising it measures the SAME "
                         "statistic more precisely. Pass 0 for the shipped config "
                         "(published-comparable noise).")
    a = ap.parse_args()

    ref = _drop_unnamed(pd.read_csv(load_schema(a.dataset).real_data_path, low_memory=False))
    pool, guarantee = carve_pool(a.dataset)
    schema = load_schema(a.dataset)
    modeled = [c for c in ref.columns
               if (c in schema.domains or c in schema.background_variables)
               or a.dataset in FULL_SOURCE]
    cols = [c for c in modeled if c in pool.columns and c != "profile_id"]
    types = TYPES.get(a.dataset, (1, 2, 3, 4, 5))

    print(f"{a.dataset}: ref={len(ref)} pool={len(pool)} [{guarantee}] "
          f"cols={len(cols)} seeds={a.seeds} n={a.n} "
          f"bootstrap_B={a.bootstrap_B or 'shipped'}")
    if guarantee == "row-disjoint":
        print("  NOTE: no person key -- identical-profile rows remain in the pool; "
              "this pool is weaker than a person-disjoint one.")
    tcols = [f"T{t}" for t in types]
    print("\n" + f"{'config':<16}" + "".join(f"{c:>8}" for c in tcols) + f"{'overall':>9}")
    print("-" * (16 + 8 * len(tcols) + 9))
    for mode in ("independence", "copula-old", "copula-fixed", "rowresample"):
        d = pd.DataFrame([score(build(pool, cols, a.n, s, mode), a.dataset, ref, types,
                                seed=1000 + s, bootstrap_B=a.bootstrap_B or None)
                          for s in range(1, a.seeds + 1)])
        cells = "".join(f"{d[c].mean():>8.3f}" if d[c].notna().any() else f"{'--':>8}"
                        for c in tcols)
        print(f"{mode:<16}{cells}{d['overall'].mean():>9.3f}")
        print(f"{'  (std)':<16}"
              + "".join(f"{d[c].std():>8.3f}" if d[c].notna().any() else f"{'--':>8}"
                        for c in tcols)
              + f"{d['overall'].std():>9.3f}")
    print("\nregimes: independence = no-donor (marginals only); copula-*/rowresample "
          "use the pool's joint = microdata. Do not compare across regimes.")


if __name__ == "__main__":
    main()
