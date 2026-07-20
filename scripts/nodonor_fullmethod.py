#!/usr/bin/env python
"""The full no-donor method (LLM conditional generation + conditional-variance repair).

This is the durable replication path for the no-donor HEADLINE, as opposed to
`nodonor_bracket.py`, which is the LLM-free reference bracket. It lives in the repo
because the scratchpad copies of the cps/cfps generation scripts were destroyed by a
/tmp wipe on 2026-07-16 -- twice now, ephemeral scripts have cost us reproducibility.

Three stages, each cached under results/nodonor_cache/ (gitignored but durable), so a
re-score costs no API calls:

  generate  Sample demographic seeds INDEPENDENTLY from the disjoint pool's marginals
            (pure aggregate -- no joint, no microdata), then have the LLM complete each
            person's downstream traits coherently. Cached: <ds>_cond_raw.csv
  elicit    Ask the LLM, reasoning as a demographer from OUTCOME NAMES ONLY, for each
            outcome's covariate-R^2. Non-circular: it never reads the pool or the test.
            Cached: <ds>_elicit.json
  repair    Per-outcome conditional-variance repair (alpha = sqrt(target/own)), every
            column mapped onto the pool's marginal, then score against the benchmark.

REGIME: no-donor. The pool supplies MARGINALS ONLY (and, at the labeled ceiling, a
low-order R^2 aggregate). Never quote these numbers against a copula/rowresample
figure from nodonor_bracket.py -- those use the pool's joint and are microdata methods.

Usage:
    export OPENROUTER_API_KEY=...            # only needed the first time (stages cached)
    .venv/bin/python scripts/nodonor_fullmethod.py cps
    .venv/bin/python scripts/nodonor_fullmethod.py cps --seeds 5 --n 5000
    .venv/bin/python scripts/nodonor_fullmethod.py cps --regenerate   # bust the cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import ssdataagent.data.conditional_variance as cv  # noqa: E402
from nodonor_bracket import _drop_unnamed, carve_pool, score  # noqa: E402
from ssdataagent.data.schema import load_schema  # noqa: E402

CACHE = REPO / "results" / "nodonor_cache"
MODEL = "anthropic/claude-sonnet-4.5"


class Spec:
    """Per-dataset wiring. `predictors` and `numeric_predictors` MUST mirror the
    dataset's type3.yaml -- the repair calibrates R^2 against the statistic T3 scores,
    so a mismatch here silently mis-sizes every alpha."""

    def __init__(self, seeds, derived, predictors, numeric_predictors, log_vars,
                 types, population, rules, glosses):
        self.seeds = seeds                          # drawn independently from marginals
        self.derived = derived                      # exact identities, computed not drawn
        self.predictors = predictors                # held coherent through the repair
        self.numeric_predictors = numeric_predictors
        self.log_vars = log_vars
        self.types = types
        self.population = population                # prose name for the prompt
        self.rules = rules                          # coherence rules for the prompt
        self.glosses = glosses                      # outcome glosses for elicitation


SPECS = {
    # cps is a 1980 cross-section of the WHOLE population, children included (age 0-88).
    # birth_year is not drawn: age + birth_year == 1980 for every row, an identity of the
    # survey design, so drawing both independently would manufacture impossible people.
    "cps": Spec(
        seeds=["age", "gender", "race"],
        derived={"birth_year": lambda df: 1980.0 - pd.to_numeric(df["age"])},
        predictors=["age", "gender", "race", "education"],
        numeric_predictors=frozenset({"age"}),
        log_vars=frozenset({"income"}),
        types=(1, 2, 3),
        population="the 1980 US Current Population Survey (March ASEC)",
        # CRITICAL semantic correction (see docs/report/2026-07-20-cps-fertility-semantics.md):
        # `child_number` / `age_first_childbirth` are NOT lifetime fertility. This is the CPS
        # household roster -- OWN children UNDER 18 currently LIVING WITH the respondent. The
        # data proves it: mean child_number 0.66 (lifetime 1980 fertility was ~2.5-3), and 60+
        # respondents average 0.16 because grown children have moved out. Describing it as
        # "children ever born" made the LLM generate lifetime fertility -- monotonically rising
        # with age -- inverting the true age relationship (real Spearman(age, first-birth) =
        # +0.62; the LLM produced -0.26) and tanking T3. This is a DEFINITION correction from
        # public metadata, NOT the pool's conditional distribution, so T3 stays non-circular.
        rules="""- This is the WHOLE resident population, children included. A 7-year-old is
  'Less than high school', not in the labor force, no occupation, no income.
- Education is bounded by age: nobody under 18 is 'College and above'.
- FERTILITY IS HOUSEHOLD-RESIDENT, NOT LIFETIME. child_number counts this person's OWN
  children UNDER 18 who currently LIVE WITH THEM; age_first_childbirth is the age at which
  their FIRST still-resident child was born. Adult children have moved out and no longer
  count. Consequences you must honour:
    * Most people have child_number 0 and 'No Child' -- including the elderly, whose
      children grew up and left. A 65-year-old almost always shows 0 resident children even
      though they were a parent. Do NOT let child_number climb with age; it PEAKS around
      ages 30-45 and falls back to ~0 by the late 50s.
    * Among people who DO have a resident child, the older they are, the LATER that child was
      born (their early children have already left home) -- so age_first_childbirth RISES
      with the respondent's age, from ~22 in their 30s to ~33 in their 60s. It is always
      >= 12 and < the person's own age.
    * child_number 0 <=> 'No Child'; a nonzero child_number needs a real first-birth age.
- income is annual 1980 dollars (a few thousand is typical); null for someone with no
  earnings such as a child. occupation is null when not in the labor force.
- Marital status, education and income should cohere as one real person.""",
        glosses={
            "marital_status": "marital status (Married / Single / Separated-Divorced-Widowed)",
            "child_number": "number of own children UNDER 18 currently living in the "
                            "household (NOT lifetime births; adult children have moved out)",
            "age_first_childbirth": "age when the first still-resident child was born "
                                    "('No Child' if none live with the respondent)",
            "education": "educational attainment (3 levels)",
            "laborforce": "in the labor force or not",
            "occupation": "broad occupation category",
            "income": "total personal income last year, 1980 dollars",
            "poverty_status": "above or below the poverty line",
        },
    ),
}


def modeled_columns(ds: str, ref: pd.DataFrame, pool: pd.DataFrame) -> list[str]:
    schema = load_schema(ds)
    cols = [c for c in ref.columns
            if (c in schema.domains or c in schema.background_variables)]
    return [c for c in cols if c in pool.columns and c != "profile_id"]


def _is_numeric(pool: pd.DataFrame, c: str) -> bool:
    s = pool[c].dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def marginal_blob(pool: pd.DataFrame, cols: list[str]) -> str:
    """The ONLY pool information that reaches the LLM: per-column marginals. This is the
    no-donor regime's supplied aggregate -- the same thing that locks T1."""
    out = {}
    for c in cols:
        s = pool[c]
        if _is_numeric(pool, c):
            v = pd.to_numeric(s, errors="coerce").dropna()
            out[c] = {"num": True, "miss": round(float(s.isna().mean()), 2),
                      "mean": round(float(v.mean()), 1), "sd": round(float(v.std()), 1)}
        else:
            vc = s.dropna().astype(str).value_counts(normalize=True).head(8)
            out[c] = {"miss": round(float(s.isna().mean()), 2),
                      "vals": {k: round(float(x), 2) for k, x in vc.items()}}
    return json.dumps(out, ensure_ascii=False)


def sample_seeds(pool: pd.DataFrame, spec: Spec, n: int, seed: int) -> pd.DataFrame:
    """Each seed column drawn INDEPENDENTLY from its own marginal -- no cross-column
    information, so nothing about the pool's joint leaks in. Derived columns are exact
    identities (cps: birth_year = 1980 - age), which is arithmetic, not microdata."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in spec.seeds:
        vals = pool[c].dropna().to_numpy()
        out[c] = vals[rng.integers(0, len(vals), n)]
    df = pd.DataFrame(out)
    for c, fn in spec.derived.items():
        df[c] = fn(df)
    return df


def _jsonable(rec: dict) -> dict:
    return {k: (int(v) if isinstance(v, np.integer) else
                float(v) if isinstance(v, np.floating) else v) for k, v in rec.items()}


def complete_batch(client, seeds_df, downstream, blob, spec, *, retries=5) -> list[dict]:
    recs = [_jsonable(r) for r in seeds_df.to_dict(orient="records")]
    prompt = f"""You are completing survey respondents from {spec.population}. Below are
{len(recs)} people, each given by their demographics. For EACH, infer the rest of their
profile as a real, coherent person.

{spec.rules}

For each input person output one JSON object with EXACTLY these keys: {downstream}.
Use null for a value that genuinely does not apply. Keep values plausible against the
population marginals below.

POPULATION MARGINALS (downstream vars): {blob}

INPUT PEOPLE (complete each, same order): {json.dumps(recs, ensure_ascii=False)}

Output {len(recs)} JSONL lines, one completion per input person, in order. No prose."""
    # The user's network drops intermittently; a single ConnectTimeout used to kill the
    # whole run and discard every completed batch. Retry each batch with backoff so a
    # transient blip costs one batch's wait, not the run. Backoff is fixed (2**k), not
    # jittered: Math.random() is unavailable here and determinism aids debugging.
    last = None
    for k in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=8000, temperature=0.8, timeout=90,
                messages=[{"role": "user", "content": prompt}])
            rows = []
            for line in (r.choices[0].message.content or "").splitlines():
                line = line.strip().strip("`")
                if line.startswith("{") and line.endswith("}"):
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return rows
        except Exception as e:  # noqa: BLE001 -- network/timeout/rate-limit all retriable
            last = e
            if k < retries - 1:
                print(f"    batch retry {k+1}/{retries-1} after {type(e).__name__}",
                      flush=True)
                time.sleep(2 ** k)
    raise RuntimeError(f"batch failed after {retries} attempts: {last}")


def _load_raw(path: Path, cols: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    for c in cols:
        if c not in raw.columns:
            raw[c] = np.nan
    return raw[cols]


def generate(ds, pool, cols, spec, n_people, batch, regenerate) -> pd.DataFrame:
    """Always returns the frame as READ BACK FROM THE CACHE, never the in-memory one
    just built. The CSV round-trip is lossy -- sentinel strings sit beside numbers in
    the same column ('No Child' / 22) and None becomes NaN -- so the in-memory frame and
    the reloaded frame score differently (measured on cps: raw T3 0.047 vs 0.007).
    Returning whichever one happened to be at hand would make the headline depend on
    whether the cache was warm. Round-trip on both paths so a first run and a re-score
    see byte-identical input."""
    out_path = CACHE / f"{ds}_cond_raw.csv"
    if out_path.exists() and not regenerate:
        print(f"generate: cached -> {out_path}")
        return _load_raw(out_path, cols)

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(REPO / ".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])

    downstream = [c for c in cols if c not in spec.seeds and c not in spec.derived]
    blob = marginal_blob(pool, downstream)
    seeds = sample_seeds(pool, spec, n_people, seed=42)
    t0, done = time.time(), []
    for i in range(0, len(seeds), batch):
        chunk = seeds.iloc[i:i + batch].reset_index(drop=True)
        got = complete_batch(client, chunk, downstream, blob, spec)
        for j in range(min(len(chunk), len(got))):
            row = dict(chunk.iloc[j])
            row.update({c: got[j].get(c, np.nan) for c in downstream})
            done.append(row)
        print(f"  seeds {i}-{i+batch}: +{min(len(chunk), len(got))} -> {len(done)}",
              flush=True)
    raw = pd.DataFrame(done)
    for c in cols:
        if c not in raw.columns:
            raw[c] = np.nan
    raw = raw[cols]
    CACHE.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_path, index=False)
    print(f"generate: {len(raw)} people in {time.time()-t0:.0f}s -> {out_path}")
    return _load_raw(out_path, cols)   # round-trip: see the docstring


def elicit(ds, cols, spec, regenerate) -> dict:
    path = CACHE / f"{ds}_elicit.json"
    if path.exists() and not regenerate:
        print(f"elicit: cached -> {path}")
    else:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(REPO / ".env")
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
        CACHE.mkdir(parents=True, exist_ok=True)
        responses = [c for c in cols if c not in spec.predictors and c in spec.glosses]
        cv.elicit_r2_targets(responses, spec.predictors, client, MODEL,
                             glosses=spec.glosses, cache_path=str(path))
    return {k: float(v) for k, v in cv._extract_last_json(path.read_text()).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(SPECS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=5000, help="simulated rows per seed")
    ap.add_argument("--people", type=int, default=480, help="LLM-generated respondents")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--regenerate", action="store_true", help="ignore cached stages")
    ap.add_argument("--bootstrap-B", type=int, default=200,
                    help="scorer Monte Carlo replicates. The shipped configs use B=1 "
                         "(T1) / B=10, which makes a single score swing by ~0.10 on T1. "
                         "B is a replicate count, not the estimand, so raising it "
                         "measures the SAME statistic more precisely. Pass 0 to use the "
                         "shipped config (published-comparable noise).")
    a = ap.parse_args()

    ds, spec = a.dataset, SPECS[a.dataset]
    ref = _drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    pool, guarantee = carve_pool(ds)
    cols = modeled_columns(ds, ref, pool)
    print(f"{ds}: ref={len(ref)} pool={len(pool)} [{guarantee}] cols={len(cols)}")
    if guarantee == "row-disjoint":
        print("  NOTE: no person key -- identical-profile rows remain in the pool.")

    raw = generate(ds, pool, cols, spec, a.people, a.batch, a.regenerate)
    elicited = elicit(ds, cols, spec, a.regenerate)

    targets = {c: elicited[c] for c in cols
               if c not in spec.predictors and elicited.get(c) is not None}
    alphas = cv.variance_repair_alphas(raw, spec.predictors, targets,
                                       numeric_predictors=spec.numeric_predictors,
                                       log_vars=spec.log_vars)
    print(f"\n{'outcome':<24}{'own R2':>9}{'target':>9}{'alpha':>8}")
    print("-" * 50)
    for c in sorted(alphas, key=lambda c: -alphas[c]):
        own = cv.covariate_r2(raw, c, spec.predictors,
                              numeric_predictors=spec.numeric_predictors,
                              log_vars=spec.log_vars)
        print(f"{c:<24}{(f'{own:.3f}' if own is not None else '--'):>9}"
              f"{targets[c]:>9.3f}{alphas[c]:>8.3f}")

    nr = raw.isna().mean()
    print(f"\nLLM generation NaN: mean rate {nr.mean():.3f} over {len(nr)} cols; "
          f"pool {pool[cols].isna().mean().mean():.3f} "
          "(gap bounds how much person-linked missingness can bite)")

    # The honesty ladder's top rung, always LABELED: targets taken from the disjoint
    # pool's covariate-R^2 -- a low-order supplied aggregate, on the same footing as
    # handing over the marginals. NOT the honest headline; it isolates how much of the
    # gap is elicitation error rather than a limit of the repair itself.
    pool_targets = {c: cv.covariate_r2(pool, c, spec.predictors,
                                       numeric_predictors=spec.numeric_predictors,
                                       log_vars=spec.log_vars)
                    for c in targets}
    pool_alphas = cv.variance_repair_alphas(
        raw, spec.predictors, {c: t for c, t in pool_targets.items() if t is not None},
        numeric_predictors=spec.numeric_predictors, log_vars=spec.log_vars)

    tcols = [f"T{t}" for t in spec.types]
    print(f"\n{'config':<28}" + "".join(f"{c:>8}" for c in tcols) + f"{'overall':>9}")
    print("-" * (28 + 8 * len(tcols) + 9))
    for name, alpha, dflt in (
        ("raw (no repair)", {c: 1.0 for c in alphas}, 1.0),
        ("repaired, elicited [honest]", alphas, 0.5),
        ("repaired, pool R2 [CEILING]", pool_alphas, 0.5),
    ):
        rows = []
        for s in range(1, a.seeds + 1):
            sim = cv.sample_variance_repaired(raw, pool, cols, spec.predictors, a.n,
                                              np.random.default_rng(s), alpha=alpha,
                                              default_alpha=dflt)
            rows.append(score(sim, ds, ref, spec.types, seed=1000 + s,
                              bootstrap_B=a.bootstrap_B or None))
        d = pd.DataFrame(rows)
        print(f"{name:<28}"
              + "".join(f"{d[c].mean():>8.3f}" if d[c].notna().any() else f"{'--':>8}"
                        for c in tcols)
              + f"{d['overall'].mean():>9.3f}")
        print(f"{'  (std)':<28}"
              + "".join(f"{d[c].std():>8.3f}" if d[c].notna().any() else f"{'--':>8}"
                        for c in tcols)
              + f"{d['overall'].std():>9.3f}")
    print("\nREGIME: no-donor (pool supplies marginals only). Do NOT compare against "
          "copula/rowresample numbers from nodonor_bracket.py -- those are microdata.")


if __name__ == "__main__":
    main()
