# Porting beyond cfps: three measurement bugs, and what survived them

**Date:** 2026-07-17
**Scope:** cfps, cps, addhealth. No test data used anywhere.
**Replication:** `scripts/nodonor_bracket.py` (no LLM, no API key).
**Code landed:** `77c02e9` (person-linked missingness in `conditional_variance.py`).

> **Precision caveat, added later the same day.** Every number here was measured with the
> benchmark's **unseeded** bootstrap at its shipped `bootstrap_B` (1 for T1, 10 for the
> rest). That scorer carries σ≈0.043 on overall and σ≈0.100 on T1 *on fixed input* — so
> nothing below a ~0.054 overall gap is resolvable at 5 seeds, and no run here is
> reproducible. The **large** findings in this report (the three bugs, the bracket's
> shape, mean-collapse, addhealth T4) clear that bar comfortably. Two fine comparisons do
> not and are struck through below. Full characterisation, corrected B=200 values, and
> the seeding fix: `docs/report/2026-07-17-scorer-noise-and-cps.md`.

## Summary

We took the cfps no-donor method to other datasets to see whether it generalized. It
did — but the trip was mostly spent discovering that our *measurement* did not. Three
bugs, all ours, made cps look far worse than cfps. Corrected, **the datasets behave the
same**: on T1–T3 the microdata ceiling is **0.827 (cps)** vs **0.791 (cfps)**. The one
durable modelling finding is that **mean-collapse is universal** and variance repair
fixes it everywhere.

This report exists to record the corrected numbers, the retractions, and a replication
path that survives a `/tmp` wipe — because the previous one did not (see *Provenance*).

## Corrected results

5000 simulated rows, scored against the paper's fixed 1000-row benchmark sample; 5 seeds
unless a table says otherwise. `independence` is the **no-donor floor** (marginals only);
`copula-*` and `rowresample` use the disjoint pool's joint and are therefore
**microdata** methods. *Regimes are not comparable* — see *Retractions*.

### cps-1980 (pool = same-year `cps-asec1980.csv`, row-disjoint, 180,488 rows)

| config | T1 | T2 | T3 | overall | regime |
|---|---|---|---|---|---|
| independence (floor) | 0.914 | 0.351 | 0.033 | 0.433 | no-donor |
| copula-old (buggy) | 0.829 | 0.560 | 0.280 | 0.556 | microdata |
| copula-fixed | 0.886 | 0.634 | 0.587 | 0.702 | microdata |
| **rowresample (ceiling)** | 0.829 | 0.858 | 0.793 | **0.827** | microdata |
| published, best single system (GPT-4o) | — | — | — | ~0.40 | per-person |
| published, per-type best of 15 LLMs | ~0.10 | ~0.71 | ~0.57 | (0.46)† | per-person |

cps is cross-sectional: its T4/T5 `event_variables` are commented out in the eval
config, so only T1–T3 exist — its overall is a genuine T1–T3 mean.

† The two published rows are different things and must not be merged. The per-type
figures are the *best across 15 different LLMs, a different model per type*; 0.46 is
their mean — an **envelope no single system achieves**, not a competitor. The fair
single-system comparator is GPT-4o's reported overall, ~0.40.

So, precisely: the trivial **independence floor (0.433) beats the best single published
system (~0.40)** — entirely on T1, which published LLMs fail (~0.10) — but it does *not*
beat the 0.46 cross-model envelope, and on the joint benchmarks (T2 0.351, T3 0.033) it
is far below every published LLM. Marginals are free; the joint is not.

> **Retracted 2026-07-17 (same day).** The floor-beats-published claim is **withdrawn**.
> Every number in this table was measured at the scorer's shipped `bootstrap_B` (1 for
> T1, 10 elsewhere) with an **unseeded** bootstrap, which carries σ≈0.043 on overall and
> σ≈0.100 on T1 — so a 5-seed mean cannot resolve an overall gap below ~0.054. The
> floor's 0.433 vs published ~0.40 is a gap of 0.033: **unresolvable**. Re-measured at
> B=200 the floor is **0.407** — a dead heat. The *shape* of this table survives (adjacent
> rungs differ by 0.10–0.18, far above the floor); the fine comparisons do not. Corrected
> values and the full noise characterisation:
> `docs/report/2026-07-17-scorer-noise-and-cps.md`.

### cfps (pool = pid-disjoint, 57,474 rows)

cfps is longitudinal, so it *does* have T4/T5 — but the runs below scored **T1–T3 only**.
The `mean(T1..T3)` column is therefore **not** comparable to the cps table's overall, nor
to any published 5-type figure. It is labelled, not renamed, to keep that explicit.

| config | T1 | T2 | T3 | mean(T1..T3) | regime |
|---|---|---|---|---|---|
| copula-old (buggy) | 0.789 | 0.627 | 0.360 | 0.592 | microdata |
| copula-fixed | 0.684 | 0.727 | 0.622 | 0.678 | microdata |
| **rowresample (ceiling)** | 0.832 | 0.812 | 0.730 | **0.791** | microdata |
| our no-donor method, shipped | 0.853 | 0.630 | 0.462 | 0.648 | **no-donor** |
| our no-donor method, + fix | 0.842 | 0.617 | 0.468 | 0.642 | **no-donor** |
| published PNAS | 0.14 | 0.62 | 0.43 | 0.40 | per-person |

The published row is the paper's cfps per-type (T1 0.14, T2 0.62, T3 0.43, T4 0.05,
T5 0.75); its **5-type overall is 0.30**, and the 0.40 above is its T1–T3 mean —
recomputed so the column compares like with like. Quoting our T1–T3 mean against the
paper's 0.30 would flatter us.

The two no-donor rows are a fresh 480-person generation + fresh elicitation (the
originals were destroyed); they reproduce the previously reported 0.874 / 0.649 / 0.446
within seed noise, so the earlier headline stands.

**All five types** (5 seeds, n=5000, ref=1000, pool=57474 person-disjoint;
reproduce with `.venv/bin/python scripts/nodonor_bracket.py cfps --seeds 5`):

| config | T1 | T2 | T3 | T4 | T5 | overall |
|---|---|---|---|---|---|---|
| independence (floor) | 0.863 | 0.533 | 0.302 | 0.000 | 0.762 | **0.492** ±.015 |
| copula-old (buggy) | 0.789 | 0.628 | 0.346 | 0.000 | 0.802 | 0.513 ±.018 |
| copula-fixed | 0.737 | 0.730 | 0.668 | 0.000 | 0.831 | 0.593 ±.008 |
| **rowresample (ceiling)** | 0.884 | 0.815 | 0.724 | **0.780** ±.130 | 0.839 | **0.808** ±.030 |
| published PNAS | 0.14 | 0.62 | 0.43 | 0.05 | 0.75 | 0.30 |

Two things worth keeping. The no-donor floor lands at **0.492 ± 0.015**, consistent
with the 0.477 recorded when it was first measured — an independent check on the new
script. And **cfps T4 is genuinely winnable**: real microdata resampling scores
**0.780 ± 0.130**, against addhealth's 0.000 ceiling. T4-winnability is a property of
the dataset, not of the method.

> The 2-seed version of this table previously reported T4 = **0.950** and overall
> **0.831**. At 5 seeds T4 is **0.780 ± 0.130** — the 0.950 was the top of a wide
> seed distribution (T4's per-seed σ ≈ 0.13 is a known property of this benchmark),
> not a stable figure. The winnable-vs-unwinnable contrast against addhealth is
> unaffected; the magnitude was overstated.

## The three bugs

All three were in measurement scaffolding, not in the method or the data.

**1. Wrong pool.** cps was run against the `cps1970` transfer wave — a decade stale,
with `race` recoded (White/Other/Black vs Non-Black-Non-Hispanic/Hispanic/Black,
intersection 1). The **same-year full source** `real_data/cps/cps-asec1980.csv`
(181,488 rows) was present all along and contains all 1000 benchmark rows.

**2. Copula missingness** (fixed in `77c02e9`; it was in *shipped* code,
`conditional_variance.py`). Two halves: NaN rows received a random latent, and
missingness was re-scattered at random, independent of the row. Missingness is
structure — the respondent with no children is the one whose age-at-first-child is
blank — so scattering it flattens the covariate structure of every missing-heavy
column. Measured: `age_first_childbirth` covariate-R² **0.51 → 0.03**. Worth
**+0.31 on cps T3** and **+0.26 on cfps T3** wherever the source carries real
missingness. The regression test pins it: a missingness pattern that is 100%
predictable from education used to emerge 51%/51% — pure noise.

**3. NaN-propagating row key.** `Series.str.cat` propagates NaN, so any row with a
missing value got a NaN key. Effects: only 508/1000 benchmark rows were keyed;
`isin` matched NaN-to-NaN; and a carve that removed "every matching row" deleted all
92k **incomplete** rows, biasing the pool toward complete records and producing a
bogus T1 = 0.14. Fix: `.map(str)` per column before joining.

## The anomaly that exposed them

cps T3 was *higher* with the stale 1970 pool (0.453) than the correct same-year pool
(0.300). Same-year real data carrying a worse joint is backwards, and chasing it paid:

**T3 is an R²-matching test, not coefficient-matching.**
`ssdatabench/evaluation/code_by_type/type3.py:177-188` computes `R2_real`, `R2_sim`,
`se_i = sqrt(4*R²*(1-R²)²/n)`, `z = (R2_real - R2_sim)/sqrt(se1²+se2²)`, and passes when
`p > alpha`. Coefficients are never compared — the β log is commented out. At the n=500
bootstrap the pass window is roughly `|ΔR²| < ~0.09`.

That explains the anomaly as **two errors cancelling**: the 1970 pool's associations are
inflated (income R² 0.274 vs real 0.224; age_first_childbirth 0.519 vs 0.420), and the
buggy copula deflates R², so the worse pool landed inside the pass window by accident.
Measured, from the correct same-year pool:

| outcome | REAL | pool80 | sim (buggy copula) |
|---|---|---|---|
| income | 0.224 | 0.193 | 0.086 |
| age_first_childbirth | 0.420 | 0.509 | **0.026** |
| child_number | 0.104 | 0.089 | 0.079 |

The pool is right (Δ vs real all inside ±0.09); the copula destroyed it.

It also **retroactively validates the variance-repair design**: targeting R² was exactly
the right instrument, because R² is literally what T3 scores.

## Retractions

- **"transfer beats the LLM method on cps" — withdrawn.** That compared a *microdata*
  method (copula over the pool's joint) against a *no-donor* one (marginals only).
  Different information regimes; the comparison was meaningless.
- **"the cfps method's edge is dataset-dependent" — withdrawn.** It rested on the three
  bugs above. Corrected, cps ≈ cfps.
- **The cfps "integrity check" (method 0.649/0.446 beats transfer 0.501) — void.**
  That transfer was the buggy copula.
- **The person-linked-missingness fix does *not* improve our no-donor method**
  (0.648 → 0.642, inside noise). Our LLM generation emits a mean NaN rate of **0.022**
  against the pool's **0.274**, so ~90% of missingness is still random top-up — there
  was no person-level structure to preserve. The fix is correct and pays in the
  microdata regime; it was not our bottleneck.

## What survived

- **Mean-collapse is universal.** Raw LLM generation inflates covariate-R² on every
  dataset (cps: `child_number` 0.10 → 0.74, `income` 0.22 → 0.43; raw T3 = 0.04).
  Per-outcome variance repair fixes it everywhere (cps T3 0.04 → 0.31).
- **addhealth T4 is unwinnable**, and this one is not a bug: resampling the *real
  microdata* scores T4 = **0.000 ± 0.000**. The permutation tolerance is so tight that
  truth itself fails (published addhealth T4 range: 0.00–0.02). The event-order module
  correctly did not help there, and hurt T5 (0.746 → 0.526). **Do not apply module A to
  addhealth.** T4-winnability is a dataset property: cfps's real-microdata ceiling is
  **0.780 ± 0.130** (5 seeds).
- ~~**Every dataset beats the best single published system on overall**~~ — **withdrawn
  2026-07-17.** At B=200 the cps floor is 0.407 against published ~0.40: a dead heat, not
  a win, and the original 0.433 was scorer noise. What survives is the *direction*: our
  cheap baselines win on T1 (which published LLMs fail at ~0.10) and lose on the joint
  benchmarks (cps floor T2 0.353 / T3 0.004 vs their ~0.71 / ~0.57). The aggregate
  "win" was always a T1 win; it is now not reliably a win at all.
  See `docs/report/2026-07-17-scorer-noise-and-cps.md`.

## Honest limits

- **cps has no person key.** A correct carve removes the exact benchmark rows but leaves
  ~36,058 *profile-identical* rows (different people, same 12 recorded attributes). This
  is weaker than cfps's `pid` carve; `scripts/nodonor_bracket.py` labels it
  `row-disjoint` vs `person-disjoint`. cps microdata numbers carry that caveat.
- **gss is blocked in the no-donor regime.** Its evals need `wealth` (T1/T2) and
  `mental_health` (T3); the `gss1994` transfer wave lacks them. The same-year
  `gss2018.csv` has them but only 2,348 rows.
- **acs** has no disjoint source registered at all.
- **Open:** the cps *no-donor LLM* result (0.413) still needs re-running against the
  1980 pool — its marginals came from the stale 1970 wave.

## Next opportunity

**Model `P(missing | covariates)` in the no-donor regime.** ~27% of every missing-heavy
column is currently structurally random: the pool's missing *rate* is a supplied
aggregate we reproduce, but the LLM supplies almost no missingness of its own (0.022),
so we scatter it. That is a real mechanism — elicit or derive the missingness model —
not plumbing, and it is now the clearest headroom.

## Replication

```bash
# The bracket for any dataset (no LLM, no API key). Defaults: 5 seeds, n=5000.
.venv/bin/python scripts/nodonor_bracket.py cps
.venv/bin/python scripts/nodonor_bracket.py cfps
.venv/bin/python scripts/nodonor_bracket.py addhealth

# The missingness regression test (pins bug 2)
.venv/bin/python -m pytest tests/test_conditional_variance.py -q
```

Seed noise is real: T1 σ ≈ 0.08–0.22, T4 σ ≈ 0.3. Quote means over ≥5 seeds, never a
single run. The script prints per-config std.

The no-donor LLM pipeline (`conditional_variance.py` + `event_order_knowledge.py`)
needs a generation cache; it lives in the **gitignored but durable**
`results/nodonor_cache/` (`cfps_cond_raw.csv`, `cfps_elicit.json`).

## Provenance

The scratchpad holding every measurement script and both LLM caches was destroyed by a
`/tmp` wipe mid-session on 2026-07-16 (`nodonor_*.py`, `bench.py`,
`nodonor_cond_raw.csv`, `varrepair_elicit.json`). Real token loss, and the
reproducibility sections of the two 2026-07-15/16 no-donor reports now point at dead
paths. Hence: measurement code belongs in `scripts/`, LLM caches in `results/`. Never
only in a session scratchpad.
