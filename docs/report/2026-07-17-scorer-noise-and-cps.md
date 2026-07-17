# The scorer is the noisiest thing we measure — and cps, finished

**Date:** 2026-07-17
**Scope:** cps (no-donor full method), plus a measurement-precision finding that
touches every number this project has produced.
**Code landed:** `23dc877` (covariate-R² fidelity), `5e1aeb9` (scorer seeding +
`bootstrap_B`).
**Replication:** `scripts/nodonor_fullmethod.py cps`, `scripts/nodonor_bracket.py cps`,
`scripts/bench_noise.py cps`.

## Summary

Two results, and the second is the important one.

**cps is finished.** The no-donor method ports: mean-collapse replicates exactly, and
per-outcome variance repair lifts T3 from **0.031 → 0.208** and overall from **0.439 →
0.502**. The previously recorded 0.413 was measured against the wrong (1970) pool and is
superseded. But the cfps headline does **not** port — on the joint benchmarks cps loses
to published PNAS (T2 0.496 vs ~0.71, T3 0.208 vs ~0.57), and a labeled ceiling proves
better elicitation cannot close it.

**The benchmark's scorer is non-deterministic, and noisier than most effects we have
been reporting.** Scoring one *fixed* frame repeatedly returns overall anywhere from
0.313 to 0.498. Two causes: the bootstrap RNG is unseeded, and `bootstrap_B: 1` for T1
in every dataset config. Our standard 5-seed protocol **cannot resolve overall
differences below ~0.054**. This is a single root cause behind several things we had
logged as separate puzzles, and it invalidates a handful of fine-grained claims — listed
under *Retractions*.

## Part 1 — the measurement floor

### What we found

Two runs of `nodonor_fullmethod.py cps` with identical seeds and identical cached input
disagreed (raw T3 0.047 / 0.007 / 0.027). The generator was not the cause. Scoring one
fixed frame four times:

| | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| call 1 | 1.000 | 0.495 | 0.000 | 0.498 |
| call 2 | 0.857 | 0.469 | 0.033 | 0.453 |
| call 3 | 0.857 | 0.481 | 0.000 | 0.446 |
| call 4 | 0.429 | 0.476 | 0.033 | 0.313 |

The input never changed. Two causes, both in `ssdatabench/evaluation/code_by_type/`:

1. **`rng = np.random.default_rng()` is unseeded** (type1.py:41, type3.py:128, and the
   rest). No run is reproducible.
2. **`bootstrap_B: 1` for T1 in every dataset config** (cps, cfps, gss, acs). A single
   500-row bootstrap draw decides each variable's pass/fail, so T1 is quantized
   coin-flipping — the 0.429 / 0.857 / 1.000 above are 3/7, 6/7, 7/7. T2–T5 use B=10.

### How much noise

`scripts/bench_noise.py cps --reps 30`, one fixed rowresample frame:

| | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| σ per call | **0.100** | 0.015 | 0.063 | 0.043 |
| range over 30 calls | 0.714–1.000 | 0.831–0.890 | 0.633–0.900 | 0.740–0.913 |
| SE at 5 seeds | 0.045 | 0.007 | 0.028 | 0.019 |
| **min resolvable, 5 seeds** | **0.126** | 0.019 | 0.079 | **0.054** |

*min resolvable* = the smallest gap two 5-seed means can distinguish at p<0.05. **Any
difference we have reported below that line is noise, not a result.**

### The fix, and why it does not redefine the benchmark

`insignificant_rate` is `not_sig / B` — the *fraction of B bootstrap iterations* in which
the test came out insignificant. B is a **Monte Carlo replicate count, not part of the
estimand**. Raising it estimates the same quantity more precisely. (The library's own
signature defaults to `B=1000`; the configs' `B=1` looks like a speed hack.)

Verified directly — one fixed frame, 12 seeds per row, varying only B:

| B | T1 mean | T1 σ | T3 mean | T3 σ | overall mean | overall σ |
|---|---|---|---|---|---|---|
| shipped (1 / 10) | 0.905 | 0.093 | 0.794 | 0.062 | 0.853 | 0.030 |
| 50 | 0.872 | 0.012 | 0.784 | 0.029 | 0.838 | 0.009 |
| 200 | 0.880 | 0.008 | 0.782 | 0.012 | 0.841 | **0.004** |

Means hold; σ falls as 1/√B (T3 from B=10→50 predicts 0.062/√5 = 0.028, observed 0.029).
B=200 makes `overall` **7× more precise**.

`score()` now takes `seed` (pins the bootstrap — same seed gives bit-identical output,
the first time anything here has been reproducible) and `bootstrap_B` (`None` = shipped
config).

**The asymmetry that must be stated whenever B is raised:** published numbers were
produced at the shipped B and carry that noise. A high-B run of ours is a *better
estimate of the same statistic*, but its error bars are not symmetric with a published
run's. Do not read a small gap against a published figure as real just because our side
is now precise.

## Part 2 — cps, finished

Pool = same-year `cps-asec1980.csv`, row-disjoint, 180,488 rows. 480 LLM-generated
respondents (claude-sonnet-4.5), 5000 simulated rows, 5 seeds, **B=200**, scored on the
paper's 1000-row benchmark sample.

### The full method

| config | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| raw generation (no repair) | 0.816 | 0.469 | 0.031 | 0.439 ±.014 |
| **repaired, elicited targets [honest]** | 0.803 | 0.496 | **0.208** | **0.502** ±.009 |
| repaired, pool R² targets [labeled ceiling] | 0.807 | 0.474 | 0.264 | 0.515 ±.014 |

### The bracket (LLM-free reference, B=200)

| config | T1 | T2 | T3 | overall | regime |
|---|---|---|---|---|---|
| independence (floor) | 0.862 | 0.353 | 0.004 | 0.407 ±.005 | no-donor |
| copula-old (buggy) | 0.863 | 0.546 | 0.337 | 0.582 ±.005 | microdata |
| copula-fixed | 0.820 | 0.634 | 0.580 | 0.678 ±.008 | microdata |
| **rowresample (ceiling)** | 0.837 | 0.858 | 0.755 | **0.816** ±.012 | microdata |

Regimes are not comparable. The bracket's *structure* survives the precision upgrade
intact — adjacent rungs differ by +0.175, +0.096, +0.138, all far above the noise floor.

### What holds

- **Mean-collapse replicates on cps and the repair works.** T3 0.031 → 0.208 (Δ 0.177,
  ~12σ); overall 0.439 → 0.502 (Δ 0.063, ~8σ). The project's core modelling finding now
  holds on a second dataset at a precision that can defend it.
- **The stale-pool result is superseded.** 0.413 (1970 pool) → **0.502** (1980 pool).
- **Elicitation error is real but small, and is not the wall.** Handing the repair
  *perfect* pool-derived targets lifts T3 only 0.208 → 0.264 (Δ 0.056, ~5σ) — nowhere
  near published ~0.57. Better prompting cannot rescue cps T3.

### What does not

- **The cfps headline does not port.** On cfps the no-donor method beats published PNAS
  on the joint benchmarks (T2 0.649 vs 0.62, T3 0.446 vs 0.43). On cps it loses badly:
  T2 0.496 vs ~0.71, T3 0.208 vs ~0.57. Both gaps are an order of magnitude above the
  noise floor. **"Genuine joint knowledge beats published" is a cfps result, not a
  property of the method.**
- **T1 does not drop under repair.** An earlier reading of this run said 0.914 → 0.857.
  At B=200 it is 0.816 → 0.803: Δ 0.013 against a min-resolvable of 0.126. That was
  scorer noise.

### An unresolved asymmetry in every published comparison

`type2`/`type3` declare `age`, `gender`, `race` as `input: true`, and the benchmark's
"real" file is named `sampled_inputs_*.csv`. **Published systems are handed the test
respondents' actual demographics** and generate the rest — which is also why T1 scores
only the 7 non-input columns. Our no-donor method instead *draws* age/gender/race from
pool marginals. Those variables are associated in real cps (mean age 34.2 F vs 30.0 M;
24.2 Hispanic vs 33.4 non-Black non-Hispanic), and T2 scores pairs *among* them, so
drawing them independently forfeits points published systems get for free.

This is the same class of error as the microdata/no-donor mixup retracted on 2026-07-17
— except it runs *against* us. Every "vs published" number in this project compares our
harder regime to their easier one. **Not yet resolved; do not treat the cps T2/T3 deficit
as fully diagnosed until it is.** The clean experiment is to run our method in the
benchmark's own protocol (inputs given) and compare there.

## Retractions

Forced by the noise floor (min resolvable at 5 seeds: overall 0.054, T1 0.126, T3 0.079):

- **"The trivial independence floor (0.433) beats the best single published system
  (~0.40)."** At B=200 the floor is **0.407**. A dead heat, and published carries its own
  noise. Withdrawn.
- **"cps no-donor 0.481 beats the 0.46 cross-model envelope."** Δ 0.02, well inside
  noise. Withdrawn.
- **"T1 drops when variance repair is applied."** Δ 0.013 vs min-resolvable 0.126. Noise.
- **cfps T4 = 0.950** (2 seeds) → 0.780 ± 0.130 at 5 seeds. Already corrected earlier
  today; the root cause is now known to be this same scorer noise, not merely "T4 is a
  seed-noisy benchmark."

Explicitly **surviving** the audit, all with gaps far above the floor: the missingness
fix's +0.26/+0.31 on T3; mean-collapse and its repair on both datasets; addhealth T4 =
0.000 from real microdata; every adjacent gap in both brackets; cps T2/T3 losing to
published.

## Consequences for how we measure

1. **Report `bootstrap_B` and the scoring seed with every number.** A figure without them
   is not reproducible.
2. **Default to B=200** for our own comparisons; use the shipped config only when the
   point is to be published-comparable, and say which.
3. **Never quote a difference below the min-resolvable line.** Run `bench_noise.py` for a
   dataset before trusting a small gap on it.
4. **Historical numbers in this repo predate the fix** and carry σ≈0.043 on overall
   (worse on T1). Treat any past gap under ~0.05 as unproven rather than false.
