# SSDataBench rewards resampling — and resampling is a privacy catastrophe

**Date:** 2026-07-15
**Scope:** what the cfps benchmark actually measures, and what beats it.
**Status:** measured and reproducible; numbers below are 12-seed unless noted.

---

## TL;DR

SSDataBench scores *fidelity* only — do the synthetic marginals, pairwise
relationships, regressions, and life-course orderings match the real ones? On
that instrument, three facts hold together and reframe the whole task:

1. **When you hold real microdata, whole-row resampling scores 0.806 — 2.7× the
   best published LLM (0.30) — and nothing can beat it.** Resampling is an
   unbiased draw from the population; fidelity is exactly what the benchmark
   measures; so resampling is a provable upper bound for any method that copies
   from the pool. The microdata regime is **saturated**.

2. **When you hold no microdata, an independent marginal draw scores 0.477 —
   still 1.6× the best published LLM.** Two separate LLM attempts to add joint
   structure on top of the marginals both returned to that floor, and an oracle
   fed the *true* correlations reaches only 0.496. There is a **~0.50 information
   wall** in the no-donor regime, and the trivial floor already sits just under
   it.

3. **The fidelity winner is the worst possible outcome on the axis the benchmark
   never measures.** Whole-row resampling republishes 100% of respondents
   verbatim (99.7% of them uniquely re-identifiable). Block-donor generation,
   which ties resampling on fidelity, sits at 0.1% — the coincidence floor.

Put plainly: **both regimes are dominated by trivial statistical baselines the
LLM-synthesis literature omits, and the method the benchmark crowns is the one
method you could never deploy.** The scientific question the benchmark's premise
is *about* — can a model's prior stand in for data that doesn't exist? — lives
entirely in the no-donor regime, and even there the interesting headroom is a
0.02–0.05 sliver above a marginal draw.

---

## The benchmark and what it measures

Five sub-benchmarks, each comparing a synthetic sample to a 1000-row real
reference for statistical distinguishability:

| | measures | mechanism |
|---|---|---|
| T1 | marginals | per-column KS / TV distance |
| T2 | pairwise | correlation / association agreement |
| T3 | regression | R² of `y ~ predictors`, delta-method z-test |
| T4 | life-course event order | chi-square on the permutation distribution |
| T5 | event order × covariate | T4 conditioned on a covariate |

Every sub-benchmark coerces to numeric then drops NA, so *which rows carry a
value* — the missingness pattern — silently selects who is scored. All five
reward one thing: a synthetic sample that is statistically **indistinguishable**
from the real one. None of them measure whether a synthetic person *is* a real
person.

### Two ceilings — the distinction the numbers turn on

Because the benchmark scores two *samples* for distinguishability, "how high can
anyone score" is itself a measurable quantity, and there are two versions of it:

- **sim = reference (0.913).** Score the 1000 reference rows against
  *themselves*. Measures only the eval's own bootstrap noise. Unreachable by
  anything that isn't the answer key — matching it means matching the
  benchmark's specific sampling noise, i.e. reading the benchmark. Not a target.
- **fresh real people vs the reference (0.806).** Draw 5000 *other* real cfps
  people from a provably-disjoint pool and score them. These are real humans —
  nothing is more "correct" — so **0.806 is the achievable ceiling for any
  method that copies from the pool.** It equals the resampling score, because
  resampling *is* "draw fresh real people".

The 0.10 gap between them is the irreducible sampling difference between any two
draws of the same population. Everything below is measured against the 0.806
achievable ceiling.

---

## Finding 1 — the microdata regime is saturated

Mean ± standard error over seeds 1–12, scored against the paper's exact 1000-row
reference, production code path, training drawn from the disjoint 57,474-row pool
(`load_disjoint_train`, verified 0 `pid` overlap):

| | T1 | T2 | T3 | T4 | T5 | overall |
|---|---|---|---|---|---|---|
| PNAS (best of 15 LLMs) | 0.14 | 0.62 | 0.43 | 0.05 | 0.75 | 0.30 |
| our shipped agent | 0.411 | 0.512 | 0.244 | 0.000 | 0.650 | 0.363 |
| **whole-row resample** (= achievable ceiling) | **0.873** | 0.813 | **0.708** | 0.825 | 0.809 | **0.806** ±.008 |
| block-donor (mega) | 0.829 | 0.813 | 0.698 | 0.758 | **0.821** | 0.784 ±.014 |
| block-donor (domain) | 0.816 | 0.774 | 0.701 | **0.867** | 0.809 | 0.793 ±.007 |
| ceiling (sim = ref, unreachable) | 0.958 | 0.949 | 0.748 | 0.960 | 0.950 | 0.913 |

Paired `block-donor(mega) − resample` = **−0.022** (t = −1.38, not significant);
no sub-benchmark shows a significant win for any modeling method over resampling.

**Why nothing can beat resampling here, in principle.** Whole-row resampling is
an unbiased draw from the pool's joint distribution. Any generative method —
block-donor, a fitted chain, an LLM — *approximates* that distribution and pays a
penalty wherever the approximation is imperfect (independence at block seams,
Gaussian smearing, prior mismatch). Fidelity is exactly the quantity T1–T5
measure. So resampling is an upper bound, and the best a modeling method can do
is tie it. We are at 0.793, within 0.016 of the ceiling. **The maximum a further
modeling can add is that 0.016, and it is bounded above by literal real people.**

The published LLMs sit at 0.30 not because synthesis is hard but because they
fail to reproduce the marginals (their T1 is 0.14, versus 0.87 for anything that
copies real values). They lose to resampling on the easiest sub-benchmark.

---

## Finding 2 — the no-donor regime hits a ~0.50 wall, and the floor already clears the field

When no microdata is available (the AGGREGATE / NO_DATA condition the PNAS
premise is actually about), resampling is impossible by construction. The only
signals are the population *marginals* and an LLM's *prior* about how variables
relate. We bracketed this regime measuring baseline and ceiling **before**
building anything (5 seeds unless noted):

| method | overall | note |
|---|---|---|
| PNAS (best published LLM) | 0.30 | — |
| **independent marginal draw (FLOOR)** | **0.477** | each column drawn from the pool marginal, zero cross-column structure |
| LLM pairwise-chain (its own priors) | 0.434 | *below* the floor — its correlation magnitudes damaged T5 |
| oracle pairwise-chain (TRUE correlations) | 0.496 | fed the real correlations; +0.019 over floor |
| unconditional LLM + marginal calibration | 0.483 | calibration lifts raw 0.168 → 0.483, lands at floor |
| **conditioned generator + calibration (BEST)** | **0.497** | PNAS-style per-person conditioning; +0.020 over floor, driven by T3 |
| achievable ceiling (microdata) | 0.806 | for reference — needs the data this regime lacks |

Two load-bearing facts:

- **The trivial marginal floor (0.477) already beats every published LLM
  (0.30).** Handing over the aggregate marginals — zero intelligence — is the
  single biggest win in this regime. Published LLMs lose to it because they can't
  reproduce the marginals they were effectively given.
- **~0.50 is an information wall.** The decisive evidence is the *oracle*: a
  method fed the true pairwise correlations still reaches only 0.496. Low-order
  structure (marginals + pairwise) tops out there. The gap from 0.50 to 0.806 is
  higher-order joint structure that only real rows carry. An LLM prior adds a
  real but small T3 lift (its income-given-education reasoning survives
  calibration: T3 0.300 → 0.343) and then hits the same wall.

So the one regime with genuine headroom offers ~0.02–0.05 of it above a marginal
draw, and reaching even that needs PNAS-style per-person conditioning plus a
marginal-calibration wrapper — not raw generation, which scores 0.168.

---

## Finding 3 — the axis the benchmark never measures: disclosure

Fidelity and re-identification risk are orthogonal, and the benchmark scores only
the first. We measured the second directly: for each generator, the fraction of
synthetic rows that **are** a real pool respondent, on the 32 modelled columns
(`src/ssdataagent/evaluation/disclosure.py`, measured against the 20k training
pool, coincidence floor from a fresh real sample):

| generator | copy_rate | unique_copy_rate | excess over floor |
|---|---|---|---|
| **whole-row resample** (the fidelity winner) | **1.000** | **0.997** | **0.999** |
| block-donor strategy (ties on fidelity) | 0.001 | 0.001 | 0.000 |
| LLM agent output | 0.000 | 0.000 | 0.000 |
| coincidence floor (fresh real vs pool) | 0.001 | — | — |

Every resampled "synthetic" person is a real respondent, and 99.7% of them are
*uniquely* identifiable on the modelled columns — the highest-risk disclosure
there is. Block-donor, which is a statistical tie with resampling on every
sub-benchmark, sits at the coincidence floor: no synthetic person is any real
person, because each is stitched from several donors.

**This is the crux.** A fidelity-only benchmark cannot distinguish these two
methods — and it crowns the one that is a maximal privacy violation. Synthetic
data exists precisely to avoid republishing respondents; a benchmark that rewards
verbatim republication is measuring the wrong thing, or at least only half of it.

---

## What this implies

**The result is not "our agent is bad."** The agent had real, diagnosed bugs
(see the companion report), and fixing them lifts it toward the ceiling. But even
a perfect agent cannot beat resampling on fidelity, because resampling is the
ceiling. The finding is about the benchmark, and it is defensible:

1. **On any dataset where microdata exists, SSDataBench is saturated by a
   one-line baseline** (`pool.sample(n, replace=True)`), and that baseline is
   undeployable. Reporting LLM fidelity scores here without the resampling line
   overstates the difficulty and the achievement.

2. **The scientific question lives in the no-donor regime**, where resampling is
   impossible — and there, the trivial marginal floor already beats the published
   LLMs, and the achievable headroom above it is a ~0.02–0.05 sliver under a hard
   ~0.50 wall.

3. **Fidelity needs a companion axis.** A fidelity-vs-disclosure frontier would
   separate resampling (fidelity 0.806, disclosure 1.0) from block-donor
   (fidelity 0.793, disclosure 0.001) — methods a fidelity-only score reports as
   equivalent. The disclosure metric shipped in
   `src/ssdataagent/evaluation/disclosure.py` is the first coordinate of that
   frontier.

The honest headline for the synthetic-data-via-LLM program on this benchmark:
*with data, a copy wins and can't be deployed; without data, a marginal draw
wins and the model's prior buys a couple of points.* That is worth stating
plainly, and it is what the numbers support.

---

## Reproducibility

- **Fidelity, 12 seeds:** production path, `load_disjoint_train(cfps, n_sample=…,
  seed)` for training, scored against `real_data/used_dataset/sampled_cfps.csv`
  (the paper's exact 1000 rows), 5000 simulated rows to cut the eval's bootstrap
  noise. Baselines and ceilings via `scratchpad/lock_headtohead.py`.
- **No-donor bracket, 5 seeds:** floor = independent per-column marginal draw
  from the disjoint pool (missingness preserved); oracle/pairwise/conditioned
  generators in `scratchpad/nodonor_*.py`; scored the same way.
- **Disclosure:** `scratchpad/disclosure_measure.py` over
  `disclosure_metrics(synthetic, pool, columns=modelled, baseline=reference)`;
  32 modelled columns; copy defined as an exact row match on those columns.
- **Leakage audit:** all 1000 benchmark rows ⊂ the 58,474-row source; disjoint
  pool excludes on `pid` (0 id-overlap); row-value audit found 1/20,000 donors
  value-identical to a benchmark row — a sparse-child collision (24/32 columns
  missing), not a duplicated person, and unable to move any score.

## Relationship to the diagnosis report

`docs/report/2026-07-13-full-agent-failure-modes.md` diagnoses *why the
tool-using agent scored badly* — four compounding bugs (marginal-only score gate,
crashing commit gate, missingness destruction, unmodellable censored columns) and
the block-donor fix. That diagnosis stands and is the mechanism behind the
"shipped agent 0.363" row above. **This report supersedes its framing:** the
agent's score was never the interesting quantity, because the ceiling it was
climbing toward is itself a trivial, undeployable baseline. Read the diagnosis
for the engineering; read this for what the benchmark is worth.
