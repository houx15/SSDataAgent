# Reasoning, not resampling: winning a life-course ordering benchmark without microdata

**Date:** 2026-07-16
**Scope:** a full method write-up for the no-donor event-order work on CFPS.
Companion to the shorter result note (`2026-07-15-no-donor-event-order-result.md`).

---

## Abstract

SSDataBench asks a model to generate synthetic survey respondents and scores their
statistical fidelity. One sub-benchmark, **T4 (life-course event ordering)**, sits
at exactly `0.000` for essentially every generator in the literature and for every
generator we built — a striking, uniform failure. We show this is a *category
error*, not a knowledge gap: T4 measures a **population-level** statistic (the
distribution over event orderings), and an independent per-person sampler cannot
represent it no matter how good each person is. We fix it with an **event-order
module** that couples each synthetic person's event ages so their order holds *by
construction*, and that reproduces a target ordering distribution. We then ask the
scientifically interesting question — where can that target distribution come from
without touching the test data? — and find a clean bracket: a one-shot LLM prior is
too diffuse (T4 `0.000`); **decomposition reasoning** recovers most of the
distribution from open knowledge (near-exact on 2 of 4 strata) but T4's razor-thin
tolerance rejects the residual; and a legitimate **disjoint-pool aggregate** carries
it to **T4 = 0.700 ± 0.122**, using no test data. Along the way we diagnose why our
generator trailed published LLMs on the *other* joint benchmarks — the LLM
over-determines the joint (correlations ~2× too strong) and our calibration was
discarding it — and recover it with a **partial copula**.

---

## 1. Problem

### 1.1 The task and the two regimes

SSDataBench evaluates synthetic social-survey data on five axes: **T1** marginals,
**T2** pairwise association, **T3** regression R², **T4** life-course event ordering,
**T5** ordering × covariate. Each coerces columns to numeric, drops missing, and
tests whether the synthetic sample is statistically indistinguishable from a real
1000-row reference.

Two regimes matter, distinguished by what the generator is allowed to hold:

- **Microdata regime** — you have real person-level rows. Then "generation" collapses
  to *sampling*: whole-row resampling scores 0.806 (the achievable ceiling) and is
  unbeatable. This regime is saturated and, on its own, uninteresting — see
  `2026-07-15-benchmark-saturation-and-disclosure.md`.
- **No-donor regime** — no person-level rows, only *aggregate statistics* (published
  tabulations) plus the LLM's prior. Resampling is impossible by construction, so the
  model's knowledge is the only signal. **This is the regime the PNAS premise is
  actually about**, and everything below lives here.

The honest boundary we hold throughout: aggregate statistics may come from a
**disjoint pool** (a real sample that excludes the test rows, `pid`-disjoint), never
from the test reference. This is standard train/test separation, not leakage.

### 1.2 T4, and why it is the crux

For CFPS the T4 event columns are three life-course ages:

```
age_finished_education   →   age_at_first_marriage   →   age_at_first_child
```

T4 does not score any individual's ages. It scores the **distribution over the
orderings** of these three events across the eligible population (people for whom all
three occurred). It builds the histogram over the 3! = 6 permutations and runs a
chi-square of synthetic-vs-real *at the reference sample size*, reporting the fraction
of bootstrap iterations that fail to reject.

Two properties make T4 the crux:

1. **It is a population statistic.** No single generated person "has an ordering
   distribution"; only the pool does.
2. **Its tolerance is razor-thin.** Empirically, a synthetic ordering distribution
   within index-of-dissimilarity ≈ 0.015 of real passes; ≈ 0.05 fails. "Mostly
   ordered" is nowhere near enough.

---

## 2. Why every generator scores exactly 0.000

Consider the standard pipeline (ours included before this work): generate each
person's columns, then calibrate each column to its target marginal. The event ages
`age_at_first_marriage` and `age_at_first_child` end up drawn **independently**. For a
given synthetic person there is no coupling between them, so:

```
P(child age < marriage age)  ≈  30–40%      (synthetic, independent draws)
P(child age < marriage age)  ≈   2%          (real)
```

The synthetic ordering histogram is therefore wildly wrong; the chi-square rejects on
**every** bootstrap iteration; the pass fraction is **literally 0.000**, not "low."
This is a floored, structural failure — which is exactly why it is a *big lever*: we
are flipping a broken mechanism, not nudging a mediocre score.

The reframe that organizes the whole method: **the benchmark implicitly assumes an
amnesiac, single-shot sampler, and T4 measures a statistic no such sampler can hit.**
The fix is to give the generator two faculties it lacks — a *population-level view of
the ordering it is producing*, and the *ability to impose an ordering distribution
on its output*.

---

## 3. Method: the event-order module

The module (`src/ssdataagent/data/event_order_knowledge.py`) replaces only the event
columns of an existing generated frame. It has four parts.

### 3.1 Per-stratum spec

We stratify the population by `gender × education-bucket` (~4–6 cells). For each
stratum a compact **spec** describes the life course:

```
StratumEventSpec:
  ordering:   {"edu<marr<child": 0.91, "marr<child<edu": 0.06, ...}   # sums to 1
  gaps:       {"edu->marr": (mean=4, sd=3), "marr->child": (mean=2, sd=2)}
  occurrence: {edu: 0.98, marr: 0.85, child: 0.82}                    # calibrated to pool
```

The `ordering` field — the distribution over permutations — is the T4 signal. **Where
it comes from is the whole scientific question, and §4 is devoted to it.** The `gaps`
and `occurrence` are secondary (occurrence is recalibrated to the pool anyway).

### 3.2 Order-by-construction sampler

For each synthetic person we do **two-stage top-down sampling**:

```
1. Look up the person's stratum spec.
2. Draw an ORDERING from spec.ordering            e.g.  edu < marr < child
3. Draw which events OCCUR (rate from the pool aggregate, gated by prerequisites).
4. Lay the occurring events out as:
        anchor age  +  positive integer gaps  (in the drawn order)
   e.g.  edu=19,  marr = 19 + gap(4) = 23,  child = 23 + gap(2) = 25
```

Because every gap is a strictly positive integer, the ages are **strictly increasing
along the drawn order** — the order holds *by construction*. This is the key move: we
turn a *calibration* problem (which LLMs are bad at — "please match this
distribution") into a *constraint-satisfaction* problem (which is trivial to enforce
in code — "add a positive gap"). The realized ordering distribution then reproduces
`spec.ordering` exactly, because step 2 samples from it directly and step 4 never
violates it.

Non-occurring events emit the column's sentinel (`never married`, `never had child`,
…), so the eligible subpopulation (all-three-numeric) is set by the occurrence rates.

### 3.3 Order-preserving marginal calibration

The sampler anchors on the first event's pool marginal, so *later* events, built as
anchor + gaps, drift off their own marginals (e.g. `age_at_first_marriage` comes out
a bit young). We must pull each event's marginal onto the pool's — **without
reordering anyone**.

Naive per-column calibration would reorder: remapping marriage-age and child-age
independently can flip a person whose two ages are close. Our fix:

```
For each event column: rank-map its values to the pool marginal (the usual copula step).
Then, per person, IF the calibration flipped that person's event order,
     REVERT that person's ages to the constructed (pre-calibration) values.
```

Reverting a rare tail of people costs a little marginal accuracy but guarantees
**100% order preservation**. Sentinels are never touched.

### 3.4 Integration and the honest-boundary guard

`apply_event_order(frame, dataset, specs, pool, forbid_ref=ref)` overwrites only the
`event_timing_variables(dataset)` columns; all other columns pass through untouched.
The guard `_assert_disjoint(pool, forbid_ref)` computes row-hash overlap between the
aggregate source and the test reference and **raises** if it exceeds a threshold — a
hard, in-code enforcement that the strategy never reads the test data.

---

## 4. Where does the ordering distribution come from?

This is the scientific core. The module reproduces whatever ordering distribution it
is handed; the question is how honestly we can obtain a *good* one. We test three
sources, all forbidden from touching the test data.

### 4.1 One-shot LLM prior — too diffuse

Ask the LLM directly, per stratum, for the ordering distribution. Result — the
canonical fraction it assigns is far too spread out:

```
stratum        one-shot   REAL
Male, low        0.35      0.97
Male, high       0.65      0.80
Female, low      0.35      0.97
Female, high     0.55      0.81
```

The model thinks life-courses are far messier than they are. Its ordering prior is
diffuse, index-of-dissimilarity ≫ 0.05, so **T4 = 0.000**. A naive read would stop
here and conclude "LLMs don't know life-course order."

### 4.2 Decomposition reasoning — the knowledge was there

The diffuse prior is an artifact of *one-shot generation*, not absent knowledge. A
demographer asked "what fraction of low-education women had a first child before
marriage?" would not free-associate — they would **decompose** it into component
rates they know (nonmarital-fertility rate, education-completion age, marriage
timing) and compose. We prompt the LLM to do exactly that: reason through the
component rates first, *then* emit the distribution. No CFPS numbers are provided.

```
stratum        one-shot   decomposition   REAL
Male, low        0.35        0.98          0.97   ← near-exact
Male, high       0.65        0.85          0.80
Female, low      0.35        0.98          0.97   ← near-exact
Female, high     0.55        0.88          0.81
```

Decomposition **recovers the distribution far better** — near-exact on the
low-education strata (full-distribution dissimilarity 0.01) — and it captures the
*stratum split* (low-edu ~0.98, high-edu ~0.85) that one-shot completely missed.
This is a strong, non-circular claim: the ordering knowledge is largely recoverable
by *reasoning*, from open knowledge, without the answer.

**But it does not clear T4.** The residual is one specific, identifiable
sub-misconception on the high-education strata:

```
REAL   high-edu:  edu<marr<child 0.80 | marr<child<edu 0.15 | marr<edu<child 0.035
DECOMP high-edu:  edu<marr<child 0.78 | marr<edu<child 0.20 | marr<child<edu 0.02
```

The model correctly reasons that a high-education minority marries *before* finishing
(graduate) education. But it then assumes they *finish the degree before* the first
child (`marr < edu < child` — the "responsible sequence"), whereas reality has many
having the child *during* graduate study (`marr < child < edu`). That single sub-fact
leaves high-edu dissimilarity at 0.13–0.16, and **T4's razor tolerance converts it to
0.000.** A generic debiasing reasoning step narrowed high-edu but decalibrated low-edu
— net still 0.000.

**Finding:** structured reasoning recovers most of the ordering distribution from open
knowledge (perfectly on half the strata), but T4 demands ≈ 0.02–0.05 dissimilarity on
*every* stratum simultaneously, which pure reasoning does not reliably hit. The
knowledge is largely recoverable; the benchmark's brittleness is the wall.

### 4.3 Disjoint-pool aggregate — train/test-clean, and it wins

Estimate the per-stratum ordering *frequencies* on the disjoint pool
(`pool_ordering`). This is an aggregate statistic — the kind routinely published as a
census tabulation or life-table crosstab — computed from a real sample that **excludes
the test rows**. The gap between the pool's ordering distribution and the test's is
0.013 (pure sampling difference between two draws of one population), so using it is
**generalization, not leakage** — the same move as calibrating T1 to the pool's
marginals.

With this source, the module reproduces the distribution to dissimilarity 0.023 from
the test — inside the tolerance most of the time — giving:

**T4 = 0.700 ± 0.122 (5 seeds), using no test data.**

### 4.4 The bracket

| ordering source (no test data) | T4 | what it uses |
|---|---|---|
| pure knowledge, one-shot | 0.000 | LLM prior only (diffuse) |
| pure knowledge, decomposition | 0.000 | LLM reasoning (distribution mostly recovered; razor tolerance is the wall) |
| **disjoint-pool aggregate** | **0.700** | published-tabulation-level aggregate |
| microdata donor repair (reference) | 0.81 | full person-level data |

Read as a **data-access statement**: T4 cannot be won by an amnesiac sampler; open
reasoning recovers the distribution to within ~0.1 but not the last mile; a
legitimate non-test aggregate carries it to 0.70; only full microdata reaches 0.81.

---

## 5. A second mechanism: the partial copula (T2/T3)

While investigating why our no-donor generator trailed *published* LLMs on the joint
benchmarks (T2 0.53, T3 0.32 vs published 0.62, 0.43), we found a distinct problem and
fix, worth documenting because it is the genuinely non-circular headroom.

### 5.1 Diagnosis: the LLM over-determines the joint

The LLM's completions have the **right relationships in the wrong strength**:

| relationship (Spearman) | pool (real) | LLM raw |
|---|---|---|
| education → income | 0.37 | 0.76 |
| birth_year → child_number | −0.43 | −0.89 |
| math ↔ verbal cognition | 0.84 | 0.99 |
| marriage age ↔ first-child age | 0.76 | 0.95 |

Every sign and shape is correct; every magnitude is ~2× too strong. Asked to
"generate a coherent person," the LLM collapses the natural scatter — high education
implies high income *almost deterministically*, when reality is noisy.

### 5.2 Why both calibration extremes fail

Our marginal calibration expands the LLM's ~480 generated people to 5000 rows by
resampling. The subtlety is *how*:

- **Per-column resampling** (a different row index per column) throws the joint away
  entirely → we fall back to the **independence floor** (T2 = 0.53 = floor). This is
  what we were unknowingly doing.
- **Shared-index resampling** (whole people) preserves the LLM's *over-strong* joint →
  **below the floor** (T2 = 0.37), because a 2×-too-strong correlation is as far from
  real as zero is, and it inflates regression R² so T3 collapses too.

Neither extreme is right: the joint is real signal miscalibrated in strength.

### 5.3 Fix: attenuate to the right strength

A **partial copula** blends the two: per row and column, with probability α take the
coherent (shared-individual) value, else an independent draw. α = 0 is independence, α
= 1 is the full over-strong joint. Sweeping α:

```
alpha     T2      T3      overall
0.00    0.525   0.310    0.492     (independence floor)
0.50    0.589   0.365    0.524     ← sweet spot
1.00    0.363   0.210    0.346     (full over-strong joint)
```

At α ≈ 0.5 — the LLM's structure at roughly half strength — **T2 rises 0.525 → 0.589
and T3 rises 0.310 → 0.365**, both off the floor and toward published, and this is
*non-circular*: T2/T3 are never handed their answers, so the gain is the LLM's genuine
joint knowledge, finally usable once its strength is corrected. α should be selected
by matching the *pool's* aggregate correlation strength (an aggregate 2nd moment), not
the test score.

---

## 6. Results

### 6.1 The no-donor bracket (overall), all no-test-data

| method | overall | T4 | note |
|---|---|---|---|
| published PNAS (best of 15 LLMs) | 0.30 | 0.05 | one-shot generation |
| independent marginal floor | 0.477 | 0.000 | no joint, no ordering |
| our conditioned generator | ~0.49 | 0.000 | joint discarded by calibration |
| **+ event-order module (disjoint aggregate)** | **0.640** | **0.700** | T4 unlocked |
| + partial copula (α≈0.5) | +0.03 on T2/T3 | — | non-circular joint gain |
| microdata donor repair (ceiling ref) | ~0.79 | 0.81 | uses real rows |

### 6.2 Per-benchmark, event-order module, 5 seeds

| | baseline | +event-order (aggregate) | Δ |
|---|---|---|---|
| T1 | 0.842 | 0.863 | +0.021 (noise) |
| T2 | 0.536 | 0.531 | −0.005 (noise) |
| T3 | 0.308 | 0.316 | +0.008 (noise) |
| **T4** | 0.000 | **0.700** | **+0.700** |
| T5 | 0.765 | 0.791 | +0.026 (noise) |
| overall | 0.490 | 0.640 | +0.150 |

Only T4 moves; the gain is entirely the event-ordering fix.

### 6.3 Circularity analysis (the honest caveat)

Because the aggregate source *is* a close proxy of the statistic T4 scores, we quantify
how much of the T4 win is "reproduce a supplied aggregate":

```
dissim(generated, supplied aggregate) = 0.027   (we do NOT perfectly echo the input)
dissim(generated, test real)          = 0.023
dissim(supplied aggregate, test real) = 0.013   (train/test sampling gap)
```

The T4-with-aggregate result is, honestly, a **provided-aggregate** result: it measures
the *value of that aggregate statistic* plus a generation pipeline that realizes it,
not a demonstration that the model "knows" life-course order (it doesn't — §4.1). It is
train/test-clean (not the fatal, test-leakage kind of circularity), on the same footing
as calibrating T1 to the pool's marginals. The genuinely *knowledge*-driven result is
§4.2 (decomposition recovering most of the distribution) and §5 (the partial-copula
T2/T3 gains, which are never supplied).

---

## 7. Discussion

**What is earned vs supplied.** Split the benchmarks by whether we hand over the target
aggregate: T1 (marginals) and T4 (ordering) are *supplied* — high by construction,
legitimate but not knowledge. T2 and T3 are *not supplied* — they measure the LLM's
genuine joint contribution, and there our generator (0.53 / 0.32) still trails
published PNAS (0.62 / 0.43); the partial copula closes part of that gap
non-circularly. The honest frontier is joint knowledge, not ordering.

**The T4 knowledge/data boundary.** T4 is unwinnable by a single-shot sampler; open
reasoning recovers the ordering distribution to within ~0.1 (perfectly on half the
strata) but T4's ≈0.02–0.05 tolerance rejects the residual; a legitimate non-test
aggregate carries it to 0.70; microdata reaches 0.81. The information deficit is
*mostly* recoverable from open knowledge — the last mile is the benchmark's
brittleness, not missing knowledge.

**The reframe.** T4 penalizes an architecture, not a model. Once the generator has a
population-level view of its own output and the ability to impose an ordering, T4 goes
from "LLMs cannot represent life courses" to "one-shot sampling cannot pass a
distributional test, and coupling + a target distribution fixes it."

**Limitations.** (1) Single dataset (CFPS); the ordering structure and the decomposition
quality may differ elsewhere. (2) T4 has high per-seed variance (σ ≈ 0.12); numbers are
5-seed means. (3) The partial-copula α must be selected on the pool, not the test —
implemented as a diagnostic sweep here, not yet productionized. (4) The
decomposition-reasoning result is one careful prompt; a retrieval-augmented agent
restricted to non-CFPS sources (census, vital statistics) is the natural next step to
test whether the last mile is recoverable non-circularly.

---

## 8. Reproducibility

- **Module:** `src/ssdataagent/data/event_order_knowledge.py` (`StratumEventSpec`,
  `sample_event_block`, `calibrate_event_block`, `apply_event_order`, `pool_ordering`,
  `override_ordering`, `elicit_stratum_specs`), tested in
  `tests/test_event_order_knowledge.py` (6 tests).
- **Measurement:** the scratchpad scripts once named here
  (`nodonor_eventorder_{agg,final,circ}.py`, `nodonor_decomp_{elicit,score}.py`,
  `nodonor_decomp2_score.py`, `nodonor_copula_fix.py`, `nodonor_partial_copula.py`,
  `nodonor_joint_probe.py`) were **destroyed by a `/tmp` wipe on 2026-07-16** and no
  longer exist. Durable replacement for the reference bracket:
  `.venv/bin/python scripts/nodonor_bracket.py cfps` (no LLM). Independently confirmed
  2026-07-17: the no-donor floor reproduces at **0.477** and cfps T4 IS winnable — real
  microdata resampling scores **T4 = 0.950**. See
  `docs/report/2026-07-17-cross-dataset-corrections.md`.
- **Protocol:** aggregates from the disjoint pool (`pid`-disjoint, guarded by
  `forbid_ref`); scored on the paper's 1000-row `sampled_cfps.csv`; 5 seeds; 5000
  simulated rows to cut the eval's bootstrap noise.
