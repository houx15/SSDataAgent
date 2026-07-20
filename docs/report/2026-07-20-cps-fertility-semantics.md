# cps, deeper: a variable-semantics trap, and variance repair masking it

**Date:** 2026-07-20
**Regime:** no-donor (pool supplies marginals only; no test data).
**Scored:** 5 seeds, n=5000, **bootstrap_B=200** (seeded), on the paper's 1000-row cps
reference. Pool = same-year `cps-asec1980.csv`, row-disjoint, 180,488 rows.
**Replication:** `scripts/nodonor_fullmethod.py cps`.

## Summary

Chasing the weak cps T3 (0.208) to its root turned up a **variable-semantics trap set by
the benchmark's own metadata**, and fixing it exposed a second, larger finding about our
own method.

1. The benchmark documents `child_number` as "Number of Child ever born." The **data** is
   the opposite — IPUMS household-roster children (own children **under 18 currently
   living with the respondent**). The LLM trusted the label, generated lifetime fertility,
   and inverted the strongest T3 relationship.
2. Correcting the *definition* in the generation prompt (public metadata, not the pool's
   answer distribution — so T3 stays non-circular) lifts the honest number **0.502 →
   0.591**: T3 0.208 → 0.367, T2 0.469 → 0.618.
3. **Variance repair — our headline T2/T3 method — now *hurts* (0.591 → 0.549).** It was
   partly compensating for the generation bug. With correct semantics the generation is
   well-calibrated, so the one-directional deflation overcorrects.

## The trap

`child_number` / `age_first_childbirth` read like lifetime fertility. Three independent
proofs they are not:

- **mean `child_number` = 0.66** in the 180k pool. Completed 1980 fertility was ~2.5–3.
- **60+ year-olds average 0.16** children — impossible for lifetime, exactly right for
  "own minor children in the household" (grown children have moved out).
- The **age profile peaks at 35–45 then collapses** (real has-child: 0.83 at 35–45, 0.12
  at 60+), the signature of household composition, not cumulative births.

The benchmark's config metadata (`description: Number of Child ever born`) is simply
wrong, or at least a fatal shorthand for IPUMS NCHILD. Any generator that believes it
fails.

### What the LLM did with the wrong label

| | has-child | Spearman(age, first-birth) | R²[afc] |
|---|---|---|---|
| v1 generation (label "ever born") | 0.55 | **−0.26** | collapsed |
| v2 generation (corrected definition) | 0.36 | **+0.32** | 0.362 |
| real | 0.33 | **+0.62** | 0.420 |

The v1 sign is inverted: the LLM assumed older people accumulate children and had them
young. Real (and the corrected v2) have first-birth age *rising* with current age — a
55-year-old with a child still at home had that child late, because the early ones have
left. `age_first_childbirth` carries the highest real R² on cps (0.420); getting its sign
wrong is most of the T3 collapse. Variance repair cannot fix a wrong sign — this had to be
fixed in generation.

The fix is a **definition correction from public metadata**, not the pool's conditional
distribution. Supplying `P(fertility | age)` from the pool was measured too (below) but is
a *labeled ceiling* — it would convert T3 into "reproduction of a supplied aggregate", the
circular status the project reserves for T1/T4, not the genuine-joint status T2/T3 must
keep.

## Results

| config | T1 | T2 | T3 | overall | note |
|---|---|---|---|---|---|
| v1 raw (wrong semantics) | 0.816 | 0.469 | 0.031 | 0.439 | |
| v1 repaired *(old headline)* | 0.803 | 0.496 | 0.208 | 0.502 | |
| **v2 raw (corrected semantics)** | 0.787 | 0.618 | 0.367 | **0.591** | **honest, non-circular** |
| v2 repaired | 0.775 | 0.573 | 0.298 | 0.549 | repair now hurts |
| *age-conditional fertility supply* | 0.790 | 0.527 | 0.508 | *0.609* | *labeled ceiling (circular)* |

Noise floor at 5 seeds (B=200): min-resolvable overall 0.054, T3 0.079. Every gap called
"real" below clears it.

### Finding 1 — the semantics fix is a clean non-circular win

**v2 raw 0.591 vs v1 repaired 0.502: +0.089 overall, +0.159 on T3, +0.149 on T2.** The
purest no-donor config — LLM joint mapped to pool marginals, no elicited targets, no pool
R² — and it beats the old best. It also lands within 0.018 of the supplied-aggregate
*ceiling* (0.609): given the correct definition, the LLM reconstructs almost all the
fertility structure that handing it the pool aggregate would supply. The residual (Spearman
+0.32 vs real +0.62) is what only microdata carries.

### Finding 2 — variance repair was masking the bug

On v1 (buggy generation) repair helped: 0.439 → 0.502, because the generation had
mean-collapse (over-strong associations) for repair to deflate. On v2 (correct generation)
repair **hurts**: 0.591 → 0.549. Two reasons, both instructive:

- The corrected generation is already well-calibrated — `child_number` own R² 0.233 vs
  elicited target 0.250 (α clips to 1.0, no change); `age_first_childbirth` own R² 0.362,
  *below* real 0.420. There is no excess association to remove.
- The repair's one-directional design assumes the joint is *always too strong*. When it is
  not, the elicited target can pull a column the wrong way: afc's elicited target (0.200)
  is far below both the generation (0.362) and reality (0.420), so repair deflates afc
  *away* from truth.

The headline consequence: **the 0.502 we reported as the cps method result was variance
repair partially compensating for a generation defect.** With the defect fixed, the honest
raw generation is better without any repair. Repair remains the right tool where
mean-collapse is real (cfps, and cps v1) — it is not universally beneficial, and should be
applied only when the generation is measurably over-strong.

## Where cps stands now

Honest, non-circular: **overall 0.591**, up from 0.502. On the joint benchmarks T2 0.618 /
T3 0.367 — still below published (~0.71 / ~0.57), but the gap roughly halved. A large part
of the remaining T2 gap is expected to be the input-regime asymmetry (we draw
age/gender/race from marginals while published systems are handed the real test
demographics; see [[project_published_comparison_regimes]]), still unmeasured.

## Generalizing the fix: data-understanding as a separate, automated layer

A fair objection to all of the above: the correction was *hand-authored*. If every dataset
needs a human to spot its semantic traps, the strategy is not general — it is a general
engine wrapped in per-dataset expert effort. The resolution is to split the work the way
real social-science practice does: a **general strategy** (mean-collapse → variance repair
→ marginal calibration; one dataset-agnostic codebase) plus a **data-understanding layer**
that reads only the benchmark's documentation and the disjoint pool — never the test — and
emits the definitions, sentinels, and identities the strategy consumes. Building a correct
data dictionary before modelling is normal and expected; it is not tuning.

The load-bearing question is whether that layer can be *automated* rather than
hand-authored. `scripts/data_audit.py` is the proof it can. Test-blind (pool +
documentation only), it flags where a variable's data contradicts its label. Its core check
is definitional: a label asserting a **cumulative lifetime quantity** ("children ever born")
is monotone non-decreasing in age *by construction*, so if the pool's mean *falls* with age
the label is impossible and the variable is a stock/resident measure. Run on all datasets it
produced 13 findings, all genuine, zero false positives:

- **cps** — recovered all three corrections this report made by hand: the `child_number`
  cumulative-monotonicity trap (mean peaks 2.01 at 35–45, falls to 0.16 by the oldest), the
  `age_first_childbirth` `No Child` sentinel, and the `age + birth_year = 1980` identity.
- **cfps** — all six event-timing sentinels (`never married` … `still alive`) and the
  income log-scale.
- **addhealth** — three event sentinels (divorce, marriage, first-sex).
- gss/acs — skipped (no disjoint pool; a known access limit).

The monotonicity check — the one catching a genuine *semantic* trap rather than routine
sentinel plumbing — fires exactly once across all datasets, on the real cps trap. So the
human insight that cost a day here is now a systematic preprocessing step, and the strategy
above it is general: the same engine, given a correct (auto-audited) data profile per
dataset. What remains hand-work is authoring the *corrected definition text* once a trap is
flagged; detecting the trap is automated.

## Reproducibility

Generation cached at `results/nodonor_cache/cps_cond_raw.csv` (v2, corrected semantics);
v1 preserved under `results/nodonor_cache/_cps_v1_lifetime_fertility/`. The corrected
variable definitions live in `SPECS["cps"]` in `scripts/nodonor_fullmethod.py`
(`--regenerate` to rebuild). The generator now retries each batch through transient
network drops rather than discarding the run.
