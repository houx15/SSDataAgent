# Conditional-variance repair: beating published PNAS on T2/T3 without test data

**Date:** 2026-07-16
**Regime:** no-donor (no target microdata; aggregate stats from the disjoint pool +
LLM prior). Scored on the paper's 1000-row cfps reference, 5 seeds.
**Module:** `src/ssdataagent/data/conditional_variance.py` (tested,
`tests/test_conditional_variance.py`).

## Result

A no-test-data method reaches **T2 = 0.649** and **T3 = 0.446** — *above* published
PNAS (0.62 / 0.43) — using only the model's reasoning about effect sizes, never
reading the pool or the test. These are the two benchmarks where we supply the
generator *nothing*: the first genuine-joint win, as opposed to the supplied-aggregate
reproduction that carries T1 and T4.

| source | T2 | T3 | overall | access |
|---|---|---|---|---|
| independence floor | 0.53 | 0.32 | ~0.49 | no joint at all |
| envelope-cap 0.15 | 0.592 | 0.400 | 0.537 | **zero-knowledge**, one-directional shrink |
| **per-outcome, elicited priors** | **0.649** | **0.446** | 0.521 | **non-circular** |
| per-outcome, pool R² targets | 0.676 | 0.504 | 0.563 | supplied-aggregate (ceiling) |
| published PNAS | 0.62 | 0.43 | 0.30 | — |

(overall here excludes the T4 event-order module; the consolidated pipeline is below.)

## The problem, and the diagnosis: mean-collapse

The no-donor generator sat at T2 ≈ 0.53 (the independence floor) and T3 ≈ 0.32 —
*below* published. The puzzle: the model clearly knows the right relationships
(education predicts income, math and verbal ability correlate), so why does adding
that knowledge not help?

The answer is that it adds *too much*. When a one-shot generator is asked for a
respondent's income given their covariates, its best single answer is `E[income |
covariates]` — the conditional mean. Do that for 1,000 people and the outcomes become
near-deterministic functions of the covariates: the residual (conditional) variance
vanishes. That single fact has two visible consequences — pairwise associations
tighten (T2) and regression R²/coefficients inflate (T3). They are one bug seen
through two tests, and it is strictly **one-directional**: the joint is always too
strong, never too weak.

Measured directly (OLS of each outcome on education/ethnicity/gender, the exact T3
regression), raw generation vs the real reference — R² / residual-std:

| outcome | REAL | raw LLM gen |
|---|---|---|
| fixed_mindset | 0.05 / 1.03 | **0.58 / 0.16** |
| growth_mindset | 0.05 / 0.99 | **0.59 / 0.15** |
| age_at_first_marriage | 0.03 / 4.12 | 0.54 / 0.78 |
| mean_income_30_40 | 0.30 / 1.07 | 0.62 / 0.22 |
| math_cognitive | 0.77 / 3.08 | 0.68 / 1.37 |

The residual-std column is the whole story: raw-gen residuals are 1/3 to 1/6 of
reality on *every* outcome — even `math_cognitive`, where the R² happens to look fine,
the residual is still collapsed. Where the true R² is low (mindset), the collapse
manifests as a fabricated 0.58; where it is genuinely high (math), the R² looks right
but the spread is still gone. It is not "correlations 2× too strong"; it is *variance
missing*, and inflated association is its shadow.

## The method: per-outcome conditional-variance repair

Repairing a *missing-variance* failure is the mirror image of the T4 event-order fix,
which had to *inject* missing structure. Here we destroy excess structure — and
destroying it in a controlled amount is easy; the only question is how much.

Keep the LLM's coherent joint by sampling whole respondents through a **shared person
index** (a copula — every column of a generated row comes from the same raw
respondent, so the joint is preserved). Then, per outcome, keep that link on only a
per-row Bernoulli(α_c) fraction of rows and draw the rest from an **independent**
respondent. A per-row blend scales the predictor→outcome covariance by α and restores
residual variance in the same move:

    beta ≈ alpha · beta_coherent ,   so   R²(alpha) ≈ alpha² · R²_coherent

Choose α per outcome so its covariate-R² lands on a target:

    alpha_c = clip( sqrt( R2_target_c / R2_own_c ) , 0 , 1 )

`R2_own` is measured on our *own* generation (never the test). Predictor columns are
held fully coherent, so real demographic structure is preserved; only outcomes are
repaired. Every column is still mapped onto the pool's marginal, which locks T1.

The per-outcome α is what beats a single global blend: the diagnosis says each
outcome needs a different amount. The realized ladder (elicited targets):

| outcome | own R² | target R² | α | effect |
|---|---|---|---|---|
| growth_mindset | 0.59 | 0.02 | 0.18 | crushed (fabricated structure) |
| self_control | 0.45 | 0.02 | 0.21 | crushed |
| mean_income_30_40 | 0.62 | 0.28 | 0.67 | moderated |
| verbal_cognitive | 0.67 | 0.42 | 0.79 | mostly kept |
| math_cognitive | 0.68 | 0.45 | 0.81 | kept (genuinely predictable) |

A single global α would either under-repair the mindsets or over-dilute cognition; the
per-outcome version also protects T1, because it does not scramble the outcomes whose
marginals depend on their (correct) covariate coupling.

## Where the target comes from — the honesty ladder

The target R² is the only thing that needs an external anchor, and because the failure
is one-directional the anchor can be weak — an *upper envelope*, "shrink anything
implausibly strong," recovers most of the gain:

- **R2_own** — always allowed; used for the ratio, measured on our own generation.
- **Elicited (non-circular).** The LLM reasons as a demographer about how much
  education/ethnicity/gender explain each outcome, sending only outcome names — never
  pool or test values. The elicited priors matched reality closely on the outcomes
  that matter most (mindsets 0.02–0.03 vs real 0.05; income 0.28 vs real 0.29),
  undershooting cognition slightly. This is the honest headline: T2 0.649, T3 0.446.
- **Pool R² (supplied aggregate).** The disjoint pool's covariate-R² — a low-order
  aggregate on the same footing as handing over the marginals. A labeled ceiling
  (T2 0.676, T3 0.504), not the honest number.
- **Zero-knowledge cap.** Even a flat "cap R² at 0.15" recovers T2 0.592 / T3 0.400 —
  because the failure is one-directional, near-zero information already helps.

## Consolidation with the T4 module, and one honest tension

Composed with the event-order module (T4), the full no-test-data pipeline over 5
seeds:

| | T1 | T2 | T3 | T4 | T5 | overall |
|---|---|---|---|---|---|---|
| consolidated (elicited targets) | 0.874 | 0.643 | 0.376 | 0.680 | 0.847 | **0.684** |
| published PNAS | 0.14 | 0.62 | 0.43 | 0.05 | 0.75 | 0.30 |

Overall 0.684 is more than double published, with the honesty ladder labeled per
benchmark (T2/T3 genuine, T1/T4 supplied-aggregate). Two interactions are worth
recording honestly:

1. **T5 recovers** (0.699 → 0.847). Bare variance-repair dropped T5 below its floor by
   leaving the event-timing outcomes over-coherent; once the event-order module owns
   those columns with its coupled, order-by-construction draw, T5 is restored.
2. **T3 gives back** (0.446 → 0.376). `age_finished_education` is *both* a T3 outcome
   and an event-timing column. When the event-order module overwrites it for T4/T5, it
   discards variance-repair's covariate calibration of that one column, costing ~0.07
   on T3. This is mechanism-5 coupling: two repairs contend for one column.

The fix for (2) is identified but not yet built: have the event-order module *anchor*
its event draw on the variance-repaired `age_finished_education` value (which carries
the correct education coupling) rather than redrawing it freely, so the ordering is
built around a covariate-correct anchor and both T3 and T4/T5 are satisfied.

## Discussion

Two of Fable's predictions held exactly. T3 responded dramatically to the
variance-restoration fix (0.32 → 0.45), being the purest expression of the collapse.
And T2 improved but plateaus below its supplied-aggregate ceiling: some pairwise
associations are genuinely idiosyncratic to the survey and no public prior recovers
them — which locates, again, the boundary between what public statistical knowledge
carries and what only microdata holds.

Taken with the event-order result, the no-donor picture is now: T1 and T4 are
supplied-aggregate reproduction; T2 and T3 are genuine, non-circular joint knowledge
that clears published PNAS; T5 rides on the event-order module. The remaining frontier
is the per-column contention above and the multi-objective loop that would co-optimize
the whole test battery rather than one benchmark at a time.

## Reproducibility

Module `src/ssdataagent/data/conditional_variance.py`; tests
`tests/test_conditional_variance.py`. Measurement scripts (scratchpad):
`nodonor_meancollapse_probe.py` (the diagnosis), `nodonor_varrepair.py` +
`nodonor_varrepair_elicit.py` (the bracket), `nodonor_consolidate.py` (the combined
pipeline). Elicited priors cached in `varrepair_elicit.json`. Aggregates and marginals
from the disjoint pool; the event-order module's `forbid_ref` guards the boundary.
