# No-donor event ordering (A): a data-access bracket, and where the real gap is

**Date:** 2026-07-15
**Regime:** no-donor (no target microdata; aggregate stats from the disjoint pool
+ LLM prior). Scored on the paper's 1000-row cfps reference, 5 seeds.

## The question

Can the model's *knowledge* of the life-course reproduce T4 (event ordering)
without microdata? T4 sat at 0.000 for every no-donor generator because event ages
were drawn independently — order random. Module A couples the event ages per
person (anchor + positive gaps, so order holds by construction) from a per-stratum
spec. The spec's ordering distribution is the T4 signal; where it comes from is the
whole question.

## Result — a clean data-access bracket

| the model holds | overall | T4 |
|---|---|---|
| pure LLM knowledge (elicited ordering) | ~0.49 | 0.000 |
| **+ aggregate ordering distribution (disjoint pool)** | **0.640 ±.032** | **0.700 ±.122** |
| full microdata (donor repair) | ~0.79 | 0.81 |

5-seed per-benchmark, strategy-2 baseline → +aggregate-ordering:

| | baseline | +agg-order | Δ |
|---|---|---|---|
| T1 | 0.842±.074 | 0.863±.080 | +0.021 (noise) |
| T2 | 0.536±.004 | 0.531±.007 | −0.005 (noise) |
| T3 | 0.308±.029 | 0.316±.030 | +0.008 (noise) |
| **T4** | 0.000±.000 | **0.700±.122** | **+0.700** |
| T5 | 0.765±.052 | 0.791±.036 | +0.026 (noise) |
| overall | 0.490±.022 | 0.640±.032 | +0.150 |

The intervention moves **only T4**; every other benchmark is unchanged within
noise. The +0.150 overall is entirely the T4 term.

## Two honest halves of the finding

**1. Negative (non-circular): the LLM's prior cannot supply the ordering
distribution.** Elicited from Claude, the ordering prior is far too diffuse —
canonical `edu<marriage<child` at 0.35–0.65 by stratum, versus real 0.91 — so T4
stays at exactly 0.000. A hand-authored fixture (canonical 0.93) also failed,
missing cfps's real minority (`marriage<child<edu`, education completing after
family formation, ~15% among the high-educated). Knowledge alone does not clear
T4's razor tolerance.

**2. Boundary: the aggregate ordering distribution unlocks it — and this is
train/test-clean, not circular.** The distribution is estimated on the **disjoint
pool** (guard enforces pool ≠ test); `dissim(INPUT, REAL) = 0.013` is the
train/test sampling gap. Using population structure learned on training data to
generate, then evaluating on held-out data, is generalization. It is the *same
move as T1*: we hand the generator the marginals (calibration) for T1 and the
ordering distribution for T4, both from the disjoint pool. If T1-via-calibration
is legitimate (it is — the accepted no-donor floor), so is this, one rung up (a
low-order joint aggregate instead of a univariate one).

### The benchmarks split by whether we supplied the aggregate

| | supplied (from disjoint data) | measures |
|---|---|---|
| T1 (0.86), T4 (0.70) | marginals / ordering distribution | reproduction of a provided aggregate |
| T2 (0.53), T3 (0.32) | nothing | the LLM's genuine joint contribution |
| T5 (0.79) | per-stratum ordering (partial) | mixed |

**Echo check** (population eligible-ordering distribution): the generator
reproduces the fed-in ordering to `dissim(treat, INPUT) = 0.027` and lands
`dissim(treat, REAL) = 0.023` from the test — right at T4's tolerance edge, which
is why T4 passes ~70% of bootstrap iterations (and is seed-noisy, σ≈0.12). So the
T4 number is largely "reproduce the provided aggregate," reported as such.

## The real gap is not circularity — it is joint knowledge

On the benchmarks where we supply nothing (T2, T3), the no-donor generator is at
**0.53 / 0.32 — below published PNAS (0.62 / 0.43).** Our overall edge over
published comes entirely from reproducing provided aggregates (T1 via calibration,
T4 via ordering), not from better joint reasoning. On genuine joint knowledge, the
generator is behind the field. That is the honest frontier, and it is what
possibility B (full-conditional elicitation for T3) targets — T3 is never fed its
own answer, so any gain there is genuinely non-circular headroom.

## Honest access-regime reading

The bracket is a statement about **data-access regimes**, not prompt engineering.
Per-stratum ordering distributions are routinely published (census tabulations,
life-table crosstabs, sequence-analysis papers), whereas microdata is
restricted-access. So "public aggregates + LLM prior, no restricted microdata"
maps onto a genuine social-science access regime. Within it: T4 requires the
aggregate ordering to move at all; the LLM's prior alone cannot supply it; and the
gap from 0.64 to the 0.79 microdata ceiling is what per-person joint data buys
beyond published aggregates.

**Never quote 0.64 bare.** It is meaningful only as the middle of the bracket
(0.49 floor, 0.79 ceiling), with T4 labeled as aggregate-ordering access.

## Reproducibility

Module: `src/ssdataagent/data/event_order_knowledge.py` (tested,
`tests/test_event_order_knowledge.py`). Measurement:
`scratchpad/nodonor_eventorder_{agg,final,circ}.py`. Aggregates from the disjoint
pool; `apply_event_order(..., forbid_ref=ref)` guards the boundary.
