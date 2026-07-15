# No-donor regime — the roadmap to answering PNAS's question

**Date:** 2026-07-15
**Premise:** a model that never sees the test data, but reasons like a data
scientist (knows to reproduce the marginals, reasons about how variables relate),
generating samples good enough to top the PNAS leaderboard. No target microdata —
only the model's knowledge plus published aggregate stats.

## Standing

- **Strategy 1 (donor-based / microdata):** parked as a diagnostic. It proves the
  benchmark is saturated when microdata exists (resampling 0.806, unbeatable) and
  is a privacy method — but it does not answer PNAS's question, because it works
  only by holding the data. See `2026-07-15-benchmark-saturation-and-disclosure.md`.
- **Strategy 2 (no-donor conditioned generator):** **KEEP as a candidate.**
  Current best no-donor method — sample demographic seeds from marginals, LLM
  completes each person, calibrate to marginals. Overall **0.497** (T1 0.807,
  T2 0.534, T3 0.343, T4 0.000, T5 0.802). Beats published 0.30 and the marginal
  floor 0.477. Lives in scratchpad today; productionize into the strategy registry.

## Where the headroom is (no-donor best 0.497 → microdata ceiling 0.806)

| | now | ceiling | gap | recoverable overall |
|---|---|---|---|---|
| T1 marginals | 0.807 | 0.873 | ~0 | — (calibration solved it) |
| T2 pairwise | 0.534 | 0.813 | 0.28 | +0.056 |
| T3 regression | 0.343 | 0.708 | 0.37 | +0.073 |
| **T4 event order** | **0.000** | 0.825 | **0.83** | **+0.165** |
| T5 order×covariate | 0.802 | 0.809 | ~0 | — (solved) |

T4 alone is worth more than T2 and T3 combined, and it sits at zero. The wall is
not uniform — it is mostly one benchmark left on the floor.

## The four possibilities (try one by one)

### A. No-donor event-order module — *start here*
The LLM knows the life-course ordering (education → work → marriage → children)
and typical age gaps. Have it specify that event structure from knowledge; sample
event ages jointly so order holds; calibrate the marginal ages. Targets the 0.165
T4 lever. We already proved T4 is winnable in principle (microdata repair took it
0.000 → 0.81). Sharpest test of "knowledge instead of data."

### B. Elicited full conditionals (LLM-as-modeler)
The LLM articulates conditional relationships it knows well — P(income | education,
age, region) — as functions/tables; assemble a Bayes net; sample; calibrate.
Targets T3 (biggest single-benchmark gap). Uses *full* conditionals, which is why
it could break the ~0.50 wall the *pairwise* oracle could not.

### C. Agentic data-scientist loop — the eventual frame
An LLM agent that generates, checks its output against the **known aggregate stats
+ its own knowledge of plausible relationships** (never the test data), diagnoses
gaps, and revises. This is "act like a data scientist" as a loop, and it can
orchestrate A and B as tools rather than replace them. Most ambitious.

### D. Fix the demographic seed (cheap component)
Today demographic seeds are drawn independently from marginals, so the seed itself
has no joint structure — likely caps T2. Seed instead from published cross-tabs
(age×gender×education). Low effort, plausibly lifts T2; useful inside A–C.

## Order of execution

0. Productionize strategy 2 into the registry (lock the candidate).
1. **A** — event-order module (biggest lever, provably winnable).
2. **B** — full-conditional elicitation (T3).
3. **D** — seed cross-tabs (T2 component; may fold into B).
4. **C** — the agentic loop that ties A/B/D together as tools.

**Discipline (learned the hard way):** measure baseline + ceiling before building,
never tune on the test reference, use the disjoint pool's marginals (not the
benchmark's 1000-row marginals) as the "known aggregate stats" so it stays honest.
