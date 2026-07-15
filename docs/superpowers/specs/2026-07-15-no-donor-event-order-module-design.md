# No-donor event-order module (A) — design spec

**Date:** 2026-07-15
**Regime:** no-donor (no target microdata; aggregate stats + LLM prior only).
**Goal:** lift the no-donor generator's T4 (event-order) off 0.000 using the
LLM's life-course knowledge, *coupling the event ages per person so order holds by
construction* and *reproducing the ordering distribution from knowledge*.

## Why this is the lever

Strategy 2 scores T4 = 0.000 because it draws each event age independently per
column → per-person order is random (child before marriage ~30–40% vs real ~2%),
and the chi-square rejects every bootstrap iteration. T4 is worth +0.165 of
overall headroom — more than T2 and T3 combined. See
`docs/report/2026-07-15-no-donor-roadmap.md`.

## Honest data boundary (a hard constraint)

- **Allowed (aggregate stats a data scientist has):** the disjoint pool's
  per-event *marginal* age distributions and *occurrence rates*; demographic
  marginals/cross-tabs. Drawn via the disjoint pool, never the 1000-row test
  reference.
- **Allowed (model knowledge):** the LLM's prior over life-course *joint*
  structure — ordering distribution and age gaps.
- **Forbidden:** any test microdata; any per-person real record; the ordering
  *distribution* as a measured statistic (that is the T4 answer — it must come
  from the LLM, per the design decision below).

A test/guard asserts the module never reads the test reference.

## Design decisions (locked with user)

1. **Hybrid strata + sampler.** The LLM emits a compact spec *per demographic
   stratum*, once; we sample every person deterministically from the matching
   stratum's spec. Not per-person LLM calls.
2. **LLM estimates the ordering distribution from knowledge** (not calibrated to
   pool ordering rates). If pure knowledge clears T4, the model genuinely knew the
   joint; if not, that is the finding.
3. **Defaults (vetoable):** stratum = `gender × highest_education` (~6 cells;
   birth-cohort is a dial we add only if timing needs it). Occurrence *rates* and
   event-age *marginals* are calibrated to the pool; ordering + gaps are the LLM's.

## Scope — the event columns

Core T4 events (canonical order): `age_finished_education` <
`age_started_work` < `age_at_first_marriage` < `age_at_first_child`.
Also handled with occurrence constraints and calibrated marginals (not the T4
focus): `age_at_first_divorce` (requires marriage), `age_exited_work` (requires
started work), `age_at_death` (usually the `still alive` sentinel).

## Architecture

### 1. Per-stratum spec (LLM output, once)

For each stratum the LLM returns, from knowledge:

```
{
  "ordering_distribution": {           # over permutations of the core events
     "edu<work<marriage<child": 0.94,
     "edu<work<child<marriage": 0.03,  # the realistic out-of-order minority
     "work<edu<...": 0.02, ...
  },
  "gaps": {                            # gap between consecutive events, in years
     "edu->work":     {"mean": 1.5, "sd": 2.0},
     "work->marriage":{"mean": 4.0, "sd": 3.5},
     "marriage->child":{"mean": 2.0, "sd": 2.0}
  },
  "occurrence_conditional": {          # structure only; RATES calibrated to pool
     "child_requires_marriage": false, # the ~2% path exists
     "divorce_requires_marriage": true
  }
}
```

Elicited via the existing OpenRouter client (`api_key_env`), parsed to a typed
`StratumEventSpec`. One call per stratum (~6), cached to a JSON artifact so
re-scoring needs no LLM calls (as `nodonor_cond_raw.csv` already does).

### 2. Deterministic sampler (per person)

Given a person's stratum (from strategy 2's demographics):

1. Draw an **ordering** from the stratum's `ordering_distribution`.
2. Draw an **anchor age** (first event in the realized order) from the pool's
   marginal for that event.
3. Draw **positive gaps** from the stratum's gap distributions (truncated at a
   small floor > 0) and cumulate → each event age, **in the drawn order**. Order
   is preserved by construction.
4. Draw **occurrence** per event; non-occurring events get the sentinel string
   (`never married`, …) or NaN, honoring the conditional constraints.

### 3. Order-preserving calibration

Naive per-column calibration reorders events and re-zeros T4. Instead:

- Calibrate the **anchor age** and each **gap** to the pool's aggregate anchor /
  gap distributions (rank → target inverse-CDF). Because gaps stay positive, the
  reconstructed ages keep their order regardless of calibration.
- Calibrate **occurrence rates** per event to the pool's aggregate rate.
- Event-age marginals then match the pool approximately (sums of calibrated
  anchor + gaps); accept the small residual, or add a final *monotone-per-event*
  refinement only if T1 on the event ages regresses.

### 4. Integration with strategy 2

This module **replaces only the event block** of the no-donor generator's output.
Demographics and non-event traits keep coming from the conditioned generator +
existing calibration; the seven event-age columns are overwritten by this module's
output. A single `apply_event_order(frame, specs, pool)` step, opt-in per run.

## Interfaces (indicative)

```python
@dataclass
class StratumEventSpec:
    ordering: dict[str, float]        # permutation label -> prob
    gaps: dict[str, tuple[float, float]]   # "a->b" -> (mean, sd)
    occurrence_conditional: dict[str, bool]

def elicit_stratum_specs(strata, descriptions, client) -> dict[stratum, StratumEventSpec]
def sample_event_block(demographics, specs, pool, rng) -> pd.DataFrame  # 7 event cols
def calibrate_event_block(block, pool, rng) -> pd.DataFrame             # order-preserving
def apply_event_order(frame, specs, pool, rng) -> pd.DataFrame          # integration step
```

## Testing

- **Order by construction:** sampled core-event ages respect the drawn ordering
  for 100% of rows (positive-gap invariant).
- **Ordering distribution:** over many samples, realized permutation frequencies
  match the input spec within tolerance.
- **Occurrence calibration:** per-event occurrence rate matches the pool aggregate
  within tolerance; conditional constraints never violated (no divorce without
  marriage).
- **Calibration keeps order:** after `calibrate_event_block`, 100% of rows still
  respect their ordering.
- **Data-boundary guard:** the module errors if handed the test reference as its
  aggregate source.
- **Golden:** a tiny hand-built 2-stratum spec produces the expected ordered ages.

## Measurement plan

- Baseline = strategy 2 today: T4 **0.000**, overall ~0.49 (re-verified 2026-07-15).
- Protocol: disjoint pool for all aggregates, scored on the 1000-row reference,
  3–5 seeds, 5000 simulated rows.
- **Win condition:** T4 meaningfully > 0. T4 ≈ 0.4–0.6 adds ~0.08–0.12 overall;
  matching the microdata event-timing repair (~0.8) is the strong result.
- Report per-benchmark means; promote to `production/` only if it clears the
  criteria there.

## Risks

- **T4's razor tolerance may reject the LLM's ordering estimate** even with order
  correct — that is the accepted risk of the "pure knowledge" choice, and a
  negative result is itself reportable ("the model doesn't know the ordering
  distribution precisely enough").
- **Gap-sum marginals** may not match the pool tightly enough for T1 on event
  ages; mitigated by the optional monotone-per-event refinement.
- **Cohort effects** (older cohorts married younger) may need birth-cohort in the
  strata; left as a dial, added only if timing is visibly off.
