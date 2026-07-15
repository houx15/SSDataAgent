# No-donor event-order module (A) — implementation plan

> **For agentic workers:** execute task-by-task, test-first. Steps use `- [ ]`.

**Goal:** lift the no-donor generator's T4 off 0.000 by coupling life-course event
ages per person (order by construction) from an LLM-estimated per-stratum spec.

**Spec:** `docs/superpowers/specs/2026-07-15-no-donor-event-order-module-design.md`.

**Architecture:** a knowledge-based sibling of the donor-based
`data/event_timing.py`. New module `src/ssdataagent/data/event_order_knowledge.py`.
The LLM emits one compact spec per demographic stratum (ordering distribution +
gaps + occurrence structure); a deterministic sampler reconstructs each person's
event ages via anchor + positive gaps (so order holds); occurrence rates and the
anchor marginal are calibrated to the disjoint pool; the block replaces only the
T4-scored columns in strategy 2's output.

## Global Constraints

- **Honest boundary:** aggregates come only from the disjoint pool, never the
  1000-row test reference. The ordering distribution comes only from the LLM.
- Reuse `event_timing.event_timing_variables(dataset)` for the exact event set.
- Sentinels are per-column (`never married`, `never had child`, `never worked`,
  `still alive`, …); non-occurring events emit the sentinel, not NaN, matching the
  real dtype.
- No test-tuning; measure vs baseline (strategy 2: T4 0.000, overall ~0.49).
- Tests: no NEW failures beyond the 4 pre-existing baseline failures.

---

### Task 1: Spec type + hand-authored fixture

**Files:** Create `src/ssdataagent/data/event_order_knowledge.py`;
Test `tests/test_event_order_knowledge.py`.

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class StratumEventSpec:
    ordering: dict[str, float]              # "edu<work<marriage<child" -> prob (sums~1)
    gaps: dict[str, tuple[float, float]]    # "edu->work" -> (mean_years, sd_years), >0
    occurrence: dict[str, float]            # event -> P(occurs), overridden by calibration
    requires: dict[str, str]                # event -> prerequisite event (e.g. divorce->marriage)

def fixture_specs(dataset: str) -> dict[tuple, StratumEventSpec]:
    """Hand-authored cfps specs keyed by (gender, edu_bucket) — life-course common
    sense, no LLM. Lets every downstream piece be built and tested LLM-free."""
```

- [ ] **Step 1 — failing test:** assert `fixture_specs("cfps")` returns ≥4 strata,
  each `ordering` sums to ~1.0 (±1e-6), all `gaps` means > 0, `requires` maps
  divorce→marriage.
- [ ] **Step 2:** run, verify fail (module missing).
- [ ] **Step 3:** implement the dataclass + `fixture_specs` (canonical order 0.94,
  child-before-marriage ~0.03, etc.; strata = gender × {low,high} education).
- [ ] **Step 4:** run, verify pass.
- [ ] **Step 5:** commit.

---

### Task 2: Deterministic sampler (order by construction)

**Files:** Modify `event_order_knowledge.py`; Test same file.

**Interfaces — Produces:**
```python
def sample_event_block(
    demographics: pd.DataFrame,      # has gender + an education column
    specs: dict[tuple, StratumEventSpec],
    pool: pd.DataFrame,              # disjoint pool — anchor marginal + occurrence
    event_vars: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:                   # one column per event_var, ordered ages / sentinels
```
Per person: map to stratum → draw ordering from `spec.ordering` → draw occurrence
per event (rate from pool aggregate, gated by `requires`) → for occurring events,
draw anchor age from the pool marginal of the person's first event, add positive
gaps (truncated > 0) from `spec.gaps`, cumulate → ages in the drawn order.
Non-occurring events get the column's sentinel.

- [ ] **Step 1 — failing tests:** with `fixture_specs` + a synthetic pool: (a) for
  every row, the *occurring* core events are strictly increasing in the drawn
  order (100%); (b) `requires` never violated — no `age_at_first_divorce` numeric
  where `age_at_first_marriage` is the sentinel; (c) per-event occurrence rate is
  within 0.05 of the pool rate.
- [ ] **Step 2:** run, verify fail.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run, verify pass.
- [ ] **Step 5:** commit.

---

### Task 3: Order-preserving marginal calibration

**Files:** Modify `event_order_knowledge.py`; Test same file.

**Interfaces — Produces:**
```python
def calibrate_event_block(block, pool, event_vars, rng) -> pd.DataFrame
```
Tighten each event's *anchor-driven* age marginal toward the pool without
reordering: calibrate via a per-event monotone (rank→pool inverse-CDF) applied
**only where it does not invert a within-person pair** — concretely, recompute
ages as `calibrated_anchor + original_positive_gaps` so order is preserved by
construction and the anchor marginal matches the pool. Occurrence untouched
(already calibrated in Task 2).

- [ ] **Step 1 — failing tests:** after calibration, (a) 100% of rows still respect
  their event order; (b) the anchor (earliest) event's marginal mean is within 1
  year of the pool's.
- [ ] **Step 2:** run, verify fail.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run, verify pass.
- [ ] **Step 5:** commit.

---

### Task 4: Integration + data-boundary guard

**Files:** Modify `event_order_knowledge.py`; Test same file.

**Interfaces — Produces:**
```python
def apply_event_order(
    frame: pd.DataFrame, dataset: str,
    specs: dict[tuple, StratumEventSpec], pool: pd.DataFrame,
    rng: np.random.Generator, *, forbid_ref: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Copy of `frame` with the T4 event columns replaced by the sampled+calibrated
    block. Uses event_timing_variables(dataset). Raises if `pool` is `forbid_ref`
    (identity/row-hash check) — the honest-boundary guard."""
```

- [ ] **Step 1 — failing tests:** (a) only the `event_timing_variables` columns
  change, all others identical; (b) passing the test reference as `pool` with
  `forbid_ref=ref` raises `ValueError`.
- [ ] **Step 2:** run, verify fail.
- [ ] **Step 3:** implement (row-hash overlap check for the guard).
- [ ] **Step 4:** run, verify pass.
- [ ] **Step 5:** commit.

---

### Task 5: LLM elicitation (isolated, cached)

**Files:** Modify `event_order_knowledge.py`; Test `tests/test_event_order_knowledge.py`.

**Interfaces — Produces:**
```python
def parse_stratum_response(text: str) -> StratumEventSpec   # tolerant JSON parse
def elicit_stratum_specs(dataset, strata, descriptions, client,
                         cache_path) -> dict[tuple, StratumEventSpec]
```
One chat call per stratum; parse to `StratumEventSpec`; cache the raw JSON so
re-scoring needs no LLM calls (mirrors `nodonor_cond_raw.csv`). Only
`parse_stratum_response` is unit-tested (no live call).

- [ ] **Step 1 — failing test:** `parse_stratum_response` on a canned JSON string
  returns a valid spec (ordering renormalised to 1.0, gaps floored > 0).
- [ ] **Step 2:** run, verify fail.
- [ ] **Step 3:** implement parser + elicitation loop.
- [ ] **Step 4:** run, verify pass.
- [ ] **Step 5:** commit.

---

### Task 6: Measurement runner (scratchpad) + verdict

**Files:** Create `scratchpad/nodonor_eventorder.py` (elicit-or-fixture → apply →
score); no src changes.

- [ ] **Step 1:** load strategy-2 cached output (`nodonor_cond_raw.csv`), build the
  demographics frame, elicit specs for cfps (real LLM, cached) — or fall back to
  `fixture_specs` if the run is offline.
- [ ] **Step 2:** `apply_event_order` → calibrate the full frame (reuse
  `nodonor_score2.calibrate` for non-event columns) → score 3 seeds vs the 1000-row
  reference, disjoint pool for aggregates.
- [ ] **Step 3:** print per-benchmark means; compare T4 and overall to baseline
  (T4 0.000, overall ~0.49).
- [ ] **Step 4 — verdict:** if T4 meaningfully > 0, write the result into the
  roadmap report and, once productionised with the tests above, add a
  `production/README.md` candidate/entry. If T4 stays ~0, record the negative
  finding (LLM ordering estimate too imprecise for T4's tolerance).

---

## Self-review

- Coverage: spec's sampler, calibration, integration, elicitation, boundary guard,
  measurement all have tasks. ✓
- Types: `StratumEventSpec` defined in Task 1, consumed unchanged in 2–5. ✓
- No placeholders: each task has concrete test assertions and interfaces. ✓
