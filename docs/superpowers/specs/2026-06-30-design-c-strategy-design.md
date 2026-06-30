# Design C strategy — retrieval-grounded generation with distributional repair

**Date:** 2026-06-30
**Status:** approved (design)
**Part:** 4 of the strategy-seam refactor (`docs/handoff/delta-plan.md` P3)
**Precedes:** implementation plan `docs/superpowers/plans/2026-06-30-design-c-strategy.md`

## 1. Goal

Add `design_c`, the **data-first** grounded generator: for each test row, retrieve
*k* nearest real individuals by background, **hot-deck** a target vector (which
preserves the real multivariate joint by construction), and run a **repair loop**
that reweights the donor draws until the synthetic aggregate matches the goal
marginals. In transfer (condition B) the LLM **transports** the borrowed target
distribution to the new population's context — used surgically on marginals only,
never for free per-case guessing.

Design C is `design-reference.md` §7's third core arm. It is *strongest in B
(transfer)* and *acknowledged-weak in C (aggregate-only)*, where there are no real
exemplars to retrieve and the pipeline degenerates to ~independent marginal
sampling. That weak data point is itself a finding (resampling can't manufacture a
joint the data never had).

## 2. Locked design decisions

- **Conditions:** A + B + C (full parity with Design B).
- **LLM role:** distributional **transport in B only** — the LLM emits per-cell
  target-context marginals; the multivariate joint stays real-donor-grounded.
  No LLM in A or C.
- **Repair:** raking over donor weights (IPF idea) + per-row candidate resampling
  (SIR), deterministic given seed. No LLM "diagnose which dimension is off" in v1.
- **C pool bootstrap:** synthetic donor pool whose backgrounds are the eval
  backgrounds and whose targets are independent draws from `known_marginals`
  (via the Design B support representation). Retrieval against this pool then
  reduces to marginal sampling — the honest weak case.
- **Defaults:** `k = 10` (matches the hot-deck baseline), repair `max_iter = 50`
  (matches `design_b.rake`), `seed = 42`.

## 3. Uniform target representation (reused from Design B)

Every target is a probability vector over a fixed support:
- categorical → `schema.allowed_values[t]`;
- numeric → `K = 10` even-width bins over `schema.numeric_ranges[t]`.

Reuse `elicitation.target_support`, `elicitation.known_vector`. Goal marginals
`q_t` (the repair target) are these prob-vectors: `known_vector(known_marginals[t])`
in A/C, the LLM-transported vector in B. A donor's target value maps to a support
index via the same binning (`np.searchsorted` on numeric edges; category index for
categorical) so donor draws and goal marginals live in one space.

## 4. Per-condition pipeline

All three share: retrieve → hot-deck candidate sets → repair → per-row draw →
assemble. They differ only in the retrieval source and the goal marginal.

- **A (FULL).** Retrieve k-NN from `gate.fit_microdata()` (= `train`) by background.
  Goal marginal `q_t = known_vector(known_marginals[t])` (train marginals). No LLM.
- **B (TRANSFER).** Retrieve k-NN from `gate.fit_microdata()` (= `source[crosswalk]`)
  by background — candidates carry the *source's* real joint. The LLM transports a
  **single population-level marginal per target** (one elicitation over the target
  population's context, anchored on the source marginal) via
  `elicitation.elicit_cell_distributions(..., transport=True)` with one
  population-level "cell"; `q_t` is that transported vector. Retrieval supplies the
  conditional/joint structure; the LLM supplies only the marginal shift; the global
  repair imposes it without flattening the retrieved conditionals.
- **C (NO_DATA).** `gate.fit_microdata()` is `None`. Bootstrap a synthetic donor
  pool: backgrounds = eval backgrounds; each donor's targets = independent draws
  from `known_marginals` (sample a support index per target ∝ `known_vector`,
  numeric → uniform within the bin). Retrieve+hot-deck+repair against this pool.
  Result ≈ independent marginal draws (no real joint to borrow). Must NOT raise.

## 5. Components (one new file + one reused helper)

New module `src/ssdataagent/strategies/design_c.py`:

```python
def retrieve_candidates(donors, background, schema, *, k=10) -> np.ndarray:
    """k-NN donor indices per eval row, by background. Reuses
    baselines.encode_numeric for shared train/eval scaling.
    Returns an (n_eval, k_eff) int array of donor-row indices."""

def bootstrap_pool(background, known_marginals, supports, schema, *, seed=42) -> pd.DataFrame:
    """Condition-C synthetic donor pool: eval backgrounds + targets drawn
    independently from known_marginals. Returns a donor frame with background
    + target columns."""

def repair_weights(neighbor_idx, donor_codes, goal_vectors, supports, targets,
                   *, max_iter=50, tol=1e-6) -> np.ndarray:
    """Global non-negative donor weights, raked so each eval row's
    candidate-weighted target marginal matches goal_vectors[t]. donor_codes[t]
    is each donor's support index for target t. Fixed IPF passes scaling weights
    by q_t(bin)/m_t(bin) per target. Returns the donor weight vector."""

def draw_targets(neighbor_idx, donor_codes, weights, donors, targets, *, seed=42) -> dict:
    """Each eval row samples one donor from its candidates ∝ weights; emit that
    donor's actual target values. Categorical -> value; numeric -> donor value
    (already a real observation, no re-uniforming needed)."""

class DesignCStrategy:  # name = "design_c"
    def generate(self, gate, run_dir, cfg) -> StrategyResult: ...
```

`DesignCStrategy.generate` orchestration (mirrors `DesignBStrategy`):
schema/bg/`fit_microdata`/`known_marginals` → target set = schema targets in
`known_marginals` (empty-target early return like Design B) → supports → donor
frame (pool bootstrap in C, else `fit_microdata`) → `retrieve_candidates` →
encode donor target values to support indices → goal vectors (single transported
vector per target via `elicit_cell_distributions` with one population-level cell in
B, else `known_vector`) → `repair_weights` → `draw_targets` → assemble via
`background_frame` + `clip_decode` → write `fit_summary.json` → `StrategyResult`.

The goal marginal `q_t` is a **single per-target vector** in all conditions —
`known_vector(known_marginals[t])` in A/C, the population-level transported vector
in B. Design C does not partition into demographic cells; the conditional structure
comes from retrieval, not from per-cell elicitation. (This is the key contrast with
Design B, which rakes per cell.)

## 6. Repair loop math (the one net-new mechanism)

Donors `d = 1..D` carry weights `w_d ≥ 0` (init 1). Eval row *i* has neighbor set
`N_i` (k donors); its candidate-selection probs are `w` normalized within `N_i`.
The induced aggregate marginal for target *t*, bin *b*:

```
m_t(b) = (1/n) · Σ_i  Σ_{d∈N_i} [w_d / Σ_{d'∈N_i} w_{d'}] · 1[code_t(d) == b]
```

Each IPF pass, for every target *t* and bin *b*, scale `w_d` for donors with
`code_t(d) == b` by `q_t(b) / m_t(b)` (guarded for `m_t(b) == 0`), then renormalize
and recompute. Loop targets for `max_iter` passes or until
`max_t ||m_t − q_t||_∞ < tol`. Final per-row draw is ∝ `w` within `N_i`.

This preserves each eval row's own background, corrects marginals, and keeps every
emitted target a **real donor vector** — Design C's thesis vs. the hot-deck
baseline (which corrects nothing) and vs. Design B (which samples synthetic
marginals + a copula joint). Property test: post-repair `||m_t − q_t||₁` is
strictly smaller than pre-repair for a constructed mismatch.

## 7. Registration & runner

- `registry.py`: `"design_c": DesignCStrategy`.
- `conditions.py`: `design_c_full` (FULL), `design_c_transfer` (TRANSFER),
  `design_c_aggregate` (NO_DATA) — all `strategy="design_c"`.
- Runner: **no change** — `_run_one_condition` already builds the transfer InfoGate
  for any `Condition.TRANSFER` spec on a `TRANSFER_PAIRS` dataset (runner.py:173),
  strategy-agnostic. A characterization test confirms `design_c_transfer` receives
  a gate with `source`/`crosswalk` populated.

## 8. Determinism, leakage, artifacts

- **Determinism:** `repair_weights` is a pure deterministic IPF (no RNG). Seeded
  RNG (`seed=42`) is used only for the C pool bootstrap and the final per-row donor
  draw. LLM transport reproducible via the existing persistent elicitation cache
  (keyed dataset/condition/model/cell/targets/prompt-version); tests mock the
  client.
- **Leakage:** B never reads target-survey targets — retrieval source and goal
  marginals are both source-only, structurally enforced by
  `gate.fit_microdata()` / `gate.known_marginals()` returning `source[crosswalk]`
  under TRANSFER. A no-leakage test asserts the generated targets are reachable
  only from source donors + transported marginals.
- **Artifacts:** `fit_summary.json` = `{backend, condition, k, n_donors, n_targets,
  repair_iters, transport: bool}`; B additionally writes the elicitation raw I/O
  under `run_dir/elicitation/` (free from `elicit_cell_distributions`).

## 9. Out of scope (v1)

- LLM "diagnose which dimension is off" in the repair loop (mechanical raking only).
- Seeding the C pool from `known_associations` (unsigned — same limitation that
  forced Design B's identity copula in C; pool stays independent-marginal).
- Sequence targets (Types 4/5) — Designs A/B/C remain tabular-first per the handoff.
- Memorization/privacy auditing of borrowed donors (noted as a watch item; not v1).

## 10. Testing strategy

Mirror Design B's test layout, client mocked:
- `tests/test_design_c_retrieve.py` — k-NN returns k_eff neighbors; shared scaling.
- `tests/test_design_c_repair.py` — `repair_weights` reduces marginal distance;
  converges; handles zero-mass bins.
- `tests/test_design_c_pool.py` — C pool bootstrap matches `known_marginals` in
  expectation; backgrounds = eval backgrounds.
- `tests/test_strategy_design_c.py` — end-to-end A/B/C: shape, no-raise in C,
  no-leakage in B, determinism (two runs identical), cache hit = zero client calls.
- Extend `tests/test_strategies_registry.py`, `tests/test_conditions.py`,
  `tests/test_runner_artifacts.py` (transfer-gate characterization; the two P0
  byte-stable tests stay green).

## 11. Gate (per `feedback_refactor_gate_philosophy`)

Our tests pass + no new failures vs. the pre-existing baseline set (the 4
`autograd`-missing failures: `test_config.py::test_unknown_provider_raises` + 3
`test_ssdatabench_integration.py *_legacy`). No bit-for-bit reproduction gate.
