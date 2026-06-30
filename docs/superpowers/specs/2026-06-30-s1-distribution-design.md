# S1 distribution diagnostic — design

**Date:** 2026-06-30
**Status:** approved (design)
**Part:** 6 of the strategy-seam refactor (`docs/handoff/design-reference.md` §7 "S1 — Distribution-as-output")
**Precedes:** implementation plan `docs/superpowers/plans/2026-06-30-s1-distribution.md`

## 1. Goal

Add the **S1 distribution diagnostic** arm. S1 does *not* generate cases: the LLM
emits the conditional distribution P(target | demographic cell) per cell, and we
sample N per cell to match the test composition, with targets sampled
**independently** (no copula). Its role is **diagnostic, not contender**: run it
with and without raking to `known_marginals` to isolate *sampling collapse* (fixed
by emitting whole distributions instead of point cases) from *prior collapse* (fixed
only by grounding). S1 is a Type 1/4 instrument and is expected to fail Types 2/3/5
— that is the intended signal. A mixture-of-personas variant tests whether the model
*knows* within-group diversity or just won't express it.

S1 is the lightest arm: almost pure reuse of the Design B machinery, minus the
copula, with raking as a toggle. The over-determination gap is already in every
run's `eval.json` (Part 2), so the raw-vs-raked diagnostic reads out for free.

## 2. Locked design decisions

- **Three variants:** `s1_raw` (elicit → independent sample, NO raking), `s1_raked`
  (elicit → rake each marginal to `known_marginals` → independent sample),
  `s1_personas` (mixture-of-personas → mixture sample).
- **Conditions span A/B/C** via the condition specs; one `s1_raked` strategy spans
  all three because the gate supplies the raking anchor (train / source / withheld).
- **Independent sampling** (identity Σ) for all variants — S1's defining property.
- **raw and raked share the anchored elicitation** (same prompt, same cache); the
  only difference is the post-step (rake or not). The isolated variable is raking
  ("grounding").
- **Personas unraked in v1** (within-group diversity is orthogonal to grounding).
- **Determinism:** persistent elicitation cache + seeded sampling (seed=42); tests
  mock the client. No microdata fit anywhere.
- **Defaults:** K=3 personas per cell; numeric support K=10 bins; cells `n_bins=4`
  (matching Design B's `_N_DEMO_BINS`/`_N_NUMERIC_BINS`).

## 3. Strategies & conditions

New module `src/ssdataagent/strategies/s1.py`. A shared `_S1Base` with two class
flags drives three thin subclasses:

| Strategy class | `name` | `rake` | `personas` |
|---|---|---|---|
| `S1RawStrategy` | `s1_raw` | False | False |
| `S1RakedStrategy` | `s1_raked` | True | False |
| `S1PersonasStrategy` | `s1_personas` | (n/a) | True |

`registry.py` adds `s1_raw`, `s1_raked`, `s1_personas`.

`conditions.py` adds 5 specs (3 strategies, A/B/C spanned by the raked variant):
- `s1_raw` → NO_DATA, strategy `s1_raw` (condition-independent; no grounding).
- `s1_raked_full` → FULL, strategy `s1_raked` (rake to train marginals).
- `s1_raked_transfer` → TRANSFER, strategy `s1_raked` (rake to source marginals).
- `s1_raked_aggregate` → NO_DATA, strategy `s1_raked` (rake to withheld marginals).
- `s1_personas` → NO_DATA, strategy `s1_personas` (condition-independent diagnostic).

Runner: **no change** (strategy-agnostic; the transfer-gate construction already fires
for any `Condition.TRANSFER` spec on a `TRANSFER_PAIRS` dataset). A characterization
test confirms `s1_raked_transfer` receives a source/crosswalk gate.

## 4. `_S1Base.generate` (raw & raked)

Mirrors `DesignBStrategy.generate`, minus the copula:

1. `schema`, `bg = gate.background()`, `known_m = gate.known_marginals() or {}`.
2. `targets = [t for t in schema.target_variables if t in known_m]`; empty → early
   return `background_frame(bg, schema)` with `meta_extras={"backend":"s1",
   "variant":<v>, "n_targets":0, "n_individuals":len(bg)}`.
3. `supports = {t: E.target_support(schema, t, n_numeric_bins=10)}`;
   `known_vecs = {t: E.known_vector(known_m.get(t), supports[t])}`.
4. cells: `scheme = cells.fit_scheme(bg, schema.background_variables, schema, n_bins=4)`;
   `eval_cell_keys = cells.assign(bg, scheme).tolist()`; `unique_cells`, `cell_weights`
   (value counts), `cell_descs = {c: cells.describe_cell(scheme, c)}`.
5. `cell_dists = E.elicit_cell_distributions(gate.client, dataset=…, condition=
   gate.condition.value, cell_descs=…, schema=…, targets=…, supports=…,
   known_vectors=known_vecs, run_dir=…, cache_dir=cfg.results_root/"_elicitation_cache",
   transport=(gate.condition is Condition.TRANSFER))`.
6. Per target: `cell_vectors_t = {c: cell_dists[c][t]}`. If `self.rake`,
   `calibrated[c][t] = rake(cell_vectors_t, cell_weights, known_vecs[t])[c]`; else
   `calibrated[c][t] = cell_vectors_t[c]` (raw, unraked).
7. Sample independently: `Sigma = np.eye(len(targets))`; `drawn =
   design_b.sample_targets(eval_cell_keys, calibrated, supports, Sigma, targets, seed=42)`.
8. Assemble via `background_frame` + `clip_decode`; write `fit_summary.json`; return
   `StrategyResult`.

S1 reuses Design B's elicitation cache — `s1_raw` and `s1_raked_aggregate` (same
dataset/condition/targets) share cache entries; only the rake step differs.

## 5. Mixture-of-personas (`s1_personas`, the net-new piece)

New functions in `s1.py`:

```python
def elicit_cell_personas(client, *, dataset, condition, cell_descs, schema, targets,
                         supports, known_vectors, run_dir, cache_dir, n_personas=3,
                         max_retries=3) -> dict[str, list[dict]]:
    """Per cell, the LLM returns K latent subtypes, each a weight + a full per-target
    distribution. Returns {cell_key: [{"weight": float, "dists": {t: np.ndarray}}, ...]}.
    Strict-JSON parse; weights normalized; each dist normalized to its support;
    fallback to a single subtype {weight:1, dists: known_vectors} on parse failure.
    Persistent cache (key includes n_personas + a personas prompt-version) + raw-I/O log."""

def sample_personas(eval_cell_keys, cell_personas, supports, targets, *, seed=42) -> dict:
    """Per eval row: pick a subtype ∝ its cell's weights, then sample each target from
    that subtype's distribution (independent within the subtype): searchsorted on the
    cumulative dist for the chosen subtype; categorical → support member, numeric →
    uniform within the chosen even-width bin. Returns {t: list-of-values}."""
```

Persona JSON shape requested from the LLM:
`{"subtypes": [{"weight": 0.5, "dists": {"<target>": [p1, p2, ...]}}, ...]}`.
Validation: keep ≤ `n_personas` subtypes; drop subtypes missing any target; if none
survive, fall back to the single known-vector subtype; renormalize weights to sum 1;
renormalize each dist to its support (length-checked) else replace with the target's
known vector. `S1PersonasStrategy.generate` follows `_S1Base` steps 1-4, then calls
`elicit_cell_personas` and `sample_personas` (no rake), assembles, writes
`fit_summary.json` with `n_personas`.

## 6. Determinism, leakage, artifacts

- **Determinism:** no microdata fit; persistent caches (elicitation / personas) +
  seeded sampling (seed=42). Tests mock the client; a cache hit means zero client calls.
- **Leakage:** raked/personas read only `known_marginals` via the gate — under
  TRANSFER that is `source[crosswalk]` marginals, never the target survey's targets
  (structural). raw reads marginals only as a prompt anchor (no calibration).
- **Artifacts:** `fit_summary.json` = `{backend:"s1", variant:"raw"|"raked"|"personas",
  condition, raked: bool, n_cells, n_targets}` (+ `n_personas` for personas);
  elicitation raw I/O under `run_dir/elicitation/` (and personas under the same).

## 7. Out of scope (v1)

- Raking the persona mixture (kept unraked — diversity is orthogonal to grounding).
- Any copula / cross-target coupling (S1 is deliberately independent — its diagnostic role).
- An anchor-free raw variant (raw shares the anchored elicitation; the isolated
  variable is raking). Notable but deferred.
- Sequence targets (Types 4/5 stay agent-only per the handoff).

## 8. Testing strategy (client mocked)

- `tests/test_s1_raw_raked.py` — `_S1Base` end-to-end for raw and raked on GSS:
  all targets present, row count, identity-Σ independence path runs; cache hit = zero
  client calls; determinism (two runs identical). For the "raked is closer to
  `known_marginals`" assertion, the mock client MUST return a per-cell distribution
  that *differs* from the anchor (a skewed prob vector) — an empty-`{}` mock falls back
  to the known-vector, making raw and raked identical and the assertion vacuous.
- `tests/test_s1_personas.py` — `elicit_cell_personas` parse/normalize/fallback;
  `sample_personas` returns valid support values, subtype weights respected in
  expectation, deterministic.
- `tests/test_strategy_s1.py` — the three strategy classes register the right `name`,
  produce valid frames in their conditions, and `s1_raked_transfer` reads source-only
  (no-leakage: a poisoned target-train target never appears via the raked anchor).
- Extend `tests/test_strategies_registry.py`, `tests/test_conditions.py`,
  `tests/test_runner_artifacts.py` (transfer-gate characterization; the two P0
  byte-stable tests stay green).

## 9. Gate (per `feedback_refactor_gate_philosophy`)

Our tests pass + no NEW failures vs. the 4 pre-existing `autograd`-missing failures
(`tests/test_config.py::test_unknown_provider_raises` + 3
`tests/test_ssdatabench_integration.py *_legacy`). No bit-for-bit reproduction gate.
