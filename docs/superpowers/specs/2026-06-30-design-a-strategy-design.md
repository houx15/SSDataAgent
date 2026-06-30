# Design A strategy — structure-first LLM-elicited hierarchical generator

**Date:** 2026-06-30
**Status:** approved (design)
**Part:** 5 of the strategy-seam refactor (`docs/handoff/delta-plan.md` P4)
**Precedes:** implementation plan `docs/superpowers/plans/2026-06-30-design-a-strategy.md`

## 1. Goal

Add `design_a`, the **structure-first** grounded generator. The LLM proposes a DAG
over the target variables (with background variables as exogenous parents), per-node
functional forms, and deliberately **wide priors** on attitude nodes. Each node is a
per-target Bayesian GLM (MAP / conjugate fit). We walk the DAG in topological order
and sample each target from its **full conditional** (not its conditional mean),
conditioned on test backgrounds and already-sampled parents.

Design A is `design-reference.md` §7's first core arm. *Fixes:* the joint via the DAG,
over-determination via wide priors + sampling the full conditional, sparse cells via
shrinkage. *Strongest in:* C and B. *Cost:* highest of the three — but realized here
without a heavy MCMC dependency.

## 2. Locked design decisions

- **Inference backend: in-house lightweight.** No PyMC/NumPyro/JAX. Per-node GLMs as
  MAP-under-Gaussian-prior using libraries already in the project:
  - numeric target → `sklearn.linear_model.BayesianRidge` (conjugate Bayesian linear
    regression; `predict(X, return_std=True)` gives the predictive mean **and std** we
    sample from).
  - categorical target → `sklearn.linear_model.LogisticRegression` (multinomial; the
    inverse-regularization `C` maps the LLM's wide-prior scale to a Gaussian prior on
    coefficients; `predict_proba` → sample a category).
- **DAG: constrained.** Background variables are exogenous roots (observed, conditioned
  on, never modeled). The LLM proposes only (a) a topological **order** over the target
  variables and (b) each target's **parents** ⊆ {backgrounds} ∪ {targets earlier in the
  order}. Acyclic by construction; the LLM cannot produce a cycle.
- **Partial pooling (B):** apply the LLM's per-target **offset** fully as an additive
  shift to the linear predictor (the offset *is* the shrunken source→target population
  difference). No separate pooling weight in v1.
- **Intercept calibration:** runs **only in C** (A/B rely on the data fit as grounding).
- **Determinism:** MAP/conjugate fits are deterministic; seeded RNG (`seed=42`) for
  predictive sampling; structure elicitation reproducible via the persistent cache.
- **Defaults:** prior scale 1.0 (wider for attitudes if the LLM says so), seed 42.

## 3. Structure elicitation (LLM, cached + logged)

New `elicit_structure(client, *, dataset, condition, schema, targets, backgrounds,
run_dir, cache_dir, transport=False) -> Structure`. Prompts once per (dataset,
condition, model) for strict JSON:

```json
{"order": ["t1","t2", ...],
 "parents": {"t1": ["age","educ"], "t2": ["age","t1"], ...},
 "prior_scale": {"t1": 1.0, ...},
 "offsets": {"t1": 0.0, ...}}
```

- `order`: topological order over the target variables.
- `parents[t]`: subset of {backgrounds} ∪ {targets before `t` in `order`}.
- `prior_scale[t]`: prior width (>1 = wider = less regularized; attitude nodes wide).
- `offsets[t]`: condition-B only — population-level additive shift in target units,
  applied to **numeric** nodes only (default 0). Categorical nodes ignore offsets in v1
  (a scalar logit shift is ill-defined across classes); condition-B categorical targets
  rely on the source fit. The LLM is told offsets are numeric-only.

Persistent cache keyed (dataset, condition, model, sorted-targets, prompt-version);
raw prompt/response logged under `run_dir/structure/`. **Validation/fallback** (never
crash on LLM output): prune each `parents[t]` to the legal set; append any missing
targets to `order`; default `prior_scale`→1.0, `offsets`→0.0; on any parse failure use
order = schema target order, parents = all backgrounds, scale 1.0, offsets 0.0.

## 4. Per-node model

Parent design matrix `X` built with `baselines.encode_numeric` over the node's parents
(backgrounds + earlier targets), fitting scaling on the training frame and reusing it
for the test/eval frame. A node with no parents uses an intercept-only model.

- **Numeric node (A/B):** `BayesianRidge().fit(X_train, y_train)`; prior width set from
  `prior_scale` (via `lambda_init`/`alpha_init` so a wider scale = weaker coefficient
  shrinkage). Predict `(mu, sd) = model.predict(X_eval, return_std=True)`; draw
  `y ~ Normal(mu_i + offset, sd_i)` per row.
- **Categorical node (A/B):** `LogisticRegression(multi_class="multinomial",
  C=prior_scale)` fit on `(X_train, y_train)`; `P = predict_proba(X_eval)`; sample a
  class ∝ P_i (offsets not applied — numeric-only in v1). Classes restricted to
  `schema.allowed_values[t]`; a degenerate single-class column falls back to that class.

## 5. Per-condition pipeline

Walk `order`; for each target, build `X` from its parents (backgrounds from the frame +
earlier targets already sampled for the eval rows), fit/construct the node, sample.

- **A (FULL):** fit each node on `gate.fit_microdata()` (= `train`). Sample for eval rows.
- **B (TRANSFER):** fit each node on `gate.fit_microdata()` (= `source[crosswalk]`); apply
  the LLM `offset` to the linear predictor; sample for target test backgrounds. Fit only
  on source → structurally no target leakage.
- **C (NO_DATA):** `gate.fit_microdata()` is `None` — no `.fit`. Construct each node's
  conditional from priors: slopes = LLM prior means (default 0 ⇒ near-independence),
  then **calibrate the intercept(s)** so the implied target marginal matches
  `known_marginals` (reuse `elicitation.known_vector` / `target_support`):
  - categorical: `intercept_class = log(known_prob_class)` (softmax of intercepts = the
    known marginal when slopes are 0; exact in that default case);
  - numeric: intercept = `known_mean − mean(Σ slope·parent)`, predictive sd = the known
    marginal's spread (estimated from its quantiles).
  Sample from those conditionals. Must NOT raise. (Honest note: with default zero
  slopes, C ≈ marginal-matched independent sampling threaded through the DAG order.)

Assemble via `background_frame` + `clip_decode`; write `fit_summary.json`; return
`StrategyResult`.

## 6. Components (one new file)

`src/ssdataagent/strategies/design_a.py`:

```python
@dataclass
class Structure:
    order: list[str]
    parents: dict[str, list[str]]
    prior_scale: dict[str, float]
    offsets: dict[str, float]

def elicit_structure(client, *, dataset, condition, schema, targets, backgrounds,
                     run_dir, cache_dir, transport=False) -> Structure: ...

def fit_numeric_node(X_train, y_train, prior_scale): ...   # -> fitted BayesianRidge
def fit_categorical_node(X_train, y_train, prior_scale, classes): ...  # -> fitted LogisticRegression

def sample_node(model_or_prior, X_eval, support, *, offset, rng): ...
    # numeric: Normal(mu+offset, sd); categorical: softmax(logits+offset) -> draw

def calibrate_intercept_c(support, known_vec, parent_contrib): ...  # C-only prior construction

class DesignAStrategy:  # name = "design_a"
    def generate(self, gate, run_dir, cfg) -> StrategyResult: ...
```

`DesignAStrategy.generate` (mirrors Design B/C orchestration): schema/bg/`fit_microdata`/
`known_marginals` → target set = schema targets in `known_marginals` (empty-target early
return) → supports → `elicit_structure` (transport under TRANSFER, cache under
`cfg.results_root/_structure_cache`) → walk `order`: per node build parent `X`, fit
(A/B) or prior-construct+calibrate (C), `sample_node` → assemble → `fit_summary.json` →
`StrategyResult`.

## 7. Registration & runner

- `registry.py`: `"design_a": DesignAStrategy`.
- `conditions.py`: `design_a_full` (FULL), `design_a_transfer` (TRANSFER),
  `design_a_aggregate` (NO_DATA) — all `strategy="design_a"`.
- Runner: **no change** (transfer-gate construction is strategy-agnostic; runner.py:173).
  A characterization test confirms `design_a_transfer` receives a source/crosswalk gate.

## 8. Determinism, leakage, artifacts

- **Determinism:** all fits are MAP/conjugate (deterministic); only predictive sampling
  uses a seeded RNG (`seed=42`). Structure elicitation reproducible via cache; tests mock
  the client.
- **Leakage:** B fits only on `source[crosswalk]` and applies LLM offsets — no
  target-survey targets are read (structural, via the gate). A no-leakage end-to-end test
  asserts the B path reads only source + elicited structure.
- **Artifacts:** `fit_summary.json` = `{backend:"design_a", condition, order, parents,
  node_types: {t: "numeric"|"categorical"}, n_train_fit, calibrated: bool, transport:
  bool}`; structure raw I/O under `run_dir/structure/`.

## 9. Out of scope (v1)

- Full MCMC/NUTS posterior (MAP + conjugate predictive instead).
- A fully Bayesian σ² treatment (BayesianRidge's conjugate estimate suffices).
- LLM-signed pairwise associations in C beyond the elicited slopes.
- Sequence targets (Types 4/5 stay agent-only per the handoff).

## 10. Testing strategy (client mocked)

- `tests/test_design_a_structure.py` — `elicit_structure` parse, parent-pruning to the
  legal set, order completion, fallback on bad JSON, cache hit = zero client calls.
- `tests/test_design_a_nodes.py` — numeric node returns mean+sd and sampled values in
  range; categorical node returns valid classes; C intercept calibration reproduces the
  known marginal (categorical exact with zero slopes; numeric mean within tolerance).
- `tests/test_strategy_design_a.py` — end-to-end A/B/C: all targets present, row count,
  no-raise in C, determinism (two runs identical), cache hit = zero client calls, and a
  **TRANSFER end-to-end no-leakage** test (generated values traceable only to source +
  structure; `transport` flag true in meta + fit_summary).
- Extend `tests/test_strategies_registry.py`, `tests/test_conditions.py`,
  `tests/test_runner_artifacts.py` (transfer-gate characterization; the two P0
  byte-stable tests stay green).

## 11. Gate (per `feedback_refactor_gate_philosophy`)

Our tests pass + no NEW failures vs. the 4 pre-existing `autograd`-missing failures
(`tests/test_config.py::test_unknown_provider_raises` + 3
`tests/test_ssdatabench_integration.py *_legacy`). No bit-for-bit reproduction gate.
