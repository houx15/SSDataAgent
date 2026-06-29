# Design — Part 2 (P1): statistical baselines + over-determination metric

**Date:** 2026-06-29
**Status:** Approved for planning
**Source plan:** `docs/handoff/delta-plan.md` (P1) and `docs/handoff/design-reference.md` (§5, §7 B1, §8)
**Scope:** The second of seven sequential sub-projects. It adds the three B1
statistical synthesizers (the non-LLM "honest bar") and the headline
over-determination diagnostic. It builds entirely behind the P0 Strategy seam —
no rewrite of the runner, agent, data, or scorer.

---

## Why this exists

P0 made the runner method-agnostic: it selects a `Strategy` by condition and
calls `strategy.generate(gate, run_dir, cfg)`. Today the only strategies are
the agent and direct-generation paths, so every result is LLM-produced and we
cannot say *which mechanism* produces the gain over the SSDataBench paper.

Part 2 adds:

1. **Three statistical baselines** (hot-deck/k-NN, sequential CART, Gaussian
   copula) — directly-runnable, non-LLM peers of the agent under the same
   information budget. This is the **honest bar**; in condition A (in-distribution)
   a good baseline may beat the LLM, which is itself a finding.
2. **The over-determination metric** — `H(target | demographics)` real vs.
   simulated, the **headline diagnostic** for H2 (case-wise LLMs underestimate
   within-group heterogeneity). Currently missing from the verify tools.

This is an **additive extension**: new files behind the seam, plus the metric
computed in the runner's common tail so *every* strategy gets it.

---

## Decomposition (full project, for context)

Each part is its own brainstorm → plan → implement cycle. **This spec covers Part 2 only.**

1. **Part 1 (P0)** — Strategy seam + thin InfoGate. **MERGED 2026-06-29.**
2. **Part 2 (P1)** — hot-deck/k-NN + CART + Gaussian-copula baselines + over-determination metric. ← *this spec*
3. **Part 3 (P2)** — Design B (marginals + copula); first consumer of the full InfoGate (`known_marginals`, A/B/C, `source_survey`).
4. **Part 4 (P3)** — Design C (retrieval + repair).
5. **Part 5 (P4)** — Design A (hierarchical Bayes).
6. **Part 6** — S1 distribution diagnostic.
7. **Part 7 (P5)** — local web console.

---

## Verified repo facts (read at design time)

- `strategies/base.py` — `Strategy` Protocol (`name: str`,
  `generate(gate, run_dir, cfg) -> StrategyResult`), `StrategyResult(generated:
  pd.DataFrame, meta_extras: dict)`, `InfoGate(condition, dataset_name,
  workspace, client, train, eval_rows, unseen_variables=())` with
  `background() -> eval_rows` and `fit_microdata() -> train | None` (train for
  `FULL/NO_SEMANTIC/UNSEEN`, `None` for `NO_DATA/DIRECT`).
- `strategies/registry.py` — `STRATEGIES = {"agent": AgentStrategy, "direct":
  DirectGenerationStrategy}`; `get_strategy(name) -> Strategy` returns a fresh
  instance, raises `KeyError` on unknown.
- `strategies/direct_strategy.py` — the pattern a non-agent strategy follows:
  `generate` writes its own artifacts into `run_dir` and returns a
  `StrategyResult`. (Direct writes `prompts.jsonl`/`responses.jsonl`; baselines
  will not.)
- `experiments/conditions.py` — `ConditionSpec(name, context_condition:
  Condition, strategy: str)`; `get_condition(name) -> ConditionSpec`. Existing
  conditions map to `"agent"`/`"direct"`.
- `agent/context.py` — `Condition` enum: `FULL, NO_SEMANTIC, NO_DATA, UNSEEN,
  DIRECT`. `FULL` = information condition A (in-distribution: sees same-survey
  train microdata).
- `experiments/runner.py` — `_run_one_condition` builds an `InfoGate`, calls
  `get_strategy(spec.strategy).generate(...)`, builds base `meta` then
  `.update(result.meta_extras)`, calls `_write_common(run_dir, meta, generated,
  dataset, run_id, eval_df)`. `_write_common` writes `meta.json` +
  `generated.csv` (`index=False`) + runs `run_evaluation` + writes `eval.json`
  via `_serialize_rates`. The held-out `eval_df` carries the **real target
  values** (it is a split of the real data, not the sealed-scorer view), so the
  metric can read real targets here without leaking into generation.
- `data/schema.py` — `load_schema(name) -> DatasetSchema` with
  `background_variables`, `target_variables`, `descriptions`, `allowed_values:
  dict[str, list]`, `numeric_ranges: dict[str, (min,max)]`, `domains`. A target
  is **numerical** iff it appears in `numeric_ranges`; **categorical/ordinal**
  iff it appears in `allowed_values`.
- `data/splitter.py` — `split_train_eval(df, ratio, seed=42)`; generated rows
  align positionally with `eval_df` because strategies generate over
  `gate.background()` in order.
- Installed deps: `numpy 2.x`, `pandas 3.x`, `scipy 1.17`, `scikit-learn 1.8`,
  `statsmodels 0.14`. **No** `copulas`/`pgmpy`. The copula backend is
  implemented in-house on numpy/scipy — **no new dependency is added**.

---

## §1 — Module layout

New files:

- `src/ssdataagent/strategies/baselines.py` — `HotDeckStrategy`,
  `CartStrategy`, `CopulaStrategy`, plus shared column-encoding helpers.
- `src/ssdataagent/evaluation/overdetermination.py` — the metric.

Changed files (minimal, additive):

- `strategies/registry.py` — add `"hotdeck"`, `"cart"`, `"copula"` to `STRATEGIES`.
- `experiments/conditions.py` — add three `ConditionSpec`s.
- `experiments/runner.py` — `_write_common` computes the metric and
  `_serialize_rates` adds it to `eval.json`.
- `scripts/generate_exp_report.py` — surface the headline gap next to T1–T5.
- `evaluation/__init__.py` — export the metric entry point.

Unchanged: `agent/`, `data/`, `generation/`, `evaluation/runner.py` (scorer),
`evaluation/comparator.py`, `direct_generation.py`, `agent_strategy.py`,
`direct_strategy.py`, `base.py`, `config/*.yaml`, `experiments.yaml`,
`dashboard/`.

---

## §2 — Baseline strategies (`baselines.py`)

All three implement the `Strategy` Protocol, fit on `gate.fit_microdata()`, and
are **deterministic** given a fixed seed (default `42`, read from a strategy
param so tests pin it). None write `prompts.jsonl`/`responses.jsonl`. Each:

- writes a small `fit_summary.json` artifact into `run_dir` (backend name, params,
  `n_train_fit`, target column kinds) for provenance;
- returns `StrategyResult(generated, meta_extras)` where `meta_extras` includes
  `{"backend": <name>, "n_individuals": len(background), "n_train_fit": len(train),
  ...backend params...}`;
- generates one row per `gate.background()` row, **in order** (positional
  alignment with `eval_df`), filling every `schema.target_variables` column;
- raises a clear `ValueError` if `gate.fit_microdata() is None` (condition C /
  no-data is deferred to Part 3 — baselines require microdata in P1).

**Shared column handling.** A target/background column is **numerical** iff it
is in `schema.numeric_ranges`, else **categorical/ordinal** (in
`schema.allowed_values`; treated as ordinal using the order of the `allowed`
list). Encoding helpers convert between the raw frame and a numeric matrix:
numerical → standardized; categorical → one-hot (for distance) or ordinal rank
(for copula latent transform). Generated categorical values are always decoded
back to members of `allowed_values`; numerical values are clipped to
`numeric_ranges`.

### `HotDeckStrategy` (name `"hotdeck"`)
1. Build a background feature matrix for train and eval rows (numeric
   standardized on train stats; categorical one-hot).
2. `sklearn.neighbors.NearestNeighbors(n_neighbors=k)` (default `k=10`) fit on
   train background.
3. For each eval row: find its `k` nearest train rows; with a seeded RNG pick
   **one** neighbor; copy that neighbor's **entire target vector**.
4. The conditional joint over targets is preserved by construction (every
   generated target vector is a real observed vector).
- `meta_extras`: `{"backend": "hotdeck", "k": k, ...}`.

### `CartStrategy` (name `"cart"`)
Synthpop-style sequential CART.
1. Choose a target order (the schema's `target_variables` order).
2. For each target `t` in order: fit a tree on `background + targets already
   generated` —
   `DecisionTreeClassifier` if `t` is categorical/ordinal,
   `DecisionTreeRegressor` if numerical (both seeded, with a `min_samples_leaf`
   floor, default `5`).
3. **Sample, do not take the mode**: classifier → draw from the predicted class
   probabilities at the row's leaf; regressor → draw a random training target
   value from the row's leaf members (preserves within-leaf spread). This is
   what keeps CART from collapsing variance.
4. Decode/clip to allowed values/ranges.
- `meta_extras`: `{"backend": "cart", "min_samples_leaf": ..., "target_order": [...]}`.

### `CopulaStrategy` (name `"copula"`)
In-house Gaussian copula over `(background + targets)`, sampled **conditioned on
each eval row's background**. No new dependency.
1. **Latent transform** each column of the train frame to standard-normal
   scores: numerical → empirical normal-score (rank → `Φ⁻¹((rank−0.5)/n)`);
   categorical/ordinal → category cumulative interval `[Φ⁻¹(F(c−1)), Φ⁻¹(F(c))]`
   (ordinal cut points; the polychoric-style construction), storing the cut
   points per column for inversion.
2. Fit the empirical correlation matrix `Σ` over all latent columns
   (background ∪ target). Regularize to PD if needed (shrinkage toward
   identity).
3. For each eval row: compute its background latent scores (numerical via train
   marginal; categorical via mid-interval); draw the target latent block from
   the **conditional Gaussian** `N(μ_{t|b}, Σ_{t|b})` with a seeded RNG.
4. **Inverse transform** target latent scores back to values: numerical → train
   marginal quantile; categorical/ordinal → the category whose cut interval
   contains the score. Clip/decode.
- `meta_extras`: `{"backend": "copula", "regularization": ...}`.

---

## §3 — Over-determination metric (`overdetermination.py`)

Computed locally on `generated.csv` (sim targets) vs. the real eval rows. Not
via SSDataBench. Both frames describe the **same background population** (sim is
generated over the eval backgrounds), so the metric is **permutation-invariant**:
each stage bins each frame independently and compares within matching demographic
cells — row order does not matter. A lightweight **sanity guard** runs first
(equal length + the background columns present in both); on violation the metric
returns the standard two-stage shape with a `reason` instead of computing,
never raising.

```python
def overdetermination(
    real: pd.DataFrame,
    sim: pd.DataFrame,
    *,
    schema: DatasetSchema,
    n_target_bins: int = 5,
    n_demo_bins: int = 4,
    min_count: int = 20,
    seed: int = 42,
) -> dict:
    """gap = H_real(target | demographics) − H_sim(target | demographics).
    Positive gap = sim is over-determined (collapsed within-group variance)."""
```

`demographics` = `schema.background_variables`; targets = `schema.target_variables`.
**Numerical targets** are discretized to `n_target_bins` bins using **bin edges
computed once from the real target marginal** and applied to both real and sim,
so entropies are comparable. Categorical/ordinal targets use their categories.

### Cell-based (headline)
- Coarsen each background var: numerical → `n_demo_bins` quantile bins (edges
  from real); categorical → its categories. Cell = cross-product.
- Keep cells with ≥ `min_count` real rows; compute per-cell conditional Shannon
  entropy of each target; **weighted mean** over kept cells (weight = real cell
  count). Compute the same `sim` entropy on the same cells (using the sim rows
  falling in each cell).
- Report per target `{h_real, h_sim, gap}`, the **headline gap** = mean gap over
  targets, **and** `coverage` = fraction of rows in kept cells and `n_cells` —
  so sparse conditioning is never silent.

### Model-based (robustness)
- For each target, fit a consistent sklearn classifier
  `P(target | all background vars)` (e.g. `HistGradientBoostingClassifier`,
  fixed `random_state=seed`) on real, and the **same architecture** on sim.
- `H_real` / `H_sim` = mean per-row Shannon entropy of the predicted class
  distribution over the eval backgrounds. `gap = H_real − H_sim`.
- Numerical targets are discretized to the same real-derived bins first.
- Report per-target `{h_real, h_sim, gap}` and the mean headline gap.

### Output
```json
"overdetermination": {
  "cell_based":  {"headline_gap": float, "coverage": float, "n_cells": int,
                  "per_target": {"<t>": {"h_real":, "h_sim":, "gap":}}},
  "model_based": {"headline_gap": float,
                  "per_target": {"<t>": {"h_real":, "h_sim":, "gap":}}}
}
```

Robustness: if a stage cannot be computed (e.g. zero kept cells, or a degenerate
target), that stage returns `null`/empty with a recorded reason rather than
raising — the metric must never break a run's scoring tail.

---

## §4 — Wiring

- `registry.py`: `STRATEGIES` gains `"hotdeck": HotDeckStrategy`, `"cart":
  CartStrategy`, `"copula": CopulaStrategy`.
- `conditions.py`: three new `ConditionSpec`s, all
  `context_condition=Condition.FULL` (information condition A), `strategy` =
  matching name:
  - `ConditionSpec("hotdeck", Condition.FULL, strategy="hotdeck")`
  - `ConditionSpec("cart", Condition.FULL, strategy="cart")`
  - `ConditionSpec("copula", Condition.FULL, strategy="copula")`
  (matching the existing keyword style in `CONDITIONS`.)
  Existing conditions untouched; `experiments.yaml`/`config/*.yaml` untouched
  (runs opt in by listing these condition names).
- `runner.py`: `_write_common` calls `overdetermination(real=eval_df,
  sim=generated, schema=load_schema(dataset), ...)` and `_serialize_rates` adds
  the block under `"overdetermination"`. Wrapped so a metric failure logs and
  yields a `null` block, never aborting the run.
- `generate_exp_report.py`: print the cell-based `headline_gap` (and coverage)
  alongside the T1–T5 pass rates.
- `evaluation/__init__.py`: export `overdetermination`.

---

## §5 — Testing & gate (TDD, all deterministic — no LLM)

Fast local loop, mocked nothing (baselines and metric are pure compute).

- `tests/test_strategy_hotdeck.py` — seeded determinism; every generated target
  vector equals some train target vector; shape + row alignment with eval;
  `fit_microdata() is None` raises.
- `tests/test_strategy_cart.py` — seeded determinism; outputs respect
  `allowed_values`/`numeric_ranges`; variance not collapsed (a sampled column is
  not constant when the conditional is non-degenerate); shape.
- `tests/test_strategy_copula.py` — seeded determinism; outputs respect
  allowed/ranges; correlation between two correlated targets is reproduced
  approximately on a constructed correlated fixture; shape.
- `tests/test_overdetermination.py` — hand-built small frames with known
  entropies → assert exact `gap` per target; collapsed sim (constant target
  within cell) gives a positive gap; numerical-target binning uses real edges;
  coverage + `n_cells` reported; divergent-background values handled (no kept
  cell has sim rows → reason); the length/column sanity guard reports a reason;
  both variants; degenerate inputs return `null` reason, not an exception.
- `tests/test_strategies_registry.py` — extend: `get_strategy` returns each new
  class; unknown still raises.
- `tests/test_conditions.py` — extend: the three specs map to the right
  strategy + `Condition.FULL`.
- `tests/test_runner_artifacts.py` — extend the P0 characterization net: a
  baseline run writes `meta.json` (with `backend`), `generated.csv`,
  `fit_summary.json`, **no** `prompts/responses.jsonl`, and `eval.json` carries
  the `overdetermination` block. (Mock `run_evaluation` so no SSDataBench
  subprocess is needed.)

**Gate (per [[feedback_refactor_gate_philosophy]]):** correctness is the full
local suite green (minus the 4 pre-existing `autograd`-missing failures), not
bit-for-bit reproduction. A cloud-box smoke run of one baseline condition is an
optional, non-blocking confidence check.

---

## §6 — Out of scope (do not build this cycle)

- Condition C (aggregate-only): IPF / max-entropy fitting and
  `InfoGate.known_marginals()` / `known_associations()` — **Part 3**.
- A pgmpy Bayesian-net backend — documented future fast-follow; the copula is
  the v1 third backend.
- Designs A/B/C, S1, the web console.
- Dashboard changes (separate AGENTS.md regeneration flow; can follow once the
  metric lands in `eval.json`).
- Any change to `config/*.yaml`, `experiments.yaml`, the scorer
  (`evaluation/runner.py`), or `agent/`.

---

## Risks & mitigations

- **Curse of dimensionality in the cell-based metric** (sparse cells → low
  coverage). Mitigated by coarse binning + `min_count` floor + **explicit
  coverage reporting** (never silent), and the model-based variant as a
  conditioning-on-all-demographics cross-check.
- **Copula latent transform on ordinal data** mis-specified → unrealistic draws.
  Mitigated by the polychoric-style cut-point construction (ordinal-aware) and a
  test asserting a known target–target correlation is reproduced.
- **CART collapsing to modal prediction** (the very failure we critique).
  Mitigated by sampling from leaf distributions, with a test asserting
  non-degenerate output variance.
- **Metric breaking the scoring tail.** Mitigated by wrapping the metric so any
  failure yields a `null` block + logged reason, never an exception.
- **Background mismatch** between `generated` and `eval_df`. The metric is
  permutation-invariant (bins each frame independently, compares within matching
  cells), so row order is irrelevant. A length/column sanity guard runs first and
  returns a `reason` rather than computing on mismatched frames.
