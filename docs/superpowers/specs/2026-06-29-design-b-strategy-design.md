# Design — Part 3b: the Design B strategy (decomposition-first grounded generator)

**Date:** 2026-06-29
**Status:** Approved for planning
**Source plan:** `docs/handoff/delta-plan.md` (P2, Design B) and `docs/handoff/design-reference.md` (§7 Design B)
**Scope:** The second half of Part 3. It builds **Design B** — the first LLM-grounded generator — on top of the Part 3a foundation: per-cell LLM marginal elicitation → rake to known marginals → data-grounded copula coupling → sample. Covers all three information conditions A/B/C. It adds the `design_b` strategy, the elicitation layer (with caching + raw-I/O logging), a shared cell-partition utility, a shared copula module, the A/B/C condition specs, and the runner's transfer-gate construction.

---

## Why this exists

Part 3a delivered the foundation (full `InfoGate`: A/B/C semantics, `known_marginals()`/`known_associations()`, transfer source + crosswalk). Part 3b is its first consumer and the project's first **grounded LLM generator**: it changes the unit of generation from the individual to the per-cell distribution, grounds that distribution in known moments, and uses the LLM only for the part data can't supply (per-cell conditional shape, especially under transfer/aggregate conditions). Design B's two knobs — **marginal raking** (LLM-elicited shape, calibrated to known marginals) and **copula coupling** (cross-target dependence, from data) — map onto the two collapse modes (prior-collapse and sampling-collapse) and are independently ablatable.

---

## Decomposition (full project, for context)

1. **Part 1 (P0)** — Strategy seam + thin InfoGate. **MERGED.**
2. **Part 2 (P1)** — statistical baselines + over-determination metric. **MERGED.**
3. **Part 3 (P2)** — Design B. **Part 3a (foundation) MERGED.** ← **Part 3b (this spec)** is the strategy.
4. **Part 4 (P3)** — Design C (retrieval + repair).
5. **Part 5 (P4)** — Design A (hierarchical Bayes).
6. **Part 6** — S1 distribution diagnostic.
7. **Part 7 (P5)** — local web console.

---

## Locked decisions (from brainstorm, binding)

- **LLM = marginals knob; copula = data-grounded in all conditions.** The LLM elicits per-cell per-target conditional distributions; cross-target correlation comes from data: train (A), source (B), `known_associations` (C). The LLM never proposes the copula.
- **All three conditions A/B/C in this cycle** (the pipeline is shared; B is a prompt/reference-marginal variation).
- **Determinism via a persistent elicitation cache** (not temperature=0); raw prompts/responses logged as artifacts. Tests mock the client.

---

## Verified repo facts (read at design time)

- `agent/llm_client.py` — `LLMClient` Protocol: `chat(messages: list[dict], system: str | None = None) -> str`. Strategies receive a built client via `gate.client`.
- `strategies/base.py` (post-3a) — `InfoGate.background()` (eval rows), `fit_microdata()` (train/source[crosswalk]/None), `known_marginals() -> dict|None`, `known_associations() -> dict|None`, fields `source`/`source_name`/`crosswalk`. `Condition` has `FULL/NO_SEMANTIC/NO_DATA/UNSEEN/DIRECT/TRANSFER`.
- `strategies/baselines.py` — copula latent helpers `_build_cuts(train, cols, schema)`, `_latent_value(col_cut, value)`, `_latent_matrix(df, cols, cuts)`, `_invert(z_array, col_cut)`, `_make_pd(M, reg)`; plus the conditional-Gaussian sampling pattern inside `copula_generate`. Shared helpers from Part 1/2: `classify_columns`, `clip_decode`, `background_frame`.
- `evaluation/overdetermination.py` — `_coarsen(real, sim, schema, n_demo_bins)`, `_bin_edges(real_vals, n_bins)`, `_discretize(values, edges)`: the existing cell coarsening, to be extracted into the shared `cells` util and re-consumed here.
- `data/transfer.py` (3a) — `TRANSFER_PAIRS`, `load_source_wave(name)`, `compute_crosswalk(target_schema, source_schema, source_df, target_df)`.
- `data/aggregates.py` (3a) — `marginals(df, vars, schema, *, n_bins=10)`, `associations(df, target_vars, schema)`; the `known_marginals()` shape is `{var: {"kind": "categorical", "probs": {...}}}` / `{"kind": "numeric", "quantiles": {...}, "mean", "std"}`.
- `experiments/runner.py` — `_run_one_condition` builds `InfoGate(condition, dataset_name, workspace, client, train, eval_rows, unseen_variables=...)`, calls `get_strategy(spec.strategy).generate(gate, run_dir, cfg)`, then `_write_common` (meta + generated.csv + eval.json incl. `overdetermination`). `_run_one_condition` is where transfer-gate construction is added.
- `experiments/conditions.py` — `ConditionSpec(name, context_condition: Condition, strategy: str)`; `get_condition`.
- `generation/formatter.py` `format_generated` fills schema targets the strategy left absent with a uniform-random baseline (the honest negative for non-crosswalk transfer targets).

---

## §1 — Module layout

New files:
- `src/ssdataagent/strategies/copula.py` — the Gaussian-copula latent machinery, **extracted** from `baselines.py` (no behavior change): `build_cuts`, `latent_value`, `latent_matrix`, `invert`, `make_pd`, and a `conditional_gaussian_sample(Sigma, cond_idx, free_idx, cond_latent, rng, *, reg)` helper factoring the conditional-Gaussian draw. `baselines.py` re-imports these (its `copula_generate` keeps working unchanged).
- `src/ssdataagent/data/cells.py` — shared cell partition: `fit_scheme(df, variables, schema, *, n_bins=4) -> CellScheme`; `assign(df, scheme) -> pd.Series` (cell-key per row); `CellScheme` holds per-variable bin edges (numeric, real-derived) / category lists. `overdetermination.py` is refactored to use it (its `_coarsen` becomes a thin wrapper).
- `src/ssdataagent/strategies/elicitation.py` — `elicit_cell_distributions(client, *, dataset, condition, cells, schema, known_marginals, run_dir, cache_dir) -> dict[cell_key, dict[target, dist]]`: prompt construction, JSON parse + schema validation + bounded retries, persistent cache, raw-I/O logging.
- `src/ssdataagent/strategies/design_b.py` — `DesignBStrategy` (`name="design_b"`) + the pipeline functions (`_rake`, `_couple`, `_sample`).

Changed files:
- `strategies/baselines.py` — import copula helpers from the new `copula.py` (delete the local copies); behavior identical.
- `evaluation/overdetermination.py` — coarsening delegates to `data/cells.py`.
- `strategies/registry.py` — add `"design_b"`.
- `experiments/conditions.py` — add `design_b_full` (`FULL`), `design_b_aggregate` (`NO_DATA`), `design_b_transfer` (`TRANSFER`).
- `experiments/runner.py` — `_run_one_condition`: when `spec.context_condition is Condition.TRANSFER`, load the source wave (`TRANSFER_PAIRS`), compute the crosswalk, and build the `InfoGate` with `source`/`source_name`/`crosswalk` set.

Untouched: the scorer, dashboard, `agent/`, `config/*.yaml` (runs opt in by condition name), `experiments.yaml`.

---

## §2 — Cell partition (`data/cells.py`)

`fit_scheme` coarsens each background variable: numeric → `n_bins` quantile-bin edges from the frame; categorical → the observed/allowed category list. `assign` maps each row to a `"|".join(parts)` cell key using the fitted scheme (so the same scheme applies to eval, train, and source frames consistently). Sparse cells (below a min count, when requested) merge into a shared fallback key. This is the **single cell definition** shared by the over-determination metric and Design B, so "cells" means the same thing everywhere. Deterministic.

---

## §3 — Elicitation layer (`strategies/elicitation.py`)

For each cell, one `client.chat` call. The prompt carries: the population context, the cell's demographic description (the coarsened background values), the target variables with descriptions + `allowed_values`/`numeric_ranges`, the **known overall marginals** (as anchors), and — in condition B — an explicit *transport* instruction (these source-population shapes must be adapted to the target context). It requests strict JSON: per target, a categorical probability vector over `allowed_values` (summing ~1) or numeric quantiles, explicitly prompted for realistic within-group **spread** (counter the modal collapse).

- **Parse + validate:** categorical probs are renormalized over `allowed_values`; numeric quantiles are sorted/clamped to `numeric_ranges`. Malformed JSON → up to N retries (default 3); final fallback = the known marginal for that target (a safe, grounded default) with the failure logged.
- **Cache:** a persistent JSON cache keyed by a hash of `(dataset, condition, model, cell-key, sorted target-set, prompt-version)`. A hit replays the stored distribution — no client call. This makes reruns stable and cheap (design-reference §13).
- **Logging:** raw prompt + raw response per cell written to `run_dir/elicitation/` for provenance.

---

## §4 — Pipeline (`strategies/design_b.py`)

`DesignBStrategy.generate(gate, run_dir, cfg)`:

1. **Reference data.** `ref = gate.fit_microdata()` (train in A, source[crosswalk] in B) or `None` in C; `known_m = gate.known_marginals()`; `known_a = gate.known_associations()`. Determine the **target set**: schema targets present in `known_m` (= crosswalk targets under B).
2. **Cells.** `scheme = cells.fit_scheme(background_of_eval_and_ref, background_vars, schema)`; assign cell keys to eval rows (and to `ref` for weights/copula).
3. **Elicit.** `cell_dists = elicit_cell_distributions(...)` → per-cell per-target distribution.
4. **Rake.** Iterative proportional fitting: scale each cell's per-target marginal so the cell-weighted mixture (weights = eval cell sizes) matches `known_m`. Fixes prior-collapse while preserving the LLM's *relative* per-cell differences.
5. **Couple.** Build the copula correlation matrix over the target latent space from the data-grounded source: `ref` (A/B) via `copula.build_cuts` + empirical correlation; or `known_a` (C) assembled into a correlation matrix (shrunk toward identity for missing/uncomputable pairs). One shared copula (cells differ in marginals, not dependence) keeps it tractable.
6. **Sample.** For each eval row: draw a latent normal vector from the copula (`copula.conditional_gaussian_sample` is available if conditioning on background latents is used; v1 draws from the unconditional target copula and relies on per-cell marginals for the demographic signal), push each component through that row's **cell's calibrated marginal** (`copula.invert`-style: categorical cut intervals, numeric quantile interpolation). Seeded RNG (default 42) → deterministic given the cache.
7. **Assemble.** Build the output via `background_frame` (no leakage) + filled targets; `clip_decode`. Non-target-set columns (non-crosswalk under B) are left absent → runner's `format_generated` applies the honest baseline fill. Return `StrategyResult(generated, meta_extras={"backend": "design_b", "condition": ..., "n_cells": ..., "n_elicited": ..., "cache_hits": ...})`. Write a `fit_summary.json`.

---

## §5 — Conditions & runner wiring

- `conditions.py`: `design_b_full`→`(FULL, "design_b")`, `design_b_aggregate`→`(NO_DATA, "design_b")`, `design_b_transfer`→`(TRANSFER, "design_b")`.
- `runner.py` `_run_one_condition`: if `spec.context_condition is Condition.TRANSFER`, look up `TRANSFER_PAIRS[dataset]`, `load_source_wave(source)`, `compute_crosswalk(load_schema(dataset), load_schema(source), source_df, eval_df)`, and construct the `InfoGate` with `source=source_df, source_name=source, crosswalk=tuple(crosswalk)`. Otherwise construct the gate as today. (This is the only runner change; existing conditions are unaffected.)
- `config/*.yaml`/`experiments.yaml` untouched — an experiment opts into Design B by listing the new condition names.

---

## §6 — Testing (deterministic; LLM mocked everywhere)

- `tests/test_cells.py` — `fit_scheme`/`assign`: numeric quantile bins, categorical passthrough, consistent assignment across frames, sparse-cell merge; **the over-determination metric tests still pass** after the refactor.
- `tests/test_copula_module.py` — the extracted helpers behave identically (port Part 2's copula assertions); `conditional_gaussian_sample` reproduces a known conditional mean/cov on a constructed Σ.
- `tests/test_elicitation.py` — prompt contains allowed values + known marginals (+ transport wording under B); parser renormalizes categorical probs and clamps numerics; malformed JSON retries then falls back to the known marginal; **cache hit replays with zero additional client calls**; raw I/O logged.
- `tests/test_strategy_design_b.py` — with a mocked client returning fixed per-cell distributions: raking makes the cell-weighted marginal match `known_marginals` (within tol); output respects `allowed_values`/`numeric_ranges`; deterministic across two runs (cache); **no leakage in B** (only target-set/crosswalk targets filled by the model); `fit_microdata() is None` in C still works (uses `known_marginals`/`known_associations`).
- `tests/test_strategies_registry.py` / `tests/test_conditions.py` — `design_b` resolves; the three condition specs map correctly.
- `tests/test_runner_artifacts.py` (extend) — a transfer-condition run (mocked client + mocked `run_evaluation`) loads the source wave, builds the crosswalk, writes `generated.csv`, `meta.json` (with `backend="design_b"`), and `eval.json` with the `overdetermination` block; the two P0 characterization tests remain byte-stable.

**Gate (per [[feedback_refactor_gate_philosophy]]):** full local suite green minus the 4 pre-existing `autograd` failures; no new failures. A cloud-box live run (real LLM, conda `ssda` + tmux) is an optional, non-blocking confidence check — and the first real Design B numbers + over-determination gap.

---

## Out of scope (do not build this cycle)

- Designs C/A, S1, the web console.
- A background-conditioned copula (v1 draws from the unconditional target copula; the demographic signal lives in the per-cell marginals). Revisit only if Type-2/3 results demand it.
- A `cheap`/`main` model-slot split for elicitation (use the configured client; defer the ablation slot).
- Multi-seed runs / `cfg.seed` plumbing (deferred project-wide).
- Any change to the scorer, dashboard, or `agent/`.

---

## Risks & mitigations

- **Elicitation cost / cell explosion.** Coarse cells (`n_bins=4`) + sparse-cell merge bound the call count; the persistent cache makes reruns free. `meta_extras` logs `n_cells`/`cache_hits` so cost is visible.
- **Malformed LLM JSON.** Bounded retries → fallback to the known marginal (grounded, never crashes the run); failures logged.
- **Raking non-convergence / degenerate marginals.** IPF capped at a fixed iteration count with a tolerance; if a target's known marginal is degenerate, skip raking it (use the elicited shape) and log.
- **Copula correlation in C from sparse `known_associations`.** Missing/uncomputable pairs shrink toward independence (identity), so the matrix is always PD (`make_pd` guards) — a conservative default that avoids stereotyped over-association.
- **Reusing private copula helpers.** Mitigated by promoting them to `strategies/copula.py` with their own tests, so both `baselines.py` and `design_b.py` consume a supported interface.
- **Runner transfer construction touching the hot path.** Mitigated by gating strictly on `Condition.TRANSFER` (all existing conditions take the unchanged branch) and a characterization test that the P0 artifacts stay byte-stable.
