# Design — Part 1 (P0): Strategy seam + thin InfoGate

**Date:** 2026-06-29
**Status:** Approved for planning
**Source plan:** `docs/handoff/delta-plan.md` (P0) and `docs/handoff/design-reference.md` (§5, §6)
**Scope:** This is the first of seven sequential sub-projects (see "Decomposition" below). It implements **only** the refactor seam and its regression gate — no new strategies, no A/B/C semantics, no metric.

---

## Why this exists

The repo today produces `generated.csv` by branching inside the experiment
runner on whether a condition is agent-based (`spec.is_agent`) or
direct-generation. To later add statistical baselines and Designs A/B/C as
peers of the agent, the runner must become **method-agnostic**: it selects a
`Strategy` by config and calls one method. This sub-project introduces that
seam with **zero behavior change** — the existing agent and direct paths become
two strategies that produce byte-identical artifacts and `eval.json`.

This is an **additive extension, not a rewrite**. Everything under `agent/`,
`data/`, `evaluation/`, `generation/` is called through the new seam, not
modified (one small exception: `log_run`, see §2).

---

## Decomposition (full project, for context)

Each part is its own brainstorm → plan → implement cycle. **This spec covers Part 1 only.**

1. **Part 1 (P0)** — Strategy seam + thin InfoGate; wrap the two existing paths; regression gate. ← *this spec*
2. **Part 2 (P1)** — hot-deck/k-NN statistical baseline + over-determination metric in `evaluation/`.
3. **Part 3 (P2)** — Design B (marginals + copula). First consumer of the full InfoGate (known_marginals, A/B/C, source_survey).
4. **Part 4 (P3)** — Design C (retrieval + repair).
5. **Part 5 (P4)** — Design A (hierarchical Bayes, LLM-elicited priors).
6. **Part 6** — S1 distribution diagnostic.
7. **Part 7 (P5)** — local web console.

---

## Verified repo facts (read at design time)

- `experiments/runner.py:136` `_run_one_condition` branches on `spec.is_agent`:
  - **agent path** (lines 183–227): `build_context` → `Orchestrator(...).run(...)` → `log_run(result, run_dir, meta)` → `run_evaluation` → write `eval.json`.
  - **direct path** (lines 139–181): `generate_direct(...)` → write `meta.json`, `prompts.jsonl`, `responses.jsonl`, `generated.csv` inline → `run_evaluation` → write `eval.json`.
- Both paths converge on the same artifact set: `meta.json`, `prompts.jsonl`, `responses.jsonl`, `generated.csv`, `eval.json`. Agent additionally writes `code/step_*.py|stdout|stderr|exit`.
- `experiments/logger.py:log_run` writes `meta.json`, `prompts.jsonl`, `responses.jsonl`, `code/`, **and** `generated.csv` (line 38).
- `experiments/conditions.py` `ConditionSpec` has fields `name`, `context_condition` (a `Condition` enum), `is_agent: bool`. Five registered conditions.
- `agent/context.py` `build_context(...)` performs the data/description gating (writes `train.csv` / `descriptions.json` into the workspace based on `Condition`) and returns `AgentContext(has_data, has_descriptions, ...)`.
- `experiments/direct_generation.py` `generate_direct(*, client, sampled, dataset_name, transcript_out)` reads the schema directly; needs only the eval rows + client.
- `tests/test_experiment_runner.py` fully mocks `build_client`, `Orchestrator`, `generate_direct`, `run_evaluation` — the runner is unit-testable without real LLM calls.

---

## §1 — Module layout

New package `src/ssdataagent/strategies/`:

- `base.py` — `Strategy` Protocol, `StrategyResult` dataclass, `InfoGate`.
- `agent_strategy.py` — `AgentStrategy` (wraps the orchestrator path verbatim).
- `direct_strategy.py` — `DirectGenerationStrategy` (wraps `generate_direct` + its artifact writing).
- `registry.py` — `get_strategy(name: str) -> Strategy` plus a `STRATEGIES` dict.
- `__init__.py`.

Changed files (minimal):

- `experiments/conditions.py` — replace `is_agent: bool` with `strategy: str` on `ConditionSpec`; map existing conditions to `"agent"` / `"direct"`.
- `experiments/runner.py` — `_run_one_condition` selects a strategy and calls it; owns the common artifact tail; no `is_agent` branch.
- `experiments/logger.py` — `log_run` no longer writes `generated.csv` (see §2). All other artifacts unchanged.

Unchanged: everything under `agent/`, `data/`, `evaluation/`, `generation/`, `dashboard/`, `direct_generation.py`, `config/*.yaml`, `experiments.yaml`.

---

## §2 — Strategy contract (`base.py`)

```python
class Strategy(Protocol):
    name: str
    def generate(self, gate: InfoGate, run_dir: Path, cfg: "ExperimentConfig") -> "StrategyResult": ...

@dataclass
class StrategyResult:
    generated: pd.DataFrame
    meta_extras: dict   # strategy-specific meta.json fields
```

**Responsibility split:**

- **Strategy writes** its method-specific artifacts into `run_dir`:
  - `AgentStrategy` → calls `log_run`, which (after this change) writes only `prompts.jsonl`, `responses.jsonl`, `code/`.
  - `DirectGenerationStrategy` → writes `prompts.jsonl`, `responses.jsonl`.
- **Runner owns the common tail** for every strategy: build the base `meta` dict (experiment, dataset, condition, run_id, git_sha, model, provider), merge `result.meta_extras`, write `meta.json`; write `result.generated` to `generated.csv`; call `run_evaluation`; write `eval.json`.

**The one behavior-preserving change:** today `generated.csv` is written by
`log_run` (agent) and inline (direct). To centralize it in the runner without
double-writing, `log_run` stops writing `generated.csv` (drop logger.py:38).
The runner writes it for both paths identically. Net file output is unchanged.

**meta.json ownership.** To keep `meta.json` written exactly once and identical,
the runner is the sole writer of `meta.json`. `log_run` must therefore also stop
writing `meta.json`; instead `AgentStrategy` returns its extra fields
(`unseen_variables`) via `meta_extras`, and the runner merges + writes. The
direct path returns `n_individuals` via `meta_extras`. This means `log_run`'s
responsibilities shrink to: `prompts.jsonl`, `responses.jsonl`, `code/`.
(Regression gate verifies the merged `meta.json` matches the pre-refactor content
key-for-key for both paths.)

---

## §3 — Thin `InfoGate` (`base.py`)

A data carrier constructed once per (condition, dataset), centralizing "what a
strategy may see." Exposes only what the two current strategies consume:

```python
@dataclass
class InfoGate:
    condition: Condition
    dataset_name: str
    workspace: Path
    client: LLMClient
    _train: pd.DataFrame
    _eval: pd.DataFrame
    unseen_variables: tuple[str, ...] = ()

    def background(self) -> pd.DataFrame:
        """Test/eval rows — always allowed."""
        return self._eval

    def fit_microdata(self) -> pd.DataFrame | None:
        """Train split when the condition permits data; None otherwise."""
        # mirrors context.has_data gating (FULL / NO_SEMANTIC / UNSEEN -> data)
```

**Deferred to Part 3 (explicitly out of scope here):** `known_marginals()`,
`known_associations()`, `source_survey`, and A/B/C condition semantics. The gate
only relocates existing gating; it introduces no new gating behavior. The
`AgentStrategy` continues to call `build_context` for the workspace-file
gating exactly as today.

---

## §4 — Config selection & dispatch

- `ConditionSpec.is_agent: bool` → `ConditionSpec.strategy: str`.
- Mapping (backward compatible):
  - `full_agent`, `agent_no_semantic`, `agent_no_data`, `full_agent_unseen` → `"agent"`
  - `direct_generation` → `"direct"`
- `registry.get_strategy("agent")` → `AgentStrategy()`; `"direct"` → `DirectGenerationStrategy()`.
- `experiments.yaml` and `config/*.yaml` are **untouched** — runs still select by condition name; the strategy is derived from the `ConditionSpec`.
- `_run_one_condition` becomes:
  1. build `InfoGate` from (spec, dataset, train, eval, workspace, unseen, client),
  2. `strategy = get_strategy(spec.strategy)`,
  3. `result = strategy.generate(gate, run_dir, cfg)`,
  4. write merged `meta.json`, `generated.csv`,
  5. `run_evaluation` → `eval.json`.
  No `is_agent` branch remains.

---

## §5 — Testing & regression gate

**TDD, fast local loop (mocked LLM/eval):**

- `tests/test_strategies_registry.py` — `get_strategy` returns the right class; unknown name raises.
- `tests/test_info_gate.py` — `background()` returns eval rows; `fit_microdata()` returns train for data conditions and `None` for `agent_no_data` / `direct_generation`.
- `tests/test_strategy_agent.py` / `tests/test_strategy_direct.py` — **characterization tests**: under mocked client/orchestrator/eval, each strategy + runner writes byte-identical artifact files (`meta.json`, `prompts.jsonl`, `responses.jsonl`, `generated.csv`, and `code/` for agent) compared to the pre-refactor behavior. Capture the pre-refactor artifacts first (golden files) on the current commit, then assert equality after the refactor.
**Existing tests to update (mechanical, behavior-preserving):**

- `tests/test_logger.py` — `test_log_run_writes_expected_files` currently asserts `log_run` writes `meta.json` (lines 31, 37–38) and `generated.csv` (line 36). After §2, `log_run` writes neither; move those two assertions to a runner-level test and shrink this test to `prompts.jsonl` / `responses.jsonl` / `code/`.
- `tests/test_conditions.py` (lines 24, 28) and `tests/test_unseen_variables.py` (line 34) — replace `spec.is_agent is True/False` with `spec.strategy == "agent"/"direct"`.
- `tests/test_experiment_runner.py` — stays green as-is; it checks `generate_direct` vs `Orchestrator` call counts through the runner, which the new dispatch preserves. Keep intact.

**Final gate (cloud box, real LLM):**

- On the cloud box (conda `ssda` + tmux — never `.venv`, never `nohup`), run a smoke experiment with a fixed config/seed **before** the refactor to capture a baseline `eval.json` (and `generated.csv`), then run the same config **after** the refactor and diff. Completion of this cycle requires the post-refactor `eval.json` to match the baseline bit-for-bit for both an agent condition and `direct_generation`.
- Capture the baseline from the current `main` commit (pre-seam) so the comparison is honest.

---

## Out of scope (do not build in this cycle)

- Any new strategy (baselines, Designs A/B/C, S1).
- `known_marginals` / `known_associations` / `source_survey` / A-B-C condition semantics.
- The over-determination metric.
- The web console.
- Any change to `config/*.yaml`, `experiments.yaml`, the dashboard, or the scorer.

---

## Risks & mitigations

- **Artifact drift (breaks the dashboard parser / regression gate).** Mitigated by golden-file characterization tests + the real smoke-run bit-for-bit diff. The `meta.json` / `generated.csv` ownership move (§2) is the highest-risk spot — covered by a dedicated test.
- **Hidden coupling in `log_run`.** Mitigated by shrinking its responsibility only (remove two writes), not rewriting it; existing dashboard reads the remaining files unchanged.
- **`ConditionSpec` field rename touching callers.** Grep for `is_agent` before editing; update all references in the same change.
