# SSDataAgent — Design Spec

**Date:** 2026-04-29
**Source spec:** `docs/SPEC.md` (engineering specification provided by user)
**This document:** brainstormed design decisions and architecture deltas, complementing SPEC.md.

## 1. Goal

Build a Python system in which an LLM agent acts as an autonomous data analyst that explores real social-survey data, builds a generative model in code, and produces synthetic individuals whose population-level statistics are evaluated against SSDataBench. The system enables direct comparison with the 15 LLMs benchmarked in the original SSDataBench paper.

## 2. Decisions resolved during brainstorming

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Phase-1 pilot dataset | **GSS 2018** | smallest schema (12 cols), richest SSDataBench coverage |
| 2 | Credentials storage | **`.env` only** | matches SPEC; keeps secrets out of tracked Python |
| 3 | LLM model | **`deepseek-v4-flash`** as configured | trust user's config; Phase-0 connectivity test will catch a wrong name |
| 4 | Sandbox isolation | **Trusted subprocess + timeout + per-run tempdir** | research-grade local use; over-isolation costs build time without benefit |
| 5 | Execution rhythm | **Strict TDD per phase, auto-continue, pause only on real blockers** | user wants the project finished |
| 6 | Sandbox state model | **Stateless subprocess + shared workspace dir** | simpler, easier to test, persistence via files works for multi-step |
| 7 | Evaluation bridge | **Subprocess wrapper around SSDataBench scripts** | survives their refactors; doesn't require ssdatabench/ to be importable |

## 3. Repository layout

```
SSDataAgent/
  .env                          # LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL  (git-ignored)
  config/
    datasets.yaml               # registry: name -> CSV path, ssdatabench yaml, eval script
    experiments.yaml            # matrix: dataset x condition
    llm.yaml                    # provider, base_url, model, temperature, max_tokens (no secrets)
  real_data/                    # canonical CSVs (gss_clean, cps_clean, acs_clean) + dataset_meta.json
  src/ssdataagent/              # importable package (`pip install -e .`)
    config.py                   # loads .env + llm.yaml, env vars take precedence
    data/
      schema.py                 # DatasetSchema; reads ssdatabench/.../data_configs/*.yaml
      loader.py                 # load real_data/*.csv into typed DataFrame
      splitter.py               # reproducible train/eval split
    agent/
      sandbox.py                # stateless subprocess + shared tempdir
      context.py                # FullContext / NoSemantic / NoData / UnseenVariable
      prompt_templates.py       # system + per-stage prompts
      orchestrator.py           # main agent loop (explore -> model -> validate -> generate)
      llm_client.py             # OpenAI-compatible + Anthropic-compatible behind one interface
    generation/
      formatter.py              # agent DataFrame -> ssdatabench/simulated_data/<dataset>/<run>/sim_profiles_*.csv
    evaluation/
      runner.py                 # subprocess wrapper around ssdatabench/scripts/evaluation/<dataset>.py
      comparator.py             # tidy results across (condition, dataset, type)
    experiments/
      conditions.py             # DirectGeneration, FullAgent, AgentNoSemantic, AgentNoData
      runner.py                 # iterate conditions x datasets, resume support
      logger.py                 # write prompts.jsonl, responses.jsonl, code/*, workspace/, eval.json
  tests/                        # pytest, mirrors src/
  scripts/
    smoke_eval.py               # Phase-0 smoke: ssdatabench eval on their existing sim data
    run_experiment.py           # CLI: --experiment <name>
  results/<exp>/<condition>/<dataset>/<run_id>/
                                # full per-run logs (see section 7)
  ssdatabench/                  # vendored repo (already present)
  docs/
    SPEC.md
    superpowers/specs/2026-04-29-ssdataagent-design.md   # this file
```

**Key deltas from SPEC layout:**
- Code is a proper package under `src/ssdataagent/`, not bare `src/`.
- Real data lives in `real_data/` (where the user placed it) — loader does *not* read from `ssdatabench/real_data/`.
- Variable metadata (allowed values, descriptions, population context) is read from existing `ssdatabench/real_data/data_configs/*.yaml`.

## 4. Sandbox

Stateless subprocess with a shared workspace directory:

```python
class Sandbox:
    def __init__(self, workspace: Path, timeout: int = 60): ...
    def stage_file(self, name: str, content: bytes | str) -> Path: ...   # initial files
    def run(self, code: str) -> SandboxResult: ...                        # writes step.py, runs `python step.py`
    def list_files(self) -> list[Path]: ...                               # for debugging
    def close(self) -> None: ...                                          # cleanup tempdir

@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    timed_out: bool
```

- Each `run()` writes `step_NN.py` to the workspace and shells out via `subprocess.run(["python", "step_NN.py"], cwd=workspace, timeout=...)`.
- Output truncated to 8 KB stdout / 8 KB stderr in the result handed to the LLM (full version is logged).
- Workspace is a `tempfile.mkdtemp()` directory; persists across `run()` calls within one agent run; deleted on `close()`. Each run’s workspace is also copied to `results/.../workspace/` for reproducibility.
- The agent's *system prompt* states: "Each code block runs in a fresh Python process. Persist state by writing files to the working directory; read them back in later steps."

## 5. Agent loop

A fixed-stage state machine:

```
EXPLORATION  -> MODELING  -> VALIDATION (loop, max N=3)  -> GENERATION
```

Per stage: build prompt from history + stage template → `LLMClient.chat()` → extract code block → `sandbox.run(code)` → append result to history.

Validation criterion is permissive in v1 (the run produced a DataFrame with the right columns and value ranges). The real judge is SSDataBench. Max 3 retries then move on.

`Orchestrator.run(condition, dataset) → (generated_df, run_log)` is the single public entry point.

## 6. LLM client

```python
@dataclass(frozen=True)
class LLMConfig:
    provider: Literal["openai", "anthropic"]
    base_url: str
    api_key: str
    model: str
    temperature: float = 1.0
    max_tokens: int = 4096

class LLMClient(Protocol):
    def chat(self, messages: list[dict], system: str | None = None) -> str: ...
```

- `OpenAICompatibleClient` uses the `openai` SDK with custom `base_url`. Used for DeepSeek.
- `AnthropicCompatibleClient` uses the `anthropic` SDK. Wired but not exercised live for this project (DeepSeek is OpenAI-compatible). Unit-tested with a mocked SDK.
- `extract_python_block(text) -> str | None` is a free function, fully unit-tested without any SDK calls.

## 7. Logging

Per `results/<experiment>/<condition>/<dataset>/<run_id>/`:

| File | Purpose |
|------|---------|
| `meta.json` | config snapshot, git SHA, timing, errors |
| `prompts.jsonl` | every message sent to LLM (role, content, stage, ts) |
| `responses.jsonl` | every response (text, model, latency_ms, usage if available) |
| `code/step_NN.py` | each executed code block |
| `code/step_NN.{stdout,stderr,exit}` | captured outputs |
| `workspace/` | snapshot of sandbox tempdir at run completion |
| `generated.csv` | formatted output handed to evaluator |
| `eval.json` | parsed pass rates from SSDataBench |

## 8. Evaluation bridge

Subprocess wrapper:

```
runner.run_evaluation(dataset, sim_dir) ->
  copy/format generated CSVs into ssdatabench/simulated_data/<dataset>/<run_id>/
  subprocess.run(
    ["python", "scripts/evaluation/<dataset>.py",
     "--sim-root", "./simulated_data/<dataset>/<run_id>",
     "--output-base", "./evaluation_results/<dataset>/<run_id>"],
    cwd="ssdatabench/")
  parse pass-rate JSONs from ssdatabench/evaluation_results/<dataset>/<run_id>/
  return PassRates{type1, type2, type3, type4, type5, by_variable}
```

Comparator builds tidy `(condition, dataset, type, pass_rate)` DataFrame and a pivot summary.

## 9. Configuration

`.env` (committed only as `.env.example`; real `.env` is git-ignored):
```
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-...
LLM_MODEL=deepseek-v4-flash
```

`config/llm.yaml` (committed, no secrets):
```yaml
provider: openai
base_url: https://api.deepseek.com
model: deepseek-v4-flash
temperature: 1.0
max_tokens: 4096
```

`load_llm_config()` precedence: env > yaml > defaults. One function used everywhere.

## 10. Testing strategy (TDD)

- **Unit tests, no LLM calls (default `pytest`):** schema, loader, splitter, sandbox, formatter, code extraction, config loading, comparator. Sub-second.
- **LLM-touching tests (gated by `RUN_LIVE_LLM_TESTS=1`):** `test_openai_api_reachable`, end-to-end orchestrator run. Skipped by default to avoid burning credits.
- **Integration tests:** Phase-0 SSDataBench smoke against their existing simulated data (no LLM needed).
- **Mocked LLM tests:** `pytest-mock` patches `OpenAI.chat.completions.create` with canned responses to deterministically test orchestrator stage transitions, retry loop, code extraction error paths.
- **Shared fixtures (`conftest.py`):** `tmp_workspace`, `gss_schema`, `tiny_train_split`, `mock_llm_client`.

## 11. Phasing — execution plan

Follow SPEC's Phase 0 → 7 in order. Per phase:

1. Write tests listed in SPEC for that phase (Red).
2. Implement minimal code to pass them (Green).
3. Run suite, confirm green.
4. Commit `phase N: <component>`.
5. Post one-paragraph status, continue.
6. Pause only on real blockers: failing API, ambiguous decision, destructive action.

**Departures from SPEC test list:**
- Skip `test_anthropic_api_reachable()` in Phase-0 live suite (DeepSeek is OpenAI-compatible only). Anthropic client still unit-tested with mocks.
- Add `test_load_llm_config_precedence()` to verify env > yaml.

## 12. Open low-risk decisions I make as I go

- Validation-loop pass criterion (start permissive: "DataFrame with right columns and in-range values").
- Unseen-variable subsets per dataset (Phase 6) — pick one demographically meaningful target per dataset.
- Random seeds — fixed (`42`) wherever reproducibility matters.

## 13. Success criteria (from SPEC, verbatim)

1. Full Agent achieves meaningfully higher average pass rates than Direct LLM Generation across SSDataBench's five pattern types.
2. Ablation: both semantic knowledge and data access contribute to performance.
3. Unseen-variable experiment: Full Agent produces non-trivial distributional predictions for variables it never observed.
4. All results reproducible from logged traces.

## 14. Out of scope (v1)

- Container-based sandboxing (Docker / Firejail).
- Multi-LLM provider testing in one experiment matrix (one provider per run).
- Distributed / parallel agent runs (sequential is fine for the matrix size we have).
- A web UI / dashboard for browsing results.
