# SSDataAgent

An LLM **data-analyst agent** for population-level social-survey simulation, evaluated against [SSDataBench](https://github.com/lszshu/SSDataBench).

Instead of prompting an LLM to generate one synthetic respondent at a time (the SSDataBench paradigm), this project gives the LLM access to real survey data and lets it explore, model, and generate via Python code execution. The agent functions as a data scientist, not a survey respondent.

## TL;DR — current state

The agent paradigm is now run on **7 datasets × 5 metric types (T1–T5)**, including longitudinal panels (NLSY, AddHealth, CFPS, Understanding Society) that exercise event-time chronology. The three LLMs we've benchmarked end-to-end:

| Model | Cross-sectional mean | Longitudinal mean | vs SSDataBench paper-best (Δ over 7 datasets) |
|---|---:|---:|---:|
| DeepSeek v4 flash    | 0.563 | 0.345 | +0.10 |
| gpt-5.4-mini         | 0.375 | 0.367 | +0.02 |
| **gpt-5.4** (current default) | 0.507 | **0.453** | **+0.13** (5/7 dataset wins) |

Two structural results worth highlighting:

- **The agent paradigm beats SSDataBench's per-individual paradigm on every dataset.** Best-of-15 paper LLMs averaged ~0.34; our agent on gpt-5.4 averages 0.476.
- **gpt-5.4 on Understanding Society is the first ever non-trivial T4 score** (0.270, vs paper-best ceiling of 0.05) — first evidence that the agent paradigm + a strong LLM can begin to capture life-course chronology, which the paper called the LLMs' hardest failure mode.

Full write-ups live in [`docs/report/`](docs/report/); the experiment ledger and current backlog are in [`docs/experiments/LEDGER.md`](docs/experiments/LEDGER.md) and [`docs/experiments/STRATEGY.md`](docs/experiments/STRATEGY.md).

## How it works

The current default agent (`rubric_tools_v3`, EXP-006e onward) is a **tool-using loop** in the same shape Claude Code and Codex use: one persistent LLM conversation, OpenAI function-calling protocol, fixed tool registry. The legacy 4-stage code-block path (`baseline`, `rubric` variants) is still wired and used for ablations — `PromptVariant.is_tool_using` routes the orchestrator.

```
   ┌──────────────────────────────────────────────────────────────┐
   │ Orchestrator                                                 │
   │   loop: ask LLM → dispatch tool_calls → return results       │
   │   exits when commit_generator succeeds or MAX_TURNS=40       │
   └────────┬─────────────────────────────────────┬───────────────┘
            │ chat_with_tools                     │ tool results
            ▼                                     │
   ┌──────────────────┐                           │
   │ LLM (gpt-5.4)    │   one persistent context,─┘
   │ function-calling │   not stage-by-stage subprocesses
   └────────┬─────────┘
            │ tool_calls
            ▼
   ┌────────────────────────────────────────────────────────────┐
   │ Tool registry — 16 tools across 4 families                 │
   │  inspect : list_columns, describe_column, cross_tab,       │
   │            missing_pattern, correlation, groupby_stat, …   │
   │  fit     : set_generation_order, fit_marginal,             │
   │            fit_conditional, fit_copy_real, replace_step    │
   │  verify  : sample_preview, score_marginal, score_pair,     │
   │            score_event_order, score_overall                │
   │  commit  : commit_generator (hard-gated on chronology),    │
   │            report_progress                                 │
   └────────┬───────────────────────────────────────────────────┘
            │ reads / mutates
            ▼
   ┌────────────────────────────────────────────────────────────┐
   │ RuntimeState — owned by the orchestrator across one run    │
   │  train_fit / held_out (deterministic 80/20 split)          │
   │  descriptions  (None when condition strips semantic info)  │
   │  Chain         — ordered list of fit Steps, built          │
   │                  incrementally; replaces model.pkl         │
   │  event_order_calls — audit trail read by commit gate       │
   └────────┬───────────────────────────────────────────────────┘
            │ chain.sample(n_rows) on commit
            ▼
   ┌──────────────────────┐    ┌────────────────────────────────┐
   │ generated.csv        │───►│ SSDataBench eval (subprocess)  │
   └──────────────────────┘    └────────────────────────────────┘
```

The loop:

1. **Inspect.** The agent reads `train_fit` (80% slice) through inspect-family tools. Tools return JSON-serializable summaries — never raw rows beyond a small `head_rows` peek. Under the `agent_no_data` condition the inspect family refuses, so the agent must rely on `descriptions.json` only.
2. **Fit incrementally.** The agent declares a `generation_order`, then fits each column with `fit_marginal` or `fit_conditional`. Each call adds one Step to the in-progress Chain. Supported families: empirical / categorical_empirical / kde / normal (marginal); empirical_lookup / linear_regression / logistic_regression (conditional, with optional `allow_missing=True` for structural NaN preservation).
3. **Verify against the held-out slice.** Verify-family tools sample the chain and score it against the 20% held-out: KS or TV distance per column, |Δ correlation| or |Δ Cramér's V| per pair, event-order compliance for life-event tuples. Each score returns `pass: true/false` against thresholds matching SSDataBench (T1 ≤ 0.10, T2 ≤ 0.15, T4 ≥ 0.80).
4. **Commit.** When the agent calls `commit_generator`, the orchestrator validates the chain and exits the loop. Two hard gates: an empty chain refuses, and a chain with ≥2 event-age columns (`age_at_first_*`, `age_finished_*`, etc.) refuses unless `score_event_order` has covered ≥2 of those columns — added in EXP-006e because the agent could otherwise optimize a marginal-only `score_overall` and ship a chain with destroyed chronology.
5. **Sample.** The orchestrator calls `chain.sample(n_rows)` to produce `generated.csv`, then hands off to SSDataBench scoring.

Every tool returns a dict; tools never raise. Bad input becomes `{"error": "...", "details": "..."}` so the LLM can self-correct — `dispatch` in `tools/__init__.py` catches every exception type. The whole tool-call transcript persists to `workspace/tool_calls.json` for offline analysis even if sampling crashes.

## Experimental conditions

The same orchestrator runs four conditions, each varying what the agent receives:

| Condition | Real data | Variable descriptions | Population context |
|---|:---:|:---:|:---:|
| **full_agent** | ✓ | ✓ | ✓ |
| agent_no_semantic | ✓ | ✗ | ✗ |
| agent_no_data | ✗ | ✓ | ✓ |
| **full_agent_unseen** (Phase 6) | partial | ✓ | ✓ |
| direct_generation (baseline) | — | per-prompt | per-prompt |

`direct_generation` reproduces SSDataBench's paradigm: one LLM call per individual, given background variables, asking for the targets in JSON.

## Repository layout

```
SSDataAgent/
  src/ssdataagent/        # importable package
    config.py             # .env + llm.yaml loader (env > yaml)
    data/                 # schema, loader, splitter
    agent/                # sandbox, context, prompts, llm_client, orchestrator
    generation/           # output formatter
    evaluation/           # subprocess wrapper around ssdatabench eval scripts
    experiments/          # conditions, logger, runner, direct_generation
  config/
    datasets.yaml         # dataset registry
    experiments.yaml      # experiment matrix (20+ entries, including EXP-001)
    llm.yaml              # provider/model defaults (no secrets)
    paper_baselines.json  # SSDataBench paper-best per (dataset, T-type)
  scripts/
    run_experiment.py     # run one experiment (writes done.flag on success)
    run_batch.py          # run a list of experiments back-to-back, resumable
    status.py             # at-a-glance progress for all experiments
    new_experiment.py     # scaffold a docs/experiments/ entry
    generate_exp_report.py# build the per-exp markdown report (vs paper-best)
    smoke_eval.py         # ssdatabench smoke test
    build_eval_subset.py  # auto-prune ssdatabench eval config to our columns
  tests/                  # 90 pytest tests, mirrors src/
  real_data/              # cleaned survey CSVs (gitignored — see Setup)
  ssdatabench/            # vendored eval suite (gitignored — see Setup)
  docs/
    SPEC.md               # engineering specification
    SSDataBench.pdf       # source benchmark paper
    LLM_Survey_Prediction_Agent.pdf
    report/               # paper-style writeups (one per pilot wave)
    experiments/          # ledger, strategy, per-exp retros, CLOUD_SETUP.md
    superpowers/          # design spec & implementation plan
  results/                # per-experiment artifacts; done.flag = experiment finished
```

## Setup

```bash
git clone git@github.com:houx15/SSDataAgent.git
cd SSDataAgent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Two paths are gitignored — copy/download separately:
#   real_data/     — cleaned survey CSVs (~6 MB)
#   ssdatabench/   — third-party eval suite (~35 MB)
# See docs/experiments/CLOUD_SETUP.md for the full upload + .env recipe.

pip install -r ssdatabench/requirements.txt
cp .env.example .env
# Edit .env: at minimum set LLM_API_KEY. LLM_MODEL/PROVIDER/BASE_URL
# are now usually set per-experiment in config/experiments.yaml — the
# .env entries are only the fallback when an experiment doesn't specify.
```

The default `llm.yaml` is configured for OpenAI (`gpt-5.4-2026-03-05`). DeepSeek and Anthropic are supported through the same `LLMClient` interface — see `src/ssdataagent/agent/llm_client.py`.

## Running experiments

The framework is built around three commands. Pick by what you want to do.

### 1. Smoke test (~1–2 min) — sanity-check after a code change

```bash
python scripts/run_experiment.py --experiment smoke_gss
# Or for the rubric variant added in EXP-001:
python scripts/run_experiment.py --experiment smoke_gss_rubric
```

One dataset, `n_rows=100`, `max_iterations=1`. Use this any time you change prompts, the orchestrator, the runner, or the LLM client.

### 2. Single experiment — focused investigation

```bash
python scripts/run_experiment.py --experiment pilot_paper_agents_gpt54
python scripts/run_experiment.py --experiment pilot_paper_agents_gpt54 --resume
```

`--resume` skips per-(condition × dataset) cells whose `eval.json` exists, and skips the experiment entirely if `results/<exp>/done.flag` exists.

### 3. Batch — many experiments in one shot, resumable

```bash
# in tmux on the cloud box (conda env `ssda` active), see CLOUD_SETUP.md
python scripts/run_batch.py exp001_rubric_cross exp001_rubric_long \
    | tee batch.log
# Ctrl-b d to detach; tmux attach -t ssda to come back later.
```

Sequential by design (resume logic is trivial that way). One failure doesn't block the rest. Per-experiment log lands at `results/<exp>/run.log` and `results/_batch_status.json` is rewritten after each step for SSH-friendly status checks.

### Status, from anywhere over SSH

```bash
python scripts/status.py                              # all experiments
python scripts/status.py exp001_rubric_cross          # filter
python scripts/status.py --watch                      # refresh every 5s
tail -f results/exp001_rubric_cross/run.log           # live output
```

Source of truth is `results/<exp>/done.flag` (success) and `failed.flag` (failure with traceback). If `done.flag` exists, the experiment finished and `summary.csv` is on disk.

### Per-experiment report (results vs the paper)

```bash
python scripts/generate_exp_report.py exp001_rubric_cross \
    --baseline pilot_paper_agents_gpt54
```

Writes `docs/experiments/<date>-<exp>-report.md` with three sections: **Strategy** (hypothesis + variant deltas), **Results** (T1–T5 per dataset), **vs Paper-best** (Δ against the strongest of SSDataBench's 15 LLMs per cell, sourced from `config/paper_baselines.json`). The optional `--baseline` adds a fourth A/B section.

### Per-run artifacts

Every per-(condition × dataset) cell writes to `results/<experiment>/<condition>/<dataset>/<run_id>/`:

```
meta.json                  run config + git SHA + model
prompts.jsonl              every message sent to the LLM
responses.jsonl            every response received
code/step_NN.py            every code block (legacy code-block path only)
code/step_NN.{stdout,stderr,exit}
workspace/
  train.csv                the 80% slice the agent saw
  descriptions.json        semantic context (omitted in no_semantic / no_data)
  chain.json               serialized Chain (tool-using path)
  tool_calls.json          full tool-call transcript with arguments + results
  transcript.json          LLM messages + tool_results in order
  progress.log             agent's free-form journal via report_progress
generated.csv              the synthetic dataset handed to scoring
eval.json                  SSDataBench T1–T5 pass rates
```

The full set of experiments is in [`config/experiments.yaml`](config/experiments.yaml) — 20+ entries covering smoke runs, the cross-sectional 4-condition matrix, longitudinal pilots, and the in-flight `exp001_rubric_*` variants.

## Deploying to a cloud box

Designed for **Ubuntu GCP server, run inside `tmux`, walk away**. The full recipe is in [`docs/experiments/CLOUD_SETUP.md`](docs/experiments/CLOUD_SETUP.md). Two-minute version:

1. **`rsync` two paths from your laptop** (both gitignored):
   - `real_data/` (~6 MB; `real_data/used_dataset/*.csv` + `dataset_meta.json` is the minimum)
   - `ssdatabench/` (~35 MB)
2. **Create `.env` on the box** with at least `LLM_API_KEY=sk-...`. Never put the key in any committed config.
3. **Inside `tmux`**, with the `ssda` conda env active, kick off a batch:
   ```bash
   tmux new -s ssda
   conda activate ssda
   python scripts/run_batch.py exp001_rubric_cross exp001_rubric_long \
       | tee batch.log
   ```
4. **Detach** (`Ctrl-b d`), **disconnect SSH freely**, **come back** to a finished batch. Reconnect via `ssh + tmux a -t ssda` and run `python scripts/status.py` to see what's done. Network drops do not affect the batch.
5. **Re-running the same `run_batch.py` command resumes** — anything with `done.flag` is silently skipped, anything with `failed.flag` is retried.

## Tests

```bash
pytest                       # 88 unit tests, ~30 seconds (live tests skipped)
pytest -m live_eval          # +2 slow tests that shell out to ssdatabench eval
RUN_LIVE_LLM_TESTS=1 pytest  # +1 test that hits the real LLM API
```

## Key design decisions

1. **The agent generates a model, not individual cases.** On the tool-using path the model is a `Chain` of fit Steps, built incrementally via tool calls and serialized to `chain.json` + sampled at commit. On the legacy code-block path the model is executable Python the agent writes to `model.pkl`. In both cases the orchestrator — not the LLM — runs the final `sample(n)`.
2. **Two execution paths share one orchestrator.** `PromptVariant.is_tool_using` decides at run start. The legacy path runs the agent's code in fresh `python` subprocesses with a shared workspace `cwd`; the tool-using path runs every tool in-process against a `RuntimeState` and never spawns subprocesses. A failure in either is caught and persisted to `tool_calls.json` / `step_NN.{stdout,stderr}` so the agent can self-correct.
3. **Two splits, two purposes.** SSDataBench's eval split (default 50/50, fixed seed) is the *external* boundary the agent never sees. Inside the tool-using loop the orchestrator further splits `train` 80/20: `train_fit` for fit-family tools and `held_out` for verify-family tools. Verify scores are local proxies for SSDataBench T1/T2/T4 — same metrics, different sample.
4. **SSDataBench is called as a subprocess.** We don't import its code; we shell out to its evaluation scripts. Survives any internal refactor of theirs.
5. **Schema drift is auto-handled.** SSDataBench's GSS-2018 evaluation expects ~28 variables; our cleaned `gss_clean.csv` has only 11. `scripts/build_eval_subset.py` regenerates a pruned config at runtime.
6. **Everything is logged.** Every prompt, response, tool call, tool result, code execution, and output is saved per-run for reproducibility and qualitative analysis.
7. **Verify-tool design is destiny.** A summary verify (e.g. `score_overall`) that omits a metric the rubric measures will train the agent to ship chains that score high on the visible part and low on the hidden one. Whenever a new structural rubric type lands, the corresponding verify tool plus a `commit_generator` gate goes in alongside it. EXP-006e's chronology gate is the canonical example.
8. **Resilient to transient errors.** OpenAI client retries `APIConnectionError`, rate limits, and 5xx with exponential backoff. The runner isolates per-condition failures so one bad run doesn't kill the matrix.

## Status

| Area | Status |
|---|:---:|
| Tool-using agent loop (`rubric_tools_v3`, current default) — 16 tools across inspect/fit/verify/commit; commit_generator hard-gated on chronology | ✓ |
| Legacy 4-stage code-block orchestrator (`baseline`, `rubric` variants) | ✓ |
| Cross-sectional pilots (GSS/CPS/ACS) | ✓ |
| Longitudinal pilots (NLSY/AddHealth/CFPS/US) — adds T4/T5 | ✓ |
| Direct-generation baseline (paper paradigm) | ✓ |
| Three-model comparison (DeepSeek / gpt-5.4-mini / gpt-5.4) | ✓ |
| Paper-comparison reporting (vs SSDataBench's 15 LLMs) | ✓ |
| Matrix-of-variants framework (prompt registry + per-exp LLM overrides) | ✓ (this PR) |
| Batch runner + status + per-exp report (cloud-deploy ready) | ✓ (this PR) |
| EXP-001 rubric variant — `exp001_rubric_*` | wired, smoke tested, full run pending |

Backlog (full list in [`docs/experiments/STRATEGY.md`](docs/experiments/STRATEGY.md)):

- **EXP-001** rubric block in SYSTEM_PROMPT (in flight)
- **EXP-002** quantitative VALIDATION thresholds (replace "if anything is clearly off")
- **EXP-003** wire `preserve_missingness=True` (currently dead code)
- **EXP-004** MODELING decision rule branching cross-sectional vs longitudinal
- **EXP-005** cross-run lessons memory injected into SYSTEM_PROMPT

## License

Research code; no license declared yet. Contact the authors before redistribution.
