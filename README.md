# SSDataAgent

An LLM **data-analyst agent** for population-level social-survey simulation, evaluated against [SSDataBench](https://github.com/lszshu/SSDataBench).

Instead of prompting an LLM to generate one synthetic respondent at a time (the SSDataBench paradigm), this project gives the LLM access to real survey data and lets it explore, model, and generate via Python code execution. The agent functions as a data scientist, not a survey respondent.

## TL;DR — cross-dataset findings (n=1000 each)

Mean pass rate (overall, types 1–3) by condition × dataset:

| Condition | GSS-2018 | CPS-1980 | ACS-1980 |
|---|---:|---:|---:|
| `full_agent` | **0.631** | failed¹ | 0.443 |
| `agent_no_semantic` | 0.506 | 0.544 | **0.655** |
| `agent_no_data` | 0.181 | 0.175 | 0.179 |
| `direct_generation` | 0.273 | 0.200 | 0.189 |

¹ *CPS `full_agent` failed in two independent runs on different stages — orchestrator stability work pending.*

The robust cross-dataset finding: **`agent_no_semantic` (data only) beats `direct_generation` (per-individual prompting) on every dataset**, with the margin growing from +0.23 → +0.34 → +0.47. Data-driven empirical sampling is uniformly stronger than per-individual LLM elicitation, regardless of variable semantics.

The full agent (data + descriptions + context) is the strongest condition only on GSS; on ACS the data-only ablation actually wins. The `agent_no_data` baseline is a near-identical floor (~0.18) on every dataset, confirming that prior knowledge alone is calibration-free.

See [`docs/report/2026-04-30-initial-findings.md`](docs/report/2026-04-30-initial-findings.md) for the full write-up.

## How it works

```
                ┌─────────────────────────────────────────────┐
                │  Orchestrator: explore → model → validate   │
                │                              → generate     │
                └─────────┬───────────────────────────────────┘
                          │ writes code
                ┌─────────▼─────────┐
                │  Sandbox          │   fresh subprocess per code block
                │  (cwd = workspace)│   shared workspace dir for files
                └─────────┬─────────┘
                          │ runs code, returns stdout/stderr
                          ▼
              ┌────────────────────────────┐
              │  Agent's generative model  │   pandas / sklearn /
              │  saved to model.pkl        │   statsmodels — agent's choice
              └────────────────────────────┘
                          │ generates synthetic individuals
                          ▼
                ┌──────────────────────┐
                │  Formatter           │   coerce to schema, write to
                │                      │   sim_profiles_*.csv
                └─────────┬────────────┘
                          ▼
              ┌────────────────────────────┐
              │  SSDataBench evaluation    │   bootstrap statistical tests
              │  (subprocess wrapper)      │   over 5 pattern types
              └────────────────────────────┘
```

The agent runs four stages in a loop:

1. **EXPLORATION** — agent writes EDA code, sees output.
2. **MODELING** — agent fits a generative model (e.g. conditional probability tables, GLMs, copulas) and saves it to `model.pkl`.
3. **VALIDATION** — agent samples from its model and compares to a holdout slice of the training data; up to N retries (default 3) if it self-diagnoses problems.
4. **GENERATION** — agent samples N synthetic individuals and writes `generated.csv`.

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
    experiments.yaml      # experiment matrix
    llm.yaml              # provider/model (no secrets)
  scripts/
    run_experiment.py     # CLI entry point
    smoke_eval.py         # ssdatabench smoke test
    build_eval_subset.py  # auto-prune ssdatabench eval config to our columns
  tests/                  # 90 pytest tests, mirrors src/
  real_data/              # cleaned GSS / CPS / ACS CSVs
  ssdatabench/            # vendored submodule
  docs/
    SPEC.md               # engineering specification
    SSDataBench.pdf       # source benchmark paper
    LLM_Survey_Prediction_Agent.pdf
    report/               # paper-style writeups
    superpowers/          # design spec & implementation plan
  results/                # per-experiment per-condition logs and pass rates
```

## Setup

```bash
git clone --recurse-submodules git@github.com:houx15/SSDataAgent.git
cd SSDataAgent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -r ssdatabench/requirements.txt

cp .env.example .env
# Edit .env: set LLM_API_KEY and adjust LLM_MODEL / LLM_BASE_URL if needed
```

The default `.env` is configured for DeepSeek (`deepseek-v4-flash`, OpenAI-compatible). Any OpenAI-compatible endpoint works (`anthropic` is also supported via the same `LLMClient` interface).

## Usage

Run the full pilot matrix on GSS:

```bash
python scripts/run_experiment.py --experiment pilot_gss
```

Per-run artifacts land in `results/<experiment>/<condition>/<dataset>/<run_id>/`:

```
meta.json            run config + git SHA + model
prompts.jsonl        every message sent to the LLM
responses.jsonl      every response received
code/step_NN.py      every code block the agent executed
code/step_NN.{stdout,stderr,exit}
workspace/           snapshot of the sandbox workspace at end of run
generated.csv        the synthetic dataset handed to evaluation
eval.json            SSDataBench pass rates
```

Resume a partially-completed experiment (skips conditions whose `eval.json` already exists):

```bash
python scripts/run_experiment.py --experiment pilot_gss --resume
```

Available experiments (in `config/experiments.yaml`):

- `smoke_gss` — single `full_agent` run, n=100, ~3 minutes
- `pilot_gss` — full 4-condition matrix, n=1000, ~30 minutes
- `pilot_gss_unseen` — held-out variable experiment

## Tests

```bash
pytest                       # 88 unit tests, ~30 seconds (live tests skipped)
pytest -m live_eval          # +2 slow tests that shell out to ssdatabench eval
RUN_LIVE_LLM_TESTS=1 pytest  # +1 test that hits the real LLM API
```

## Key design decisions

1. **The agent generates a model, not individual cases.** Final agent output is executable Python that produces a pandas DataFrame.
2. **Stateless sandbox + shared workspace.** Each code block runs in a fresh `python` subprocess in a shared `cwd`. State persists via files (CSV / cloudpickle / JSON), never via interpreter state. The system prompt warns the agent about the pickle-by-reference trap.
3. **The agent sees a train split; SSDataBench evaluates on the eval split.** Default 50/50, fixed seed.
4. **SSDataBench is called as a subprocess.** We don't import its code; we shell out to its evaluation scripts. Survives any internal refactor of theirs.
5. **Schema drift is auto-handled.** SSDataBench's GSS-2018 evaluation expects ~28 variables; our cleaned `gss_clean.csv` has only 11. `scripts/build_eval_subset.py` regenerates a pruned config at runtime.
6. **Everything is logged.** Every prompt, response, code execution, and output is saved per-run for reproducibility and qualitative analysis.
7. **Resilient to transient errors.** OpenAI client retries `APIConnectionError`, rate limits, and 5xx with exponential backoff. The runner isolates per-condition failures so one bad run doesn't kill the matrix.

## Status

| Phase | Description | Status |
|---|---|:---:|
| 0 | Project setup, SSDataBench integration, LLM connectivity | ✓ |
| 1 | Data layer (schema, loader, splitter) | ✓ |
| 2 | Sandbox + context builder | ✓ |
| 3 | Agent orchestrator (explore/model/validate/generate) | ✓ |
| 4 | Output formatting + evaluation bridge | ✓ |
| 5 | Experiment runner + CLI | ✓ |
| 6 | Unseen-variable experiment | wired, not yet run |
| 7 | Direct LLM generation baseline | ✓ |

90/90 unit tests passing. One full pilot run completed (above).

Next steps: run `pilot_gss_unseen`; expand to CPS and ACS; add longitudinal datasets per SPEC Phase 3.

## License

Research code; no license declared yet. Contact the authors before redistribution.
