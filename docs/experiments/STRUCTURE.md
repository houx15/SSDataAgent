# Experiments structure

> How experiments are organized in this repo — from a hypothesis in
> `STRATEGY.md` to a row in `LEDGER.md`. Use this as the map when
> adding a new experiment or reading an old one.

## Two-axis docs: forward vs backward

Everything in `docs/experiments/` falls on one of two axes.

- **Forward-looking — `STRATEGY.md`**. The current state of our beliefs
  about why the agent scores what it scores and what to try next.
  Edited continuously. Backlog of experiments in expected-lift order.
  Persistent "Lessons" section for things that have stayed true across
  multiple runs.
- **Backward-looking — `LEDGER.md`**. Append-only, one row per
  experiment that actually ran. Headline number + link to the retro.
  Newest on top. The receipt.

Per-experiment files sit between the two:

- `YYYY-MM-DD-<exp_name>-report.md` — the retro for a finished run.
  Filed via `_template.md`. Linked from LEDGER.
- `RUN-EXP-<NNN[letter]>.md` — the runbook for a cloud-bound experiment.
  Written *before* the run; tells you exactly what to type. Removed or
  archived after the experiment is done and the retro exists.
- `<YYYY-MM-DD>-<exp_name>_<scope>-report.md` — same convention as
  retros, used for narrow diagnostic write-ups (e.g.
  `2026-05-11-exp006e_cps_t3_regression.md`).
- `CLOUD_SETUP.md` — one-time cloud-box bootstrap (conda env, tmux,
  submodule). Cited from every runbook.
- `_template.md` — frontmatter + section headings for a retro.

## Experiment naming

```
expNNN[letter]_<variant>_<scope>
```

- `NNN` — sequence number from STRATEGY's backlog (001, 006, …). Once
  assigned it doesn't move, even if the experiment is split into
  letter-suffixed sub-runs.
- `letter` — sub-run within a number, in time order: `a` (spike), `b`
  (broaden), `c` (refine), `d` (ablation), `e` (re-fix), `f`
  (diagnostic). Use letters when the same hypothesis-family produces
  multiple cloud runs.
- `variant` — what's under test: `rubric`, `tools`, `rubric_tools_v2`,
  etc. Mirrors `prompt_variant` in the yaml.
- `scope` — which datasets: `cross` (gss/cps/acs), `long`
  (nlsy/addhealth/cfps/us), `acs`/`nlsy` (single-dataset spike),
  `ablation_cross`, `diag`.

Examples: `exp006a_tools_acs` (spike), `exp006b_tools_long`
(broaden), `exp006c_tools_cross` (refine), `exp006f_tools_diag`
(diagnostic on a subset that crashed in `exp006e_tools_long`).

`smoke_*` and `pilot_*` are reserved prefixes:

- `smoke_*` — cheap (<$0.10), single-dataset, fast confirmation that
  the deployed code path works. Always paired with the next experiment.
- `pilot_*` — paper-baseline reproductions (`pilot_paper_agents_gpt54`
  etc.). Used as `--baseline` for report generation.

## How a run is configured

All experiments are declared in **`config/experiments.yaml`** under the
top-level `experiments:` key. Each entry is parsed into
`ExperimentConfig` (`src/ssdataagent/experiments/runner.py:22`):

```yaml
exp006f_tools_diag:
  datasets: [cfps, addhealth]      # which sampled_*.csv files to load
  conditions: [full_agent]         # which context shapes to test
  max_iterations: 1                # outer validation cycles (legacy 4-stage path)
  sandbox_timeout: 90              # per-sandbox-exec seconds (legacy path)
  train_eval_split: 0.5            # train/eval split ratio (seed=42)
  n_rows: 1000                     # rows to generate per dataset/condition
  prompt_variant: rubric_tools_v3  # registry key in prompt_templates.py
  llm_model: gpt-5.4-2026-03-05    # passed through to llm overrides
  llm_provider: openai
  llm_base_url: https://api.openai.com/v1
```

The orchestrator's `max_turns` (tool-using loop cap, default 40) is
**not** in the yaml — it's a constant in `orchestrator.py`. The yaml's
`max_iterations` is the legacy-stage cycle count and is effectively
ignored by tool-using variants.

## Conditions (context shapes)

`conditions` controls what context the agent gets, not the prompt
variant. Three are defined:

- `full_agent` — agent sees data preview + semantic descriptions.
- `agent_no_semantic` — agent sees data preview only, no descriptions.
- `agent_no_data` — agent sees descriptions only, no data preview.
- (`unseen` — used by some pilots; agent gets descriptions minus a
  held-out target variable.)

A condition × dataset pair is a "cell". An experiment with
`datasets: [gss, cps, acs]` and `conditions: [full_agent,
agent_no_semantic, agent_no_data]` produces 9 cells.

## Running an experiment

```bash
# Single experiment (one yaml entry):
python scripts/run_experiment.py --experiment exp006f_tools_diag

# Multiple experiments in sequence (one process per experiment,
# isolated failure handling, resume-aware):
python scripts/run_batch.py exp006e_tools_long exp006e_tools_cross

# Status of one or more experiments:
python scripts/status.py exp006e_tools_long exp006e_tools_cross

# Generate a markdown report comparing against a pilot baseline:
python scripts/generate_exp_report.py exp006e_tools_long \
    --baseline pilot_paper_longitudinal_gpt54
```

Resume semantics: `run_experiment.py --resume` re-runs only the cells
that don't yet have a `done.flag`. `run_batch.py` skips experiments
whose `done.flag` already exists and re-attempts ones with `failed.flag`.

Cloud-box runs use **tmux + conda `ssda`** (see `CLOUD_SETUP.md`) —
**never** `nohup` and **never** `.venv` on the cloud box.

## Results layout

```
results/<exp_name>/
├── done.flag              # JSON stamp; written only after summary.csv lands
├── failed.flag            # JSON stamp w/ traceback if the run blew up
├── summary.csv            # condition × dataset × type → pass_rate (long form)
├── run.log                # combined stdout/stderr of run_experiment.py
└── <condition>/<dataset>/<run_id>/
    ├── meta.json          # exp/dataset/condition/run_id/git_sha/model/provider
    ├── eval.json          # raw rubric outputs for this cell
    ├── generated.csv      # the n_rows synthetic sample
    ├── prompts.jsonl      # user/system/tool prompts seen by the LLM
    ├── responses.jsonl    # assistant turns
    ├── code/              # any executable code the agent emitted (legacy path)
    └── workspace/         # tool-using path artifacts:
        ├── chain.json     # final generation_order + per-step (col, family, given)
        ├── tool_calls.json # ordered list of every tool call (args + result)
        ├── transcript.json # role/content pairs across the loop
        ├── progress.log   # narrative notes from report_progress
        ├── train.csv      # the train half the agent saw
        ├── descriptions.json # semantic context provided
        └── generated.csv  # final sample (copied to the parent run_dir too)
```

`<run_id>` is `YYYYMMDD-HHMMSS`. Re-runs of the same dataset/condition
get a new run_id; the latest sorts last, so reading with
`sorted(os.listdir(...))[-1]` gives you the freshest.

A few things worth knowing for forensics:

- **`tool_calls.json` + `transcript.json` are now persistence-guaranteed**
  even on a crashed cell (commit `b604880`, EXP-006f). Before that, a
  raise during force-commit lost both.
- **`chain.json` carries `t4_unverified`** when EXP-006f's soft-fail
  gate fired. Reporting code should treat `t4_unverified=True` as "T4
  is expected to be penalized" rather than a clean number.
- **`done.flag`'s `prompt_variant`+`llm_model` are authoritative** for
  what actually ran, in case the yaml changes after the fact.

## Lifecycle of an experiment

```
                    ┌────────────────────────┐
   hypothesis  ──▶  │ STRATEGY.md backlog    │  (one bullet, expected lift)
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼───────────┐
                    │ config/experiments.yaml │  (yaml entry under experiments:)
                    └────────────┬───────────┘
                                 │
                    optional ┌───▼───┐
                    cloud ─▶ │ RUN-* │  (runbook with exact commands)
                             └───┬───┘
                                 │
                    ┌────────────▼───────────┐
                    │ scripts/run_experiment │  → results/<exp_name>/
                    │ scripts/run_batch       │    done.flag + summary.csv
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼───────────┐
                    │ generate_exp_report.py  │  (compares to a baseline)
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼───────────┐
                    │ YYYY-MM-DD-<exp>-report │  (retro via _template.md)
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼───────────┐
                    │ LEDGER.md row + update  │  (newest on top)
                    │ STRATEGY.md             │  (mark done; add follow-up)
                    └────────────────────────┘
```

The two terminal docs (retro + LEDGER row) are the contract. If they
exist, the experiment is done.

## Adding a new experiment — short checklist

1. Bump STRATEGY's backlog with a one-line "Expected lift × inverse
   cost" bullet. Pick an `expNNN[letter]_<variant>_<scope>` name.
2. Add a yaml entry under `experiments:` in
   `config/experiments.yaml`. Mirror the closest existing entry — only
   change what's under test.
3. If the run is cloud-bound, add `RUN-EXP-<N>.md` modeled on
   `RUN-EXP-006f.md`: pull command, smoke (if useful), main run,
   what to look for, pull-back rsync, pitfalls.
4. Smoke locally first when a code path changed. Smoke configs are
   single-dataset, n_rows ≤ 100.
5. Run. Pull results to laptop (`rsync -av --progress`).
6. Generate the report with `generate_exp_report.py
   --baseline <pilot_…>`, file the retro from `_template.md`.
7. Add a `LEDGER.md` row. Update STRATEGY (`[x]` the backlog item,
   add follow-up `[ ]` if there is one, promote any cross-experiment
   lesson to the "Lessons" section).

## Files you'll touch most

| file | when |
|---|---|
| `config/experiments.yaml` | every new experiment |
| `docs/experiments/STRATEGY.md` | planning + after-action |
| `docs/experiments/LEDGER.md` | after each run finishes |
| `docs/experiments/_template.md` | copy → retro |
| `docs/experiments/RUN-EXP-*.md` | cloud-bound runs |
| `scripts/run_experiment.py` | the entry point |
| `scripts/generate_exp_report.py` | post-run comparison |
| `src/ssdataagent/agent/prompt_templates.py` | when the change under test is a prompt variant |
| `src/ssdataagent/agent/orchestrator.py`, `src/ssdataagent/agent/tools/*.py` | when the change is in the loop or tools |
