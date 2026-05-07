---
exp_name: exp006_tool_using_agent
date: 2026-05-07
model: gpt-5.4-2026-03-05
git_sha: _pending_
baseline_exp: pilot_paper_agents_gpt54 (cross) / pilot_paper_longitudinal_gpt54 (long) / exp001_rubric_* (rubric A/B)
status: planned
hypothesis: Replacing the 4-stage exec'd-script orchestrator with a tool-using agent loop (Claude Code / Codex style) eliminates the "ambitious-code-with-runtime-bugs" failure mode seen in EXP-001 and lifts overall mean by +0.05 to +0.15 across all conditions, primarily by recovering the cells that crashed under the rubric and by letting the agent verify before it commits.
---

# exp006_tool_using_agent — design

> **This is a project-wide architecture change, not a single prompt swap.** It supersedes the 4-stage `EXPLORATION → MODELING → VALIDATION → GENERATION` flow currently in `src/ssdataagent/agent/orchestrator.py`. The condition system (`full_agent`, `agent_no_semantic`, `agent_no_data`, `full_agent_unseen`), the experiment runner, and the SSDataBench scoring path are unchanged.

## Hypothesis

Three observations from EXP-001 motivate this:

1. **The rubric reaches the LLM but pushes it into code it can't get right.** Every successful `step_002.py` from `exp001_rubric_*` references T1–T5 in comments. Three cells (gss, cps, addhealth) failed in code execution rather than for lack of trying — `~(miss | structural_na)` type mismatch, `"No Child"` string written into a `float64` column, no code block on retry.
2. **`agent_no_semantic` outperforms `full_agent` on ACS by +0.27 overall on the same rubric.** Stripping `descriptions.json` removes the structural rules the agent overcommits to; the simpler empirical strategy ships working code.
3. **The agent never gets to test small pieces of its idea before committing.** Today it must produce one ~400-line `step_002.py` that fits the entire generative model in one shot, then a separate `step_003.py` that validates everything, with no way to walk back a bad fit incrementally.

A tool-using loop addresses all three. The agent calls small tools (inspect this column, fit this conditional, score this marginal against held-out data) and assembles the generator piece by piece, with the runtime owning the in-progress state across turns. Failures become localized — a bad `fit_conditional` returns a poor verify score and the agent retries that one piece, instead of crashing the whole pipeline.

## Architecture

### The loop (Claude Code / Codex shape)

The orchestrator becomes a single tool-using loop, not 4 stages:

```
loop:
    response = llm.chat(history, system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS)
    if response has no tool_calls:
        if model has been committed: break (success)
        else: nudge "you must call a tool; emit `commit_generator` when done"
    for each tool_call in response:
        result = execute_tool(tool_call.name, tool_call.arguments, runtime_state)
        history.append(tool_result)
    if turn_count > MAX_TURNS: raise (failure)
```

`runtime_state` is a Python object held by the orchestrator across turns. It owns:

- The held-out 20% slice (split once, deterministically, when the loop starts).
- The in-progress generative chain (an ordered list of `Step` objects: marginal fits, conditional fits, sampling order).
- A frozen view of the train.csv (the agent's tools read from it but never mutate).

When the agent calls `commit_generator()`, the runtime ends the loop, samples N=1000 rows from the chain, writes `generated.csv`, and returns to the experiment runner — same downstream contract as today.

### Prompt scaffold

One system prompt, not four stage prompts. Skeleton:

```
You are a data analyst building a generative model that matches a target survey
distribution. Your job is to fit a generator that scores well on T1 (univariate
distributions), T2 (pairwise associations), T3 (regression preservation), T4
(event-time chronology), and T5 (event-order × covariate) — see rubric below.

You have tools to inspect data, fit pieces of the generator, and verify each
piece against a held-out slice before committing. Use them. Do not write
generation logic from scratch.

Workflow you should follow:
  1. inspect: list_columns, describe_column, missing_pattern on suspicious vars.
  2. fit incrementally: start with marginals, then conditionals; declare a
     generation order before fitting conditionals on later columns.
  3. verify: after each fit, call score_marginal / score_pair / score_event_order
     to confirm the piece works. If it doesn't, swap the family or refit.
  4. commit: when sample_preview looks right, call commit_generator().

[T1-T5 rubric block — same content as the `rubric` PromptVariant]
```

The `agent_no_semantic` condition gets the same prompt but `descriptions.json` is not appended; tools that would read it (e.g. `list_columns` returning per-column human descriptions) return only schema-derived metadata. `agent_no_data` gets descriptions but tools that read `train.csv` return `{"error": "data withheld in this condition"}`.

### Tool surface

Tools are grouped into four families. JSON schemas are in `src/ssdataagent/agent/tools/schemas.py` (to be created); the contract here is the tool name, what the runtime executes, and what it returns to the agent.

#### Inspect family — read-only on `train.csv`

| Tool | Input | Output |
|---|---|---|
| `list_columns` | — | `[{name, dtype, n_unique, n_missing, missing_rate}]` |
| `describe_column` | `col` | numeric: `{mean, std, min, max, q25, q50, q75, n_missing, missing_rate}`; categorical: `{value_counts: [(val, count)], n_categories, n_missing}` |
| `cross_tab` | `col1, col2, normalize=False` | 2D table as `{rows: [...], cols: [...], counts: [[...]]}` |
| `missing_pattern` | `cols: [str]` | `[{pattern: "1010", count, fraction}]` (1 = present, 0 = NA) |
| `correlation` | `col1, col2, method` | `{coef, n}` |
| `groupby_stat` | `group_col, value_col, stat` | `[{group_value, stat_value, n}]` |
| `head_rows` | `n=5` | first N rows of `train.csv` (same as `df.head()` printed) |

These do not change runtime state. They're stateless reads.

#### Fit family — builds the generative chain incrementally

| Tool | Input | Effect |
|---|---|---|
| `set_generation_order` | `cols: [str]` | declares column order; required before any `fit_conditional` |
| `fit_marginal` | `col, family ∈ {empirical, kde, normal, categorical_empirical}` | fits and registers; returns `{aic_or_loglik, n_params}` |
| `fit_conditional` | `col, given: [str], family ∈ {empirical_lookup, kde, linear_regression, logistic_regression, decision_tree}, allow_missing: bool` | fits and registers; returns `{r2_or_acc, n_params}` |
| `fit_copy_real` | `col` | sentinel: at sample time, draws this column iid from the empirical distribution (useful for unseen-variable fallback) |
| `replace_step` | `col` | drops the existing fit for `col` so the agent can refit it |

The runtime maintains a `Chain` object internally. Calls accumulate; out-of-order calls (e.g., `fit_conditional("X", given=["Y"])` before `Y` is registered) return an error in the tool result, which the agent can correct.

#### Verify family — scores the in-progress chain against held-out 20%

| Tool | Input | Output |
|---|---|---|
| `sample_preview` | `n=100` | column summaries for a sample drawn from the current chain — same shape as `describe_column` so the agent can compare |
| `score_marginal` | `col` | `{ks_stat, p_value, pass}` (numeric) or `{chi2, p_value, tv_distance, pass}` (categorical) |
| `score_pair` | `col1, col2` | `{real_corr, sim_corr, abs_diff, pass}` |
| `score_event_order` | `events: [str]` | `{compliance_rate, expected, pass}` (longitudinal only) |
| `score_overall` | — | runs all of the above on every fitted column / pair declared in the chain; returns a table |

`pass` thresholds match EXP-002's planned hard thresholds: TV ≤ 0.10 for T1 categorical, |Δr| ≤ 0.15 for T2, etc. These are *proxies* for SSDataBench's full scoring, not replacements — A/B in the spike confirms they correlate well with the actual T1-T5 scores.

#### Commit family

| Tool | Input | Effect |
|---|---|---|
| `commit_generator` | — | ends loop; runtime samples N rows from the chain → `generated.csv` |
| `report_progress` | `message: str` | free-form note appended to `run.log`; no effect on state |

### Runtime state and serialization

`runtime_state` lives in memory during a run and is serialized to the workspace at the end so failures stay debuggable:

```
workspace/
    train.csv                  (existing — unchanged)
    descriptions.json          (existing — unchanged, condition-dependent)
    held_out.csv               (NEW — 20% slice for verify tools)
    chain.json                 (NEW — serialized Chain at end of run)
    transcript.json            (NEW — full tool-using conversation, replaces step_NNN.* files)
    generated.csv              (existing — final output)
```

`chain.json` shape (illustrative):

```json
{
  "generation_order": ["age", "gender", "education", "income"],
  "steps": [
    {"col": "age",       "kind": "marginal",    "family": "kde",                 "params": {...}},
    {"col": "gender",    "kind": "marginal",    "family": "categorical_empirical", "params": {...}},
    {"col": "education", "kind": "conditional", "given": ["age", "gender"], "family": "decision_tree", "params": {...}},
    {"col": "income",    "kind": "conditional", "given": ["age", "education"], "family": "linear_regression", "params": {...}}
  ]
}
```

Replays from `chain.json` should be deterministic enough that re-running `commit_generator` from a saved chain produces the same `generated.csv` ± seed.

## How conditions still apply

The condition flag still controls two binary inputs to the agent's environment, just routed through the prompt and tools instead of through which files the agent reads:

| condition | descriptions in system prompt? | inspect-family tools see train.csv? |
|---|---|---|
| `full_agent` | yes | yes |
| `agent_no_semantic` | no | yes |
| `agent_no_data` | yes | no (returns `{"error": "data withheld"}`) |
| `full_agent_unseen` | yes (minus unseen vars) | yes (with unseen cols dropped) |
| `direct_generation` | not an agent run; unchanged |

`build_context` in `src/ssdataagent/agent/context.py` keeps its current behavior; the orchestrator just consults the same flags when assembling the system prompt and constructing the toolbox.

## Files touched

**Rewrite:**
- `src/ssdataagent/agent/orchestrator.py` — replace 4-stage code-exec flow with tool loop. Net likely smaller, not larger.
- `src/ssdataagent/agent/prompt_templates.py` — collapse 4 per-stage prompts into one system prompt + tool description block. Keep the `rubric` variant; it now affects tool-selection guidance rather than per-stage text.
- `src/ssdataagent/agent/llm_client.py` — add `chat_with_tools(history, system, tools)` returning structured tool calls; existing `chat()` stays for non-agent codepaths.

**New:**
- `src/ssdataagent/agent/tools/__init__.py` — registry mapping tool name → callable
- `src/ssdataagent/agent/tools/schemas.py` — JSON schemas for each tool (consumed by OpenAI function-calling)
- `src/ssdataagent/agent/tools/inspect.py` — inspect-family impls
- `src/ssdataagent/agent/tools/fit.py` — fit-family impls + `Chain` class
- `src/ssdataagent/agent/tools/verify.py` — verify-family impls; KS / χ² / corr-diff calculators
- `src/ssdataagent/agent/tools/commit.py` — `commit_generator` + chain → samples logic

**Untouched:**
- `src/ssdataagent/experiments/runner.py`, `experiments/conditions.py`, `data/schema.py`, `evaluation/runner.py`, `agent/sandbox.py` (still useful for `direct_generation` and one-off scripts)
- All dataset configs, `config/experiments.yaml`, `config/datasets.yaml`
- The condition flags (`Condition.FULL`, etc.) and their meaning
- SSDataBench scoring path

## Spike plan

Two stages. **Don't generalize until the spike scores ≥ baseline on ACS.**

### Stage A — single-dataset spike (target: 1–2 days)

1. Implement the tool surface for inspect + fit + verify + commit on `acs` only.
2. Implement the orchestrator loop with a hardcoded `MAX_TURNS=40`.
3. Add a new prompt variant `rubric_tools` (same rubric block, tool-using workflow guidance instead of stage prompts).
4. Add a new yaml entry `exp006a_tools_acs` (1 dataset × `full_agent` only, gpt-5.4).
5. Run locally on a smoke subset (n_rows=100) to validate the loop end-to-end. Then full n_rows=1000 against the existing baseline.
6. Generate report; compare to `pilot_paper_agents_gpt54[acs]` (full_agent overall=0.505) and `exp001_rubric_cross[acs]` (full_agent overall=0.369).

**Gate:** spike score ≥ baseline AND no crashes. If yes, proceed to stage B. If no, retro and adjust before generalizing.

### Stage B — generalize (target: 1–2 days after stage A)

1. Run on all 7 datasets × full_agent.
2. Add `agent_no_semantic` and `agent_no_data` cells to confirm the conditions still work.
3. Compare against EXP-001 to confirm the failure modes (gss, cps, addhealth crashes) are gone.
4. If the tool-using path consistently matches or beats the legacy path, deprecate the 4-stage orchestrator (keep it behind a flag for a few weeks, then delete).

## Risks and open questions

1. **Latency / cost.** A 4-call run becomes a 20–40-turn run. Tool descriptions in every turn (~500 tokens) inflate input. Rough budget estimate: ~5× current per-cell cost. The cloud bill matters here; let's establish a per-cell cap (e.g. $1) before stage B.
2. **Verify-tool calibration.** If `score_marginal` is too lenient, the agent commits a bad chain. If too strict, it loops past `MAX_TURNS`. We need to A/B verify-pass rates against actual T1/T2 scores in the spike before trusting them.
3. **Max-turns fallback.** What does the runtime do if the agent hits `MAX_TURNS` without `commit_generator`? Proposal: auto-commit whatever chain exists, fall back to `fit_marginal` empirical for any unfitted column. Worst case = empirical-marginal-only generation (≈ `agent_no_semantic` baseline behavior). Logged as a soft failure, not a hard one — the cell still scores instead of crashing.
4. **Function-calling support across providers.** gpt-5.4 supports it natively. DeepSeek v4 flash should via OpenAI-compatible endpoint, but needs verification — we may need a string-protocol fallback for non-tool-capable models. Defer this to a follow-up; the spike runs on gpt-5.4 only.
5. **Tool-call schema validation.** OpenAI returns tool args as JSON-parsed dicts but they may not match our schema (e.g., agent passes a list where we expect a string). Runtime validates and returns an error in the tool result, not as a runtime exception. Prompt nudges the agent toward correct usage.

## Decision points before code

- **Confirmed:** OpenAI function-calling protocol (Claude Code / Codex shape).
- **Confirmed:** This replaces the 4-stage path for all agent conditions; not a separate condition flag.
- **Open (default `MAX_TURNS=40`):** ok with you?
- **Open (default verify-pass thresholds = EXP-002's planned thresholds):** ok with you?
- **Open (spike on `acs` only first vs all 3 cross-sectional):** I'd recommend `acs` only — fastest signal. ok?

## Retro
<!-- filled in after the spike completes -->

- **What worked:**
- **What didn't:**
- **Surprises:**
- **Lesson worth preserving:**
- **Next experiment:**
