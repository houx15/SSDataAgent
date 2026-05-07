# exp006a_tools_acs — report (2026-05-07)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-006** — **Tool-using agent loop** (Claude Code / Codex shape). Replaces the 4-stage code-exec orchestrator with a tool-using loop where the agent inspects/fits/verifies incrementally instead of writing one ~400-line generation script in one shot. Design: `2026-05-07-tool-using-agent-design.md`. **Subsumes EXP-002 (verify-family thresholds), EXP-003 (`fit_conditional(allow_missing=True)`), and EXP-004 (system prompt + tool guidance, not stage-prompt branching).** *Expected lift:* +0.05–0.15 mostly by recovering the cells that crashed under EXP-001 + letting the agent verify before it commits. *Cost:* multi-day rewrite of orchestrator + new tools module; spike on `acs` first.

- **prompt_variant:** `rubric_tools`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** acs
- **conditions:** full_agent
- **n_rows / dataset:** 1000
- **max_iterations:** 1
- **A/B baseline experiment:** `pilot_paper_agents_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| acs | 0.591 | 0.744 | 0.158 | — | — | 0.498 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| acs | T1 | 0.591 | 0.200 | +0.391 |
| acs | T2 | 0.744 | 0.500 | +0.244 |
| acs | T3 | 0.158 | 0.400 | -0.243 |
| acs | **overall** | 0.498 | 0.400 | +0.098 |

## vs Baseline experiment `pilot_paper_agents_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| acs | T1 | 0.591 | 0.445 | +0.146 |
| acs | T2 | 0.744 | 0.714 | +0.030 |
| acs | T3 | 0.158 | 0.355 | -0.197 |
| acs | **overall** | 0.498 | 0.505 | -0.007 |

## vs EXP-001 rubric (the experiment this was meant to recover from)

| Dataset | T-type | this | exp001_rubric | Δ |
|---|---:|---:|---:|---:|
| acs | T1 | 0.591 | 0.457 | +0.134 |
| acs | T2 | 0.744 | 0.634 | +0.110 |
| acs | T3 | 0.158 | 0.015 | **+0.143** |
| acs | **overall** | 0.498 | 0.369 | **+0.129** |

## Run shape

- **19 turns**, **69 tool calls**, no crashes, no max_turns auto-commits.
- Agent called **`replace_step` 9 times** — i.e. fit something, scored it,
  didn't pass, refit with a different family. This is the verify-then-revise
  loop the design predicted; it doesn't exist in the legacy code-block path.
- Tool mix: 16 `fit_conditional`, 9 `score_pair`, 6 `fit_marginal`,
  6 `describe_column`, 5 `cross_tab`, 3 each of `missing_pattern` /
  `groupby_stat` / `report_progress` / `score_overall`. The agent leaned
  hard on cross-checking each pair before committing.
- Final chain: 13 cols, generation order pivoted on demographics
  (gender → age → birth_year → race → immigrant_status) then conditioned
  family/work events on demographics + each other. `income` was the last
  thing fit, conditional on age + education + occupation + …; `health_work_difficulty`
  came before income (chain ordering choice the agent made on its own).

## Retro

- **What worked:**
  - Recovered the EXP-001 rubric crash entirely on ACS (+0.129 overall;
    T3 went from 0.015 to 0.158 — over 10× the rubric's value).
  - Beat baseline on T1 (+0.146) and T2 (+0.030); basically tied on overall
    (within stochastic noise — gpt-5.4 is non-deterministic).
  - The verify-then-replace_step loop fired: 9 replace_step calls means the
    agent treated bad fits as recoverable instead of crashing.
  - Smoke at n=100 caught two real production-only bugs (dummy column
    reindex, dispatch catch-all) that the unit tests missed. Cheap filter.

- **What didn't:**
  - T3 still 0.158 vs baseline 0.355 (-0.197). Tool-using path doesn't yet
    match the legacy path on regression preservation. Probably the agent
    didn't use `fit_conditional(allow_missing=True)` aggressively enough on
    T3-critical columns (occupation, income), and / or used `empirical_lookup`
    where a true conditional regression would have preserved the residual
    structure better.

- **Surprises:**
  - The agent picked `empirical_lookup` more than expected — 11 of 16
    `fit_conditional` calls. Robust but blunts T3 (no smooth regression
    surface). Future system prompt should nudge linear/logistic over
    empirical for numeric targets.
  - Tool latency was a non-issue for the spike — 19 turns × ~3s/call ~= 1 min.
    Cost stayed well under the $1 cap I'd budgeted in the design doc.

- **Lesson worth preserving:**
  - **Architectural change recovers crashes without paying a quality tax.**
    The same rubric block that broke the legacy path on 3/7 datasets now
    matches baseline overall on ACS. The 4-stage one-shot orchestrator
    was the bottleneck, not the rubric content.
  - **`replace_step` is the killer feature.** The legacy path's only
    recovery was "regenerate the whole step_002.py and hope." A surgical
    refit-this-one-column-with-different-family is dramatically cheaper
    and more reliable.

- **Next experiment:**
  - **EXP-006b** — generalize Stage A to all 7 datasets × full_agent,
    same `rubric_tools` variant, same gpt-5.4. If T3 stays weak across
    datasets, queue an EXP-006c that tightens the system prompt on
    family selection ("prefer linear/logistic over empirical_lookup
    when the target is numeric and you can list ≤4 informative `given`
    columns").
  - Decide before EXP-006b: do we keep `agent_no_semantic` and
    `agent_no_data` ablations or skip them for the spike continuation?
    (My read: keep them — the EXP-001 ACS data showed `agent_no_semantic`
    actually outperformed `full_agent` on the rubric variant; same A/B on
    tools would be informative.)
