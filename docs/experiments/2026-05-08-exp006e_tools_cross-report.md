# exp006e_tools_cross — report (2026-05-08)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-006** — **Tool-using agent loop** (Claude Code / Codex shape). Replaces the 4-stage code-exec orchestrator with a tool-using loop where the agent inspects/fits/verifies incrementally instead of writing one ~400-line generation script in one shot. Design: `2026-05-07-tool-using-agent-design.md`. **Subsumes EXP-002 (verify-family thresholds), EXP-003 (`fit_conditional(allow_missing=True)`), and EXP-004 (system prompt + tool guidance, not stage-prompt branching).** *Expected lift:* +0.05–0.15 mostly by recovering the cells that crashed under EXP-001 + letting the agent verify before it commits. *Cost:* multi-day rewrite of orchestrator + new tools module; spike on `acs` first.

- **prompt_variant:** `rubric_tools_v3`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** gss, cps, acs
- **conditions:** full_agent
- **n_rows / dataset:** 1000
- **max_iterations:** 1
- **A/B baseline experiment:** `pilot_paper_agents_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| gss | 0.572 | 0.686 | 0.330 | — | — | 0.529 |
| cps | 0.546 | 0.775 | 0.147 | — | — | 0.489 |
| acs | 0.634 | 0.595 | 0.000 | — | — | 0.409 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| gss | T1 | 0.572 | 0.130 | +0.442 |
| gss | T2 | 0.686 | 0.710 | -0.024 |
| gss | T3 | 0.330 | 0.550 | -0.220 |
| gss | **overall** | 0.529 | 0.390 | +0.139 |
| cps | T1 | 0.546 | 0.100 | +0.446 |
| cps | T2 | 0.775 | 0.710 | +0.065 |
| cps | T3 | 0.147 | 0.570 | -0.423 |
| cps | **overall** | 0.489 | 0.400 | +0.089 |
| acs | T1 | 0.634 | 0.200 | +0.434 |
| acs | T2 | 0.595 | 0.500 | +0.095 |
| acs | T3 | 0.000 | 0.400 | -0.400 |
| acs | **overall** | 0.409 | 0.400 | +0.009 |

## vs Baseline experiment `pilot_paper_agents_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| gss | T1 | 0.572 | 0.220 | +0.353 |
| gss | T2 | 0.686 | 0.651 | +0.035 |
| gss | T3 | 0.330 | 0.407 | -0.078 |
| gss | **overall** | 0.529 | 0.426 | +0.103 |
| cps | T1 | 0.546 | 0.573 | -0.027 |
| cps | T2 | 0.775 | 0.743 | +0.033 |
| cps | T3 | 0.147 | 0.457 | -0.310 |
| cps | **overall** | 0.489 | 0.591 | -0.102 |
| acs | T1 | 0.634 | 0.445 | +0.189 |
| acs | T2 | 0.595 | 0.714 | -0.119 |
| acs | T3 | 0.000 | 0.355 | -0.355 |
| acs | **overall** | 0.409 | 0.505 | -0.095 |

## Retro

- **What worked:**
- **What didn't:**
- **Surprises:**
- **Lesson worth preserving:**
- **Next experiment:**
