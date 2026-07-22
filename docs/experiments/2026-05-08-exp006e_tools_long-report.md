# exp006e_tools_long — report (2026-05-08)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-006** — **Tool-using agent loop** (Claude Code / Codex shape). Replaces the 4-stage code-exec orchestrator with a tool-using loop where the agent inspects/fits/verifies incrementally instead of writing one ~400-line generation script in one shot. Design: `2026-05-07-tool-using-agent-design.md`. **Subsumes EXP-002 (verify-family thresholds), EXP-003 (`fit_conditional(allow_missing=True)`), and EXP-004 (system prompt + tool guidance, not stage-prompt branching).** *Expected lift:* +0.05–0.15 mostly by recovering the cells that crashed under EXP-001 + letting the agent verify before it commits. *Cost:* multi-day rewrite of orchestrator + new tools module; spike on `acs` first.

- **prompt_variant:** `rubric_tools_v3`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** nlsy, addhealth, cfps, us
- **conditions:** full_agent
- **n_rows / dataset:** 1000
- **max_iterations:** 1
- **A/B baseline experiment:** `pilot_paper_longitudinal_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| nlsy | 0.346 | 0.789 | 0.350 | 0.165 | 0.677 | 0.465 |
| addhealth | — | — | — | — | — | (no eval) |
| cfps | — | — | — | — | — | (no eval) |
| us | 0.436 | 0.618 | 0.266 | 0.290 | 0.763 | 0.475 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| nlsy | T1 | 0.346 | 0.130 | +0.216 |
| nlsy | T2 | 0.789 | 0.710 | +0.079 |
| nlsy | T3 | 0.350 | 0.420 | -0.070 |
| nlsy | T4 | 0.165 | 0.050 | +0.115 |
| nlsy | T5 | 0.677 | 0.750 | -0.073 |
| nlsy | **overall** | 0.465 | 0.300 | +0.165 |
| addhealth | — | (no eval) | — | — |
| cfps | — | (no eval) | — | — |
| us | T1 | 0.436 | 0.200 | +0.236 |
| us | T2 | 0.618 | 0.620 | -0.002 |
| us | T3 | 0.266 | 0.540 | -0.274 |
| us | T4 | 0.290 | 0.050 | +0.240 |
| us | T5 | 0.763 | 0.750 | +0.013 |
| us | **overall** | 0.475 | 0.360 | +0.115 |

## vs Baseline experiment `pilot_paper_longitudinal_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| nlsy | T1 | 0.346 | 0.536 | -0.190 |
| nlsy | T2 | 0.789 | 0.760 | +0.030 |
| nlsy | T3 | 0.350 | 0.552 | -0.202 |
| nlsy | T4 | 0.165 | 0.070 | +0.095 |
| nlsy | T5 | 0.677 | 0.681 | -0.004 |
| nlsy | **overall** | 0.465 | 0.520 | -0.054 |
| addhealth | — | — | 0.438 | — |
| cfps | — | — | 0.358 | — |
| us | — | 0.475 | — | — |

## Retro

- **What worked:**
- **What didn't:**
- **Surprises:**
- **Lesson worth preserving:**
- **Next experiment:**
