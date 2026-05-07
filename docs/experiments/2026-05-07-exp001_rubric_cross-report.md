# exp001_rubric_cross — report (2026-05-07)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-001** — Add T1–T5 rubric block to `SYSTEM_PROMPT` as a new prompt variant. Yaml entries: `exp001_rubric_cross` + `exp001_rubric_long`. Run on cloud box via `python scripts/run_batch.py exp001_rubric_cross exp001_rubric_long`; report via `scripts/generate_exp_report.py exp001_rubric_cross --baseline pilot_paper_agents_gpt54`. *Expected lift:* +0.03–0.08 across the board. *Status:* in progress (matrix runner + variant landed 2026-05-06).

- **prompt_variant:** `rubric`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** gss, cps, acs
- **conditions:** full_agent, agent_no_semantic, agent_no_data
- **n_rows / dataset:** 1000
- **max_iterations:** 3
- **A/B baseline experiment:** `pilot_paper_agents_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| gss | — | — | — | — | — | (no eval) |
| cps | — | — | — | — | — | (no eval) |
| acs | 0.457 | 0.634 | 0.015 | — | — | 0.369 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| gss | — | (no eval) | — | — |
| cps | — | (no eval) | — | — |
| acs | T1 | 0.457 | 0.200 | +0.257 |
| acs | T2 | 0.634 | 0.500 | +0.134 |
| acs | T3 | 0.015 | 0.400 | -0.385 |
| acs | **overall** | 0.369 | 0.400 | -0.031 |

## vs Baseline experiment `pilot_paper_agents_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| gss | — | — | 0.426 | — |
| cps | — | — | 0.591 | — |
| acs | T1 | 0.457 | 0.445 | +0.012 |
| acs | T2 | 0.634 | 0.714 | -0.080 |
| acs | T3 | 0.015 | 0.355 | -0.340 |
| acs | **overall** | 0.369 | 0.505 | -0.136 |

## Retro

- **What worked:**
- **What didn't:**
- **Surprises:**
- **Lesson worth preserving:**
- **Next experiment:**
