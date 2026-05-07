# exp001_rubric_long — report (2026-05-07)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-001** — Add T1–T5 rubric block to `SYSTEM_PROMPT` as a new prompt variant. Yaml entries: `exp001_rubric_cross` + `exp001_rubric_long`. Run on cloud box via `python scripts/run_batch.py exp001_rubric_cross exp001_rubric_long`; report via `scripts/generate_exp_report.py exp001_rubric_cross --baseline pilot_paper_agents_gpt54`. *Expected lift:* +0.03–0.08 across the board. *Status:* in progress (matrix runner + variant landed 2026-05-06).

- **prompt_variant:** `rubric`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** nlsy, addhealth, cfps, us
- **conditions:** full_agent
- **n_rows / dataset:** 1000
- **max_iterations:** 3
- **A/B baseline experiment:** `pilot_paper_longitudinal_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| nlsy | 0.516 | 0.756 | — | 0.000 | 0.585 | 0.464 |
| addhealth | — | — | — | — | — | (no eval) |
| cfps | 0.590 | 0.607 | 0.414 | 0.000 | 0.434 | 0.409 |
| us | 0.444 | 0.726 | — | 0.195 | 0.773 | 0.534 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| nlsy | T1 | 0.516 | 0.130 | +0.386 |
| nlsy | T2 | 0.756 | 0.710 | +0.046 |
| nlsy | T3 | — | 0.420 | — |
| nlsy | T4 | 0.000 | 0.050 | -0.050 |
| nlsy | T5 | 0.585 | 0.750 | -0.165 |
| nlsy | **overall** | 0.464 | 0.300 | +0.164 |
| addhealth | — | (no eval) | — | — |
| cfps | T1 | 0.590 | 0.140 | +0.450 |
| cfps | T2 | 0.607 | 0.620 | -0.013 |
| cfps | T3 | 0.414 | 0.430 | -0.016 |
| cfps | T4 | 0.000 | 0.050 | -0.050 |
| cfps | T5 | 0.434 | 0.750 | -0.316 |
| cfps | **overall** | 0.409 | 0.300 | +0.109 |
| us | T1 | 0.444 | 0.200 | +0.244 |
| us | T2 | 0.726 | 0.620 | +0.106 |
| us | T3 | — | 0.540 | — |
| us | T4 | 0.195 | 0.050 | +0.145 |
| us | T5 | 0.773 | 0.750 | +0.023 |
| us | **overall** | 0.534 | 0.360 | +0.174 |

## vs Baseline experiment `pilot_paper_longitudinal_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| nlsy | T1 | 0.516 | 0.536 | -0.020 |
| nlsy | T2 | 0.756 | 0.760 | -0.004 |
| nlsy | T3 | — | 0.552 | — |
| nlsy | T4 | 0.000 | 0.070 | -0.070 |
| nlsy | T5 | 0.585 | 0.681 | -0.096 |
| nlsy | **overall** | 0.464 | 0.520 | -0.056 |
| addhealth | — | — | 0.438 | — |
| cfps | T1 | 0.590 | 0.267 | +0.323 |
| cfps | T2 | 0.607 | 0.625 | -0.018 |
| cfps | T3 | 0.414 | 0.341 | +0.073 |
| cfps | T4 | 0.000 | 0.000 | +0.000 |
| cfps | T5 | 0.434 | 0.557 | -0.124 |
| cfps | **overall** | 0.409 | 0.358 | +0.051 |
| us | — | 0.534 | — | — |

## Retro

- **What worked:**
- **What didn't:**
- **Surprises:**
- **Lesson worth preserving:**
- **Next experiment:**
