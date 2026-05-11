# exp_demo_b_cross + exp_demo_b_long — report (2026-05-11)

## Strategy
Combined cross+long demo. Tests that the parser handles `## Strategy`
instead of `## Hypothesis` and pulls bullet metadata from any section.

- **prompt_variant:** rubric_tools_v2
- **llm_model:** gpt-5.4-2026-03-05
- **datasets:** demo_x, demo_y
- **conditions:** full_agent

## Results — full_agent

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| demo_x | 0.60 | 0.70 | 0.40 | — | — | 0.567 |
| demo_y | 0.50 | 0.65 | 0.30 | 0.20 | 0.70 | 0.470 |

## vs Baseline experiment `pilot_demo`
Δ overall: +0.17. T3 improved on demo_x, T4 still weak on demo_y.

## What worked / didn't
- Family-selection recipe lifted T3 on demo_x.
- Chronology recipe didn't fire on demo_y.
