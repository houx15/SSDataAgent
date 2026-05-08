# exp006c_tools_cross — report (2026-05-08)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-006c** — Tighten `rubric_tools` system prompt with two concrete recipes: (1) prefer `linear_regression` / `logistic_regression` over `empirical_lookup` when the target is numeric / categorical with a small label set and ≥3 informative `given` columns (T3 lift); (2) for longitudinal datasets, declare event-age columns adjacent in generation_order, fit conditional on previous events, call `score_event_order` before committing (T4 lift). *Expected lift:* +0.10 to +0.20 on T3 across cross-sectional, +0.05 to +0.15 on T4 across longitudinal. *Status:* prompt + yaml ready (variant `rubric_tools_v2`, yaml entries `smoke_acs_tools_v2` / `exp006c_tools_cross` / `exp006c_tools_long`); user runs on cloud box per `RUN-EXP-006c.md`.

- **prompt_variant:** `rubric_tools_v2`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** gss, cps, acs
- **conditions:** full_agent
- **n_rows / dataset:** 1000
- **max_iterations:** 1
- **A/B baseline experiment:** `pilot_paper_agents_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| gss | 0.547 | 0.736 | 0.485 | — | — | 0.590 |
| cps | 0.506 | 0.777 | 0.537 | — | — | 0.606 |
| acs | 0.670 | 0.615 | 0.025 | — | — | 0.437 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| gss | T1 | 0.547 | 0.130 | +0.417 |
| gss | T2 | 0.736 | 0.710 | +0.026 |
| gss | T3 | 0.485 | 0.550 | -0.065 |
| gss | **overall** | 0.590 | 0.390 | +0.200 |
| cps | T1 | 0.506 | 0.100 | +0.406 |
| cps | T2 | 0.777 | 0.710 | +0.067 |
| cps | T3 | 0.537 | 0.570 | -0.033 |
| cps | **overall** | 0.606 | 0.400 | +0.206 |
| acs | T1 | 0.670 | 0.200 | +0.470 |
| acs | T2 | 0.615 | 0.500 | +0.115 |
| acs | T3 | 0.025 | 0.400 | -0.375 |
| acs | **overall** | 0.437 | 0.400 | +0.037 |

## vs Baseline experiment `pilot_paper_agents_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| gss | T1 | 0.547 | 0.220 | +0.328 |
| gss | T2 | 0.736 | 0.651 | +0.085 |
| gss | T3 | 0.485 | 0.407 | +0.078 |
| gss | **overall** | 0.590 | 0.426 | +0.163 |
| cps | T1 | 0.506 | 0.573 | -0.067 |
| cps | T2 | 0.777 | 0.743 | +0.034 |
| cps | T3 | 0.537 | 0.457 | +0.080 |
| cps | **overall** | 0.606 | 0.591 | +0.016 |
| acs | T1 | 0.670 | 0.445 | +0.225 |
| acs | T2 | 0.615 | 0.714 | -0.099 |
| acs | T3 | 0.025 | 0.355 | -0.330 |
| acs | **overall** | 0.437 | 0.505 | -0.068 |

## Retro

vs EXP-006b (same model, same tool path, no recipes):

| Dataset | Δ T1 | Δ T2 | Δ T3 | Δ overall |
|---|---:|---:|---:|---:|
| gss | -0.070 | +0.025 | **+0.278** | +0.078 |
| cps | -0.204 | +0.150 | **+0.340** | +0.095 |
| acs | +0.120 | -0.039 | +0.022 | +0.035 |

- **What worked:** FAMILY-SELECTION recipe lifted gss/cps T3 dramatically (0.207 → 0.485, 0.197 → 0.537), well past the +0.10–0.20 expected band. gss/cps now beat paper-best overall (+0.20). The agent's `tool_calls.json` shows it preferred logistic/linear conditional fits over `empirical_lookup` on `marital_status / education / occupation / income / wealth` — exactly the substitutions the recipe targeted.
- **What didn't:** ACS T3 stayed near zero (0.025). Only 8 of acs's 24 conditional-with-given fits were regressions (16 were `empirical_lookup`). Even where regressions were used, the agent didn't preserve missingness — and ACS T3 is missingness-bound (memory: `feedback_acs_t3_fragile_to_imputation`). Family-selection alone can't fix that bottleneck.
- **What didn't, part 2:** cross-sectional T1 dipped on gss (-0.070) and cps (-0.204). Switching to parametric conditionals tightens the joint at the cost of marginal fidelity — the same agent that gets T3 right loses some T1.
- **Surprises:** cps T2 jumped +0.150 in addition to T3. Conditioning more carefully also fixed bivariate dependence the marginal-heavy chain was missing.
- **Lesson worth preserving:** A per-target family-selection rule beats a generic "free choice" prompt for cross-sectional T3 when the chain has full demographics + clear numeric/small-categorical targets. ACS-style missingness still needs separate machinery (preserve-NA conditionals).
- **Next experiment:** EXP-006e — fix the longitudinal ablation (event-order verify gate, see long-report retro) and add an ACS-specific path that keeps NaN structure on conditionally-missing predictors before the regression step.
