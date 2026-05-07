# exp006b_tools_cross — report (2026-05-07)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-006** — **Tool-using agent loop** (Claude Code / Codex shape). Replaces the 4-stage code-exec orchestrator with a tool-using loop where the agent inspects/fits/verifies incrementally instead of writing one ~400-line generation script in one shot. Design: `2026-05-07-tool-using-agent-design.md`. **Subsumes EXP-002 (verify-family thresholds), EXP-003 (`fit_conditional(allow_missing=True)`), and EXP-004 (system prompt + tool guidance, not stage-prompt branching).** *Expected lift:* +0.05–0.15 mostly by recovering the cells that crashed under EXP-001 + letting the agent verify before it commits. *Cost:* multi-day rewrite of orchestrator + new tools module; spike on `acs` first.

- **prompt_variant:** `rubric_tools`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** gss, cps, acs
- **conditions:** full_agent
- **n_rows / dataset:** 1000
- **max_iterations:** 1
- **A/B baseline experiment:** `pilot_paper_agents_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| gss | 0.617 | 0.711 | 0.207 | — | — | 0.512 |
| cps | 0.710 | 0.627 | 0.197 | — | — | 0.511 |
| acs | 0.550 | 0.654 | 0.003 | — | — | 0.402 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| gss | T1 | 0.617 | 0.130 | +0.487 |
| gss | T2 | 0.711 | 0.710 | +0.001 |
| gss | T3 | 0.207 | 0.550 | -0.343 |
| gss | **overall** | 0.512 | 0.390 | +0.122 |
| cps | T1 | 0.710 | 0.100 | +0.610 |
| cps | T2 | 0.627 | 0.710 | -0.083 |
| cps | T3 | 0.197 | 0.570 | -0.373 |
| cps | **overall** | 0.511 | 0.400 | +0.111 |
| acs | T1 | 0.550 | 0.200 | +0.350 |
| acs | T2 | 0.654 | 0.500 | +0.154 |
| acs | T3 | 0.003 | 0.400 | -0.398 |
| acs | **overall** | 0.402 | 0.400 | +0.002 |

## vs Baseline experiment `pilot_paper_agents_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| gss | T1 | 0.617 | 0.220 | +0.397 |
| gss | T2 | 0.711 | 0.651 | +0.060 |
| gss | T3 | 0.207 | 0.407 | -0.200 |
| gss | **overall** | 0.512 | 0.426 | +0.086 |
| cps | T1 | 0.710 | 0.573 | +0.137 |
| cps | T2 | 0.627 | 0.743 | -0.115 |
| cps | T3 | 0.197 | 0.457 | -0.260 |
| cps | **overall** | 0.511 | 0.591 | -0.079 |
| acs | T1 | 0.550 | 0.445 | +0.105 |
| acs | T2 | 0.654 | 0.714 | -0.060 |
| acs | T3 | 0.003 | 0.355 | -0.352 |
| acs | **overall** | 0.402 | 0.505 | -0.102 |

## vs EXP-001 rubric (the experiment this was meant to recover from)

| Dataset | T-type | this | exp001_rubric | Δ |
|---|---:|---:|---:|---:|
| gss | overall | 0.512 | (crashed) | **full recovery** |
| cps | overall | 0.511 | (crashed) | **full recovery** |
| acs | overall | 0.402 | 0.369 | +0.033 |

## Retro

- **What worked:** All 3 datasets ran end-to-end with no crashes. gss and cps
  — both of which CRASHED under the legacy rubric path in EXP-001 — produced
  scores in the 0.51 range, beating paper-best on overall by +0.11 to +0.12.
  T1 jumped enormously vs paper-best on every dataset (+0.35 to +0.61) — the
  tool-using path consistently nails univariate marginals.

- **What didn't:** vs same-model baseline (legacy gpt-5.4 code-block path),
  cross-sectional regressed: gss +0.086, cps −0.079, acs −0.102. The
  regressions are concentrated in T2 (cps −0.115, acs −0.060) and T3
  (gss −0.200, cps −0.260, acs −0.352). ACS overall here is 0.402 vs Stage A's
  0.498 on the *same* config — gpt-5.4 stochasticity is real and large
  (run-to-run variance ≈ 0.10 on overall, mostly driven by which family the
  agent picks for T3-critical numeric conditionals).

- **Surprises:** ACS T3 collapsed to 0.003. The agent in this run picked
  empirical_lookup for income, occupation-related conditionals where Stage A
  had used some linear_regression. Confirms the Stage A retro's hypothesis
  that empirical_lookup is a T3-killer for numeric targets.

- **Lesson worth preserving:** **Family choice for T3-critical conditionals
  dominates T3 score** more than anything else the agent does. The system
  prompt currently lists `empirical_lookup` first as "safe for any dtype"
  — for the next experiment, demote it for numeric targets.

- **Next experiment:** **EXP-006c** — tighten the `rubric_tools` system
  prompt to nudge `linear_regression` over `empirical_lookup` whenever
  the target is numeric and ≥3 informative `given` columns exist. Re-run
  this same yaml after the prompt change; expect T3 lift of +0.10 to +0.20
  on the 3 cross-sectional datasets.
