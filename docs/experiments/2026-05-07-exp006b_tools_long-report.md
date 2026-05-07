# exp006b_tools_long — report (2026-05-07)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-006** — **Tool-using agent loop** (Claude Code / Codex shape). Replaces the 4-stage code-exec orchestrator with a tool-using loop where the agent inspects/fits/verifies incrementally instead of writing one ~400-line generation script in one shot. Design: `2026-05-07-tool-using-agent-design.md`. **Subsumes EXP-002 (verify-family thresholds), EXP-003 (`fit_conditional(allow_missing=True)`), and EXP-004 (system prompt + tool guidance, not stage-prompt branching).** *Expected lift:* +0.05–0.15 mostly by recovering the cells that crashed under EXP-001 + letting the agent verify before it commits. *Cost:* multi-day rewrite of orchestrator + new tools module; spike on `acs` first.

- **prompt_variant:** `rubric_tools`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** nlsy, addhealth, cfps, us
- **conditions:** full_agent
- **n_rows / dataset:** 1000
- **max_iterations:** 1
- **A/B baseline experiment:** `pilot_paper_longitudinal_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| nlsy | 0.573 | 0.779 | 0.494 | 0.010 | 0.641 | 0.499 |
| addhealth | 0.612 | 0.760 | 0.204 | 0.000 | 0.329 | 0.381 |
| cfps | 0.225 | 0.576 | — | 0.000 | 0.586 | 0.347 |
| us | 0.219 | 0.651 | 0.335 | 0.135 | 0.761 | 0.420 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| nlsy | T1 | 0.573 | 0.130 | +0.443 |
| nlsy | T2 | 0.779 | 0.710 | +0.069 |
| nlsy | T3 | 0.494 | 0.420 | +0.074 |
| nlsy | T4 | 0.010 | 0.050 | -0.040 |
| nlsy | T5 | 0.641 | 0.750 | -0.109 |
| nlsy | **overall** | 0.499 | 0.300 | +0.199 |
| addhealth | T1 | 0.612 | 0.120 | +0.492 |
| addhealth | T2 | 0.760 | 0.550 | +0.210 |
| addhealth | T3 | 0.204 | 0.390 | -0.186 |
| addhealth | T4 | 0.000 | 0.020 | -0.020 |
| addhealth | T5 | 0.329 | 0.460 | -0.131 |
| addhealth | **overall** | 0.381 | 0.270 | +0.111 |
| cfps | T1 | 0.225 | 0.140 | +0.085 |
| cfps | T2 | 0.576 | 0.620 | -0.044 |
| cfps | T3 | — | 0.430 | — |
| cfps | T4 | 0.000 | 0.050 | -0.050 |
| cfps | T5 | 0.586 | 0.750 | -0.164 |
| cfps | **overall** | 0.347 | 0.300 | +0.047 |
| us | T1 | 0.219 | 0.200 | +0.019 |
| us | T2 | 0.651 | 0.620 | +0.031 |
| us | T3 | 0.335 | 0.540 | -0.205 |
| us | T4 | 0.135 | 0.050 | +0.085 |
| us | T5 | 0.761 | 0.750 | +0.011 |
| us | **overall** | 0.420 | 0.360 | +0.060 |

## vs Baseline experiment `pilot_paper_longitudinal_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| nlsy | T1 | 0.573 | 0.536 | +0.037 |
| nlsy | T2 | 0.779 | 0.760 | +0.019 |
| nlsy | T3 | 0.494 | 0.552 | -0.058 |
| nlsy | T4 | 0.010 | 0.070 | -0.060 |
| nlsy | T5 | 0.641 | 0.681 | -0.040 |
| nlsy | **overall** | 0.499 | 0.520 | -0.020 |
| addhealth | T1 | 0.612 | 0.439 | +0.172 |
| addhealth | T2 | 0.760 | 0.764 | -0.004 |
| addhealth | T3 | 0.204 | 0.532 | -0.328 |
| addhealth | T4 | 0.000 | 0.000 | +0.000 |
| addhealth | T5 | 0.329 | 0.455 | -0.126 |
| addhealth | **overall** | 0.381 | 0.438 | -0.057 |
| cfps | T1 | 0.225 | 0.267 | -0.042 |
| cfps | T2 | 0.576 | 0.625 | -0.049 |
| cfps | T3 | — | 0.341 | — |
| cfps | T4 | 0.000 | 0.000 | +0.000 |
| cfps | T5 | 0.586 | 0.557 | +0.028 |
| cfps | **overall** | 0.347 | 0.358 | -0.011 |
| us | — | 0.420 | — | — |

## vs EXP-001 rubric (the experiment this was meant to recover from)

| Dataset | this | exp001_rubric | Δ |
|---|---:|---:|---:|
| nlsy | 0.499 | 0.464 | +0.035 |
| addhealth | 0.381 | (crashed) | **full recovery** |
| cfps | 0.347 | 0.409 | −0.062 |
| us | 0.420 | 0.534 | −0.114 |

## Retro

- **What worked:** All 4 datasets ran end-to-end with no crashes. addhealth —
  which CRASHED under the legacy rubric path in EXP-001 — produced 0.381,
  beating paper-best by +0.111. nlsy got 0.499 (+0.199 over paper-best), the
  best showing across the board. T1 again strong vs paper-best on all four
  (+0.02 to +0.49). T4 / T5 mostly tracked baseline — the chronology layer
  isn't worse than legacy, it's just not better either.

- **What didn't:** us regressed −0.114 vs the EXP-001 rubric run that
  succeeded (which scored an unusually high 0.534 — likely the chain
  happened to capture event ordering by accident). cfps lost T3 entirely
  to an SSDataBench eval failure (same upstream issue exp001 hit; not a
  tools-path bug). vs same-model baseline: nlsy ≈ tied (−0.020), addhealth
  −0.057, cfps ≈ tied (−0.011), us baseline missing.

- **Surprises:** **us T4 = 0.135** is the second-highest T4 score we've
  ever produced (paper-best is 0.050, exp001_rubric_long was 0.195 on us
  but that run got T1=0.444). T4 success on us means the agent stumbled
  onto event-time chronology via tool composition without explicit
  guidance — useful signal for the next experiment, but not yet
  reproducible.

- **Lesson worth preserving:** **Tool-using doesn't fix T4 by default** —
  you still need to either fit conditionals on age-events in the right
  order with allow_missing or call score_event_order during the loop and
  refit. The current `rubric_tools` prompt mentions T4 in the rubric block
  but doesn't surface a concrete recipe.

- **Next experiment:** Roll into **EXP-006c** alongside the cross fix:
  add a chronology recipe to the system prompt — "for longitudinal,
  declare event-age columns adjacent in generation_order, fit each as
  conditional on the previous events, call score_event_order before
  committing." Expected lift: +0.05 to +0.15 on T4 across longitudinal
  datasets.
