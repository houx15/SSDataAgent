# Preserve-Missingness Ablation

**Date:** 2026-05-03
**Experiment:** `pilot_paper_agents_preserve` (controlled A/B vs `pilot_paper_agents`).
**Intervention:** added one paragraph to `modeling_prompt` instructing the agent to keep NaN values where the survey design makes them conditionally missing (e.g., `age_first_marriage` for never-married, `spouse_occupation` for unmarried, `income` for not-in-laborforce). Otherwise identical to the 2026-05-02 run.

## Headline comparison

| Cell | Prior | Preserve | Δ | Note |
|---|---:|---:|---:|---|
| full_agent / GSS | 0.570 | **0.681** | +0.111 | T1/T2 up, T3 → NaN |
| full_agent / CPS | **0.729** | 0.671 | -0.058 | T3 went 0.683 → NaN |
| full_agent / ACS | 0.389 | **0.730** | **+0.341** | T1 0.179→0.645, T2 0.599→0.815 |
| agent_no_semantic / GSS | 0.764 | FAILED | — | nudge retry failed (no code block) |
| agent_no_semantic / CPS | 0.474 | **0.751** | **+0.277** | T3 0.523 → 0.737 |
| agent_no_semantic / ACS | 0.211 | FAILED | — | nudge retry failed (no code block) |
| agent_no_data / GSS | 0.187 | 0.272 | +0.085 | |
| agent_no_data / CPS | 0.176 | 0.324 | +0.148 | T3 0.007 → 0.497 |
| agent_no_data / ACS | 0.176 | FAILED | — | GENERATION crashed |

Three patterns emerged:

### 1. Preserve is robustly good for `full_agent` on T1/T2; ambivalent on T3.

`full_agent` overall improved on GSS (+0.111) and ACS (+0.341), and lost only 6 points on CPS. The biggest single jump was ACS: marginal pass rate (T1) went from 0.179 to 0.645, and bivariate (T2) from 0.599 to 0.815.

But T3 (regression) **collapsed to NaN on all three datasets** — the opposite of what we hypothesized. The mechanism: when the agent preserves more conditional missingness, predictor columns in the bootstrap regression have more NaN rows, and SSDataBench drops those rows. With enough drops the regression either fails to fit (constant predictor variance) or the bootstrap estimate explodes.

So preserve-missingness trades T3 fittability for T1/T2 fidelity — a different kind of failure mode than the original imputation collapse.

### 2. Preserve unlocks `agent_no_semantic` on the dataset it works on.

`agent_no_semantic` on CPS jumped from 0.474 to **0.751** — the largest gain in the matrix. T3 went from 0.523 to 0.737. Without semantic descriptions the agent can't decide which variables "should" be conditional, so the explicit instruction tells it: copy the empirical missingness pattern from training. That maps cleanly to a `df.isna()` join, which the agent does.

### 3. Preserve breaks `agent_no_semantic` on schemas it can't structure.

Both GSS and ACS `agent_no_semantic` runs failed because the agent emitted **no code block even after the nudge retry**. The preserve instruction is now ~80 words of structured natural-language detail. Combined with the no-semantic schema view (no variable descriptions, no domain tags), the agent went into long planning mode and produced narrative responses without code.

This is a real cost — adding important behavioral guidance to the prompt makes the schema-light condition fragile.

## Per-type breakdown (preserve only)

| Condition | Dataset | T1 | T2 | T3 |
|---|---|---:|---:|---:|
| `full_agent` | GSS | 0.589 | 0.773 | NaN |
| `full_agent` | CPS | 0.661 | 0.680 | NaN |
| `full_agent` | ACS | 0.645 | 0.815 | NaN |
| `agent_no_semantic` | CPS | 0.713 | 0.805 | 0.737 |
| `agent_no_data` | GSS | 0.046 | 0.655 | 0.115 |
| `agent_no_data` | CPS | 0.007 | 0.470 | 0.497 |

Note `agent_no_semantic` on CPS is the **only condition that scored well on T3 with preserve**. It's also the only condition where the agent didn't have semantics to mislead it about which vars are "conditional" but did have data and the explicit preserve instruction. That combination — empirical pattern + literal preservation directive — appears to be what type-3 needs.

## Per-domain (preserve only, full_agent)

| Domain | GSS | CPS | ACS |
|---|---:|---:|---:|
| Demography | 0.824 | 0.704 | 0.839 |
| Marriage | 0.823 | 0.728 | 0.712 |
| SES | 0.548 | 0.621 | 0.676 |
| Health | 0.635 | — | **0.929** |
| Attitude | 0.779 | — | — |
| Ability | 0.656 | — | — |

Striking: ACS `full_agent` Health domain hits **0.929** under preserve. The legacy ACS health variable (`health_work_difficulty`) was being imputed before; preserving its NaN structure recovers near-perfect domain fit.

## What this tells us

- The original 2026-04-30 type-3 collapse was real, but the fix is more nuanced than "always preserve missingness." Preserve fixes T1/T2 calibration but introduces T3 fittability problems via a different route.
- The **right intervention is conditional**: preserve the NaN pattern of *structurally-missing* variables but allow the agent to model the conditional distribution (e.g., spouse_occupation given married). That requires the agent to distinguish "missing because not asked" (preserve) from "missing because unknown" (impute or model).
- The instruction style itself matters: long behavioral paragraphs degrade the schema-light agent's ability to emit code at all. A future iteration should test a shorter form, or only inject the instruction in conditions that have semantics to ground it.

## Recommended next steps

1. **Two-tier preserve instruction.** Short version for `agent_no_semantic`; full version for `full_agent`. Avoids the code-block failures.
2. **Per-variable conditional generation.** Have the agent explicitly model `P(target | condition_vars)` rather than just preserving the NaN mask wholesale. This should rescue T3.
3. **Multi-seed runs.** Two big positive deltas (ACS full_agent +0.34, CPS agent_no_semantic +0.28) deserve variance bands before being treated as load-bearing.
4. **Don't roll the preserve change forward yet.** Three runs failed and T3 broke for `full_agent`. Treat this as an instructive ablation, not a default.

## Reproducing

```bash
python scripts/run_experiment.py --experiment pilot_paper_agents_preserve
python scripts/summarize_pilot.py pilot_paper_agents_preserve
```

Compare against `pilot_paper_agents` (2026-05-02 run, before the modeling-prompt change at commit `1f62b80`).
