# exp006c_tools_long — report (2026-05-08)

## Strategy

From STRATEGY.md backlog:

> - [~] **EXP-006c** — Tighten `rubric_tools` system prompt with two concrete recipes: (1) prefer `linear_regression` / `logistic_regression` over `empirical_lookup` when the target is numeric / categorical with a small label set and ≥3 informative `given` columns (T3 lift); (2) for longitudinal datasets, declare event-age columns adjacent in generation_order, fit conditional on previous events, call `score_event_order` before committing (T4 lift). *Expected lift:* +0.10 to +0.20 on T3 across cross-sectional, +0.05 to +0.15 on T4 across longitudinal. *Status:* prompt + yaml ready (variant `rubric_tools_v2`, yaml entries `smoke_acs_tools_v2` / `exp006c_tools_cross` / `exp006c_tools_long`); user runs on cloud box per `RUN-EXP-006c.md`.

- **prompt_variant:** `rubric_tools_v2`
- **llm_model:** `gpt-5.4-2026-03-05`
- **datasets:** nlsy, addhealth, cfps, us
- **conditions:** full_agent
- **n_rows / dataset:** 1000
- **max_iterations:** 1
- **A/B baseline experiment:** `pilot_paper_longitudinal_gpt54`

## Results — `full_agent`

| Dataset | T1 | T2 | T3 | T4 | T5 | overall |
|---|---:|---:|---:|---:|---:|---:|
| nlsy | 0.694 | 0.715 | 0.298 | 0.022 | 0.534 | 0.453 |
| addhealth | 0.617 | 0.775 | 0.282 | 0.000 | 0.400 | 0.415 |
| cfps | 0.134 | 0.569 | — | 0.000 | 0.430 | 0.283 |
| us | 0.425 | 0.629 | 0.291 | 0.230 | 0.744 | 0.464 |

## vs Paper-best (best of 15 LLMs in SSDataBench)

Side-by-side per (dataset, T-type) cell. Δ = ours − paper-best.

| Dataset | T-type | ours | paper-best | Δ |
|---|---:|---:|---:|---:|
| nlsy | T1 | 0.694 | 0.130 | +0.564 |
| nlsy | T2 | 0.715 | 0.710 | +0.005 |
| nlsy | T3 | 0.298 | 0.420 | -0.122 |
| nlsy | T4 | 0.022 | 0.050 | -0.028 |
| nlsy | T5 | 0.534 | 0.750 | -0.216 |
| nlsy | **overall** | 0.453 | 0.300 | +0.153 |
| addhealth | T1 | 0.617 | 0.120 | +0.497 |
| addhealth | T2 | 0.775 | 0.550 | +0.225 |
| addhealth | T3 | 0.282 | 0.390 | -0.108 |
| addhealth | T4 | 0.000 | 0.020 | -0.020 |
| addhealth | T5 | 0.400 | 0.460 | -0.060 |
| addhealth | **overall** | 0.415 | 0.270 | +0.145 |
| cfps | T1 | 0.134 | 0.140 | -0.006 |
| cfps | T2 | 0.569 | 0.620 | -0.051 |
| cfps | T3 | — | 0.430 | — |
| cfps | T4 | 0.000 | 0.050 | -0.050 |
| cfps | T5 | 0.430 | 0.750 | -0.320 |
| cfps | **overall** | 0.283 | 0.300 | -0.017 |
| us | T1 | 0.425 | 0.200 | +0.225 |
| us | T2 | 0.629 | 0.620 | +0.009 |
| us | T3 | 0.291 | 0.540 | -0.249 |
| us | T4 | 0.230 | 0.050 | +0.180 |
| us | T5 | 0.744 | 0.750 | -0.006 |
| us | **overall** | 0.464 | 0.360 | +0.104 |

## vs Baseline experiment `pilot_paper_longitudinal_gpt54`

Δ = this experiment − baseline. Positive = improvement.

| Dataset | T-type | this | baseline | Δ |
|---|---:|---:|---:|---:|
| nlsy | T1 | 0.694 | 0.536 | +0.159 |
| nlsy | T2 | 0.715 | 0.760 | -0.045 |
| nlsy | T3 | 0.298 | 0.552 | -0.254 |
| nlsy | T4 | 0.022 | 0.070 | -0.048 |
| nlsy | T5 | 0.534 | 0.681 | -0.147 |
| nlsy | **overall** | 0.453 | 0.520 | -0.067 |
| addhealth | T1 | 0.617 | 0.439 | +0.178 |
| addhealth | T2 | 0.775 | 0.764 | +0.011 |
| addhealth | T3 | 0.282 | 0.532 | -0.250 |
| addhealth | T4 | 0.000 | 0.000 | +0.000 |
| addhealth | T5 | 0.400 | 0.455 | -0.055 |
| addhealth | **overall** | 0.415 | 0.438 | -0.023 |
| cfps | T1 | 0.134 | 0.267 | -0.133 |
| cfps | T2 | 0.569 | 0.625 | -0.056 |
| cfps | T3 | — | 0.341 | — |
| cfps | T4 | 0.000 | 0.000 | +0.000 |
| cfps | T5 | 0.430 | 0.557 | -0.127 |
| cfps | **overall** | 0.283 | 0.358 | -0.075 |
| us | — | 0.464 | — | — |

## Retro

vs EXP-006b (same model, same tool path, no recipes):

| Dataset | Δ T1 | Δ T2 | Δ T3 | Δ T4 | Δ T5 | Δ overall |
|---|---:|---:|---:|---:|---:|---:|
| nlsy | +0.121 | -0.064 | **-0.196** | +0.013 | -0.107 | -0.046 |
| addhealth | +0.005 | +0.015 | +0.078 | 0.000 | +0.071 | +0.034 |
| cfps | -0.091 | -0.007 | — | 0.000 | -0.155 | -0.064 |
| us | +0.206 | -0.022 | -0.044 | **+0.095** | -0.017 | +0.044 |

- **What worked:** us T4 0.135 → 0.230 (+0.095) — *but* the worked chain still has every event-age column fit marginally with `categorical_empirical`. US's narrow age ranges (most marriages 22-28, divorces 30-40) make marginal sampling produce roughly correct event ordering ~23% of the time. **The chronology recipe didn't fire on any longitudinal dataset.** The lift is coincidence, not recipe.
- **What didn't:** nlsy T3 regressed −0.196. Inspection of `tool_calls.json` shows: agent followed the recipe at first (turn 8, fit `age_at_first_marriage given=[ever_married,…] family=empirical_lookup`), then verified with `score_overall` (0.44, low — because score_overall is per-column-marginal-only and parametric conditionals widen marginal tails), panicked, called `replace_step` on 15 columns, refit them all as **marginal** `categorical_empirical` (turn 13), and committed at score_overall=0.56. Chronology destroyed for the score-overall optimization.
- **What didn't, part 2:** `score_event_order` was never invoked in any longitudinal run, even though the recipe explicitly says "call `score_event_order` before committing". The tool is registered (`tools/__init__.py:39`) and surfaces in the schema. The recipe is advisory — nothing forces the agent to call it before `commit_generator`.
- **Surprises:** addhealth T3 +0.078 and T5 +0.071 with T4 unchanged at 0. The recipe nudged the conditional structure positively for non-event metrics without touching event ordering at all.
- **Lesson worth preserving:** **Verify-tool design is destiny.** If the agent's score_overall doesn't include the metric you care about (event ordering, missingness pattern), the agent will optimize against it and ship a chain that scores high on the visible metric and low on the hidden one. A prompt instruction to "call score_event_order before committing" is not enough — the orchestrator must make `commit_generator` *require* a recent `score_event_order` call on longitudinal datasets.
- **Next experiment:** EXP-006e — make `commit_generator` reject commits on longitudinal datasets unless `score_event_order` has been called for at least one event-list. Optionally: make `score_overall` include a chronology penalty when the chain has multiple `age_*` event-age columns. Pair with the family-selection refinement: don't apply the recipe to many-valued event-age targets (nlsy regression on age_at_first_marriage is wrong; empirical_lookup conditioned on prior events is right).
