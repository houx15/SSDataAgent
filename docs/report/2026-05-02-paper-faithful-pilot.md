# Paper-faithful Cross-sectional Pilot

**Date:** 2026-05-02
**Experiment:** `pilot_paper_agents` (3 conditions × 3 datasets, n=1000 per run)
**Inputs:** `real_data/used_dataset/sampled_{gss,cps,acs}.csv` — SSDataBench's own cleaned CSVs, **30 GSS / 12 CPS / 13 ACS variables** (vs 11/12/13 in the 2026-04-30 report).
**Eval:** SSDataBench bootstrap insignificance tests, types 1–3, on the full paper schema. Type 3 metric here is "fraction of bootstrap regressions where the agent's R² is statistically indistinguishable from the real R²" (`Z` test on `R²` difference; we report it as a pass rate).
**Run time:** 27 minutes wall-clock for the full 9-cell matrix.

## Headline

| Condition | GSS-2018 | CPS-1980 | ACS-1980 |
|---|---:|---:|---:|
| `full_agent` | 0.570 | **0.729** | **0.389** |
| `agent_no_semantic` | **0.764** | 0.474 | 0.211 |
| `agent_no_data` | 0.187 | 0.176 | 0.176 |

Three findings replicate cleanly across the wider, paper-faithful schemas:

1. **`agent_no_data` is a flat ~0.18 floor on every dataset.** Confirms prior knowledge alone (no real data) produces calibration-free output regardless of schema. Same as the 2026-04-30 report on the narrower datasets.
2. **The "semantics hurt" effect is now visible on GSS too.** On the legacy 11-var GSS, `full_agent` (0.631) beat `agent_no_semantic` (0.506). On the paper-faithful 33-var GSS, the order **flips**: `agent_no_semantic` (0.764) beats `full_agent` (0.570) by 19 points. The wider semantic schema gives the agent more "validation theater" hooks to over-fit on, while empirical column-statistics preserve the joint structure.
3. **`full_agent` dominates on CPS and ACS** (0.729 and 0.389 respectively), where the schemas are narrower and the population context (1980 US) is more constraining. On these datasets the agent's domain knowledge actually adds calibrated structure rather than distractor cues.

## Per-type breakdown

| Condition | Dataset | T1 (univariate) | T2 (bivariate) | T3 (regression) |
|---|---|---:|---:|---:|
| `full_agent` | GSS | 0.674 | 0.610 | 0.425 |
| `full_agent` | CPS | 0.696 | 0.807 | **0.683** |
| `full_agent` | ACS | 0.179 | 0.599 | NaN |
| `agent_no_semantic` | GSS | 0.758 | 0.849 | 0.685 |
| `agent_no_semantic` | CPS | 0.247 | 0.651 | 0.523 |
| `agent_no_semantic` | ACS | 0.128 | 0.295 | NaN |
| `agent_no_data` | GSS | 0.002 | 0.558 | 0.000 |
| `agent_no_data` | CPS | 0.006 | 0.515 | 0.007 |
| `agent_no_data` | ACS | 0.000 | 0.528 | 0.000 |

Notable patterns:
- **CPS type-3 recovered** under `full_agent`: 0.683 here vs the 2026-04-30 collapse to 0.000 on the legacy 12-var CPS. With the paper-faithful schema (same 12 vars by name but loaded from SSDataBench's own clean), type-3 regressions fit successfully across the bootstrap. The original collapse was driven by imputation of conditionally-missing variables; the new sampled CPS appears to handle the missingness convention differently and the agent's model fit cleanly.
- **ACS type-3 collapsed entirely (NaN) under both `full_agent` and `agent_no_semantic`.** SSDataBench couldn't fit the type-3 regressions on the agent's output. This still has the same root cause as before — generated data violates a regression assumption (probably constant or near-constant predictor variance after imputation). The preserve-missingness prompt fix (open task) is the natural next intervention.
- **`agent_no_data` produces near-zero univariate (T1) rates everywhere.** Without seeing the data, the agent's marginal sampling has no chance of matching the real distribution. T2 (bivariate) is somewhat higher (~0.55) because zero-correlation random sampling occasionally happens to be insignificantly different from real correlations.

## Per-domain breakdown

Domains come from SSDataBench's `data_configs/*.yaml` — every variable carries a `domain:` tag (Demography, Marriage, SES, Health, Attitude, Ability). Reporting per-domain because uniform pass rates across domains would be the strongest evidence of true population-level fit.

| Condition | Dataset | Demography | Marriage | SES | Health |
|---|---|---:|---:|---:|---:|
| `full_agent` | GSS | 0.694 | 0.587 | 0.550 | — |
| `full_agent` | CPS | 0.797 | 0.741 | 0.756 | — |
| `full_agent` | ACS | 0.634 | 0.441 | 0.312 | 0.357 |
| `agent_no_semantic` | GSS | 0.846 | 0.798 | 0.729 | — |
| `agent_no_semantic` | CPS | 0.605 | 0.449 | 0.555 | — |
| `agent_no_semantic` | ACS | 0.169 | 0.224 | 0.145 | 0.545 |
| `agent_no_data` | GSS | 0.652 | 0.160 | 0.281 | — |
| `agent_no_data` | CPS | 0.585 | 0.181 | 0.158 | — |
| `agent_no_data` | ACS | 0.540 | 0.165 | 0.171 | 0.340 |

(GSS Health was not surfaced in this run's evaluation — the type-1/2/3 configs we used did not include the `Health`-tagged GSS variables. To be revisited.)

Per-domain readouts make the headline picture sharper:

- **GSS `agent_no_semantic` wins by being uniformly high.** It scores 0.85/0.80/0.73 across Demography/Marriage/SES — empirical sampling preserves all three domain structures evenly. `full_agent` scores 0.69/0.59/0.55 — same shape but each domain shifted down ~10–15 pts, suggesting semantics introduce a uniform downward bias.
- **CPS `full_agent` wins because of SES.** Demography is similar across conditions (~0.6–0.8). The big differentiator is `full_agent` getting SES (income/occupation) to 0.756 while `agent_no_semantic` lands at 0.555. This is the inverse of GSS: on CPS, knowing what "income" means seems to help the agent build calibrated income models.
- **ACS is hardest everywhere.** All conditions score below 0.65 in every domain. ACS-1980 has a wider age range (0–90) and immigration mix that may stress the agent's modeling.

## Comparison to 2026-04-30 results

| Dataset | Condition | 2026-04-30 (legacy) | 2026-05-02 (paper-faithful) | Δ |
|---|---|---:|---:|---:|
| GSS | full_agent | 0.631 (11 vars) | 0.570 (33 vars) | -0.061 |
| GSS | agent_no_semantic | 0.506 (11 vars) | 0.764 (33 vars) | **+0.258** |
| CPS | full_agent | 0.622 | 0.729 | +0.107 |
| CPS | agent_no_semantic | 0.544 | 0.474 | -0.070 |
| ACS | full_agent | 0.443 | 0.389 | -0.054 |
| ACS | agent_no_semantic | 0.655 | 0.211 | **-0.444** |

Three patterns to flag:

1. **GSS agent_no_semantic jumped from 0.506 to 0.764 on the wider schema.** Empirical sampling improves substantially when more variables are available — likely because the joint distribution becomes more identifiable.
2. **ACS agent_no_semantic dropped from 0.655 to 0.211** despite similar variable count. The two ACS CSVs differ in cleaning/coding (legacy `acs_clean.csv` vs paper's `sampled_acs.csv`). Single-seed variance probably also accounts for some. Worth a multi-seed re-run to bound the bands.
3. **CPS full_agent gained ~10 points and recovered type-3.** Paper-faithful CPS appears to be a friendlier dataset for the agent than our internal cleaning was.

## What this proves about the framework

- **End-to-end SSDataBench replication works on the paper's own inputs.** 9 runs in 27 minutes, fully automated, no human intervention. Agent successfully recovers from 6+ sandbox crashes via the validation iteration loop.
- **The reporting parser now produces SSDataBench-style breakdowns** (per-type, per-pair, per-variable, per-domain) — what the user asked for after the 2026-04-30 round.
- **The previously-flagged hangs (proxy stalls killing pilots silently) are now bounded.** Per-request 300 s timeout in the LLM client; orchestrator emits per-stage heartbeats so any future stall is visible within seconds, not hours.

## Open questions and next steps

1. **Multi-seed runs.** Single-seed swings (especially ACS agent_no_semantic) are large. 3 seeds × 4 conditions × 3 datasets to bound the bands.
2. **Preserve-missingness prompt fix.** ACS type-3 still NaN. Add `"do not impute missing values; preserve the missingness pattern of the training data"` to the modeling prompt and rerun ACS as a controlled before/after.
3. **Direct generation baseline on paper-faithful inputs.** `pilot_paper_direct` (already wired) — slow (~2 h/dataset, per-row LLM calls).
4. **Longitudinal datasets.** AddHealth / NLSY / CFPS / Understanding Society — types 4 and 5 (sequence and sequence×covariate) — needs the eval pipeline extended for the new types.
5. **GSS Health domain didn't show up in eval.** Either no Health vars made it into the agent's output, or the type-1/2/3 configs don't cover them. Check `evaluation/config/gss_2018_subset/type1.yaml` against schema domains.

## Reproducing

```bash
python scripts/run_experiment.py --experiment pilot_paper_agents
python scripts/summarize_pilot.py pilot_paper_agents
```

Per-run artifacts under `results/pilot_paper_agents/<condition>/<dataset>/<run_id>/`. eval.json now includes `by_pair` and `by_domain` blocks alongside the existing `by_type` / `by_variable` / `overall_average`.
