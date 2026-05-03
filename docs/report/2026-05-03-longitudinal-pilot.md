# Longitudinal Pilot — Extending Coverage to T4 + T5

**Date:** 2026-05-03
**Inputs:** SSDataBench's own `sampled_{nlsy,addhealth,cfps,US}.csv` (paper-faithful, n=500 eval split)
**Model:** DeepSeek-v4-flash as data-analyst agent (`pilot_paper_longitudinal`, with one retry round for the cells that crashed mid-run)
**Wall-clock:** ~3.5h total (incl. one full retry of CFPS+US)

This extends the cross-sectional pilot to the four longitudinal panels SSDataBench ships and to all five evaluation types. Cross-sectional GSS/CPS/ACS only support T1–T3 (the upstream T4/T5 configs have empty `event_variables`); life-course event-order metrics require panel data with structured event timings (`age_finished_education`, `age_at_first_marriage`, `age_at_first_child`, `age_started_work`, etc.).

## Headline (per-type pass rate, full T1–T5 suite)

| Dataset | T1 | T2 | T3 | T4 | T5 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| NLSY79 | 0.084 | 0.758 | NaN | 0.000 | 0.603 | 0.361 |
| AddHealth | 0.102 | 0.677 | NaN | 0.020 | 0.370 | 0.292 |
| CFPS | 0.129 | 0.590 | NaN | 0.000 | 0.674 | 0.348 |
| Understanding Society | 0.201 | 0.627 | **0.267** | 0.045 | 0.747 | 0.377 |
| **Mean (excl. NaN)** | **0.129** | **0.663** | 0.267 | **0.016** | **0.598** | **0.345** |

US is the one cell where T3 fit cleanly. The other three datasets hit the same regression-fit failure pattern documented for cross-sectional ACS in `2026-05-03-preserve-missingness-ablation.md`: the agent's sim regressions either fail to converge or the bootstrap estimate explodes because too many predictor rows drop to NaN.

## What T4 and T5 measure (and why one works, one doesn't)

**T4 (event order, chi-sq).** For each triple of life events (E, W, C, M = education-finish, work-start, child, marriage), build the 6-class permutation label "E→W→C", "W→C→E", etc. Chi-sq compares the categorical distribution between real and sim. Across all four datasets the agent fails badly here (0.000–0.045) — life-course chronology isn't something the agent's typical "Gaussian copula on age columns" approach captures, because the *joint structure* of when events happen relative to each other matters, not the marginals.

**T5 (event order × covariate, delta-method z on eta²/Cramér's V).** For each (event-triple, predictor) pair, test whether the *strength* of association between the order label and the predictor is the same in real vs sim. The agent does much better here (0.37–0.75). Two reasons:
1. T5 only tests whether the *strength* of the association matches, not the actual ordering. Even when the agent gets the absolute order distribution wrong (T4=0), the *gradient* with respect to gender/race/education can still be approximately right.
2. The bootstrap on η² / Cramér's V is more forgiving: the variance of the test statistic scales with sample size, so when n=500 and the predictor is balanced (gender, race), small absolute differences don't trigger rejection.

The implication: the agent paradigm partially captures *who* tends to follow which ordering pattern (T5) without capturing *what the typical ordering looks like* (T4). That's a real model deficiency, not a measurement artifact.

## Per-domain breakdown (mean across T1, T2, T3, T5; T4 listed separately as "Sequence")

| Domain | NLSY79 | AddHealth | CFPS | US |
|---|---:|---:|---:|---:|
| Demography | 0.739 | 0.569 | 0.712 | 0.760 |
| Marriage | 0.515 | 0.421 | 0.673 | 0.482 |
| SES | 0.387 | 0.285 | 0.227 | 0.401 |
| Health | 0.526 | 0.296 | 0.670 | 0.520 |
| Abilities/Ability | 0.488 | 0.349 | 0.400 | 0.543 |
| Attitudes/Attitude | — | 0.591 | 0.486 | — |
| Sequence (T4) | 0.000 | 0.020 | 0.000 | 0.045 |

Demography is the consistent strong domain (0.57–0.76) — gender, race, immigrant_status, parental education are easy to model. SES is the weakest among non-T4 domains (0.23–0.40) because it bundles `mean_income_30_40`, `occupation_30_40`, `highest_education` — variables with skewed distributions and conditional missingness that imputation distorts.

## Cross-sectional vs longitudinal headline

Combining this report with `2026-05-03-paper-comparison.md`:

| Dataset | Type | Overall pass rate (agent) |
|---|---|---:|
| GSS-2018 | cross-sectional | 0.570 |
| CPS-1980 | cross-sectional | 0.729 |
| ACS-1980 | cross-sectional | 0.389 |
| NLSY79 | longitudinal (T1–T5) | 0.361 |
| AddHealth | longitudinal (T1–T5) | 0.292 |
| CFPS | longitudinal (T1–T5) | 0.348 |
| Understanding Society | longitudinal (T1–T5) | 0.377 |

Two effects compound to make longitudinal lower:
1. T4 is uniformly catastrophic for the agent and drags down the average by ~0.07.
2. T1 is also weaker on longitudinal (0.08–0.20) than on cross-sectional GSS/CPS (0.67/0.70). Longitudinal schemas have more variables (~25 vs 11), more conditional missingness, and richer cognitive/attitude domains the agent doesn't fit well.

T2 and T5 are solid across all 7 datasets, suggesting the agent reliably captures pairwise associations and association-strength gradients — but not the absolute distributions or temporal structure that T1 and T4 demand.

## Stability findings (DeepSeek + this prompt)

The pilot needed two retry rounds because the orchestrator + prompt combination is fragile on schemas with 25+ variables:
- **EXPLORATION crashed on first try in 3/4 datasets** (NLSY, CFPS, US). Cause: the LLM's generated exploration code calls `.mean()` on columns it assumed numeric but are actually string-categorical (e.g., `physical_health` in NLSY is stored as a string label, not a number).
- **MODELING then crashed in 4/4 datasets on first try.** The agent recovered via the validation/retry loop in 2/4 cases (NLSY, AddHealth iter 0); the other 2 (CFPS, US) only recovered after a fresh run.
- **One run hit `APIConnectionError`** mid-pilot, blowing away the US cell entirely.
- **One run produced a sim CSV that broke T2's autograd Cramér's V** with `np.choice(a=0)` — a category present in real but not sim, leaving the bootstrap with zero rows to draw from. Different seed on retry sidestepped it.

Each of these is recoverable but the cumulative effect was ~3.5h wall-clock for what should have been a ~75min pilot. Worth hardening: a stricter EXPLORATION prompt that reads schema types before computing summaries, and per-stage retry on transient API errors.

## Pipeline changes shipped with this pilot

- `src/ssdataagent/data/schema.py` — filter `type: sequential` from background/target so per-age sequential cols don't enter the agent's prompt or generation.
- `src/ssdataagent/agent/context.py` — write only schema-relevant columns into `train.csv` (longitudinal raw CSVs ship 230–770 cols).
- `src/ssdataagent/evaluation/runner.py` — extended `parse_pass_rates` to handle T4 (`combo`-keyed) and T5 (`combo × predictor`); `by_domain` treats T4 combos as a single `Sequence` bucket; eval subprocess uses `sys.executable` + venv-prepended PATH so the nested `python evaluation/run_all_types.py` finds venv-only deps (`autograd`).
- `config/datasets.yaml` — added `nlsy / addhealth / cfps / us` entries.
- `config/experiments.yaml` — added `smoke_nlsy`, `pilot_paper_longitudinal`, `pilot_paper_longitudinal_retry`.

## Open follow-ups

1. **T4 is the weakest link.** A targeted fix would prompt the agent to model event timings as a joint with order-preserving constraints (e.g., sample from the empirical joint of (E, W, C, M) ages, or fit a sequential conditional `P(M | E, W, C)` chain). Today's Gaussian-copula-on-margins approach can't get this right.
2. **T3 NaN on 3/4 datasets** — same root cause as cross-sectional ACS. Per-variable conditional generation `P(target | condition_vars)` would likely rescue this on NLSY/AddHealth/CFPS where the regression has too many NaN rows to fit.
3. **Direct-generation baseline for longitudinal.** Cross-sectional has DeepSeek-direct numbers in `2026-05-03-paper-comparison.md` for an apples-to-apples comparison. Longitudinal needs the same. Estimated ~6h per dataset → 24h total.
4. **Multi-seed runs** would give variance bands. T4 = 0 might just be variance, not a structural failure.
5. **Stricter EXPLORATION prompt.** Tell the agent to inspect dtypes before computing means/correlations. Would have saved ~6 of the iteration retries seen in this pilot.

## Reproducing

```bash
python scripts/run_experiment.py --experiment pilot_paper_longitudinal
# If any cells fail (transient APIConnectionError or sandbox bugs), retry only the failed cells:
python scripts/run_experiment.py --experiment pilot_paper_longitudinal_retry
python scripts/summarize_pilot.py pilot_paper_longitudinal
```

Per-cell results in `results/pilot_paper_longitudinal/full_agent/<dataset>/<run_id>/eval.json`. Raw bootstrap CSVs in `ssdatabench/evaluation_results/<dataset>/agent_<run_id>/`.
