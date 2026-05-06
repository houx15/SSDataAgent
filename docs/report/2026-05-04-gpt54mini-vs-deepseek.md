# Model Swap: gpt-5.4-mini-2026-03-17 vs deepseek-v4-flash

**Date:** 2026-05-04
**Same agent paradigm, same datasets, same evaluation pipeline.** Only the LLM behind the data-analyst orchestrator changed (config swap in `config/llm.yaml` + `.env`; no code changes other than `OpenAI` chat completions requiring `max_completion_tokens` instead of the legacy `max_tokens`).

**Inputs:** SSDataBench's own `sampled_*.csv`, n=500 holdout, identical schemas, identical condition definitions (`full_agent` only for the headline).

## Headline (overall pass rate, full_agent, average across evaluated types)

| Dataset | Type | DeepSeek-v4-flash | **gpt-5.4-mini** | Δ |
|---|---|---:|---:|---:|
| GSS-2018 | cross-sectional | 0.570 | **0.393** | −0.177 |
| CPS-1980 | cross-sectional | 0.729 | **0.479** | −0.250 |
| ACS-1980 | cross-sectional | 0.389 | **0.253** | −0.136 |
| NLSY79 | longitudinal | 0.361 | **0.398** | +0.037 |
| AddHealth | longitudinal | 0.292 | **0.423** | +0.131 |
| CFPS | longitudinal | 0.348 | **0.284** | −0.064 |
| Understanding Society | longitudinal | 0.377 | **0.363** | −0.014 |

**Mean across 7 datasets:** DeepSeek 0.438, gpt-5.4-mini 0.370 (−0.068).

The model swap is **not** a uniform win. Cross-sectional regresses sharply (GSS, CPS, ACS all drop ≥ 0.14); longitudinal is mixed (NLSY/AddHealth improve, CFPS/US flat-or-down).

## Per-type breakdown (full_agent)

### Cross-sectional

| | GSS DeepSeek | GSS gpt-5.4-mini | CPS DeepSeek | CPS gpt-5.4-mini | ACS DeepSeek | ACS gpt-5.4-mini |
|---|---:|---:|---:|---:|---:|---:|
| T1 (univariate) | 0.674 | **0.278** | 0.696 | **0.377** | 0.179 | 0.144 |
| T2 (bivariate)  | 0.610 | 0.625 | 0.807 | **0.679** | 0.599 | 0.536 |
| T3 (regression) | 0.425 | **0.275** | 0.683 | **0.380** | NaN | **0.080** |

The biggest cross-sectional regression is on **T1**. DeepSeek's "univariate-collapse" rescue (sample from empirical marginals) had GSS/CPS at 0.67/0.70; gpt-5.4-mini lands at 0.28/0.38. The agent's modeling code on gpt-5.4-mini doesn't preserve marginals as cleanly — likely fitting parametric distributions and losing tails.

The lone cross-sectional win: **ACS T3 went from NaN → 0.080** (small but nonzero — the regression actually fit). gpt-5.4-mini handles the conditional-missingness imputation differently than DeepSeek did.

### Longitudinal

| | NLSY DS | NLSY mini | AH DS | AH mini | CFPS DS | CFPS mini | US DS | US mini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.084 | **0.333** | 0.102 | **0.612** | 0.129 | 0.224 | 0.201 | 0.097 |
| T2 | 0.758 | 0.708 | 0.677 | **0.774** | 0.590 | 0.481 | 0.627 | 0.622 |
| T3 | NaN | **0.392** | NaN | **0.294** | NaN | **0.206** | 0.267 | 0.282 |
| T4 | 0.000 | 0.000 | 0.020 | 0.000 | 0.000 | 0.000 | 0.045 | 0.053 |
| T5 | 0.603 | 0.556 | 0.370 | **0.434** | 0.674 | 0.509 | 0.747 | **0.760** |

The two **structural wins** of the swap on longitudinal:
1. **T3 NaN-issue largely resolved.** DeepSeek produced NaN on NLSY/AddHealth/CFPS T3 (regression failed to fit on imputed-missingness sims). gpt-5.4-mini produces real numbers on every cell — the agent generates conditional structures the regression can fit.
2. **T1 jumps on AddHealth and NLSY** (0.10→0.61, 0.08→0.33). For these schemas with ~25 vars and rich categorical/health domains, gpt-5.4-mini's marginals fit better than DeepSeek's.

The **structural loss**: T4 stays at 0 across the board. The model swap doesn't fix the fundamental "joint event-order chronology" failure.

## Speed

| Stage | DeepSeek-v4-flash | gpt-5.4-mini |
|---|---|---|
| Per LLM call (chat) | ~14s (reasoning_content tail) | ~3–9s |
| Cross-sectional pilot wall-clock (3 ds × 3 cond × ≤3 iter) | ~hours (multiple retries) | **~28 min, zero retries** |
| Longitudinal pilot wall-clock (4 ds × 1 cond × ≤3 iter) | ~3.5 h (incl. 1 retry round) | **~80 min total (incl. 2 US retries)** |

**Reliability:** Cross-sectional ran clean on first try (0/9 cells failed). Longitudinal lost US on first try and on retry-1 (different failure modes — `pd.NA in tuple` boolean check, then `'nan' as bool` in eval); US recovered on retry-2. Same broad fragility envelope as DeepSeek but each individual run is shorter.

## What this comparison says

1. **Model choice matters at the agent layer just as much as it does for direct generation.** The same paradigm + same prompts + smaller-faster model = significantly different scores. The "agent paradigm beats the best paper LLM" headline from `2026-05-03-paper-comparison-full.md` is **specific to DeepSeek-v4-flash on cross-sectional**; on gpt-5.4-mini, GSS drops back into the paper LLMs' upper range (0.39, vs paper best 0.39 for GPT-4) instead of dominating.

2. **Different models have different blind spots in the same paradigm.**
   - DeepSeek wins T1/T2/T3 on cross-sectional (esp. CPS), gets NaN on regressions when conditional missingness is present.
   - gpt-5.4-mini wins T1/T2/T3 on longitudinal (esp. AddHealth, NLSY), and rescues every T3 NaN, but loses ground on cross-sectional T1.

3. **T4 is paradigm-bound, not model-bound.** Both models score 0–0.05. Fixing T4 needs an architectural change to the agent (joint event-time modeling), not a different LLM.

4. **gpt-5.4-mini's headline is `paper-best-LLM range`, not "beats every paper LLM."** Paper Fig.2 best LLMs scored ~0.39 on GSS and ~0.40 on CPS in the per-individual paradigm. gpt-5.4-mini under our agent paradigm gets 0.39 on GSS and 0.48 on CPS — comparable on GSS, modestly above on CPS. DeepSeek under our agent paradigm got 0.57 / 0.73 — clearly dominating.

## Reproducing

```bash
# Set OpenAI key in .env: LLM_API_KEY=sk-...
# config/llm.yaml: provider=openai, base_url=https://api.openai.com/v1, model=gpt-5.4-mini-2026-03-17

python scripts/run_experiment.py --experiment pilot_paper_agents_gpt54mini
python scripts/run_experiment.py --experiment pilot_paper_longitudinal_gpt54mini
# US needs retries; rerun until eval.json appears for results/pilot_paper_longitudinal_gpt54mini_retry/full_agent/us/<run_id>/
python scripts/run_experiment.py --experiment pilot_paper_longitudinal_gpt54mini_retry
python scripts/summarize_pilot.py pilot_paper_agents_gpt54mini
python scripts/summarize_pilot.py pilot_paper_longitudinal_gpt54mini
```

Per-cell results in `results/<experiment>/<condition>/<dataset>/<run_id>/eval.json`. Each `meta.json` records the active `model` and `provider` so you can confirm what produced the numbers without consulting the experiment name.

## Open follow-ups

1. **Cross-sectional T1 regression.** Why did gpt-5.4-mini lose 0.4 on GSS T1? Inspect the modeling code each model produces — likely gpt-5.4-mini chooses parametric distributions where DeepSeek copied empirical marginals.
2. **Longitudinal T1 win on AddHealth.** Why did AddHealth T1 jump 0.10→0.61? If the modeling pattern is identifiable, generalize it.
3. **Polish targets are now model-specific.** "Per-variable conditional generation" was the gp DeepSeek polish target for T3 NaN; gpt-5.4-mini largely already handles that. The polish target for gpt-5.4-mini is different (T1 marginal preservation).
4. **A polish-then-rerun cycle should target the same model that will run the production benchmark.** Don't tune for DeepSeek then evaluate on gpt-5.4-mini.
