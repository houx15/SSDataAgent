# Data-Driven Agents Beat Per-Individual Prompting on Population Simulation

**Cross-dataset findings from SSDataAgent on GSS-2018, CPS-1980, ACS-1980**
*April 30, 2026*

## Abstract

We compare two LLM-based paradigms for generating synthetic social-survey data with population-level statistical realism:

- **Direct generation** (the SSDataBench paradigm): one LLM call per simulated individual, conditioned on background variables, returning JSON of target values.
- **Data-analyst agent**: a single LLM session given the real training split and asked to *write Python code* that fits a generative model and samples from it.

Across **GSS-2018, CPS-1980, and ACS-1980** (n=1000 each, ~10 variables each) using DeepSeek's `deepseek-v4-flash` reasoning model, the agent paradigm — even when stripped of variable descriptions and population context (`agent_no_semantic`) — beats per-individual prompting on every dataset, with the gap *widening* across datasets (+0.23 / +0.34 / +0.47 mean pass rate). The robust finding is not "agents win" but rather **"data-driven empirical sampling beats per-individual LLM elicitation, regardless of variable semantics."** Removing data access collapses the agent to a near-identical floor (~0.18) on every dataset. The full agent (with data + semantics) is the strongest condition only on GSS; on harder datasets it is unstable or underperforms its data-only ablation.

## 1. Setup

### 1.1 Model and infrastructure

All conditions use the same LLM: DeepSeek `deepseek-v4-flash`, an OpenAI-compatible reasoning model accessed via `https://api.deepseek.com`. Temperature = 1.0, max_tokens = 4096.

We re-implement SSDataBench's "direct generation" baseline using the same client (rather than calling SSDataBench's own simulation code, which is hard-wired to OpenRouter), so the only varying factor across conditions is the *paradigm* — same model, same temperature, same data.

### 1.2 Datasets

Three SSDataBench-cleaned cross-sectional samples, each n=1000:

| Dataset | Period | Vars retained | Target vars | Notable conditioning |
|---|---|---:|---|---|
| **GSS-2018** | General Social Survey | 11 | 5 | `age_first_childbirth` conditional on `child_number > 0` |
| **CPS-1980** | Current Population Survey | 13 | 8 | `income`, `laborforce` with high NaN |
| **ACS-1980** | American Community Survey | 14 | 9 | `age_first_marriage` and `age_first_childbirth` both conditional |

Train/eval split: 50/50 with fixed seed (500/500) per dataset. We auto-prune SSDataBench's per-dataset evaluation config down to columns present in our cleaned CSVs.

### 1.3 Conditions

| Condition | Real data shown | Variable descriptions | Population context |
|---|:---:|:---:|:---:|
| `full_agent` | train split | yes | yes |
| `agent_no_semantic` | train split | no | no |
| `agent_no_data` | none | yes | yes |
| `direct_generation` | none | per-prompt | per-prompt |

Agent conditions run a fixed-stage pipeline: `EXPLORATION → MODELING → VALIDATION → GENERATION`. Each stage is a single LLM call returning Python that runs in a sandboxed subprocess. Validation has up to 3 retry iterations.

### 1.4 Evaluation

We use SSDataBench's bootstrap-based statistical tests over five pattern types (we report the three that have data in our subsets):

- **Type 1** — univariate marginal distributions (proportions for categoricals, KS test for numerics).
- **Type 2** — bivariate association strength (chi-square for cat×cat, correlation for num×num).
- **Type 3** — regression coefficient preservation (OLS fit on real vs sim, bootstrap test of coefficient differences).

Higher pass rate = the bootstrap test fails to reject the null that real and synthetic distributions are identical, meaning more realistic generation.

## 2. Cross-dataset results

### 2.1 Headline

Mean pass rate (overall, averaged over types 1–3) per condition × dataset:

| Condition | GSS-2018 | CPS-1980 | ACS-1980 |
|---|---:|---:|---:|
| `full_agent` | **0.631** | failed¹ | 0.443 |
| `agent_no_semantic` | 0.506 | 0.544 | **0.655** |
| `agent_no_data` | 0.181 | 0.175 | 0.179 |
| `direct_generation` | 0.273 | 0.200 | 0.189 |

¹ *CPS `full_agent` failed in two independent runs at different stages (GENERATION script produced no output file; MODELING response had no fenced code block). The orchestrator currently has no retry on malformed agent output. See §4.*

The robust finding: **`agent_no_semantic` beats `direct_generation` on every dataset, by a margin that grows from +0.23 (GSS) → +0.34 (CPS) → +0.47 (ACS)**. The data-only agent is *uniformly stronger* than per-individual prompting that has full per-call descriptions.

### 2.2 Per-type breakdown

| Dataset | Condition | Type 1 | Type 2 | Type 3 | Mean |
|---|---|---:|---:|---:|---:|
| GSS | full_agent          | 0.586 | 0.767 | 0.540 | 0.631 |
| GSS | agent_no_semantic   | 0.458 | 0.585 | 0.475 | 0.506 |
| GSS | direct_generation   | 0.084 | 0.455 | 0.280 | 0.273 |
| GSS | agent_no_data       | 0.002 | 0.540 | 0.000 | 0.181 |
| CPS | agent_no_semantic   | 0.501 | 0.773 | 0.357 | 0.544 |
| CPS | direct_generation   | 0.253 | 0.346 | 0.000 | 0.200 |
| CPS | agent_no_data       | 0.001 | 0.520 | 0.003 | 0.175 |
| ACS | agent_no_semantic   | 0.414 | 0.861 | 0.690 | 0.655 |
| ACS | full_agent          | 0.361 | 0.526 | 0.000 | 0.443 |
| ACS | direct_generation   | 0.114 | 0.452 | 0.003 | 0.189 |
| ACS | agent_no_data       | 0.005 | 0.531 | 0.000 | 0.179 |

Three patterns hold across all datasets:

1. **`agent_no_data` is a flat floor at ~0.18** — without real data, prior knowledge produces calibration-free distributions independent of which dataset we're emulating. Its type 1 pass rate is essentially zero everywhere; only type 2 (bivariate associations) survives, because broad demographic correlations are part of the LLM's training prior.
2. **`agent_no_semantic` is the consistent ceiling**, often the best condition.
3. **`direct_generation` degrades across datasets** (0.273 → 0.200 → 0.189), suggesting per-individual prompting struggles more as the target variable count grows (GSS: 5, CPS: 8, ACS: 9).

### 2.3 Surprise: semantics can hurt

On ACS, **`agent_no_semantic` (0.655) outperforms `full_agent` (0.443) by +0.21**. The data-only agent fitted a tighter generative model than the agent given both data and descriptions. Looking at type 3 (regression preservation), the gap is even more striking: `agent_no_semantic` scores 0.690 while `full_agent` collapses to 0.000.

A plausible explanation: with descriptions, the agent inserts plausibility-based constraints (allowed-value enforcement, sanity assertions, conditional gates inferred from variable names) that distort empirical conditional distributions. Without descriptions, the agent has no temptation to "fix" the data and just samples from joint structure as observed. We do not yet have direct evidence — the ACS `full_agent` final code differs structurally from `agent_no_semantic`'s — but this is consistent with the failure pattern.

This is the most actionable finding for the design space: **giving the agent both data and semantics is not Pareto-better than data alone**.

### 2.4 Cross-dataset GSS still has the headline win

On GSS, `full_agent` (0.631) is the best by +0.13 over `agent_no_semantic` (0.506). The original "agent paradigm wins" finding holds for GSS — but does not generalize to CPS (where it failed to run) or ACS (where it underperformed). The narrower claim — *data-driven sampling beats per-individual prompting* — is what generalizes.

## 3. What the agent built

For `full_agent` on GSS, the LLM autonomously produced (in `MODELING`):

1. A `SimpleGaussianCopulaModel` with:
   - Empirical marginals (with Laplace smoothing) for each categorical variable
   - A Gaussian copula on rank-transformed numerics (multivariate normal on `norm.ppf(ECDF(x))`)
   - Inverse-ECDF interpolation when sampling
2. Mode/median imputation for missing values
3. Rounding for integer-valued numerics (`child_number`)

The agent diagnosed that `income` was fully NaN and dropped it; noted `birth_year` was collinear with `age` and dropped it. Validated by sampling 500 rows and printing marginal comparisons before generating the final 1000-row dataset.

For `full_agent` on CPS (the run that failed), the agent built a sequential `ConditionalChain` (root demographics → sequential GLM/logistic per target) and used it for generation, but the final-step generation script either silently emitted no `generated.csv` (first run) or the LLM returned a planning narrative without code (retry). Both failures suggest an instability in agent code emission for larger target schemas (CPS has 8 targets vs GSS's 5).

For `agent_no_data` across all three datasets, the LLM defaulted to wide-prior distributions (e.g. `age ~ Uniform(18, 89)`, marital-status proportions guessed). The near-identical scores across datasets (0.175–0.181) confirm this is a prior-knowledge floor invariant to which population we're trying to simulate.

## 4. What goes wrong, and what's robust

**Robust:** The data-driven > per-individual finding (`agent_no_semantic` vs `direct_generation`) replicates on three datasets with growing margins. The `agent_no_data` floor is essentially constant. Both signals come from condition-by-design rather than agent-quality variance.

**Fragile:** The full agent (data + semantics) is the highest-variance condition. On GSS it wins; on CPS it fails to produce output twice; on ACS it underperforms its own data-only ablation. Two distinct CPS failure modes were observed:
- *Generation script produces no output file* (assertions inside agent code likely tripped on out-of-range values; full diagnosis blocked by missing per-step stdout, since fixed in commit `62bcd5b`).
- *LLM returns prose without a fenced code block* (orchestrator has no retry policy for this).

**Engineering follow-ups (already in the issue list):**
- Add a single-shot retry in the orchestrator when LLM response has no code block (would have rescued the CPS retry).
- Make `format_generated` tolerant of missing schema columns so unseen-variable runs produce a low pass rate instead of crashing the eval (fixed in commit `62bcd5b`).
- Persist sandbox stdout/stderr to disk incrementally so failed runs are debuggable without rerunning (fixed in commit `62bcd5b`).

## 5. Limitations

- **Three datasets, all U.S. cross-sectional surveys.** Findings may not transfer to longitudinal (NLSY, Add Health) or non-U.S. (CFPS) datasets.
- **One model.** All numbers come from `deepseek-v4-flash`. Whether the no_semantic > direct_generation gap holds at lower-capability models is open.
- **Schema-pruned evaluation.** Our eval covers 10–14 variables of each dataset's documented 25–30; direct comparisons to SSDataBench's paper numbers are not 1-to-1.
- **n=1000.** Pass rates likely improve at larger sample sizes (especially type 1 marginals).
- **No unseen-variable headline yet.** `pilot_gss_unseen` ran but the agent omitted the held-out column from its output, crashing the original eval. The eval is now robust to missing columns; rerun pending.
- **Single seed.** No variance estimates across reruns. The CPS full_agent failures suggest variance is non-trivial for the headline condition.

## 6. Reproducing

```bash
git clone --recurse-submodules git@github.com:houx15/SSDataAgent.git
cd SSDataAgent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -r ssdatabench/requirements.txt
cp .env.example .env  # then add your LLM_API_KEY

# Run the three cross-dataset pilots
python scripts/run_experiment.py --experiment pilot_gss
python scripts/run_experiment.py --experiment pilot_cps
python scripts/run_experiment.py --experiment pilot_acs
```

Per-condition logs under `results/<experiment>/<condition>/<dataset>/<run_id>/`. The agent's Python code, every LLM exchange, sandbox stdout/stderr (now preserved on failure), and the final generated CSV are all there for inspection.

Approximate runtime per pilot: ~15 min for the three agent conditions, ~2 hours for `direct_generation` (1000 sequential per-individual LLM calls, network-bound).

## 7. Next steps

1. **Stabilize `full_agent`** — add code-block-missing retry; investigate why ACS's full agent collapsed on type 3 vs the no-semantic ablation. The `agent_no_semantic > full_agent` gap on ACS is the most interesting structural finding to chase.
2. **Rerun `pilot_gss_unseen`** with the now-tolerant eval to get a real number for SPEC criterion 3 (zero-shot held-out variable prediction).
3. **Multiple seeds** for the four conditions × three datasets to get variance bands. Particularly important given the CPS full_agent instability.
4. **Vary model capability** — re-run `agent_no_semantic` and `direct_generation` against a smaller model. If the data-driven advantage requires reasoning capability, that's an interesting ceiling claim.
5. **Investigate why semantics hurt on ACS** — diff the `full_agent` and `agent_no_semantic` MODELING code for ACS to identify the specific plausibility-constraint that distorted the joint distribution.
6. **Longitudinal datasets** (NLSY, Add Health) — the per-individual paradigm has a clearer claim there because individuals have temporal context the agent paradigm has to model explicitly.

---

*Code, full traces, and per-run logs are available at the project repository root. Per-pilot results: `results/pilot_gss/`, `results/pilot_cps/`, `results/pilot_acs/`.*
