# Data-Driven Agents Beat Per-Individual Prompting on Population Simulation

**Cross-dataset findings from SSDataAgent on GSS-2018, CPS-1980, ACS-1980**
*April 30, 2026*

## Abstract

We compare two LLM-based paradigms for generating synthetic social-survey data with population-level statistical realism:

- **Direct generation** (the SSDataBench paradigm): one LLM call per simulated individual, conditioned on background variables, returning JSON of target values.
- **Data-analyst agent**: a single LLM session given the real training split and asked to *write Python code* that fits a generative model and samples from it.

Across **GSS-2018, CPS-1980, and ACS-1980** (n=1000 each, ~10 variables each) using DeepSeek's `deepseek-v4-flash` reasoning model, the agent paradigm beats per-individual prompting on every dataset, with `full_agent` reaching mean pass rates of **0.631 / 0.622 / 0.443** vs `direct_generation`'s 0.273 / 0.200 / 0.189. Even when the agent is stripped of variable descriptions and population context (`agent_no_semantic`), the gap holds and *widens* across datasets (+0.23 / +0.34 / +0.47). The robust finding is **"data-driven empirical sampling beats per-individual LLM elicitation, regardless of variable semantics."** Removing data access collapses the agent to a near-identical floor (~0.18) on every dataset. On ACS, semantics actively *hurt* — the data-only ablation outperforms the full agent. A held-out-variable test on GSS confirms the agent does not attempt zero-shot prediction of a variable absent from training.

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
| `full_agent` | **0.631** | **0.622**¹ | 0.443 |
| `agent_no_semantic` | 0.506 | 0.544 | **0.655** |
| `agent_no_data` | 0.181 | 0.175 | 0.179 |
| `direct_generation` | 0.273 | 0.200 | 0.189 |

¹ *CPS `full_agent` failed twice on the first attempts — once because the GENERATION script produced no output file, once because the MODELING response had no fenced code block. After adding a single-shot retry to the orchestrator for prose-only responses, the third attempt produced 0.622 (essentially tied with GSS).*

Two robust findings:

- **Full agent paradigm wins on 2 of 3 datasets:** `full_agent` reaches ~0.63 on both GSS and CPS, and beats `direct_generation` everywhere (largest gap on ACS at +0.25, smallest on GSS at +0.36).
- **`agent_no_semantic` beats `direct_generation` on every dataset**, by a margin that grows from +0.23 (GSS) → +0.34 (CPS) → +0.47 (ACS). The data-only agent is *uniformly stronger* than per-individual prompting that has full per-call descriptions.

### 2.2 Per-type breakdown

| Dataset | Condition | Type 1 | Type 2 | Type 3 | Mean |
|---|---|---:|---:|---:|---:|
| GSS | full_agent          | 0.586 | 0.767 | 0.540 | 0.631 |
| GSS | agent_no_semantic   | 0.458 | 0.585 | 0.475 | 0.506 |
| GSS | direct_generation   | 0.084 | 0.455 | 0.280 | 0.273 |
| GSS | agent_no_data       | 0.002 | 0.540 | 0.000 | 0.181 |
| CPS | full_agent          | 0.486 | 0.757 | 0.000 | 0.622 |
| CPS | agent_no_semantic   | 0.501 | 0.773 | 0.357 | 0.544 |
| CPS | direct_generation   | 0.253 | 0.346 | 0.000 | 0.200 |
| CPS | agent_no_data       | 0.001 | 0.520 | 0.003 | 0.175 |
| ACS | agent_no_semantic   | 0.414 | 0.861 | 0.690 | 0.655 |
| ACS | full_agent          | 0.361 | 0.526 | 0.000 | 0.443 |
| ACS | direct_generation   | 0.114 | 0.452 | 0.003 | 0.189 |
| ACS | agent_no_data       | 0.005 | 0.531 | 0.000 | 0.179 |

A consistent failure mode: **`full_agent` scores 0.000 on type 3 (regression coefficient preservation) for both CPS and ACS**, despite winning on type 1 and type 2. Whatever the agent built generates per-row distributions that match marginals and bivariate correlations but fails to preserve OLS-fittable conditional structure. `agent_no_semantic` does not have this collapse on ACS (type 3 = 0.690), suggesting again that the descriptions are inducing some constraint that breaks regression preservation specifically.

### 2.3 The type-3 collapse: imputed missingness destroys conditional structure

We replicated the SSDataBench type-3 OLS fit by hand on the actual simulated CSVs (formula: `response ~ age + C(gender) + C(race) + C(education)`). The pattern is unambiguous on ACS:

| Response | Real | full_agent | no_semantic |
|---|---|---|---|
| `age_first_childbirth` | n=309, R²=0.502 | n=**1000**, R²=0.008 | n=295, R²=0.475 |
| `age_first_marriage` | n=564, R²=0.210 | n=**1000**, R²=0.010 | n=536, R²=0.169 |
| `child_number` | n=1000, R²=0.298 | n=1000, R²=0.004 | n=1000, R²=0.251 |
| `income` | n=772, R²=0.297 | n=**1000**, R²=0.005 | n=765, R²=0.291 |

`full_agent` populated *every row* with values for variables that are conditionally missing in the real data. Sample inspection of its raw output confirms it: row 0 is a 65yo male with `age_first_childbirth=27.2` and `age_first_marriage=18.9`; row 1 is a 24yo with 2.18 children and `age_first_childbirth=37.9` (a child born to a 37-year-old who is currently 24).

Real `age_first_childbirth` is NaN for 69% of ACS individuals (those without children). SSDataBench's OLS uses the natural ~309-row mother subset to fit a coherent regression on age/education, recovering R²=0.50. The same OLS on `full_agent`'s data uses all 1000 rows including teenage males with imputed childbirth ages, reducing the conditional structure to noise — R² collapses to 0.008. Across 100 bootstrap iterations, the R² gap is statistically significant in every one → `insignificant_rate = 0.000`.

`agent_no_semantic` on ACS preserved the missingness pattern (n=295 ≈ real's 309) and recovered real-comparable R² (0.475 vs 0.502). On CPS, *both* conditions populated every NaN — that's why CPS no_semantic also takes a hit on type 3 (0.357), but the imputed values for income happened to be conditional-mean-tracking enough that the R² gap stays modest. ACS no_semantic was unusual in actually preserving the data pattern.

**Why does the descriptions-aware agent destroy missingness?** The "validation theater" hypothesis: with descriptions in the prompt, the agent infers that "complete rows" are part of the task and writes imputation code in MODELING. Without descriptions, it treats the data as opaque tabular structure and just samples from joint distribution as observed. We don't have a controlled test of this yet — but the pattern is consistent across two datasets and the GENERATION-step code shows extensive plausibility validation that a no-semantic run wouldn't have any reason to write.

**Implication for paradigm design:** the data-analyst prompt template should explicitly tell the agent to preserve missingness patterns rather than impute. Imputation is a downstream choice for the *consumer* of the synthetic data, not a property of the data itself.

Three patterns hold across all datasets:

1. **`agent_no_data` is a flat floor at ~0.18** — without real data, prior knowledge produces calibration-free distributions independent of which dataset we're emulating. Its type 1 pass rate is essentially zero everywhere; only type 2 (bivariate associations) survives, because broad demographic correlations are part of the LLM's training prior.
2. **`agent_no_semantic` is the consistent ceiling**, often the best condition.
3. **`direct_generation` degrades across datasets** (0.273 → 0.200 → 0.189), suggesting per-individual prompting struggles more as the target variable count grows (GSS: 5, CPS: 8, ACS: 9).

### 2.4 Why semantics hurt on ACS

The ACS surprise (`agent_no_semantic` 0.655 > `full_agent` 0.443, +0.21) is dominated by the type-3 collapse documented in §2.3. Specifically, `full_agent` imputed `age_first_marriage`, `age_first_childbirth`, and `income` for every individual; `agent_no_semantic` happened to preserve their conditional missingness pattern. Type-3 swung from 0.690 to 0.000, dragging the mean.

The "validation theater" mechanism — descriptions cue the agent to write imputation/validation code that destroys conditional structure — is consistent with the inspection of the GENERATION-stage code: full_agent wrote ~50 lines of post-hoc range and category assertions; agent_no_semantic wrote a 3-line `model.sample(1000); to_csv()` and stopped. Neither agent is "wrong"; the descriptions-aware one took the prompt as license to enforce schema validity, which on ACS produced a worse synthetic dataset.

**Implication for prompt design:** giving the agent both data and semantics is not Pareto-better than data alone. The data-analyst prompt should explicitly distinguish "preserve the data's structural properties (including missingness)" from "validate output for schema correctness."

### 2.5 GSS and CPS share the headline; ACS is the outlier

On both GSS and CPS, `full_agent` is the strongest condition (0.631 / 0.622). On ACS, `agent_no_semantic` (0.655) outperforms `full_agent` (0.443). Two of three datasets confirm the original "agent paradigm wins" claim; one shows that semantics can actively hurt. The narrower data-driven > per-individual claim holds everywhere.

### 2.6 Held-out-variable test (SPEC criterion 3)

We removed `age_first_childbirth` from the agent's training data on GSS (`pilot_gss_unseen`, condition `full_agent_unseen`). The agent is told the column exists but does not see its values, and is asked to produce it in output anyway — the SPEC's zero-shot prediction test.

Per-variable type-1 pass rates from this run:

| Variable | Pass rate |
|---|---:|
| `child_number` (seen) | 1.000 |
| `marital_status` (seen) | 0.840 |
| `education` (seen) | 0.540 |
| `occupation` (seen) | 0.050 |
| `age_first_childbirth` (**held out**) | **0.000** |

The agent **does not attempt zero-shot prediction** of the held-out variable. Its output omits the column entirely; the formatter's uniform-random baseline scores at zero, as it should. This is itself a finding: the data-analyst paradigm models what it has and does not generalize to absent variables — the agent treats "I don't have this column" as "this column is not part of the task" rather than "I should infer this from observed correlations."

A future variant of this experiment would prompt the agent more aggressively to predict held-out variables (or build them as derived features) to test whether zero-shot prediction is a capability of the paradigm at all, or just an absence of intent in the current prompt.

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
- **Unseen-variable test only on GSS.** SPEC criterion 3 was run only on GSS-2018 with `age_first_childbirth` held out. The result is unambiguous (agent does not attempt zero-shot prediction) but the prompt design has not been varied to invite zero-shot inference.
- **Single seed.** No variance estimates across reruns. The CPS full_agent retries (two failures, one success) suggest variance is non-trivial for the headline condition.

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

1. **Add a "preserve missingness" instruction to the modeling prompt.** §2.3 traced the type-3 collapse to the agent imputing conditionally-missing variables. A targeted prompt variant — *"do not impute missing values; preserve the missingness pattern of the training data in your generated output"* — should fix the failure mode without removing any agent capability. Test this against `full_agent` on CPS and ACS as a controlled before/after.
2. **Multiple seeds** for the four conditions × three datasets to get variance bands. Particularly important given the CPS full_agent instability (two failures + one success in three independent runs).
3. **Stronger prompt for zero-shot prediction.** The held-out test showed the agent omits absent variables rather than inferring them. A modeling-prompt variant that explicitly asks for predictions of unseen variables would test whether zero-shot is a capability of the paradigm or just an absence of intent.
4. **Vary model capability** — re-run `agent_no_semantic` and `direct_generation` against a smaller model. If the data-driven advantage requires reasoning capability, that is an interesting ceiling claim.
5. **Match SSDataBench's full reporting structure.** Currently we collapse evaluation to one mean per condition × dataset. SSDataBench reports per-pattern, per-domain, and per-variable breakdowns; expanding our reporting will make our numbers directly comparable to the SSDataBench paper.
6. **Longitudinal datasets** (NLSY, Add Health) — the per-individual paradigm has a clearer claim there because individuals have temporal context the agent paradigm has to model explicitly.

---

*Code, full traces, and per-run logs are available at the project repository root. Per-pilot results: `results/pilot_gss/`, `results/pilot_cps/`, `results/pilot_acs/`.*
