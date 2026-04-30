# Data-Analyst Agents Beat Per-Individual Prompting on Population Simulation

**Initial findings from SSDataAgent on GSS-2018**
*April 30, 2026*

## Abstract

We compare two LLM-based paradigms for generating synthetic social-survey data with population-level statistical realism:

- **Direct generation** (the SSDataBench paradigm): one LLM call per simulated individual, conditioned on background variables, returning JSON of target values.
- **Data-analyst agent**: a single LLM session given the real training split and asked to *write Python code* that fits a generative model and samples from it.

On the GSS-2018 dataset (n=1000, 11 variables) using DeepSeek's `deepseek-v4-flash` model, the agent paradigm achieves a mean SSDataBench pass rate of **0.631**, compared to **0.273** for direct generation — a 2.3× improvement. Ablations confirm both training-data access and semantic context contribute, with data being the larger lever.

## 1. Setup

### 1.1 Model and infrastructure

All conditions use the same LLM: DeepSeek `deepseek-v4-flash`, an OpenAI-compatible reasoning model accessed via `https://api.deepseek.com`. Temperature = 1.0, max_tokens = 4096.

We re-implement SSDataBench's "direct generation" baseline using the same client (rather than calling SSDataBench's own simulation code, which is hard-wired to OpenRouter), so the only varying factor across conditions is the *paradigm* — same model, same temperature, same data.

### 1.2 Dataset

- Source: SSDataBench's cleaned GSS-2018 sample, 1000 individuals.
- Columns retained (after dropping `income` which was 100% NaN): `gender`, `age`, `birth_year`, `race`, `immigrant_status`, `marital_status`, `child_number`, `age_first_childbirth`, `education`, `occupation`.
- Train/eval split: 50/50 with fixed seed (500/500).

We auto-pruned SSDataBench's evaluation config (which expects ~28 GSS variables) down to the 10 columns we have, so only available variables enter the bootstrap tests.

### 1.3 Conditions

| Condition | Real data shown | Variable descriptions | Population context |
|---|:---:|:---:|:---:|
| `full_agent` | train split | yes | yes |
| `agent_no_semantic` | train split | no | no |
| `agent_no_data` | none | yes | yes |
| `direct_generation` | none | per-prompt | per-prompt |

Agent conditions run a fixed-stage pipeline: `EXPLORATION → MODELING → VALIDATION → GENERATION`. Each stage is a single LLM call returning Python that runs in a sandboxed subprocess. Validation has up to 3 retry iterations.

### 1.4 Evaluation

We use SSDataBench's bootstrap-based statistical tests over five pattern types (we report the three that have data in our subset):

- **Type 1** — univariate marginal distributions (proportions for categoricals, KS test for numerics).
- **Type 2** — bivariate association strength (chi-square for cat×cat, correlation for num×num).
- **Type 3** — regression coefficient preservation (OLS fit on real vs sim, bootstrap test of coefficient differences).

Higher pass rate = the bootstrap test fails to reject the null that real and synthetic distributions are identical, meaning more realistic generation.

## 2. Results

### 2.1 Headline

| Condition | Type 1 | Type 2 | Type 3 | **Mean** | Δ vs direct |
|---|---:|---:|---:|---:|---:|
| **full_agent** | 0.586 | 0.767 | 0.540 | **0.631** | +0.358 |
| agent_no_semantic | 0.458 | 0.585 | 0.475 | 0.506 | +0.233 |
| direct_generation | 0.084 | 0.455 | 0.280 | 0.273 | — |
| agent_no_data | 0.002 | 0.540 | 0.000 | 0.181 | −0.092 |

The agent paradigm with full context **more than doubles** the mean pass rate of direct generation, with the largest gain on univariate marginals (0.586 vs 0.084 — a 7× difference).

### 2.2 Ablation: where does the lift come from?

Removing components from `full_agent`:

- **Remove descriptions and population context** (`agent_no_semantic`): 0.631 → 0.506, **−0.125**. The agent loses about 20% of its pass rate without knowing what the columns *mean*. It still has the data and can fit empirical distributions, so the loss is modest.
- **Remove training data** (`agent_no_data`): 0.631 → 0.181, **−0.450**. Without data, the agent must invent plausible distributions purely from descriptions and prior knowledge of the GSS population. This is a much larger loss than removing semantics, and brings the agent below `direct_generation`.

Conclusion: **data access is the larger lever**, but semantics are not redundant — removing them costs about a third of what removing data costs.

### 2.3 Per-variable breakdown (Type 1, full_agent)

| Variable | Pass rate |
|---|---:|
| `child_number` | 0.93 |
| `education` | 0.74 |
| `marital_status` | 0.69 |
| `occupation` | 0.57 |
| `age_first_childbirth` | 0.00 |

The agent does very well on marginals where the empirical distribution is straightforward to sample (categorical proportions, child counts). It fails on `age_first_childbirth` — because this variable is conditional on `child_number > 0`, and the agent's Gaussian-copula model didn't preserve this conditional missingness pattern, leaving the variable defined for individuals who shouldn't have one.

### 2.4 Surprising secondary finding

`agent_no_semantic` (0.506) **outperforms `direct_generation` (0.273) by +0.23**, even though both are denied variable descriptions/context. This is striking: the agent gets only column names and unlabeled numbers, yet still beats per-individual prompting that has full descriptions for every call.

Two implications:
- The data-driven modeling step itself — fitting joint and conditional distributions on real samples — is doing most of the work.
- Per-individual LLM elicitation appears to be a *weaker* signal than empirical sampling, even with rich context per call.

## 3. What the agent actually built

For `full_agent`, the LLM autonomously produced (in `MODELING`):

1. A `SimpleGaussianCopulaModel` with:
   - Empirical marginals (with Laplace smoothing) for each categorical variable
   - A Gaussian copula on rank-transformed numerics (multivariate normal on `norm.ppf(ECDF(x))`)
   - Inverse-ECDF interpolation when sampling
2. Mode/median imputation for missing values
3. Rounding for integer-valued numerics (`child_number`)

The agent diagnosed that `income` was fully NaN and dropped it; noted `birth_year` was collinear with `age` and dropped it. It then validated by sampling 500 rows and printing marginal comparisons before generating the final 1000-row dataset. None of this was scripted — it emerged from a generic `data analyst` system prompt.

For `agent_no_data`, the LLM had only descriptions and tried to encode plausible distributions from prior knowledge (e.g. `age ~ Uniform(18, 89)`, marital-status proportions guessed). It got near-zero on type 1 and 3 — the prior knowledge is good for *coarse* shapes (which is why type 2 still scored 0.54) but specific marginals require empirical calibration.

## 4. Limitations

- **Single dataset.** Results are from GSS-2018 only. The pilot has not yet been run on CPS or ACS, nor on longitudinal datasets (NLSY, Add Health, etc.).
- **One model.** All numbers come from `deepseek-v4-flash`. Whether the agent advantage holds at lower-capability models is open.
- **Schema-pruned evaluation.** Our eval covers 10 of GSS-2018's documented 28 variables — direct comparisons to the SSDataBench paper's reported numbers are not 1-to-1.
- **n=1000.** Pass rates would likely improve at larger sample sizes by the law of large numbers (especially type 1 marginals).
- **No unseen-variable test yet.** SPEC criterion 3 (zero-shot prediction of held-out variables) is wired but not run.
- **Single seed.** No variance estimates across reruns.

## 5. Reproducing

```bash
git clone --recurse-submodules git@github.com:houx15/SSDataAgent.git
cd SSDataAgent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -r ssdatabench/requirements.txt
cp .env.example .env  # then add your LLM_API_KEY

python scripts/run_experiment.py --experiment pilot_gss
```

Per-condition logs under `results/pilot_gss/<condition>/gss/<run_id>/`. The agent's Python code, every LLM exchange, sandbox stdout/stderr, and the final generated CSV are all preserved per run for inspection.

## 6. Next steps

1. **Run `pilot_gss_unseen`** (test SPEC criterion 3 — zero-shot distributional prediction).
2. **Add CPS-1980 and ACS-1980** to test cross-dataset generality.
3. **Vary model capability** — re-run `full_agent` and `direct_generation` against a smaller model to test whether the agent advantage scales with reasoning capability.
4. **Multiple seeds** to get variance estimates.
5. **Investigate the `age_first_childbirth` failure** — explicitly prompt the agent about conditional missingness, see if it builds a hurdle/two-stage model.

---

*Code, full traces, and per-run logs are available at the project repository root.*
