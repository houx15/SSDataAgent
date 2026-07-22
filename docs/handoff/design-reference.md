# Design reference — Grounded Whole-View Simulation of Social Science Data

> **Read `delta-plan.md` first.** This file is the *conceptual* design (thesis, strategies, conditions, metrics, console). `delta-plan.md` is the operative doc — it maps this design onto the existing `SSDataAgent` repo and owns the build order. Where this file's §13 build order (greenfield) and `delta-plan.md` disagree, **`delta-plan.md` wins** because the repo already implements much of the harness.

**Status:** Locked for implementation (v1). Conceptual reference for a coding agent.
**Date:** 2026-06-29
**Evaluation target:** SSDataBench (Xie et al., 2025).
**Constraint:** No model training/fine-tuning. We use existing datasets as inputs and orchestrate with off-the-shelf LLMs + classical statistics.

---

## 0. How to read this doc

This spec defines (1) the story we are testing, (2) the strategies to implement, and (3) the experiment harness. A coding agent should be able to build the harness and all five strategies from this document alone. Where a choice is left open, it is listed in §13. Sections §4–§11 are the implementation contract; keep interfaces stable so strategies and the scorer evolve independently.

---

## 1. Problem

SSDataBench evaluates whether LLM-generated synthetic individuals reproduce **population-level** statistical patterns of real survey data (five pattern types across seven surveys). Pass rates are low (<0.5) and do **not** improve with model capacity. The benchmark generates one synthetic individual at a time, conditioned only on that individual's demographics, then aggregates.

**Our diagnosis:** the failure is an artifact of **case-wise modal generation**, not a knowledge gap. Independent per-case draws from "guess the typical answer" collapse variance, shrink tails, and **over-determine** opinions/behaviors from demographics (inflated Cramér's V and R², low entropy). This is reproducible by any case-wise modal predictor.

## 2. Thesis & hypotheses

**Thesis.** Change the *unit of generation* from the individual to the distribution ("whole-view"), **ground** that distribution in available data or known moments, and orchestrate method selection + semantic priors with an LLM. Sell it as **simulation** (silicon sampling: demographics → unobserved opinion/behavior data); prove it with **distributional realism** tests.

**Primary hypotheses.**
- **H1 (collapse is mechanism, not capability).** A whole-view generator (emit the conditional distribution, then sample) substantially raises pass rates over case-wise LLMs without any training.
- **H2 (over-determination is the headline failure for opinion).** Case-wise LLMs systematically *underestimate within-group heterogeneity*; whole-view + grounding closes the conditional-entropy gap H(target | demographics).
- **H3 (grounding earns the LLM its keep).** With partial/transferred data, an LLM that supplies structure + priors beats both case-wise LLMs and pure statistical synthesizers — most clearly in transfer (B) and aggregate-only (C) conditions.

**Headline metric:** the **over-determination gap** = H_real(target | demographics) − H_sim(target | demographics), reported alongside SSDataBench pass rates.

## 3. Locked scope (v1)

| Decision | Choice |
|---|---|
| Primary simulation target | **GSS attitudes (opinion)** primary; **one longitudinal behavior dataset (NLSY life-events)** secondary (covers Type 4/5). |
| Information conditions | **All three rungs (A in-distribution, B transfer, C aggregate-only)** treated as equal first-class axes. |
| Strategies | B0 (floor), B1 (bar), S1 (diagnostic), **Designs A/B/C (core)**, S3 (capstone). |
| Training | None. |
| Base LLM | One strong main model + one cheap model for ablation (parameterized, §12). |

## 4. Data layer

**Source.** SSDataBench code + processed data from the project repos (confirm exact URLs at build time):
- Code: `github.com/lzszhu/SSDataBench`
- Data: `github.com/LemengLiang/SSDataBench-Data`

Seven surveys: NLSY, CFPS, Add Health, Understanding Society (longitudinal); U.S. Census, CPS-ASEC, GSS (cross-sectional). Six domains: demographics, SES, marriage, health, abilities, attitudes.

**Unified schema.** Ingest every survey to one internal representation:

```python
@dataclass
class Survey:
    name: str
    context: str                  # historical & regional context string
    rows: pd.DataFrame            # one row per real individual
    background_vars: list[str]    # demographics used as conditioning inputs
    target_vars: list[Var]        # variables to be generated
    # longitudinal only:
    sequence_vars: list[str] | None   # life-event sequence fields

@dataclass
class Var:
    name: str
    kind: Literal["numerical","categorical","ordinal","sequence"]
    allowed: list | tuple | None  # categories, or (min,max) for numerical
    description: str              # natural-language description from instrument
```

**Splits.** Per survey, deterministic split `train / dev / test` (e.g., 60/20/20, fixed seed). The **test split's target values are sealed**: only the scorer may read them. The 1,000-case sampling and bootstrap from the paper operate *inside* the scorer on the test split.

## 5. Information conditions (what a strategy may see)

A strategy is always given the **background variables of the test rows** plus the survey context. The condition gates everything else:

- **A — In-distribution.** May fit on the **train split of the same survey** (background + targets). Upper bound / ceiling; largely reducible to density estimation — report it as such.
- **B — Transfer.** May fit on a **source survey** (different population/wave/country) with targets; must predict targets for a **different target survey's** test rows. Never sees the target survey's targets. Define explicit source→target pairs, e.g. `NLSY→CFPS`, `GSS(year t)→GSS(year t+k)`, `CPS-ASEC(t)→CPS-ASEC(t+k)`. This is where LLM contextual knowledge is tested.
- **C — Aggregate-only.** May see **only marginals** (and an optional small set of published pairwise associations) of the target survey's target variables — **no microdata targets**. Reconstruct the joint. Implement marginals as computed-from-train-then-withheld-microdata, or from published toplines.

Implement as a single gate object that each strategy must call to obtain data; the gate enforces the budget and logs exactly what was exposed.

```python
class InfoGate:
    def __init__(self, condition: Literal["A","B","C"], survey, source_survey=None): ...
    def background(self) -> pd.DataFrame: ...        # always allowed (test rows)
    def fit_microdata(self) -> pd.DataFrame | None:  # A: train split; B: source survey; C: None
    def known_marginals(self) -> dict | None:        # C (and optional elsewhere)
    def known_associations(self) -> dict | None:     # C optional
```

## 6. Strategy interface (the contract)

Every strategy — baseline or proposed — implements one method:

```python
class Strategy(Protocol):
    name: str
    def generate(self, gate: InfoGate, targets: list[Var], config: dict) -> pd.DataFrame:
        """Return a synthetic dataset: one row per test background row,
        with all target_vars filled. Must only use data exposed via `gate`."""
```

A **run** is fully described by a YAML config:

```yaml
strategy: s2_grounded_hybrid
condition: B
survey: CFPS
source_survey: NLSY        # condition B only
base_model: <main|cheap>   # LLM strategies only
seed: 0
params: { ... }            # strategy-specific
```

## 7. Strategies

### B0 — Case-wise LLM (reference floor)
Replicates the paper. For each test row, prompt the base model with context + background variables + target descriptions/allowed values; require strict JSON of target values (≤3 retries on schema failure). Aggregate to a synthetic dataset. No data access beyond background (condition-independent; this is the paradigm we are beating).

### B1 — Statistical synthesizers (non-LLM baselines)
Fit classical conditional generators on whatever the gate exposes, sample targets. Implement three backends and report the strongest per dataset: (a) **hot-deck / k-NN imputation** — draw a real neighbor's target; reproduces the conditional joint *by construction* and is the honest bar (expect it to beat the LLM in condition A); (b) **sequential CART / synthpop-style** chained models; (c) a **Bayesian network** (`pgmpy`) or **copula** (`copulas`) for mixed types. In condition C, fit via **IPF / max-entropy** to known marginals. If a cheap baseline already passes, that is itself a headline finding.

### S1 — Distribution-as-output (diagnostic arm)
Do **not** generate cases. Prompt the base model to emit the **conditional distribution** P(target | demographic cell) directly — full categorical probabilities or parametric form for numerical targets — then sample N per cell to match the test composition. Optional raking to `known_marginals`. **Role: diagnostic, not contender.** It isolates *sampling collapse* (fixed by whole-view) from *prior collapse* (fixed only by grounding): run S1 with and without raking to show whole-view alone closes part of the over-determination gap and grounding closes the rest. It samples targets independently, so it is mainly a Type 1/4 instrument and is expected to fail Types 2/3/5. *Variant — mixture-of-personas:* the LLM enumerates latent subtypes per cell and their weights before sampling, to test whether the model *knows* within-group diversity or just won't express it.

### Core designs — the grounded generators (the scientific contribution)
Three distinct engines, all grounded, all covering conditions A/B/C. Build order: **B → C → A.** Each must explicitly model the joint over targets (Types 2/3/5) and report the over-determination gap.

**Design A — Structure-first: LLM-elicited hierarchical Bayesian generator.** LLM proposes a DAG over background+target variables, per-node functional forms, and deliberately **wide priors** on attitude nodes; build in PyMC/NumPyro. Calibrate: A → fit posterior on train; B → fit on source, LLM specifies population-level offsets, partial pooling blends; C → constrain the likelihood to known marginals, priors fill the rest. Sample from the posterior predictive conditioned on test backgrounds. *Fixes:* joint via the DAG, over-determination via wide priors + calibration, sparse cells via partial pooling. *Strongest in:* C and B. *Cost:* highest; risk = LLM mis-specifying the DAG.

**Design B — Decomposition-first: whole-view marginals + copula coupling.** Partition into demographic cells; the LLM emits each target's full conditional distribution per cell (prompted for spread/tails, optionally via elicited quantiles); **rake** each marginal to the known/train marginal (the grounding that fixes prior collapse); fit a **copula** (Gaussian/vine on latent scores for categoricals) for cross-target dependence — from data in A/B, LLM-proposed-with-shrinkage in C (shrinkage toward independence counters stereotyped over-association). Sample correlated uniforms → push through the calibrated marginals. *Two independent, ablatable knobs* (marginal raking vs. copula) map onto the two collapse modes. *Strongest in:* A/B; most interpretable. *Cost:* medium. **Build this first.**

**Design C — Data-first: retrieval-grounded generation with distributional repair.** Retrieve k nearest real individuals by background (A: same-survey train; B: source survey; C: a synthetic pool seeded from marginals); **hot-deck** a target vector (preserves the real joint by construction); in condition B the LLM **transports** the borrowed target to the new context as a distributional mapping (not free per-case guessing); a **repair loop** then compares the synthetic aggregate to known marginals/associations and applies **SIR reweighting / raking** until moments match, with the LLM diagnosing which dimension is off. *Fixes:* resampling can't collapse; LLM used surgically. *Strongest in:* B (transfer). *Weakest in:* C (no exemplars — must bootstrap a pool first). Watch memorization/privacy.

### S3 — Data-analyst agent (capstone, not a core arm)
The autonomous explore → select → implement & fit → self-validate (dev split, frozen metrics) → iterate (bounded budget) → emit-generative-artifact loop, respecting the gate budget and **sealed from the test split**. **Role: demonstrate end-to-end automation/generality, not to test H1–H3** — it muddies attribution, so build it last and frame it as "can this be automated," internally deploying Design A/B/C tactics.

## 8. Scorer & metrics (frozen)

Local, deterministic reimplementation of SSDataBench's evaluation. Fixed seeds for all bootstrap. Inputs: real test data + a strategy's synthetic dataset. Outputs: per-pattern-type, per-variable pass rates + diagnostics.

**Significance tests (paper Table 1):**

| Type | Pattern | Test |
|---|---|---|
| 1 | Univariate, numerical | Kolmogorov–Smirnov |
| 1 | Univariate, categorical | Pearson χ² |
| 2 | Bivariate, num×num | Fisher's z on Pearson r |
| 2 | Bivariate, cat×cat | delta-method z on Cramér's V |
| 2 | Bivariate, cat×num | delta-method z on η² |
| 3 | Multivariate prediction | delta-method z on R² |
| 4 | Life-event sequence distribution | Pearson χ² |
| 5 | Sequence × covariate (cat) | delta-method z on Cramér's V |
| 5 | Sequence × covariate (num) | delta-method z on η² |

**Pass rate.** Bootstrap: 100 iterations; each draws 500 with replacement from real and from synthetic; run the test at p=0.05; pass rate = fraction of iterations **not** significant. Higher = more realistic.

**Diagnostics (the explanatory layer):**
- **Over-determination gap (headline):** H_real(target | demographics) − H_sim(target | demographics).
- Marginal **entropy gap** per categorical target.
- **Cramér's V inflation** per categorical pair.
- **R² inflation** per regression.

Validate the reimplementation against the original repo's numbers on B0 before trusting any result (§13 risk).

## 9. Runner & results store

- **Run unit:** `strategy × condition × survey (× source_survey) × pattern_type × seed`.
- **Log per run:** all pass rates + diagnostics (per variable), token/$ cost, wall-clock, base model, config hash, git commit, and artifacts (generated distributions/code, raw synthetic dataset).
- **Store:** parquet or DuckDB/SQLite results table + an `artifacts/` dir keyed by run id. One row per (run, pattern_type, variable) for easy slicing.
- **CLI:** `run --config <yaml>`, `score --run <id>`, `report --compare <ids>`.

## 10. Leaderboard & lab notebook

- **GitHub** (connected): version strategies, configs, scorer, results summaries. Never edit a strategy in place — fork the config.
- **Notion** (connected): a leaderboard database (run → metrics) + a lab notebook with one entry per iteration: **hypothesis → change → result → interpretation → next step.** This is the "continually note strategy and results" requirement.

## 11. Anti-leakage discipline (non-negotiable)

- Iterate only against **dev**; the **test** split's targets are read solely by the scorer.
- S3's agent must be sandboxed from the test split entirely.
- Because the pass criterion *is* a significance test, any strategy that self-validates must do so on dev — otherwise it p-hacks the metric and results won't reproduce. Final test scoring happens **once** per locked config.

## 12. Tech stack & layout

Python 3.11+. `pandas`, `numpy`, `scipy`, `statsmodels` (tests); `pgmpy`/`copulas`/synthpop-style CART (B1, S2); an LLM client with `main` and `cheap` model slots set via env/config; `duckdb` or `sqlite` + `pyarrow` (results). Optional Notion via MCP.

```
configs/strategies/*.yaml      # one per run
src/data/                      # ingest, schema, splits
src/conditions/                # InfoGate (A/B/C)
src/strategies/{b0,b1,s1,s2,s3}.py
src/scoring/                   # tests, pass_rate, diagnostics
src/runner/                    # run, log, report
results/                       # db + artifacts/
notebook/                      # local mirror of Notion lab log
```

## 13. Build order & open questions

**Build order:**
1. Data layer + InfoGate + sealed test splits.
2. Frozen scorer — **GATE: reproduce the paper's B0 pass rates against the original repo before anything else proceeds.** No design is trusted until this matches.
3. B0 + B1 (floor and bar).
4. S1 diagnostic (isolates the two collapse modes).
5. **Design B** (first real design), then **Design C**, then **Design A**.
6. S3 capstone last.
7. Runner/results + Notion leaderboard wired in parallel from step 3.

**Must be decided by a human before/at handoff (research decisions a coding agent cannot invent):**
- **Transfer pairs + variable crosswalk (condition B).** Confirm which source→target pairs have alignable variables (e.g., is an NLSY↔CFPS attitude/demographic crosswalk actually feasible?). Without a concrete crosswalk, condition B is undefined. Safest first pair is *within-survey temporal* transfer (GSS year t → t+k), which shares a schema by construction.
- **Condition-C "known moments" rule.** Per dataset: which marginals (and which, if any, pairwise associations) count as "known" — computed-from-train-then-withheld vs. published toplines.
- **Sequence scope (Type 4/5).** All three designs are specified for tabular targets; life-event *sequences* need an explicit representation and may need a dedicated sub-design. **Recommended: v1 = attitudes/tabular first; sequences as a fast-follow**, consistent with "attitudes primary."
- **Base model identities** for `main` and `cheap`.

**Defaults the coding agent may take (stated, overridable):**
- Demographic-cell definition (Design B / S1): exact strata; fall back to a learned/coarsened partition when a cell is below a min-count threshold.
- **LLM reproducibility:** cache every one-time elicitation (Design A DAG/priors, Design B per-cell marginals, Design C transport mappings) keyed by config hash; fix temperature; log raw LLM I/O as artifacts so reruns are stable and cheap.
- S3 compute/$ budget ceiling per run.

**Top risks:** (a) scorer not matching the paper → fix before any claims; (b) condition-A numbers look great but are "just density estimation" → always report A as ceiling, lead with B/C + the over-determination gap; (c) S3 cost/variance → cap budget, prefer Design A/B/C tactics internally; (d) LLM nondeterminism → caching + fixed temperature + logged I/O.

---

## 14. Experiment control plane (web gateway)

A **local, single-user web console** that is the operational surface of the iteration loop: launch/manage runs, browse results, compare strategies, export reports, keep the lab notebook. Localhost only; no auth in v1.

**Architecture.**
- **Backend:** FastAPI (same language as the harness — imports the runner directly).
- **Store:** the existing DuckDB/SQLite results table + `artifacts/` dir; a `runs` table holds status (queued/running/done/failed), config hash, git commit, cost, timings.
- **Job execution:** lightweight subprocess-backed queue; status polled into the DB. Sequential by default, small concurrency cap configurable. (No Celery/Redis in v1.)
- **Frontend:** server-rendered **FastAPI + Jinja + HTMX** (default — fewest moving parts for a single-user local console; a React+Vite SPA is acceptable if the agent prefers), charts via Plotly.
- Run tracking is the project's own DuckDB/SQLite store (no MLflow/W&B dependency in v1 — keep it self-contained).

**Decided (v1):** custom local web app; report export in **HTML + Markdown**.

**Views (the console must provide all six):**
1. **Leaderboard** — sortable/filterable table: row per run/strategy, columns = pass rate by pattern type, the **over-determination gap**, and cost; filter by condition / dataset / base model. Highlights current best per (condition × dataset).
2. **Run launcher + status board** — pick or fork a config (form over the YAML in §6), enqueue, watch queued/running/done/failed with live logs; cancel and re-run.
3. **Run detail** — all metrics + diagnostics with per-variable breakdown; artifacts (generated distributions, code, raw synthetic data, **logged LLM I/O**); the exact config + git commit for reproducibility.
4. **Compare** — select N runs → metric diff, the **strategy × pattern-type matrix** heatmap, and real-vs-simulated distribution plots side by side.
5. **Report export** — select runs → render a templated report in **HTML** (self-contained, interactive plots) and **Markdown** (paste-ready for the notebook / Notion / a paper draft), containing methods, leaderboard, over-determination gap, key real-vs-sim plots, and the linked notebook interpretation; downloadable.
6. **Lab notebook** — entries (hypothesis → change → result → interpretation → next) each linked to the runs that motivated them; mirrors to Notion if connected.

**Endpoint sketch:** `POST /runs` (enqueue), `GET /runs` / `GET /runs/{id}`, `GET /leaderboard`, `POST /compare`, `POST /reports`, `GET/POST /notebook`.

**The console must make one loop fast** (this is the whole point): *spot a failure in a run (which pattern type / which variable) → fork its config with a tweak → launch → compare against the parent → log the result.* If forking + launching + comparing takes more than a few clicks, the design has failed its purpose.
