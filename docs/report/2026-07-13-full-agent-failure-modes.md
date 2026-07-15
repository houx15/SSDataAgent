# Why `full_agent` scores badly — four compounding bugs

**Date:** 2026-07-13
**Scope:** diagnosis of the tool-using agent (`full_agent`) on `acs` and `cfps`.
**Status:** diagnosed, not yet fixed. Fix plan at the bottom.

---

## TL;DR

`full_agent`'s poor T3/T4 scores are **not a model-quality problem**. We built a
scoring instrument that rewards destroying conditional structure, a commit gate
that cannot be satisfied, a `fit_marginal` that annihilates missingness, and no
tool at all for the censored columns that T3/T4/T5 are built on. The LLM behaved
rationally inside that box.

Evidence is from two committed runs:

- acs: `results/exp006e_tools_cross/full_agent/acs/20260508-035039`
- cfps: `results/exp006f_tools_diag/full_agent/cfps/20260511-155049`

---

## Architecture recap

`full_agent` never writes rows. It is a tool-using LLM that builds a **sequential
generative chain** (a hand-rolled Bayes net) over the dataset's columns, then
samples from it.

- **Inspect** — `list_columns`, `describe_column`, `cross_tab`, `correlation`,
  `groupby_stat`, `missing_pattern`
- **Fit** — `set_generation_order`, then one *Step* per column:
  - `fit_marginal` → iid draw (empirical / KDE / normal). Ignores all other columns.
  - `fit_conditional(col, given=[...])` → `empirical_lookup`,
    `linear_regression`, or `logistic_regression` on earlier columns.
  - `fit_copy_real` → iid draw from the real column.
- **Verify** — `score_marginal`, `score_pair`, `score_event_order`,
  `score_overall`, `sample_preview`
- **Commit** — `commit_generator`, then `chain.sample(n)` walks the order.

Budget: 40 turns. Code: `src/ssdataagent/agent/tools/{fit,verify,commit}.py`.

---

## Bug 1 — `score_overall` is marginal-only, so the agent Goodharts it

`score_overall` (`verify.py:297`) runs `score_marginal` on every column and
returns a `pass_rate`. It measures **per-column KS / TV-distance and nothing
else**. There is no T3 tool anywhere in the toolset.

Given that instrument, `replace_step(col) → fit_marginal(col, "empirical")` is
the **optimal** move: an iid draw from the real column reproduces the real
marginal *perfectly, by construction*.

**acs, verbatim from `tool_calls.json`:**

Turns 1–8, the agent built a good chain:

```
age                  | birth_year, gender, race                    R² = 0.9996
marital_status       | age, gender, race, immigrant_status        acc = 0.78
child_number         | marital_status, age_first_marriage, gender  R² = 0.479
income               | education, marital_status, child_number, …  R² = 0.293
```

Turn 15: `score_overall` → `pass_rate: 0.538`.

Turn 16–17 — it deleted every conditional for **exactly the four T3 response
variables** and refit them as marginals:

```
replace_step(child_number), replace_step(income),
replace_step(age_first_marriage), replace_step(age_first_childbirth)
fit_marginal(child_number, empirical)   fit_marginal(income, empirical) …
```

Turn 19: committed — with 21 of 40 turns unused.

**The trade did not even work.** `pass_rate` stayed at 0.538, and the pairwise
scores got *worse* (`child_number × income` |Δr| blew up from 0.090 → 0.426).
The agent had that on screen at turn 18 and committed anyway.

### Consequence: T3 measures R², and ours goes to zero

`type3.py` does **not** compare regression coefficients. It fits
`y ~ predictors` separately on real and sim and compares **R²** via a
delta-method z-test, `B` bootstrap iterations, score = fraction with p > 0.05.

Measured on acs (`full_agent` vs real):

| response | R² real | R² sim |
|---|---|---|
| age_first_marriage | 0.210 | 0.009 |
| age_first_childbirth | 0.502 | 0.024 |
| child_number | 0.298 | 0.011 |
| income | 0.202 | 0.008 |

z = 6 to 14. Every bootstrap iteration fails. **T3 = 0.000, exactly.**

### It generalizes across the fleet

| dataset | T3 responses left as independent marginals | T3 score |
|---|---|---|
| nlsy | 0 / 6 | **0.548** |
| gss | 0 / 4 | 0.330 |
| cps | 0 / 3 | 0.147 |
| addhealth | 3 / 5 | 0.220 |
| cfps | 6 / 10 | 0.244 |
| us | 9 / 10 | 0.266 |
| **acs** | **4 / 4** | **0.000** |

Orphaning a response variable is *sufficient* to guarantee failure. (cps shows
it is not the whole story — the fitted R² must also land — but it is the
dominant term.)

---

## Bug 2 — `score_event_order` crashes, making the commit gate unsatisfiable

The event-age columns are `object` dtype: they mix numeric ages with **string
sentinels**.

| cfps column | sentinel | count / 1000 |
|---|---|---|
| age_started_work | `'never worked'` | 410 |
| age_at_first_marriage | `'never married'` | 223 |
| age_at_first_child | `'never had child'` | 74 |
| age_at_death | `'still alive'` | 969 |

`score_event_order` does a naive float cast:

```
ValueError: could not convert string to float: 'never worked'
```

**On cfps it can never succeed.** And `commit_generator` gates on it having
succeeded (`error: missing_event_order_check`). **The exit is welded shut.**

**cfps death spiral, verbatim:**

- t9, t14, t17, t20: `score_event_order` → `tool_internal_error` (4×)
- t16: `commit_generator` → `missing_event_order_check`
- t18–19: agent replaces all 5 event ages with `copy_real` (iid — destroys
  chronology outright)
- t21: commit → error again
- t23–38: one column at a time → `categorical_empirical` marginals, retry, fail.
  **Six consecutive failed commits.**
- t42: max_turns → force-commit with `t4_unverified=True`

The agent cannot know the *tool* is broken, so it concludes its *model* is wrong
and progressively mutilates exactly the columns T4 and T5 measure. **It burns
turns 21–42 — half its budget — flailing.**

We have been reading `T4 = 0.000` as "the benchmark is degenerate at N=1000."
Part of it is. But **on cfps the agent was never even trying.**

---

## Bug 3 — `fit_marginal` destroys missingness

`fit_marginal` does `nn = s.dropna()` and samples from what's left. It **never
re-emits NaN**. There is no `allow_missing` parameter. A column that is 53%
missing in real comes out **0% missing** in sim.

| cfps column | occurred: real | occurred: sim |
|---|---|---|
| age_at_first_marriage | 43.5% | **68.0%** |
| age_at_first_child | 32.4% | **81.4%** |
| age_started_work | 24.2% | **38.0%** |

`age_at_first_child` = 398 numeric + 74 `'never had child'` + 528 NaN. Drop the
NaN and you draw 398/472 = 84.3% numeric — which is the 81.4% observed.

**T4-eligible rows (all three events numeric): real = 200, sim = 555.** The sim
is 2.8× too eventful. T4 was never going to score — sim and real are not even
scoring the same subpopulation.

**Bugs 1 and 3 compound:** the instrument *pushes* the agent toward
`fit_marginal`, which is the one step type that annihilates missingness.

---

## Bug 4 — censored columns are unmodellable

`fit_conditional` checks `is_numeric_dtype(col)`. `age_at_first_marriage` is
`object`, so:

- `linear_regression` → **rejected** (`family_dtype_mismatch`)
- `logistic_regression` → "accepted", treating **each distinct age as a class**
  (~60 classes)
- `empirical_lookup` → the only sane option

**The agent literally cannot fit a timing model on the columns T3, T4 and T5 are
all built on.** It wasn't being lazy — we gave it no tool.

---

## The right representation: censored numerics + a hurdle model

These columns are not dirty. They are **three-state**, and the state carries the
meaning:

```
NaN             → not observed         (we don't know)
'never married' → event never occurred (censored)
34              → occurred at age 34   (timing)
```

**This is exactly the structure the eval is built on.** T1, T3 and T4 all do
`to_numeric(errors='coerce')` then `dropna()`. So *which rows carry a numeric
age determines who enters the regression at all*:

- **T3** — the regression for `age_at_first_marriage` is fit only on the
  married. Wrong occurrence rate → the R² is computed on a different
  subpopulation.
- **T4** — only rows where *all three* events are numeric enter at all.
- **T1** — scored among the occurred.

So the correct model is a **hurdle**:

1. `P(status | X)` where status ∈ {missing, never, occurred} — multinomial logistic
2. `P(age | occurred, X)` — regression or lookup, fit only on those who occurred
3. **Sample:** draw status; emit `NaN` / the sentinel string / a drawn age

One new tool — `fit_hurdle` — reproduces missingness and occurrence *by
construction* and makes the timing genuinely modellable for the first time.

---

## Minor bugs found along the way

- `ConditionalStep.to_meta()` never records `given`, so `chain.json` loses the
  entire dependency graph. We cannot audit our own models from the artifact.
- `score_overall` compares **100 real rows to 200 sim rows**. `birth_year`
  "failed" its KS test at p = 0.24. Half of what the agent reacts to is
  sampling noise.

---

## Data findings (cfps)

- `real_data/used_dataset/sampled_cfps.csv` = **1,000 rows**;
  `real_data/cfps/cfps_2010_2022.csv` = **58,474 rows**. `pid` is a unique key
  and **all 1,000 benchmark rows are a subset of the full source** → a
  provably-disjoint training pool of 57,474 rows is available.
- `train_eval_split: 0.5` means the agent currently fits on **500 rows**. In
  those 500: `self_control` ≈ 45 non-null, `mean_income_30_40` ≈ 122,
  `fixed_mindset` ≈ 182. Several T3 responses cannot be fit at all. **Part of
  the marginal-collapse is genuine starvation, not only Goodhart.**
- The eval's `real_csv` is the **500-row held-out half** (`experiments/runner.py`
  passes `sampled=eval_df` → `write_simulated`). The paper scores against the
  full 1,000. **Our numbers are not strictly comparable today**, and the real
  side of every bootstrap is noisy.

**Decision (2026-07-13, with user):** pin the eval reference to the paper's exact
1,000-row `sampled_cfps.csv`, and draw training data from the disjoint pool.
Training-set size `N` becomes a clean experimental axis. This is only possible
*because* we have the full source — today the 50/50 split is the only thing
preventing leakage.

---

## Bug 5 — `empirical_lookup` is a marginal in disguise

It keys on the **exact tuple** of `given` values. With a continuous column in the
key (`birth_year`, any `age_*`), most sample-time keys were never seen at fit time
and silently fall back to the global marginal. Measured on cfps, the fraction of
reference rows finding *no* matching cell:

| column | fallback @ train N=500 | @ N=20k |
|---|---|---|
| occupation_30_40 | **95.2%** | 69.8% |
| age_started_work | **90.6%** | 43.1% |
| age_at_first_child | **83.7%** | 81.5% |
| age_at_first_marriage | **61.1%** | 11.5% |

And the **median donor pool among rows that *do* match is 1** — even a "hit"
deterministically copies one specific person. The agent believed it was fitting a
conditional; it was fitting a marginal plus noise. (This also explains an earlier
false lead: scaling training data "improved the model" mostly by making
`empirical_lookup` stop falling back.)

---

## The real defect: no step family can preserve both structure and marginal

| family | conditional structure | marginal fidelity |
|---|---|---|
| `linear_regression` | yes | **no** — Gaussian noise smears the distribution |
| `logistic_regression` | yes | refuses numeric; treats each age as a class |
| `empirical_lookup` | **no** — 60–95% fallback | yes |
| `fit_marginal` | **no** — none at all | yes, but destroys missingness |

The agent was choosing the least-bad option from a menu with no good item. Its
"Goodhart collapse" was a *rational trade of T3 for T1* under a marginal-only
instrument. Confirmed by replay: the turn-8 chain it discarded scores **better on
T3 (+0.10) but much worse on T1 (−0.26)**, and is a wash overall. Fixing the
instrument alone would only have let it *see* the trade, not escape it.

---

## The fix: block-donor generation

`src/ssdataagent/strategies/block_donor.py`, `agent/tools/donor.py`.

Columns are grouped into **blocks** (from the `domain` field SSDataBench's own
schema already carries). For each simulated row, one real donor is matched on a
**coarsened** key over already-generated columns — with progressive fallback to
shorter keys and a `min_cell` floor — and that donor's **whole block is copied
verbatim**. Every failure above dies at once, by construction rather than tuning:

* values are real → the marginal is exact (**T1**)
* NaN and `'never married'` copy across → occurrence rates, and therefore *which
  rows survive the eval's dropna and enter the scored regressions*, are right
* a block moves as a unit → chronology and flag/value coherence hold exactly; no
  childless person acquires a first-birth age (**T2/T3/T4**)
* later blocks condition on the event ages already drawn → life-course order
  survives across block boundaries (**T4/T5**)
* the key is binned, not exact → conditioning without the fallback cliff

**Granularity is the modeling dial, and it is a genuine trade-off**: `"mega"`
(demography, then everything else as one block) maximizes pairwise fidelity and
wins T2; `"domain"` conditions harder and wins T3. That choice is exactly what a
modeler — or the agent — should be making deliberately.

## Results (cfps, scored on the paper's exact 1000-row reference)

> **Correction (2026-07-15).** An earlier version of this section claimed
> block-donor beat a resampling baseline 0.819 vs 0.765 and cleared the ceiling on
> four of five benchmarks. That was **seed noise read as signal**: those were
> 5-seed means, and T4 alone carries a per-seed standard deviation near 0.15, so
> the error bars were wider than the effect. Re-run over **12 paired seeds**,
> block-donor does **not** beat copying real rows — it is marginally worse — and
> the whole regime is **saturated**. The corrected numbers and the reasoning are
> below. The diagnosis of the four agent bugs (above) is unaffected and stands.

Two protocol changes, both required:

* **No more halving the benchmark.** `load_disjoint_train` draws training rows from
  the 58k full source *excluding* the 1000 benchmark rows, proven on `pid`. We used
  to cut the reference in half to manufacture held-out rows — which starved the
  model (400 fitting rows; `self_control` had ~45 observations) *and* scored us at
  half the paper's N.
* **More simulated rows.** The eval bootstraps 500 rows from sim per iteration, so
  handing it 5000 instead of 1000 cuts Monte-Carlo noise.

### Two different "ceilings" — this is the crux

The benchmark scores two *samples* for statistical distinguishability. That makes
"how high can anyone score" a real, measurable quantity, and there are two versions
of it that the earlier draft conflated:

* **sim = reference (0.913).** Score the benchmark's 1000 rows against *themselves*.
  This measures only the eval's own unseeded bootstrap noise. It is unreachable by
  anything that isn't literally the answer key, so it is not an achievable target.
* **fresh real people vs the benchmark (0.806).** Draw 5000 *other* real CFPS
  people (from the disjoint pool) and score them against the benchmark. These are
  real humans — nothing can be more "correct" — so **0.806 is the achievable
  ceiling for any method that copies from this pool.** The 0.10 gap up to 0.913 is
  the irreducible sampling difference between any two draws of the same population;
  closing it means matching the benchmark's specific sampling noise, which requires
  reading the benchmark = leakage.

The achievable ceiling, 0.806, is the number that matters. It *is* the resampling
baseline's own score — because resampling **is** "draw fresh real people".

### Head-to-head, 12 paired seeds, vs the paper's 1000-row reference

Mean ± standard error over seeds 1–12, production code path:

| | T1 | T2 | T3 | T4 | T5 | overall |
|---|---|---|---|---|---|---|
| PNAS (best of 15 LLMs) | 0.14 | 0.62 | 0.43 | 0.05 | 0.75 | 0.30 |
| our shipped agent | 0.411 | 0.512 | 0.244 | 0.000 | 0.650 | 0.363 |
| **whole-row resample** (= achievable ceiling) | **0.873** | 0.813 | **0.708** | 0.825 | 0.809 | **0.806** ±.008 |
| block-donor (mega) | 0.829 | 0.813 | 0.698 | 0.758 | **0.821** | 0.784 ±.014 |
| block-donor (domain) | 0.816 | 0.774 | 0.701 | **0.867** | 0.809 | 0.793 ±.007 |
| ceiling (sim = ref, not achievable) | 0.958 | 0.949 | 0.748 | 0.960 | 0.950 | 0.913 |

Paired `block-donor(mega) − resample`: overall **−0.022** (t = −1.38, not
significant); **no benchmark shows a significant win** for block-donor. It is a
statistical tie with resampling, landing very slightly below it.

**Why block-donor cannot win here, in principle.** Whole-row resampling is an
*unbiased* draw from the donor pool's distribution. Block-donor *approximates* that
same distribution and pays an independence penalty at the block seams. Fidelity is
exactly what T1–T5 measure, so resampling is an **upper bound** for any
copy-from-pool method — block-donor can at best tie it, never beat it. The
0.819-vs-0.765 "win" was an artifact.

### The real finding: the microdata regime is saturated

When you hold real microdata, **"reuse it" scores 0.806 — 2.7× the best published
LLM (0.30) — and nothing can beat it.** We are at 0.79, within 0.016 of the
achievable ceiling. The maximum score any modeling work can add here is that
0.016, and it is bounded above by literal real people. **Optimizing cfps-with-
microdata further is optimizing into a wall the benchmark itself imposes.**

This reframes what block-donor is for. Its advantages are the ones this
fidelity-only benchmark cannot see:

* **Privacy / disclosure.** Resampling republishes actual respondents (100%
  re-identification); block-donor stitches each synthetic person from several real
  donors, so no synthetic person is any real person. The benchmark never measures
  this, which is why it rewards a method that could not be deployed.
* **Transfer.** Donors can come from another wave or population — you take their
  *conditional structure* without importing their marginals. Resampling cannot.
* **The no-donor regime.** Where there are no donors at all (AGGREGATE / NO_DATA),
  resampling is impossible by construction and an LLM's prior is the only signal.
  That is the regime the PNAS premise is actually about, and the only one with real
  headroom left.

### T4 is winnable at N=1000 — this part still holds

The shipped agent's `T4 = 0.000` was our bugs, not the benchmark being degenerate.
Both resample (0.825) and block-donor-domain (0.867) score T4 well against the exact
1000-row reference. The three things that were missing from the agent:

1. reproduce the **missingness** (the agent admitted 555 rows into T4 where reality
   admits 200 — a 2.8× inflated, wholly different population),
2. draw the event ages **jointly** from one donor (40% of the agent's simulated life
   courses had the child before the wedding; reality: 2%),
3. hand the eval **more simulated rows**.

An earlier probe held all three fixed and blamed the *test*. The test was fine.

---

## Fix status

| Step | Work | Status |
|---|---|---|
| 1 | Offline T1–T5 bench + chain replay (`scratchpad/bench.py`) | done |
| 2 | Block-donor step family + strategy, granularity dial | done |
| 3 | Advisory (never-blocking) gates | done |
| 4 | `allow_missing` on `fit_marginal` (defaults **on**) | done |
| 5 | `ConditionalStep.to_meta()` records `given` | done |
| 6 | Leakage-proof disjoint training pool (`load_disjoint_train`) | done |
| 7 | `score_conditional` + composite `score_overall` (the instrument) | **pending** |
| 8 | Generalize to the other six datasets as full sources land | **pending** |

**Design decision (user, 2026-07-13):** gates are **advisory, never blocking**. A
gate warns and records but can never refuse a commit, so a broken or failing check
can never trap the agent again. The forcing function moves into the *score*, where
the cost of a bad choice is visible rather than punitive.

### Still open: the no-donor regime — where the actual problem now lives

The 12-seed result moves this from "future work" to "the main event". With microdata,
the benchmark is saturated (§Results): resampling scores 0.806 and is unbeatable, so
there is nothing left to model. Block-donor needs microdata — it has no meaning
without donors. `score_conditional` likewise cannot exist without a real R² to compare
against. In AGGREGATE / NO_DATA mode the agent must supply conditional structure from
knowledge alone, resampling is impossible by construction, and an LLM's prior is the
only available signal. **That is the regime with real headroom, and it needs its own
design pass.**
