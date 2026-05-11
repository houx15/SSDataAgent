# EXP-006e cps T3 collapse — root cause

> Investigation of why cps T3 fell from 0.537 (EXP-006c, rubric_tools_v2)
> to 0.147 (EXP-006e, rubric_tools_v3). Both runs are cross-sectional,
> same model (gpt-5.4), same n=1000, same data — only the system prompt
> changed.

## Final chain diff (cps)

| Column | v2 family (T3=0.537) | v3 family (T3=0.147) |
|---|---|---|
| birth_year | `empirical` (marginal) | `linear_regression` |
| marital_status | `empirical_lookup` | `logistic_regression` |
| child_number | `empirical_lookup` | `empirical_lookup` |
| age_first_childbirth | `empirical_lookup` | `empirical_lookup` |
| education | `logistic_regression` | `logistic_regression` |
| laborforce | `empirical_lookup` | `logistic_regression` |
| occupation | `empirical_lookup` | `empirical_lookup` |
| **income** | **`linear_regression`** | **`empirical_lookup`** |
| poverty_status | `logistic_regression` | `logistic_regression` |

Marital_status/laborforce flipping to logistic_regression is plausibly
fine — both are small-label categoricals. The income flip is the load-
bearing change.

## Why income matters

cps income is a continuous numeric target with hundreds of distinct
values. T3 (regression preservation) measures whether the joint
distribution preserves the coefficient β in `income = β · covariates`.

- `linear_regression(income | gender, age, race, marital_status, education, laborforce, occupation, …)`
  produces a smooth conditional mean that preserves β by construction.
- `empirical_lookup(income | 5+ categorical parents)` builds cells with
  ~5-10 training rows each at n=500 train. Within-cell empirical
  distributions are near-constant or fall back to the marginal, which
  flattens the conditional mean and destroys the regression.

Both runs followed the same `linear_regression → empirical_lookup`
trajectory for income at cycle 2 after `score_pair(age, income)` showed
|Δr| = 0.21–0.30 on the parametric fit. The difference is what the
agent did *after* empirical_lookup passed score_pair (|Δr| = 0.09):

- **v2** ran a third refit cycle that switched income BACK to
  `linear_regression` and re-tested. It still passed score_pair (|Δr| =
  0.08), and v2 committed with linear_regression.
- **v3** stopped after the second cycle and committed with
  empirical_lookup.

## Prompt diff that caused the difference

v2's `_FAMILY_RECIPE` block had a forbidding anchor for high-cardinality
numerics:

> "Use `empirical_lookup` for numeric targets ONLY when the column has
> ≤8 distinct values and is essentially categorical-encoded."

v3's `_FAMILY_RECIPE_V3` rewrote the numeric clause to add a separate
LIFE-EVENT-AGE category, and in doing so **deleted that ≤8-distinct
anchor**. v3 retains the `|Δr|>0.15` fallback rule (`fall back to
empirical_lookup if score_pair shows |Δr| > 0.15`) but no longer
forbids the fallback on high-cardinality targets.

Result: when the first-cycle linear_regression on income produced a
score_pair failure, v3's agent fell back to empirical_lookup and saw it
pass — no further rule pushed it back toward linear_regression. v2's
agent was told that empirical_lookup is forbidden on >8-value numerics,
so it had to keep trying linear_regression variants until one passed.

## Proposed fix (for an EXP-006g prompt patch)

Restore the cardinality anchor in `_FAMILY_RECIPE_V3`, scoped to make
clear it overrides the fallback rule. Suggested wording, to insert as
the third bullet under "Target is a CONTINUOUS NUMERIC variable":

> Hard rule: **do not** use `empirical_lookup` for a continuous numeric
> target with more than 8 distinct training values, even if the
> `|Δr| > 0.15` fallback above suggests it. Sparse high-cardinality
> cells collapse T3 worse than a noisy parametric fit. If
> `linear_regression` fails score_pair, try a different `given` set
> (drop noisy parents, add a stronger predictor) or accept the
> degradation — empirical_lookup is not the answer for income, age,
> or vocabulary-score-style columns.

The v3 LIFE-EVENT AGE clause is independent — it correctly tells the
agent to use empirical_lookup for `age_at_first_*`-style columns, and
those have small distinct-value sets per cell when conditioned on the
prior event. Keep it.

## Open questions

1. Did cps T3 collapse on cpc6e_long for the same reason? The
   longitudinal v3 prompt has the same `_FAMILY_RECIPE_V3` block; if
   v3 also flipped income to empirical_lookup on us/nlsy/cfps/addhealth,
   the same fix applies across all v3 runs. Worth checking the chain.json
   for each longitudinal cell before patching the prompt.
2. Should the `|Δr| > 0.15` fallback rule be removed entirely instead of
   anchored? The fallback was added in v2 specifically to fix the
   EXP-006b "linear_regression overconfidence" failure. If we instead
   relied only on the LIFE-EVENT AGE exception, we'd lose the v2 fix.
   So the anchor-the-fallback approach is the conservative patch.
3. The agent's third-cycle behavior in v2 wasn't deterministic — it
   happened to swing back to linear_regression. With a stronger
   forbidding rule in v3, the agent might just give up faster on a
   passing chain. Worth a smoke before the long batch.
