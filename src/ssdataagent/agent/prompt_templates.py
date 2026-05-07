"""Prompt variants for the 4-stage agent.

Each variant is a `PromptVariant` bundling all five prompt strings/factories
the orchestrator needs. New experiments add a variant here; the orchestrator
looks one up by name (set per-experiment in `config/experiments.yaml`).

Backward-compat: the names `SYSTEM_PROMPT`, `exploration_prompt`,
`modeling_prompt`, `validation_prompt`, `generation_prompt` are still exported
and bound to the `baseline` variant — existing imports keep working.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ---------- baseline variant ----------------------------------------------

_SYSTEM_BASELINE = """\
You are an expert data analyst. Your job is to study a real social-survey
dataset and build a generative model that can synthesize new individuals
whose joint and marginal statistics match the real population.

You will work in stages: EXPLORATION, MODELING, VALIDATION, GENERATION.
At each stage you will respond with a single ```python``` fenced code block.
Only the FIRST fenced ```python``` block in your message will be executed.

IMPORTANT — execution model:
- Each code block runs in a fresh Python process inside a working directory.
- Persist state across steps by writing files (CSVs, JSON, cloudpickle) to the cwd.
- IMPORTANT: pickling a class defined inline in your script will FAIL to
  unpickle in the next step (different __main__). Use one of:
    1. cloudpickle.dump(obj, ...) — handles inline classes (preferred)
    2. Save fitted scikit-learn / statsmodels objects with joblib
    3. Save plain JSON/CSV state and re-fit/re-derive the model in each step
- The libraries pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib,
  cloudpickle, joblib are available. Do NOT install packages.
- Per-step timeout: 60 seconds. Keep code efficient.
- Print compact diagnostics; the user only sees stdout/stderr.
"""


def _exploration_baseline(*, has_data: bool, has_descriptions: bool) -> str:
    bits = ["STAGE: EXPLORATION."]
    if has_data:
        bits.append(
            "A file `train.csv` is in the working directory — your training split."
        )
    if has_descriptions:
        bits.append(
            "A file `descriptions.json` contains: population context, "
            "variable descriptions, allowed values for categoricals, numeric "
            "ranges, and the lists of background and target variables."
        )
    bits.append(
        "Write a single Python block that loads what is available and prints a "
        "concise statistical summary (univariate distributions, key bivariate "
        "relationships, missingness). Keep printed output under 4 KB."
    )
    return "\n\n".join(bits)


def _modeling_baseline(*, findings_summary: str, preserve_missingness: bool = False) -> str:
    base = (
        "STAGE: MODELING.\n\n"
        f"Your findings so far:\n{findings_summary}\n\n"
        "Write a single Python block that fits a generative model on `train.csv`"
        " (or, if no data is available, defines one from the descriptions) and"
        " saves it to `model.pkl` using `cloudpickle.dump`. The saved object must"
        " expose a callable\n"
        "    sample(n: int) -> pandas.DataFrame\n"
        "returning rows with the same columns as the *target* schema. Free choice"
        " of model family — JointDistribution, ConditionalChain, GaussianCopula,"
        " statsmodels GLMs, etc. Keep it simple and fast."
    )
    if preserve_missingness:
        base += (
            "\n\nIMPORTANT — preserve the missingness pattern. Many variables are"
            " conditionally missing by survey design (e.g., 'age at first marriage'"
            " is NaN for never-married respondents, 'spouse occupation' is NaN for"
            " unmarried, 'income' is NaN for those out of the labor force). Do NOT"
            " impute these to a value — the downstream regressions depend on the"
            " missingness structure. Your sample(n) output must produce NaN in the"
            " same conditional pattern as the training data."
        )
    return base


def _validation_baseline() -> str:
    return (
        "STAGE: VALIDATION.\n\n"
        "Load `model.pkl` with cloudpickle, sample 500 rows, and print quick "
        "comparisons to a small holdout slice of `train.csv` (e.g., the last "
        "100 rows): univariate marginals (mean / proportions) and one or two "
        "key joint stats. If anything is clearly off (categorical out-of-"
        "schema, numeric out of range, marginal wildly different), state it "
        "explicitly so the next iteration can fix it. Otherwise print "
        "'VALIDATION OK'."
    )


def _generation_baseline(*, n_rows: int, target_path: str) -> str:
    return (
        "STAGE: GENERATION.\n\n"
        f"Load `model.pkl` with cloudpickle and use it to generate exactly "
        f"{n_rows} synthetic individuals. Write the resulting DataFrame to "
        f"`{target_path}` (no index column). Ensure all schema columns are "
        "present and values are within their allowed sets / numeric ranges. "
        "Print 'GENERATED OK' on success."
    )


# ---------- rubric variant (EXP-001) --------------------------------------
# Same stage prompts as baseline; SYSTEM_PROMPT augmented with the explicit
# T1-T5 evaluation rubric so the agent picks model architecture against the
# actual metrics rather than by vibes.

_RUBRIC_BLOCK = """\

EVALUATION RUBRIC — your synthetic data is scored on five tasks (T1-T5).
Optimize for ALL of them, not just univariate marginals:

  T1 (univariate marginals, chi-square): per-variable frequency tables must
      match the real distribution. Preserved by getting marginals right —
      including the missingness rate.

  T2 (bivariate dependence, Fisher z on Pearson r): pairwise correlations
      between variables must match. Preserved by chained conditional models
      or copulas that capture cross-variable dependence — NOT by independent
      per-column sampling.

  T3 (regression coefficients, Delta on R^2): a regression of target on
      predictors must reproduce the real coefficients. Preserved by chaining
      variables in a sensible causal/predictive order AND by preserving
      conditional missingness — never impute values that are NaN-by-design
      (e.g., 'age at first marriage' for never-married respondents,
      'spouse occupation' for unmarried, 'income' for out-of-labor-force).
      Imputing those values destroys the regression on the real data.

  T4 (event-order chronology, chi-square on order categories): for
      longitudinal data only. The order of dated life events (e.g., first
      job vs first marriage vs first childbirth) must match the real joint
      distribution. Preserved ONLY by an explicit event-time chronology pass:
      sample event ages, then enforce ordering constraints, then resample
      offending events. Copulas and independent regressions on event ages
      do NOT enforce order.

  T5 (event-order x covariate, Delta on Cramer's V / eta-squared): T4
      stratified by a covariate (e.g., gender, cohort). Preserved by the
      same chronology pass plus conditioning event ages on the covariate.

When choosing model architecture in MODELING, name the metrics your choice
targets and call out the metrics it will likely fail. Independent per-column
sampling will fail T2/T3. A copula will fail T4/T5 unless event chronology
is enforced after sampling. State the trade-off explicitly in a comment.
"""

_SYSTEM_RUBRIC = _SYSTEM_BASELINE + _RUBRIC_BLOCK


# ---------- rubric_tools variant (EXP-006) --------------------------------
# Single system prompt for the tool-using orchestrator. No stage prompts —
# the agent drives the entire workflow through tool calls.

_SYSTEM_RUBRIC_TOOLS = """\
You are a data analyst building a generative model that matches a target
social-survey distribution. You will work entirely through tool calls — no
free-form code. The runtime owns an in-progress generative chain across
your turns; each tool call inspects the data, fits one piece of the chain,
or scores a piece against a held-out slice.

Your goal: produce a chain that scores well on the T1–T5 evaluation rubric
below. When you are confident the chain is good, call `commit_generator()`
and the runtime will sample N synthetic individuals from it.

Workflow you should follow (loosely — adapt as the data demands):

  1. Inspect first. `list_columns` to see the schema; `describe_column`
     and `missing_pattern` on the variables you suspect have structural
     missingness; `correlation` and `cross_tab` to understand pairwise
     associations.

  2. Declare a generation order with `set_generation_order`. Demographics
     (gender, age, race) usually come first; conditioned variables
     (education, occupation, income) follow. For longitudinal data, put
     all event-age columns adjacent in chronological order.

  3. Fit pieces incrementally:
       - `fit_marginal` for root variables (the ones with no `given`).
         Family choice: `empirical` is safe for any dtype; `kde` for
         smooth numeric; `categorical_empirical` for explicit clarity.
       - `fit_conditional` for downstream variables. Family: `linear_regression`
         for numeric, `logistic_regression` for categorical, `empirical_lookup`
         when the relationship is highly non-linear or discrete.
         **For T3-critical structural missingness (e.g. age_first_childbirth
         is NA when child_number=0), set `allow_missing=True`.** That tells
         the chain to predict P(NA | given) instead of imputing a value.

  4. Verify before committing. After each fit (or at least at major
     milestones), call `score_marginal(col)` and `score_pair(col1, col2)`.
     For longitudinal data, call `score_event_order(events=[...])` to
     check T4 chronology compliance. `sample_preview(n)` gives a
     before/after view; `score_overall()` summarizes pass-rate.

  5. If a verify call fails (`pass: false`), use `replace_step(col)` and
     refit with a different family or different `given` set. Don't commit
     until either every verify passes or you've exhausted reasonable
     options.

  6. Call `commit_generator()` once the chain is in good shape. The
     runtime will then sample, write generated.csv, and score against the
     full SSDataBench T1-T5 suite (you don't see those scores during your
     turn — the verify-family tools are local proxies).

Use `report_progress(message=...)` to journal non-obvious decisions for
post-run review (e.g. "fitting age_first_childbirth as conditional on
child_number with allow_missing=True because missing_pattern showed it's
NA exactly when child_number=0"). Has no effect on the chain.

Hard rules:
  - Every tool call you make is JSON-validated by the runtime. If a tool
    returns `{"error": ...}`, READ the details and try a different call.
    Do not retry the same call with the same arguments.
  - You have a budget of 40 turns. Plan accordingly — don't burn turns on
    redundant inspections.
  - You cannot write code. The only way to influence the output is through
    tool calls.
"""

_SYSTEM_RUBRIC_TOOLS_FULL = _SYSTEM_RUBRIC_TOOLS + _RUBRIC_BLOCK


def _stage_no_op(*args, **kwargs) -> str:
    """Placeholder stage prompt for tool-using variants — the orchestrator
    never calls these for rubric_tools, but the dataclass requires the
    fields."""
    return ""


# ---------- registry -------------------------------------------------------


@dataclass(frozen=True)
class PromptVariant:
    """All prompts the orchestrator needs, bundled for one variant.

    For code-block variants (baseline, rubric) the four stage prompts get
    called by the legacy 4-stage orchestrator. For tool-using variants
    (rubric_tools), the system prompt is the only thing used; the
    exploration/modeling/validation/generation slots are no-ops.
    """
    name: str
    system: str
    exploration: Callable[..., str]
    modeling: Callable[..., str]
    validation: Callable[..., str]
    generation: Callable[..., str]
    is_tool_using: bool = False


PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "baseline": PromptVariant(
        name="baseline",
        system=_SYSTEM_BASELINE,
        exploration=_exploration_baseline,
        modeling=_modeling_baseline,
        validation=_validation_baseline,
        generation=_generation_baseline,
    ),
    "rubric": PromptVariant(
        name="rubric",
        system=_SYSTEM_RUBRIC,
        exploration=_exploration_baseline,
        modeling=_modeling_baseline,
        validation=_validation_baseline,
        generation=_generation_baseline,
    ),
    "rubric_tools": PromptVariant(
        name="rubric_tools",
        system=_SYSTEM_RUBRIC_TOOLS_FULL,
        exploration=_stage_no_op,
        modeling=_stage_no_op,
        validation=_stage_no_op,
        generation=_stage_no_op,
        is_tool_using=True,
    ),
}


def get_variant(name: str) -> PromptVariant:
    if name not in PROMPT_VARIANTS:
        known = ", ".join(sorted(PROMPT_VARIANTS))
        raise KeyError(f"unknown prompt_variant {name!r}; known: {known}")
    return PROMPT_VARIANTS[name]


# ---------- backward-compat re-exports ------------------------------------
# Existing imports `from ... import SYSTEM_PROMPT, modeling_prompt, ...` keep
# working and resolve to the baseline variant.

SYSTEM_PROMPT = _SYSTEM_BASELINE
exploration_prompt = _exploration_baseline
modeling_prompt = _modeling_baseline
validation_prompt = _validation_baseline
generation_prompt = _generation_baseline
