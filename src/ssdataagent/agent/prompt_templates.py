from __future__ import annotations


SYSTEM_PROMPT = """\
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


def exploration_prompt(*, has_data: bool, has_descriptions: bool) -> str:
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


def modeling_prompt(*, findings_summary: str) -> str:
    return (
        "STAGE: MODELING.\n\n"
        f"Your findings so far:\n{findings_summary}\n\n"
        "Write a single Python block that fits a generative model on `train.csv`"
        " (or, if no data is available, defines one from the descriptions) and"
        " saves it to `model.pkl` using `cloudpickle.dump`. The saved object must"
        " expose a callable\n"
        "    sample(n: int) -> pandas.DataFrame\n"
        "returning rows with the same columns as the *target* schema. Free choice"
        " of model family — JointDistribution, ConditionalChain, GaussianCopula,"
        " statsmodels GLMs, etc. Keep it simple and fast.\n\n"
        "IMPORTANT — preserve the missingness pattern. Many variables are"
        " conditionally missing by survey design (e.g., 'age at first marriage'"
        " is NaN for never-married respondents, 'spouse occupation' is NaN for"
        " unmarried, 'income' is NaN for those out of the labor force). Do NOT"
        " impute these to a value — the downstream regressions depend on the"
        " missingness structure. Your sample(n) output must produce NaN in the"
        " same conditional pattern as the training data."
    )


def validation_prompt() -> str:
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


def generation_prompt(*, n_rows: int, target_path: str) -> str:
    return (
        "STAGE: GENERATION.\n\n"
        f"Load `model.pkl` with cloudpickle and use it to generate exactly "
        f"{n_rows} synthetic individuals. Write the resulting DataFrame to "
        f"`{target_path}` (no index column). Ensure all schema columns are "
        "present and values are within their allowed sets / numeric ranges. "
        "Print 'GENERATED OK' on success."
    )
