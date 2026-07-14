"""OpenAI function-calling schemas for every tool the agent can invoke.

The shape matches the `tools=[...]` parameter of `chat.completions.create`:
each entry is `{"type": "function", "function": {name, description, parameters}}`
where `parameters` is a JSON Schema object describing the args.

The agent sees the `description` text — keep it action-oriented and tight,
and call out edge cases the agent should know about (e.g. NA handling).
"""
from __future__ import annotations


def _fn(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    out: dict = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        out["required"] = required
    return out


# ---------- Inspect family ----------

INSPECT_SCHEMAS = [
    _fn(
        "list_columns",
        "Return per-column metadata (name, dtype, n_unique, n_missing, missing_rate) "
        "for every column in the training sample. Use this first to learn the schema.",
        _obj({}),
    ),
    _fn(
        "describe_column",
        "Stats summary for one column. Numeric: mean/std/quartiles/min/max. "
        "Categorical: top 20 value_counts + n_categories. Always includes n_missing and missing_rate.",
        _obj({"col": {"type": "string", "description": "Column name."}}, ["col"]),
    ),
    _fn(
        "cross_tab",
        "Contingency table between two (typically categorical) columns. "
        "Set normalize=true to return relative frequencies instead of raw counts.",
        _obj(
            {
                "col1": {"type": "string"},
                "col2": {"type": "string"},
                "normalize": {"type": "boolean", "default": False},
            },
            ["col1", "col2"],
        ),
    ),
    _fn(
        "missing_pattern",
        "Distribution of NA patterns across the supplied columns. Pattern is a string "
        "of '1' (present) and '0' (NA) in the column order. Use this to discover "
        "structural missingness (e.g. age_first_childbirth NA exactly when child_number=0).",
        _obj(
            {
                "cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to compute the pattern over. Order matters.",
                },
            },
            ["cols"],
        ),
    ),
    _fn(
        "correlation",
        "Pearson or Spearman correlation between two numeric columns. Drops rows where "
        "either is NA. Returns coef and n. Use to check pairwise associations before "
        "deciding which conditionals to fit for T2.",
        _obj(
            {
                "col1": {"type": "string"},
                "col2": {"type": "string"},
                "method": {"type": "string", "enum": ["pearson", "spearman"], "default": "pearson"},
            },
            ["col1", "col2"],
        ),
    ),
    _fn(
        "groupby_stat",
        "Per-group statistic of a numeric column. stat ∈ {mean, median, std, min, max, count}. "
        "Capped at 50 groups; for higher-cardinality grouping pre-bin or pick a different group_col.",
        _obj(
            {
                "group_col": {"type": "string"},
                "value_col": {"type": "string"},
                "stat": {
                    "type": "string",
                    "enum": ["mean", "median", "std", "min", "max", "count"],
                    "default": "mean",
                },
            },
            ["group_col", "value_col"],
        ),
    ),
    _fn(
        "head_rows",
        "First N rows of the training sample as a list of dicts. Useful for sanity-checking "
        "the data shape. n is capped at 50.",
        _obj(
            {"n": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50}},
        ),
    ),
]


# ---------- Fit family ----------

FIT_SCHEMAS = [
    _fn(
        "set_generation_order",
        "Declare the order columns will be sampled in. Required before any fit_conditional. "
        "Calling again replaces the order. Best practice: list every column you intend to model, "
        "with conditioned-on (background) variables before their dependents.",
        _obj(
            {"cols": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
            ["cols"],
        ),
    ),
    _fn(
        "fit_marginal",
        "Fit P(col) and register it — an iid draw that ignores every other column. Families: "
        "'empirical' (bootstrap from the column, any dtype), 'kde' (numeric only), 'normal' "
        "(numeric only), 'categorical_empirical' (alias for empirical). `allow_missing` defaults "
        "to true so the simulated missing rate matches real; setting it false emits a complete "
        "column, which changes which rows survive the eval's dropna and therefore which "
        "subpopulation T1/T3/T4 score. NOTE: a marginal reproduces the distribution of `col` "
        "perfectly but destroys every association it has with other columns, so T2/T3 for this "
        "column collapse. Prefer fit_block_donor or fit_conditional unless the column really is "
        "independent.",
        _obj(
            {
                "col": {"type": "string"},
                "family": {
                    "type": "string",
                    "enum": ["empirical", "categorical_empirical", "kde", "normal"],
                    "default": "empirical",
                },
                "allow_missing": {"type": "boolean", "default": True},
            },
            ["col"],
        ),
    ),
    _fn(
        "fit_conditional",
        "Fit P(col | given) and register it. `given` columns must already appear before "
        "`col` in generation_order — call set_generation_order first. Families: "
        "'empirical_lookup' (per-key bootstrap, any dtype, falls back to global empirical "
        "for unseen keys), 'linear_regression' (numeric col), 'logistic_regression' "
        "(categorical col, ≥2 classes). Set allow_missing=true to also model P(col is NA | given) "
        "— required for T3 when col has structural missingness.",
        _obj(
            {
                "col": {"type": "string"},
                "given": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "family": {
                    "type": "string",
                    "enum": ["empirical_lookup", "linear_regression", "logistic_regression"],
                    "default": "empirical_lookup",
                },
                "allow_missing": {"type": "boolean", "default": False},
            },
            ["col", "given"],
        ),
    ),
    _fn(
        "fit_copy_real",
        "Sentinel: register `col` to be drawn iid from its empirical distribution at "
        "sample time. Useful for columns you don't want to condition on anything else.",
        _obj({"col": {"type": "string"}}, ["col"]),
    ),
    _fn(
        "fit_block_donor",
        "PREFERRED for any group of related columns. Register `cols` as one BLOCK: for each "
        "simulated row a single real donor row is matched on `given` (binned, with fallback to "
        "coarser keys when a cell is thin) and that donor's whole block is copied verbatim. "
        "Because the values are real and copied together: the marginal is exact, NaN and "
        "censoring sentinels ('never married') are reproduced, and everything inside the block "
        "stays mutually consistent — chronological order of life events, and flag/value pairs "
        "like ever_married vs age_at_first_marriage. Put columns in the SAME block when they "
        "constrain each other; condition on the upstream causes. Unlike empirical_lookup this "
        "does not need an exact key match, so it keeps conditioning instead of silently falling "
        "back to the global marginal.",
        _obj(
            {
                "cols": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "given": {"type": "array", "items": {"type": "string"},
                          "description": "already-generated columns to match donors on"},
                "min_cell": {"type": "integer", "default": 25,
                             "description": "minimum donors in a cell before conditioning on it"},
            },
            ["cols"],
        ),
    ),
    _fn(
        "replace_step",
        "Drop the existing fit for `col` so a new fit_marginal / fit_conditional can replace it. "
        "Use when verify-tools say a fit isn't good enough. If `col` belongs to a block, the "
        "whole block is dropped — a block is only meaningful as a unit.",
        _obj({"col": {"type": "string"}}, ["col"]),
    ),
]


# ---------- Verify family ----------

VERIFY_SCHEMAS = [
    _fn(
        "sample_preview",
        "Sample n rows from the in-progress chain and return per-column summaries "
        "(same shape as describe_column). Use this to sanity-check fits before scoring "
        "or committing. Unfit columns are temporarily filled with empirical for the preview.",
        _obj(
            {"n": {"type": "integer", "default": 100, "minimum": 10, "maximum": 1000}},
        ),
    ),
    _fn(
        "score_marginal",
        "Score one column's marginal: KS for numeric (pass if KS ≤ 0.10), TV-distance "
        "for categorical (pass if TV ≤ 0.10). Compares chain's sample against the held-out "
        "20% slice. Returns the metric value, pass bool, and threshold.",
        _obj({"col": {"type": "string"}}, ["col"]),
    ),
    _fn(
        "score_pair",
        "Score one pair: |Δ pearson| for two numeric cols, |Δ Cramér's V| for two categorical. "
        "Pass threshold 0.15. Use to verify T2 (pairwise associations).",
        _obj({"col1": {"type": "string"}, "col2": {"type": "string"}}, ["col1", "col2"]),
    ),
    _fn(
        "score_event_order",
        "Compute the fraction of sim rows where the supplied age-event columns are "
        "non-decreasing in the given order. Pass threshold 0.80. Use only on longitudinal "
        "datasets to verify T4 (event-time chronology).",
        _obj(
            {"events": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
            ["events"],
        ),
    ),
    _fn(
        "score_overall",
        "Composite health check for the whole chain: the marginal (T1) pass rate AND the "
        "conditional (T3) pass rate, averaged into `composite_score`. Optimize composite_score, "
        "not marginal_pass_rate — replacing a conditional step with fit_marginal makes that "
        "column's marginal perfect by construction while destroying its conditional structure, "
        "so chasing the marginal rate alone will drive T3 to zero.",
        _obj({}),
    ),
    _fn(
        "score_conditional",
        "Score P(col | given) the way the benchmark's T3 does: fit an OLS of `col` on `given` "
        "in both the real held-out data and the current chain's sample, and compare the two R² "
        "values (T3 grades R², NOT coefficients). Returns r2_real, r2_sim, a p-value, and n_real "
        "vs n_sim. Two failure signatures to watch for: r2_sim near 0 means `col` is effectively "
        "independent of its predictors (a marginal step, or an empirical_lookup falling back to "
        "the global pool); n_sim far larger than n_real means the regression is fit on a "
        "different subpopulation, usually because `col` lost its missingness.",
        _obj(
            {
                "col": {"type": "string"},
                "given": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
            ["col", "given"],
        ),
    ),
]


# ---------- Commit family ----------

COMMIT_SCHEMAS = [
    _fn(
        "commit_generator",
        "Signal end-of-modeling. The orchestrator will sample N rows from your chain. "
        "Call this only after sample_preview / score_overall confirm the chain looks right. "
        "Any column in generation_order that's still unfit will be auto-filled with empirical.",
        _obj({}),
    ),
    _fn(
        "report_progress",
        "Append a free-form narrative note to the run log (e.g. 'fitting age_first_child as "
        "linear_regression on (age, child_number) because of high correlation observed'). "
        "Has no effect on the chain — purely for human debugging of the run.",
        _obj({"message": {"type": "string"}}, ["message"]),
    ),
]


def all_tool_schemas() -> list[dict]:
    """Concatenate all tool schemas. Called by the orchestrator to assemble
    the `tools=[...]` argument for chat_with_tools()."""
    return [*INSPECT_SCHEMAS, *FIT_SCHEMAS, *VERIFY_SCHEMAS, *COMMIT_SCHEMAS]
