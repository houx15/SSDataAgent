"""Inspect-family tools — read-only views over `train_fit`.

Tools always return a JSON-serializable dict. On error they return
`{"error": "...", "details": "..."}` — they never raise. The orchestrator
forwards the dict back to the LLM as a `tool` message; the agent recovers
by calling a different tool or different arguments.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ssdataagent.agent.tools.state import RuntimeState, data_withheld_error


_TOP_VALUE_COUNTS = 20         # cap categorical value_counts so tool results stay small
_MAX_GROUPBY_GROUPS = 50       # cap groupby_stat to avoid pathological wide outputs


def _to_jsonable(x: Any) -> Any:
    """Make a value JSON-serializable — np scalar → python scalar, NaN → None."""
    if x is None:
        return None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, (np.ndarray,)):
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def _missing_check(state: RuntimeState) -> dict | None:
    """Return data_withheld_error() if the run shouldn't see data; else None."""
    if not state.has_data:
        return data_withheld_error()
    return None


def _column_missing(state: RuntimeState, col: str) -> dict | None:
    """Return an error dict if `col` isn't in train; else None."""
    if col not in state.train_fit.columns:
        return {
            "error": "unknown_column",
            "details": f"column {col!r} not in train; available: {list(state.train_fit.columns)}",
        }
    return None


def list_columns(state: RuntimeState) -> dict:
    """Return per-column metadata for every column in train_fit."""
    if (refusal := _missing_check(state)) is not None:
        return refusal
    df = state.train_fit
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "name": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "n_missing": int(s.isna().sum()),
            "missing_rate": _to_jsonable(s.isna().mean()),
        })
    return {"columns": rows, "n_rows": int(len(df))}


def describe_column(state: RuntimeState, col: str) -> dict:
    """Stats summary for one column. Numeric vs categorical view chosen by dtype."""
    if (refusal := _missing_check(state)) is not None:
        return refusal
    if (err := _column_missing(state, col)) is not None:
        return err
    s = state.train_fit[col]
    n_missing = int(s.isna().sum())
    missing_rate = _to_jsonable(s.isna().mean())
    if pd.api.types.is_numeric_dtype(s):
        nn = s.dropna()
        if len(nn) == 0:
            return {
                "col": col, "kind": "numeric", "all_missing": True,
                "n_missing": n_missing, "missing_rate": missing_rate,
            }
        return {
            "col": col,
            "kind": "numeric",
            "n_missing": n_missing,
            "missing_rate": missing_rate,
            "mean": _to_jsonable(nn.mean()),
            "std": _to_jsonable(nn.std()),
            "min": _to_jsonable(nn.min()),
            "q25": _to_jsonable(nn.quantile(0.25)),
            "q50": _to_jsonable(nn.quantile(0.50)),
            "q75": _to_jsonable(nn.quantile(0.75)),
            "max": _to_jsonable(nn.max()),
            "n_unique": int(nn.nunique()),
        }
    # Categorical / object / bool path
    counts = s.value_counts(dropna=True)
    head = counts.head(_TOP_VALUE_COUNTS)
    return {
        "col": col,
        "kind": "categorical",
        "n_missing": n_missing,
        "missing_rate": missing_rate,
        "n_categories": int(s.nunique(dropna=True)),
        "value_counts": [
            {"value": _to_jsonable(v), "count": int(c)}
            for v, c in head.items()
        ],
        "value_counts_truncated": int(len(counts) > _TOP_VALUE_COUNTS),
    }


def cross_tab(state: RuntimeState, col1: str, col2: str, normalize: bool = False) -> dict:
    """Contingency table between two categorical-ish columns."""
    if (refusal := _missing_check(state)) is not None:
        return refusal
    for c in (col1, col2):
        if (err := _column_missing(state, c)) is not None:
            return err
    df = state.train_fit
    ct = pd.crosstab(df[col1], df[col2], dropna=False)
    if normalize:
        total = ct.values.sum()
        if total == 0:
            return {"error": "empty_crosstab", "details": "both columns are entirely NaN"}
        ct = ct / total
    return {
        "col1": col1,
        "col2": col2,
        "rows": [_to_jsonable(v) for v in ct.index.tolist()],
        "cols": [_to_jsonable(v) for v in ct.columns.tolist()],
        "values": [[_to_jsonable(v) for v in row] for row in ct.values.tolist()],
        "normalized": bool(normalize),
    }


def missing_pattern(state: RuntimeState, cols: list[str]) -> dict:
    """Distribution of missingness patterns across the supplied columns.

    Pattern is a string of '1' (present) and '0' (NA), in the order of `cols`.
    Returned rows are sorted by descending fraction.
    """
    if (refusal := _missing_check(state)) is not None:
        return refusal
    if not cols:
        return {"error": "empty_cols", "details": "supply at least one column"}
    for c in cols:
        if (err := _column_missing(state, c)) is not None:
            return err
    df = state.train_fit[list(cols)]
    presence = df.notna().astype(int).astype(str)
    pattern = presence.agg("".join, axis=1)
    counts = pattern.value_counts()
    n = int(len(df))
    out = []
    for pat, count in counts.items():
        out.append({
            "pattern": pat,
            "count": int(count),
            "fraction": _to_jsonable(count / n),
        })
    return {"cols": list(cols), "n_rows": n, "patterns": out}


def correlation(state: RuntimeState, col1: str, col2: str, method: str = "pearson") -> dict:
    """Pearson or Spearman correlation between two numeric columns."""
    if (refusal := _missing_check(state)) is not None:
        return refusal
    for c in (col1, col2):
        if (err := _column_missing(state, c)) is not None:
            return err
    if method not in ("pearson", "spearman"):
        return {"error": "unknown_method", "details": f"method must be 'pearson' or 'spearman', got {method!r}"}
    df = state.train_fit[[col1, col2]].dropna()
    if len(df) < 3:
        return {"error": "too_few_rows", "details": f"only {len(df)} non-NaN paired rows"}
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": "non_numeric", "details": f"correlation needs numeric columns; got {df.dtypes.to_dict()!r}"}
    coef = df[col1].corr(df[col2], method=method)
    return {
        "col1": col1, "col2": col2, "method": method,
        "coef": _to_jsonable(coef),
        "n": int(len(df)),
    }


def groupby_stat(
    state: RuntimeState,
    group_col: str,
    value_col: str,
    stat: str = "mean",
) -> dict:
    """Per-group summary statistic of a numeric column."""
    if (refusal := _missing_check(state)) is not None:
        return refusal
    for c in (group_col, value_col):
        if (err := _column_missing(state, c)) is not None:
            return err
    if stat not in ("mean", "median", "std", "min", "max", "count"):
        return {"error": "unknown_stat", "details": f"stat must be one of mean/median/std/min/max/count; got {stat!r}"}
    s_value = state.train_fit[value_col]
    if stat != "count" and not pd.api.types.is_numeric_dtype(s_value):
        return {"error": "non_numeric_value", "details": f"stat={stat} needs numeric value_col; got {s_value.dtype}"}
    grouped = state.train_fit.groupby(group_col, dropna=False)[value_col]
    if stat == "count":
        agg = grouped.count()
    else:
        agg = getattr(grouped, stat)()
    n_per_group = grouped.size()
    if len(agg) > _MAX_GROUPBY_GROUPS:
        return {
            "error": "too_many_groups",
            "details": f"group_col has {len(agg)} unique values; cap is {_MAX_GROUPBY_GROUPS}. Try a different grouping or pre-bin.",
        }
    rows = []
    for grp_value in agg.index:
        rows.append({
            "group_value": _to_jsonable(grp_value),
            "stat_value": _to_jsonable(agg.loc[grp_value]),
            "n": int(n_per_group.loc[grp_value]),
        })
    return {
        "group_col": group_col,
        "value_col": value_col,
        "stat": stat,
        "groups": rows,
    }


def head_rows(state: RuntimeState, n: int = 5) -> dict:
    """First N rows of train_fit as records (list of dicts)."""
    if (refusal := _missing_check(state)) is not None:
        return refusal
    if n < 1 or n > 50:
        return {"error": "bad_n", "details": "n must be between 1 and 50"}
    head = state.train_fit.head(n)
    rows = [
        {col: _to_jsonable(val) for col, val in row.items()}
        for _, row in head.iterrows()
    ]
    return {"n": len(rows), "rows": rows, "columns": list(head.columns)}
