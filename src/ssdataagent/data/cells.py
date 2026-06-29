from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema


def bin_edges(vals, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(pd.to_numeric(vals, errors="coerce").dropna(), qs))
    if len(edges) < 2:
        edges = (np.array([edges[0] - 1e-9, edges[0] + 1e-9])
                 if len(edges) else np.array([0.0, 1.0]))
    return edges


def discretize(values, edges) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy()
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)


@dataclass
class CellScheme:
    variables: list[str]
    edges: dict[str, np.ndarray] = field(default_factory=dict)


def fit_scheme(df, variables, schema: DatasetSchema, *, n_bins: int = 4) -> CellScheme:
    edges: dict[str, np.ndarray] = {}
    for v in variables:
        if v in schema.numeric_ranges:
            edges[v] = bin_edges(df[v], n_bins)
    return CellScheme(variables=list(variables), edges=edges)


def assign(df, scheme: CellScheme) -> pd.Series:
    parts = []
    for v in scheme.variables:
        if v in scheme.edges:
            parts.append(discretize(df[v], scheme.edges[v]).astype(str))
        else:
            parts.append(df[v].astype(str).to_numpy())
    keys = ["|".join(t) for t in zip(*parts)] if parts else ["_"] * len(df)
    return pd.Series(keys, index=df.index)


def describe_cell(scheme: CellScheme, key: str) -> dict[str, str]:
    """Map a cell key back to human-readable demographic values: numeric vars
    -> the bin's "[lo,hi)" range string; categorical vars -> their value."""
    parts = key.split("|")
    desc: dict[str, str] = {}
    for v, p in zip(scheme.variables, parts):
        if v in scheme.edges:
            edges = scheme.edges[v]
            try:
                i = int(p)
            except ValueError:
                desc[v] = p
                continue
            i = max(0, min(i, len(edges) - 2))
            desc[v] = f"[{edges[i]:.4g},{edges[i + 1]:.4g})"
        else:
            desc[v] = p
    return desc
