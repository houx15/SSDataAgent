from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ssdataagent.data.schema import DatasetSchema, load_schema
from ssdataagent.strategies.base import InfoGate, StrategyResult


def classify_columns(schema: DatasetSchema, columns) -> tuple[list[str], list[str]]:
    """Split columns into (numerical, categorical). Numerical iff in
    schema.numeric_ranges; everything else is treated as categorical/ordinal."""
    numerical, categorical = [], []
    for c in columns:
        (numerical if c in schema.numeric_ranges else categorical).append(c)
    return numerical, categorical


def encode_numeric(df, columns, schema, *, stats=None):
    """Float feature matrix: numerical z-scored, categorical one-hot over
    schema.allowed_values. `stats` (means/sds fitted on train) is reused for
    eval rows so the scaling origin matches. Returns (X, stats)."""
    num, cat = classify_columns(schema, columns)
    if stats is None:
        means = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in num}
        sds = {c: (float(pd.to_numeric(df[c], errors="coerce").std(ddof=0)) or 1.0) for c in num}
        stats = {"means": means, "sds": sds}
    blocks = []
    for c in num:
        col = pd.to_numeric(df[c], errors="coerce").fillna(stats["means"][c]).to_numpy()
        blocks.append(((col - stats["means"][c]) / stats["sds"][c]).reshape(-1, 1))
    for c in cat:
        cats = schema.allowed_values.get(c) or sorted(df[c].dropna().unique().tolist())
        idx = {v: i for i, v in enumerate(cats)}
        oh = np.zeros((len(df), len(cats)), dtype=float)
        for r, v in enumerate(df[c].tolist()):
            if v in idx:
                oh[r, idx[v]] = 1.0
        blocks.append(oh)
    X = np.hstack(blocks) if blocks else np.zeros((len(df), 0))
    return X, stats


def ordinal_encode(df, columns, schema) -> np.ndarray:
    """Integer-code columns for tree models: numerical kept as float;
    categorical mapped to its index in allowed_values (unknown -> -1)."""
    cols = []
    for c in columns:
        if c in schema.numeric_ranges:
            cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1))
        else:
            cats = schema.allowed_values.get(c) or sorted(df[c].dropna().unique().tolist())
            idx = {v: i for i, v in enumerate(cats)}
            cols.append(np.array([idx.get(v, -1) for v in df[c].tolist()], dtype=float).reshape(-1, 1))
    return np.hstack(cols) if cols else np.zeros((len(df), 0))


def clip_decode(df, schema) -> pd.DataFrame:
    """Clip numerical columns to schema.numeric_ranges; leave categoricals."""
    out = df.copy()
    for c in out.columns:
        if c in schema.numeric_ranges:
            lo, hi = schema.numeric_ranges[c]
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lo, hi)
    return out


def background_frame(background, schema) -> pd.DataFrame:
    """Background columns (+ profile_id), target columns dropped (no leakage)."""
    cols = [c for c in background.columns
            if c in schema.background_variables or c == "profile_id"]
    out = background[cols].reset_index(drop=True).copy()
    if "profile_id" not in out.columns:
        out.insert(0, "profile_id", range(len(out)))
    return out


def hotdeck_generate(train, background, schema, *, k=10, seed=42) -> pd.DataFrame:
    """Generate targets for background rows via k-NN hot-deck imputation.

    For each background row, find its k nearest neighbors in the training set
    (by background variables), randomly pick one, and donate its targets.
    Returns a DataFrame with background variables + all targets, clipped to range.
    """
    bg_vars = list(schema.background_variables)
    targets = list(schema.target_variables)
    Xtr, stats = encode_numeric(train, bg_vars, schema)
    Xev, _ = encode_numeric(background, bg_vars, schema, stats=stats)
    k_eff = max(1, min(k, len(train)))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(Xtr)
    _, idx = nn.kneighbors(Xev)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k_eff, size=len(Xev))
    chosen = idx[np.arange(len(idx)), pick]
    donor = train.iloc[chosen][targets].reset_index(drop=True)
    out = background_frame(background, schema)
    for c in targets:
        out[c] = donor[c].to_numpy()
    return clip_decode(out, schema)


class HotDeckStrategy:
    """Hot-deck / k-NN baseline: donor imputation by nearest background neighbors."""

    name = "hotdeck"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        """Generate synthetic data using hot-deck imputation.

        Raises ValueError if no microdata is available (fit_microdata() is None).
        """
        train = gate.fit_microdata()
        if train is None:
            raise ValueError("hotdeck requires microdata; this condition exposes none")
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        generated = hotdeck_generate(train, bg, schema, k=10, seed=42)
        Path(run_dir, "fit_summary.json").write_text(
            json.dumps({"backend": "hotdeck", "k": 10, "n_train_fit": len(train)}, indent=2)
        )
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "hotdeck", "k": 10,
                         "n_train_fit": len(train), "n_individuals": len(bg)},
        )
