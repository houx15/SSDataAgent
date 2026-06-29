from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import ordinal_encode


def _bin_edges(real_vals, n_bins):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(pd.to_numeric(real_vals, errors="coerce").dropna(), qs))
    if len(edges) < 2:
        edges = np.array([edges.min() - 1e-9, edges.max() + 1e-9]) if len(edges) else np.array([0.0, 1.0])
    return edges


def _discretize(values, edges):
    return np.clip(np.digitize(pd.to_numeric(values, errors="coerce").to_numpy(), edges[1:-1]), 0, len(edges) - 2)


def _coarsen(real, sim, schema, n_demo_bins):
    """Return (real_cells, sim_cells): a string cell key per row."""
    real_keys, sim_keys = [], []
    parts_real, parts_sim = [], []
    for v in schema.background_variables:
        if v in schema.numeric_ranges:
            edges = _bin_edges(real[v], n_demo_bins)
            parts_real.append(_discretize(real[v], edges).astype(str))
            parts_sim.append(_discretize(sim[v], edges).astype(str))
        else:
            parts_real.append(real[v].astype(str).to_numpy())
            parts_sim.append(sim[v].astype(str).to_numpy())
    real_keys = ["|".join(t) for t in zip(*parts_real)] if parts_real else ["_"] * len(real)
    sim_keys = ["|".join(t) for t in zip(*parts_sim)] if parts_sim else ["_"] * len(sim)
    return np.array(real_keys), np.array(sim_keys)


def _target_series(df, t, schema, edges_map):
    if t in schema.numeric_ranges:
        return pd.Series(_discretize(df[t], edges_map[t]), index=df.index)
    return df[t].astype(str)


def _entropy_bits(labels) -> float:
    counts = pd.Series(labels).value_counts().to_numpy().astype(float)
    if counts.sum() == 0:
        return 0.0
    return float(entropy(counts, base=2))


def _cell_based(real, sim, schema, n_target_bins, n_demo_bins, min_count):
    edges_map = {t: _bin_edges(real[t], n_target_bins)
                 for t in schema.target_variables if t in schema.numeric_ranges}
    real_cells, sim_cells = _coarsen(real, sim, schema, n_demo_bins)
    real = real.reset_index(drop=True); sim = sim.reset_index(drop=True)
    real_cells = pd.Series(real_cells); sim_cells = pd.Series(sim_cells)
    kept = [c for c, n in real_cells.value_counts().items() if n >= min_count]
    if not kept:
        return {"headline_gap": None, "coverage": 0.0, "n_cells": 0,
                "per_target": {}, "reason": "no cells met min_count"}
    per_target = {}
    for t in schema.target_variables:
        rt = _target_series(real, t, schema, edges_map)
        st = _target_series(sim, t, schema, edges_map)
        num_r = num_s = denom = 0.0
        for c in kept:
            rmask = (real_cells == c).to_numpy()
            smask = (sim_cells == c).to_numpy()
            if smask.sum() == 0:
                continue
            w = float(rmask.sum())
            num_r += w * _entropy_bits(rt[rmask])
            num_s += w * _entropy_bits(st[smask])
            denom += w
        if denom == 0:
            continue
        h_real, h_sim = num_r / denom, num_s / denom
        per_target[t] = {"h_real": h_real, "h_sim": h_sim, "gap": h_real - h_sim}
    coverage = float(real_cells.isin(kept).sum()) / len(real)
    gaps = [v["gap"] for v in per_target.values()]
    headline = float(np.mean(gaps)) if gaps else None
    return {"headline_gap": headline, "coverage": coverage,
            "n_cells": len(kept), "per_target": per_target}


def _model_based(real, sim, schema, n_target_bins, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    edges_map = {t: _bin_edges(real[t], n_target_bins)
                 for t in schema.target_variables if t in schema.numeric_ranges}
    Xr = ordinal_encode(real, schema.background_variables, schema)
    Xs = ordinal_encode(sim, schema.background_variables, schema)
    per_target = {}
    for t in schema.target_variables:
        yr = _target_series(real, t, schema, edges_map).astype(str).to_numpy()
        ys = _target_series(sim, t, schema, edges_map).astype(str).to_numpy()
        if len(np.unique(yr)) < 2:
            continue
        try:
            mr = HistGradientBoostingClassifier(random_state=seed).fit(Xr, yr)
            ms = HistGradientBoostingClassifier(random_state=seed).fit(Xs, ys)
            h_real = float(np.mean([entropy(p, base=2) for p in mr.predict_proba(Xr)]))
            h_sim = float(np.mean([entropy(p, base=2) for p in ms.predict_proba(Xs)]))
        except Exception:
            continue
        per_target[t] = {"h_real": h_real, "h_sim": h_sim, "gap": h_real - h_sim}
    gaps = [v["gap"] for v in per_target.values()]
    return {"headline_gap": float(np.mean(gaps)) if gaps else None,
            "per_target": per_target}


def overdetermination(*, real: pd.DataFrame, sim: pd.DataFrame, schema: DatasetSchema,
                      n_target_bins: int = 5, n_demo_bins: int = 4,
                      min_count: int = 20, seed: int = 42) -> dict:
    """gap = H_real(target | demographics) - H_sim(target | demographics), in bits.
    Positive gap => sim is over-determined (collapsed within-group variance).
    Never raises: a failing stage returns a dict with a 'reason' instead."""
    try:
        cell = _cell_based(real, sim, schema, n_target_bins, n_demo_bins, min_count)
    except Exception as e:
        cell = {"headline_gap": None, "per_target": {}, "reason": f"{type(e).__name__}: {e}"}
    try:
        model = _model_based(real, sim, schema, n_target_bins, seed)
    except Exception as e:
        model = {"headline_gap": None, "per_target": {}, "reason": f"{type(e).__name__}: {e}"}
    return {"cell_based": cell, "model_based": model}
