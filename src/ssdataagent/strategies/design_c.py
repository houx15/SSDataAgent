from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.baselines import background_frame, clip_decode, encode_numeric
from ssdataagent.strategies.base import InfoGate, StrategyResult

_K = 10
_N_NUMERIC_BINS = 10
_REPAIR_ITERS = 50


def retrieve_candidates(donors, background, schema, *, k: int = 10) -> np.ndarray:
    """k-NN donor row-indices per eval row, matched on the background variables
    present in BOTH frames (the crosswalk subset under TRANSFER). encode_numeric
    fits scaling on donors and reuses it for eval rows. Returns (n_eval, k_eff)."""
    bg_vars = [c for c in schema.background_variables
               if c in donors.columns and c in background.columns]
    Xtr, stats = encode_numeric(donors, bg_vars, schema)
    Xev, _ = encode_numeric(background, bg_vars, schema, stats=stats)
    k_eff = max(1, min(k, len(donors)))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(Xtr)
    _, idx = nn.kneighbors(Xev)
    return idx


def encode_to_codes(donors, targets, supports) -> dict[str, np.ndarray]:
    """Map each donor's target value to its support index. Categorical -> index
    in the support list (unknown -> 0); numeric -> bin via searchsorted on
    interior edges, clamped to a valid bin."""
    codes: dict[str, np.ndarray] = {}
    for t in targets:
        sup = supports[t]
        if sup["kind"] == "cat":
            order = {v: i for i, v in enumerate(sup["support"])}
            codes[t] = np.array([order.get(v, 0) for v in donors[t].tolist()], dtype=int)
        else:
            edges = np.asarray(sup["edges"], float)
            vals = pd.to_numeric(donors[t], errors="coerce").fillna(float(edges[0])).to_numpy()
            idx = np.searchsorted(edges[1:-1], vals, side="right")
            codes[t] = np.clip(idx, 0, len(edges) - 2).astype(int)
    return codes


def repair_weights(neighbor_idx, donor_codes, goal_vectors, supports, targets,
                   *, max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Global non-negative donor weights, raked so each eval row's
    candidate-weighted target marginal matches goal_vectors[t]. Each IPF pass,
    for every target/bin, scale the weight of donors coded to that bin by
    q_t(bin)/m_t(bin); renormalize for scale stability. Approximate IPF (per-row
    normalization couples the bins), so iterate. Returns (n_donor,) weights."""
    if not targets:
        return np.ones(0, dtype=float)
    n_donor = len(donor_codes[targets[0]])
    w = np.ones(n_donor, dtype=float)
    if neighbor_idx.size == 0 or n_donor == 0:
        return w
    n_eval = neighbor_idx.shape[0]
    for _ in range(max_iter):
        max_gap = 0.0
        for t in targets:
            q = np.asarray(goal_vectors[t], float)
            n_bins = len(q)
            codes_t = donor_codes[t]
            cand_codes = codes_t[neighbor_idx]
            cand_w = w[neighbor_idx]
            row_sum = cand_w.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum > 0, row_sum, 1.0)
            sel = cand_w / row_sum
            m = np.bincount(cand_codes.ravel(), weights=sel.ravel(),
                            minlength=n_bins)[:n_bins] / n_eval
            max_gap = max(max_gap, float(np.max(np.abs(m - q))))
            ratio = np.divide(q, m, out=np.ones_like(q), where=m > 0)
            w = w * ratio[codes_t]
        s = w.sum()
        if s > 0:
            w = w * (n_donor / s)
        if max_gap < tol:
            break
    return w
