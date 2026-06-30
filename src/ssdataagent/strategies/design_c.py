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
