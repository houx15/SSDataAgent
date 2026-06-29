from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ssdataagent.agent.context import Condition
from ssdataagent.data import cells
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import copula, elicitation as E
from ssdataagent.strategies.baselines import background_frame, clip_decode
from ssdataagent.strategies.base import InfoGate, StrategyResult

_N_DEMO_BINS = 4
_N_NUMERIC_BINS = 10


def rake(cell_vectors, cell_weights, known_vec, *, max_iter: int = 50, tol: float = 1e-6):
    """IPF: scale each cell's prob vector so the cell-weighted mixture matches
    known_vec, preserving relative cross-cell differences. Per-cell vectors
    stay normalized."""
    known = np.asarray(known_vec, float)
    cells = list(cell_vectors)
    P = {c: np.asarray(cell_vectors[c], float).copy() for c in cells}
    total_w = sum(cell_weights[c] for c in cells) or 1.0
    w = {c: cell_weights[c] / total_w for c in cells}
    for _ in range(max_iter):
        mix = sum(w[c] * P[c] for c in cells)
        if np.max(np.abs(mix - known)) < tol:
            break
        ratio = np.divide(known, mix, out=np.ones_like(known), where=mix > 0)
        for c in cells:
            v = P[c] * ratio
            s = v.sum()
            if s > 0:
                P[c] = v / s
    return P


def build_target_copula(ref, targets, schema, *, reg: float = 1e-6) -> np.ndarray:
    """T×T copula correlation over targets: signed empirical correlation from
    `ref` microdata when available (A/B); identity (independence) when ref is
    None (C) or fewer than 2 targets."""
    t = len(targets)
    if ref is None or t < 2 or len(ref) < 2:
        return np.eye(t)
    cuts = copula.build_cuts(ref, list(targets), schema)
    Z = copula.latent_matrix(ref, list(targets), cuts)
    corr = np.corrcoef(Z, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return np.eye(t)
    return copula.make_pd(corr, reg)


def sample_targets(eval_cell_keys, calibrated, supports, Sigma, targets, *, seed: int = 42):
    """Draw a correlated latent per row, map each component through the row's
    cell's calibrated marginal. Categorical -> support member; numeric -> uniform
    within the chosen even-width bin."""
    rng = np.random.default_rng(seed)
    n, t = len(eval_cell_keys), len(targets)
    Z = copula.correlated_normal(Sigma, n, rng) if t else np.zeros((n, 0))
    U = np.clip(norm.cdf(Z), copula.EPS, 1 - copula.EPS)
    cums = {c: {tt: np.cumsum(calibrated[c][tt]) for tt in targets} for c in calibrated}
    out: dict[str, list] = {tt: [None] * n for tt in targets}
    for i in range(n):
        c = eval_cell_keys[i]
        for j, tt in enumerate(targets):
            cum = cums[c][tt]
            idx = int(np.searchsorted(cum, U[i, j], side="left"))
            idx = min(max(idx, 0), len(cum) - 1)
            sup = supports[tt]
            if sup["kind"] == "cat":
                out[tt][i] = sup["support"][idx]
            else:
                lo, hi = float(sup["edges"][idx]), float(sup["edges"][idx + 1])
                out[tt][i] = float(lo + rng.random() * (hi - lo))
    return out


class DesignBStrategy:
    name = "design_b"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        ref = gate.fit_microdata()            # train (A) / source[crosswalk] (B) / None (C)
        known_m = gate.known_marginals() or {}

        # target set: schema targets present in known marginals (= crosswalk targets in B)
        targets = [t for t in schema.target_variables if t in known_m]
        if not targets:
            generated = background_frame(bg, schema)
            return StrategyResult(generated=generated,
                                  meta_extras={"backend": "design_b", "n_targets": 0,
                                               "n_individuals": len(bg)})

        supports = {t: E.target_support(schema, t, n_numeric_bins=_N_NUMERIC_BINS) for t in targets}
        known_vecs = {t: E.known_vector(known_m.get(t), supports[t]) for t in targets}

        # cells from the eval backgrounds
        scheme = cells.fit_scheme(bg, schema.background_variables, schema, n_bins=_N_DEMO_BINS)
        eval_cell_keys = cells.assign(bg, scheme).tolist()
        unique_cells = sorted(set(eval_cell_keys))
        counts = pd.Series(eval_cell_keys).value_counts()
        cell_weights = {c: float(counts[c]) for c in unique_cells}
        cell_descs = {c: dict(zip(scheme.variables, c.split("|"))) for c in unique_cells}

        # elicit per-cell vectors (cached + logged)
        cell_dists = E.elicit_cell_distributions(
            gate.client, dataset=gate.dataset_name, condition=gate.condition.value,
            cell_descs=cell_descs, schema=schema, targets=targets, supports=supports,
            known_vectors=known_vecs, run_dir=run_dir,
            cache_dir=Path(getattr(cfg, "results_root", run_dir)) / "_elicitation_cache",
            transport=(gate.condition is Condition.TRANSFER),
        )

        # rake each target to its known marginal
        calibrated: dict[str, dict[str, "np.ndarray"]] = {c: {} for c in unique_cells}
        for t in targets:
            cell_vectors_t = {c: cell_dists[c][t] for c in unique_cells}
            raked = rake(cell_vectors_t, cell_weights, known_vecs[t])
            for c in unique_cells:
                calibrated[c][t] = raked[c]

        # copula (signed from ref in A/B; identity in C) + sample
        Sigma = build_target_copula(ref, targets, schema)
        drawn = sample_targets(eval_cell_keys, calibrated, supports, Sigma, targets, seed=42)

        out = background_frame(bg, schema)
        for t in targets:
            out[t] = drawn[t]
        generated = clip_decode(out, schema)

        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "design_b", "condition": gate.condition.value,
             "n_cells": len(unique_cells), "n_targets": len(targets),
             "copula": "identity" if (ref is None or len(targets) < 2) else "data"}, indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "design_b", "condition": gate.condition.value,
                         "n_cells": len(unique_cells), "n_targets": len(targets),
                         "n_individuals": len(bg)},
        )
