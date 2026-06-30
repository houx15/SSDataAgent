# src/ssdataagent/strategies/s1.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data import cells
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import design_b, elicitation as E
from ssdataagent.strategies.baselines import background_frame, clip_decode
from ssdataagent.strategies.base import InfoGate, StrategyResult

_N_DEMO_BINS = 4
_N_NUMERIC_BINS = 10
_N_PERSONAS = 3
_SEED = 42


def _prepare(gate) -> dict:
    """Shared setup for every S1 variant: schema, background, target set, supports,
    known vectors, and the demographic-cell partition over the eval rows."""
    schema = load_schema(gate.dataset_name)
    bg = gate.background()
    known_m = gate.known_marginals() or {}
    targets = [t for t in schema.target_variables if t in known_m]
    supports = {t: E.target_support(schema, t, n_numeric_bins=_N_NUMERIC_BINS) for t in targets}
    known_vecs = {t: E.known_vector(known_m.get(t), supports[t]) for t in targets}
    if targets:
        scheme = cells.fit_scheme(bg, schema.background_variables, schema, n_bins=_N_DEMO_BINS)
        eval_cell_keys = cells.assign(bg, scheme).tolist()
        unique_cells = sorted(set(eval_cell_keys))
        counts = pd.Series(eval_cell_keys).value_counts()
        cell_weights = {c: float(counts[c]) for c in unique_cells}
        cell_descs = {c: cells.describe_cell(scheme, c) for c in unique_cells}
    else:
        eval_cell_keys, unique_cells, cell_weights, cell_descs = [], [], {}, {}
    return dict(schema=schema, bg=bg, known_m=known_m, targets=targets, supports=supports,
                known_vecs=known_vecs, eval_cell_keys=eval_cell_keys, unique_cells=unique_cells,
                cell_weights=cell_weights, cell_descs=cell_descs)


def _empty_result(p, variant):
    return StrategyResult(generated=background_frame(p["bg"], p["schema"]),
                          meta_extras={"backend": "s1", "variant": variant,
                                       "n_targets": 0, "n_individuals": len(p["bg"])})


class _S1Base:
    name = "s1"
    rake = False
    variant = "raw"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        p = _prepare(gate)
        if not p["targets"]:
            return _empty_result(p, self.variant)
        cell_dists = E.elicit_cell_distributions(
            gate.client, dataset=gate.dataset_name, condition=gate.condition.value,
            cell_descs=p["cell_descs"], schema=p["schema"], targets=p["targets"],
            supports=p["supports"], known_vectors=p["known_vecs"], run_dir=run_dir,
            cache_dir=Path(getattr(cfg, "results_root", run_dir)) / "_elicitation_cache",
            transport=(gate.condition is Condition.TRANSFER),
        )
        calibrated = {c: {} for c in p["unique_cells"]}
        for t in p["targets"]:
            cell_vectors_t = {c: cell_dists[c][t] for c in p["unique_cells"]}
            if self.rake:
                raked = design_b.rake(cell_vectors_t, p["cell_weights"], p["known_vecs"][t])
                for c in p["unique_cells"]:
                    calibrated[c][t] = raked[c]
            else:
                for c in p["unique_cells"]:
                    calibrated[c][t] = cell_vectors_t[c]
        Sigma = np.eye(len(p["targets"]))
        drawn = design_b.sample_targets(p["eval_cell_keys"], calibrated, p["supports"],
                                        Sigma, p["targets"], seed=_SEED)
        out = background_frame(p["bg"], p["schema"])
        for t in p["targets"]:
            out[t] = drawn[t]
        generated = clip_decode(out, p["schema"])
        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "s1", "variant": self.variant, "condition": gate.condition.value,
             "raked": self.rake, "n_cells": len(p["unique_cells"]),
             "n_targets": len(p["targets"])}, indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "s1", "variant": self.variant,
                         "condition": gate.condition.value, "raked": self.rake,
                         "n_cells": len(p["unique_cells"]), "n_targets": len(p["targets"]),
                         "n_individuals": len(p["bg"])})


class S1RawStrategy(_S1Base):
    name = "s1_raw"
    rake = False
    variant = "raw"


class S1RakedStrategy(_S1Base):
    name = "s1_raked"
    rake = True
    variant = "raked"
