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


import hashlib

_PERSONAS_PROMPT_VERSION = "s1personas-v1"
_PERSONAS_SYSTEM = (
    "You are a survey-distribution estimator. For a demographic subgroup, enumerate a "
    "few latent SUBTYPES (distinct attitude/behavior clusters) with population weights, "
    "and for each subtype give each target's distribution. Return ONLY a JSON object."
)


def _personas_cache_key(dataset, condition, model, cell_key, targets, n_personas) -> str:
    blob = json.dumps([dataset, condition, model, cell_key, sorted(targets), n_personas,
                       _PERSONAS_PROMPT_VERSION], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _build_personas_prompt(*, schema, cell_desc, targets, supports, known_vectors, n_personas) -> str:
    lines = [
        f"Population: {schema.population_context}",
        f"Demographic subgroup: {json.dumps(cell_desc, default=str)}",
        "",
        f"Enumerate up to {n_personas} latent SUBTYPES within this subgroup. Give each a "
        "population weight (weights sum to ~1) and, for each target, that subtype's "
        "probability vector over its support:",
    ]
    for t in targets:
        desc = schema.descriptions.get(t, "")
        lines.append(f"- {t}{(': ' + desc) if desc else ''} — {E._describe_support(supports[t])}.")
    lines += ["",
              'Respond with ONLY JSON: {"subtypes": [{"weight": 0.5, "dists": '
              '{"<target>": [p1, p2, ...]}}, ...]}']
    return "\n".join(lines)


def _validate_personas(obj, targets, supports, known_vectors, n_personas) -> list[dict]:
    out: list[dict] = []
    subs = obj.get("subtypes") if isinstance(obj, dict) else None
    if isinstance(subs, list):
        for s in subs[:n_personas]:
            if not isinstance(s, dict):
                continue
            dists_raw = s.get("dists") or {}
            dists, ok = {}, True
            for t in targets:
                v = E._normalize_to_support(dists_raw.get(t), supports[t])
                if v is None:
                    ok = False
                    break
                dists[t] = v
            if not ok:
                continue
            try:
                w = float(s.get("weight", 1.0))
            except (TypeError, ValueError):
                w = 1.0
            out.append({"weight": max(w, 0.0), "dists": dists})
    if not out:
        out = [{"weight": 1.0, "dists": {t: np.array(known_vectors[t], float) for t in targets}}]
    total = sum(s["weight"] for s in out) or 1.0
    for s in out:
        s["weight"] = s["weight"] / total
    return out


def elicit_cell_personas(client, *, dataset, condition, cell_descs, schema, targets,
                         supports, known_vectors, run_dir, cache_dir, n_personas=3,
                         max_retries=3) -> dict[str, list[dict]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(run_dir) / "elicitation"
    log_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(getattr(client, "cfg", None), "model", "unknown")
    result: dict[str, list[dict]] = {}
    for cell_key, cell_desc in cell_descs.items():
        key = _personas_cache_key(dataset, condition, model, cell_key, targets, n_personas)
        cache_file = cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                result[cell_key] = [{"weight": float(s["weight"]),
                                     "dists": {t: np.array(s["dists"][t], float) for t in targets}}
                                    for s in cached]
                continue
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass  # corrupt cache -> re-elicit
        prompt = _build_personas_prompt(schema=schema, cell_desc=cell_desc, targets=targets,
                                        supports=supports, known_vectors=known_vectors,
                                        n_personas=n_personas)
        subs, raw = None, ""
        for _ in range(max_retries + 1):
            raw = client.chat(messages=[{"role": "user", "content": prompt}], system=_PERSONAS_SYSTEM)
            m = E._JSON_OBJ.search(raw or "")
            obj = None
            if m:
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    obj = None
            if obj is not None:
                subs = _validate_personas(obj, targets, supports, known_vectors, n_personas)
                break
        if subs is None:
            subs = _validate_personas(None, targets, supports, known_vectors, n_personas)
        (log_dir / f"{cell_key.replace('|', '_')}.personas.prompt.txt").write_text(prompt)
        (log_dir / f"{cell_key.replace('|', '_')}.personas.response.txt").write_text(raw or "")
        cache_file.write_text(json.dumps(
            [{"weight": s["weight"], "dists": {t: s["dists"][t].tolist() for t in targets}}
             for s in subs]))
        result[cell_key] = subs
    return result
