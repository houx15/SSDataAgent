from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LogisticRegression

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.baselines import background_frame, clip_decode, encode_numeric
from ssdataagent.strategies.base import InfoGate, StrategyResult

_SEED = 42
_N_NUMERIC_BINS = 10
_PROMPT_VERSION = "designa-v1"
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_SYSTEM = (
    "You are a survey-data modeler. Given background variables and target variables, "
    "propose a predictive structure: an order over the targets and, for each target, "
    "which variables predict it. Return ONLY a JSON object."
)


@dataclass
class Structure:
    order: list
    parents: dict
    prior_scale: dict
    offsets: dict


def _validate_structure(obj, targets, backgrounds, transport) -> "Structure":
    tset = set(targets)
    order = [t for t in (obj.get("order") or []) if t in tset]
    for t in targets:
        if t not in order:
            order.append(t)
    legal_bg = set(backgrounds)
    parents, seen = {}, []
    for t in order:
        allowed = legal_bg | set(seen)
        raw = (obj.get("parents") or {}).get(t, []) or []
        parents[t] = [p for p in raw if p in allowed]
        seen.append(t)
    prior_scale = {}
    for t in order:
        try:
            v = float((obj.get("prior_scale") or {}).get(t, 1.0))
        except (TypeError, ValueError):
            v = 1.0
        prior_scale[t] = v if v > 0 else 1.0
    offsets = {}
    for t in order:
        try:
            offsets[t] = float((obj.get("offsets") or {}).get(t, 0.0)) if transport else 0.0
        except (TypeError, ValueError):
            offsets[t] = 0.0
    return Structure(order=order, parents=parents, prior_scale=prior_scale, offsets=offsets)


def _default_structure(targets, backgrounds, transport) -> "Structure":
    return Structure(order=list(targets),
                     parents={t: list(backgrounds) for t in targets},
                     prior_scale={t: 1.0 for t in targets},
                     offsets={t: 0.0 for t in targets})


def _build_structure_prompt(schema, targets, backgrounds, transport) -> str:
    lines = [
        f"Population: {schema.population_context}",
        f"Background variables (always observed, may be parents): {list(backgrounds)}",
        "Target variables to model:",
    ]
    for t in targets:
        desc = schema.descriptions.get(t, "")
        lines.append(f"- {t}{(': ' + desc) if desc else ''}")
    lines += [
        "",
        "Propose: (1) `order` — a topological order over the targets; (2) `parents` — "
        "for each target, the variables that predict it, chosen ONLY from the background "
        "variables and targets that appear EARLIER in your order; (3) `prior_scale` — a "
        "number per target, >1 for attitude/opinion targets with wide natural spread; "
        "(4) `offsets` — per numeric target, a population shift (use 0 unless adapting "
        "across populations).",
        'Respond with ONLY JSON: {"order": [...], "parents": {"t": [...]}, '
        '"prior_scale": {"t": 1.0}, "offsets": {"t": 0.0}}',
    ]
    if transport:
        lines.append("NOTE: you are adapting from a different source population; set "
                     "numeric `offsets` to the expected target-population shift.")
    return "\n".join(lines)


def _cache_key(dataset, condition, model, targets) -> str:
    blob = json.dumps([dataset, condition, model, sorted(targets), _PROMPT_VERSION],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def elicit_structure(client, *, dataset, condition, schema, targets, backgrounds,
                     run_dir, cache_dir, transport=False) -> "Structure":
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(run_dir) / "structure"
    log_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(getattr(client, "cfg", None), "model", "unknown")
    cache_file = cache_dir / f"{_cache_key(dataset, condition, model, targets)}.json"
    if cache_file.exists():
        try:
            d = json.loads(cache_file.read_text())
            return Structure(order=d["order"], parents=d["parents"],
                             prior_scale=d["prior_scale"], offsets=d["offsets"])
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt cache -> re-elicit
    prompt = _build_structure_prompt(schema, targets, backgrounds, transport)
    raw = ""
    try:
        raw = client.chat(messages=[{"role": "user", "content": prompt}], system=_SYSTEM)
        m = _JSON_OBJ.search(raw or "")
        obj = json.loads(m.group(0) if m else "null")
        struct = _validate_structure(obj, targets, backgrounds, transport)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        struct = _default_structure(targets, backgrounds, transport)
    (log_dir / "structure.prompt.txt").write_text(prompt)
    (log_dir / "structure.response.txt").write_text(raw or "")
    cache_file.write_text(json.dumps({"order": struct.order, "parents": struct.parents,
                                      "prior_scale": struct.prior_scale,
                                      "offsets": struct.offsets}))
    return struct
