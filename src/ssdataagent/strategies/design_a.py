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


def _design_matrix(df, parents, schema, stats=None):
    """Parent design matrix via encode_numeric; an intercept-only ones column
    when there are no parents (so sklearn always has >=1 feature)."""
    if not parents:
        return np.ones((len(df), 1)), (stats or {})
    X, st = encode_numeric(df, list(parents), schema, stats=stats)
    if X.shape[1] == 0:
        return np.ones((len(df), 1)), st
    return X, st


def fit_numeric_node(X_train, y_train, prior_scale):
    """Conjugate Bayesian linear regression. Wider prior_scale -> weaker
    coefficient shrinkage (smaller lambda_init)."""
    lam = 1.0 / max(float(prior_scale), 1e-6)
    try:
        model = BayesianRidge(lambda_init=lam)
    except TypeError:               # older sklearn without lambda_init
        model = BayesianRidge()
    model.fit(X_train, np.asarray(y_train, float))
    return model


def fit_categorical_node(X_train, y_train, prior_scale, classes):
    """Multinomial logistic MAP (C = prior_scale = inverse Gaussian-prior
    precision). Single-class training -> a constant model."""
    y = np.asarray(y_train, dtype=object).astype(str)
    uniq = list(dict.fromkeys(y.tolist()))
    if len(uniq) < 2:
        return ("constant", uniq[0] if uniq else (classes[0] if classes else None))
    # sklearn>=1.7 removed the `multi_class` kwarg; multinomial is the default for
    # multiclass. C = inverse Gaussian-prior precision (larger prior_scale = wider).
    model = LogisticRegression(C=max(float(prior_scale), 1e-6), max_iter=1000)
    model.fit(X_train, y)
    return model


def sample_node(model, X_eval, support, *, offset, rng):
    """Numeric: draw Normal(mu+offset, sd) from BayesianRidge predictive.
    Categorical: draw a class ~ predict_proba (offset not applied)."""
    n = len(X_eval)
    if support["kind"] == "num":
        mu, sd = model.predict(X_eval, return_std=True)
        sd = np.where(np.asarray(sd) > 0, sd, 1e-6)
        return rng.normal(np.asarray(mu) + float(offset), sd)
    if isinstance(model, tuple) and model[0] == "constant":
        return np.array([model[1]] * n, dtype=object)
    P = model.predict_proba(X_eval)
    classes = list(model.classes_)
    out = []
    for i in range(n):
        p = np.asarray(P[i], float)
        s = p.sum()
        p = p / s if s > 0 else np.full(len(classes), 1.0 / len(classes))
        out.append(classes[int(rng.choice(len(classes), p=p))])
    return np.array(out, dtype=object)


def sample_from_known(support, known_vec, n, rng):
    """Condition-C draw: sample a support index from the known marginal, then a
    category (categorical) or a uniform value within the chosen bin (numeric)."""
    vec = np.asarray(known_vec, float)
    s = vec.sum()
    vec = vec / s if s > 0 else np.full(len(vec), 1.0 / len(vec))
    idx = rng.choice(len(vec), size=n, p=vec)
    if support["kind"] == "cat":
        return np.array([support["support"][i] for i in idx], dtype=object)
    edges = np.asarray(support["edges"], float)
    lo, hi = edges[idx], edges[idx + 1]
    return lo + rng.random(n) * (hi - lo)


class DesignAStrategy:
    name = "design_a"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        train = gate.fit_microdata()           # train (A) / source[crosswalk] (B) / None (C)
        known_m = gate.known_marginals() or {}

        targets = [t for t in schema.target_variables if t in known_m]
        if not targets:
            return StrategyResult(generated=background_frame(bg, schema),
                                  meta_extras={"backend": "design_a", "n_targets": 0,
                                               "n_individuals": len(bg)})

        supports = {t: E.target_support(schema, t, n_numeric_bins=_N_NUMERIC_BINS) for t in targets}
        backgrounds = list(schema.background_variables)
        struct = elicit_structure(
            gate.client, dataset=gate.dataset_name, condition=gate.condition.value,
            schema=schema, targets=targets, backgrounds=backgrounds, run_dir=run_dir,
            cache_dir=Path(getattr(cfg, "results_root", run_dir)) / "_structure_cache",
            transport=(gate.condition is Condition.TRANSFER),
        )

        rng = np.random.default_rng(_SEED)
        sampled = background_frame(bg, schema)
        node_types: dict[str, str] = {}
        is_transfer = gate.condition is Condition.TRANSFER
        for t in struct.order:
            if t not in supports:
                continue
            sup = supports[t]
            is_num = sup["kind"] == "num"
            node_types[t] = "numeric" if is_num else "categorical"
            if train is None:                                   # condition C
                vals = sample_from_known(sup, E.known_vector(known_m.get(t), sup), len(bg), rng)
            else:
                parents = [p for p in struct.parents.get(t, [])
                           if p in train.columns and p in sampled.columns]
                X_train, stats = _design_matrix(train, parents, schema)
                X_eval, _ = _design_matrix(sampled, parents, schema, stats=stats)
                offset = float(struct.offsets.get(t, 0.0)) if (is_transfer and is_num) else 0.0
                if is_num:
                    y = pd.to_numeric(train[t], errors="coerce").fillna(0.0).to_numpy()
                    model = fit_numeric_node(X_train, y, struct.prior_scale.get(t, 1.0))
                else:
                    model = fit_categorical_node(X_train, train[t].astype(str).to_numpy(),
                                                 struct.prior_scale.get(t, 1.0), sup["support"])
                vals = sample_node(model, X_eval, sup, offset=offset, rng=rng)
            sampled[t] = vals
        generated = clip_decode(sampled, schema)

        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "design_a", "condition": gate.condition.value,
             "order": struct.order, "parents": struct.parents, "node_types": node_types,
             "n_train_fit": (0 if train is None else len(train)),
             "calibrated": train is None, "transport": is_transfer}, indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "design_a", "condition": gate.condition.value,
                         "n_targets": len(targets), "calibrated": train is None,
                         "transport": is_transfer, "n_individuals": len(bg)},
        )
