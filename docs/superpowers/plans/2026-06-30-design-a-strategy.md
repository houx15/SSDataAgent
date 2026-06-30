# Design A strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `design_a` strategy — a structure-first, LLM-elicited hierarchical generator — covering information conditions A/B/C, with no MCMC dependency.

**Architecture:** One new module `src/ssdataagent/strategies/design_a.py`. The LLM proposes a constrained DAG over the target variables (backgrounds are exogenous parents; LLM gives a topological order + each target's parents from {backgrounds, earlier targets}) plus per-node prior scales and condition-B numeric offsets. Each target node is a per-node Bayesian GLM (MAP/conjugate): numeric → `sklearn.BayesianRidge` (predictive mean+std), categorical → `sklearn.LogisticRegression` (multinomial). We walk the order and sample each target from its full conditional given test backgrounds + already-sampled parents. Condition C (no microdata) samples each target from its known marginal.

**Tech Stack:** Python 3.11+, numpy, pandas, scikit-learn (`BayesianRidge`, `LogisticRegression`), the existing `elicitation`/`baselines`/`base` strategy modules. No new dependency.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-30-design-a-strategy-design.md` is authoritative.
- **Backend is in-house:** per-node penalized GLMs = MAP under Gaussian priors. NO PyMC/NumPyro/JAX.
- **DAG is constrained:** backgrounds exogenous; LLM gives only `order` (over targets) + `parents[t]` ⊆ {backgrounds} ∪ {targets earlier in order}. Acyclic by construction. Validation prunes illegal parents and completes a partial order.
- **Offsets are numeric-only and condition-B-only** (categorical nodes ignore offsets in v1).
- **Intercept calibration / known-marginal sampling runs only in C.** A/B rely on the data fit.
- **Sampling draws the full conditional** (BayesianRidge predictive Normal; full `predict_proba` multinomial), never the conditional mean — this is the over-determination fix.
- **Determinism:** all fits are deterministic (MAP/conjugate); only predictive sampling uses a seeded RNG (`seed=42`). Structure elicitation reproducible via the persistent cache; tests mock the client. Always renormalize a probability vector before `rng.choice` (float drift).
- **Leakage:** condition B fits only on `source[crosswalk]` (via `gate.fit_microdata()`), reads `known_marginals` from source — never the target survey's targets.
- **Condition C must NOT raise** (`fit_microdata()` is None): sample each target from `known_marginals`.
- **Defaults:** prior scale 1.0, seed 42, numeric support K=10 bins.
- **Reuse, don't reimplement:** `baselines.encode_numeric` / `background_frame` / `clip_decode`; `elicitation.target_support` / `known_vector`; `InfoGate` / `StrategyResult`.
- **Gate (per `feedback_refactor_gate_philosophy`):** our tests pass + no NEW failures vs. the 4 pre-existing `autograd`-missing failures (`tests/test_config.py::test_unknown_provider_raises` + 3 `tests/test_ssdatabench_integration.py *_legacy`). No bit-for-bit gate.
- **Git hygiene:** stage explicit paths (never `git add -A`); avoid the literal word "eval" in commit messages; do not stage the `ssdatabench` submodule pointer. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: `design_a.py` — structure elicitation

**Files:**
- Create: `src/ssdataagent/strategies/design_a.py`
- Test: `tests/test_design_a_structure.py`

**Interfaces:**
- Consumes: `client.chat(messages, system)`, `schema.descriptions`, `schema.population_context`.
- Produces:
  - `@dataclass Structure(order: list, parents: dict, prior_scale: dict, offsets: dict)`
  - `elicit_structure(client, *, dataset, condition, schema, targets, backgrounds, run_dir, cache_dir, transport=False) -> Structure` — prompts once, parses strict JSON, validates/prunes, persistent cache + raw-I/O log; never raises on bad LLM output.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_design_a_structure.py
import json
from pathlib import Path

from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.design_a import Structure, elicit_structure, _validate_structure


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()

    def chat(self, messages, system=None):
        self.calls += 1
        return self.payload


def test_validate_prunes_illegal_parents():
    targets = ["t1", "t2"]
    backgrounds = ["age", "educ"]
    obj = {"order": ["t1", "t2"],
           "parents": {"t1": ["age", "t2", "bogus"], "t2": ["t1", "educ"]},
           "prior_scale": {"t1": 2.0}, "offsets": {}}
    s = _validate_structure(obj, targets, backgrounds, transport=False)
    # t1 cannot have t2 (later) or bogus (unknown) as parent; t2 may have t1 (earlier)
    assert s.parents["t1"] == ["age"]
    assert s.parents["t2"] == ["t1", "educ"]
    assert s.prior_scale["t1"] == 2.0
    assert s.prior_scale["t2"] == 1.0          # default
    assert s.offsets["t1"] == 0.0              # transport=False zeros offsets


def test_validate_completes_partial_order():
    s = _validate_structure({"order": ["t2"]}, ["t1", "t2"], ["age"], transport=False)
    assert set(s.order) == {"t1", "t2"} and len(s.order) == 2


def test_elicit_fallback_on_bad_json(tmp_path):
    schema = load_schema("gss")
    targets = list(schema.target_variables)[:2]
    bgs = list(schema.background_variables)
    c = _Client("not json at all")
    s = elicit_structure(c, dataset="gss", condition="full", schema=schema, targets=targets,
                         backgrounds=bgs, run_dir=tmp_path, cache_dir=tmp_path / "cache")
    assert set(s.order) == set(targets)
    assert all(s.parents[t] == bgs for t in targets)   # default = all backgrounds


def test_elicit_caches(tmp_path):
    schema = load_schema("gss")
    targets = list(schema.target_variables)[:2]
    bgs = list(schema.background_variables)
    payload = json.dumps({"order": targets, "parents": {t: bgs[:1] for t in targets},
                          "prior_scale": {}, "offsets": {}})
    c = _Client(payload)
    kw = dict(dataset="gss", condition="full", schema=schema, targets=targets,
              backgrounds=bgs, run_dir=tmp_path, cache_dir=tmp_path / "cache")
    elicit_structure(c, **kw)
    assert c.calls == 1
    elicit_structure(c, **kw)            # cache hit
    assert c.calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_design_a_structure.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/strategies/design_a.py
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
        obj = json.loads(m.group(0)) if m else {}
        struct = _validate_structure(obj, targets, backgrounds, transport)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        struct = _default_structure(targets, backgrounds, transport)
    (log_dir / "structure.prompt.txt").write_text(prompt)
    (log_dir / "structure.response.txt").write_text(raw or "")
    cache_file.write_text(json.dumps({"order": struct.order, "parents": struct.parents,
                                      "prior_scale": struct.prior_scale,
                                      "offsets": struct.offsets}))
    return struct
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_design_a_structure.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_a.py tests/test_design_a_structure.py
git commit -m "design_a: LLM structure elicitation (constrained DAG)"
```

---

### Task 2: `design_a.py` — per-node GLM fit + sample

**Files:**
- Modify: `src/ssdataagent/strategies/design_a.py`
- Test: `tests/test_design_a_nodes.py`

**Interfaces:**
- Consumes: `baselines.encode_numeric`.
- Produces:
  - `_design_matrix(df, parents, schema, stats=None) -> (np.ndarray, dict)` — parent design matrix; an intercept-only `ones` column when there are no parents.
  - `fit_numeric_node(X_train, y_train, prior_scale) -> BayesianRidge`
  - `fit_categorical_node(X_train, y_train, prior_scale, classes) -> object` (a fitted `LogisticRegression`, or `("constant", value)` when training labels are single-class)
  - `sample_node(model, X_eval, support, *, offset, rng) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_design_a_nodes.py
import numpy as np

from ssdataagent.strategies.design_a import (
    _design_matrix, fit_numeric_node, fit_categorical_node, sample_node,
)


class _Schema:
    numeric_ranges = {"x": (0.0, 10.0)}
    allowed_values = {"c": ["a", "b"]}


def test_design_matrix_empty_parents_is_ones():
    import pandas as pd
    X, stats = _design_matrix(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), [], _Schema())
    assert X.shape == (3, 1) and np.allclose(X, 1.0)


def test_numeric_node_samples_with_spread():
    rng = np.random.default_rng(0)
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    y = 2.0 * X[:, 0] + rng.normal(0, 0.5, 50)
    m = fit_numeric_node(X, y, prior_scale=1.0)
    sup = {"kind": "num", "edges": np.linspace(0, 10, 11)}
    vals = sample_node(m, X[:5], sup, offset=0.0, rng=np.random.default_rng(1))
    assert vals.shape == (5,)
    # two draws from the same fitted model differ (full-conditional sampling, not the mean)
    a = sample_node(m, X[:5], sup, offset=0.0, rng=np.random.default_rng(1))
    b = sample_node(m, X[:5], sup, offset=0.0, rng=np.random.default_rng(2))
    assert not np.allclose(a, b)


def test_numeric_offset_shifts_mean():
    rng = np.random.default_rng(0)
    X = np.zeros((100, 1))
    y = rng.normal(5.0, 0.1, 100)
    m = fit_numeric_node(X, y, prior_scale=1.0)
    sup = {"kind": "num", "edges": np.linspace(0, 100, 11)}
    base = sample_node(m, np.zeros((100, 1)), sup, offset=0.0, rng=np.random.default_rng(3))
    shifted = sample_node(m, np.zeros((100, 1)), sup, offset=20.0, rng=np.random.default_rng(3))
    assert shifted.mean() - base.mean() > 15.0


def test_categorical_node_valid_classes_and_constant_fallback():
    rng = np.random.default_rng(0)
    X = np.random.default_rng(0).normal(size=(60, 2))
    y = np.array(["a", "b"] * 30)
    m = fit_categorical_node(X, y, prior_scale=1.0, classes=["a", "b"])
    sup = {"kind": "cat", "support": ["a", "b"]}
    vals = sample_node(m, X[:10], sup, offset=0.0, rng=rng)
    assert set(np.unique(vals)).issubset({"a", "b"})
    # single-class training -> constant model
    mc = fit_categorical_node(X, np.array(["a"] * 60), prior_scale=1.0, classes=["a", "b"])
    cv = sample_node(mc, X[:4], sup, offset=0.0, rng=rng)
    assert list(cv) == ["a", "a", "a", "a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_design_a_nodes.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `design_a.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_design_a_nodes.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_a.py tests/test_design_a_nodes.py
git commit -m "design_a: per-node Bayesian GLM fit + full-conditional sampling"
```

---

### Task 3: `design_a.py` — condition-C known-marginal sampling

**Files:**
- Modify: `src/ssdataagent/strategies/design_a.py`
- Test: `tests/test_design_a_known.py`

**Interfaces:**
- Consumes: `elicitation.known_vector` / `target_support` (the support dict shape).
- Produces: `sample_from_known(support, known_vec, n, rng) -> np.ndarray` — condition-C draw: a support index ∝ `known_vec`, then category (categorical) or uniform-within-bin (numeric). This is the zero-slope, intercept-calibrated conditional from the spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_design_a_known.py
import numpy as np

from ssdataagent.strategies.design_a import sample_from_known


def test_known_categorical_matches_marginal():
    sup = {"kind": "cat", "support": ["a", "b"]}
    vals = sample_from_known(sup, np.array([0.8, 0.2]), 5000, np.random.default_rng(0))
    assert abs((vals == "a").mean() - 0.8) < 0.03


def test_known_numeric_within_range_and_deterministic():
    sup = {"kind": "num", "edges": np.linspace(0.0, 10.0, 11)}
    vec = np.full(10, 0.1)
    a = sample_from_known(sup, vec, 200, np.random.default_rng(7))
    b = sample_from_known(sup, vec, 200, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() <= 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_design_a_known.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `design_a.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_design_a_known.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_a.py tests/test_design_a_known.py
git commit -m "design_a: condition-C known-marginal sampling"
```

---

### Task 4: `DesignAStrategy.generate` — A/B/C orchestration

**Files:**
- Modify: `src/ssdataagent/strategies/design_a.py`
- Test: `tests/test_strategy_design_a.py`

**Interfaces:**
- Consumes: `InfoGate` (`background`, `fit_microdata`, `known_marginals`, `condition`, `client`, `dataset_name`); all Task 1-3 functions; `background_frame`, `clip_decode`, `E.target_support`, `E.known_vector`.
- Produces: `class DesignAStrategy` with `name = "design_a"` and `generate(self, gate, run_dir, cfg) -> StrategyResult`.

Notes for the implementer:
- Target set = `[t for t in schema.target_variables if t in known_marginals]`. Empty → early return `background_frame(bg, schema)` with `meta_extras={"backend":"design_a","n_targets":0,"n_individuals":len(bg)}`.
- `train = gate.fit_microdata()` is train (A) / source[crosswalk] (B) / None (C).
- Walk `struct.order`; maintain a `sampled` frame (starts as `background_frame`) and add each sampled target column as you go, so later nodes can use earlier targets as parents. In A/B the design matrix for a node is built from `train[parents]` (fit) and `sampled[parents]` (predict) with shared `stats`. Restrict each node's parents to columns actually present in `train` (crosswalk safety in B).
- Offsets apply only when `gate.condition is Condition.TRANSFER` AND the target is numeric.
- C path: `sample_from_known(support, E.known_vector(known_m.get(t), support), len(bg), rng)`.
- `calibrated = train is None`; write `fit_summary.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_design_a.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.design_a import DesignAStrategy


class _FakeClient:
    """Empty JSON -> elicit_structure falls back to the default structure."""
    def __init__(self):
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()

    def chat(self, messages, system=None):
        self.calls += 1
        return "{}"


def _frame(schema, n, seed):
    rng = np.random.default_rng(seed)
    data = {}
    for c in list(schema.background_variables) + list(schema.target_variables):
        if c in schema.numeric_ranges:
            lo, hi = schema.numeric_ranges[c]
            data[c] = rng.uniform(lo, hi, n)
        else:
            cats = schema.allowed_values.get(c) or ["a", "b"]
            data[c] = rng.choice(cats, n)
    return pd.DataFrame(data)


def _cfg(tmp_path):
    return type("Cfg", (), {"results_root": tmp_path})()


def _gate(condition, schema, tmp_path, *, train=None, source=None, crosswalk=()):
    if train is None:
        train = _frame(schema, 200, 0)
    bg = _frame(schema, 30, 1)
    return InfoGate(condition=condition, dataset_name="gss", workspace=tmp_path,
                    client=_FakeClient(), train=train, eval_rows=bg,
                    source=source, source_name="gss1994" if source is not None else None,
                    crosswalk=crosswalk)


def test_full_generates_all_targets(tmp_path):
    schema = load_schema("gss")
    g = _gate(Condition.FULL, schema, tmp_path)
    res = DesignAStrategy().generate(g, tmp_path, _cfg(tmp_path))
    for t in schema.target_variables:
        assert t in res.generated.columns
    assert len(res.generated) == 30
    fs = json.loads(Path(tmp_path, "fit_summary.json").read_text())
    assert fs["backend"] == "design_a" and fs["transport"] is False


def test_aggregate_does_not_raise(tmp_path):
    schema = load_schema("gss")
    g = _gate(Condition.NO_DATA, schema, tmp_path)
    res = DesignAStrategy().generate(g, tmp_path, _cfg(tmp_path))
    assert len(res.generated) == 30
    for t in schema.target_variables:
        assert t in res.generated.columns


def test_full_is_deterministic(tmp_path):
    schema = load_schema("gss")
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    g1 = _gate(Condition.FULL, schema, tmp_path, train=_frame(schema, 200, 0))
    g2 = _gate(Condition.FULL, schema, tmp_path, train=_frame(schema, 200, 0))
    # identical eval rows: rebuild the same bg seed by reusing the same gate construction
    r1 = DesignAStrategy().generate(g1, tmp_path / "a", _cfg(tmp_path))
    r2 = DesignAStrategy().generate(g2, tmp_path / "b", _cfg(tmp_path))
    pd.testing.assert_frame_equal(r1.generated, r2.generated)


def test_transfer_end_to_end_no_leakage(tmp_path):
    """B fits on source, not target train. Poison the target-survey target with a
    high mean; source has a low mean. Generated must track SOURCE, proving no leak."""
    schema = load_schema("gss")
    num_t = next(t for t in schema.target_variables if t in schema.numeric_ranges)
    lo, hi = schema.numeric_ranges[num_t]
    train = _frame(schema, 200, 0); train[num_t] = hi - 0.01      # target survey: high
    source = _frame(schema, 200, 2); source[num_t] = lo + 0.01    # source: low
    crosswalk = tuple(list(schema.background_variables) + list(schema.target_variables))
    g = _gate(Condition.TRANSFER, schema, tmp_path, train=train, source=source, crosswalk=crosswalk)
    res = DesignAStrategy().generate(g, tmp_path, _cfg(tmp_path))
    fs = json.loads(Path(tmp_path, "fit_summary.json").read_text())
    assert fs["transport"] is True
    mid = (lo + hi) / 2
    assert res.generated[num_t].mean() < mid     # tracks source (low), not target train (high)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategy_design_a.py -q`
Expected: FAIL — `ImportError: cannot import name 'DesignAStrategy'`.

- [ ] **Step 3: Write minimal implementation** (append to `design_a.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategy_design_a.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_a.py tests/test_strategy_design_a.py
git commit -m "design_a: DesignAStrategy A/B/C orchestration"
```

---

### Task 5: Register strategy + A/B/C conditions + runner characterization

**Files:**
- Modify: `src/ssdataagent/strategies/registry.py`
- Modify: `src/ssdataagent/experiments/conditions.py`
- Test: `tests/test_strategies_registry.py` (extend), `tests/test_conditions.py` (extend), `tests/test_runner_artifacts.py` (extend)

**Interfaces:**
- Consumes: `DesignAStrategy`; `Condition.FULL/TRANSFER/NO_DATA`.
- Produces: registry key `"design_a"`; conditions `design_a_full`, `design_a_transfer`, `design_a_aggregate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_strategies_registry.py  (add)
def test_design_a_registered():
    from ssdataagent.strategies.registry import get_strategy
    assert get_strategy("design_a").name == "design_a"
```

```python
# tests/test_conditions.py  (add)
def test_design_a_conditions():
    from ssdataagent.agent.context import Condition
    from ssdataagent.experiments.conditions import get_condition
    assert get_condition("design_a_full").context_condition is Condition.FULL
    assert get_condition("design_a_full").strategy == "design_a"
    assert get_condition("design_a_transfer").context_condition is Condition.TRANSFER
    assert get_condition("design_a_transfer").strategy == "design_a"
    assert get_condition("design_a_aggregate").context_condition is Condition.NO_DATA
    assert get_condition("design_a_aggregate").strategy == "design_a"
```

For `tests/test_runner_artifacts.py`: add a transfer-gate characterization test mirroring the existing Design B/C ones (`_fake_design_b_generate` / `test_transfer_condition_builds_source_gate` around line 131, and the Design C analogue added later in the file). Copy the structure: a module-level `_fake_design_a_generate(self, gate, run_dir, cfg)` that asserts `gate.source is not None and len(gate.crosswalk) > 0` then returns a trivial `StrategyResult(generated=..., meta_extras={"backend": "design_a"})`; a test `test_design_a_transfer_builds_source_gate` patched with `@patch("ssdataagent.strategies.design_a.DesignAStrategy.generate", _fake_design_a_generate)`, running an experiment with `conditions=["design_a_transfer"]`, dataset `["gss"]`, exp name `daexp`, run-dir segment `design_a_transfer`, asserting the produced meta `backend == "design_a"`. Reuse the SAME decorator/fixture stack the Design B/C tests use. Do NOT modify the two byte-stable tests (`test_agent_artifacts_are_stable`, `test_direct_artifacts_are_stable`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py -q`
Expected: FAIL — `KeyError: unknown strategy 'design_a'` / `unknown condition 'design_a_full'`.

- [ ] **Step 3: Write minimal implementation**

In `registry.py`: import `DesignAStrategy` and add `"design_a": DesignAStrategy` after the `"design_c"` entry.

```python
from ssdataagent.strategies.design_a import DesignAStrategy
# ...
    "design_c": DesignCStrategy,
    "design_a": DesignAStrategy,
```

In `conditions.py`, add to `CONDITIONS`:

```python
    "design_a_full": ConditionSpec("design_a_full", Condition.FULL, strategy="design_a"),
    "design_a_aggregate": ConditionSpec("design_a_aggregate", Condition.NO_DATA, strategy="design_a"),
    "design_a_transfer": ConditionSpec("design_a_transfer", Condition.TRANSFER, strategy="design_a"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py tests/test_runner_artifacts.py -q`
Expected: PASS (including the new characterization test and the unchanged byte-stable tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/registry.py src/ssdataagent/experiments/conditions.py tests/test_strategies_registry.py tests/test_conditions.py tests/test_runner_artifacts.py
git commit -m "design_a: register strategy + A/B/C condition specs"
```

---

## Self-Review (plan author)

- **Spec coverage:** §3 structure elicitation → Task 1; §4 per-node model → Task 2; §5 C path → Task 3 + Task 4 wiring; §5 A/B orchestration → Task 4; §7 registration/runner → Task 5; §8 determinism/leakage/artifacts → asserted across Tasks 1-5 (determinism + transfer-no-leakage in Task 4, structure cache in Task 1). All covered.
- **Placeholder scan:** none — every step ships complete code. The Task 4 `_FakeClient` returns `"{}"` (explicit default-structure fallback). The Task 5 runner test references the existing Design B/C pattern by location rather than re-pasting it (the implementer copies an in-repo test).
- **Type consistency:** `Structure(order,parents,prior_scale,offsets)` produced in Task 1, consumed in Task 4; `_design_matrix -> (ndarray, dict)`, `fit_numeric_node -> BayesianRidge`, `fit_categorical_node -> model|("constant",v)`, `sample_node -> ndarray`, `sample_from_known -> ndarray` — all consumed consistently in Task 4. Offsets numeric+TRANSFER-only in both spec and Task 4.
- **Leakage:** Task 4 fits on `gate.fit_microdata()` (source under TRANSFER) and the end-to-end test proves generated values track source, not poisoned target-train targets.
- **Determinism risk:** `rng.choice` is fed a renormalized probability vector in both `sample_node` and `sample_from_known` (guards float drift); all model fits are deterministic.
