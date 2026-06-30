# S1 distribution diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the S1 distribution diagnostic arm — three variants (`s1_raw`, `s1_raked`, `s1_personas`) that emit per-cell conditional distributions and sample targets independently, spanning conditions A/B/C.

**Architecture:** One new module `src/ssdataagent/strategies/s1.py`. A shared `_S1Base` (raw & raked) reuses Design B's machinery — `elicitation.elicit_cell_distributions`, `cells.*`, `design_b.rake`, `design_b.sample_targets` with an **identity Σ** (independent sampling) — differing only by a `rake` flag. The net-new piece is the mixture-of-personas variant (`elicit_cell_personas` + `sample_personas`).

**Tech Stack:** Python 3.11+, numpy, pandas; the existing `elicitation` / `cells` / `design_b` / `baselines` / `base` modules. No new dependency.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-30-s1-distribution-design.md` is authoritative.
- **Independent sampling (identity Σ)** for all variants — S1's defining property; it is *meant* to fail Types 2/3/5.
- **raw vs raked share the same anchored elicitation** (and cache); the only difference is the rake step. The isolated variable is raking ("grounding").
- **One `s1_raked` strategy spans A/B/C** — the gate supplies the raking anchor (train/source/withheld via `known_marginals()`); no per-condition code.
- **Personas unraked in v1.** K=3 personas per cell (`_N_PERSONAS`).
- **Determinism:** no microdata fit; persistent elicitation/personas caches + seeded sampling (`seed=42`); tests mock the client; a cache hit means zero client calls. Renormalize any probability vector before `rng.choice`.
- **Leakage:** raked/personas read only `known_marginals` via the gate — under TRANSFER that is source-only; never the target survey's targets.
- **Reuse, don't reimplement:** `design_b.rake`, `design_b.sample_targets`; `elicitation.target_support`/`known_vector`/`elicit_cell_distributions` and its helpers `_normalize_to_support`/`_describe_support`/`_JSON_OBJ`; `cells.fit_scheme`/`assign`/`describe_cell`; `baselines.background_frame`/`clip_decode`.
- **Defaults:** numeric support K=10 bins; cells `n_bins=4`; seed 42; K=3 personas.
- **Gate (per `feedback_refactor_gate_philosophy`):** our tests pass + no NEW failures vs. the 4 pre-existing `autograd`-missing failures (`tests/test_config.py::test_unknown_provider_raises` + 3 `tests/test_ssdatabench_integration.py *_legacy`). No bit-for-bit gate.
- **Git hygiene:** stage explicit paths (never `git add -A`); avoid the literal word "eval" in commit messages; do not stage the `ssdatabench` submodule pointer. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: `s1.py` — `_S1Base` + raw & raked strategies

**Files:**
- Create: `src/ssdataagent/strategies/s1.py`
- Test: `tests/test_s1_raw_raked.py`

**Interfaces:**
- Consumes: `cells.*`, `design_b.rake`/`sample_targets`, `elicitation.*`, `baselines.background_frame`/`clip_decode`, `InfoGate`/`StrategyResult`.
- Produces: `_prepare(gate) -> dict`; `class _S1Base` (`generate`); `class S1RawStrategy` (`name="s1_raw"`, `rake=False`); `class S1RakedStrategy` (`name="s1_raked"`, `rake=True`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_s1_raw_raked.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.s1 import S1RawStrategy, S1RakedStrategy


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


class _SkewClient:
    """Returns a fixed, NON-degenerate skewed prob vector per target (mass on every
    bin, decaying). Skewed away from the train marginal so raking visibly moves it."""
    def __init__(self, supports):
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()
        self._lens = {t: (len(s["support"]) if s["kind"] == "cat" else len(s["edges"]) - 1)
                      for t, s in supports.items()}

    def chat(self, messages, system=None):
        self.calls += 1
        obj = {}
        for t, L in self._lens.items():
            v = np.array([0.5 ** i for i in range(L)], float)
            obj[t] = (v / v.sum()).tolist()
        return json.dumps(obj)


def _gate(condition, schema, tmp_path, *, client, train=None, source=None, crosswalk=(), n_eval=300):
    if train is None:
        train = _frame(schema, 400, 0)
    bg = _frame(schema, n_eval, 1)
    return InfoGate(condition=condition, dataset_name="gss", workspace=tmp_path,
                    client=client, train=train, eval_rows=bg,
                    source=source, source_name="gss1994" if source is not None else None,
                    crosswalk=crosswalk)


def _supports(schema, targets):
    return {t: E.target_support(schema, t, n_numeric_bins=10) for t in targets}


def test_raw_generates_all_targets(tmp_path):
    schema = load_schema("gss")
    targets = list(schema.target_variables)
    g = _gate(Condition.FULL, schema, tmp_path, client=_SkewClient(_supports(schema, targets)))
    res = S1RawStrategy().generate(g, tmp_path, _cfg(tmp_path))
    for t in targets:
        assert t in res.generated.columns
    fs = json.loads(Path(tmp_path, "fit_summary.json").read_text())
    assert fs["backend"] == "s1" and fs["variant"] == "raw" and fs["raked"] is False


def test_raked_marginal_closer_than_raw(tmp_path):
    schema = load_schema("gss")
    # pick a categorical target with a stable known marginal
    t = next(c for c in schema.target_variables if c not in schema.numeric_ranges)
    sup = E.target_support(schema, t, n_numeric_bins=10)
    order = sup["support"]
    train = _frame(schema, 600, 0)
    client = _SkewClient(_supports(schema, list(schema.target_variables)))
    known = E.known_vector({"probs": pd.Series(train[t]).value_counts(normalize=True).to_dict()}, sup)

    def marg(df):
        vc = pd.Series(df[t]).value_counts(normalize=True)
        return np.array([float(vc.get(v, 0.0)) for v in order])

    g_raw = _gate(Condition.FULL, schema, tmp_path / "r", client=client, train=train)
    g_rake = _gate(Condition.FULL, schema, tmp_path / "k", client=client, train=train)
    (tmp_path / "r").mkdir(); (tmp_path / "k").mkdir()
    raw = S1RawStrategy().generate(g_raw, tmp_path / "r", _cfg(tmp_path))
    rake = S1RakedStrategy().generate(g_rake, tmp_path / "k", _cfg(tmp_path))
    d_raw = np.abs(marg(raw.generated) - known).sum()
    d_rake = np.abs(marg(rake.generated) - known).sum()
    assert d_rake < d_raw


def test_raw_deterministic_and_cache(tmp_path):
    schema = load_schema("gss")
    client = _SkewClient(_supports(schema, list(schema.target_variables)))
    g = _gate(Condition.FULL, schema, tmp_path, client=client)
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    r1 = S1RawStrategy().generate(g, tmp_path / "a", _cfg(tmp_path))
    calls_after_first = client.calls
    r2 = S1RawStrategy().generate(g, tmp_path / "b", _cfg(tmp_path))
    assert client.calls == calls_after_first      # cache hit -> zero new calls
    pd.testing.assert_frame_equal(r1.generated, r2.generated)


def test_raked_transfer_no_leakage(tmp_path):
    schema = load_schema("gss")
    num_t = next(t for t in schema.target_variables if t in schema.numeric_ranges)
    lo, hi = schema.numeric_ranges[num_t]
    train = _frame(schema, 300, 0); train[num_t] = hi - 0.01      # target survey: high
    source = _frame(schema, 300, 2); source[num_t] = lo + 0.01    # source: low
    crosswalk = tuple(list(schema.background_variables) + list(schema.target_variables))
    client = _SkewClient(_supports(schema, list(schema.target_variables)))
    g = _gate(Condition.TRANSFER, schema, tmp_path, client=client, train=train,
              source=source, crosswalk=crosswalk)
    res = S1RakedStrategy().generate(g, tmp_path, _cfg(tmp_path))
    assert res.generated[num_t].mean() < (lo + hi) / 2            # tracks source, not target train
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_s1_raw_raked.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_s1_raw_raked.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/s1.py tests/test_s1_raw_raked.py
git commit -m "s1: _S1Base + raw/raked distribution strategies"
```

---

### Task 2: `s1.py` — mixture-of-personas elicitation

**Files:**
- Modify: `src/ssdataagent/strategies/s1.py`
- Test: `tests/test_s1_personas_elicit.py`

**Interfaces:**
- Consumes: `elicitation._normalize_to_support`, `_describe_support`, `_JSON_OBJ` (existing helpers); `schema.population_context`/`descriptions`.
- Produces:
  - `_validate_personas(obj, targets, supports, known_vectors, n_personas) -> list[dict]`
  - `elicit_cell_personas(client, *, dataset, condition, cell_descs, schema, targets, supports, known_vectors, run_dir, cache_dir, n_personas=3, max_retries=3) -> dict[str, list[dict]]` — each cell → list of `{"weight": float, "dists": {t: np.ndarray}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_s1_personas_elicit.py
import json

import numpy as np

from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.s1 import _validate_personas, elicit_cell_personas


def _sup(schema, t):
    return E.target_support(schema, t, n_numeric_bins=10)


def test_validate_normalizes_weights_and_dists():
    schema = load_schema("gss")
    t = next(c for c in schema.target_variables if c not in schema.numeric_ranges)
    sup = _sup(schema, t)
    L = len(sup["support"])
    kv = {t: np.full(L, 1.0 / L)}
    obj = {"subtypes": [
        {"weight": 3.0, "dists": {t: [1.0] + [0.0] * (L - 1)}},
        {"weight": 1.0, "dists": {t: [0.0] * (L - 1) + [2.0]}},   # unnormalized dist
    ]}
    subs = _validate_personas(obj, [t], {t: sup}, kv, n_personas=3)
    assert len(subs) == 2
    assert abs(sum(s["weight"] for s in subs) - 1.0) < 1e-9
    assert subs[0]["weight"] == 0.75 and subs[1]["weight"] == 0.25
    for s in subs:
        assert abs(s["dists"][t].sum() - 1.0) < 1e-9


def test_validate_fallback_to_single_known_subtype():
    schema = load_schema("gss")
    t = next(c for c in schema.target_variables if c not in schema.numeric_ranges)
    sup = _sup(schema, t)
    L = len(sup["support"])
    kv = {t: np.full(L, 1.0 / L)}
    subs = _validate_personas({"garbage": 1}, [t], {t: sup}, kv, n_personas=3)
    assert len(subs) == 1 and subs[0]["weight"] == 1.0
    assert np.allclose(subs[0]["dists"][t], kv[t])


class _PersonaClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()

    def chat(self, messages, system=None):
        self.calls += 1
        return self.payload


def test_elicit_personas_caches(tmp_path):
    schema = load_schema("gss")
    t = next(c for c in schema.target_variables if c not in schema.numeric_ranges)
    sup = _sup(schema, t)
    L = len(sup["support"])
    kv = {t: np.full(L, 1.0 / L)}
    payload = json.dumps({"subtypes": [{"weight": 1.0, "dists": {t: [1.0 / L] * L}}]})
    c = _PersonaClient(payload)
    kw = dict(dataset="gss", condition="no_data", cell_descs={"c0": {"x": "y"}},
              schema=schema, targets=[t], supports={t: sup}, known_vectors=kv,
              run_dir=tmp_path, cache_dir=tmp_path / "cache", n_personas=3)
    r1 = elicit_cell_personas(c, **kw)
    assert c.calls == 1 and "c0" in r1 and len(r1["c0"]) >= 1
    elicit_cell_personas(c, **kw)
    assert c.calls == 1            # cache hit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_s1_personas_elicit.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `s1.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_s1_personas_elicit.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/s1.py tests/test_s1_personas_elicit.py
git commit -m "s1: mixture-of-personas elicitation"
```

---

### Task 3: `s1.py` — persona sampling + `S1PersonasStrategy`

**Files:**
- Modify: `src/ssdataagent/strategies/s1.py`
- Test: `tests/test_s1_personas_sample.py`

**Interfaces:**
- Consumes: `_prepare`, `elicit_cell_personas`, `_empty_result`; `background_frame`/`clip_decode`.
- Produces:
  - `sample_personas(eval_cell_keys, cell_personas, supports, targets, *, seed=42) -> dict[str, list]`
  - `class S1PersonasStrategy(_S1Base)` (`name="s1_personas"`, `variant="personas"`, own `generate`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_s1_personas_sample.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.s1 import sample_personas, S1PersonasStrategy


def test_sample_personas_respects_weights_and_support():
    # one categorical target, two subtypes: subtype 0 -> always "a", subtype 1 -> always "b"
    sup = {"kind": "cat", "support": ["a", "b"]}
    cell_personas = {"c0": [
        {"weight": 0.8, "dists": {"t": np.array([1.0, 0.0])}},
        {"weight": 0.2, "dists": {"t": np.array([0.0, 1.0])}},
    ]}
    keys = ["c0"] * 4000
    out = sample_personas(keys, cell_personas, {"t": sup}, ["t"], seed=0)
    vals = np.array(out["t"], dtype=object)
    assert set(np.unique(vals)).issubset({"a", "b"})
    assert abs((vals == "a").mean() - 0.8) < 0.03            # weight 0.8 on the "a" subtype


def test_sample_personas_deterministic():
    sup = {"kind": "num", "edges": np.linspace(0.0, 10.0, 11)}
    cp = {"c0": [{"weight": 1.0, "dists": {"t": np.full(10, 0.1)}}]}
    a = sample_personas(["c0", "c0"], cp, {"t": sup}, ["t"], seed=5)
    b = sample_personas(["c0", "c0"], cp, {"t": sup}, ["t"], seed=5)
    assert np.array_equal(np.array(a["t"]), np.array(b["t"]))


class _PersonaClient:
    def __init__(self, supports):
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()
        self._lens = {t: (len(s["support"]) if s["kind"] == "cat" else len(s["edges"]) - 1)
                      for t, s in supports.items()}

    def chat(self, messages, system=None):
        self.calls += 1
        dists = {t: [1.0 / L] * L for t, L in self._lens.items()}
        return json.dumps({"subtypes": [{"weight": 0.6, "dists": dists},
                                        {"weight": 0.4, "dists": dists}]})


def _frame(schema, n, seed):
    rng = np.random.default_rng(seed)
    data = {}
    for c in list(schema.background_variables) + list(schema.target_variables):
        if c in schema.numeric_ranges:
            lo, hi = schema.numeric_ranges[c]
            data[c] = rng.uniform(lo, hi, n)
        else:
            data[c] = rng.choice(schema.allowed_values.get(c) or ["a", "b"], n)
    return pd.DataFrame(data)


def test_personas_strategy_end_to_end(tmp_path):
    schema = load_schema("gss")
    targets = list(schema.target_variables)
    supports = {t: E.target_support(schema, t, n_numeric_bins=10) for t in targets}
    g = InfoGate(condition=Condition.NO_DATA, dataset_name="gss", workspace=tmp_path,
                 client=_PersonaClient(supports), train=_frame(schema, 300, 0),
                 eval_rows=_frame(schema, 40, 1))
    res = S1PersonasStrategy().generate(g, tmp_path, type("Cfg", (), {"results_root": tmp_path})())
    for t in targets:
        assert t in res.generated.columns
    assert len(res.generated) == 40
    fs = json.loads(Path(tmp_path, "fit_summary.json").read_text())
    assert fs["variant"] == "personas" and fs["n_personas"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_s1_personas_sample.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `s1.py`)

```python
def sample_personas(eval_cell_keys, cell_personas, supports, targets, *, seed=42) -> dict[str, list]:
    """Per eval row: pick a subtype ~ its cell's weights, then sample each target
    independently from that subtype's distribution. Categorical -> support member;
    numeric -> uniform within the chosen even-width bin."""
    rng = np.random.default_rng(seed)
    n = len(eval_cell_keys)
    cell_w, cell_cum = {}, {}
    for c, subs in cell_personas.items():
        w = np.array([s["weight"] for s in subs], float)
        s = w.sum()
        cell_w[c] = w / s if s > 0 else np.full(len(subs), 1.0 / len(subs))
        cell_cum[c] = [{t: np.cumsum(np.asarray(sub["dists"][t], float)) for t in targets}
                       for sub in subs]
    out: dict[str, list] = {t: [None] * n for t in targets}
    for i in range(n):
        c = eval_cell_keys[i]
        w, cums = cell_w[c], cell_cum[c]
        j = int(rng.choice(len(w), p=w))
        for t in targets:
            cum = cums[j][t]
            idx = int(np.searchsorted(cum, rng.random(), side="left"))
            idx = min(max(idx, 0), len(cum) - 1)
            sup = supports[t]
            if sup["kind"] == "cat":
                out[t][i] = sup["support"][idx]
            else:
                lo, hi = float(sup["edges"][idx]), float(sup["edges"][idx + 1])
                out[t][i] = float(lo + rng.random() * (hi - lo))
    return out


class S1PersonasStrategy(_S1Base):
    name = "s1_personas"
    variant = "personas"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        p = _prepare(gate)
        if not p["targets"]:
            return _empty_result(p, self.variant)
        cell_personas = elicit_cell_personas(
            gate.client, dataset=gate.dataset_name, condition=gate.condition.value,
            cell_descs=p["cell_descs"], schema=p["schema"], targets=p["targets"],
            supports=p["supports"], known_vectors=p["known_vecs"], run_dir=run_dir,
            cache_dir=Path(getattr(cfg, "results_root", run_dir)) / "_elicitation_cache",
            n_personas=_N_PERSONAS,
        )
        drawn = sample_personas(p["eval_cell_keys"], cell_personas, p["supports"],
                                p["targets"], seed=_SEED)
        out = background_frame(p["bg"], p["schema"])
        for t in p["targets"]:
            out[t] = drawn[t]
        generated = clip_decode(out, p["schema"])
        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "s1", "variant": "personas", "condition": gate.condition.value,
             "raked": False, "n_cells": len(p["unique_cells"]),
             "n_targets": len(p["targets"]), "n_personas": _N_PERSONAS}, indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "s1", "variant": "personas",
                         "condition": gate.condition.value, "raked": False,
                         "n_cells": len(p["unique_cells"]), "n_targets": len(p["targets"]),
                         "n_personas": _N_PERSONAS, "n_individuals": len(p["bg"])})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_s1_personas_sample.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/s1.py tests/test_s1_personas_sample.py
git commit -m "s1: persona sampling + S1PersonasStrategy"
```

---

### Task 4: Register strategies + 5 condition specs + runner characterization

**Files:**
- Modify: `src/ssdataagent/strategies/registry.py`
- Modify: `src/ssdataagent/experiments/conditions.py`
- Test: `tests/test_strategies_registry.py` (extend), `tests/test_conditions.py` (extend), `tests/test_runner_artifacts.py` (extend)

**Interfaces:**
- Consumes: `S1RawStrategy`, `S1RakedStrategy`, `S1PersonasStrategy`; `Condition.FULL/TRANSFER/NO_DATA`.
- Produces: registry keys `s1_raw`, `s1_raked`, `s1_personas`; conditions `s1_raw`, `s1_raked_full`, `s1_raked_transfer`, `s1_raked_aggregate`, `s1_personas`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_strategies_registry.py  (add)
def test_s1_strategies_registered():
    from ssdataagent.strategies.registry import get_strategy
    assert get_strategy("s1_raw").name == "s1_raw"
    assert get_strategy("s1_raked").name == "s1_raked"
    assert get_strategy("s1_personas").name == "s1_personas"
```

```python
# tests/test_conditions.py  (add)
def test_s1_conditions():
    from ssdataagent.agent.context import Condition
    from ssdataagent.experiments.conditions import get_condition
    assert get_condition("s1_raw").context_condition is Condition.NO_DATA
    assert get_condition("s1_raw").strategy == "s1_raw"
    assert get_condition("s1_raked_full").context_condition is Condition.FULL
    assert get_condition("s1_raked_full").strategy == "s1_raked"
    assert get_condition("s1_raked_transfer").context_condition is Condition.TRANSFER
    assert get_condition("s1_raked_transfer").strategy == "s1_raked"
    assert get_condition("s1_raked_aggregate").context_condition is Condition.NO_DATA
    assert get_condition("s1_raked_aggregate").strategy == "s1_raked"
    assert get_condition("s1_personas").context_condition is Condition.NO_DATA
    assert get_condition("s1_personas").strategy == "s1_personas"
```

For `tests/test_runner_artifacts.py`: add a transfer-gate characterization test mirroring the existing design_b/design_c/design_a ones. Define a module-level `_fake_s1_raked_generate(self, gate, run_dir, cfg)` that asserts `gate.source is not None and len(gate.crosswalk) > 0` then returns a trivial `StrategyResult(generated=..., meta_extras={"backend": "s1"})`; a test `test_s1_raked_transfer_builds_source_gate` patched with `@patch("ssdataagent.strategies.s1.S1RakedStrategy.generate", _fake_s1_raked_generate)`, running an experiment with `conditions=["s1_raked_transfer"]`, dataset `["gss"]`, exp name `s1exp`, run-dir segment `s1_raked_transfer`, asserting the produced meta `backend == "s1"`. Reuse the SAME decorator/fixture stack the existing transfer tests use. Do NOT modify the two byte-stable tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py -q`
Expected: FAIL — `KeyError: unknown strategy 's1_raw'` / `unknown condition 's1_raw'`.

- [ ] **Step 3: Write minimal implementation**

In `registry.py`: import the three classes and add entries after `"design_a"`:

```python
from ssdataagent.strategies.s1 import S1PersonasStrategy, S1RakedStrategy, S1RawStrategy
# ...
    "design_a": DesignAStrategy,
    "s1_raw": S1RawStrategy,
    "s1_raked": S1RakedStrategy,
    "s1_personas": S1PersonasStrategy,
```

In `conditions.py`, add to `CONDITIONS`:

```python
    "s1_raw": ConditionSpec("s1_raw", Condition.NO_DATA, strategy="s1_raw"),
    "s1_raked_full": ConditionSpec("s1_raked_full", Condition.FULL, strategy="s1_raked"),
    "s1_raked_transfer": ConditionSpec("s1_raked_transfer", Condition.TRANSFER, strategy="s1_raked"),
    "s1_raked_aggregate": ConditionSpec("s1_raked_aggregate", Condition.NO_DATA, strategy="s1_raked"),
    "s1_personas": ConditionSpec("s1_personas", Condition.NO_DATA, strategy="s1_personas"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py tests/test_runner_artifacts.py -q`
Expected: PASS (including the new characterization test and the unchanged byte-stable tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/registry.py src/ssdataagent/experiments/conditions.py tests/test_strategies_registry.py tests/test_conditions.py tests/test_runner_artifacts.py
git commit -m "s1: register 3 variants + 5 condition specs"
```

---

## Self-Review (plan author)

- **Spec coverage:** §3 strategies/conditions → Tasks 1+3+4; §4 raw/raked generate → Task 1; §5 personas → Tasks 2+3; §6 determinism/leakage/artifacts → asserted across Tasks 1-3 (determinism, cache, transfer-no-leakage in Task 1; persona determinism in Task 3); §8 testing → one file per task. All covered.
- **Placeholder scan:** none — every step ships complete code. The Task 4 runner test references the existing transfer-characterization pattern by location (the implementer copies an in-repo test), consistent with Parts 3b/4/5.
- **Type consistency:** `_prepare -> dict` consumed by `_S1Base.generate` and `S1PersonasStrategy.generate`; `elicit_cell_distributions -> {cell:{t:vec}}` and `design_b.sample_targets` consumed in Task 1; `elicit_cell_personas -> {cell:[{weight,dists}]}` consumed by `sample_personas` in Task 3; identity `Sigma = np.eye(len(targets))` matches `sample_targets`'s `Sigma` arg. `_validate_personas`/`_empty_result` defined and reused consistently.
- **Diagnostic integrity:** raw and raked share `elicit_cell_distributions` (only the rake step differs); the raked-closer test uses a non-degenerate skewed mock (an empty/degenerate mock would make raking a no-op).
- **Leakage:** raked transfer reads `known_marginals` (= source under TRANSFER); Task 1's `test_raked_transfer_no_leakage` proves output tracks source, not the poisoned target train.
