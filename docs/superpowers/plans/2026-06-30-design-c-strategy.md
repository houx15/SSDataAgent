# Design C strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `design_c` strategy — retrieval-grounded generation with distributional repair — covering information conditions A/B/C.

**Architecture:** One new module `src/ssdataagent/strategies/design_c.py` orchestrating existing parts. For each eval row, retrieve *k* nearest real donors by background, hot-deck candidate target vectors (preserving the real joint), rake donor weights toward a goal marginal (known in A/C, LLM-transported in B), then draw each row's final target from its candidates ∝ weights. The only net-new math is `repair_weights` (IPF over donor weights). Retrieval, encoding, assembly, and LLM elicitation are all reused.

**Tech Stack:** Python 3.11+, numpy, pandas, scipy, scikit-learn (`NearestNeighbors`), the existing `elicitation`/`baselines`/`base` strategy modules.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-30-design-c-strategy-design.md` is authoritative.
- **Defaults (exact values):** `k = 10`, repair `max_iter = 50`, `tol = 1e-6`, `seed = 42`, numeric support `K = 10` even-width bins.
- **Uniform target representation:** prob vector over support — categorical → `schema.allowed_values[t]`; numeric → `K=10` even-width bins over `schema.numeric_ranges[t]`. Reuse `elicitation.target_support` / `elicitation.known_vector`.
- **Goal marginal `q_t` is a single per-target vector in all conditions** — `known_vector(known_marginals[t])` in A/C, the population-level transported vector in B. Design C does NOT partition into demographic cells.
- **LLM only in B**, only on marginals: a single population-level transport elicitation per target via `elicitation.elicit_cell_distributions(..., transport=True)` with one `"__population__"` cell. No LLM in A/C.
- **Every emitted target is a real donor value** (`draw_targets` emits the chosen donor's actual value — no synthetic resampling within bins).
- **Determinism:** `repair_weights` is pure (no RNG); seeded RNG only for the C pool bootstrap and the final per-row draw. LLM transport reproducible via the persistent elicitation cache; tests mock the client.
- **Leakage:** condition B never reads target-survey targets — donors are `source[crosswalk]` and the goal marginal is source-anchored + LLM-transported, both source-only via the gate.
- **Condition C must NOT raise** (`fit_microdata()` is `None`): bootstrap a synthetic donor pool from `known_marginals`.
- **Gate (per `feedback_refactor_gate_philosophy`):** our tests pass + no NEW failures vs. the 4 pre-existing `autograd`-missing failures (`tests/test_config.py::test_unknown_provider_raises` + 3 `tests/test_ssdatabench_integration.py *_legacy`). No bit-for-bit reproduction gate.
- **Git hygiene:** stage explicit paths (never `git add -A`); avoid the literal word "eval" in commit messages (hook blocks it — say "the test"/"evaluation"); do not stage the `ssdatabench` submodule pointer. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: `design_c.py` — retrieval + donor coding

**Files:**
- Create: `src/ssdataagent/strategies/design_c.py`
- Test: `tests/test_design_c_retrieve.py`

**Interfaces:**
- Consumes: `baselines.encode_numeric(df, columns, schema, *, stats=None) -> (X, stats)`; `elicitation.target_support`; `schema.background_variables`, `schema.allowed_values`, `schema.numeric_ranges`.
- Produces:
  - `retrieve_candidates(donors, background, schema, *, k=10) -> np.ndarray` — `(n_eval, k_eff)` int array of donor row-indices, matched on background variables common to both frames.
  - `encode_to_codes(donors, targets, supports) -> dict[str, np.ndarray]` — each donor's support index per target (`(n_donor,)` int arrays).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_design_c_retrieve.py
import numpy as np
import pandas as pd

from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.design_c import retrieve_candidates, encode_to_codes


def _toy_schema():
    s = load_schema("gss")
    return s


def test_retrieve_returns_k_neighbors():
    s = _toy_schema()
    bgv = list(s.background_variables)
    donors = pd.DataFrame({c: list(range(20)) for c in bgv})
    bg = pd.DataFrame({c: [0, 19] for c in bgv})
    idx = retrieve_candidates(donors, bg, s, k=5)
    assert idx.shape == (2, 5)
    # nearest neighbor of row 0 (all-zeros) should include donor 0
    assert 0 in idx[0]


def test_retrieve_keff_clamped_to_donor_count():
    s = _toy_schema()
    bgv = list(s.background_variables)
    donors = pd.DataFrame({c: [0, 1, 2] for c in bgv})
    bg = pd.DataFrame({c: [0] for c in bgv})
    idx = retrieve_candidates(donors, bg, s, k=10)
    assert idx.shape == (1, 3)


def test_encode_to_codes_categorical_and_numeric():
    s = load_schema("gss")
    targets = list(s.target_variables)[:2]
    supports = {}
    from ssdataagent.strategies import elicitation as E
    for t in targets:
        supports[t] = E.target_support(s, t, n_numeric_bins=10)
    donors = pd.DataFrame({t: [list((s.allowed_values.get(t) or [0]))[0]] * 3
                           if supports[t]["kind"] == "cat"
                           else [float(s.numeric_ranges[t][0])] * 3 for t in targets})
    codes = encode_to_codes(donors, targets, supports)
    for t in targets:
        assert codes[t].shape == (3,)
        n_bins = len(supports[t]["support"]) if supports[t]["kind"] == "cat" else len(supports[t]["edges"]) - 1
        assert codes[t].min() >= 0 and codes[t].max() < n_bins
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_design_c_retrieve.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (design_c not yet created).

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/strategies/design_c.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_design_c_retrieve.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_c.py tests/test_design_c_retrieve.py
git commit -m "design_c: k-NN retrieval + donor support-coding"
```

---

### Task 2: `design_c.py` — `repair_weights` (the net-new IPF)

**Files:**
- Modify: `src/ssdataagent/strategies/design_c.py`
- Test: `tests/test_design_c_repair.py`

**Interfaces:**
- Consumes: `retrieve_candidates` / `encode_to_codes` outputs; `goal_vectors` (`dict[str, np.ndarray]`, each summing to 1).
- Produces: `repair_weights(neighbor_idx, donor_codes, goal_vectors, supports, targets, *, max_iter=50, tol=1e-6) -> np.ndarray` — `(n_donor,)` non-negative weights raked so each eval row's candidate-weighted target marginal approaches `goal_vectors[t]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_design_c_repair.py
import numpy as np

from ssdataagent.strategies.design_c import repair_weights


def _induced_marginal(neighbor_idx, codes_t, w, n_bins):
    cand_codes = codes_t[neighbor_idx]
    cand_w = w[neighbor_idx]
    row_sum = cand_w.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    sel = cand_w / row_sum
    m = np.bincount(cand_codes.ravel(), weights=sel.ravel(), minlength=n_bins)[:n_bins]
    return m / len(neighbor_idx)


def test_repair_reduces_marginal_distance():
    # 4 donors, target with 2 bins; donors 0,1 -> bin 0, donors 2,3 -> bin 1
    codes = {"t": np.array([0, 0, 1, 1])}
    # every eval row sees all 4 donors as candidates
    neighbor_idx = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    supports = {"t": {"kind": "cat", "support": ["a", "b"]}}
    goal = {"t": np.array([0.25, 0.75])}  # want bin 1 dominant
    w = repair_weights(neighbor_idx, codes, goal, supports, ["t"])
    m_before = _induced_marginal(neighbor_idx, codes["t"], np.ones(4), 2)
    m_after = _induced_marginal(neighbor_idx, codes["t"], w, 2)
    d_before = np.abs(m_before - goal["t"]).sum()
    d_after = np.abs(m_after - goal["t"]).sum()
    assert d_after < d_before
    assert d_after < 1e-3  # converges on this separable case


def test_repair_handles_zero_mass_bin():
    codes = {"t": np.array([0, 0, 0])}  # no donor in bin 1
    neighbor_idx = np.array([[0, 1, 2]])
    supports = {"t": {"kind": "cat", "support": ["a", "b"]}}
    goal = {"t": np.array([0.5, 0.5])}
    w = repair_weights(neighbor_idx, codes, goal, supports, ["t"])
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)


def test_repair_no_targets_returns_unit_weights():
    w = repair_weights(np.zeros((0, 0), dtype=int), {}, {}, {}, [])
    assert w.shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_design_c_repair.py -q`
Expected: FAIL — `ImportError: cannot import name 'repair_weights'`.

- [ ] **Step 3: Write minimal implementation** (append to `design_c.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_design_c_repair.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_c.py tests/test_design_c_repair.py
git commit -m "design_c: repair_weights IPF over donor weights"
```

---

### Task 3: `design_c.py` — C pool bootstrap + final draw

**Files:**
- Modify: `src/ssdataagent/strategies/design_c.py`
- Test: `tests/test_design_c_pool.py`

**Interfaces:**
- Consumes: `elicitation.known_vector`, `elicitation.target_support`; `repair_weights` output.
- Produces:
  - `bootstrap_pool(background, known_marginals, supports, schema, *, seed=42) -> pd.DataFrame` — synthetic donor pool: backgrounds = eval backgrounds; targets drawn independently from `known_marginals`.
  - `draw_targets(neighbor_idx, weights, donors, targets, *, seed=42) -> dict[str, np.ndarray]` — each eval row samples one donor from its candidates ∝ weights; emits that donor's actual target value.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_design_c_pool.py
import numpy as np
import pandas as pd

from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.design_c import bootstrap_pool, draw_targets


def test_bootstrap_pool_matches_marginal_in_expectation():
    # one categorical target, known marginal 70/30 over two categories
    schema = type("S", (), {"background_variables": ["age"],
                            "allowed_values": {"x": ["a", "b"]},
                            "numeric_ranges": {}})()
    bg = pd.DataFrame({"age": list(range(2000))})
    supports = {"x": {"kind": "cat", "support": ["a", "b"]}}
    known = {"x": {"probs": {"a": 0.7, "b": 0.3}}}
    pool = bootstrap_pool(bg, known, supports, schema, seed=1)
    assert len(pool) == 2000
    assert list(pool["age"]) == list(bg["age"])  # backgrounds preserved
    frac_a = (pool["x"] == "a").mean()
    assert abs(frac_a - 0.7) < 0.05


def test_draw_targets_emits_real_donor_values():
    donors = pd.DataFrame({"t": ["a", "b", "c"]})
    neighbor_idx = np.array([[0, 1, 2], [0, 1, 2]])
    # weight donor 2 ('c') overwhelmingly
    weights = np.array([1e-9, 1e-9, 1.0])
    drawn = draw_targets(neighbor_idx, weights, donors, ["t"], seed=0)
    assert list(drawn["t"]) == ["c", "c"]


def test_draw_targets_deterministic():
    donors = pd.DataFrame({"t": [10.0, 20.0, 30.0, 40.0]})
    neighbor_idx = np.array([[0, 1], [2, 3], [0, 3]])
    weights = np.array([1.0, 2.0, 1.5, 0.5])
    a = draw_targets(neighbor_idx, weights, donors, ["t"], seed=7)
    b = draw_targets(neighbor_idx, weights, donors, ["t"], seed=7)
    assert np.array_equal(a["t"], b["t"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_design_c_pool.py -q`
Expected: FAIL — `ImportError: cannot import name 'bootstrap_pool'`.

- [ ] **Step 3: Write minimal implementation** (append to `design_c.py`)

```python
def bootstrap_pool(background, known_marginals, supports, schema, *, seed: int = 42) -> pd.DataFrame:
    """Condition-C synthetic donor pool: backgrounds = eval backgrounds; each
    donor's targets drawn independently from known_marginals (numeric -> uniform
    within the sampled bin; categorical -> the sampled category)."""
    rng = np.random.default_rng(seed)
    pool = background.reset_index(drop=True).copy()
    n = len(pool)
    for t, sup in supports.items():
        vec = np.asarray(E.known_vector(known_marginals.get(t), sup), float)
        idx = rng.choice(len(vec), size=n, p=vec)
        if sup["kind"] == "cat":
            pool[t] = [sup["support"][i] for i in idx]
        else:
            edges = np.asarray(sup["edges"], float)
            lo = edges[idx]
            hi = edges[idx + 1]
            pool[t] = lo + rng.random(n) * (hi - lo)
    return pool


def draw_targets(neighbor_idx, weights, donors, targets, *, seed: int = 42) -> dict[str, np.ndarray]:
    """Each eval row samples one donor from its candidate set ∝ weights, then
    emits that donor's actual target value (a real observation)."""
    rng = np.random.default_rng(seed)
    n_eval, k = neighbor_idx.shape
    chosen = np.empty(n_eval, dtype=int)
    for i in range(n_eval):
        cand = neighbor_idx[i]
        wv = weights[cand]
        s = wv.sum()
        p = wv / s if s > 0 else np.full(k, 1.0 / k)
        chosen[i] = cand[rng.choice(k, p=p)]
    donor_vals = donors.reset_index(drop=True)
    return {t: donor_vals[t].to_numpy()[chosen] for t in targets}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_design_c_pool.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_c.py tests/test_design_c_pool.py
git commit -m "design_c: C pool bootstrap + per-row donor draw"
```

---

### Task 4: `DesignCStrategy.generate` — A/B/C orchestration

**Files:**
- Modify: `src/ssdataagent/strategies/design_c.py`
- Test: `tests/test_strategy_design_c.py`

**Interfaces:**
- Consumes: `InfoGate` (`background`, `fit_microdata`, `known_marginals`, `condition`, `client`, `dataset_name`); `elicitation.elicit_cell_distributions`; all Task 1-3 functions; `background_frame`, `clip_decode`.
- Produces: `class DesignCStrategy` with `name = "design_c"` and `generate(self, gate, run_dir, cfg) -> StrategyResult`.

Notes for the implementer:
- Target set = `[t for t in schema.target_variables if t in known_marginals]` (mirrors Design B). Empty → early return with `background_frame` only and `meta_extras={"backend":"design_c","n_targets":0,"n_individuals":len(bg)}`.
- Donor frame: `bootstrap_pool(...)` when `gate.fit_microdata()` is `None` (condition C), else `gate.fit_microdata().reset_index(drop=True)`.
- Goal marginal: under `Condition.TRANSFER`, elicit ONE population-level cell (`cell_descs={"__population__": {"population": schema.population_context}}`, `transport=True`, anchors = `known_vector` of the source marginals) and use `transported["__population__"][t]`; otherwise `known_vector(known_marginals[t], supports[t])`.
- The cache dir mirrors Design B: `Path(getattr(cfg, "results_root", run_dir)) / "_elicitation_cache"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_design_c.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.design_c import DesignCStrategy


class _FakeClient:
    """Returns a uniform prob vector per target; counts calls."""
    def __init__(self):
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()

    def chat(self, messages, system=None):
        self.calls += 1
        # echo a valid-but-uniform JSON object across whatever targets are asked
        import re
        text = messages[-1]["content"]
        tnames = re.findall(r"\n- (\w+)", text)
        obj = {}
        for t in tnames:
            # 10 numeric bins or N categories — uniform of length found in the prompt
            obj[t] = [1.0]  # replaced below; see fallback note
        return json.dumps(obj)


def _train(schema, n=200, seed=0):
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


def _gate(condition, schema, tmp_path, source=None, crosswalk=()):
    train = _train(schema)
    bg = _train(schema, n=30, seed=1)
    return InfoGate(condition=condition, dataset_name="gss", workspace=tmp_path,
                    client=_FakeClient(), train=train, eval_rows=bg,
                    source=source, source_name="gss1994" if source is not None else None,
                    crosswalk=crosswalk)


def test_full_condition_generates_all_targets(tmp_path):
    schema = load_schema("gss")
    gate = _gate(Condition.FULL, schema, tmp_path)
    res = DesignCStrategy().generate(gate, tmp_path, cfg=type("Cfg", (), {"results_root": tmp_path})())
    for t in schema.target_variables:
        assert t in res.generated.columns
    assert len(res.generated) == 30
    assert res.meta_extras["backend"] == "design_c"
    assert json.loads(Path(tmp_path, "fit_summary.json").read_text())["transport"] is False


def test_aggregate_condition_does_not_raise(tmp_path):
    schema = load_schema("gss")
    gate = _gate(Condition.NO_DATA, schema, tmp_path)
    res = DesignCStrategy().generate(gate, tmp_path, cfg=type("Cfg", (), {"results_root": tmp_path})())
    assert len(res.generated) == 30
    assert res.meta_extras["transport"] is False


def test_full_condition_is_deterministic(tmp_path):
    schema = load_schema("gss")
    g1 = _gate(Condition.FULL, schema, tmp_path / "a")
    g2 = _gate(Condition.FULL, schema, tmp_path / "b")
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    r1 = DesignCStrategy().generate(g1, tmp_path / "a", cfg=type("Cfg", (), {"results_root": tmp_path})())
    r2 = DesignCStrategy().generate(g2, tmp_path / "b", cfg=type("Cfg", (), {"results_root": tmp_path})())
    # same train/bg seeds -> identical output
    pd.testing.assert_frame_equal(r1.generated, r2.generated)
```

Implementer note: the `_FakeClient` above must return a JSON object whose vector length matches each target's support. Replace the placeholder body so each target maps to a uniform vector of the correct length (parse the `numeric bins [...]` / `categories [...]` hint from the prompt, or simpler: return an empty `{}` so `elicit_cell_distributions` falls back to the anchor `known_vector` for every target — the fallback path is explicitly tested behavior and keeps the test robust). Prefer the empty-`{}` fallback for simplicity.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategy_design_c.py -q`
Expected: FAIL — `ImportError: cannot import name 'DesignCStrategy'`.

- [ ] **Step 3: Write minimal implementation** (append to `design_c.py`)

```python
class DesignCStrategy:
    name = "design_c"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        ref = gate.fit_microdata()            # train (A) / source[crosswalk] (B) / None (C)
        known_m = gate.known_marginals() or {}

        targets = [t for t in schema.target_variables if t in known_m]
        if not targets:
            return StrategyResult(generated=background_frame(bg, schema),
                                  meta_extras={"backend": "design_c", "n_targets": 0,
                                               "n_individuals": len(bg)})

        supports = {t: E.target_support(schema, t, n_numeric_bins=_N_NUMERIC_BINS) for t in targets}

        if ref is None:
            donors = bootstrap_pool(bg, known_m, supports, schema, seed=42)
        else:
            donors = ref.reset_index(drop=True)

        if gate.condition is Condition.TRANSFER:
            anchors = {t: E.known_vector(known_m.get(t), supports[t]) for t in targets}
            transported = E.elicit_cell_distributions(
                gate.client, dataset=gate.dataset_name, condition=gate.condition.value,
                cell_descs={"__population__": {"population": schema.population_context}},
                schema=schema, targets=targets, supports=supports, known_vectors=anchors,
                run_dir=run_dir,
                cache_dir=Path(getattr(cfg, "results_root", run_dir)) / "_elicitation_cache",
                transport=True,
            )
            goal = {t: transported["__population__"][t] for t in targets}
            transport_used = True
        else:
            goal = {t: E.known_vector(known_m.get(t), supports[t]) for t in targets}
            transport_used = False

        neighbor_idx = retrieve_candidates(donors, bg, schema, k=_K)
        donor_codes = encode_to_codes(donors, targets, supports)
        weights = repair_weights(neighbor_idx, donor_codes, goal, supports, targets,
                                 max_iter=_REPAIR_ITERS)
        drawn = draw_targets(neighbor_idx, weights, donors, targets, seed=42)

        out = background_frame(bg, schema)
        for t in targets:
            out[t] = drawn[t]
        generated = clip_decode(out, schema)

        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "design_c", "condition": gate.condition.value, "k": _K,
             "n_donors": len(donors), "n_targets": len(targets),
             "repair_iters": _REPAIR_ITERS, "transport": transport_used}, indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "design_c", "condition": gate.condition.value,
                         "k": _K, "n_donors": len(donors), "n_targets": len(targets),
                         "transport": transport_used, "n_individuals": len(bg)},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategy_design_c.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_c.py tests/test_strategy_design_c.py
git commit -m "design_c: DesignCStrategy A/B/C orchestration"
```

---

### Task 5: Register strategy + A/B/C conditions + runner characterization

**Files:**
- Modify: `src/ssdataagent/strategies/registry.py`
- Modify: `src/ssdataagent/experiments/conditions.py`
- Test: `tests/test_strategies_registry.py` (extend), `tests/test_conditions.py` (extend), `tests/test_runner_artifacts.py` (extend)

**Interfaces:**
- Consumes: `DesignCStrategy`; `Condition.FULL/TRANSFER/NO_DATA`.
- Produces: registry key `"design_c"`; conditions `design_c_full`, `design_c_transfer`, `design_c_aggregate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_strategies_registry.py  (add)
def test_design_c_registered():
    from ssdataagent.strategies.registry import get_strategy
    s = get_strategy("design_c")
    assert s.name == "design_c"
```

```python
# tests/test_conditions.py  (add)
def test_design_c_conditions():
    from ssdataagent.agent.context import Condition
    from ssdataagent.experiments.conditions import get_condition
    assert get_condition("design_c_full").context_condition is Condition.FULL
    assert get_condition("design_c_full").strategy == "design_c"
    assert get_condition("design_c_transfer").context_condition is Condition.TRANSFER
    assert get_condition("design_c_aggregate").context_condition is Condition.NO_DATA
```

For `tests/test_runner_artifacts.py`, add a transfer-gate characterization test mirroring the existing Design B one: a `_fake_design_c_generate` that asserts the gate it receives under `design_c_transfer` (on a `TRANSFER_PAIRS` dataset) has `gate.source is not None` and `gate.crosswalk`. Reuse the existing test's monkeypatch pattern (look at the Design B transfer characterization test already in this file and copy its structure, swapping the strategy name to `design_c` and condition to `design_c_transfer`). Keep the two P0 byte-stable tests (`test_agent_artifacts_are_stable`, `test_direct_artifacts_are_stable`) untouched.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py -q`
Expected: FAIL — `KeyError: unknown strategy 'design_c'` / `unknown condition 'design_c_full'`.

- [ ] **Step 3: Write minimal implementation**

In `registry.py`: import `DesignCStrategy` and add `"design_c": DesignCStrategy` to `STRATEGIES`.

```python
from ssdataagent.strategies.design_c import DesignCStrategy
# ...
    "design_b": DesignBStrategy,
    "design_c": DesignCStrategy,
```

In `conditions.py`, add to `CONDITIONS`:

```python
    "design_c_full": ConditionSpec("design_c_full", Condition.FULL, strategy="design_c"),
    "design_c_aggregate": ConditionSpec("design_c_aggregate", Condition.NO_DATA, strategy="design_c"),
    "design_c_transfer": ConditionSpec("design_c_transfer", Condition.TRANSFER, strategy="design_c"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py tests/test_runner_artifacts.py -q`
Expected: PASS (including the new characterization test and the unchanged P0 byte-stable tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/registry.py src/ssdataagent/experiments/conditions.py tests/test_strategies_registry.py tests/test_conditions.py tests/test_runner_artifacts.py
git commit -m "design_c: register strategy + A/B/C condition specs"
```

---

## Self-Review (plan author)

- **Spec coverage:** §4 conditions → Tasks 3 (C pool) + 4 (A/B/C orchestration); §5 components → Tasks 1-4 (one function group each); §6 repair math → Task 2; §7 registration/runner → Task 5; §8 determinism/leakage/artifacts → asserted in Tasks 3-5 tests; §10 testing → one test file per task. All covered.
- **Placeholders:** none — every step ships complete code, except the Task 4 `_FakeClient` body, which is explicitly resolved via the stated empty-`{}` fallback path.
- **Type consistency:** `retrieve_candidates → (n_eval, k_eff) int`, `encode_to_codes → dict[t, (n_donor,) int]`, `repair_weights → (n_donor,) float`, `draw_targets → dict[t, (n_eval,)]`, `bootstrap_pool → DataFrame` — consumed consistently in Task 4. `goal`/`supports`/`targets` shapes match across Tasks 2-4.
- **Leakage:** Task 4 routes B's goal through `elicit_cell_distributions(transport=True)` on source-anchored marginals; donors are `gate.fit_microdata()` (= `source[crosswalk]` under TRANSFER). No target-survey targets are read.
