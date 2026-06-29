# Design B Strategy (Part 3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Design B — the first LLM-grounded generator: per-cell LLM marginal elicitation → IPF rake to known marginals → data-grounded copula coupling → sample, across conditions A/B/C.

**Architecture:** Every target is a probability vector over a fixed support (categorical → `allowed_values`; numeric → K even-width bins over range). The LLM elicits per-cell vectors (cached, logged); IPF raking calibrates them so the cell-weighted mixture matches the known marginal; a Gaussian copula over targets (signed correlation from microdata in A/B, identity in C) supplies cross-target dependence; sampling draws a correlated latent, maps each component through the row's cell's calibrated marginal. Built on the Part 3a foundation; reuses extracted copula latent helpers and a shared cell-partition utility.

**Tech Stack:** Python 3.11+, numpy, pandas, scipy (installed). No new dependencies.

## Global Constraints

- **No new dependencies.**
- **Uniform target representation:** every target is a prob vector over a fixed support — categorical/ordinal → `allowed_values`; numeric → `K` even-width bins over `numeric_ranges` (default `K=10`). One code path for both. (This concretizes the spec's "categorical probs or numeric quantiles".)
- **Copula:** signed correlation from microdata in A/B; **identity (independence) in C** (`known_associations` are unsigned, so no signed copula is fabricated). `known_associations` is NOT consumed by Design B v1.
- **Background-conditioned copula is out of scope** — v1 draws from the unconditional target copula; the demographic signal lives in the per-cell calibrated marginals. Therefore `copula.py` adds only the unconditional `correlated_normal` helper (no `conditional_gaussian_sample` this cycle — YAGNI, refines spec §1).
- **Determinism via a persistent elicitation cache** (keyed by dataset/condition/model/cell/target-set/prompt-version), not temperature=0. Sampling/raking use a seeded RNG (default 42). Tests mock the LLM client → fully deterministic, no network.
- **No target leakage:** outputs built via `background_frame`; only target-set (crosswalk under B) targets are filled by the model; non-crosswalk targets fall through to the runner's `format_generated` baseline fill.
- **Behavior-preserving refactors:** extracting `strategies/copula.py` (baselines re-imports) and pointing `evaluation/overdetermination.py` coarsening at `data/cells.py` must not change any existing behavior — guarded by the existing tests (`test_strategy_copula.py`, `test_overdetermination.py`) still passing.
- Test runner `.venv/bin/pytest`. Pre-existing failures NOT in scope (do not fix): `tests/test_config.py::test_unknown_provider_raises` + 3 `tests/test_ssdatabench_integration.py *_legacy` (missing `autograd`). Gate = our tests pass + no NEW failures.
- Follow repo style: `from __future__ import annotations`, module-level functions, `StrategyResult(generated, meta_extras)` contract, raise `ValueError` when required microdata is absent (note: Design B in C legitimately has `fit_microdata()==None` and must NOT raise — it uses `known_marginals`).

---

### Task 1: Extract `strategies/copula.py` (shared latent machinery)

**Files:**
- Create: `src/ssdataagent/strategies/copula.py`
- Modify: `src/ssdataagent/strategies/baselines.py` (import the helpers instead of defining them)
- Test: `tests/test_copula_module.py`

**Interfaces:**
- Produces (used by Task 5): `build_cuts(train, cols, schema) -> dict`, `latent_value(col_cut, value) -> float`, `latent_matrix(df, cols, cuts) -> np.ndarray`, `invert(z_array, col_cut) -> list`, `make_pd(M, reg) -> np.ndarray`, `correlated_normal(Sigma, n_samples, rng) -> np.ndarray`, and `EPS`.

This is a pure move: the helper bodies are copied verbatim from `baselines.py` (lines ~188–248), renamed from `_build_cuts`/`_latent_value`/`_latent_matrix`/`_invert`/`_make_pd`/`_EPS` to public names, with `correlated_normal` added.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_copula_module.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies import copula


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=[], target_variables=["vote", "income"],
        descriptions={}, allowed_values={"vote": ["A", "B"]},
        numeric_ranges={"income": (0.0, 100.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_build_cuts_and_roundtrip():
    s = toy_schema()
    df = pd.DataFrame({"vote": ["A", "B", "A", "B"], "income": [10.0, 90.0, 20.0, 80.0]})
    cuts = copula.build_cuts(df, ["vote", "income"], s)
    assert cuts["vote"]["kind"] == "cat" and cuts["income"]["kind"] == "num"
    # latent then invert returns valid support members / in-range numerics
    z = copula.latent_matrix(df, ["vote", "income"], cuts)
    assert z.shape == (4, 2)
    inv_vote = copula.invert(z[:, 0], cuts["vote"])
    assert set(inv_vote).issubset({"A", "B"})


def test_make_pd_is_positive_definite():
    M = np.array([[1.0, 0.9], [0.9, 1.0]])
    pd_M = copula.make_pd(M, 1e-6)
    assert np.all(np.linalg.eigvalsh(pd_M) > 0)


def test_correlated_normal_reproduces_correlation():
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    rng = np.random.default_rng(0)
    samples = copula.correlated_normal(Sigma, 20000, rng)
    assert samples.shape == (20000, 2)
    emp = np.corrcoef(samples, rowvar=False)[0, 1]
    assert abs(emp - 0.8) < 0.05


def test_baselines_still_imports_helpers():
    # behavior-preserving extraction: copula strategy path still works
    from ssdataagent.strategies.baselines import copula_generate
    s = toy_schema()
    train = pd.DataFrame({"vote": ["A", "B"] * 20, "income": list(np.linspace(0, 100, 40))})
    out = copula_generate(train, train[[]].assign(profile_id=range(40)), s, seed=1)
    assert set(out["vote"].unique()).issubset({"A", "B"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_copula_module.py -v`
Expected: FAIL (`ModuleNotFoundError: ssdataagent.strategies.copula`)

- [ ] **Step 3: Create `src/ssdataagent/strategies/copula.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-6


def build_cuts(train, cols, schema) -> dict:
    """Per-column inversion data. numeric -> sorted train values;
    categorical -> (categories, cumulative upper edges)."""
    cuts: dict[str, dict] = {}
    for c in cols:
        if c in schema.numeric_ranges:
            vals = pd.to_numeric(train[c], errors="coerce").dropna().to_numpy()
            cuts[c] = {"kind": "num", "sorted": np.sort(vals)}
        else:
            cats = schema.allowed_values.get(c) or sorted(train[c].dropna().unique().tolist())
            counts = train[c].value_counts()
            probs = np.array([max(counts.get(v, 0), 0) for v in cats], dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(cats), 1.0 / len(cats))
            cuts[c] = {"kind": "cat", "cats": list(cats), "cum": np.cumsum(probs)}
    return cuts


def latent_value(col_cut, value) -> float:
    if col_cut["kind"] == "num":
        s = col_cut["sorted"]
        if len(s) == 0 or pd.isna(value):
            return 0.0
        pos = int(np.searchsorted(s, float(value), side="right"))
        u = min(max((pos - 0.5) / len(s), EPS), 1 - EPS)
        return float(norm.ppf(u))
    cats, cum = col_cut["cats"], col_cut["cum"]
    if value not in cats:
        return 0.0
    i = cats.index(value)
    lo = cum[i - 1] if i > 0 else 0.0
    u = min(max((lo + cum[i]) / 2.0, EPS), 1 - EPS)
    return float(norm.ppf(u))


def latent_matrix(df, cols, cuts) -> np.ndarray:
    out = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        out[:, j] = [latent_value(cuts[c], v) for v in df[c].tolist()]
    return out


def invert(z_array, col_cut) -> list:
    u = np.clip(norm.cdf(z_array), EPS, 1 - EPS)
    if col_cut["kind"] == "num":
        s = col_cut["sorted"]
        return list(np.quantile(s, u)) if len(s) else [0.0] * len(u)
    cats, cum = col_cut["cats"], col_cut["cum"]
    idx = np.searchsorted(cum, u, side="left")
    idx = np.clip(idx, 0, len(cats) - 1)
    return [cats[i] for i in idx]


def make_pd(M, reg) -> np.ndarray:
    M = (M + M.T) / 2.0
    M = M + reg * np.eye(M.shape[0])
    w, V = np.linalg.eigh(M)
    w = np.clip(w, reg, None)
    return (V * w) @ V.T


def correlated_normal(Sigma, n_samples, rng) -> np.ndarray:
    """Draw n_samples rows from N(0, Sigma) using the Cholesky factor."""
    d = Sigma.shape[0]
    if d == 0:
        return np.zeros((n_samples, 0))
    L = np.linalg.cholesky(make_pd(Sigma, 1e-9))
    return rng.standard_normal((n_samples, d)) @ L.T
```

- [ ] **Step 4: Update `src/ssdataagent/strategies/baselines.py`**

Delete the local `_EPS`, `_build_cuts`, `_latent_value`, `_latent_matrix`, `_invert`, `_make_pd` definitions (lines ~188–248). Add an import near the top with the other imports:

```python
from ssdataagent.strategies.copula import (
    build_cuts as _build_cuts,
    invert as _invert,
    latent_matrix as _latent_matrix,
    make_pd as _make_pd,
)
```

(The aliases keep `copula_generate`'s body — which calls `_build_cuts`/`_latent_matrix`/`_make_pd`/`_invert` — unchanged. `_latent_value` and `_EPS` were only used by the moved functions, so they need no alias.)

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_copula_module.py tests/test_strategy_copula.py -v`
Expected: PASS (new module tests + the existing copula-strategy tests prove the extraction is behavior-preserving)

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/strategies/copula.py src/ssdataagent/strategies/baselines.py tests/test_copula_module.py
git commit -m "copula: extract shared Gaussian-copula latent helpers + correlated_normal"
```

---

### Task 2: Shared cell partition (`data/cells.py`) + metric refactor

**Files:**
- Create: `src/ssdataagent/data/cells.py`
- Modify: `src/ssdataagent/evaluation/overdetermination.py` (coarsening delegates here)
- Test: `tests/test_cells.py`

**Interfaces:**
- Produces (used by Task 6 + the metric): `bin_edges(vals, n_bins) -> np.ndarray`, `discretize(values, edges) -> np.ndarray`, `fit_scheme(df, variables, schema, *, n_bins=4) -> CellScheme`, `assign(df, scheme) -> pd.Series`. `CellScheme` is a dataclass holding `variables: list[str]`, `edges: dict[str, np.ndarray]` (numeric vars only).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cells.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.data import cells


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={"age": (18.0, 90.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_fit_and_assign_consistent_across_frames():
    s = toy_schema()
    train = pd.DataFrame({"age": np.linspace(20, 80, 40), "region": ["N", "S"] * 20})
    scheme = cells.fit_scheme(train, ["age", "region"], s, n_bins=4)
    a = cells.assign(train, scheme)
    # same scheme applied to a second frame yields keys drawn from the same space
    other = pd.DataFrame({"age": [25.0, 75.0], "region": ["N", "S"]})
    b = cells.assign(other, scheme)
    assert len(a) == 40 and len(b) == 2
    assert all("|" in k for k in a)            # composite cell keys (age_bin|region)


def test_bin_edges_and_discretize():
    edges = cells.bin_edges(pd.Series([0.0, 25.0, 50.0, 75.0, 100.0]), 4)
    idx = cells.discretize(pd.Series([0.0, 100.0]), edges)
    assert idx[0] == 0 and idx[1] == len(edges) - 2


def test_metric_still_works_after_refactor():
    # the over-determination metric consumes cells.* now; smoke it
    from ssdataagent.evaluation.overdetermination import overdetermination
    s = toy_schema()
    real = pd.DataFrame({"region": ["N"] * 40, "age": np.linspace(20, 80, 40),
                         "vote": ["A", "B"] * 20})
    sim = pd.DataFrame({"region": ["N"] * 40, "age": np.linspace(20, 80, 40),
                        "vote": ["A"] * 40})
    res = overdetermination(real=real, sim=sim, schema=s, min_count=10)
    assert "cell_based" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cells.py -v`
Expected: FAIL (`ModuleNotFoundError: ssdataagent.data.cells`)

- [ ] **Step 3: Create `src/ssdataagent/data/cells.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema


def bin_edges(vals, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(pd.to_numeric(vals, errors="coerce").dropna(), qs))
    if len(edges) < 2:
        edges = (np.array([edges[0] - 1e-9, edges[0] + 1e-9])
                 if len(edges) else np.array([0.0, 1.0]))
    return edges


def discretize(values, edges) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy()
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)


@dataclass
class CellScheme:
    variables: list[str]
    edges: dict[str, np.ndarray] = field(default_factory=dict)


def fit_scheme(df, variables, schema: DatasetSchema, *, n_bins: int = 4) -> CellScheme:
    edges: dict[str, np.ndarray] = {}
    for v in variables:
        if v in schema.numeric_ranges:
            edges[v] = bin_edges(df[v], n_bins)
    return CellScheme(variables=list(variables), edges=edges)


def assign(df, scheme: CellScheme) -> pd.Series:
    parts = []
    for v in scheme.variables:
        if v in scheme.edges:
            parts.append(discretize(df[v], scheme.edges[v]).astype(str))
        else:
            parts.append(df[v].astype(str).to_numpy())
    keys = ["|".join(t) for t in zip(*parts)] if parts else ["_"] * len(df)
    return pd.Series(keys, index=df.index)
```

- [ ] **Step 4: Refactor `src/ssdataagent/evaluation/overdetermination.py`**

Replace the local `_bin_edges`/`_discretize`/`_coarsen` with delegation to `cells`. At the top, add:

```python
from ssdataagent.data import cells
```

Replace the three helpers with:

```python
def _bin_edges(real_vals, n_bins):
    return cells.bin_edges(real_vals, n_bins)


def _discretize(values, edges):
    return cells.discretize(values, edges)


def _coarsen(real, sim, schema, n_demo_bins):
    scheme = cells.fit_scheme(real, schema.background_variables, schema, n_bins=n_demo_bins)
    return cells.assign(real, scheme).to_numpy(), cells.assign(sim, scheme).to_numpy()
```

(Everything else in `overdetermination.py` is unchanged — `_target_series` still calls `_bin_edges`/`_discretize`, which now delegate.)

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_cells.py tests/test_overdetermination.py -v`
Expected: PASS (new cells tests + the existing metric tests prove the refactor is behavior-preserving)

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/data/cells.py src/ssdataagent/evaluation/overdetermination.py tests/test_cells.py
git commit -m "cells: shared cell-partition util; over-determination metric delegates to it"
```

---

### Task 3: Elicitation layer (`strategies/elicitation.py`)

**Files:**
- Create: `src/ssdataagent/strategies/elicitation.py`
- Test: `tests/test_elicitation.py`

**Interfaces:**
- Consumes: `LLMClient.chat`; `DatasetSchema`.
- Produces (used by Tasks 4-6): `target_support(schema, target, *, n_numeric_bins=10) -> dict`; `known_vector(known_m_t, support) -> np.ndarray`; `elicit_cell_distributions(client, *, dataset, condition, cell_descs, schema, targets, supports, known_vectors, run_dir, cache_dir, transport=False, max_retries=3) -> dict[str, dict[str, np.ndarray]]` (cell_key → target → prob vector).
- `target_support` returns `{"kind":"cat","support":[...]}` or `{"kind":"num","edges":np.ndarray}` (support length = `len(support)` or `len(edges)-1`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_elicitation.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies import elicitation as E


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["vote", "income"],
        descriptions={"vote": "party"}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={"income": (0.0, 100.0)},
        population_context="ctx", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
    def chat(self, messages, system=None):
        self.calls += 1
        self.last_messages = messages
        return self.responses.pop(0)


def test_target_support_and_known_vector():
    s = toy_schema()
    sup_v = E.target_support(s, "vote")
    assert sup_v == {"kind": "cat", "support": ["A", "B"]}
    sup_i = E.target_support(s, "income", n_numeric_bins=4)
    assert sup_i["kind"] == "num" and len(sup_i["edges"]) == 5
    kv = E.known_vector({"kind": "categorical", "probs": {"A": 0.75, "B": 0.25}}, sup_v)
    assert np.allclose(kv, [0.75, 0.25])


def test_parse_renormalizes_and_caches(tmp_path):
    s = toy_schema()
    sup = {"vote": E.target_support(s, "vote"), "income": E.target_support(s, "income", n_numeric_bins=4)}
    kv = {"vote": np.array([0.5, 0.5]), "income": np.full(4, 0.25)}
    resp = json.dumps({"vote": [3, 1], "income": [1, 1, 1, 1]})  # unnormalized
    client = FakeClient([resp])
    out = E.elicit_cell_distributions(
        client, dataset="toy", condition="full_agent",
        cell_descs={"N": {"region": "N"}}, schema=s,
        targets=["vote", "income"], supports=sup, known_vectors=kv,
        run_dir=tmp_path, cache_dir=tmp_path / "cache",
    )
    assert np.allclose(out["N"]["vote"], [0.75, 0.25])      # renormalized
    assert abs(out["N"]["income"].sum() - 1.0) < 1e-9
    # cache hit: a second call with the same args makes NO new client call
    client2 = FakeClient([])  # would IndexError if called
    out2 = E.elicit_cell_distributions(
        client2, dataset="toy", condition="full_agent",
        cell_descs={"N": {"region": "N"}}, schema=s,
        targets=["vote", "income"], supports=sup, known_vectors=kv,
        run_dir=tmp_path, cache_dir=tmp_path / "cache",
    )
    assert client2.calls == 0
    assert np.allclose(out2["N"]["vote"], [0.75, 0.25])
    # raw I/O logged
    assert (tmp_path / "elicitation").exists()


def test_malformed_json_falls_back_to_known(tmp_path):
    s = toy_schema()
    sup = {"vote": E.target_support(s, "vote")}
    kv = {"vote": np.array([0.6, 0.4])}
    client = FakeClient(["not json", "still not json", "{bad", "{bad}"])  # all retries fail
    out = E.elicit_cell_distributions(
        client, dataset="toy", condition="full_agent",
        cell_descs={"N": {"region": "N"}}, schema=s,
        targets=["vote"], supports=sup, known_vectors=kv,
        run_dir=tmp_path, cache_dir=tmp_path / "cache", max_retries=3,
    )
    assert np.allclose(out["N"]["vote"], [0.6, 0.4])        # fell back to known marginal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_elicitation.py -v`
Expected: FAIL (`ModuleNotFoundError: ssdataagent.strategies.elicitation`)

- [ ] **Step 3: Create `src/ssdataagent/strategies/elicitation.py`**

```python
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ssdataagent.data.schema import DatasetSchema

_PROMPT_VERSION = "designb-v1"
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You are a survey-distribution estimator. For a demographic subgroup, you "
    "estimate the DISTRIBUTION of each target variable across people in that "
    "subgroup — not a single typical value. Real subgroups have substantial "
    "internal variation; reflect that spread. Return ONLY a JSON object."
)


def target_support(schema: DatasetSchema, target: str, *, n_numeric_bins: int = 10) -> dict:
    if target in schema.numeric_ranges:
        lo, hi = schema.numeric_ranges[target]
        return {"kind": "num", "edges": np.linspace(float(lo), float(hi), n_numeric_bins + 1)}
    cats = schema.allowed_values.get(target) or []
    return {"kind": "cat", "support": list(cats)}


def _support_len(support: dict) -> int:
    return len(support["support"]) if support["kind"] == "cat" else len(support["edges"]) - 1


def known_vector(known_m_t: dict, support: dict) -> np.ndarray:
    if support["kind"] == "cat":
        probs = (known_m_t or {}).get("probs", {})
        v = np.array([float(probs.get(str(c), 0.0)) for c in support["support"]], float)
    else:
        quant = (known_m_t or {}).get("quantiles", {})
        edges = support["edges"]
        if quant:
            qs = sorted(float(q) for q in quant.keys())
            vals = [float(quant[str(q)]) if str(q) in quant else float(quant[q]) for q in qs]
            cdf = np.interp(edges, vals, qs, left=0.0, right=1.0)
            v = np.clip(np.diff(cdf), 0.0, None)
        else:
            v = np.zeros(_support_len(support))
    s = v.sum()
    return v / s if s > 0 else np.full(_support_len(support), 1.0 / _support_len(support))


def _normalize_to_support(raw, support: dict) -> np.ndarray | None:
    n = _support_len(support)
    if not isinstance(raw, (list, tuple)) or len(raw) != n:
        return None
    try:
        v = np.array([max(float(x), 0.0) for x in raw], float)
    except (TypeError, ValueError):
        return None
    s = v.sum()
    return v / s if s > 0 else None


def _describe_support(support: dict) -> str:
    if support["kind"] == "cat":
        return f"categories {support['support']} (give one probability per category, in order)"
    e = support["edges"]
    ranges = [f"[{e[i]:.4g},{e[i+1]:.4g})" for i in range(len(e) - 1)]
    return f"numeric bins {ranges} (give one probability per bin, in order)"


def _build_prompt(*, dataset, cell_desc, schema, targets, supports, known_vectors, transport) -> str:
    lines = [
        f"Population: {schema.population_context}",
        f"Demographic subgroup: {json.dumps(cell_desc, default=str)}",
        "",
        "For EACH target below, return a probability vector over its support "
        "(probabilities for that subgroup; reflect realistic within-subgroup spread):",
    ]
    for t in targets:
        desc = schema.descriptions.get(t, "")
        anchor = np.round(known_vectors[t], 4).tolist()
        lines.append(f"- {t}{(': ' + desc) if desc else ''} — {_describe_support(supports[t])}. "
                     f"Population-wide marginal (anchor, do not copy blindly): {anchor}")
    if transport:
        lines.append("")
        lines.append("NOTE: the anchors come from a DIFFERENT source population. Adapt each "
                     "subgroup distribution to THIS population's context, not the source's.")
    lines.append("")
    lines.append('Respond with ONLY JSON: {"<target>": [p1, p2, ...], ...}')
    return "\n".join(lines)


def _cache_key(dataset, condition, model, cell_key, targets) -> str:
    blob = json.dumps([dataset, condition, model, cell_key, sorted(targets), _PROMPT_VERSION],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def elicit_cell_distributions(
    client, *, dataset, condition, cell_descs, schema, targets, supports,
    known_vectors, run_dir, cache_dir, transport=False, max_retries=3,
) -> dict[str, dict[str, np.ndarray]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(run_dir) / "elicitation"
    log_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(getattr(client, "cfg", None), "model", "unknown")
    result: dict[str, dict[str, np.ndarray]] = {}

    for cell_key, cell_desc in cell_descs.items():
        key = _cache_key(dataset, condition, model, cell_key, targets)
        cache_file = cache_dir / f"{key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            result[cell_key] = {t: np.array(cached[t], float) for t in targets}
            continue

        prompt = _build_prompt(dataset=dataset, cell_desc=cell_desc, schema=schema,
                               targets=targets, supports=supports,
                               known_vectors=known_vectors, transport=transport)
        parsed: dict[str, np.ndarray] = {}
        raw = ""
        for attempt in range(max_retries + 1):
            raw = client.chat(messages=[{"role": "user", "content": prompt}], system=_SYSTEM)
            m = _JSON_OBJ.search(raw or "")
            obj = {}
            if m:
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    obj = {}
            ok = True
            parsed = {}
            for t in targets:
                vec = _normalize_to_support(obj.get(t), supports[t])
                if vec is None:
                    ok = False
                    break
                parsed[t] = vec
            if ok:
                break
        else:
            parsed = {}
        # fallback for any target not successfully parsed
        for t in targets:
            if t not in parsed:
                parsed[t] = np.array(known_vectors[t], float)

        (log_dir / f"{cell_key.replace('|', '_')}.prompt.txt").write_text(prompt)
        (log_dir / f"{cell_key.replace('|', '_')}.response.txt").write_text(raw or "")
        cache_file.write_text(json.dumps({t: parsed[t].tolist() for t in targets}))
        result[cell_key] = parsed
    return result
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_elicitation.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/elicitation.py tests/test_elicitation.py
git commit -m "design_b: LLM per-cell elicitation layer with cache + raw-IO logging"
```

---

### Task 4: Raking (IPF calibration) in `strategies/design_b.py`

**Files:**
- Create: `src/ssdataagent/strategies/design_b.py`
- Test: `tests/test_design_b_rake.py`

**Interfaces:**
- Produces (used by Task 6): `rake(cell_vectors: dict[str, np.ndarray], cell_weights: dict[str, float], known_vec: np.ndarray, *, max_iter=50, tol=1e-6) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_design_b_rake.py
import numpy as np

from ssdataagent.strategies.design_b import rake


def test_rake_matches_known_marginal():
    # two cells, equal weight; known marginal [0.5, 0.5]; LLM gave skewed cells
    cell_vectors = {"c0": np.array([0.9, 0.1]), "c1": np.array([0.3, 0.7])}
    cell_weights = {"c0": 0.5, "c1": 0.5}
    known = np.array([0.5, 0.5])
    out = rake(cell_vectors, cell_weights, known)
    mix = 0.5 * out["c0"] + 0.5 * out["c1"]
    assert np.allclose(mix, known, atol=1e-4)
    # relative ordering within each cell preserved (c0 still favors index 0)
    assert out["c0"][0] > out["c0"][1]


def test_rake_each_cell_sums_to_one():
    out = rake({"c0": np.array([0.2, 0.8])}, {"c0": 1.0}, np.array([0.5, 0.5]))
    assert abs(out["c0"].sum() - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_design_b_rake.py -v`
Expected: FAIL (`ModuleNotFoundError` / `ImportError: rake`)

- [ ] **Step 3: Create `src/ssdataagent/strategies/design_b.py` with `rake`**

```python
from __future__ import annotations

import numpy as np


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_design_b_rake.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_b.py tests/test_design_b_rake.py
git commit -m "design_b: IPF raking of per-cell marginals to known marginal"
```

---

### Task 5: Copula build + sampling in `strategies/design_b.py`

**Files:**
- Modify: `src/ssdataagent/strategies/design_b.py`
- Test: `tests/test_design_b_sample.py`

**Interfaces:**
- Consumes: `copula.build_cuts`, `copula.latent_matrix`, `copula.make_pd`, `copula.correlated_normal` (Task 1); `scipy.stats.norm`.
- Produces (used by Task 6): `build_target_copula(ref, targets, schema, *, reg=1e-6) -> np.ndarray` (a T×T correlation matrix; identity when `ref is None` or `< 2` targets); `sample_targets(eval_cell_keys, calibrated, supports, Sigma, targets, *, seed=42) -> dict[str, list]` (target → per-row values).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_design_b_sample.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.design_b import build_target_copula, sample_targets
from ssdataagent.strategies import elicitation as E


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["vote", "income"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={"income": (0.0, 100.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_build_target_copula_identity_when_no_ref():
    s = toy_schema()
    Sig = build_target_copula(None, ["vote", "income"], s)
    assert np.allclose(Sig, np.eye(2))


def test_build_target_copula_from_ref_is_correlation():
    s = toy_schema()
    ref = pd.DataFrame({"vote": ["A", "B"] * 50, "income": list(np.linspace(0, 100, 100))})
    Sig = build_target_copula(ref, ["vote", "income"], s)
    assert Sig.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(Sig) > 0)


def test_sample_respects_support_and_ranges():
    s = toy_schema()
    supports = {"vote": E.target_support(s, "vote"),
                "income": E.target_support(s, "income", n_numeric_bins=4)}
    calibrated = {"N": {"vote": np.array([1.0, 0.0]),          # always A
                        "income": np.array([1.0, 0.0, 0.0, 0.0])}}  # lowest bin
    eval_cells = ["N", "N", "N"]
    out = sample_targets(eval_cells, calibrated, supports, np.eye(2),
                         ["vote", "income"], seed=1)
    assert out["vote"] == ["A", "A", "A"]                       # degenerate marginal honored
    assert all(0.0 <= x <= 25.0 for x in out["income"])         # lowest of 4 even bins of [0,100]


def test_sample_is_deterministic():
    s = toy_schema()
    supports = {"vote": E.target_support(s, "vote"),
                "income": E.target_support(s, "income", n_numeric_bins=4)}
    calibrated = {"N": {"vote": np.array([0.5, 0.5]), "income": np.full(4, 0.25)}}
    ev = ["N"] * 20
    a = sample_targets(ev, calibrated, supports, np.eye(2), ["vote", "income"], seed=7)
    b = sample_targets(ev, calibrated, supports, np.eye(2), ["vote", "income"], seed=7)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_design_b_sample.py -v`
Expected: FAIL (`ImportError: build_target_copula`)

- [ ] **Step 3: Add to `src/ssdataagent/strategies/design_b.py`**

```python
from scipy.stats import norm

from ssdataagent.strategies import copula


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_design_b_sample.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/design_b.py tests/test_design_b_sample.py
git commit -m "design_b: target copula (signed in A/B, identity in C) + per-cell sampling"
```

---

### Task 6: `DesignBStrategy.generate` orchestration

**Files:**
- Modify: `src/ssdataagent/strategies/design_b.py`
- Test: `tests/test_strategy_design_b.py`

**Interfaces:**
- Consumes: `InfoGate` (`background`, `fit_microdata`, `known_marginals`, `dataset_name`, `client`, `crosswalk`, `condition`); `load_schema`; `cells.fit_scheme`/`assign`; `elicitation.target_support`/`known_vector`/`elicit_cell_distributions`; `rake`, `build_target_copula`, `sample_targets`; `background_frame`/`clip_decode` from `baselines`; `StrategyResult`.
- Produces (used by Task 7): class `DesignBStrategy` with `name = "design_b"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_design_b.py
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.design_b import DesignBStrategy


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={}, population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


class FixedClient:
    cfg = type("C", (), {"model": "m"})()
    def chat(self, messages, system=None):
        # always emit vote distribution [0.5, 0.5] over support ["A","B"]
        return json.dumps({"vote": [0.5, 0.5]})


def _gate(condition, train, ev, **kw):
    base = dict(condition=condition, dataset_name="toy", workspace=Path("/tmp"),
                client=FixedClient(), train=train, eval_rows=ev)
    base.update(kw)
    return InfoGate(**base)


@patch("ssdataagent.strategies.design_b.load_schema", side_effect=lambda n: toy_schema())
@patch("ssdataagent.strategies.elicitation.DatasetSchema", DatasetSchema)
def test_design_b_full_calibrates_to_known_marginal(_ls, tmp_path):
    s = toy_schema()
    # train marginal heavily favors A (0.8/0.2)
    train = pd.DataFrame({"region": ["N", "S"] * 50,
                          "vote": (["A"] * 80 + ["B"] * 20)})
    ev = pd.DataFrame({"region": ["N", "S"] * 50})
    gate = _gate(Condition.FULL, train, ev)
    cfg = type("Cfg", (), {"results_root": tmp_path})()
    res = DesignBStrategy().generate(gate, tmp_path, cfg)
    assert len(res.generated) == 100
    assert set(res.generated["vote"].unique()).issubset({"A", "B"})
    # elicited [0.5,0.5] raked toward the train marginal 0.8 -> majority A
    share_A = (res.generated["vote"] == "A").mean()
    assert share_A > 0.6
    assert res.meta_extras["backend"] == "design_b"


@patch("ssdataagent.strategies.design_b.load_schema", side_effect=lambda n: toy_schema())
def test_design_b_aggregate_condition_no_microdata(_ls, tmp_path):
    s = toy_schema()
    train = pd.DataFrame({"region": ["N", "S"] * 50, "vote": (["A"] * 60 + ["B"] * 40)})
    ev = pd.DataFrame({"region": ["N", "S"] * 50})
    gate = _gate(Condition.NO_DATA, train, ev)   # fit_microdata() is None; known_marginals from train
    cfg = type("Cfg", (), {"results_root": tmp_path})()
    res = DesignBStrategy().generate(gate, tmp_path, cfg)   # must NOT raise
    assert len(res.generated) == 100
    assert (tmp_path / "fit_summary.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategy_design_b.py -v`
Expected: FAIL (`ImportError: DesignBStrategy`)

- [ ] **Step 3: Add the strategy to `src/ssdataagent/strategies/design_b.py`**

```python
import json
from pathlib import Path

import pandas as pd

from ssdataagent.data import cells
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.baselines import background_frame, clip_decode
from ssdataagent.strategies.base import InfoGate, StrategyResult

_N_DEMO_BINS = 4
_N_NUMERIC_BINS = 10


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
```

Add `from ssdataagent.agent.context import Condition` to the imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategy_design_b.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the design_b + foundation tests together**

Run: `.venv/bin/pytest tests/test_design_b_rake.py tests/test_design_b_sample.py tests/test_strategy_design_b.py tests/test_elicitation.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/strategies/design_b.py tests/test_strategy_design_b.py
git commit -m "design_b: DesignBStrategy orchestration (elicit -> rake -> couple -> sample)"
```

---

### Task 7: Register strategy + add A/B/C condition specs

**Files:**
- Modify: `src/ssdataagent/strategies/registry.py`
- Modify: `src/ssdataagent/experiments/conditions.py`
- Test: `tests/test_strategies_registry.py` (extend), `tests/test_conditions.py` (extend)

**Interfaces:**
- Consumes: `DesignBStrategy` (Task 6); `ConditionSpec`, `Condition`.
- Produces: registry key `"design_b"`; conditions `design_b_full`/`design_b_aggregate`/`design_b_transfer`.

- [ ] **Step 1: Write the failing tests** (append)

```python
# tests/test_strategies_registry.py  (append)
from ssdataagent.strategies.design_b import DesignBStrategy


def test_get_strategy_returns_design_b():
    assert isinstance(get_strategy("design_b"), DesignBStrategy)
```

```python
# tests/test_conditions.py  (append)
def test_design_b_conditions_registered():
    assert get_condition("design_b_full").context_condition is Condition.FULL
    assert get_condition("design_b_aggregate").context_condition is Condition.NO_DATA
    assert get_condition("design_b_transfer").context_condition is Condition.TRANSFER
    for n in ("design_b_full", "design_b_aggregate", "design_b_transfer"):
        assert get_condition(n).strategy == "design_b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py -v`
Expected: FAIL (`KeyError: 'design_b'`)

- [ ] **Step 3: Implement**

In `src/ssdataagent/strategies/registry.py` add the import + entry:

```python
from ssdataagent.strategies.design_b import DesignBStrategy
```
```python
    "design_b": DesignBStrategy,
```

In `src/ssdataagent/experiments/conditions.py` add to `CONDITIONS`:

```python
    "design_b_full": ConditionSpec("design_b_full", Condition.FULL, strategy="design_b"),
    "design_b_aggregate": ConditionSpec("design_b_aggregate", Condition.NO_DATA, strategy="design_b"),
    "design_b_transfer": ConditionSpec("design_b_transfer", Condition.TRANSFER, strategy="design_b"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/registry.py src/ssdataagent/experiments/conditions.py tests/test_strategies_registry.py tests/test_conditions.py
git commit -m "design_b: register strategy + A/B/C condition specs"
```

---

### Task 8: Runner transfer-gate construction + characterization net

**Files:**
- Modify: `src/ssdataagent/experiments/runner.py` (`_run_one_condition`)
- Test: `tests/test_runner_artifacts.py` (extend)

**Interfaces:**
- Consumes: `TRANSFER_PAIRS`, `load_source_wave`, `compute_crosswalk` (3a `data/transfer.py`); `load_schema`; `Condition`.
- Produces: when a condition's `context_condition is Condition.TRANSFER`, the runner builds the `InfoGate` with `source`/`source_name`/`crosswalk` populated.

- [ ] **Step 1: Write the failing test** (append to `tests/test_runner_artifacts.py`)

```python
def _fake_design_b_generate(self, gate, run_dir, cfg):
    # assert the runner built a transfer gate, then emit a trivial frame
    from ssdataagent.strategies.base import StrategyResult
    import pandas as pd
    assert gate.source is not None and len(gate.crosswalk) > 0
    out = pd.DataFrame({"profile_id": range(len(gate.background()))})
    return StrategyResult(generated=out, meta_extras={"backend": "design_b"})


@patch("ssdataagent.strategies.design_b.DesignBStrategy.generate", _fake_design_b_generate)
@patch("ssdataagent.experiments.runner._git_sha", return_value="testsha")
@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_transfer_condition_builds_source_gate(_cfg, _client, _eval, _sha, tmp_path):
    _cfg.return_value = MagicMock(model="m1", provider="p1")
    cfg = ExperimentConfig(
        name="dbexp", datasets=["gss"], conditions=["design_b_transfer"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    run_dir = _only_run_dir(tmp_path / "dbexp" / "design_b_transfer" / "gss")
    meta = json.loads(_read(run_dir, "meta.json"))
    assert meta["backend"] == "design_b"
    assert json.loads(_read(run_dir, "eval.json"))  # eval.json written
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_runner_artifacts.py::test_transfer_condition_builds_source_gate -v`
Expected: FAIL (`gate.source is None` — runner doesn't build the transfer gate yet)

- [ ] **Step 3: Implement the transfer branch in `_run_one_condition`**

In `src/ssdataagent/experiments/runner.py`, add imports:

```python
from ssdataagent.agent.context import Condition
from ssdataagent.data.transfer import TRANSFER_PAIRS, compute_crosswalk, load_source_wave
```

In `_run_one_condition`, replace the `gate = InfoGate(...)` construction with a transfer-aware build:

```python
    if spec.context_condition is Condition.TRANSFER and dataset in TRANSFER_PAIRS:
        source_name = TRANSFER_PAIRS[dataset]
        source_df = load_source_wave(source_name)
        crosswalk = compute_crosswalk(
            load_schema(dataset), load_schema(source_name), source_df, eval_df,
        )
        gate = InfoGate(
            condition=spec.context_condition, dataset_name=dataset, workspace=workspace,
            client=client, train=train, eval_rows=eval_df,
            unseen_variables=tuple(cfg.unseen_variables.get(dataset, [])),
            source=source_df, source_name=source_name, crosswalk=tuple(crosswalk),
        )
    else:
        gate = InfoGate(
            condition=spec.context_condition, dataset_name=dataset, workspace=workspace,
            client=client, train=train, eval_rows=eval_df,
            unseen_variables=tuple(cfg.unseen_variables.get(dataset, [])),
        )
```

(The `else` branch is the existing construction verbatim — existing conditions are unaffected.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_runner_artifacts.py -v`
Expected: PASS (the 2 P0 characterization tests remain byte-stable + the new transfer test)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS except the 4 pre-existing failures (`tests/test_config.py::test_unknown_provider_raises` + 3 `tests/test_ssdatabench_integration.py *_legacy`). No NEW failures.

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/experiments/runner.py tests/test_runner_artifacts.py
git commit -m "runner: build transfer InfoGate (source wave + crosswalk) for transfer conditions"
```

---

## Self-Review

**Spec coverage:**
- §1 module layout → copula.py (T1), cells.py (T2), elicitation.py (T3), design_b.py (T4-6), registry/conditions (T7), runner (T8). ✓
- §2 cell partition shared util + metric refactor → T2. ✓
- §3 elicitation (support-vector prompt, parse/renormalize, retries→fallback, persistent cache, raw-IO logging) → T3. ✓
- §4 pipeline (reference data, cells, elicit, rake, couple, sample, assemble, StrategyResult, fit_summary) → T4 (rake) + T5 (couple+sample) + T6 (orchestration). ✓
- Locked decisions: LLM=marginals + copula data-grounded/identity-in-C → T5 `build_target_copula`; uniform prob-vector-over-support → T3 `target_support`/`known_vector` + T5 sampling; cache-based determinism → T3. ✓
- §5 conditions + runner transfer construction → T7 + T8. ✓
- §6 testing (cells, copula module, elicitation incl. cache-hit-no-call, design_b raking/leakage/determinism, registry/conditions, runner transfer + char-net byte-stable) → tests across T1-T8. ✓
- Out-of-scope: no background-conditioned copula (only `correlated_normal` added, no `conditional_gaussian_sample` — documented refinement); known_associations not consumed; no model-slot split; no scorer/dashboard change. ✓

**Placeholder scan:** No TBD/TODO; complete code in every code step; commands have expected output.

**Type consistency:** `target_support`/`known_vector` (T3) signatures match their use in T6. `rake(cell_vectors, cell_weights, known_vec)` (T4) matches T6's call. `build_target_copula(ref, targets, schema)` / `sample_targets(eval_cell_keys, calibrated, supports, Sigma, targets, *, seed)` (T5) match T6. `copula.build_cuts/latent_matrix/make_pd/invert/correlated_normal/EPS` (T1) match uses in T5 and baselines. `cells.fit_scheme/assign` (T2) match T6 + the metric refactor. `DesignBStrategy.name == "design_b"` matches the registry key (T7) and condition specs (T7). `InfoGate(... source=, source_name=, crosswalk=)` (T8) matches the 3a field names.
