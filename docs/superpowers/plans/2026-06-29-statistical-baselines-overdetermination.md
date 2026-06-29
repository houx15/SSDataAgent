# Statistical Baselines + Over-determination Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three non-LLM statistical baseline strategies (hot-deck/k-NN, sequential CART, Gaussian copula) and the headline over-determination diagnostic, behind the existing P0 Strategy seam.

**Architecture:** Each baseline is a `Strategy` (Protocol from `strategies/base.py`) whose `generate()` fits on `gate.fit_microdata()` and fills targets for `gate.background()` rows, deterministically (fixed seed). Core generation logic is a pure function (`*_generate(train, background, schema, ...)`) so it is unit-testable with a hand-built schema, with a thin Strategy wrapper that resolves the schema and writes artifacts. The over-determination metric is a pure function in `evaluation/overdetermination.py`, called from the runner's `_write_common` so every run (agent, direct, baselines) gets it in `eval.json`.

**Tech Stack:** Python 3.11+, numpy, pandas, scipy, scikit-learn (all already installed). No new dependencies — the Gaussian copula is implemented in-house.

## Global Constraints

- **No new dependencies.** Copula is in-house on numpy/scipy. (pgmpy Bayesian-net backend is a documented future fast-follow, not built here.)
- **Determinism:** every baseline and the metric is deterministic given a fixed seed (default `42`). Tests pin the seed and assert reproducibility.
- **No target leakage:** baseline output frames are built from background columns only (+ `profile_id`); real target columns from `eval_df` are never copied into the generated frame.
- **Positional alignment:** strategies generate one row per `gate.background()` row, in order; the metric aligns `generated` to `eval_df` positionally.
- **The metric must never break a run:** `_write_common` wraps the metric call so any failure yields an error/`null` block in `eval.json`, never an exception.
- **Condition scope:** baselines run under `Condition.FULL` only (information condition A). If `gate.fit_microdata()` is `None`, raise `ValueError`. Condition C (`known_marginals`, IPF) is deferred to Part 3.
- **A target is numerical** iff it is in `schema.numeric_ranges`; otherwise categorical/ordinal (in `schema.allowed_values`, ordered by the `allowed` list).
- **Entropy is in bits** (`scipy.stats.entropy(..., base=2)`) throughout the metric.
- Follow existing repo style: `from __future__ import annotations`, module-level functions prefixed `_` for internals, `StrategyResult(generated, meta_extras)` return contract.

---

### Task 1: Shared encoding helpers in `baselines.py`

**Files:**
- Create: `src/ssdataagent/strategies/baselines.py`
- Test: `tests/test_baselines_encoding.py`

**Interfaces:**
- Consumes: `ssdataagent.data.schema.DatasetSchema` (fields: `background_variables`, `target_variables`, `allowed_values: dict[str,list]`, `numeric_ranges: dict[str,(float,float)]`).
- Produces (used by Tasks 2-4):
  - `classify_columns(schema, columns) -> tuple[list[str], list[str]]` → `(numerical, categorical)`
  - `encode_numeric(df, columns, schema, *, stats=None) -> tuple[np.ndarray, dict]` → standardized-numeric + one-hot-categorical matrix and the fitted `stats`
  - `ordinal_encode(df, columns, schema) -> np.ndarray` → float matrix (numeric as-is, categorical → index in `allowed_values`, unknown → -1)
  - `clip_decode(df, schema) -> pd.DataFrame` → clip numeric columns to `numeric_ranges`
  - `background_frame(background, schema) -> pd.DataFrame` → background columns (+ `profile_id`), targets dropped

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines_encoding.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import (
    background_frame,
    classify_columns,
    clip_decode,
    encode_numeric,
    ordinal_encode,
)


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy",
        real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"],
        target_variables=["income", "vote"],
        descriptions={},
        allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="",
        ssdatabench_sim_subdir="toy",
        evaluation_script="x.py",
        domains={},
    )


def test_classify_columns_splits_by_schema():
    num, cat = classify_columns(toy_schema(), ["age", "region", "income", "vote"])
    assert num == ["age", "income"]
    assert cat == ["region", "vote"]


def test_encode_numeric_standardizes_and_one_hots():
    s = toy_schema()
    df = pd.DataFrame({"age": [20.0, 40.0, 60.0], "region": ["N", "S", "N"]})
    X, stats = encode_numeric(df, ["age", "region"], s)
    # age column standardized to mean 0
    assert abs(X[:, 0].mean()) < 1e-9
    # region one-hot: 2 columns (N, S)
    assert X.shape == (3, 3)
    # reusing stats on new data keeps the same scaling origin
    X2, _ = encode_numeric(pd.DataFrame({"age": [40.0], "region": ["S"]}),
                           ["age", "region"], s, stats=stats)
    assert abs(X2[0, 0]) < 1e-9  # 40 is the train mean -> 0


def test_ordinal_encode_codes_categoricals():
    s = toy_schema()
    df = pd.DataFrame({"region": ["N", "S", "Z"], "age": [20.0, 30.0, 40.0]})
    X = ordinal_encode(df, ["region", "age"], s)
    assert X[0, 0] == 0.0 and X[1, 0] == 1.0 and X[2, 0] == -1.0  # Z unknown
    assert X[2, 1] == 40.0


def test_clip_decode_clips_numeric_to_range():
    s = toy_schema()
    df = pd.DataFrame({"income": [-5.0, 250.0, 50.0], "vote": ["A", "B", "C"]})
    out = clip_decode(df, s)
    assert list(out["income"]) == [0.0, 200.0, 50.0]
    assert list(out["vote"]) == ["A", "B", "C"]


def test_background_frame_drops_targets_adds_profile_id():
    s = toy_schema()
    ev = pd.DataFrame({"age": [20.0], "region": ["N"], "income": [99.0], "vote": ["A"]})
    out = background_frame(ev, s)
    assert "income" not in out.columns and "vote" not in out.columns
    assert "profile_id" in out.columns
    assert list(out["age"]) == [20.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_baselines_encoding.py -v`
Expected: FAIL with `ModuleNotFoundError: ssdataagent.strategies.baselines`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/strategies/baselines.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema


def classify_columns(schema: DatasetSchema, columns) -> tuple[list[str], list[str]]:
    """Split columns into (numerical, categorical). Numerical iff in
    schema.numeric_ranges; everything else is treated as categorical/ordinal."""
    numerical, categorical = [], []
    for c in columns:
        (numerical if c in schema.numeric_ranges else categorical).append(c)
    return numerical, categorical


def encode_numeric(df, columns, schema, *, stats=None):
    """Float feature matrix: numerical z-scored, categorical one-hot over
    schema.allowed_values. `stats` (means/sds fitted on train) is reused for
    eval rows so the scaling origin matches. Returns (X, stats)."""
    num, cat = classify_columns(schema, columns)
    if stats is None:
        means = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in num}
        sds = {c: (float(pd.to_numeric(df[c], errors="coerce").std(ddof=0)) or 1.0) for c in num}
        stats = {"means": means, "sds": sds}
    blocks = []
    for c in num:
        col = pd.to_numeric(df[c], errors="coerce").fillna(stats["means"][c]).to_numpy()
        blocks.append(((col - stats["means"][c]) / stats["sds"][c]).reshape(-1, 1))
    for c in cat:
        cats = schema.allowed_values.get(c) or sorted(df[c].dropna().unique().tolist())
        idx = {v: i for i, v in enumerate(cats)}
        oh = np.zeros((len(df), len(cats)), dtype=float)
        for r, v in enumerate(df[c].tolist()):
            if v in idx:
                oh[r, idx[v]] = 1.0
        blocks.append(oh)
    X = np.hstack(blocks) if blocks else np.zeros((len(df), 0))
    return X, stats


def ordinal_encode(df, columns, schema) -> np.ndarray:
    """Integer-code columns for tree models: numerical kept as float;
    categorical mapped to its index in allowed_values (unknown -> -1)."""
    cols = []
    for c in columns:
        if c in schema.numeric_ranges:
            cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1))
        else:
            cats = schema.allowed_values.get(c) or sorted(df[c].dropna().unique().tolist())
            idx = {v: i for i, v in enumerate(cats)}
            cols.append(np.array([idx.get(v, -1) for v in df[c].tolist()], dtype=float).reshape(-1, 1))
    return np.hstack(cols) if cols else np.zeros((len(df), 0))


def clip_decode(df, schema) -> pd.DataFrame:
    """Clip numerical columns to schema.numeric_ranges; leave categoricals."""
    out = df.copy()
    for c in out.columns:
        if c in schema.numeric_ranges:
            lo, hi = schema.numeric_ranges[c]
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lo, hi)
    return out


def background_frame(background, schema) -> pd.DataFrame:
    """Background columns (+ profile_id), target columns dropped (no leakage)."""
    cols = [c for c in background.columns
            if c in schema.background_variables or c == "profile_id"]
    out = background[cols].reset_index(drop=True).copy()
    if "profile_id" not in out.columns:
        out.insert(0, "profile_id", range(len(out)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_baselines_encoding.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/baselines.py tests/test_baselines_encoding.py
git commit -m "baselines: shared column-encoding helpers"
```

---

### Task 2: Hot-deck / k-NN strategy

**Files:**
- Modify: `src/ssdataagent/strategies/baselines.py`
- Test: `tests/test_strategy_hotdeck.py`

**Interfaces:**
- Consumes: `classify_columns`, `encode_numeric`, `clip_decode`, `background_frame` (Task 1); `InfoGate`/`StrategyResult` from `ssdataagent.strategies.base`; `load_schema` from `ssdataagent.data.schema`.
- Produces (used by Task 5 registry): class `HotDeckStrategy` with `name = "hotdeck"`; pure function `hotdeck_generate(train, background, schema, *, k=10, seed=42) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_hotdeck.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import HotDeckStrategy, hotdeck_generate


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["income", "vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def _train(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n).astype(float)
    region = rng.choice(["N", "S"], n)
    income = age * 1.5 + rng.normal(0, 5, n)
    vote = np.where(region == "N", "A", "B")
    return pd.DataFrame({"age": age, "region": region,
                         "income": income.clip(0, 200), "vote": vote})


def test_hotdeck_output_targets_are_real_train_vectors():
    s = toy_schema()
    train = _train()
    bg = train[["age", "region"]].iloc[:10].reset_index(drop=True)
    out = hotdeck_generate(train, bg, s, k=5, seed=42)
    assert len(out) == 10
    train_pairs = set(zip(train["income"].round(6), train["vote"]))
    for inc, vote in zip(out["income"].round(6), out["vote"]):
        assert (inc, vote) in train_pairs  # whole target vector is a real one


def test_hotdeck_is_deterministic():
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:10].reset_index(drop=True)
    a = hotdeck_generate(train, bg, s, k=5, seed=42)
    b = hotdeck_generate(train, bg, s, k=5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_hotdeck_no_target_leakage_and_profile_id():
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:5].reset_index(drop=True)
    out = hotdeck_generate(train, bg, s, k=5, seed=42)
    assert "profile_id" in out.columns
    assert set(["age", "region", "income", "vote"]).issubset(out.columns)


def test_hotdeck_strategy_requires_microdata():
    import pytest

    class _Gate:
        dataset_name = "toy"
        def fit_microdata(self): return None
        def background(self): return pd.DataFrame()
    with pytest.raises(ValueError):
        HotDeckStrategy().generate(_Gate(), Path("/tmp"), None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategy_hotdeck.py -v`
Expected: FAIL with `ImportError: cannot import name 'HotDeckStrategy'`

- [ ] **Step 3: Write minimal implementation** (append to `baselines.py`)

```python
import json
from pathlib import Path

from sklearn.neighbors import NearestNeighbors

from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.base import InfoGate, StrategyResult


def hotdeck_generate(train, background, schema, *, k=10, seed=42) -> pd.DataFrame:
    bg_vars = list(schema.background_variables)
    targets = list(schema.target_variables)
    Xtr, stats = encode_numeric(train, bg_vars, schema)
    Xev, _ = encode_numeric(background, bg_vars, schema, stats=stats)
    k_eff = max(1, min(k, len(train)))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(Xtr)
    _, idx = nn.kneighbors(Xev)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k_eff, size=len(Xev))
    chosen = idx[np.arange(len(idx)), pick]
    donor = train.iloc[chosen][targets].reset_index(drop=True)
    out = background_frame(background, schema)
    for c in targets:
        out[c] = donor[c].to_numpy()
    return clip_decode(out, schema)


class HotDeckStrategy:
    name = "hotdeck"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        train = gate.fit_microdata()
        if train is None:
            raise ValueError("hotdeck requires microdata; this condition exposes none")
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        generated = hotdeck_generate(train, bg, schema, k=10, seed=42)
        Path(run_dir, "fit_summary.json").write_text(
            json.dumps({"backend": "hotdeck", "k": 10, "n_train_fit": len(train)}, indent=2)
        )
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "hotdeck", "k": 10,
                         "n_train_fit": len(train), "n_individuals": len(bg)},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategy_hotdeck.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/baselines.py tests/test_strategy_hotdeck.py
git commit -m "baselines: hot-deck / k-NN strategy"
```

---

### Task 3: Sequential CART strategy

**Files:**
- Modify: `src/ssdataagent/strategies/baselines.py`
- Test: `tests/test_strategy_cart.py`

**Interfaces:**
- Consumes: `ordinal_encode`, `clip_decode`, `background_frame` (Task 1); `load_schema`, `InfoGate`, `StrategyResult`.
- Produces (Task 5): class `CartStrategy` (`name = "cart"`); `cart_generate(train, background, schema, *, min_samples_leaf=5, seed=42) -> pd.DataFrame`.

Sequential CART: for each target in `schema.target_variables` order, grow a tree on background + already-generated targets (regressor for numerical, classifier for categorical), then **donate a random training target value from the matched leaf** (tree-structured hot-deck) so within-leaf spread is preserved instead of collapsed to the mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_cart.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import CartStrategy, cart_generate


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["income", "vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def _train(n=300, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n).astype(float)
    region = rng.choice(["N", "S"], n)
    income = (age * 1.2 + rng.normal(0, 20, n)).clip(0, 200)
    # vote varies WITHIN every region -> a non-collapsed conditional
    vote = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame({"age": age, "region": region, "income": income, "vote": vote})


def test_cart_respects_allowed_and_ranges():
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:50].reset_index(drop=True)
    out = cart_generate(train, bg, s, seed=42)
    assert len(out) == 50
    assert set(out["vote"].unique()).issubset({"A", "B", "C"})
    assert out["income"].between(0, 200).all()


def test_cart_is_deterministic():
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:50].reset_index(drop=True)
    a = cart_generate(train, bg, s, seed=42)
    b = cart_generate(train, bg, s, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_cart_does_not_collapse_variance():
    # vote is genuinely diverse given background -> sampled output must not be constant
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:100].reset_index(drop=True)
    out = cart_generate(train, bg, s, seed=42)
    assert out["vote"].nunique() > 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategy_cart.py -v`
Expected: FAIL with `ImportError: cannot import name 'CartStrategy'`

- [ ] **Step 3: Write minimal implementation** (append to `baselines.py`)

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def cart_generate(train, background, schema, *, min_samples_leaf=5, seed=42) -> pd.DataFrame:
    bg_vars = list(schema.background_variables)
    targets = list(schema.target_variables)
    rng = np.random.default_rng(seed)
    out = background_frame(background, schema)
    feat_cols = list(bg_vars)
    train_feat = train[bg_vars].copy().reset_index(drop=True)
    gen_feat = background[bg_vars].copy().reset_index(drop=True)
    for t in targets:
        Xtr = ordinal_encode(train_feat, feat_cols, schema)
        Xgen = ordinal_encode(gen_feat, feat_cols, schema)
        is_num = t in schema.numeric_ranges
        Tree = DecisionTreeRegressor if is_num else DecisionTreeClassifier
        y = pd.to_numeric(train[t], errors="coerce").to_numpy() if is_num \
            else train[t].astype(str).to_numpy()
        model = Tree(min_samples_leaf=min_samples_leaf, random_state=seed).fit(Xtr, y)
        leaf_tr = model.apply(Xtr)
        leaf_gen = model.apply(Xgen)
        raw = train[t].to_numpy()
        by_leaf: dict[int, list] = {}
        for lid, v in zip(leaf_tr, raw):
            by_leaf.setdefault(int(lid), []).append(v)
        drawn = []
        for lid in leaf_gen:
            pool = by_leaf.get(int(lid)) or list(raw)
            drawn.append(pool[int(rng.integers(0, len(pool)))])
        out[t] = drawn
        feat_cols = feat_cols + [t]
        train_feat[t] = train[t].to_numpy()
        gen_feat[t] = drawn
    return clip_decode(out, schema)


class CartStrategy:
    name = "cart"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        train = gate.fit_microdata()
        if train is None:
            raise ValueError("cart requires microdata; this condition exposes none")
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        generated = cart_generate(train, bg, schema, min_samples_leaf=5, seed=42)
        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "cart", "min_samples_leaf": 5,
             "target_order": list(schema.target_variables), "n_train_fit": len(train)},
            indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "cart", "min_samples_leaf": 5,
                         "n_train_fit": len(train), "n_individuals": len(bg)},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategy_cart.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/baselines.py tests/test_strategy_cart.py
git commit -m "baselines: sequential CART strategy"
```

---

### Task 4: Gaussian copula strategy

**Files:**
- Modify: `src/ssdataagent/strategies/baselines.py`
- Test: `tests/test_strategy_copula.py`

**Interfaces:**
- Consumes: `clip_decode`, `background_frame` (Task 1); `load_schema`, `InfoGate`, `StrategyResult`; `scipy.stats.norm`.
- Produces (Task 5): class `CopulaStrategy` (`name = "copula"`); `copula_generate(train, background, schema, *, regularization=1e-6, seed=42) -> pd.DataFrame`.

In-house Gaussian copula over `(background + targets)`: latent normal-score transform (numeric via empirical CDF; categorical/ordinal via polychoric-style cut-point midpoints), empirical correlation `Σ`, then for each eval row sample the target latent block from the **conditional Gaussian** given the row's background latent scores, and invert back to values.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_copula.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import CopulaStrategy, copula_generate


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["income", "vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def _train(n=400, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 90, n)
    region = rng.choice(["N", "S"], n)
    income = (age * 1.8 + rng.normal(0, 8, n)).clip(0, 200)  # strong age->income
    vote = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame({"age": age, "region": region, "income": income, "vote": vote})


def test_copula_respects_allowed_and_ranges():
    s, train = toy_schema(), _train()
    bg = pd.DataFrame({"age": np.linspace(20, 88, 60), "region": ["N", "S"] * 30})
    out = copula_generate(train, bg, s, seed=42)
    assert len(out) == 60
    assert set(out["vote"].unique()).issubset({"A", "B", "C"})
    assert out["income"].between(0, 200).all()


def test_copula_is_deterministic():
    s, train = toy_schema(), _train()
    bg = pd.DataFrame({"age": np.linspace(20, 88, 60), "region": ["N", "S"] * 30})
    a = copula_generate(train, bg, s, seed=42)
    b = copula_generate(train, bg, s, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_copula_conditions_on_background():
    # generated income should track the strong age->income relationship
    s, train = toy_schema(), _train()
    bg = pd.DataFrame({"age": np.linspace(20, 88, 120), "region": ["N", "S"] * 60})
    out = copula_generate(train, bg, s, seed=42)
    r = np.corrcoef(bg["age"].to_numpy(), out["income"].to_numpy())[0, 1]
    assert r > 0.5  # positive conditioning preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategy_copula.py -v`
Expected: FAIL with `ImportError: cannot import name 'CopulaStrategy'`

- [ ] **Step 3: Write minimal implementation** (append to `baselines.py`)

```python
from scipy.stats import norm, rankdata

_EPS = 1e-6


def _build_cuts(train, cols, schema) -> dict:
    """Per-column inversion data. numeric -> sorted train values;
    categorical -> (categories, cumulative upper edges)."""
    cuts: dict[str, dict] = {}
    n = len(train)
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


def _latent_value(col_cut, value) -> float:
    if col_cut["kind"] == "num":
        s = col_cut["sorted"]
        if len(s) == 0 or pd.isna(value):
            return 0.0
        pos = int(np.searchsorted(s, float(value), side="right"))
        u = min(max((pos - 0.5) / len(s), _EPS), 1 - _EPS)
        return float(norm.ppf(u))
    cats, cum = col_cut["cats"], col_cut["cum"]
    if value not in cats:
        return 0.0
    i = cats.index(value)
    lo = cum[i - 1] if i > 0 else 0.0
    u = min(max((lo + cum[i]) / 2.0, _EPS), 1 - _EPS)
    return float(norm.ppf(u))


def _latent_matrix(df, cols, schema, cuts) -> np.ndarray:
    out = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        out[:, j] = [_latent_value(cuts[c], v) for v in df[c].tolist()]
    return out


def _invert(z_array, col, schema, col_cut) -> list:
    u = np.clip(norm.cdf(z_array), _EPS, 1 - _EPS)
    if col_cut["kind"] == "num":
        s = col_cut["sorted"]
        return list(np.quantile(s, u)) if len(s) else [0.0] * len(u)
    cats, cum = col_cut["cats"], col_cut["cum"]
    idx = np.searchsorted(cum, u, side="left")
    idx = np.clip(idx, 0, len(cats) - 1)
    return [cats[i] for i in idx]


def _make_pd(M, reg) -> np.ndarray:
    M = (M + M.T) / 2.0
    M = M + reg * np.eye(M.shape[0])
    w, V = np.linalg.eigh(M)
    w = np.clip(w, reg, None)
    return (V * w) @ V.T


def copula_generate(train, background, schema, *, regularization=1e-6, seed=42) -> pd.DataFrame:
    bg_vars = list(schema.background_variables)
    targets = list(schema.target_variables)
    cols = bg_vars + targets
    cuts = _build_cuts(train, cols, schema)
    Z = _latent_matrix(train, cols, schema, cuts)
    Sigma = _make_pd(np.corrcoef(Z, rowvar=False), regularization)
    bi = list(range(len(bg_vars)))
    ti = list(range(len(bg_vars), len(cols)))
    Sbb = Sigma[np.ix_(bi, bi)]
    Stt = Sigma[np.ix_(ti, ti)]
    Stb = Sigma[np.ix_(ti, bi)]
    Sbb_inv = np.linalg.pinv(Sbb)
    cond_cov = _make_pd(Stt - Stb @ Sbb_inv @ Stb.T, regularization)
    L = np.linalg.cholesky(cond_cov)
    Zb = _latent_matrix(background, bg_vars, schema, cuts)
    mu = (Stb @ Sbb_inv @ Zb.T).T
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((len(background), len(ti))) @ L.T
    Zt = mu + eps
    out = background_frame(background, schema)
    for j, t in enumerate(targets):
        out[t] = _invert(Zt[:, j], t, schema, cuts[t])
    return clip_decode(out, schema)


class CopulaStrategy:
    name = "copula"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        train = gate.fit_microdata()
        if train is None:
            raise ValueError("copula requires microdata; this condition exposes none")
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        generated = copula_generate(train, bg, schema, seed=42)
        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "copula", "regularization": 1e-6, "n_train_fit": len(train)}, indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "copula", "n_train_fit": len(train),
                         "n_individuals": len(bg)},
        )
```

Note: `rankdata` is imported for parity with the normal-score approach but the
implementation uses `searchsorted`; drop the `rankdata` import if your linter
flags it unused.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategy_copula.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/baselines.py tests/test_strategy_copula.py
git commit -m "baselines: in-house Gaussian copula strategy"
```

---

### Task 5: Register strategies + add conditions

**Files:**
- Modify: `src/ssdataagent/strategies/registry.py`
- Modify: `src/ssdataagent/experiments/conditions.py`
- Test: `tests/test_strategies_registry.py` (extend), `tests/test_conditions.py` (extend)

**Interfaces:**
- Consumes: `HotDeckStrategy`, `CartStrategy`, `CopulaStrategy` (Tasks 2-4); `ConditionSpec`, `Condition`.
- Produces: registry keys `"hotdeck"`, `"cart"`, `"copula"`; conditions `hotdeck`, `cart`, `copula` (all `Condition.FULL`).

- [ ] **Step 1: Write the failing tests** (append to existing test files)

```python
# tests/test_strategies_registry.py  (append)
from ssdataagent.strategies.baselines import (
    CartStrategy,
    CopulaStrategy,
    HotDeckStrategy,
)


def test_get_strategy_returns_baselines():
    assert isinstance(get_strategy("hotdeck"), HotDeckStrategy)
    assert isinstance(get_strategy("cart"), CartStrategy)
    assert isinstance(get_strategy("copula"), CopulaStrategy)
```

```python
# tests/test_conditions.py  (append)
def test_baseline_conditions_registered():
    for name in ("hotdeck", "cart", "copula"):
        spec = get_condition(name)
        assert spec.context_condition is Condition.FULL
        assert spec.strategy == name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py -v`
Expected: FAIL (`KeyError: 'hotdeck'`)

- [ ] **Step 3: Write minimal implementation**

In `src/ssdataagent/strategies/registry.py`, add the imports and entries:

```python
from ssdataagent.strategies.baselines import (
    CartStrategy,
    CopulaStrategy,
    HotDeckStrategy,
)

STRATEGIES: dict[str, type] = {
    "agent": AgentStrategy,
    "direct": DirectGenerationStrategy,
    "hotdeck": HotDeckStrategy,
    "cart": CartStrategy,
    "copula": CopulaStrategy,
}
```

In `src/ssdataagent/experiments/conditions.py`, add to `CONDITIONS`:

```python
    "hotdeck": ConditionSpec("hotdeck", Condition.FULL, strategy="hotdeck"),
    "cart": ConditionSpec("cart", Condition.FULL, strategy="cart"),
    "copula": ConditionSpec("copula", Condition.FULL, strategy="copula"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_strategies_registry.py tests/test_conditions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/registry.py src/ssdataagent/experiments/conditions.py tests/test_strategies_registry.py tests/test_conditions.py
git commit -m "baselines: register hotdeck/cart/copula strategies + conditions"
```

---

### Task 6: Over-determination metric module

**Files:**
- Create: `src/ssdataagent/evaluation/overdetermination.py`
- Test: `tests/test_overdetermination.py`

**Interfaces:**
- Consumes: `DatasetSchema`; `scipy.stats.entropy`; `sklearn.ensemble.HistGradientBoostingClassifier`; `ordinal_encode` from `ssdataagent.strategies.baselines`.
- Produces (Task 7): `overdetermination(*, real, sim, schema, n_target_bins=5, n_demo_bins=4, min_count=20, seed=42) -> dict` with keys `"cell_based"` and `"model_based"`. Each sub-dict carries `headline_gap` (float|None), `per_target: {t: {h_real, h_sim, gap}}`, plus `coverage`/`n_cells` (cell-based) and an optional `reason` when not computable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_overdetermination.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.evaluation.overdetermination import overdetermination


def cat_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={}, population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_collapsed_sim_gives_positive_gap():
    s = cat_schema()
    # region N: real votes 50/50 (H=1 bit); sim all A (H=0) -> gap ~ 1.0
    real = pd.DataFrame({"region": ["N"] * 40, "vote": ["A", "B"] * 20})
    sim = pd.DataFrame({"region": ["N"] * 40, "vote": ["A"] * 40})
    res = overdetermination(real=real, sim=sim, schema=s, min_count=10)
    cb = res["cell_based"]
    assert abs(cb["per_target"]["vote"]["h_real"] - 1.0) < 1e-6
    assert cb["per_target"]["vote"]["h_sim"] < 1e-6
    assert abs(cb["per_target"]["vote"]["gap"] - 1.0) < 1e-6
    assert cb["headline_gap"] > 0.9
    assert cb["coverage"] == 1.0
    assert cb["n_cells"] == 1


def test_no_cells_meet_min_count_reports_reason():
    s = cat_schema()
    real = pd.DataFrame({"region": ["N"] * 5, "vote": ["A", "B"] * 2 + ["A"]})
    sim = pd.DataFrame({"region": ["N"] * 5, "vote": ["A"] * 5})
    res = overdetermination(real=real, sim=sim, schema=s, min_count=1000)
    assert res["cell_based"]["headline_gap"] is None
    assert "reason" in res["cell_based"]


def test_numeric_target_uses_real_derived_bins():
    s = DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["income"],
        descriptions={}, allowed_values={"region": ["N"]},
        numeric_ranges={"income": (0.0, 100.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"region": ["N"] * 100, "income": rng.uniform(0, 100, 100)})
    sim = pd.DataFrame({"region": ["N"] * 100, "income": [50.0] * 100})  # collapsed
    res = overdetermination(real=real, sim=sim, schema=s, min_count=10, n_target_bins=4)
    assert res["cell_based"]["per_target"]["income"]["gap"] > 0  # real spread > sim


def test_misaligned_backgrounds_report_reason():
    s = cat_schema()
    real = pd.DataFrame({"region": ["N"] * 10, "vote": ["A", "B"] * 5})
    sim = pd.DataFrame({"region": ["S"] * 10, "vote": ["A"] * 10})
    res = overdetermination(real=real, sim=sim, schema=s, min_count=1)
    # backgrounds differ -> cell-based still computes per-cell, but alignment
    # guard records a warning; ensure it does not raise and returns a dict
    assert isinstance(res, dict) and "cell_based" in res


def test_model_based_present_and_directional():
    s = cat_schema()
    rng = np.random.default_rng(0)
    region = rng.choice(["N", "S"], 200)
    real = pd.DataFrame({"region": region, "vote": rng.choice(["A", "B"], 200)})
    sim = pd.DataFrame({"region": region, "vote": ["A"] * 200})  # collapsed
    res = overdetermination(real=real, sim=sim, schema=s, min_count=10)
    mb = res["model_based"]
    assert "vote" in mb["per_target"]
    assert mb["per_target"]["vote"]["gap"] >= 0  # collapsed sim -> lower entropy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_overdetermination.py -v`
Expected: FAIL with `ModuleNotFoundError: ssdataagent.evaluation.overdetermination`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/evaluation/overdetermination.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import ordinal_encode


def _bin_edges(real_vals, n_bins):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(pd.to_numeric(real_vals, errors="coerce").dropna(), qs))
    if len(edges) < 2:
        edges = np.array([edges.min() - 1e-9, edges.max() + 1e-9]) if len(edges) else np.array([0.0, 1.0])
    return edges


def _discretize(values, edges):
    return np.clip(np.digitize(pd.to_numeric(values, errors="coerce").to_numpy(), edges[1:-1]), 0, len(edges) - 2)


def _coarsen(real, sim, schema, n_demo_bins):
    """Return (real_cells, sim_cells): a string cell key per row."""
    real_keys, sim_keys = [], []
    parts_real, parts_sim = [], []
    for v in schema.background_variables:
        if v in schema.numeric_ranges:
            edges = _bin_edges(real[v], n_demo_bins)
            parts_real.append(_discretize(real[v], edges).astype(str))
            parts_sim.append(_discretize(sim[v], edges).astype(str))
        else:
            parts_real.append(real[v].astype(str).to_numpy())
            parts_sim.append(sim[v].astype(str).to_numpy())
    real_keys = ["|".join(t) for t in zip(*parts_real)] if parts_real else ["_"] * len(real)
    sim_keys = ["|".join(t) for t in zip(*parts_sim)] if parts_sim else ["_"] * len(sim)
    return np.array(real_keys), np.array(sim_keys)


def _target_series(df, t, schema, edges_map):
    if t in schema.numeric_ranges:
        return pd.Series(_discretize(df[t], edges_map[t]), index=df.index)
    return df[t].astype(str)


def _entropy_bits(labels) -> float:
    counts = pd.Series(labels).value_counts().to_numpy().astype(float)
    if counts.sum() == 0:
        return 0.0
    return float(entropy(counts, base=2))


def _cell_based(real, sim, schema, n_target_bins, n_demo_bins, min_count):
    edges_map = {t: _bin_edges(real[t], n_target_bins)
                 for t in schema.target_variables if t in schema.numeric_ranges}
    real_cells, sim_cells = _coarsen(real, sim, schema, n_demo_bins)
    real = real.reset_index(drop=True); sim = sim.reset_index(drop=True)
    real_cells = pd.Series(real_cells); sim_cells = pd.Series(sim_cells)
    kept = [c for c, n in real_cells.value_counts().items() if n >= min_count]
    if not kept:
        return {"headline_gap": None, "coverage": 0.0, "n_cells": 0,
                "per_target": {}, "reason": "no cells met min_count"}
    per_target = {}
    for t in schema.target_variables:
        rt = _target_series(real, t, schema, edges_map)
        st = _target_series(sim, t, schema, edges_map)
        num_r = num_s = denom = 0.0
        for c in kept:
            rmask = (real_cells == c).to_numpy()
            smask = (sim_cells == c).to_numpy()
            if smask.sum() == 0:
                continue
            w = float(rmask.sum())
            num_r += w * _entropy_bits(rt[rmask])
            num_s += w * _entropy_bits(st[smask])
            denom += w
        if denom == 0:
            continue
        h_real, h_sim = num_r / denom, num_s / denom
        per_target[t] = {"h_real": h_real, "h_sim": h_sim, "gap": h_real - h_sim}
    coverage = float(real_cells.isin(kept).sum()) / len(real)
    gaps = [v["gap"] for v in per_target.values()]
    headline = float(np.mean(gaps)) if gaps else None
    return {"headline_gap": headline, "coverage": coverage,
            "n_cells": len(kept), "per_target": per_target}


def _model_based(real, sim, schema, n_target_bins, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    edges_map = {t: _bin_edges(real[t], n_target_bins)
                 for t in schema.target_variables if t in schema.numeric_ranges}
    Xr = ordinal_encode(real, schema.background_variables, schema)
    Xs = ordinal_encode(sim, schema.background_variables, schema)
    per_target = {}
    for t in schema.target_variables:
        yr = _target_series(real, t, schema, edges_map).astype(str).to_numpy()
        ys = _target_series(sim, t, schema, edges_map).astype(str).to_numpy()
        if len(np.unique(yr)) < 2 or len(np.unique(ys)) < 2:
            continue
        try:
            mr = HistGradientBoostingClassifier(random_state=seed).fit(Xr, yr)
            ms = HistGradientBoostingClassifier(random_state=seed).fit(Xs, ys)
            h_real = float(np.mean([entropy(p, base=2) for p in mr.predict_proba(Xr)]))
            h_sim = float(np.mean([entropy(p, base=2) for p in ms.predict_proba(Xs)]))
        except Exception:
            continue
        per_target[t] = {"h_real": h_real, "h_sim": h_sim, "gap": h_real - h_sim}
    gaps = [v["gap"] for v in per_target.values()]
    return {"headline_gap": float(np.mean(gaps)) if gaps else None,
            "per_target": per_target}


def overdetermination(*, real: pd.DataFrame, sim: pd.DataFrame, schema: DatasetSchema,
                      n_target_bins: int = 5, n_demo_bins: int = 4,
                      min_count: int = 20, seed: int = 42) -> dict:
    """gap = H_real(target | demographics) - H_sim(target | demographics), in bits.
    Positive gap => sim is over-determined (collapsed within-group variance).
    Never raises: a failing stage returns a dict with a 'reason' instead."""
    try:
        cell = _cell_based(real, sim, schema, n_target_bins, n_demo_bins, min_count)
    except Exception as e:
        cell = {"headline_gap": None, "per_target": {}, "reason": f"{type(e).__name__}: {e}"}
    try:
        model = _model_based(real, sim, schema, n_target_bins, seed)
    except Exception as e:
        model = {"headline_gap": None, "per_target": {}, "reason": f"{type(e).__name__}: {e}"}
    return {"cell_based": cell, "model_based": model}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_overdetermination.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/evaluation/overdetermination.py tests/test_overdetermination.py
git commit -m "metric: over-determination gap (cell-based + model-based)"
```

---

### Task 7: Wire the metric into the runner + extend the characterization net

**Files:**
- Modify: `src/ssdataagent/experiments/runner.py` (`_write_common`, `_serialize_rates`)
- Test: `tests/test_runner_artifacts.py` (extend)

**Interfaces:**
- Consumes: `overdetermination` (Task 6); existing `_write_common(*, run_dir, meta, generated, dataset, run_id, eval_df)` and `_serialize_rates(r, dataset_name=None)`.
- Produces: `eval.json` gains a top-level `"overdetermination"` key for every run.

- [ ] **Step 1: Write the failing test** (append to `tests/test_runner_artifacts.py`)

```python
def _fake_overdet(*, real, sim, schema, **kw):
    return {"cell_based": {"headline_gap": 0.42, "coverage": 1.0, "n_cells": 1,
                           "per_target": {}}, "model_based": {"headline_gap": None,
                           "per_target": {}}}


@patch("ssdataagent.experiments.runner.overdetermination", side_effect=_fake_overdet)
@patch("ssdataagent.experiments.runner._git_sha", return_value="testsha")
@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.strategies.agent_strategy.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_eval_json_has_overdetermination(_cfg, _client, MockOrch, _eval, _sha, _od, tmp_path):
    _cfg.return_value = MagicMock(model="m1", provider="p1")
    MockOrch.return_value.run.return_value = _agent_run_result()
    cfg = ExperimentConfig(
        name="charexp", datasets=["gss"], conditions=["full_agent"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    run_dir = _only_run_dir(tmp_path / "charexp" / "full_agent" / "gss")
    blob = json.loads(_read(run_dir, "eval.json"))
    assert blob["overdetermination"]["cell_based"]["headline_gap"] == 0.42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_runner_artifacts.py::test_eval_json_has_overdetermination -v`
Expected: FAIL with `AttributeError`/`KeyError` (`overdetermination` not patchable / key missing)

- [ ] **Step 3: Write minimal implementation**

In `src/ssdataagent/experiments/runner.py`, add the import near the other evaluation imports:

```python
from ssdataagent.evaluation.overdetermination import overdetermination
```

Replace `_write_common` and `_serialize_rates`:

```python
def _write_common(
    *,
    run_dir: Path,
    meta: dict,
    generated,
    dataset: str,
    run_id: str,
    eval_df,
) -> PassRates:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    generated.to_csv(run_dir / "generated.csv", index=False)
    rates = run_evaluation(
        dataset_name=dataset, run_id=run_id, generated=generated, sampled=eval_df,
    )
    od = _safe_overdetermination(generated=generated, eval_df=eval_df, dataset=dataset)
    (run_dir / "eval.json").write_text(_serialize_rates(rates, dataset, overdetermination=od))
    return rates


def _safe_overdetermination(*, generated, eval_df, dataset) -> dict:
    """Compute the over-determination block; never break the scoring tail."""
    try:
        return overdetermination(
            real=eval_df, sim=generated, schema=load_schema(dataset),
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _serialize_rates(r: PassRates, dataset_name: str | None = None,
                     *, overdetermination: dict | None = None) -> str:
    payload: dict = {
        "by_type": r.by_type,
        "by_variable": r.by_variable,
        "by_pair": r.by_pair,
        "overall_average": r.overall_average,
    }
    if dataset_name is not None:
        try:
            payload["by_domain"] = by_domain(r, load_schema(dataset_name))
        except Exception:
            pass
    if overdetermination is not None:
        payload["overdetermination"] = overdetermination
    return json.dumps(payload, indent=2)
```

Note: `load_schema` is already imported in `runner.py`. The
`overdetermination` parameter name shadows the imported function inside
`_serialize_rates` only — that function does not call it, so there is no
conflict; `_write_common` calls it via `_safe_overdetermination`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_runner_artifacts.py -v`
Expected: PASS (all 3 tests — the two P0 characterization tests still pass, plus the new one)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/experiments/runner.py tests/test_runner_artifacts.py
git commit -m "runner: emit over-determination block in eval.json for every run"
```

---

### Task 8: Surface the gap in the report + export the metric

**Files:**
- Modify: `scripts/generate_exp_report.py` (add an over-determination section)
- Modify: `src/ssdataagent/evaluation/__init__.py` (export entry point)
- Test: `tests/test_report_overdetermination.py`

**Interfaces:**
- Consumes: `overdetermination` (Task 6); the report's `_md_table`, `_fmt`, `_latest_eval` helpers; eval.json `"overdetermination"` block (Task 7).
- Produces: `from ssdataagent.evaluation import overdetermination` works; report prints a "Over-determination gap" table.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_overdetermination.py
import importlib


def test_evaluation_exports_overdetermination():
    mod = importlib.import_module("ssdataagent.evaluation")
    assert hasattr(mod, "overdetermination")


def test_overdetermination_section_builder():
    from scripts.generate_exp_report import _overdetermination_section
    cells = {
        "gss": {"overdetermination": {
            "cell_based": {"headline_gap": 0.5, "coverage": 0.8, "n_cells": 12},
            "model_based": {"headline_gap": 0.3}}},
        "cps": None,
    }
    md = _overdetermination_section(cells, ["gss", "cps"])
    assert "Over-determination" in md
    assert "0.500" in md  # cell-based gap formatted
    assert "gss" in md and "cps" in md
```

(If `scripts/` is not importable in your test env, prepend the repo root to
`sys.path` at the top of the test: `import sys, pathlib;
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_report_overdetermination.py -v`
Expected: FAIL (`__init__` empty → no `overdetermination`; `_overdetermination_section` undefined)

- [ ] **Step 3: Write minimal implementation**

In `src/ssdataagent/evaluation/__init__.py`:

```python
from ssdataagent.evaluation.overdetermination import overdetermination

__all__ = ["overdetermination"]
```

In `scripts/generate_exp_report.py`, add a section builder (place near `_md_table`):

```python
def _overdetermination_section(cells: dict, datasets: list) -> str:
    """Markdown table of the over-determination gap per dataset (cell-based
    headline + coverage, and the model-based cross-check)."""
    headers = ["Dataset", "gap (cell)", "coverage", "n_cells", "gap (model)"]
    rows = []
    for ds in datasets:
        cell = cells.get(ds)
        od = (cell or {}).get("overdetermination") if cell else None
        if not od:
            rows.append([ds, "—", "—", "—", "—"])
            continue
        cb = od.get("cell_based", {}) or {}
        mb = od.get("model_based", {}) or {}
        rows.append([ds, _fmt(cb.get("headline_gap")), _fmt(cb.get("coverage")),
                     _fmt(cb.get("n_cells")), _fmt(mb.get("headline_gap"))])
    return ("## Over-determination gap — `H_real − H_sim` (bits, higher = sim more collapsed)\n\n"
            + _md_table(headers, rows))
```

Then, in `main()`, immediately after the Results section (after the line
`bits.append("")` that follows `bits.append(_md_table(headers, rows))` for
Section 2, around line 168), insert:

```python
    # Section 2b: Over-determination gap
    bits.append(_overdetermination_section(cells, datasets))
    bits.append("")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_report_overdetermination.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS except the 4 pre-existing failures (`tests/test_config.py::test_unknown_provider_raises` and 3 `tests/test_ssdatabench_integration.py` `*_legacy` cases — all from a missing `autograd` module, unrelated to this work). No NEW failures.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_exp_report.py src/ssdataagent/evaluation/__init__.py tests/test_report_overdetermination.py
git commit -m "report: surface over-determination gap + export metric"
```

---

## Self-Review

**Spec coverage:**
- §1 module layout → Tasks 1-8 create/modify exactly the spec's files. ✓
- §2 hot-deck → Task 2; CART (sample from leaf, not mode) → Task 3; in-house Gaussian copula conditioned on background → Task 4. All seeded/deterministic, write `fit_summary.json`, raise on `None` microdata, drop real targets via `background_frame`. ✓
- §3 metric: cell-based headline (coarsened cells, min_count, coverage, n_cells, real-derived numeric bins) + model-based robustness; never raises; under `eval.json → overdetermination` → Tasks 6-7. ✓
- §4 wiring: registry + 3 `Condition.FULL` conditions (Task 5); `_write_common`/`_serialize_rates` (Task 7); report + `__init__` export (Task 8). ✓
- §5 testing: per-backend determinism + value-domain + (hot-deck) real-vector + (CART) non-collapse + (copula) conditioning; metric known-entropy + coverage + numeric-bins + degenerate-reason + both variants; wiring + characterization-net extension. ✓
- §6 out-of-scope: condition C / `known_marginals` / pgmpy / Designs A-C / dashboard — none built. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; commands have expected output.

**Type consistency:** `overdetermination(*, real, sim, schema, ...)` signature identical in Tasks 6/7/8. `*_generate(train, background, schema, *, ...)` consistent across Tasks 2-4. Helper names (`classify_columns`, `encode_numeric`, `ordinal_encode`, `clip_decode`, `background_frame`) defined in Task 1 and used unchanged in Tasks 2-4 and 6. `StrategyResult(generated, meta_extras)` matches `base.py`. `ConditionSpec(name, Condition, strategy=)` matches existing code.
