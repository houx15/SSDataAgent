# Full InfoGate + Transfer Wiring (Part 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared foundation for all grounded designs — the full `InfoGate` (A/B/C semantics, `known_marginals`/`known_associations`, transfer source) plus condition-B transfer data wiring (source waves + variable crosswalk).

**Architecture:** Two new pure data-layer modules (`data/aggregates.py` for marginal/association computations, `data/transfer.py` for source-wave loading + crosswalk), a new `Condition.TRANSFER` enum value, and an extended `InfoGate` whose methods are the single budget-enforcement point per A/B/C. No strategy and no runner wiring — those land in Part 3b. Everything is deterministic and LLM-free.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy (all installed). No new dependencies.

## Global Constraints

- **No new dependencies.**
- **No strategy, no runner wiring, no condition specs in this part** — 3a delivers mechanisms only; Part 3b adds the `design_b` strategy + condition specs + runner wiring that consume them.
- **Backward-compatible:** existing strategies (agent, direct, Part 2 baselines) and existing `InfoGate` construction must keep working unchanged. New `InfoGate` fields are defaulted.
- **`data/aggregates.py` is self-contained** — it must NOT import from `ssdataagent.strategies.*` (that would create a `data → strategies → base → data` import cycle). It re-derives the numeric/categorical split inline from `schema.numeric_ranges`. (This is a deliberate refinement of spec §2's "reuse classify_columns" note; behavior — numerical iff in `schema.numeric_ranges`, else categorical — is identical.)
- **Budget enforcement (the gate is the single point of truth):** A=`FULL` → `fit_microdata`=train, aggregates from train; B=`TRANSFER` → `fit_microdata`=source(crosswalk cols), aggregates from source, NEVER target targets; C=`NO_DATA` → `fit_microdata`=None, aggregates from train (rows withheld); `DIRECT` → all None.
- **Transfer pairs:** `TRANSFER_PAIRS = {"gss": "gss1994", "cps": "cps1970"}` (source = earlier wave; target = the scored wave).
- **No target leakage in B:** a transfer gate's `fit_microdata()` returns only crosswalk columns of the **source** frame.
- **Determinism:** all functions are pure/deterministic; no randomness. Entropy/stat helpers tolerate NaN per-pair and never raise on degenerate input.
- Test runner: `.venv/bin/pytest`. Pre-existing failures NOT in scope (do not fix): `tests/test_config.py::test_unknown_provider_raises` + 3 `tests/test_ssdatabench_integration.py *_legacy` (missing `autograd`). Gate = our tests pass + no NEW failures.
- Follow repo style: `from __future__ import annotations`, module-level functions.

---

### Task 1: Aggregate computations (`data/aggregates.py`)

**Files:**
- Create: `src/ssdataagent/data/aggregates.py`
- Test: `tests/test_aggregates.py`

**Interfaces:**
- Consumes: `ssdataagent.data.schema.DatasetSchema` (`numeric_ranges`, `allowed_values`); `scipy.stats.chi2_contingency`.
- Produces (used by Task 3): `marginals(df, variables, schema, *, n_bins=10) -> dict[str, dict]`; `associations(df, target_variables, schema) -> dict[str, dict[str, float]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregates.py
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.data.aggregates import marginals, associations


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"],
        target_variables=["vote", "income", "age"],
        descriptions={},
        allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"income": (0.0, 200.0), "age": (18.0, 90.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_marginals_categorical_normalizes_over_allowed():
    s = toy_schema()
    df = pd.DataFrame({"vote": ["A", "A", "B"]})  # no C present
    m = marginals(df, ["vote"], s)
    assert m["vote"]["kind"] == "categorical"
    probs = m["vote"]["probs"]
    assert set(probs) == {"A", "B", "C"}          # missing C present at 0
    assert abs(probs["A"] - 2 / 3) < 1e-9 and probs["C"] == 0.0
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_marginals_numeric_quantiles_and_moments():
    s = toy_schema()
    df = pd.DataFrame({"income": [0.0, 50.0, 100.0, 150.0, 200.0]})
    m = marginals(df, ["income"], s, n_bins=4)
    assert m["income"]["kind"] == "numeric"
    assert m["income"]["quantiles"][1.0] == 200.0 and m["income"]["quantiles"][0.0] == 0.0
    assert abs(m["income"]["mean"] - 100.0) < 1e-9


def test_associations_perfect_and_independent():
    s = toy_schema()
    # vote perfectly determined by region -> Cramer's V ~ 1
    perfect = pd.DataFrame({"region": ["N", "N", "S", "S"] * 10,
                            "vote": ["A", "A", "B", "B"] * 10})
    a = associations(perfect, ["region", "vote"], s)
    assert a["region"]["vote"] > 0.9
    assert a["vote"]["region"] == a["region"]["vote"]  # symmetric
    # region independent of vote -> ~0
    indep = pd.DataFrame({"region": (["N", "S"] * 20),
                          "vote": (["A", "B"] * 20)})
    indep["vote"] = (["A", "A", "B", "B"] * 10)  # uncorrelated with alternating region
    a2 = associations(indep, ["region", "vote"], s)
    assert a2.get("region", {}).get("vote", 0.0) < 0.4


def test_associations_mixed_and_numnum_and_degenerate():
    s = toy_schema()
    df = pd.DataFrame({"region": ["N", "S"] * 20,
                       "income": list(np.linspace(0, 200, 40)),
                       "age": list(np.linspace(18, 90, 40))})
    a = associations(df, ["region", "income", "age"], s)
    assert 0.0 <= a["region"]["income"] <= 1.0      # cat x num -> eta
    assert a["income"]["age"] > 0.9                 # num x num -> |r| (both linear)
    # degenerate: constant column produces no entry, no raise
    dgn = pd.DataFrame({"vote": ["A"] * 10, "income": [5.0] * 10})
    a3 = associations(dgn, ["vote", "income"], s)
    assert a3 == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_aggregates.py -v`
Expected: FAIL with `ModuleNotFoundError: ssdataagent.data.aggregates`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/data/aggregates.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ssdataagent.data.schema import DatasetSchema


def _is_numeric(schema: DatasetSchema, col: str) -> bool:
    return col in schema.numeric_ranges


def marginals(df: pd.DataFrame, variables, schema: DatasetSchema, *, n_bins: int = 10) -> dict:
    """Univariate marginal per variable.
    Categorical -> {"kind": "categorical", "probs": {value: prob}} over allowed_values
      (missing categories at 0.0; normalized over non-null rows).
    Numeric -> {"kind": "numeric", "quantiles": {q: value}, "mean": float, "std": float}."""
    out: dict[str, dict] = {}
    for v in variables:
        if v not in df.columns:
            continue
        col = df[v].dropna()
        if _is_numeric(schema, v):
            x = pd.to_numeric(col, errors="coerce").dropna()
            if len(x) == 0:
                out[v] = {"kind": "numeric", "quantiles": {}, "mean": None, "std": None}
                continue
            qs = {round(float(q), 3): float(np.quantile(x, q))
                  for q in np.linspace(0, 1, n_bins + 1)}
            out[v] = {"kind": "numeric", "quantiles": qs,
                      "mean": float(x.mean()), "std": float(x.std(ddof=0))}
        else:
            cats = schema.allowed_values.get(v) or sorted(col.unique().tolist())
            counts = col.value_counts()
            total = float(counts.sum()) or 1.0
            out[v] = {"kind": "categorical",
                      "probs": {str(c): float(counts.get(c, 0)) / total for c in cats}}
    return out


def _cramers_v(a: pd.Series, b: pd.Series):
    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = float(tab.to_numpy().sum())
    denom = n * (min(tab.shape) - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(chi2 / denom))


def _corr_ratio(categories: pd.Series, values: pd.Series):
    vals = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"c": categories.to_numpy(), "x": vals.to_numpy()}).dropna()
    if frame["c"].nunique() < 2 or len(frame) < 2:
        return None
    grand = frame["x"].mean()
    ss_total = float(((frame["x"] - grand) ** 2).sum())
    if ss_total <= 0:
        return None
    ss_between = float(sum(len(g) * (g["x"].mean() - grand) ** 2
                          for _, g in frame.groupby("c")))
    return float(np.sqrt(ss_between / ss_total))


def associations(df: pd.DataFrame, target_variables, schema: DatasetSchema) -> dict:
    """Symmetric pairwise association among target variables in [0,1]:
    cat x cat -> Cramer's V; num x num -> |Pearson r|; cat x num -> correlation ratio eta.
    Degenerate/uncomputable pairs are omitted (never raises)."""
    out: dict[str, dict[str, float]] = {}
    tv = [v for v in target_variables if v in df.columns]
    for i, a in enumerate(tv):
        for b in tv[i + 1:]:
            sub = df[[a, b]].dropna()
            an, bn = _is_numeric(schema, a), _is_numeric(schema, b)
            if len(sub) < 2:
                val = None
            elif an and bn:
                x = pd.to_numeric(sub[a], errors="coerce")
                y = pd.to_numeric(sub[b], errors="coerce")
                val = None if x.std(ddof=0) == 0 or y.std(ddof=0) == 0 \
                    else float(abs(np.corrcoef(x, y)[0, 1]))
            elif not an and not bn:
                val = _cramers_v(sub[a], sub[b])
            else:
                cat, num = (a, b) if not an else (b, a)
                val = _corr_ratio(sub[cat], sub[num])
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            out.setdefault(a, {})[b] = val
            out.setdefault(b, {})[a] = val
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_aggregates.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/data/aggregates.py tests/test_aggregates.py
git commit -m "data: marginal + pairwise-association aggregate computations"
```

---

### Task 2: Transfer mechanics + source-wave dataset entries (`data/transfer.py`)

**Files:**
- Create: `src/ssdataagent/data/transfer.py`
- Modify: `config/datasets.yaml`
- Test: `tests/test_transfer.py`, `tests/test_datasets_source_waves.py`

**Interfaces:**
- Consumes: `ssdataagent.data.loader.load_real_data`; `ssdataagent.data.schema.DatasetSchema`/`load_schema`.
- Produces (used by Part 3b): `TRANSFER_PAIRS: dict[str, str]`; `load_source_wave(source_name) -> pd.DataFrame`; `compute_crosswalk(target_schema, source_schema, source_df, target_df) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer.py
from pathlib import Path

import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.data.transfer import TRANSFER_PAIRS, compute_crosswalk


def _schema(bg, tgt, name="s") -> DatasetSchema:
    return DatasetSchema(
        name=name, real_data_path=Path("/nonexistent.csv"),
        background_variables=bg, target_variables=tgt,
        descriptions={}, allowed_values={}, numeric_ranges={},
        population_context="", ssdatabench_sim_subdir="x",
        evaluation_script="x.py", domains={},
    )


def test_transfer_pairs_mapping():
    assert TRANSFER_PAIRS["gss"] == "gss1994"
    assert TRANSFER_PAIRS["cps"] == "cps1970"


def test_compute_crosswalk_intersection_and_order():
    target = _schema(["age", "region"], ["vote", "income", "wealth"])
    source = _schema(["age", "region"], ["vote", "income"])  # source lacks 'wealth'
    target_df = pd.DataFrame(columns=["age", "region", "vote", "income", "wealth"])
    source_df = pd.DataFrame(columns=["age", "region", "vote", "income"])  # lacks wealth col
    cw = compute_crosswalk(target, source, source_df, target_df)
    assert cw == ["age", "region", "vote", "income"]      # target-schema order, wealth dropped


def test_compute_crosswalk_excludes_column_absent_in_a_frame():
    target = _schema(["age"], ["vote"])
    source = _schema(["age"], ["vote"])
    target_df = pd.DataFrame(columns=["age", "vote"])
    source_df = pd.DataFrame(columns=["age"])              # 'vote' column missing in source
    assert compute_crosswalk(target, source, source_df, target_df) == ["age"]
```

```python
# tests/test_datasets_source_waves.py
import pytest

from ssdataagent.data.loader import load_real_data
from ssdataagent.data.schema import load_schema
from ssdataagent.data.transfer import compute_crosswalk, load_source_wave


def test_source_wave_schemas_resolve():
    assert load_schema("gss1994").target_variables  # resolves, non-empty
    assert load_schema("cps1970").target_variables


def test_gss_crosswalk_is_substantial():
    cw = compute_crosswalk(load_schema("gss"), load_schema("gss1994"),
                           load_source_wave("gss1994"), load_real_data("gss"))
    # observed ~25 common variables across GSS 1994/2018; floor well below that.
    assert len(cw) >= 18


def test_cps_crosswalk_is_substantial():
    cw = compute_crosswalk(load_schema("cps"), load_schema("cps1970"),
                           load_source_wave("cps1970"), load_real_data("cps"))
    # observed 12 common variables across CPS-ASEC 1970/1980.
    assert len(cw) >= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_transfer.py tests/test_datasets_source_waves.py -v`
Expected: FAIL (`ModuleNotFoundError: ssdataagent.data.transfer`; later `KeyError: unknown dataset 'gss1994'`)

- [ ] **Step 3a: Add source-wave entries to `config/datasets.yaml`**

Insert these two entries under the `datasets:` map (e.g. after the `acs` entry). Source entries reuse the target wave's `ssdatabench_yaml`/sim-subdir/eval-script — they are loaded only as fitting microdata, never scored:

```yaml
  # Transfer sources (Part 3a) — earlier waves, loaded as condition-B source
  # microdata only; never scored. Reuse the target wave's ssdatabench_yaml since
  # GSS/CPS variable names are stable across waves (verified: gss1994 shares ~25
  # vars with gss2018; cps-asec1970 shares all 12 with cps1980).
  gss1994:
    real_data_path: real_data/gss/gss1994.csv
    ssdatabench_yaml: ssdatabench/real_data/data_configs/gss2018.yaml
    ssdatabench_sim_subdir: gss_2018
    evaluation_script: scripts/evaluation/gss_2018.py
    type: cross-sectional
  cps1970:
    real_data_path: real_data/cps/cps-asec1970.csv
    ssdatabench_yaml: ssdatabench/real_data/data_configs/cps1980.yaml
    ssdatabench_sim_subdir: cps_1980
    evaluation_script: scripts/evaluation/cps_1980.py
    type: cross-sectional
```

- [ ] **Step 3b: Write `data/transfer.py`**

```python
# src/ssdataagent/data/transfer.py
from __future__ import annotations

import logging

import pandas as pd

from ssdataagent.data.loader import load_real_data
from ssdataagent.data.schema import DatasetSchema

log = logging.getLogger(__name__)

# target dataset name -> source dataset name (source = earlier wave).
TRANSFER_PAIRS: dict[str, str] = {"gss": "gss1994", "cps": "cps1970"}


def load_source_wave(source_name: str) -> pd.DataFrame:
    """Load a source wave's cleaned CSV as fitting microdata (full wave)."""
    return load_real_data(source_name)


def compute_crosswalk(
    target_schema: DatasetSchema,
    source_schema: DatasetSchema,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> list[str]:
    """Variables usable for transfer: (background + target) vars present in BOTH
    schemas AND as columns in BOTH frames, ordered by the target schema."""
    candidate = list(target_schema.background_variables) + list(target_schema.target_variables)
    src_vars = set(source_schema.background_variables) | set(source_schema.target_variables)
    common = [v for v in candidate
              if v in src_vars and v in source_df.columns and v in target_df.columns]
    dropped = [v for v in candidate if v not in common]
    log.info("crosswalk: %d common variables (dropped %d: %s)",
             len(common), len(dropped), dropped)
    return common
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_transfer.py tests/test_datasets_source_waves.py -v`
Expected: PASS (3 + 3 tests). If a real-data crosswalk test fails because the floor is wrong, print `len(cw)` and adjust the floor to just below the observed count (and update the comment) — but a count far below ~25 (GSS) / 12 (CPS) means a schema/column mismatch to investigate, not a floor to lower.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/data/transfer.py config/datasets.yaml tests/test_transfer.py tests/test_datasets_source_waves.py
git commit -m "data: transfer pairs, source-wave loading, variable crosswalk"
```

---

### Task 3: `Condition.TRANSFER` + InfoGate extension (`strategies/base.py`)

**Files:**
- Modify: `src/ssdataagent/agent/context.py` (add `Condition.TRANSFER`)
- Modify: `src/ssdataagent/strategies/base.py` (InfoGate fields + methods)
- Test: `tests/test_info_gate_transfer.py`

**Interfaces:**
- Consumes: `marginals`, `associations` (Task 1); `Condition` (`agent/context.py`); `load_schema` (`data/schema.py`).
- Produces (used by Part 3b): `InfoGate` with `source`/`source_name`/`crosswalk` fields, `fit_microdata()` TRANSFER branch, `known_marginals()`, `known_associations()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_info_gate_transfer.py
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.base import InfoGate


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["vote", "income"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def _gate(condition, **kw):
    train = pd.DataFrame({"age": [20.0, 40.0], "region": ["N", "S"],
                          "vote": ["A", "B"], "income": [10.0, 90.0]})
    ev = pd.DataFrame({"age": [30.0], "region": ["N"], "vote": ["A"], "income": [50.0]})
    base = dict(condition=condition, dataset_name="toy", workspace=Path("/tmp"),
                client=None, train=train, eval_rows=ev)
    base.update(kw)
    return InfoGate(**base)


@patch("ssdataagent.strategies.base.load_schema", side_effect=lambda n: toy_schema())
def test_fit_microdata_per_condition(_ls):
    assert _gate(Condition.FULL).fit_microdata() is not None
    assert _gate(Condition.NO_DATA).fit_microdata() is None
    assert _gate(Condition.DIRECT).fit_microdata() is None
    src = pd.DataFrame({"age": [25.0], "region": ["S"], "vote": ["B"], "income": [70.0]})
    g = _gate(Condition.TRANSFER, source=src, source_name="toy_src",
              crosswalk=("age", "region", "vote"))
    fm = g.fit_microdata()
    assert list(fm.columns) == ["age", "region", "vote"]   # crosswalk cols only
    assert "income" not in fm.columns                       # non-crosswalk target excluded


@patch("ssdataagent.strategies.base.load_schema", side_effect=lambda n: toy_schema())
def test_known_marginals_and_associations_sources(_ls):
    # A/C compute from train; DIRECT -> None
    assert _gate(Condition.FULL).known_marginals() is not None
    assert _gate(Condition.NO_DATA).known_marginals() is not None   # C: aggregates w/o rows
    assert _gate(Condition.NO_DATA).known_associations() is not None
    assert _gate(Condition.DIRECT).known_marginals() is None
    assert _gate(Condition.DIRECT).known_associations() is None
    # B: from source, and a target var absent from crosswalk is not in the marginals
    src = pd.DataFrame({"age": [25.0, 30.0], "region": ["S", "N"], "vote": ["B", "A"],
                        "income": [70.0, 20.0]})
    g = _gate(Condition.TRANSFER, source=src, crosswalk=("age", "region", "vote"))
    km = g.known_marginals()
    assert "vote" in km and "income" not in km   # only crosswalk targets


@patch("ssdataagent.strategies.base.load_schema", side_effect=lambda n: toy_schema())
def test_transfer_no_target_leakage(_ls):
    # source rows are the ONLY microdata exposed; target eval targets never appear
    src = pd.DataFrame({"age": [25.0], "region": ["S"], "vote": ["B"], "income": [70.0]})
    g = _gate(Condition.TRANSFER, source=src, crosswalk=("age", "region", "vote"))
    fm = g.fit_microdata()
    assert fm.equals(src[["age", "region", "vote"]])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_info_gate_transfer.py -v`
Expected: FAIL (`AttributeError: TRANSFER` / `InfoGate` has no `known_marginals`)

- [ ] **Step 3a: Add the enum value** in `src/ssdataagent/agent/context.py`

```python
class Condition(str, Enum):
    FULL = "full_agent"
    NO_SEMANTIC = "agent_no_semantic"
    NO_DATA = "agent_no_data"
    UNSEEN = "full_agent_unseen"
    DIRECT = "direct_generation"
    TRANSFER = "transfer"
```

- [ ] **Step 3b: Extend `InfoGate`** in `src/ssdataagent/strategies/base.py`

Add the imports near the top (module level — no cycle: `base → aggregates → schema`):

```python
from ssdataagent.data.aggregates import associations, marginals
from ssdataagent.data.schema import load_schema
```

Add the three fields to the `InfoGate` dataclass (after `unseen_variables`):

```python
    source: pd.DataFrame | None = None
    source_name: str | None = None
    crosswalk: tuple[str, ...] = ()
```

Replace `fit_microdata` and add the new methods:

```python
    def fit_microdata(self) -> pd.DataFrame | None:
        """Microdata a strategy may fit on. Source (crosswalk cols) under
        TRANSFER; train under FULL/NO_SEMANTIC/UNSEEN; None under NO_DATA/DIRECT."""
        if self.condition is Condition.TRANSFER:
            return None if self.source is None else self.source[list(self.crosswalk)]
        if self.condition in (Condition.FULL, Condition.NO_SEMANTIC, Condition.UNSEEN):
            return self.train
        return None

    def _reference_microdata(self) -> pd.DataFrame | None:
        """Frame the aggregates are computed from: source for TRANSFER, train for
        FULL/NO_SEMANTIC/UNSEEN/NO_DATA, None for DIRECT."""
        if self.condition is Condition.DIRECT:
            return None
        if self.condition is Condition.TRANSFER:
            return None if self.source is None else self.source[list(self.crosswalk)]
        return self.train

    def known_marginals(self) -> dict | None:
        ref = self._reference_microdata()
        if ref is None:
            return None
        schema = load_schema(self.dataset_name)
        targets = [t for t in schema.target_variables if t in ref.columns]
        return marginals(ref, targets, schema)

    def known_associations(self) -> dict | None:
        ref = self._reference_microdata()
        if ref is None:
            return None
        schema = load_schema(self.dataset_name)
        targets = [t for t in schema.target_variables if t in ref.columns]
        return associations(ref, targets, schema)
```

(Note: the existing `fit_microdata` only handled `FULL/NO_SEMANTIC/UNSEEN`; the replacement adds the `TRANSFER` branch and is otherwise behavior-identical for existing conditions. Keep the existing docstring style.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_info_gate_transfer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the existing gate/strategy tests to confirm backward compatibility**

Run: `.venv/bin/pytest tests/test_info_gate.py tests/test_strategy_hotdeck.py tests/test_strategies_registry.py -v`
Expected: PASS (existing `InfoGate`/strategy behavior unchanged)

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS except the 4 pre-existing failures (`tests/test_config.py::test_unknown_provider_raises` + 3 `tests/test_ssdatabench_integration.py *_legacy`). No NEW failures.

- [ ] **Step 7: Commit**

```bash
git add src/ssdataagent/agent/context.py src/ssdataagent/strategies/base.py tests/test_info_gate_transfer.py
git commit -m "infogate: Condition.TRANSFER + known_marginals/associations + transfer source"
```

---

## Self-Review

**Spec coverage:**
- §1 module layout → Tasks 1-3 create/modify exactly `data/aggregates.py`, `data/transfer.py`, `base.py`, `agent/context.py`, `datasets.yaml`. ✓
- §2 aggregates (marginals + associations, mixed types) → Task 1 (self-contained per Global Constraints, a documented refinement of the spec's classify_columns note). ✓
- §3 transfer (TRANSFER_PAIRS, load_source_wave, compute_crosswalk + logged coverage) → Task 2. ✓
- §4 InfoGate extension (fields, fit_microdata TRANSFER, known_marginals/associations, budget table) → Task 3; the budget table is realized by `fit_microdata` + `_reference_microdata`. ✓
- §5 Condition.TRANSFER + datasets.yaml source entries → Task 3 (enum) + Task 2 (yaml). ✓
- §6 scope boundary (no strategy/specs/runner) → honored; no condition specs or runner edits in any task. ✓
- §7 testing (aggregates, info_gate_transfer incl. leakage, transfer crosswalk, real-data smoke) → Tasks 1-3 tests. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; commands have expected output. The Task 2 floor-adjustment note is a conditional empirical check with a concrete fallback, not a placeholder.

**Type consistency:** `marginals(df, variables, schema, *, n_bins=10)` and `associations(df, target_variables, schema)` are identical in Task 1 (definition) and Task 3 (call from the gate). `compute_crosswalk(target_schema, source_schema, source_df, target_df)` consistent across Task 2 def and its tests. `InfoGate` field names (`source`, `source_name`, `crosswalk`) consistent between Task 3 definition and the Task 3 tests. `Condition.TRANSFER` value `"transfer"` used consistently.
