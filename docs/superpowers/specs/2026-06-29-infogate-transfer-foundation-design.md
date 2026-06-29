# Design — Part 3a: full InfoGate + transfer wiring (foundation)

**Date:** 2026-06-29
**Status:** Approved for planning
**Source plan:** `docs/handoff/delta-plan.md` (P2 prerequisites, condition↔A/B/C mapping) and `docs/handoff/design-reference.md` (§5 information conditions, §5 InfoGate sketch)
**Scope:** The first half of Part 3. It builds the shared **foundation** every grounded design (B/C/A) will consume — the full `InfoGate` (A/B/C semantics, `known_marginals`/`known_associations`, transfer source) plus the condition-B transfer data wiring (source waves + variable crosswalk). It adds **no strategy** and **no runner wiring** — those arrive in Part 3b alongside the Design B strategy.

---

## Why this exists

P0 introduced a *thin* `InfoGate` exposing only `background()` (eval rows) and
`fit_microdata()` (train, gated by condition). The A/B/C information semantics —
`known_marginals()`, `known_associations()`, `source_survey` transfer — were
explicitly deferred. The grounded designs (B/C/A) all need them, and condition B
(transfer) needs a real *source survey ≠ target survey* setup, which the current
single-wave data + held-out-variable `UNSEEN` condition does not provide.

Part 3a delivers that foundation as pure, deterministic, LLM-free data/gating
logic so it can be tested in isolation and reused by every later design. It is an
**additive extension**: new fields with defaults + new gate methods + two new
utility modules + two source-wave `datasets.yaml` entries. Existing strategies
(agent, direct, Part 2 baselines) are untouched and keep working — they only call
`fit_microdata()`/`background()`, whose `FULL/NO_SEMANTIC/UNSEEN/NO_DATA/DIRECT`
behavior is unchanged.

---

## Decomposition (full project, for context)

Each part is its own brainstorm → plan → implement cycle.

1. **Part 1 (P0)** — Strategy seam + thin InfoGate. **MERGED 2026-06-29.**
2. **Part 2 (P1)** — statistical baselines + over-determination metric. **MERGED 2026-06-29.**
3. **Part 3 (P2)** — Design B (marginals + copula). Split into:
   - **Part 3a** — full InfoGate + transfer wiring (foundation). ← *this spec*
   - **Part 3b** — the Design B strategy (LLM per-cell elicitation → rake → copula → sample), consuming 3a; adds condition specs + runner wiring.
4. **Part 4 (P3)** — Design C (retrieval + repair).
5. **Part 5 (P4)** — Design A (hierarchical Bayes).
6. **Part 6** — S1 distribution diagnostic.
7. **Part 7 (P5)** — local web console.

---

## Research decisions (settled at brainstorm, binding here)

- **Conditions:** A (in-distribution), B (transfer), C (aggregate-only) all in scope for Part 3.
- **Transfer pairs (B):** `GSS 1994 → 2018` and `CPS-ASEC 1970 → 1980`. Source = the earlier wave; target = the already-scored wave (`gss2018` / `cps_1980`). Within-survey temporal transfer (shared variable definitions by construction).
- **Condition-C known moments:** train-derived **univariate marginals** **plus** a small set of train-derived **pairwise associations**. (No external/published toplines.)

---

## Verified repo facts (read at design time)

- `strategies/base.py` — `InfoGate(condition, dataset_name, workspace, client, train, eval_rows, unseen_variables=())`; `background()→eval_rows`; `fit_microdata()→train` for `FULL/NO_SEMANTIC/UNSEEN`, else `None`. `Strategy` Protocol + `StrategyResult` also live here.
- `agent/context.py` — `class Condition(str, Enum)` with `FULL="full_agent"`, `NO_SEMANTIC="agent_no_semantic"`, `NO_DATA="agent_no_data"`, `UNSEEN="full_agent_unseen"`, `DIRECT="direct_generation"`. `build_context` computes `has_data = condition in (FULL, NO_SEMANTIC, UNSEEN)` — a new enum member it does not list is treated as no-data, which is correct (the agent never runs transfer).
- `data/schema.py` — `load_schema(name) → DatasetSchema(background_variables, target_variables, allowed_values: dict[str,list], numeric_ranges: dict[str,(float,float)], descriptions, domains, real_data_path, ...)`. A var is **numerical** iff in `numeric_ranges`, else **categorical/ordinal** (in `allowed_values`).
- `data/loader.py` — `load_real_data(name)` reads `schema.real_data_path`, drops `Unnamed: N` columns.
- `strategies/baselines.py` (Part 2) — `classify_columns(schema, cols) → (numerical, categorical)` is reusable for the aggregate computations.
- Data on disk: `real_data/gss/gss1994.csv` + `gss1994.yaml`, `real_data/gss/gss2018.csv`; `real_data/cps/cps-asec1970.csv` + `asec1970.yaml` (and 1980/1990/2000). The target waves' sampled CSVs (`real_data/used_dataset/sampled_gss.csv`, `samples_cps.csv`) back the existing `gss`/`cps` dataset entries.
- `config/datasets.yaml` — entries map a dataset name to `real_data_path`, `ssdatabench_yaml`, `ssdatabench_sim_subdir`, `evaluation_script`, `type`. Source waves need their own entries to be `load_schema`-able.

---

## §1 — Module layout

New files:
- `src/ssdataagent/data/aggregates.py` — `marginals(...)`, `associations(...)`.
- `src/ssdataagent/data/transfer.py` — `TRANSFER_PAIRS`, `load_source_wave(...)`, `compute_crosswalk(...)`.

Changed files (additive):
- `src/ssdataagent/strategies/base.py` — `InfoGate` gains `source`/`source_name`/`crosswalk` fields and `known_marginals()`/`known_associations()` methods; `fit_microdata()` gains a `TRANSFER` branch.
- `src/ssdataagent/agent/context.py` — add `Condition.TRANSFER = "transfer"`.
- `config/datasets.yaml` — add `gss1994`, `cps1970` source-wave entries.

Untouched: `experiments/runner.py`, `experiments/conditions.py`, all strategies, `evaluation/`, the scorer, `generation/`. (Part 3b wires the gate into the runner and adds the A/B/C condition specs.)

---

## §2 — Aggregate computations (`data/aggregates.py`)

Pure functions over a DataFrame + schema. Used by the gate to expose "known
moments." Reuse `classify_columns` for the numeric/categorical split.

```python
def marginals(df: pd.DataFrame, variables: list[str], schema: DatasetSchema,
              *, n_bins: int = 10) -> dict[str, dict]:
    """Univariate marginal per variable.
    Categorical/ordinal → {"kind": "categorical", "probs": {value: prob}} over
      schema.allowed_values (missing categories get prob 0.0; normalized over non-null rows).
    Numerical → {"kind": "numeric", "quantiles": {q: value for q in 0..1 step},
      "mean": float, "std": float} using `n_bins` quantile points from the column."""

def associations(df: pd.DataFrame, target_variables: list[str], schema: DatasetSchema
                 ) -> dict[str, dict[str, float]]:
    """Symmetric pairwise association among target variables:
      cat × cat  → Cramér's V
      num × num  → |Pearson r|
      cat × num  → correlation ratio η (sqrt of η²)
    Returned as a nested dict assoc[a][b] = assoc[b][a] = value in [0, 1];
    self-pairs omitted. NaN/degenerate pairs are skipped (absent from the dict)."""
```

Determinism: no randomness. Both functions tolerate missing/NaN cells (drop
per-pair) and never raise on degenerate input — a pair that cannot be computed is
simply absent from the result.

---

## §3 — Transfer mechanics (`data/transfer.py`)

```python
TRANSFER_PAIRS: dict[str, str] = {"gss": "gss1994", "cps": "cps1970"}
# target dataset name -> source dataset name (source = earlier wave).

def load_source_wave(source_name: str) -> pd.DataFrame:
    """Load the source wave's cleaned CSV via its datasets.yaml schema
    (full wave; no sampling — more rows is better for fitting). Reuses
    load_real_data."""

def compute_crosswalk(target_schema: DatasetSchema, source_schema: DatasetSchema,
                      source_df: pd.DataFrame, target_df: pd.DataFrame) -> list[str]:
    """Common variables usable for transfer: the (background ∪ target) variables
    that appear in BOTH schemas AND as columns in BOTH frames. Ordered by the
    target schema's declared order. Logs the resulting count (and which target
    vars were dropped) so a thin crosswalk is visible, never silent."""
```

The crosswalk defines what transfer can address. Target variables **not** in the
crosswalk cannot be transferred; Part 3b leaves them to the runner's existing
"no informed prediction" baseline fill (`format_generated`), which scores them
low — the correct honest negative.

---

## §4 — InfoGate extension (`strategies/base.py`)

New fields (defaulted → existing construction unchanged):

```python
source: pd.DataFrame | None = None      # source-wave microdata (condition B)
source_name: str | None = None
crosswalk: tuple[str, ...] = ()         # common vars source∩target (condition B)
```

Methods:

```python
def fit_microdata(self) -> pd.DataFrame | None:
    if self.condition is Condition.TRANSFER:
        return None if self.source is None else self.source[list(self.crosswalk)]
    if self.condition in (Condition.FULL, Condition.NO_SEMANTIC, Condition.UNSEEN):
        return self.train
    return None  # NO_DATA (C), DIRECT

def _reference_microdata(self) -> pd.DataFrame | None:
    """Frame the aggregates are computed from: source for B, train for A and C,
    None for DIRECT."""
    if self.condition is Condition.DIRECT:
        return None
    if self.condition is Condition.TRANSFER:
        return None if self.source is None else self.source[list(self.crosswalk)]
    return self.train

def known_marginals(self) -> dict | None:
    ref = self._reference_microdata()
    if ref is None:
        return None
    targets = [t for t in load_schema(self.dataset_name).target_variables if t in ref.columns]
    return marginals(ref, targets, load_schema(self.dataset_name))

def known_associations(self) -> dict | None:
    ref = self._reference_microdata()
    if ref is None:
        return None
    targets = [t for t in load_schema(self.dataset_name).target_variables if t in ref.columns]
    return associations(ref, targets, load_schema(self.dataset_name))
```

**Budget enforcement — the single point of truth:**

| Condition | `fit_microdata()` | `known_marginals()` / `known_associations()` | reference |
|---|---|---|---|
| A (`FULL`) | target `train` | from target `train` | train |
| B (`TRANSFER`) | source (crosswalk cols) | from **source** | source |
| C (`NO_DATA`) | **None** | from target `train` (rows withheld) | train |
| `DIRECT` | None | None | none |

So C exposes the aggregates but no rows (aggregate-only); A exposes aggregates +
rows; B exposes source aggregates + source rows, **never** the target's target
values. (Note: `load_schema` is imported in `base.py` for these methods; it is a
cheap YAML read and is already used elsewhere per-call.)

---

## §5 — Condition enum & `datasets.yaml`

- `agent/context.py`: add `TRANSFER = "transfer"` to `Condition`. The agent's
  `build_context` is unaffected (it lists the data conditions explicitly; TRANSFER
  falls through to no-data, and the agent strategy is never paired with TRANSFER).
- `config/datasets.yaml`: add source-wave entries so `load_schema`/`load_real_data`
  resolve them:

```yaml
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

(Source entries reuse the **target** wave's `ssdatabench_yaml`/sim-subdir/eval
script — they exist only to be loaded as microdata for fitting; they are never
scored themselves. The crosswalk handles any variable-set differences between the
source CSV and the target schema. Verify at implementation time that the source
CSVs load and the schema's variables resolve against them; if a source wave needs
its own variable yaml, point `ssdatabench_yaml` at `real_data/gss/gss1994.yaml` /
`real_data/cps/asec1970.yaml` instead — pick whichever yields the larger crosswalk
and record the choice.)

---

## §6 — Scope boundary (3a vs 3b)

**3a delivers** the mechanisms only: `Condition.TRANSFER`, the extended `InfoGate`
methods/fields, `aggregates.py`, `transfer.py`, source `datasets.yaml` entries —
all unit-testable without any strategy or runner change.

**3b (next cycle) delivers** the consumers: the `design_b` strategy, the A/B/C
condition specs (e.g. `design_b_full`/`design_b_transfer`/`design_b_aggregate`)
that pair `Condition.{FULL,TRANSFER,NO_DATA}` with `strategy="design_b"`, and the
runner wiring that loads the source wave + builds the crosswalk + constructs the
transfer `InfoGate`. No condition spec referencing `design_b` is added in 3a (it
would reference a nonexistent strategy).

---

## §7 — Testing (deterministic, no LLM)

- `tests/test_aggregates.py` — `marginals`: categorical probs normalize to 1 over
  `allowed_values`, missing categories present at 0.0; numeric returns quantiles +
  mean/std. `associations`: cat×cat Cramér's V in [0,1], symmetric; num×num |r|;
  cat×num η; perfectly-associated toy pair → ~1.0; independent toy pair → ~0.0;
  degenerate pair absent (no raise).
- `tests/test_info_gate_transfer.py` — on toy frames + a constructed transfer
  setup: `fit_microdata()` returns source(crosswalk) under `TRANSFER`, `train`
  under A, `None` under C/DIRECT; `known_marginals`/`known_associations` non-None
  for A/B/C, None for DIRECT, and computed from the correct reference (source for
  B, train for A/C); **no target leakage** — a transfer gate's `fit_microdata()`
  contains only crosswalk columns and never the target wave's target values.
- `tests/test_transfer.py` — `compute_crosswalk` returns the correct intersection,
  excludes vars absent from either schema or frame, preserves target-schema order;
  `TRANSFER_PAIRS` maps gss→gss1994, cps→cps1970.
- `tests/test_datasets_source_waves.py` (real-data smoke) — `load_schema("gss1994")`
  / `load_schema("cps1970")` resolve; `load_source_wave` reads both; the GSS and
  CPS crosswalks against their targets are non-trivial (assert ≥ a sane floor,
  e.g. ≥ 5 common variables — adjust to the real count observed, and the test
  documents that count).

**Gate (per [[feedback_refactor_gate_philosophy]]):** correctness = the full local
suite green (minus the 4 pre-existing `autograd`-missing failures), no new
failures. No LLM or cloud run needed for 3a.

---

## Out of scope (do not build this cycle)

- The Design B strategy (per-cell elicitation, raking, copula coupling, sampling) — Part 3b.
- Condition specs and runner wiring for A/B/C Design B — Part 3b.
- LLM elicitation caching / raw-I/O logging — Part 3b.
- Designs C/A, S1, the web console.
- Any change to the scorer, `generation/`, the dashboard, or existing strategies.

---

## Risks & mitigations

- **Thin or empty crosswalk** (source/target variable names diverge across waves).
  Mitigated by `compute_crosswalk` **logging the common-variable count + dropped
  targets**, and a smoke test asserting a sane floor — a thin crosswalk surfaces
  loudly rather than silently producing an unusable transfer setup.
- **Source CSV schema mismatch** (a source wave's columns don't match the target
  schema's variable names). Mitigated by the crosswalk operating on the
  intersection and by the implementation-time check in §5 (pick the
  `ssdatabench_yaml` that yields the larger crosswalk; record it).
- **Aggregates on mixed/degenerate columns.** Mitigated by reusing the proven
  `classify_columns` split and by per-pair NaN tolerance (degenerate pairs absent,
  never raising).
- **Accidental target leakage in B.** Mitigated by `fit_microdata()`/
  `_reference_microdata()` restricting to `crosswalk` columns of the **source**
  frame only, plus a dedicated leakage test.
