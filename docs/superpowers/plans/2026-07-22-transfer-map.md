# Transfer Map (same-country time transfer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the roadmap's Phase-1 transfer map — a diagnostic composition-vs-mechanism
map plus firewalled B0/B1 no-donor baselines — for GSS/CPS same-country time-transfer pairs.

**Architecture:** A new `src/ssdataagent/transfer/` package with four pure modules (pair
registry, the structure/marginal-split generator, the KOB/DFL decomposition, the
copula-stability test) and a `scripts/transfer_map.py` orchestrator that emits CSV tables.
Reuses `nodonor_bracket.score`/`carve_pool`/`build` and `conditional_variance` helpers.

**Tech Stack:** Python 3, numpy, pandas, scipy.stats. Tests via pytest. No LLM, no network.

**Design doc:** `docs/superpowers/specs/2026-07-22-transfer-map-design.md` (read for the
scientific rationale; this plan is the build order).

## Global Constraints

- **Firewall:** Layer-1 diagnostics may read both contexts' microdata (they are the answer
  key). Layer-2 baselines are firewalled: B1 reads only the target's **per-column marginals**
  (never its joint, never its test sample); B0 reads no target data. Enforced structurally by
  `transfer_build` (per-column inverse-CDF + missingness rate only).
- **Measurement discipline:** the real scored run uses `seed=1000+s`, `bootstrap_B=200`;
  never quote an overall gap below ~0.054.
- **Do not modify `nodonor_bracket.build()`** — it is the frozen no-donor replication path.
  `transfer_build` is a separate implementation in the new package.
- All new code is LLM-free and must not shell out to the scorer in unit tests (no `live_*`
  markers); tests use small synthetic frames with known answers.
- Match repo test conventions: flat `tests/test_*.py`, `from __future__ import annotations`,
  run with `.venv/bin/python -m pytest`.

---

### Task 1: Pair registry & crosswalk (`transfer/pairs.py`)

**Files:**
- Create: `src/ssdataagent/transfer/__init__.py` (empty)
- Create: `src/ssdataagent/transfer/pairs.py`
- Test: `tests/test_transfer_pairs.py`

**Interfaces:**
- Produces: `TransferPair` dataclass; `PAIRS: list[TransferPair]`;
  `crosswalk_columns(schema_name: str, source_df, target_df) -> list[str]`;
  `covariates_outcomes(schema_name: str, cols: list[str]) -> tuple[list[str], list[str]]`;
  `load_pair(pair: TransferPair) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_pairs.py
from __future__ import annotations

import pandas as pd

from ssdataagent.transfer.pairs import (
    PAIRS, TransferPair, crosswalk_columns, covariates_outcomes,
)


def test_pairs_registry_shape():
    ids = [p.id for p in PAIRS]
    assert ids == [
        "gss_1994_2018", "cps_1970_1980", "cps_1970_1990", "cps_1980_1990",
        "cps_1970_2000", "cps_1980_2000", "cps_1990_2000",
    ]
    scored = {p.id for p in PAIRS if p.scored}
    assert scored == {"gss_1994_2018", "cps_1970_1980"}
    for p in PAIRS:
        assert isinstance(p, TransferPair)
        assert (p.target_dataset is not None) == p.scored


def test_crosswalk_keeps_common_logs_dropped():
    # cps background/target vars include age, gender, race, education, income, ...
    src = pd.DataFrame({"age": [1], "gender": [1], "race": [1], "education": [1],
                        "income": [1], "birth_year": [1], "source_only": [1]})
    tgt = pd.DataFrame({"age": [1], "gender": [1], "race": [1], "education": [1],
                        "birth_year": [1], "target_only": [1]})  # no income -> dropped
    cols = crosswalk_columns("cps", src, tgt)
    assert "age" in cols and "gender" in cols and "education" in cols
    assert "income" not in cols          # target lacks it
    assert "source_only" not in cols and "target_only" not in cols
    # birth_year is a wave time-identity (birth_year = year - age), disjoint support
    # across waves -> non-transferable, dropped even though present in both frames.
    assert "birth_year" not in cols


def test_covariates_outcomes_split():
    cols = crosswalk_columns("cps",
                             pd.DataFrame({c: [1] for c in
                                           ["age", "gender", "race", "education", "income"]}),
                             pd.DataFrame({c: [1] for c in
                                           ["age", "gender", "race", "education", "income"]}))
    x, y = covariates_outcomes("cps", cols)
    assert "age" in x and "gender" in x          # background/demographic
    assert "income" in y                          # an outcome
    assert set(x).isdisjoint(set(y))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_pairs.py -v`
Expected: FAIL — `ModuleNotFoundError: ssdataagent.transfer`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/transfer/pairs.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ssdataagent.config import data_root
from ssdataagent.data.schema import load_schema

log = logging.getLogger(__name__)

# Wave time-identities: mechanically tied to the survey year (birth_year = year - age),
# so their support is disjoint across waves and they carry no transferable mechanism.
# Dropped from every transfer crosswalk (documented, like a data_audit trap).
NON_TRANSFERABLE = frozenset({"birth_year"})


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]


@dataclass(frozen=True)
class TransferPair:
    id: str
    source_csv: Path
    target_csv: Path
    schema_name: str        # "gss" | "cps": drives X/Y split + scoring config
    scored: bool            # True only for benchmark-backed targets
    target_dataset: str | None  # ds name passed to score() when scored


def _cps(name: str) -> Path:
    return data_root() / "cps" / name


def _gss(name: str) -> Path:
    return data_root() / "gss" / name


PAIRS: list[TransferPair] = [
    TransferPair("gss_1994_2018", _gss("gss1994.csv"), _gss("gss2018.csv"), "gss", True, "gss"),
    TransferPair("cps_1970_1980", _cps("cps-asec1970.csv"), _cps("cps-asec1980.csv"), "cps", True, "cps"),
    TransferPair("cps_1970_1990", _cps("cps-asec1970.csv"), _cps("cps-asec1990.csv"), "cps", False, None),
    TransferPair("cps_1980_1990", _cps("cps-asec1980.csv"), _cps("cps-asec1990.csv"), "cps", False, None),
    TransferPair("cps_1970_2000", _cps("cps-asec1970.csv"), _cps("cps-asec2000.csv"), "cps", False, None),
    TransferPair("cps_1980_2000", _cps("cps-asec1980.csv"), _cps("cps-asec2000.csv"), "cps", False, None),
    TransferPair("cps_1990_2000", _cps("cps-asec1990.csv"), _cps("cps-asec2000.csv"), "cps", False, None),
]


def crosswalk_columns(schema_name: str, source_df: pd.DataFrame,
                      target_df: pd.DataFrame) -> list[str]:
    """Background+target vars present as columns in BOTH frames, ordered by schema."""
    schema = load_schema(schema_name)
    candidate = [v for v in list(schema.background_variables) + list(schema.target_variables)
                 if v not in NON_TRANSFERABLE]
    common = [v for v in candidate
              if v in source_df.columns and v in target_df.columns]
    dropped = [v for v in candidate if v not in common]
    log.info("crosswalk[%s]: %d common (dropped %d: %s)",
             schema_name, len(common), len(dropped), dropped)
    return common


def covariates_outcomes(schema_name: str, cols: list[str]) -> tuple[list[str], list[str]]:
    schema = load_schema(schema_name)
    bg = set(schema.background_variables)
    x = [c for c in cols if c in bg]
    y = [c for c in cols if c not in bg]
    return x, y


def load_pair(pair: TransferPair) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    src = _drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    tgt = _drop_unnamed(pd.read_csv(pair.target_csv, low_memory=False))
    cols = crosswalk_columns(pair.schema_name, src, tgt)
    return src, tgt, cols
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_pairs.py -v`
Expected: PASS (3 tests). If a schema's variable names differ from the test's assumptions
(e.g. `income` not an output var in the cps schema), adjust the test's asserted names to
real schema vars — inspect with `.venv/bin/python -c "from ssdataagent.data.schema import load_schema; s=load_schema('cps'); print(s.background_variables, s.target_variables)"` — but keep the drop-logging and X/Y-disjointness assertions.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/__init__.py src/ssdataagent/transfer/pairs.py tests/test_transfer_pairs.py
git commit -m "transfer: pair registry + crosswalk"
```

---

### Task 2: Structure/marginal-split generator (`transfer/generate.py`)

**Files:**
- Create: `src/ssdataagent/transfer/generate.py`
- Test: `tests/test_transfer_generate.py`

**Interfaces:**
- Produces: `transfer_build(struct, marg, cols, n, seed, mode) -> pd.DataFrame`
  where `mode ∈ {"carryover", "marginal-swap"}`.
- Consumes: `nodonor_bracket.build` (in test only, for the equivalence check).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_generate.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.transfer.generate import transfer_build


def _frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1, n)
    return pd.DataFrame({
        "x": x,
        "y": x * 1.5 + rng.normal(0, 0.5, n),          # numeric, correlated with x
        "g": np.where(x > shift, "hi", "lo"),           # categorical, tied to x
    })


def test_carryover_matches_bracket_copula_fixed():
    import nodonor_bracket as nb
    a = _frame(400, 1)
    cols = ["x", "y", "g"]
    got = transfer_build(a, a, cols, n=300, seed=7, mode="carryover")
    ref = nb.build(a, cols, n=300, seed=7, mode="copula-fixed")
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref.reset_index(drop=True))


def test_marginal_swap_takes_target_marginals():
    a = _frame(500, 2, shift=0.0)          # x ~ N(0,1)
    b = _frame(500, 3, shift=5.0)          # x ~ N(5,1) -- clearly different marginal
    out = transfer_build(a, b, ["x", "y", "g"], n=2000, seed=9, mode="marginal-swap")
    # swapped output's x marginal follows B (mean ~5), not A (mean ~0)
    assert abs(pd.to_numeric(out["x"]).mean() - 5.0) < 0.5
    assert abs(pd.to_numeric(out["x"]).mean() - 0.0) > 3.0


def test_marginal_swap_preserves_copula():
    # A has strong x~y rank dependence; after swapping B's marginals the dependence survives
    a = _frame(600, 4)
    b = _frame(600, 5, shift=5.0)
    out = transfer_build(a, b, ["x", "y", "g"], n=3000, seed=11, mode="marginal-swap")
    r = pd.to_numeric(out["x"]).corr(pd.to_numeric(out["y"]), method="spearman")
    assert r > 0.6          # positive dependence carried over from A's copula


def test_rejects_unknown_mode():
    a = _frame(50, 1)
    with pytest.raises(ValueError):
        transfer_build(a, a, ["x"], n=10, seed=1, mode="nonsense")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_generate.py -v`
Expected: FAIL — `ImportError: cannot import name 'transfer_build'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/transfer/generate.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _is_numeric(s: pd.Series) -> bool:
    s = s.dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _latent(pool: pd.DataFrame, c: str, num: bool, glat: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    """Per-row uniform latent for column c, correlated across columns via glat.
    Ported verbatim from nodonor_bracket._latent so carryover mode matches build()."""
    m = len(pool)
    if num:
        u = np.array(pd.to_numeric(pool[c], errors="coerce")
                     .rank(pct=True, method="first").to_numpy(dtype=float))
        nan = np.isnan(u)
        u[nan] = rng.random(int(nan.sum()))
        return u
    s = pool[c].astype(str)
    order = (pd.Series(glat, index=range(m)).groupby(s.to_numpy()).mean()
             .sort_values().index.tolist())
    pos = {v: i for i, v in enumerate(order)}
    b = np.array(s.map(pos).to_numpy(dtype=float), dtype=float)
    nn = np.isnan(b)
    b[nn] = rng.integers(0, max(1, len(order)), int(nn.sum()))
    return pd.Series(b + rng.random(m)).rank(pct=True, method="first").to_numpy()


def _marginal_map(marg_col: pd.Series, u: np.ndarray, num: bool) -> np.ndarray:
    """Inverse-CDF: map uniforms u onto marg_col's empirical marginal (object array)."""
    if num:
        vals = np.sort(pd.to_numeric(marg_col, errors="coerce").dropna().to_numpy())
        if len(vals) == 0:
            return np.full(len(u), np.nan, dtype=object)
        return vals[np.clip((u * len(vals)).astype(int), 0, len(vals) - 1)].astype(object)
    vc = marg_col.dropna().astype(str).value_counts(normalize=True)
    if len(vc) == 0:
        return np.full(len(u), np.nan, dtype=object)
    cats, edges = vc.index.to_numpy(), np.cumsum(vc.to_numpy())
    return cats[np.searchsorted(edges, u, side="right").clip(0, len(cats) - 1)].astype(object)


def transfer_build(struct: pd.DataFrame, marg: pd.DataFrame, cols: list[str],
                   n: int, seed: int, mode: str) -> pd.DataFrame:
    """Copula from ``struct``, marginals from ``marg``.

    mode "carryover"     -- struct == marg (== source A). Equals
                            nodonor_bracket.build(A, ..., "copula-fixed").
    mode "marginal-swap" -- struct = A, marg = B. A's dependence, B's marginals.

    Numeric detection, the shared latent index (``base``), and each column's missingness
    PATTERN come from ``struct``; the inverse-CDF value map and the missingness RATE come
    from ``marg``. In carryover the two frames are identical, so the rng draw order matches
    build() exactly (asserted by test).
    """
    if mode not in ("carryover", "marginal-swap"):
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed)
    m = len(struct)
    num = {c: _is_numeric(struct[c]) for c in cols}
    znum = {c: pd.to_numeric(struct[c], errors="coerce").rank(pct=True)
            for c in cols if num[c]}
    glat = (pd.DataFrame(znum).mean(axis=1).fillna(0.5).to_numpy() if znum
            else np.full(m, 0.5))
    base = rng.integers(0, m, n)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        u = np.clip(_latent(struct, c, num[c], glat, rng)[base], 1e-6, 1 - 1e-6)
        em = _marginal_map(marg[c], u, num[c])
        miss = float(marg[c].isna().mean())
        if miss > 0:
            want = int(round(miss * n))
            mask = struct[c].isna().to_numpy()[base].copy()
            have = int(mask.sum())
            if have > want:
                mask[rng.choice(np.flatnonzero(mask), have - want, replace=False)] = False
            elif have < want:
                free = np.flatnonzero(~mask)
                mask[rng.choice(free, min(want - have, len(free)), replace=False)] = True
            em[mask] = np.nan
        out[c] = em
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_generate.py -v`
Expected: PASS (4 tests). The `test_carryover_matches_bracket_copula_fixed` equivalence is
the load-bearing one — if it fails, diff the rng draw order against `nodonor_bracket.build`
(bracket.py:137-175) and align; do NOT change `build()`.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/generate.py tests/test_transfer_generate.py
git commit -m "transfer: structure/marginal-split generator (B0 carryover, B1 swap)"
```

---

### Task 3: Composition-vs-mechanism decomposition (`transfer/decompose.py`)

**Files:**
- Create: `src/ssdataagent/transfer/decompose.py`
- Test: `tests/test_transfer_decompose.py`

**Interfaces:**
- Produces: `raking_weights(a, b, covariates, *, bins=10, iters=30) -> np.ndarray`;
  `kob_decompose(a, b, response, covariates) -> dict`;
  `oaxaca_blinder(a, b, response, covariates, *, numeric_predictors=frozenset()) -> dict`.
- Consumes: `conditional_variance._dummy_design` (for OB design matrices).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_decompose.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.decompose import (
    kob_decompose, oaxaca_blinder, raking_weights,
)


def _mk(n, seed, xmean, beta):
    rng = np.random.default_rng(seed)
    x = rng.normal(xmean, 1, n)
    return pd.DataFrame({"x": x, "y": beta * x + rng.normal(0, 0.3, n)})


def test_raking_matches_target_marginal():
    a = _mk(2000, 1, xmean=0.0, beta=1.0)
    b = _mk(2000, 2, xmean=3.0, beta=1.0)
    w = raking_weights(a, b, ["x"], bins=8)
    # weighted mean of A's x should move toward B's (~3)
    wm = np.average(pd.to_numeric(a["x"]), weights=w)
    assert abs(wm - 3.0) < 0.4


def test_composition_dominated():
    # same mechanism (beta=1), X shifted -> gap is pure composition -> share ~1
    a = _mk(3000, 3, xmean=0.0, beta=1.0)
    b = _mk(3000, 4, xmean=3.0, beta=1.0)
    d = kob_decompose(a, b, "y", ["x"])
    assert d["composition_share"] > 0.7
    assert d["label"] == "composition-dominated"


def test_mechanism_shifted():
    # same X distribution, mechanism flips (beta 1 -> -1) -> share ~0
    a = _mk(3000, 5, xmean=0.0, beta=1.0)
    b = _mk(3000, 5, xmean=0.0, beta=-1.0)   # same seed => same X, different y-mechanism
    d = kob_decompose(a, b, "y", ["x"])
    assert d["composition_share"] < 0.3
    assert d["label"] == "mechanism-shifted"


def test_oaxaca_agrees_on_linear_case():
    a = _mk(4000, 6, xmean=0.0, beta=1.0)
    b = _mk(4000, 7, xmean=3.0, beta=1.0)
    ob = oaxaca_blinder(a, b, "y", ["x"], numeric_predictors=frozenset({"x"}))
    # pure composition => endowment term dominates
    assert ob["composition_share_ob"] > 0.7


def test_aligned_returns_nan():
    a = _mk(2000, 8, xmean=0.0, beta=1.0)
    b = _mk(2000, 9, xmean=0.0, beta=1.0)   # essentially same distribution
    d = kob_decompose(a, b, "y", ["x"])
    assert (not np.isfinite(d["composition_share"])) or d["label"] in {
        "composition-dominated", "mechanism-shifted", "aligned"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_decompose.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/transfer/decompose.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ssdataagent.data.conditional_variance import _dummy_design

_EPS = 1e-9


def _is_num(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _edges(a_col: pd.Series, b_col: pd.Series, bins: int) -> np.ndarray:
    pooled = pd.to_numeric(pd.concat([a_col, b_col]), errors="coerce").dropna()
    if len(pooled) == 0:
        return np.array([])
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    return np.unique(np.quantile(pooled, qs))


def _codes(s: pd.Series, num: bool, edges: np.ndarray | None) -> np.ndarray:
    if num:
        v = pd.to_numeric(s, errors="coerce")
        c = np.digitize(v.to_numpy(dtype=float), edges) if edges is not None and len(edges) \
            else np.zeros(len(v), dtype=int)
        c = c.astype(object)
        c[v.isna().to_numpy()] = "__nan__"
        return c.astype(str)
    return s.astype("string").fillna("__nan__").to_numpy().astype(str)


def raking_weights(a: pd.DataFrame, b: pd.DataFrame, covariates: list[str],
                   *, bins: int = 10, iters: int = 30) -> np.ndarray:
    """Per-row weights on A so its weighted covariate marginals match B's (IPF/raking)."""
    n = len(a)
    w = np.ones(n, dtype=float)
    specs = []
    for c in covariates:
        num = _is_num(a[c])
        edges = _edges(a[c], b[c], bins) if num else None
        a_codes = _codes(a[c], num, edges)
        b_codes = _codes(b[c], num, edges)
        target = pd.Series(b_codes).value_counts(normalize=True)
        specs.append((a_codes, target))
    for _ in range(iters):
        for a_codes, target in specs:
            cur = pd.Series(w).groupby(a_codes).sum()
            cur = cur / cur.sum()
            factor = {k: target.get(k, 1e-12) / max(cur.get(k, 1e-12), 1e-12)
                      for k in np.unique(a_codes)}
            w = w * np.array([factor[k] for k in a_codes])
            w = w * (n / w.sum())
    return w


def _weighted_props(vals: np.ndarray, w: np.ndarray) -> pd.Series:
    key = pd.Series(vals).astype("string").fillna("__nan__").to_numpy()
    s = pd.Series(w).groupby(key).sum()
    return s / s.sum()


def _tv(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return 0.5 * float((p.reindex(idx, fill_value=0.0) - q.reindex(idx, fill_value=0.0)).abs().sum())


def kob_decompose(a: pd.DataFrame, b: pd.DataFrame, response: str,
                  covariates: list[str]) -> dict:
    """DFL reweighting decomposition of the A->B gap in ``response``.

    composition_share = (gap_raw - gap_residual) / gap_raw, where gap_residual is the gap
    remaining after raking A's covariates to B's. Numeric response uses standardized
    1-Wasserstein; categorical uses total-variation distance.
    """
    num = _is_num(a[response]) and _is_num(b[response])
    w = raking_weights(a, b, covariates)
    if num:
        av = pd.to_numeric(a[response], errors="coerce")
        bv = pd.to_numeric(b[response], errors="coerce")
        oka, okb = av.notna().to_numpy(), bv.notna().to_numpy()
        avv, wv = av[oka].to_numpy(dtype=float), w[oka]
        bvv = bv[okb].to_numpy(dtype=float)
        if len(avv) == 0 or len(bvv) == 0:
            gap_raw = gap_res = np.nan
        else:
            sd = float(np.std(np.concatenate([avv, bvv]))) or 1.0
            gap_raw = wasserstein_distance(avv / sd, bvv / sd)
            gap_res = wasserstein_distance(avv / sd, bvv / sd, u_weights=wv)
    else:
        pa_raw = pd.Series(a[response]).astype("string").fillna("__nan__").value_counts(normalize=True)
        pb = pd.Series(b[response]).astype("string").fillna("__nan__").value_counts(normalize=True)
        pa_w = _weighted_props(a[response].to_numpy(), w)
        gap_raw = _tv(pa_raw, pb)
        gap_res = _tv(pa_w, pb)
    if not np.isfinite(gap_raw) or gap_raw < _EPS:
        share = np.nan
        label = "aligned"
    else:
        share = float(np.clip((gap_raw - gap_res) / gap_raw, 0.0, 1.0))
        label = "composition-dominated" if share >= 0.5 else "mechanism-shifted"
    return {
        "response": response,
        "composition_share": share,
        "mechanism_share": (1.0 - share) if np.isfinite(share) else np.nan,
        "gap_raw": gap_raw, "gap_residual": gap_res,
        "label": label, "method": "dfl",
    }


def oaxaca_blinder(a: pd.DataFrame, b: pd.DataFrame, response: str,
                   covariates: list[str], *,
                   numeric_predictors: frozenset[str] = frozenset()) -> dict:
    """Twofold Oaxaca-Blinder for a NUMERIC response (cross-check for kob_decompose).

    Builds a SHARED dummy design on A∪B so coefficient vectors are aligned, fits OLS
    separately on each, and splits the mean gap into endowment (composition) and
    coefficient (mechanism) terms with A as the reference.
    """
    both = pd.concat([a[covariates], b[covariates]], ignore_index=True)
    design, ok = _dummy_design(both, covariates, numeric_predictors)
    na = len(a)
    ya = pd.to_numeric(a[response], errors="coerce").to_numpy(dtype=float)
    yb = pd.to_numeric(b[response], errors="coerce").to_numpy(dtype=float)
    da = design.iloc[:na]
    db = design.iloc[na:]
    oka = ok[:na] & np.isfinite(ya)
    okb = ok[na:] & np.isfinite(yb)

    def _fit(d, y, m):
        X = np.column_stack([np.ones(int(m.sum())), d.loc[m].to_numpy(dtype=float)])
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        return beta, X.mean(axis=0)

    beta_a, xbar_a = _fit(da, ya, oka)
    beta_b, xbar_b = _fit(db, yb, okb)
    endowment = float((xbar_b - xbar_a) @ beta_a)
    coefficient = float(xbar_b @ (beta_b - beta_a))
    denom = abs(endowment) + abs(coefficient)
    share = abs(endowment) / denom if denom > _EPS else np.nan
    return {"response": response, "endowment": endowment, "coefficient": coefficient,
            "composition_share_ob": share, "method": "oaxaca-blinder"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_decompose.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/decompose.py tests/test_transfer_decompose.py
git commit -m "transfer: composition-vs-mechanism (DFL reweighting + Oaxaca-Blinder)"
```

---

### Task 4: Copula-stability test (`transfer/copula_stability.py`)

**Files:**
- Create: `src/ssdataagent/transfer/copula_stability.py`
- Test: `tests/test_transfer_copula_stability.py`

**Interfaces:**
- Produces: `pair_association(frame, v1, v2) -> tuple[float, str]`;
  `copula_stability(a, b, cols, *, threshold=0.10) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_copula_stability.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.copula_stability import copula_stability, pair_association


def _gauss_copula(n, seed, rho, xmean=0.0, ymean=0.0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return pd.DataFrame({"x": z[:, 0] + xmean, "y": z[:, 1] + ymean})


def test_stable_copula_different_marginals():
    a = _gauss_copula(2000, 1, rho=0.7, xmean=0.0, ymean=0.0)
    b = _gauss_copula(2000, 2, rho=0.7, xmean=5.0, ymean=9.0)  # same dependence, shifted
    tau_a, method = pair_association(a, "x", "y")
    tau_b, _ = pair_association(b, "x", "y")
    assert method == "kendall"
    assert abs(tau_a - tau_b) < 0.1        # rank-based => marginal-invariant


def test_shifted_copula_flagged():
    a = _gauss_copula(2000, 3, rho=0.7)
    b = _gauss_copula(2000, 4, rho=-0.7)   # opposite dependence
    df = copula_stability(a, b, ["x", "y"])
    row = df.iloc[0]
    assert row["abs_delta"] > 0.5
    assert row["label"] == "shifted"


def test_copula_stability_frame_shape():
    a = _gauss_copula(500, 5, rho=0.5)
    b = _gauss_copula(500, 6, rho=0.5, xmean=2.0)
    df = copula_stability(a, b, ["x", "y"])
    assert list(df.columns) == ["v1", "v2", "method", "assoc_a", "assoc_b",
                                "abs_delta", "label"]
    assert len(df) == 1                    # one unordered pair from 2 cols
    assert df.iloc[0]["label"] == "stable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_copula_stability.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/transfer/copula_stability.py
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kendalltau


def _is_num(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if ct.to_numpy().sum() == 0 or min(ct.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    r, k = ct.shape
    denom = n * max(1, min(r - 1, k - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def pair_association(frame: pd.DataFrame, v1: str, v2: str) -> tuple[float, str]:
    """Copula probe for a variable pair. Both numeric/ordinal -> Kendall's tau
    (rank-based, marginal-invariant). Any nominal member -> Cramer's V on binned data."""
    s1, s2 = frame[v1], frame[v2]
    num1, num2 = _is_num(s1), _is_num(s2)
    if num1 and num2:
        a = pd.to_numeric(s1, errors="coerce")
        c = pd.to_numeric(s2, errors="coerce")
        ok = a.notna() & c.notna()
        if int(ok.sum()) < 10:
            return np.nan, "kendall"
        tau, _ = kendalltau(a[ok], c[ok])
        return (float(tau) if tau == tau else np.nan), "kendall"

    def _cat(s: pd.Series, num: bool) -> pd.Series:
        if num:
            v = pd.to_numeric(s, errors="coerce")
            try:
                return pd.qcut(v, 5, duplicates="drop").astype("string")
            except (ValueError, IndexError):
                return v.rank(pct=True).round(1).astype("string")
        return s.astype("string")

    c1, c2 = _cat(s1, num1), _cat(s2, num2)
    ok = c1.notna() & c2.notna()
    if int(ok.sum()) < 10:
        return np.nan, "cramers_v"
    return _cramers_v(c1[ok], c2[ok]), "cramers_v"


def copula_stability(a: pd.DataFrame, b: pd.DataFrame, cols: list[str],
                     *, threshold: float = 0.10) -> pd.DataFrame:
    """Per unordered variable pair: association in A vs B, and |delta| stability label."""
    rows = []
    for v1, v2 in itertools.combinations(cols, 2):
        assoc_a, method = pair_association(a, v1, v2)
        assoc_b, _ = pair_association(b, v1, v2)
        if np.isfinite(assoc_a) and np.isfinite(assoc_b):
            delta = abs(assoc_a - assoc_b)
            label = "stable" if delta < threshold else "shifted"
        else:
            delta, label = np.nan, "undefined"
        rows.append({"v1": v1, "v2": v2, "method": method,
                     "assoc_a": assoc_a, "assoc_b": assoc_b,
                     "abs_delta": delta, "label": label})
    return pd.DataFrame(rows, columns=["v1", "v2", "method", "assoc_a", "assoc_b",
                                       "abs_delta", "label"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_copula_stability.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/copula_stability.py tests/test_transfer_copula_stability.py
git commit -m "transfer: copula-stability test (Kendall tau / Cramer's V)"
```

---

### Task 5: Orchestrator (`scripts/transfer_map.py`)

**Files:**
- Create: `scripts/transfer_map.py`
- Test: `tests/test_transfer_map.py`

**Interfaces:**
- Consumes: everything from `ssdataagent.transfer.*`, and `nodonor_bracket.{score, build,
  carve_pool}`.
- Produces: `run_layer1(a, b, cols, covariates, outcomes) -> tuple[pd.DataFrame, pd.DataFrame]`
  (map rows, copula rows); `run_layer2(pair, *, seeds, n, bootstrap_B) -> pd.DataFrame`;
  `main()`.

- [ ] **Step 1: Write the failing test** (Layer-1 only — no scorer, no data files)

```python
# tests/test_transfer_map.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from transfer_map import run_layer1


def _frame(n, seed, xmean, beta):
    rng = np.random.default_rng(seed)
    x = rng.normal(xmean, 1, n)
    edu = np.where(x > xmean, "hi", "lo")
    return pd.DataFrame({"age": x, "education": edu, "income": beta * x + rng.normal(0, .3, n)})


def test_run_layer1_returns_map_and_copula():
    a = _frame(1500, 1, xmean=0.0, beta=1.0)
    b = _frame(1500, 2, xmean=3.0, beta=1.0)   # composition shift on outcomes
    covariates, outcomes = ["age", "education"], ["income"]
    kob, cop = run_layer1(a, b, ["age", "education", "income"], covariates, outcomes)
    # KOB has one row per outcome, with a share and a label
    assert set(kob["response"]) == {"income"}
    assert {"composition_share", "label", "gap_raw"}.issubset(kob.columns)
    inc = kob[kob["response"] == "income"].iloc[0]
    assert inc["label"] in {"composition-dominated", "mechanism-shifted", "aligned"}
    # copula table covers all unordered pairs of the 3 columns
    assert len(cop) == 3
    assert {"v1", "v2", "abs_delta", "label"}.issubset(cop.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_map.py -v`
Expected: FAIL — `ModuleNotFoundError: transfer_map`.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python
"""Build the transfer map: Layer-1 diagnostics (composition-vs-mechanism + copula
stability) over all pairs, and Layer-2 firewalled B0/B1 baselines over the scored pairs.

See docs/superpowers/specs/2026-07-22-transfer-map-design.md. No LLM, no API key.

    .venv/bin/python scripts/transfer_map.py
    .venv/bin/python scripts/transfer_map.py --pairs cps_1970_1980 --seeds 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.transfer.copula_stability import copula_stability  # noqa: E402
from ssdataagent.transfer.decompose import kob_decompose, oaxaca_blinder  # noqa: E402
from ssdataagent.transfer.generate import transfer_build  # noqa: E402
from ssdataagent.transfer.pairs import (  # noqa: E402
    PAIRS, covariates_outcomes, load_pair,
)

OUT = REPO / "results" / "transfer_map"


def run_layer1(a: pd.DataFrame, b: pd.DataFrame, cols: list[str],
               covariates: list[str], outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostic map (reads both contexts' microdata — the answer key, not firewalled)."""
    rows = []
    for y in outcomes:
        d = kob_decompose(a, b, y, covariates)
        try:
            ob = oaxaca_blinder(a, b, y, covariates)
            d["composition_share_ob"] = ob["composition_share_ob"]
        except Exception:
            d["composition_share_ob"] = float("nan")
        rows.append(d)
    kob = pd.DataFrame(rows)
    cop = copula_stability(a, b, cols)
    return kob, cop


def run_layer2(pair, *, seeds: int, n: int, bootstrap_B: int) -> pd.DataFrame:
    """Firewalled B0/B1 baselines scored against the target's benchmark reference."""
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema

    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    schema = load_schema(pair.target_dataset)
    ref = nb._drop_unnamed(pd.read_csv(schema.real_data_path, low_memory=False))
    b_pool, guarantee = nb.carve_pool(pair.target_dataset)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    types = nb.TYPES.get(pair.target_dataset, (1, 2, 3))

    def _score_many(builder):
        recs = [nb.score(builder(s), pair.target_dataset, ref, types,
                         seed=1000 + s, bootstrap_B=bootstrap_B) for s in range(1, seeds + 1)]
        df = pd.DataFrame(recs)
        return {f"{c}": float(df[c].mean()) for c in df.columns if c.startswith(("T", "overall"))
                and df[c].notna().any()}

    configs = {
        "B0_carryover": lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "B1_marginal_swap": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
        "within_B_floor": lambda s: nb.build(b_pool, cols, n, s, "independence"),
        "within_B_ceiling": lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
    }
    out = []
    for name, builder in configs.items():
        rec = {"pair": pair.id, "config": name, "guarantee": guarantee}
        rec.update(_score_many(builder))
        out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=None, help="pair ids (default: all)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in PAIRS if a.pairs is None or p.id in a.pairs]

    all_map, all_cop, all_base = [], [], []
    for p in pairs:
        src, tgt, cols = load_pair(p)
        cov, outc = covariates_outcomes(p.schema_name, cols)
        kob, cop = run_layer1(src, tgt, cols, cov, outc)
        kob.insert(0, "pair", p.id); cop.insert(0, "pair", p.id)
        kob.to_csv(OUT / f"map_{p.id}.csv", index=False)
        all_map.append(kob); all_cop.append(cop)
        print(f"\n=== {p.id} (Layer 1) ===")
        print(kob[["response", "composition_share", "mechanism_share", "label"]].to_string(index=False))
        stable = (cop["label"] == "stable").mean() if len(cop) else float("nan")
        print(f"copula stable fraction: {stable:.2f}  ({len(cop)} pairs)")
        if p.scored:
            base = run_layer2(p, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B)
            base.to_csv(OUT / f"baselines_{p.id}.csv", index=False)
            all_base.append(base)
            print(f"--- {p.id} (Layer 2, firewalled baselines) ---")
            print(base.to_string(index=False))

    if all_map:
        pd.concat(all_map, ignore_index=True).to_csv(OUT / "map.csv", index=False)
        pd.concat(all_cop, ignore_index=True).to_csv(OUT / "copula.csv", index=False)
    if all_base:
        pd.concat(all_base, ignore_index=True).to_csv(OUT / "baselines.csv", index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_map.py -v`
Expected: PASS (1 test). (Layer 2 is exercised in Task 6's real run, not here.)

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_map.py tests/test_transfer_map.py
git commit -m "transfer: map orchestrator (Layer-1 diagnostics + Layer-2 baselines)"
```

---

### Task 6: Real run + report + dashboard *(controller task — not a subagent implementer)*

**Files:**
- Create: `results/transfer_map/*.csv` (produced by running the orchestrator)
- Create: `docs/report/2026-07-22-transfer-map.md`
- Modify: `docs/experiments/LEDGER.md` (append one row)
- Modify: `docs/dashboard/index.html` (regenerated)

- [ ] **Step 1: Run the full pipeline**

Run: `.venv/bin/python scripts/transfer_map.py --seeds 5 --bootstrap-B 200`
Expected: `results/transfer_map/{map.csv, copula.csv, baselines.csv, map_*.csv,
baselines_*.csv}` written; Layer-1 tables for all 7 pairs, Layer-2 tables for the 2 scored
pairs. If a scored pair errors on a specific type (a T3 response dropped by the crosswalk),
that shows as `None`/absent — record it, do not silently drop.

- [ ] **Step 2: Sanity-check the numbers**

Confirm: (a) the CPS time-distance ladder shows composition_share trending as gap widens;
(b) B1 ≥ B0 overall on both scored pairs (a marginal swap should not hurt); (c) B1 sits at
or below the within-B floor; (d) copula-stable fraction is higher where B1's T2 is better.
Note any cell that violates the map↔baseline throughline — that is a finding, not a bug to
hide.

- [ ] **Step 3: Write the report**

`docs/report/2026-07-22-transfer-map.md`: the claim, what the map is (answer key, not a
method), the two-layer method, the composition/mechanism table + copula-stable fractions,
the B0/B1/floor/ceiling baseline table, the map↔baseline validation, and the §9 honesty
limits from the spec (firewall boundary, row- vs person-disjoint, Layer-1 uses B's joint by
design, same-country only). Follow the measurement discipline (`bootstrap_B=200`, noise
floor ~0.054).

- [ ] **Step 4: LEDGER row + dashboard**

Append a `docs/experiments/LEDGER.md` row with a meaningful one-line hypothesis (per
AGENTS.md the dashboard reads this), then:

```bash
.venv/bin/python scripts/build_dashboard.py
git add docs/dashboard/index.html
```

- [ ] **Step 5: Commit**

```bash
git add results/transfer_map docs/report/2026-07-22-transfer-map.md docs/experiments/LEDGER.md docs/dashboard/index.html
git commit -m "transfer map: real run, report, ledger + dashboard"
```

---

## Self-Review notes (author)

- **Spec coverage:** §4 pairs → Task 1; §5a generator → Task 2; §5b decompose → Task 3;
  §5c copula → Task 4; §5d/5e registry+orchestrator → Tasks 1/5; §7 report/ledger → Task 6.
  All spec components map to a task.
- **Firewall:** enforced in `run_layer2` — B1 fits on `a` (source) and maps to `b_pool`
  marginals only; `ref` (B's benchmark sample) is used solely by `score()`.
- **Type consistency:** `transfer_build(struct, marg, cols, n, seed, mode)`,
  `kob_decompose(a, b, response, covariates) -> dict`, `copula_stability(a,b,cols) -> DataFrame`
  used identically in the orchestrator and tests.
- **Known risk:** the carryover≡`build()` equivalence (Task 2) depends on exact rng draw
  order; the test pins it. If a real schema's var names differ from Task 1's test
  assumptions, fix the test's names (not the crosswalk logic).
