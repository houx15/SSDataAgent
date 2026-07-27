# Transfer Characterization Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on the real cps/gss/cfps data, how heterogeneous contexts are and why — the composition-vs-mechanism split (Q1), X-composition distance (Q2), mechanism difference (Q3), and shape-vs-level of mechanism moves (Q4) — and render the answers as a self-contained HTML report.

**Architecture:** A new pure/analytical module `transfer/characterize.py` holds a context/pair registry and two new pure metrics (`marginal_distance`, `shape_level_split`), and orchestrates the four questions over each pair by reusing the existing tested primitives (`decompose.kob_decompose`, `decompose.oaxaca_blinder`, `copula_stability.copula_stability`) into a tidy long-format DataFrame. A thin CLI writes the table; a report builder renders matplotlib figures embedded as base64 into one standalone HTML.

**Tech Stack:** Python 3, pandas, numpy, scipy, matplotlib 3.10.9 (all present in `.venv`).

## Global Constraints

- **Analyst-side, no firewall constraint.** This study reads A and B microdata freely. Do not add firewall guards; this is not a generator.
- **Reuse, don't reinvent.** Q1 = `ssdataagent.transfer.decompose.kob_decompose` / `oaxaca_blinder`; Q3 = `ssdataagent.transfer.copula_stability.copula_stability`; numeric/categorical detection = `ssdataagent.transfer.decompose._is_num` (single authority). Only `marginal_distance` and `shape_level_split` are new metrics.
- **No `/dev/null` or out-of-repo redirects** (project hook blocks them). Never write outside the repo.
- **`results/` is gitignored.** The tidy CSV goes to `results/characterization/` AND a committed copy to `docs/report/2026-07-27-characterization-data.csv`.
- **Never stage the `ssdatabench` submodule.** Stage only the specific files each task names.
- **Avoid the literal word "eval" in commit messages** (project hook blocks it); use "the check" / "the run".
- **Per-schema constants** (module-level dicts in `characterize.py`):
  - `CORE_DEMOGRAPHICS = {"cps": ("age","gender","race"), "gss": ("age","gender","race"), "cfps": ("gender","sib_number")}`
  - `FOCAL = {"cps": "age", "gss": "age", "cfps": "birth_year"}`
  - `GROUP_COL = {"cps": "race", "gss": "race", "cfps": "minzu"}`
- For a **group** pair, `GROUP_COL[schema]` is excluded from both the Q1 core and the Q2 X-sweep.
- Q4 focal gate checks `focal in a.columns and focal in b.columns` (NOT `in cols`): the crosswalk drops `birth_year` via `pairs.NON_TRANSFERABLE`, but `birth_year`/`age` are real CSV columns and remain valid Q4 focals.

---

### Task 1: Pure metrics — `marginal_distance` and `shape_level_split`

**Files:**
- Create: `src/ssdataagent/transfer/characterize.py`
- Test: `tests/test_transfer_characterize.py`

**Interfaces:**
- Consumes: `ssdataagent.transfer.decompose._is_num`, `scipy.stats.wasserstein_distance`.
- Produces:
  - `marginal_distance(a_col: pd.Series, b_col: pd.Series) -> tuple[float, str]` — `(distance, kind)`, `kind in {"wasserstein","tv"}`.
  - `shape_level_split(a: pd.DataFrame, b: pd.DataFrame, response: str, focal: str, *, bins: int = 10) -> dict` — keys `response, focal, level, shape, shape_ratio, n_bins`.
  - `_EPS = 1e-9` (module constant).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer_characterize.py
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_marginal_distance_identical_is_zero():
    from ssdataagent.transfer.characterize import marginal_distance
    num = pd.Series([1.0, 2.0, 3.0, 4.0])
    d, kind = marginal_distance(num, num.copy())
    assert kind == "wasserstein" and abs(d) < 1e-9
    cat = pd.Series(["a", "b", "a", "c"])
    d, kind = marginal_distance(cat, cat.copy())
    assert kind == "tv" and abs(d) < 1e-9


def test_marginal_distance_disjoint_categoricals_is_one():
    from ssdataagent.transfer.characterize import marginal_distance
    a = pd.Series(["x", "x", "x"])
    b = pd.Series(["y", "y", "y"])
    d, kind = marginal_distance(a, b)
    assert kind == "tv" and abs(d - 1.0) < 1e-9


def test_marginal_distance_known_numeric_shift():
    from ssdataagent.transfer.characterize import marginal_distance
    # a all 0, b all 1 -> pooled SD 0.5 -> standardized values 0 vs 2 -> Wasserstein 2.0
    a = pd.Series([0.0] * 100)
    b = pd.Series([1.0] * 100)
    d, kind = marginal_distance(a, b)
    assert kind == "wasserstein" and abs(d - 2.0) < 1e-6


def test_shape_level_split_pure_level_shift():
    from ssdataagent.transfer.characterize import shape_level_split
    focal = np.repeat(np.arange(10), 20).astype(float)
    a = pd.DataFrame({"f": focal, "y": focal.copy()})
    b = pd.DataFrame({"f": focal, "y": focal + 5.0})
    r = shape_level_split(a, b, "y", "f", bins=10)
    assert abs(r["level"] - 5.0) < 1e-6
    assert r["shape"] < 1e-6
    assert r["shape_ratio"] < 1e-3


def test_shape_level_split_pure_shape_change():
    from ssdataagent.transfer.characterize import shape_level_split
    focal = np.repeat(np.arange(-5, 6), 20).astype(float)   # symmetric about 0
    a = pd.DataFrame({"f": focal, "y": np.zeros(len(focal))})
    b = pd.DataFrame({"f": focal, "y": focal.copy()})        # gradient changes, mean gap ~ 0
    r = shape_level_split(a, b, "y", "f", bins=10)
    assert abs(r["level"]) < 0.5
    assert r["shape"] > 1.0
    assert r["shape_ratio"] > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_characterize.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (module not created yet).

- [ ] **Step 3: Create the module with the two metrics**

```python
# src/ssdataagent/transfer/characterize.py
"""Transfer characterization study: measure how heterogeneous contexts are and why.

Analyst-side (reads A and B freely) -- see
docs/superpowers/specs/2026-07-27-transfer-characterization-study-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ssdataagent.transfer.decompose import _is_num

_EPS = 1e-9


def marginal_distance(a_col: pd.Series, b_col: pd.Series) -> tuple[float, str]:
    """Distance between the marginal of ``a_col`` and ``b_col``. Both numeric ->
    standardized 1-Wasserstein (divide by pooled SD of the non-missing values);
    otherwise -> total variation (0.5 * sum|p-q|) with NaN bucketed as its own category."""
    if _is_num(a_col) and _is_num(b_col):
        av = pd.to_numeric(a_col, errors="coerce").dropna().to_numpy(dtype=float)
        bv = pd.to_numeric(b_col, errors="coerce").dropna().to_numpy(dtype=float)
        if len(av) == 0 or len(bv) == 0:
            return np.nan, "wasserstein"
        sd = float(np.std(np.concatenate([av, bv]))) or 1.0
        return float(wasserstein_distance(av / sd, bv / sd)), "wasserstein"
    pa = a_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    pb = b_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    tv = 0.5 * float((pa.reindex(idx, fill_value=0.0)
                      - pb.reindex(idx, fill_value=0.0)).abs().sum())
    return tv, "tv"


def shape_level_split(a: pd.DataFrame, b: pd.DataFrame, response: str, focal: str,
                      *, bins: int = 10) -> dict:
    """Split the A->B conditional-mean gap of a NUMERIC ``response`` over bins of ``focal``
    into a level term (mean gap) and a shape term (rms residual). g(x) = E_B[Y|x] - E_A[Y|x]
    over shared quantile bins of ``focal``; level = mean_x g(x); shape = rms_x(g(x)-level);
    shape_ratio = shape/(|level|+shape+eps). ~0 => pure level shift; ~1 => shape change.
    Bins with no data in either context are skipped."""
    fa = pd.to_numeric(a[focal], errors="coerce")
    fb = pd.to_numeric(b[focal], errors="coerce")
    ya = pd.to_numeric(a[response], errors="coerce")
    yb = pd.to_numeric(b[response], errors="coerce")
    pooled = pd.concat([fa, fb]).dropna()
    empty = {"response": response, "focal": focal, "level": np.nan,
             "shape": np.nan, "shape_ratio": np.nan, "n_bins": 0}
    if len(pooled) == 0:
        return empty
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(pooled, qs))
    ca = np.digitize(fa.to_numpy(dtype=float), edges)
    cb = np.digitize(fb.to_numpy(dtype=float), edges)
    fa_ok, fb_ok = fa.notna().to_numpy(), fb.notna().to_numpy()
    ya_ok, yb_ok = ya.notna().to_numpy(), yb.notna().to_numpy()
    yav, ybv = ya.to_numpy(dtype=float), yb.to_numpy(dtype=float)
    gaps = []
    for k in np.unique(np.concatenate([ca, cb])):
        ma = (ca == k) & fa_ok & ya_ok
        mb = (cb == k) & fb_ok & yb_ok
        if ma.sum() == 0 or mb.sum() == 0:
            continue
        gaps.append(float(ybv[mb].mean() - yav[ma].mean()))
    if not gaps:
        return empty
    g = np.array(gaps, dtype=float)
    level = float(g.mean())
    shape = float(np.sqrt(np.mean((g - level) ** 2)))
    return {"response": response, "focal": focal, "level": level, "shape": shape,
            "shape_ratio": float(shape / (abs(level) + shape + _EPS)), "n_bins": len(g)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_characterize.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/characterize.py tests/test_transfer_characterize.py
git commit -m "feat(characterize): marginal_distance + shape_level_split metrics"
```

---

### Task 2: Context / Pair registry and `load_context`

**Files:**
- Modify: `src/ssdataagent/transfer/characterize.py`
- Test: `tests/test_transfer_characterize.py`

**Interfaces:**
- Consumes: `ssdataagent.config.data_root`, `ssdataagent.transfer.pairs._drop_unnamed`.
- Produces:
  - `@dataclass(frozen=True) Context(dataset: str, csv: Path, label: str, group_col: str | None = None, group_val: str | None = None, negate: bool = False)`
  - `@dataclass(frozen=True) Pair(id: str, family: str, a: Context, b: Context, schema_name: str)` (`family in {"time","group"}`)
  - `load_context(ctx: Context) -> pd.DataFrame`
  - `CONTEXTS: dict[str, Context]`, `PAIRS: list[Pair]`
  - The three constant dicts `CORE_DEMOGRAPHICS`, `FOCAL`, `GROUP_COL` (from Global Constraints).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_transfer_characterize.py

def test_load_context_group_filter_and_negate(tmp_path):
    from ssdataagent.transfer.characterize import Context, load_context
    csv = tmp_path / "toy.csv"
    pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3],
        "race": ["Black", "White", None, "Black"],
        "x": [1, 2, 3, 4],
    }).to_csv(csv, index=False)
    minority = Context("toy", csv, "black", "race", "Black")
    df = load_context(minority)
    assert list(df.columns) == ["race", "x"]      # Unnamed dropped
    assert len(df) == 2 and set(df["race"]) == {"Black"}
    majority = Context("toy", csv, "rest", "race", "Black", negate=True)
    dfm = load_context(majority)
    assert len(dfm) == 1 and set(dfm["race"]) == {"White"}   # NaN excluded from both


def test_pairs_registry_shape_and_paths():
    from ssdataagent.transfer.characterize import PAIRS, CORE_DEMOGRAPHICS, FOCAL, GROUP_COL
    ids = [p.id for p in PAIRS]
    assert ids == [
        "cps_1970_1980", "cps_1980_1990", "cps_1990_2000", "cps_1970_2000",
        "gss_1994_2018", "cps_1980_race", "gss_2018_race", "cfps_minzu",
    ]
    fams = {p.id: p.family for p in PAIRS}
    assert fams["gss_1994_2018"] == "time" and fams["cfps_minzu"] == "group"
    for p in PAIRS:
        assert p.a.csv.exists() and p.b.csv.exists(), f"missing csv for {p.id}"
    assert CORE_DEMOGRAPHICS["cfps"] == ("gender", "sib_number")
    assert FOCAL["cfps"] == "birth_year" and GROUP_COL["gss"] == "race"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_characterize.py -k "load_context or registry" -v`
Expected: FAIL with `ImportError` (`Context` / `PAIRS` not defined yet).

- [ ] **Step 3: Add the registry and loader**

Add these imports at the top of `characterize.py` (below the existing imports):

```python
from dataclasses import dataclass
from pathlib import Path

from ssdataagent.config import data_root
from ssdataagent.transfer.pairs import _drop_unnamed
```

Then append:

```python
CORE_DEMOGRAPHICS: dict[str, tuple[str, ...]] = {
    "cps": ("age", "gender", "race"),
    "gss": ("age", "gender", "race"),
    "cfps": ("gender", "sib_number"),
}
FOCAL: dict[str, str] = {"cps": "age", "gss": "age", "cfps": "birth_year"}
GROUP_COL: dict[str, str] = {"cps": "race", "gss": "race", "cfps": "minzu"}


@dataclass(frozen=True)
class Context:
    dataset: str
    csv: Path
    label: str
    group_col: str | None = None
    group_val: str | None = None
    negate: bool = False   # True -> keep rows where group_col != group_val (majority/rest)


@dataclass(frozen=True)
class Pair:
    id: str
    family: str            # "time" | "group"
    a: Context
    b: Context
    schema_name: str


def load_context(ctx: Context) -> pd.DataFrame:
    """Read a context's CSV, drop ``Unnamed:`` index columns, and apply the group filter.
    NaN in the grouping column is excluded from BOTH subgroups (== and != both drop it)."""
    df = _drop_unnamed(pd.read_csv(ctx.csv, low_memory=False))
    if ctx.group_col is not None:
        col = df[ctx.group_col].astype("string")
        mask = (col != ctx.group_val) if ctx.negate else (col == ctx.group_val)
        df = df[mask.fillna(False)].reset_index(drop=True)
    return df


def _cps(name: str) -> Path:
    return data_root() / "cps" / name


def _gss(name: str) -> Path:
    return data_root() / "gss" / name


def _cfps() -> Path:
    return data_root() / "cfps" / "cfps_2010_2022.csv"


CONTEXTS: dict[str, Context] = {
    "cps_1970": Context("cps", _cps("cps-asec1970.csv"), "cps 1970"),
    "cps_1980": Context("cps", _cps("cps-asec1980.csv"), "cps 1980"),
    "cps_1990": Context("cps", _cps("cps-asec1990.csv"), "cps 1990"),
    "cps_2000": Context("cps", _cps("cps-asec2000.csv"), "cps 2000"),
    "gss_1994": Context("gss", _gss("gss1994.csv"), "gss 1994"),
    "gss_2018": Context("gss", _gss("gss2018.csv"), "gss 2018"),
    "cps_1980_maj": Context("cps", _cps("cps-asec1980.csv"), "cps1980 non-Black",
                            "race", "Black", negate=True),
    "cps_1980_min": Context("cps", _cps("cps-asec1980.csv"), "cps1980 Black", "race", "Black"),
    "gss_2018_maj": Context("gss", _gss("gss2018.csv"), "gss2018 non-Black",
                            "race", "Black", negate=True),
    "gss_2018_min": Context("gss", _gss("gss2018.csv"), "gss2018 Black", "race", "Black"),
    "cfps_han": Context("cfps", _cfps(), "cfps han", "minzu", "han"),
    "cfps_min": Context("cfps", _cfps(), "cfps minority", "minzu", "minority"),
}

PAIRS: list[Pair] = [
    Pair("cps_1970_1980", "time", CONTEXTS["cps_1970"], CONTEXTS["cps_1980"], "cps"),
    Pair("cps_1980_1990", "time", CONTEXTS["cps_1980"], CONTEXTS["cps_1990"], "cps"),
    Pair("cps_1990_2000", "time", CONTEXTS["cps_1990"], CONTEXTS["cps_2000"], "cps"),
    Pair("cps_1970_2000", "time", CONTEXTS["cps_1970"], CONTEXTS["cps_2000"], "cps"),
    Pair("gss_1994_2018", "time", CONTEXTS["gss_1994"], CONTEXTS["gss_2018"], "gss"),
    Pair("cps_1980_race", "group", CONTEXTS["cps_1980_maj"], CONTEXTS["cps_1980_min"], "cps"),
    Pair("gss_2018_race", "group", CONTEXTS["gss_2018_maj"], CONTEXTS["gss_2018_min"], "gss"),
    Pair("cfps_minzu", "group", CONTEXTS["cfps_han"], CONTEXTS["cfps_min"], "cfps"),
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_characterize.py -v`
Expected: all tests pass (7 total).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/characterize.py tests/test_transfer_characterize.py
git commit -m "feat(characterize): context/pair registry and load_context"
```

---

### Task 3: Orchestration — `resolve_columns`, `pair_records`, `run_characterization`

**Files:**
- Modify: `src/ssdataagent/transfer/characterize.py`
- Test: `tests/test_transfer_characterize.py`

**Interfaces:**
- Consumes: `pairs.crosswalk_columns`, `pairs.covariates_outcomes`, `decompose.kob_decompose`, `decompose.oaxaca_blinder`, `decompose._is_num`, `copula_stability.copula_stability`, and Task 1/2 symbols.
- Produces:
  - `resolve_columns(pair: Pair, a=None, b=None) -> dict` — keys `a, b, cols, x, y, core, x_sweep, focal`.
  - `pair_records(pair: Pair) -> list[dict]` — tidy rows, each with at least `pair, family, dataset, question, metric, key, value`.
  - `run_characterization(pairs=PAIRS) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_transfer_characterize.py

def test_resolve_columns_group_excludes_grouping_var():
    from ssdataagent.transfer.characterize import PAIRS, resolve_columns
    pair = next(p for p in PAIRS if p.id == "gss_2018_race")
    r = resolve_columns(pair)
    assert "race" not in r["core"], "grouping var must be excluded from Q1 core"
    assert "race" not in r["x_sweep"], "grouping var must be excluded from Q2 sweep"
    assert r["core"] == ["age", "gender"]
    assert r["focal"] == "age"


def test_run_characterization_tidy_schema_on_one_pair():
    from ssdataagent.transfer.characterize import PAIRS, run_characterization
    pair = next(p for p in PAIRS if p.id == "gss_2018_race")   # smallest (single gss wave)
    df = run_characterization([pair])
    for col in ("pair", "family", "dataset", "question", "metric", "key", "value"):
        assert col in df.columns
    assert set(df["question"]) >= {"Q1", "Q2", "Q3", "Q4"}
    q1 = df[(df["question"] == "Q1") & (df["metric"] == "composition_share")]
    assert len(q1) > 0 and q1["value"].notna().any()
    assert (df["pair"] == "gss_2018_race").all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_characterize.py -k "resolve or tidy_schema" -v`
Expected: FAIL with `ImportError` (`resolve_columns` / `run_characterization` not defined).

- [ ] **Step 3: Add the orchestration**

Add to the imports block in `characterize.py`:

```python
from ssdataagent.transfer import copula_stability as _cop
from ssdataagent.transfer import decompose as _dec
from ssdataagent.transfer.pairs import covariates_outcomes, crosswalk_columns
```

Then append:

```python
def resolve_columns(pair: Pair, a: pd.DataFrame | None = None,
                    b: pd.DataFrame | None = None) -> dict:
    """Load both contexts and resolve the column sets for one pair. For a group pair the
    grouping variable (GROUP_COL[schema]) is dropped from both the Q1 core and Q2 sweep."""
    if a is None:
        a = load_context(pair.a)
    if b is None:
        b = load_context(pair.b)
    cols = crosswalk_columns(pair.schema_name, a, b)
    x, y = covariates_outcomes(pair.schema_name, cols)
    is_group = pair.family == "group"
    gcol = GROUP_COL.get(pair.schema_name)
    core = [c for c in CORE_DEMOGRAPHICS[pair.schema_name]
            if c in cols and not (is_group and c == gcol)]
    x_sweep = [c for c in x if not (is_group and c == gcol)]
    return {"a": a, "b": b, "cols": cols, "x": x, "y": y,
            "core": core, "x_sweep": x_sweep, "focal": FOCAL[pair.schema_name]}


def pair_records(pair: Pair) -> list[dict]:
    """Run Q1-Q4 for one pair and return tidy long-format rows."""
    r = resolve_columns(pair)
    a, b, cols = r["a"], r["b"], r["cols"]
    core, x_sweep, focal = r["core"], r["x_sweep"], r["focal"]
    base = {"pair": pair.id, "family": pair.family, "dataset": pair.schema_name}
    recs: list[dict] = []

    # Q1 -- composition vs mechanism (per outcome)
    for resp in r["y"]:
        d = _dec.kob_decompose(a, b, resp, core)
        recs.append({**base, "question": "Q1", "metric": "composition_share", "key": resp,
                     "value": d["composition_share"], "mechanism_share": d["mechanism_share"],
                     "ess_ratio": d["ess_ratio"], "label": d["label"]})
        if _dec._is_num(a[resp]) and _dec._is_num(b[resp]):
            ob = _dec.oaxaca_blinder(a, b, resp, core)
            recs.append({**base, "question": "Q1", "metric": "composition_share_ob",
                         "key": resp, "value": ob["composition_share_ob"],
                         "endowment": ob["endowment"], "coefficient": ob["coefficient"]})

    # Q2 -- X composition distance (per covariate)
    for xc in x_sweep:
        dist, kind = marginal_distance(a[xc], b[xc])
        recs.append({**base, "question": "Q2", "metric": "marginal_distance", "key": xc,
                     "value": dist, "kind": kind})

    # Q3 -- mechanism / association stability
    stab = _cop.copula_stability(a, b, cols)
    n = len(stab)
    counts = stab["label"].value_counts()
    for lab in ("stable", "shifted", "undefined"):
        recs.append({**base, "question": "Q3", "metric": f"pct_{lab}", "key": "all_pairs",
                     "value": (float(counts.get(lab, 0)) / n) if n else np.nan})
    recs.append({**base, "question": "Q3", "metric": "median_abs_delta", "key": "all_pairs",
                 "value": float(stab["abs_delta"].median(skipna=True)) if n else np.nan})

    # Q4 -- shape vs level (numeric outcomes; focal must be a real column in both frames)
    if focal in a.columns and focal in b.columns:
        for resp in r["y"]:
            if _dec._is_num(a[resp]) and _dec._is_num(b[resp]):
                s = shape_level_split(a, b, resp, focal)
                recs.append({**base, "question": "Q4", "metric": "shape_ratio", "key": resp,
                             "value": s["shape_ratio"], "level": s["level"],
                             "shape": s["shape"], "focal": focal, "n_bins": s["n_bins"]})
    return recs


def run_characterization(pairs: list[Pair] = PAIRS) -> pd.DataFrame:
    """Concatenate pair_records over all pairs into one tidy long-format table."""
    rows: list[dict] = []
    for p in pairs:
        rows.extend(pair_records(p))
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_characterize.py -v`
Expected: all pass (9 total). The one-pair `run_characterization` test loads a single gss wave (~2.3k rows) and takes a few seconds — acceptable.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/characterize.py tests/test_transfer_characterize.py
git commit -m "feat(characterize): pair orchestration into tidy Q1-Q4 table"
```

---

### Task 4: CLI runner — `write_outputs` and `scripts/characterize.py`

**Files:**
- Modify: `src/ssdataagent/transfer/characterize.py`
- Create: `scripts/characterize.py`
- Test: `tests/test_transfer_characterize.py`

**Interfaces:**
- Consumes: `run_characterization`, `ssdataagent.config.REPO_ROOT`.
- Produces: `write_outputs(df: pd.DataFrame, repo_root: Path) -> tuple[Path, Path]` — writes `results/characterization/characterization.csv` and `docs/report/2026-07-27-characterization-data.csv`, returns both paths.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transfer_characterize.py

def test_write_outputs_writes_both_copies(tmp_path):
    from ssdataagent.transfer.characterize import write_outputs
    df = pd.DataFrame({"pair": ["p"], "question": ["Q1"], "value": [0.5]})
    results_csv, committed_csv = write_outputs(df, tmp_path)
    assert results_csv.exists() and committed_csv.exists()
    assert results_csv == tmp_path / "results" / "characterization" / "characterization.csv"
    assert committed_csv == tmp_path / "docs" / "report" / "2026-07-27-characterization-data.csv"
    back = pd.read_csv(committed_csv)
    assert list(back["pair"]) == ["p"] and float(back["value"].iloc[0]) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_characterize.py -k write_outputs -v`
Expected: FAIL with `ImportError` (`write_outputs` not defined).

- [ ] **Step 3: Add `write_outputs` and the CLI script**

Append to `characterize.py`:

```python
def write_outputs(df: pd.DataFrame, repo_root: Path) -> tuple[Path, Path]:
    """Write the tidy table to results/ (gitignored) and a committed copy under docs/report."""
    out_dir = repo_root / "results" / "characterization"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "characterization.csv"
    df.to_csv(results_csv, index=False)
    committed = repo_root / "docs" / "report" / "2026-07-27-characterization-data.csv"
    committed.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(committed, index=False)
    return results_csv, committed
```

Create `scripts/characterize.py`:

```python
#!/usr/bin/env python
"""Run the transfer characterization sweep (Q1-Q4) and write the tidy results table."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssdataagent.config import REPO_ROOT
from ssdataagent.transfer.characterize import run_characterization, write_outputs


def main() -> None:
    df = run_characterization()
    results_csv, committed = write_outputs(df, REPO_ROOT)
    print(f"characterization: {len(df)} rows across {df['pair'].nunique()} pairs")
    print(f"  results copy   -> {results_csv}")
    print(f"  committed copy -> {committed}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the real sweep**

Run: `.venv/bin/python -m pytest tests/test_transfer_characterize.py -k write_outputs -v`
Expected: PASS.

Then run the real sweep end-to-end (this exercises all 8 pairs on real data):

Run: `.venv/bin/python scripts/characterize.py`
Expected: prints `characterization: <N> rows across 8 pairs` and two file paths; both CSVs exist. If any pair raises, fix the underlying issue before proceeding — do not narrow the pair list.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/characterize.py scripts/characterize.py tests/test_transfer_characterize.py docs/report/2026-07-27-characterization-data.csv
git commit -m "feat(characterize): CLI runner + committed tidy results table"
```

---

### Task 5: HTML report builder — `scripts/characterize_report.py`

**Files:**
- Create: `scripts/characterize_report.py`
- Test: `tests/test_characterize_report.py`

**Interfaces:**
- Consumes: the tidy CSV (`docs/report/2026-07-27-characterization-data.csv`), matplotlib, `ssdataagent.config.REPO_ROOT`.
- Produces: `build_report_html(df: pd.DataFrame) -> str` (full standalone HTML) and `main()` writing `docs/report/2026-07-27-transfer-characterization.html`.

**Interface note:** `build_report_html` must accept the tidy DataFrame and return a complete HTML document string beginning with `<!doctype html>` — so it is testable without touching the filesystem. Use the non-interactive matplotlib backend (`matplotlib.use("Agg")` before importing pyplot).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_characterize_report.py
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import pandas as pd


def _toy_df():
    rows = []
    for pair, fam in [("cps_1970_1980", "time"), ("gss_2018_race", "group")]:
        ds = "cps" if pair.startswith("cps") else "gss"
        rows += [
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q1",
             "metric": "composition_share", "key": "income", "value": 0.6},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q2",
             "metric": "marginal_distance", "key": "age", "value": 0.3, "kind": "wasserstein"},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q3",
             "metric": "pct_stable", "key": "all_pairs", "value": 0.7},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q3",
             "metric": "pct_shifted", "key": "all_pairs", "value": 0.2},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q3",
             "metric": "pct_undefined", "key": "all_pairs", "value": 0.1},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q4",
             "metric": "shape_ratio", "key": "income", "value": 0.4, "level": 2.0, "shape": 1.3},
        ]
    return pd.DataFrame(rows)


def test_build_report_html_is_self_contained():
    from characterize_report import build_report_html
    html = build_report_html(_toy_df())
    assert html.lstrip().startswith("<!doctype html>")
    assert "</html>" in html
    # figures embedded, no external asset references
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html
    assert "src=\"http" not in html
    # all four questions titled
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert q in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_characterize_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'characterize_report'`.

- [ ] **Step 3: Write the report builder**

Create `scripts/characterize_report.py`:

```python
#!/usr/bin/env python
"""Render the transfer characterization report: one self-contained HTML with Q1-Q4 figures."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssdataagent.config import REPO_ROOT   # noqa: E402

DATA_CSV = REPO_ROOT / "docs" / "report" / "2026-07-27-characterization-data.csv"
OUT_HTML = REPO_ROOT / "docs" / "report" / "2026-07-27-transfer-characterization.html"
_FAMILY_COLOR = {"time": "#3b6ea5", "group": "#a5533b"}


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig) -> str:
    return f'<img src="data:image/png;base64,{_b64(fig)}" style="max-width:100%;height:auto"/>'


def _fig_q1(df):
    d = df[(df["question"] == "Q1") & (df["metric"] == "composition_share")].dropna(subset=["value"])
    fig, ax = plt.subplots(figsize=(7, 4))
    fams = ["time", "group"]
    data = [d[d["family"] == f]["value"].to_numpy() for f in fams]
    ax.boxplot(data, labels=[f"{f}\n(n={len(v)})" for f, v in zip(fams, data)], showmeans=True)
    for i, f in enumerate(fams, start=1):
        sub = d[d["family"] == f]
        ax.scatter([i] * len(sub), sub["value"], alpha=0.5,
                   color=[_FAMILY_COLOR[f]] * len(sub), zorder=3, s=18)
    ax.set_ylabel("composition_share (per outcome)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Q1 — how much of each Y-gap is composition vs mechanism")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    return fig


def _fig_q2(df):
    d = df[(df["question"] == "Q2") & (df["metric"] == "marginal_distance")].dropna(subset=["value"])
    means = d.groupby(["pair", "family"])["value"].mean().reset_index().sort_values("value")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(means))))
    ax.barh(means["pair"], means["value"],
            color=[_FAMILY_COLOR.get(f, "#777") for f in means["family"]])
    ax.set_xlabel("mean X-composition distance (TV / std-Wasserstein)")
    ax.set_title("Q2 — how different is demographic composition, per pair")
    return fig


def _fig_q3(df):
    d = df[df["question"] == "Q3"]
    piv = d[d["metric"].isin(["pct_stable", "pct_shifted", "pct_undefined"])]
    piv = piv.pivot_table(index="pair", columns="metric", values="value", aggfunc="first").fillna(0)
    for c in ("pct_stable", "pct_shifted", "pct_undefined"):
        if c not in piv:
            piv[c] = 0.0
    piv = piv[["pct_stable", "pct_shifted", "pct_undefined"]]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(piv))))
    left = [0.0] * len(piv)
    for c, color in [("pct_stable", "#4a9"), ("pct_shifted", "#c74"), ("pct_undefined", "#bbb")]:
        ax.barh(piv.index, piv[c], left=left, label=c.replace("pct_", ""), color=color)
        left = [l + v for l, v in zip(left, piv[c])]
    ax.set_xlabel("fraction of variable-pairs")
    ax.set_title("Q3 — mechanism (association) stability, per pair")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def _fig_q4(df):
    d = df[(df["question"] == "Q4") & (df["metric"] == "shape_ratio")].dropna(subset=["value"])
    fig, ax = plt.subplots(figsize=(7, 4))
    fams = ["time", "group"]
    data = [d[d["family"] == f]["value"].to_numpy() for f in fams]
    data = [v if len(v) else [float("nan")] for v in data]
    ax.boxplot(data, labels=fams, showmeans=True)
    ax.set_ylabel("shape_ratio  (0 = pure level shift, 1 = shape change)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Q4 — when mechanism moves, is it level or shape? (numeric Y)")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    return fig


_SECTIONS = [
    ("Q1 — composition vs mechanism", _fig_q1,
     "Per outcome, the share of the A→B gap explained by reweighting A's demographics to B's "
     "(composition); the remainder is mechanism. Above the dashed line = composition-dominated."),
    ("Q2 — X-composition distance", _fig_q2,
     "Mean distance between the two contexts' demographic marginals; larger = the populations "
     "differ more in who they contain."),
    ("Q3 — mechanism stability", _fig_q3,
     "Fraction of variable-pairs whose association is stable vs shifted between contexts "
     "(|Δ| threshold 0.10). More 'stable' = the dependence structure transfers."),
    ("Q4 — shape vs level", _fig_q4,
     "For numeric outcomes, whether the conditional curve moved by a constant offset (level, "
     "cheaply correctable) or changed slope (shape, needs real adaptation)."),
]


def build_report_html(df: pd.DataFrame) -> str:
    blocks = []
    for title, fn, caption in _SECTIONS:
        blocks.append(f"<section><h2>{title}</h2><p class='cap'>{caption}</p>{_img(fn(df))}</section>")
    body = "\n".join(blocks)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transfer characterization — cps / gss / cfps</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto;
          padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
  .cap {{ color: #555; font-size: 0.9rem; }}
  section {{ border-top: 1px solid #eee; padding-top: 0.5rem; }}
  footer {{ color: #888; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #eee; padding-top: 1rem; }}
</style></head><body>
<h1>Transfer characterization: how heterogeneous are contexts, and why?</h1>
<p class="cap">cps / gss / cfps · time and group (ethnicity) families · analyst-side, reads A and B.
Q5 (a learned composition model) is not built: with ~{df['pair'].nunique()} pairs we stay below the
corpus threshold to learn transport, so it remains a corpus-gated follow-on.</p>
{body}
<footer>Generated by scripts/characterize_report.py from
docs/report/2026-07-27-characterization-data.csv. See
docs/superpowers/specs/2026-07-27-transfer-characterization-study-design.md.</footer>
</body></html>"""


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    OUT_HTML.write_text(build_report_html(df), encoding="utf-8")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then build the real report**

Run: `.venv/bin/python -m pytest tests/test_characterize_report.py -v`
Expected: PASS.

Then build against the real tidy CSV (produced in Task 4):

Run: `.venv/bin/python scripts/characterize_report.py`
Expected: prints `wrote .../2026-07-27-transfer-characterization.html`; open it and confirm four figures render with data.

- [ ] **Step 5: Commit**

```bash
git add scripts/characterize_report.py tests/test_characterize_report.py docs/report/2026-07-27-transfer-characterization.html
git commit -m "feat(characterize): self-contained HTML report with Q1-Q4 figures"
```

---

## Self-Review

**Spec coverage:** Q1 (Task 3 kob/oaxaca rows), Q2 (Task 3 marginal_distance rows + Task 1 metric), Q3 (Task 3 copula_stability summary), Q4 (Task 1 shape_level_split + Task 3 rows), Q5 disposition (Task 5 report note). Contexts/pairs registry (Task 2). Deliverable HTML + committed CSV (Tasks 4–5). Per-schema constants, grouping-var exclusion, Q4 `.columns` gate — all in Global Constraints and Tasks 2–3.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `Context`/`Pair` fields, `resolve_columns` dict keys (`a,b,cols,x,y,core,x_sweep,focal`), and the tidy row keys (`pair,family,dataset,question,metric,key,value`) are used identically across Tasks 3–5. `marginal_distance` returns `(float, str)`; `shape_level_split` returns the documented dict; `write_outputs` returns `(Path, Path)`; `build_report_html` takes a DataFrame and returns a str — consistent with every call site.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-27-transfer-characterization.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — I execute tasks in this session with checkpoints for review.

Which approach?
