# B3 LLM-Prior-Pointed-At-Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add B3 — the no-donor LLM full method pointed at the *target* context — as the final Phase-2 baseline, scored on the identical footing as B0/B1/B2, reporting three rungs (raw / elicited-R² / pool-R²).

**Architecture:** A thin orchestrator `scripts/transfer_b3.py` reuses the existing durable LLM stages from `scripts/nodonor_fullmethod.py` (seed→LLM-complete→variance-repair) and scores through the same restricted-config path as `scripts/transfer_map.py`. Two shared seams are extracted into `src/` so no script imports another: scoring helpers into `src/ssdataagent/transfer/scoring.py`, and the `Spec` dataclass + per-dataset specs into `src/ssdataagent/transfer/b3_specs.py`. `transfer_map.py` stays LLM-free.

**Tech Stack:** Python 3, pandas, numpy, pytest, OpenRouter (OpenAI client) for the GSS generation only; cps runs off a warm cache.

## Global Constraints

- **Publication protocol for every scored run:** `seeds=3`, `n=3000`, `bootstrap_B=200`.
- **Comparability is sacred:** B3 uses the *identical* crosswalk `cols`, the *identical* `restrict_config_dir` output, the *identical* reference (`load_schema(ds).real_data_path`), and the *identical* seed offset (`1000+s`) as B0/B1/B2. Compute `cols` exactly as `transfer_map.run_layer2` does: `[c for c in load_pair(pair)[2] if c in a.columns and c in b_pool.columns and c in ref.columns]`.
- **`transfer_map.py` must remain LLM-free / no-API-key** — never import an LLM stage into it.
- **Repair predictors MUST mirror the dataset's `type3.yaml`** exactly, or every alpha is silently mis-sized. cps: `["age","gender","race","education"]`; gss: `["age","gender","race","education"]`.
- **Firewall (row-level):** B3 reads from the target only pool marginals, pool covariate-R² (the `B3_pool_R2` rung only), and public codebook/context. Never the target joint, never the reference/test sample.
- **Model:** `anthropic/claude-sonnet-4.5` (reuse `nodonor_fullmethod.MODEL`).
- **Cache location:** LLM stages cache under `results/nodonor_cache/<ds>_cond_raw.csv` and `<ds>_elicit.json`. Output ladder rows go to `results/transfer_map/b3_<pair>.csv`.
- **Target dataset name** for a pair is `pair.target_dataset` (`"cps"` / `"gss"`); it drives `carve_pool`, the cache filename, `nb.TYPES`, and the schema. Do not derive it from the pair id string.

---

## File Structure

- `src/ssdataagent/transfer/scoring.py` (new) — `restrict_config_dir`, `mean_scores`. Pure scoring helpers, no LLM.
- `src/ssdataagent/transfer/b3_specs.py` (new) — `Spec` dataclass; `SPECS = {"cps": ..., "gss": ...}`. Prompt/repair wiring per dataset.
- `scripts/transfer_b3.py` (new) — column derivation, rung-alpha computation, orchestrator, CLI.
- `scripts/transfer_map.py` (modify) — import the two helpers from `scoring.py` instead of defining them.
- `scripts/nodonor_fullmethod.py` (modify) — import `Spec` and `SPECS["cps"]` from `b3_specs.py` instead of defining them.
- Tests: `tests/test_transfer_scoring.py`, `tests/test_b3_specs.py`, `tests/test_transfer_b3.py`.

---

## Task 1: Extract scoring helpers into `src/ssdataagent/transfer/scoring.py`

**Files:**
- Create: `src/ssdataagent/transfer/scoring.py`
- Modify: `scripts/transfer_map.py` (remove the two function defs; import them)
- Test: `tests/test_transfer_scoring.py`

**Interfaces:**
- Produces: `restrict_config_dir(subdir: str, cols: set[str], types, dest: Path) -> Path` and `mean_scores(df: pd.DataFrame) -> dict` — moved verbatim, same signatures.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_scoring.py
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_mean_scores_selects_only_numeric_type_and_overall():
    from ssdataagent.transfer.scoring import mean_scores
    df = pd.DataFrame([
        {"T1": 0.8, "T2": 0.5, "T3": 0.6, "overall": 0.63, "T3_error": "boom"},
        {"T1": 0.7, "T2": 0.6, "T3": 0.4, "overall": 0.57, "T3_error": "boom"},
    ])
    out = mean_scores(df)
    assert out == {"T1": 0.75, "T2": 0.55, "T3": 0.5, "overall": 0.6}
    assert "T3_error" not in out  # string column starting with 'T' must be excluded


def test_transfer_map_reexports_helpers_from_scoring():
    # transfer_map must import the helpers, not redefine them (single source of truth).
    import transfer_map
    from ssdataagent.transfer import scoring
    assert transfer_map.mean_scores is scoring.mean_scores
    assert transfer_map.restrict_config_dir is scoring.restrict_config_dir
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ssdataagent.transfer.scoring'`.

- [ ] **Step 3: Create `scoring.py` with the two functions moved verbatim**

Cut these two functions out of `scripts/transfer_map.py` (currently `restrict_config_dir` at lines ~53-86 and `mean_scores` at ~89-101) and paste them into the new module. The code is exactly:

```python
# src/ssdataagent/transfer/scoring.py
"""Scoring helpers shared by the transfer scripts (LLM-free).

Extracted from scripts/transfer_map.py so scripts/transfer_b3.py can reuse them
without a script importing a script.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def restrict_config_dir(subdir: str, cols: set[str], types, dest: Path) -> Path:
    """Write type configs restricted to the transferable (crosswalk) ``cols`` under
    ``dest/subdir/``, and return ``dest`` (a ``config_dir`` for nb.score).

    A transfer sim can only carry variables the SOURCE context has, so the target's stock
    config — which may test variables the source lacks (gss2018's depress/mental_health) —
    would KeyError. Restricting ``variables``/``predictors``/``response`` to ``cols`` scores
    exactly the transferable variables. For a pair whose crosswalk already covers the config
    (cps 1970->1980), this is a no-op and reproduces the stock score.
    """
    import yaml
    import nodonor_bracket as nb
    src = nb.CONFIG_DIR / subdir
    (dest / subdir).mkdir(parents=True, exist_ok=True)
    for t in types:
        p = src / f"type{t}.yaml"
        if not p.exists():
            continue
        cfg = yaml.safe_load(p.read_text())
        # T3 carries a model_type list aligned positionally to `response`; when we drop a
        # response we must drop its model_type entry too, or the runner raises
        # "N model types but M responses".
        resp = cfg.get("response")
        mt = cfg.get("model_type")
        if isinstance(resp, dict) and isinstance(mt, list) and len(mt) == len(resp):
            keep_mask = [k in cols for k in resp]
            cfg["model_type"] = [m for m, keep in zip(mt, keep_mask) if keep]
        for key in ("variables", "predictors", "response"):
            if isinstance(cfg.get(key), dict):
                cfg[key] = {k: v for k, v in cfg[key].items() if k in cols}
        # sort_keys=False: preserve `response` dict order so T3's positional `model_type`
        # list stays paired with the right response (see nodonor_bracket._cfg_with_B).
        (dest / subdir / f"type{t}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dest


def mean_scores(df: pd.DataFrame) -> dict:
    """Average the numeric per-type / overall score columns across seeds.

    nb.score() emits per-type rates as ``T1``..``T5`` and ``overall``, but also stores
    per-type FAILURES as string columns ``T{t}_error``. Those also start with 'T', so a
    naive ``startswith('T')`` would call ``.mean()`` on a string column and crash. Select
    only ``overall`` and ``T<digit>`` columns explicitly.
    """
    keep = [c for c in df.columns
            if (c == "overall" or (c.startswith("T") and c[1:].isdigit()))
            and df[c].notna().any()]
    return {c: float(df[c].mean()) for c in keep}
```

- [ ] **Step 4: Rewire `transfer_map.py` to import them**

In `scripts/transfer_map.py`, delete the two function definitions and add to the import block (near the other `ssdataagent.transfer` imports):

```python
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402
```

Leave the call sites (`restrict_config_dir(...)`, `mean_scores(...)`) unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_scoring.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Verify the map still runs offline (regression)**

Run: `.venv/bin/python scripts/transfer_map.py --pairs cps_1970_1980 --no-scoring`
Expected: prints the Layer-1 map for `cps_1970_1980` with no import error and no API key.

- [ ] **Step 7: Commit**

```bash
git add src/ssdataagent/transfer/scoring.py scripts/transfer_map.py tests/test_transfer_scoring.py
git commit -m "refactor(transfer): extract scoring helpers into scoring.py"
```

---

## Task 2: Move `Spec` + cps spec into `src/ssdataagent/transfer/b3_specs.py`

**Files:**
- Create: `src/ssdataagent/transfer/b3_specs.py`
- Modify: `scripts/nodonor_fullmethod.py` (remove `Spec` + `SPECS`; import them)
- Test: `tests/test_b3_specs.py`

**Interfaces:**
- Produces: `Spec` dataclass-like class (constructor `Spec(seeds, derived, predictors, numeric_predictors, log_vars, types, population, rules, glosses)`) and `SPECS: dict[str, Spec]` with key `"cps"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_b3_specs.py
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_cps_spec_is_single_source_of_truth():
    import nodonor_fullmethod as nf
    from ssdataagent.transfer import b3_specs
    assert nf.SPECS["cps"] is b3_specs.SPECS["cps"]  # moved, not duplicated


def test_cps_spec_fields_intact():
    from ssdataagent.transfer.b3_specs import SPECS
    cps = SPECS["cps"]
    assert cps.seeds == ["age", "gender", "race"]
    assert cps.predictors == ["age", "gender", "race", "education"]
    assert cps.numeric_predictors == frozenset({"age"})
    assert cps.log_vars == frozenset({"income"})
    assert cps.types == (1, 2, 3)
    assert "1980" in cps.population
    assert "child_number" in cps.glosses
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_b3_specs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ssdataagent.transfer.b3_specs'`.

- [ ] **Step 3: Create `b3_specs.py`; move `Spec` + `SPECS["cps"]` verbatim**

Cut the `Spec` class (nodonor_fullmethod.py lines ~56-71) and the `SPECS` dict (lines ~74-127) out of `nodonor_fullmethod.py` and paste into the new module. Preserve the cps spec's prose byte-for-byte (it carries the fertility-semantics correction). The module head:

```python
# src/ssdataagent/transfer/b3_specs.py
"""Per-dataset generation specs for the no-donor LLM full method (B3 + the durable
no-donor headline path). Shared by scripts/nodonor_fullmethod.py and scripts/transfer_b3.py
so the cps prompt has one source of truth.
"""
from __future__ import annotations

import pandas as pd


class Spec:
    """Per-dataset wiring. `predictors` and `numeric_predictors` MUST mirror the
    dataset's type3.yaml -- the repair calibrates R^2 against the statistic T3 scores,
    so a mismatch here silently mis-sizes every alpha."""

    def __init__(self, seeds, derived, predictors, numeric_predictors, log_vars,
                 types, population, rules, glosses):
        self.seeds = seeds                          # drawn independently from marginals
        self.derived = derived                      # exact identities, computed not drawn
        self.predictors = predictors                # held coherent through the repair
        self.numeric_predictors = numeric_predictors
        self.log_vars = log_vars
        self.types = types
        self.population = population                 # prose name for the prompt
        self.rules = rules                           # coherence rules for the prompt
        self.glosses = glosses                       # outcome glosses for elicitation


SPECS = {
    # ... paste the existing "cps": Spec(...) entry here verbatim ...
}
```

(Paste the full existing `"cps": Spec(...)` block from nodonor_fullmethod.py into the `SPECS` dict unchanged.)

- [ ] **Step 4: Rewire `nodonor_fullmethod.py` to import them**

In `scripts/nodonor_fullmethod.py`, delete the `Spec` class and the `SPECS` dict, and add near the other `ssdataagent` imports:

```python
from ssdataagent.transfer.b3_specs import SPECS, Spec  # noqa: E402,F401
```

(`Spec` is re-exported for any downstream import; `F401` silences the unused-in-this-file warning.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_b3_specs.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Verify the durable path still imports (regression)**

Run: `.venv/bin/python -c "import sys; sys.path[:0]=['src','scripts','.']; import nodonor_fullmethod as nf; print(sorted(nf.SPECS))"`
Expected: prints `['cps']` with no error.

- [ ] **Step 7: Commit**

```bash
git add src/ssdataagent/transfer/b3_specs.py scripts/nodonor_fullmethod.py tests/test_b3_specs.py
git commit -m "refactor(transfer): move Spec + cps spec into b3_specs.py"
```

---

## Task 3: Add the GSS 2018 spec

**Files:**
- Modify: `src/ssdataagent/transfer/b3_specs.py` (add `SPECS["gss"]`)
- Test: `tests/test_b3_specs.py` (add GSS invariants)

**Interfaces:**
- Consumes: `Spec`, `SPECS` from Task 2.
- Produces: `SPECS["gss"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_b3_specs.py  (append)
def test_gss_spec_invariants():
    from ssdataagent.transfer.b3_specs import SPECS
    gss = SPECS["gss"]
    assert gss.seeds == ["age", "gender", "race"]
    assert gss.predictors == ["age", "gender", "race", "education"]
    assert gss.numeric_predictors == frozenset({"age"})
    assert gss.log_vars == frozenset()          # income is categorical brackets in GSS
    assert gss.derived == {}
    assert gss.types == (1, 2, 3)
    assert "2018" in gss.population and "GSS" in gss.population
    # glosses scoped to exactly the numeric T3 outcomes surviving crosswalk restriction
    assert set(gss.glosses) == {"child_number", "age_first_childbirth", "vocabulary_test"}
    # lifetime-fertility rule present (opposite of the CPS household-roster trap)
    assert "EVER BORN" in gss.glosses["child_number"]
    assert "LIFETIME" in gss.rules or "lifetime" in gss.rules


def test_gss_seeds_and_predictors_are_crosswalk_columns():
    from ssdataagent.transfer.b3_specs import SPECS
    from ssdataagent.transfer.pairs import PAIRS, load_pair
    pair = [p for p in PAIRS if p.id == "gss_1994_2018"][0]
    _, _, cols = load_pair(pair)
    gss = SPECS["gss"]
    assert set(gss.seeds) <= set(cols)
    assert set(gss.predictors) <= set(cols)
    assert set(gss.glosses) <= set(cols)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_b3_specs.py -k gss -v`
Expected: FAIL — `KeyError: 'gss'`.

- [ ] **Step 3: Add the GSS spec to the `SPECS` dict**

Append this entry inside `SPECS` in `src/ssdataagent/transfer/b3_specs.py`:

```python
    # gss is a 2018 cross-section of ADULTS (age 18-89). Unlike cps, child_number is
    # LIFETIME "children ever born" (pool mean ~1.8; stock type3.yaml confirms), so the age
    # relationship runs the OTHER way -- it rises then plateaus with age. income is a
    # categorical bracket, not a dollar amount, so it is not numeric and gets no R^2 repair.
    "gss": Spec(
        seeds=["age", "gender", "race"],
        derived={},
        predictors=["age", "gender", "race", "education"],
        numeric_predictors=frozenset({"age"}),
        log_vars=frozenset(),
        types=(1, 2, 3),
        population="the 2018 US General Social Survey (GSS), adults 18 and older",
        rules="""- Every respondent is an ADULT (18-89). There are no children in this
  sample: everyone can have completed education, a labor-force status, and attitudes.
- FERTILITY IS LIFETIME here, the opposite of a household roster. child_number counts ALL
  children the person has EVER had, so it RISES with age and then plateaus: near 0 in the
  late teens/early 20s, climbing to ~2 by the 40s and staying there into old age. Do NOT
  let it fall for older people.
    * child_number 0 <=> age_first_childbirth 'No Child'. A nonzero count needs a real
      first-birth age, always >= 12 and < the person's own age.
    * age_first_childbirth is the LIFETIME first birth, typically 18-30, only weakly
      related to current age; more-educated people tend to start later.
- Education (Less than high school / High school / College and above) is bounded by age
  only at the young end (a 19-year-old is rarely 'College and above' yet).
- vocabulary_test (0-10) rises with education and is roughly flat in age.
- income is a categorical BRACKET ('$10000 OR MORE' dominates; 'Unemployed' when not
  earning), not a dollar amount. occupation is a broad census category; 'Unemployed' or
  'Military Occupations' are valid. spouse_occupation is 'No spouse' for the unmarried.
- Attitudes (gender_role_attitude, political_view, trust, happy, satisfy_job, work_hard)
  and health should read as one coherent person, plausible against the marginals below.
- immigrant_status and parental background (mother/father education & occupation) are
  inferred coherently, kept plausible against the marginals.""",
        glosses={
            "child_number": "total number of children EVER BORN in the respondent's "
                            "lifetime (GSS lifetime fertility, NOT resident children); "
                            "0..8, mean ~1.8",
            "age_first_childbirth": "age at the respondent's FIRST live birth over their "
                                    "lifetime ('No Child' if they never had a child)",
            "vocabulary_test": "number of correct words (0-10) on the GSS WORDSUM "
                               "vocabulary test, an indicator of verbal/cognitive skill",
        },
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_b3_specs.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/b3_specs.py tests/test_b3_specs.py
git commit -m "feat(transfer): GSS 2018 spec for B3 (lifetime-fertility rules)"
```

---

## Task 4: B3 column derivation + rung-alpha computation (pure logic)

**Files:**
- Create: `scripts/transfer_b3.py` (the two pure functions + imports)
- Test: `tests/test_transfer_b3.py`

**Interfaces:**
- Consumes: `SPECS` (b3_specs), `load_pair`/`PAIRS` (pairs), `covariate_r2`/`variance_repair_alphas` (conditional_variance).
- Produces:
  - `b3_columns(pair) -> tuple[str, list[str], "Spec", list[str], list[str]]` returning `(ds, cols, spec, predictors, downstream)` where `cols` is the crosswalk restricted to source∩pool∩ref columns (mirroring `run_layer2`), `predictors = spec.predictors` filtered to `cols`, `downstream = [c for c in cols if c not in spec.seeds and c not in spec.derived]`.
  - `rung_alphas(raw, spec, elicited, pool, predictors) -> dict[str, dict[str, float]]` returning `{"B3_raw": {c: 1.0 ...}, "B3_elicited": {...}, "B3_pool_R2": {...}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_b3.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_b3_columns_restricts_and_splits():
    import transfer_b3
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    ds, cols, spec, predictors, downstream = transfer_b3.b3_columns(pair)
    assert ds == "cps"
    assert "birth_year" not in cols                       # non-transferable, dropped
    assert set(predictors) <= set(cols)
    assert set(downstream) == set(cols) - set(spec.seeds) - set(spec.derived)
    for c in spec.seeds:
        assert c not in downstream


def test_rung_alphas_raw_is_all_ones_and_pool_uses_pool_r2():
    import transfer_b3
    from ssdataagent.transfer.b3_specs import SPECS
    rng = np.random.default_rng(0)
    n = 300
    # synthetic raw + pool with a real age->income signal so R^2 is estimable
    age = rng.integers(18, 80, n)
    income = 500 * age + rng.normal(0, 3000, n)
    frame = pd.DataFrame({
        "age": age, "gender": rng.choice(["Male", "Female"], n),
        "race": rng.choice(["Black", "Hispanic", "Non-Black, Non-Hispanic"], n),
        "education": rng.choice(["High school", "College and above"], n),
        "income": income,
    })
    spec = SPECS["cps"]
    predictors = ["age", "gender", "race", "education"]
    elicited = {"income": 0.5}
    alphas = transfer_b3.rung_alphas(frame, spec, elicited, frame, predictors)
    assert set(alphas) == {"B3_raw", "B3_elicited", "B3_pool_R2"}
    assert alphas["B3_raw"] == {"income": 1.0}            # raw = no repair
    assert alphas["B3_pool_R2"]["income"] > 0             # a real alpha from pool R^2
    # pool R^2 alpha differs from the elicited-0.5 alpha (different targets)
    assert alphas["B3_pool_R2"]["income"] != alphas["B3_elicited"]["income"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_b3.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transfer_b3'`.

- [ ] **Step 3: Create `scripts/transfer_b3.py` with the two pure functions**

```python
#!/usr/bin/env python
"""B3 -- the no-donor LLM full method (nodonor_fullmethod stages) pointed at the TARGET
context, restricted to the crosswalk columns and scored on the identical footing as
B0/B1/B2. Reports three rungs: B3_raw (no repair), B3_elicited (LLM-elicited R^2), and
B3_pool_R2 (pool covariate-R^2, same aggregate footing as B2).

See docs/superpowers/specs/2026-07-23-b3-llm-prior-pointed-at-target-design.md.

    export OPENROUTER_API_KEY=...            # first run only (gss); cps is cached
    .venv/bin/python scripts/transfer_b3.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
    .venv/bin/python scripts/transfer_b3.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import ssdataagent.data.conditional_variance as cv  # noqa: E402
from ssdataagent.transfer.b3_specs import SPECS  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, load_pair  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def b3_columns(pair):
    """(ds, cols, spec, predictors, downstream). `cols` is the crosswalk restricted to the
    source, target pool, and reference columns -- IDENTICAL to run_layer2's `cols` so B3 is
    scored on the same variables as B0/B1/B2."""
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema
    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    ref = nb._drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    b_pool, _ = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    spec = SPECS[ds]
    predictors = [c for c in spec.predictors if c in cols]
    downstream = [c for c in cols if c not in spec.seeds and c not in spec.derived]
    return ds, cols, spec, predictors, downstream


def rung_alphas(raw, spec, elicited, pool, predictors):
    """The three rungs' alpha dicts. raw = all 1.0 (no repair); elicited = alphas from the
    LLM-elicited R^2 targets; pool_R2 = alphas from the pool's covariate-R^2 (a low-order
    aggregate, same footing as B2). Numeric-only outcomes are repaired; cv skips the rest."""
    outcomes = [c for c in raw.columns if c not in predictors]
    elic_targets = {c: elicited[c] for c in outcomes
                    if c not in spec.predictors and elicited.get(c) is not None}
    pool_targets = {c: cv.covariate_r2(pool, c, predictors,
                                       numeric_predictors=spec.numeric_predictors,
                                       log_vars=spec.log_vars)
                    for c in elic_targets}
    elic_alpha = cv.variance_repair_alphas(
        raw, predictors, elic_targets,
        numeric_predictors=spec.numeric_predictors, log_vars=spec.log_vars)
    pool_alpha = cv.variance_repair_alphas(
        raw, predictors, {c: t for c, t in pool_targets.items() if t is not None},
        numeric_predictors=spec.numeric_predictors, log_vars=spec.log_vars)
    return {
        "B3_raw": {c: 1.0 for c in elic_alpha},
        "B3_elicited": elic_alpha,
        "B3_pool_R2": pool_alpha,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_b3.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_b3.py tests/test_transfer_b3.py
git commit -m "feat(transfer): B3 column derivation + rung-alpha logic"
```

---

## Task 5: B3 orchestrator + CLI + cps warm-cache end-to-end gate

**Files:**
- Modify: `scripts/transfer_b3.py` (add `run_b3` + `main`)
- Test: `tests/test_transfer_b3.py` (add the end-to-end cps test)

**Interfaces:**
- Consumes: `b3_columns`, `rung_alphas` (Task 4); `generate`, `elicit` (nodonor_fullmethod); `sample_variance_repaired`, `score`, `carve_pool` (cv / nb).
- Produces: `run_b3(pair, *, seeds, n, bootstrap_B, people=480, batch=20, regenerate=False) -> pd.DataFrame` (rows: `pair`, `config` in {B3_raw,B3_elicited,B3_pool_R2}, `guarantee`, `T1`,`T2`,`T3`,`overall`); writes `results/transfer_map/b3_<pair.id>.csv`.

- [ ] **Step 1: Write the failing test (deterministic, off the warm cps cache)**

```python
# tests/test_transfer_b3.py  (append)
import pytest


def _cps_cache_warm():
    cache = REPO / "results" / "nodonor_cache"
    return (cache / "cps_cond_raw.csv").exists() and (cache / "cps_elicit.json").exists()


@pytest.mark.skipif(not _cps_cache_warm(), reason="cps LLM cache not present")
def test_run_b3_cps_off_warm_cache(tmp_path, monkeypatch):
    import transfer_b3
    from ssdataagent.transfer.pairs import PAIRS
    # redirect the output CSV into tmp so the test never clobbers a real result
    monkeypatch.setattr(transfer_b3, "OUT", tmp_path)
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    # small but real: 2 seeds, n=800, cheap bootstrap -- still deterministic off the cache
    df = transfer_b3.run_b3(pair, seeds=2, n=800, bootstrap_B=50)
    assert list(df["config"]) == ["B3_raw", "B3_elicited", "B3_pool_R2"]
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all()
        assert (df[col] >= 0).all() and (df[col] <= 1).all()
    out = tmp_path / "b3_cps_1970_1980.csv"
    assert out.exists()
    # B3_pool_R2 should not score below B3_raw on T3 by more than noise (repair helps or ties)
    raw_t3 = float(df.loc[df.config == "B3_raw", "T3"].iloc[0])
    pool_t3 = float(df.loc[df.config == "B3_pool_R2", "T3"].iloc[0])
    assert pool_t3 >= raw_t3 - 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_b3.py -k run_b3 -v`
Expected: FAIL — `AttributeError: module 'transfer_b3' has no attribute 'run_b3'`.

- [ ] **Step 3: Add `run_b3` and `main` to `scripts/transfer_b3.py`**

```python
# append to scripts/transfer_b3.py

def run_b3(pair, *, seeds, n, bootstrap_B, people=480, batch=20, regenerate=False):
    """Generate (LLM, cached) -> elicit (cached) -> score three rungs through the restricted
    config against the target reference. Firewalled: reads only the target pool's marginals
    and (pool_R2 rung) its covariate-R^2, never its joint or the reference sample."""
    import tempfile

    import nodonor_bracket as nb
    import nodonor_fullmethod as nf
    from ssdataagent.data.schema import load_schema

    ds, cols, spec, predictors, downstream = b3_columns(pair)
    pool, guarantee = nb.carve_pool(ds)
    ref = nb._drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    types = nb.TYPES.get(ds, (1, 2, 3))

    raw = nf.generate(ds, pool, cols, spec, people, batch, regenerate)
    elicited = nf.elicit(ds, cols, spec, regenerate)
    alphas = rung_alphas(raw, spec, elicited, pool, predictors)

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(load_schema(ds).ssdatabench_sim_subdir,
                                      set(cols), types, Path(cfg_td))
        for name, alpha in alphas.items():
            recs = []
            for s in range(1, seeds + 1):
                sim = cv.sample_variance_repaired(raw, pool, cols, predictors, n,
                                                  np.random.default_rng(s),
                                                  alpha=alpha, default_alpha=0.5)
                recs.append(nb.score(sim, ds, ref, types, seed=1000 + s,
                                     bootstrap_B=bootstrap_B, config_dir=cfg_dir))
            row = {"pair": pair.id, "config": name, "guarantee": guarantee}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"b3_{pair.id}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", choices=[p.id for p in PAIRS if p.scored])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--people", type=int, default=480)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()
    pair = [p for p in PAIRS if p.id == a.pair][0]
    df = run_b3(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B,
                people=a.people, batch=a.batch, regenerate=a.regenerate)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'b3_{pair.id}.csv'}")
    print("REGIME: no-donor. Target supplies marginals + (pool_R2 rung) covariate-R^2 only.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the end-to-end test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_b3.py -k run_b3 -v`
Expected: PASS (deterministic off the warm cps cache; no API calls).

- [ ] **Step 5: Run the full B3 test file**

Run: `.venv/bin/python -m pytest tests/test_transfer_b3.py -v`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/transfer_b3.py tests/test_transfer_b3.py
git commit -m "feat(transfer): B3 orchestrator + CLI + cps warm-cache end-to-end"
```

---

## Task 6: Run both pairs, write the report, add the LEDGER row, rebuild the dashboard

**Files:**
- Create: `results/transfer_map/b3_cps_1970_1980.csv`, `results/transfer_map/b3_gss_1994_2018.csv` (run outputs)
- Create: `docs/report/2026-07-23-b3-llm-prior-pointed-at-target.md`
- Modify: `docs/experiments/LEDGER.md`
- Modify: `docs/dashboard/index.html` (regenerated)

**Interfaces:**
- Consumes: `run_b3` / the CLI (Task 5).

> This task runs live code and records real numbers, so its steps produce values that cannot be pre-written. Follow the sequence; fill the report tables from the actual CSV outputs.

- [ ] **Step 1: Run cps B3 (deterministic, off the warm cache), full protocol**

Run: `.venv/bin/python scripts/transfer_b3.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200`
Expected: prints three rung rows with finite T1/T2/T3/overall; writes `results/transfer_map/b3_cps_1970_1980.csv`. Confirm the file mtime is now (per the sentinel-bug lesson — a stale file must not be mistaken for a fresh run).

- [ ] **Step 2: Sanity-check cps B3_raw against the durable no-donor path**

Run: `.venv/bin/python scripts/nodonor_fullmethod.py cps --seeds 3 --n 3000 --bootstrap-B 200`
Expected: its "raw (no repair)" overall is within noise (~0.05) of `b3_cps_1970_1980.csv`'s `B3_raw` overall. A large divergence means the restricted-config scoring path differs from the durable path — investigate before proceeding. (Small differences are expected: B3 scores on the crosswalk-restricted config; for cps 1970→1980 the crosswalk covers the config, so they should be close.)

- [ ] **Step 3: Run gss B3 (live API — generates the cold cache, then scores)**

Ensure `OPENROUTER_API_KEY` is in `.env` (it is). Run:
`.venv/bin/python scripts/transfer_b3.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200`
Expected: the `generate` stage makes ~24 batches of LLM completions (writing `results/nodonor_cache/gss_cond_raw.csv`), `elicit` writes `gss_elicit.json`, then three rung rows print and `results/transfer_map/b3_gss_1994_2018.csv` is written. If a batch retries on a network blip, that is normal (fixed backoff). Confirm the output CSV mtime is now.

- [ ] **Step 4: Read both CSVs and the matching B1/B2 rows for the comparison**

Run: `.venv/bin/python -c "import pandas as pd,glob; [print(f,'\n',pd.read_csv(f).to_string(index=False),'\n') for f in ['results/transfer_map/b3_cps_1970_1980.csv','results/transfer_map/b3_gss_1994_2018.csv','results/transfer_map/baselines_cps_1970_1980.csv','results/transfer_map/baselines_gss_1994_2018.csv']]"`
Expected: prints the three B3 rungs per pair and the B0/B1/B2/floor/ceiling rows to compare against.

- [ ] **Step 5: Write the report**

Create `docs/report/2026-07-23-b3-llm-prior-pointed-at-target.md`. Structure (fill tables from Step 4's numbers — no placeholders in the final file):
- **The question**: LLM prior as the structure source vs transplanted source A; the decision gate.
- **Result 1**: the B0/B1/B2/B3 ladder table per pair (T1/T2/T3/overall), floor + ceiling.
- **Result 2 — the two gate comparisons**: `B3_pool_R2` vs `B2` (structure source, strength held equal) and `B3_elicited` vs `B2` (fully firewalled prior). State which, if either, closes B2's residual.
- **Firewall** paragraph (row-level; marginals + pool R² + codebook only).
- **Decision gate**: does a large mechanism-shift residual survive both B2 and B3? → the Phase-3 verdict.
- **Limits**: carry forward the spec's Limits section (two genuine pairs, `mental_health` absent, attitude measurement non-invariance, same-country only, row-level firewall). If cps B3_raw diverged from the durable path in Step 2, disclose it.

- [ ] **Step 6: Add the LEDGER row**

Append a `b3_llm_prior` row to `docs/experiments/LEDGER.md` following the existing column format, with a one-line `hypothesis` (e.g. "LLM prior pointed at the target closes B2's residual for free — decides Phase 3") and `git_sha` `_pending_` (set after the final commit).

- [ ] **Step 7: Rebuild the dashboard**

Run: `.venv/bin/python scripts/build_dashboard.py`
Expected: regenerates `docs/dashboard/index.html` without error.

- [ ] **Step 8: Commit**

```bash
git add results/transfer_map/b3_cps_1970_1980.csv results/transfer_map/b3_gss_1994_2018.csv \
        docs/report/2026-07-23-b3-llm-prior-pointed-at-target.md \
        docs/experiments/LEDGER.md docs/dashboard/index.html
git commit -m "report: B3 LLM-prior-pointed-at-target ladder + decision gate"
```

---

## Self-Review

**1. Spec coverage:**
- Approach A (thin script + extracted `scoring.py` + `b3_specs.py`, map stays LLM-free) → Tasks 1, 2, 4, 5. ✓
- GSS Spec with lifetime-fertility rules + 3-outcome R² scope → Task 3. ✓
- Three rungs (raw/elicited/pool-R²) + two gate comparisons → Tasks 4 (alphas), 5 (scoring), 6 (report). ✓
- Firewall (row-level, marginals + pool R² + codebook) → enforced in `run_b3`/`rung_alphas`, disclosed in Task 6 report. ✓
- Variable set pinned to benchmark → `b3_columns` uses the identical crosswalk; predictors filtered to cols; glosses scoped to numeric T3 outcomes (Task 3 test). ✓
- Testing: extraction parity (T1), cps identity (T2), GSS invariants (T3), column restriction + rung alphas (T4), cps warm-cache end-to-end + mtime (T5/T6). ✓
- `mental_health` gap, attitude non-invariance, two-pairs limit → Task 6 report Limits. ✓

**2. Placeholder scan:** No TBD/TODO. Task 6 legitimately fills report tables from live numbers (a run-then-write task, flagged as such); all code steps carry complete code.

**3. Type consistency:** `b3_columns` returns `(ds, cols, spec, predictors, downstream)` — consumed with those exact names in `run_b3` and the tests. `rung_alphas(raw, spec, elicited, pool, predictors)` — same arg order at its one call site. `run_b3` writes columns `pair, config, guarantee, T1..overall` — matched by the Task 5 test and the Task 6 comparison. `SPECS`/`Spec` names consistent across Tasks 2/3/4. `restrict_config_dir`/`mean_scores` signatures unchanged from the verbatim move.
