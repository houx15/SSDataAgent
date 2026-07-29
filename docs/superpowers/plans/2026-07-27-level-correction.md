# Level-Correction Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a level-correction channel that keeps source A's marginal shape + copula and corrects only each numeric outcome's location for the target, estimated from oracle / LLM-description / sibling-pooled / ESS-gated-hybrid signals, scored T1–T5 against the ladder.

**Architecture:** A new pure module `transfer/levelcorrect.py` holds the shift math (`numeric_outcomes`, `outcome_mean`, `oracle_shifts`, `pooled_shifts`, `hybrid_shifts`, `apply_level_shift`), the LLM location elicitation (`llm_level_prompt`, `parse_levels`, `llm_shifts`), and the arm-assembly orchestrator (`assemble_shifts`). A runner `scripts/transfer_levelcorrect.py` clones the face-swap harness, builds five `transfer_build` configs plus three references, and scores them.

**Tech Stack:** Python 3, pandas, numpy; reuses `transfer.generate.transfer_build`, `transfer.retrieval`, `transfer.rescue.select_r2_source`, `transfer.blind` (MODEL + `_last_json_object` + `blind_specs`), and the `nodonor_bracket` scorer.

## Global Constraints

- **Analyst arms read only what their firewall allows:** `LC_llm` reads B's audited *description* only (`blind_specs.BLIND_SPECS`), `LC_pooled`/`LC_hybrid` read B's *public X-margins* only (raking), `LC_oracle` reads B's Y (labeled ceiling). Do not let a feasible arm read B's Y.
- **Reuse, don't reinvent:** copula/marginals = `transfer_build`; siblings = `retrieval.sibling_csvs`/`reweighted_pool`; hybrid gate = `rescue.select_r2_source`; numeric test mirrors `nodonor_bracket._is_numeric`; LLM client/cache pattern + `_last_json_object` from `transfer.blind`; scoring = `nodonor_bracket.score` + `scoring.restrict_config_dir`/`mean_scores`.
- **Location statistic = mean; shift = additive** (`marg[y] = to_numeric(a[y]) + Δ_y`).
- **`OPENROUTER_API_KEY` lives in `.env`** — never print it; source silently. LLM elicitation is cached durably to `results/levelcorrect_cache/<ds>_levels.json` (gitignored) so scoring needs no key.
- **No `/dev/null` / out-of-repo redirects** (hook blocks them). **Avoid the literal word "eval"** in commit messages. **`results/` is gitignored.** **Never stage the `ssdatabench` submodule** — stage only the files each task names.
- Model for LLM elicitation: `anthropic/claude-sonnet-4.5` (== `blind.MODEL`).

---

### Task 1: Pure shift math

**Files:**
- Create: `src/ssdataagent/transfer/levelcorrect.py`
- Test: `tests/test_transfer_levelcorrect.py`

**Interfaces:**
- Consumes: `ssdataagent.transfer.rescue.select_r2_source`.
- Produces:
  - `numeric_outcomes(a, b, outs) -> list[str]`
  - `outcome_mean(frame, y) -> float`
  - `oracle_shifts(a, b, ys) -> dict[str, float]`
  - `pooled_shifts(a, sib_rew, ys) -> dict[str, float]`
  - `hybrid_shifts(pooled, llm, n_siblings, ess) -> dict[str, float]`
  - `apply_level_shift(marg, shifts) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer_levelcorrect.py
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_numeric_outcomes_picks_numeric_in_both():
    from ssdataagent.transfer.levelcorrect import numeric_outcomes
    a = pd.DataFrame({"inc": [1, 2, 3], "occ": ["x", "y", "z"], "kid": [0, 1, 2]})
    b = pd.DataFrame({"inc": [4, 5, 6], "occ": ["p", "q", "r"], "kid": [1, 2, 3]})
    assert numeric_outcomes(a, b, ["inc", "occ", "kid"]) == ["inc", "kid"]


def test_oracle_and_pooled_shifts_are_mean_diffs():
    from ssdataagent.transfer.levelcorrect import oracle_shifts, pooled_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0, 0.0, 0.0]})
    b = pd.DataFrame({"inc": [5.0, 5.0, 5.0, 5.0]})
    assert abs(oracle_shifts(a, b, ["inc"])["inc"] - 5.0) < 1e-9
    assert abs(pooled_shifts(a, b, ["inc"])["inc"] - 5.0) < 1e-9


def test_apply_level_shift_moves_mean_preserves_shape_and_others():
    from ssdataagent.transfer.levelcorrect import apply_level_shift, outcome_mean
    a = pd.DataFrame({"inc": [1.0, 2.0, 3.0, np.nan], "occ": ["x", "y", "z", "w"]})
    out = apply_level_shift(a, {"inc": 10.0})
    assert abs(outcome_mean(out, "inc") - (outcome_mean(a, "inc") + 10.0)) < 1e-9
    assert abs(pd.to_numeric(out["inc"]).var() - pd.to_numeric(a["inc"]).var()) < 1e-9
    assert out["inc"].isna().sum() == 1            # NaN preserved
    assert list(out["occ"]) == list(a["occ"])      # other columns untouched


def test_apply_level_shift_zero_nonfinite_and_missing_are_noops():
    from ssdataagent.transfer.levelcorrect import apply_level_shift
    a = pd.DataFrame({"inc": [1.0, 2.0]})
    assert list(apply_level_shift(a, {"inc": 0.0})["inc"]) == [1.0, 2.0]
    assert list(apply_level_shift(a, {"inc": float("nan")})["inc"]) == [1.0, 2.0]
    assert list(apply_level_shift(a, {"missing": 5.0}).columns) == ["inc"]


def test_hybrid_shifts_gate_routes_pooled_vs_llm():
    from ssdataagent.transfer.levelcorrect import hybrid_shifts
    pooled, llm = {"inc": 3.0}, {"inc": 7.0}
    assert hybrid_shifts(pooled, llm, n_siblings=3, ess=0.6)["inc"] == 3.0   # plural+effective
    assert hybrid_shifts(pooled, llm, n_siblings=1, ess=0.6)["inc"] == 7.0   # thin -> llm
    # an outcome absent from the chosen arm falls back to the other
    assert hybrid_shifts({}, {"inc": 7.0}, n_siblings=3, ess=0.6)["inc"] == 7.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_levelcorrect.py -v`
Expected: FAIL with `ModuleNotFoundError` (module not created).

- [ ] **Step 3: Create the module with the pure functions**

```python
# src/ssdataagent/transfer/levelcorrect.py
"""Level-correction channel: keep source A's marginal SHAPE + copula, correct only the
per-outcome LOCATION for the target. See
docs/superpowers/specs/2026-07-27-level-correction-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.rescue import select_r2_source


def _is_numeric(s: pd.Series) -> bool:
    """>90% of non-missing values parse as numbers. Mirrors nodonor_bracket._is_numeric
    (the scorer's test) so the shifted set matches what the scorer grades as numeric."""
    s = s.dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def numeric_outcomes(a: pd.DataFrame, b: pd.DataFrame, outs: list[str]) -> list[str]:
    """Outcomes numeric in BOTH frames (order preserved)."""
    return [y for y in outs if y in a.columns and y in b.columns
            and _is_numeric(a[y]) and _is_numeric(b[y])]


def outcome_mean(frame: pd.DataFrame, y: str) -> float:
    """NaN-aware mean of a numeric column (NaN if empty)."""
    v = pd.to_numeric(frame[y], errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def oracle_shifts(a: pd.DataFrame, b: pd.DataFrame, ys: list[str]) -> dict[str, float]:
    """Δ_y = mean_B(y) - mean_A(y) (reads B -- the ceiling arm)."""
    return {y: outcome_mean(b, y) - outcome_mean(a, y) for y in ys}


def pooled_shifts(a: pd.DataFrame, sib_rew: pd.DataFrame, ys: list[str]) -> dict[str, float]:
    """Δ_y = mean(sib_rew[y]) - mean_A(y): siblings raked to B's public X-margins."""
    return {y: outcome_mean(sib_rew, y) - outcome_mean(a, y) for y in ys}


def hybrid_shifts(pooled: dict[str, float], llm: dict[str, float],
                  n_siblings: int, ess: float) -> dict[str, float]:
    """Per outcome, the ESS-gated fuse (B6 gate): the pooled (retrieval) shift when the
    sibling pool is plural AND effectively-sized, else the llm (description) shift. An
    outcome missing/non-finite in the chosen arm falls back to the other (0.0 if neither)."""
    use_pooled = select_r2_source(n_siblings, ess)
    primary, backup = (pooled, llm) if use_pooled else (llm, pooled)
    out: dict[str, float] = {}
    for y in set(pooled) | set(llm):
        val = primary.get(y)
        if val is None or not np.isfinite(val):
            val = backup.get(y, 0.0)
        out[y] = float(val if val is not None and np.isfinite(val) else 0.0)
    return out


def apply_level_shift(marg: pd.DataFrame, shifts: dict[str, float]) -> pd.DataFrame:
    """Copy of ``marg`` with each numeric column named in ``shifts`` shifted by its Δ (NaN
    preserved); all other columns byte-identical. A zero / non-finite Δ or a column absent
    from ``marg`` is a no-op."""
    out = marg.copy()
    for y, d in shifts.items():
        if y not in out.columns or not np.isfinite(d) or d == 0.0:
            continue
        out[y] = pd.to_numeric(out[y], errors="coerce") + d
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_levelcorrect.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/levelcorrect.py tests/test_transfer_levelcorrect.py
git commit -m "feat(levelcorrect): pure location-shift math + ESS-gated hybrid"
```

---

### Task 2: LLM location elicitation

**Files:**
- Modify: `src/ssdataagent/transfer/levelcorrect.py`
- Test: `tests/test_transfer_levelcorrect.py`

**Interfaces:**
- Consumes: `ssdataagent.transfer.blind.MODEL`, `ssdataagent.transfer.blind._last_json_object`, `ssdataagent.transfer.blind_specs.BLIND_SPECS`.
- Produces:
  - `llm_level_prompt(ds, a, ys) -> str`
  - `parse_levels(text, ys) -> dict[str, float]`
  - `llm_shifts(a, ds, ys, *, client=None, cache_dir=None, regenerate=False) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_transfer_levelcorrect.py

class _FakeClient:
    """Minimal stand-in for the OpenRouter client: .chat.completions.create(...) returns an
    object whose choices[0].message.content is the canned text."""
    def __init__(self, content):
        self._content = content
    @property
    def chat(self):
        return self
    @property
    def completions(self):
        return self
    def create(self, model, messages):
        msg = type("M", (), {"content": self._content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_llm_shifts_uses_elicited_mean(tmp_path):
    from ssdataagent.transfer.levelcorrect import llm_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0, 0.0, 0.0], "kid": [2.0, 2.0, 2.0, 2.0]})
    client = _FakeClient('{"inc": 5, "kid": 3}')
    sh = llm_shifts(a, "cps", ["inc", "kid"], client=client, cache_dir=tmp_path)
    assert abs(sh["inc"] - 5.0) < 1e-9        # 5 - mean 0
    assert abs(sh["kid"] - 1.0) < 1e-9        # 3 - mean 2
    assert (tmp_path / "cps_levels.json").exists()


def test_llm_shifts_drops_junk_entries(tmp_path):
    from ssdataagent.transfer.levelcorrect import llm_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0]})
    sh = llm_shifts(a, "cps", ["inc"], client=_FakeClient('{"inc": "NaN-ish"}'), cache_dir=tmp_path)
    assert "inc" not in sh                    # dropped -> carryover for that outcome


def test_llm_shifts_cache_hit_skips_client(tmp_path):
    import json
    from ssdataagent.transfer.levelcorrect import llm_shifts
    (tmp_path / "cps_levels.json").write_text(json.dumps({"inc": 9.0}))
    a = pd.DataFrame({"inc": [1.0, 1.0]})

    class Boom:
        @property
        def chat(self):
            raise AssertionError("client must not be called on a cache hit")
    sh = llm_shifts(a, "cps", ["inc"], client=Boom(), cache_dir=tmp_path)
    assert abs(sh["inc"] - 8.0) < 1e-9        # 9 - mean 1, from cache
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_levelcorrect.py -k llm_shifts -v`
Expected: FAIL with `ImportError` (`llm_shifts` not defined).

- [ ] **Step 3: Add the elicitation code**

Add these imports to the existing import block in `levelcorrect.py`:

```python
import json
import os
from pathlib import Path

from ssdataagent.transfer.blind import MODEL, _last_json_object
```

Add a module constant below the imports:

```python
_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO / "results" / "levelcorrect_cache"
```

Then append:

```python
def llm_level_prompt(ds: str, a: pd.DataFrame, ys: list[str]) -> str:
    """Ask ONLY for the population MEAN of each numeric outcome in the described target
    context. Reads B's audited description (blind_specs); never B's data."""
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    spec = BLIND_SPECS[ds]
    body = "\n".join(f'- "{y}": {spec.glosses.get(y, y)}' for y in ys)
    return (
        f"You are estimating population averages in {spec.population}.\n"
        f"Context:\n{spec.description}\n\n"
        f"Using ONLY your knowledge of this described population -- no external data -- give "
        f"the approximate population MEAN (average) value of EACH numeric variable below. "
        f"Reply with ONE JSON object mapping each variable name to a single number.\n\n"
        f"{body}\n\nOutput only the JSON."
    )


def parse_levels(text: str, ys: list[str]) -> dict[str, float]:
    """Parse the LLM JSON into {y: mean}; keep only finite numeric entries for ys, drop junk."""
    raw = _last_json_object(text)
    out: dict[str, float] = {}
    for y in ys:
        try:
            fv = float(raw.get(y))
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out[y] = fv
    return out


def llm_shifts(a: pd.DataFrame, ds: str, ys: list[str], *, client=None,
               cache_dir: Path | None = None, regenerate: bool = False) -> dict[str, float]:
    """Δ_y = llm_mean(y) - mean_A(y), with llm_mean elicited from B's DESCRIPTION only and
    cached to <cache_dir>/<ds>_levels.json. An outcome the LLM omits/garbles is dropped (no
    shift -> carryover for it). Reads no B data. Warm cache returns without calling the LLM."""
    cache_dir = cache_dir or _DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ds}_levels.json"
    if path.exists() and not regenerate:
        levels = json.loads(path.read_text())
    else:
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url="https://openrouter.ai/api/v1",
                            api_key=os.environ["OPENROUTER_API_KEY"])
        prompt = llm_level_prompt(ds, a, ys)
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}])
        levels = parse_levels(resp.choices[0].message.content, ys)
        path.write_text(json.dumps(levels, ensure_ascii=False, indent=2))
    return {y: float(levels[y]) - outcome_mean(a, y) for y in ys if y in levels}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_levelcorrect.py -v`
Expected: all pass (8 total).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/levelcorrect.py tests/test_transfer_levelcorrect.py
git commit -m "feat(levelcorrect): LLM location elicitation from description (cached)"
```

---

### Task 3: Arm assembly + runner script + real run

**Files:**
- Modify: `src/ssdataagent/transfer/levelcorrect.py`
- Create: `scripts/transfer_levelcorrect.py`
- Test: `tests/test_transfer_levelcorrect.py`

**Interfaces:**
- Consumes: all Task 1/2 symbols; `retrieval.sibling_csvs`/`reweighted_pool`; `nodonor_bracket` (`_drop_unnamed`, `carve_pool`, `build`, `score`, `TYPES`); `generate.transfer_build`; `pairs` (`PAIRS`, `covariates_outcomes`, `load_pair`); `scoring` (`restrict_config_dir`, `mean_scores`); `schema.load_schema`.
- Produces:
  - `assemble_shifts(a, b, sib_rew, ds, ys, n_siblings, ess, *, client=None, cache_dir=None) -> dict[str, dict]` (keys `oracle, llm, pooled, hybrid`).
  - `scripts/transfer_levelcorrect.py` with `run_levelcorrect(pair, *, seeds, n, bootstrap_B, dry_run=False)` and a `--dry-run` CLI flag.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transfer_levelcorrect.py

def test_assemble_shifts_wires_all_four_arms(tmp_path):
    from ssdataagent.transfer.levelcorrect import assemble_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0, 0.0, 0.0]})
    b = pd.DataFrame({"inc": [10.0, 10.0, 10.0, 10.0]})          # oracle Δ = 10
    sib_rew = pd.DataFrame({"inc": [4.0, 4.0, 4.0, 4.0]})        # pooled Δ = 4
    shifts = assemble_shifts(a, b, sib_rew, "cps", ["inc"], n_siblings=3, ess=0.6,
                             client=_FakeClient('{"inc": 7}'), cache_dir=tmp_path)  # llm Δ = 7
    assert abs(shifts["oracle"]["inc"] - 10.0) < 1e-9
    assert abs(shifts["pooled"]["inc"] - 4.0) < 1e-9
    assert abs(shifts["llm"]["inc"] - 7.0) < 1e-9
    assert abs(shifts["hybrid"]["inc"] - 4.0) < 1e-9             # 3 sib, ess .6 -> pooled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_levelcorrect.py -k assemble -v`
Expected: FAIL with `ImportError` (`assemble_shifts` not defined).

- [ ] **Step 3a: Add `assemble_shifts` to the module**

Append to `levelcorrect.py`:

```python
def assemble_shifts(a: pd.DataFrame, b: pd.DataFrame, sib_rew: pd.DataFrame, ds: str,
                    ys: list[str], n_siblings: int, ess: float, *, client=None,
                    cache_dir: Path | None = None) -> dict[str, dict]:
    """Assemble the four arms' shift dicts for one pair: oracle (reads B), llm (description),
    pooled (siblings raked to B's public X-margins), hybrid (ESS-gated fuse of pooled+llm)."""
    oracle = oracle_shifts(a, b, ys)
    llm = llm_shifts(a, ds, ys, client=client, cache_dir=cache_dir)
    pooled = pooled_shifts(a, sib_rew, ys)
    hybrid = hybrid_shifts(pooled, llm, n_siblings, ess)
    return {"oracle": oracle, "llm": llm, "pooled": pooled, "hybrid": hybrid}
```

- [ ] **Step 3b: Create the runner script**

Create `scripts/transfer_levelcorrect.py`:

```python
#!/usr/bin/env python
"""Level-correction channel -- location-shifted marginals for cross-context transfer.
Keeps source A's marginal SHAPE + copula; corrects only each numeric outcome's LOCATION for
the target from oracle / LLM-description / sibling-pooled / ESS-gated-hybrid estimates. See
docs/superpowers/specs/2026-07-27-level-correction-design.md.

    export OPENROUTER_API_KEY=...     # first run only; level elicitation is cached
    .venv/bin/python scripts/transfer_levelcorrect.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.data.schema import load_schema  # noqa: E402
from ssdataagent.transfer.generate import transfer_build  # noqa: E402
from ssdataagent.transfer.levelcorrect import (  # noqa: E402
    apply_level_shift, assemble_shifts, numeric_outcomes, oracle_shifts, pooled_shifts,
)
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes, load_pair  # noqa: E402
from ssdataagent.transfer.retrieval import reweighted_pool, sibling_csvs  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

RAKE_CANDIDATES = ("age", "gender", "race")
POOL_N = 50000          # weighted resample size for a stable pooled-mean estimate
OUT = REPO / "results" / "transfer_map"


def _pool(pair, a, b_pool, cols, covs, *, seed=0):
    """Sibling pool raked to B's public X-margins -> (sib_rew, ess, n_siblings)."""
    import nodonor_bracket as nb
    sib_paths = sibling_csvs(pair)
    sibs = [nb._drop_unnamed(pd.read_csv(p, low_memory=False)) for p in sib_paths]
    rake_cols = [c for c in RAKE_CANDIDATES if c in covs]
    rng = np.random.default_rng(seed)
    sib_rew, ess = reweighted_pool(sibs, b_pool, cols, rake_cols, POOL_N, rng)
    return sib_rew, ess, len(sibs)


def run_levelcorrect(pair, *, seeds, n, bootstrap_B, dry_run=False):
    """Score LC_none(==B0), LC_oracle, LC_llm, LC_pooled, LC_hybrid, ref_oracle_comp(==B1),
    ref_floor, ref_ceiling on the crosswalk cols -- identical protocol to B0-B6/face-swap.
    Only ref_oracle_comp + LC_oracle + the scorer touch B's pool; LC_llm reads B's
    description; LC_pooled/LC_hybrid read B's public X-margins."""
    import nodonor_bracket as nb

    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    schema = load_schema(ds)
    ref = nb._drop_unnamed(pd.read_csv(schema.real_data_path, low_memory=False))
    b_pool, guarantee = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    covs, outs = covariates_outcomes(pair.schema_name, cols)
    types = nb.TYPES.get(ds, (1, 2, 3))

    ys = numeric_outcomes(a, b_pool, outs)
    sib_rew, ess, n_sib = _pool(pair, a, b_pool, cols, covs)

    if dry_run:
        # Real-data wiring check without the LLM or the scorer (deterministic arms only).
        print(f"{pair.id}: numeric outcomes={ys}", flush=True)
        print(f"  siblings={n_sib} ess={ess:.3f} rake_cols={[c for c in RAKE_CANDIDATES if c in covs]}",
              flush=True)
        print(f"  oracle Δ={ {y: round(v, 3) for y, v in oracle_shifts(a, b_pool, ys).items()} }",
              flush=True)
        print(f"  pooled Δ={ {y: round(v, 3) for y, v in pooled_shifts(a, sib_rew, ys).items()} }",
              flush=True)
        return None

    shifts = assemble_shifts(a, b_pool, sib_rew, ds, ys, n_sib, ess)
    print(f"{pair.id}: numeric outcomes {ys}; siblings {n_sib} ess {ess:.3f}", flush=True)

    configs = {
        "LC_none":         lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "LC_oracle":       lambda s: transfer_build(a, apply_level_shift(a, shifts["oracle"]),
                                                    cols, n, s, "marginal-swap"),
        "LC_llm":          lambda s: transfer_build(a, apply_level_shift(a, shifts["llm"]),
                                                    cols, n, s, "marginal-swap"),
        "LC_pooled":       lambda s: transfer_build(a, apply_level_shift(a, shifts["pooled"]),
                                                    cols, n, s, "marginal-swap"),
        "LC_hybrid":       lambda s: transfer_build(a, apply_level_shift(a, shifts["hybrid"]),
                                                    cols, n, s, "marginal-swap"),
        "ref_oracle_comp": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
        "ref_floor":       lambda s: nb.build(b_pool, cols, n, s, "independence"),
        "ref_ceiling":     lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
    }

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(schema.ssdatabench_sim_subdir, set(cols), types,
                                      Path(cfg_td))
        for name, builder in configs.items():
            recs = [nb.score(builder(s), ds, ref, types, seed=1000 + s,
                             bootstrap_B=bootstrap_B, config_dir=cfg_dir)
                    for s in range(1, seeds + 1)]
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "n_numeric": len(ys), "ess": round(ess, 3), "n_siblings": n_sib}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"levelcorrect_{pair.id}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_id")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print shifts on real data (no LLM, no scoring)")
    args = ap.parse_args()
    pair = next(p for p in PAIRS if p.id == args.pair_id)
    run_levelcorrect(pair, seeds=args.seeds, n=args.n, bootstrap_B=args.bootstrap_B,
                     dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit test, then the real-data dry-run**

Run: `.venv/bin/python -m pytest tests/test_transfer_levelcorrect.py -v`
Expected: all pass (9 total).

Then verify the real-data wiring (no API key, no scorer) for both pairs:

Run: `.venv/bin/python scripts/transfer_levelcorrect.py cps_1970_1980 --dry-run`
Run: `.venv/bin/python scripts/transfer_levelcorrect.py gss_1994_2018 --dry-run`
Expected: each prints the numeric outcomes, `siblings N ess X.XXX`, and finite oracle/pooled Δ dicts. cps should report 3 siblings, gss 1 sibling (thin ess). If either raises, report BLOCKED with the traceback — do NOT wrap arms in try/except or narrow the pair.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/levelcorrect.py scripts/transfer_levelcorrect.py tests/test_transfer_levelcorrect.py
git commit -m "feat(levelcorrect): arm assembly + scored runner (LC ladder)"
```

**Note for the controller (post-implementation, not the implementer):** the full scored run
across both pairs (all five arms incl. `LC_llm`) needs `OPENROUTER_API_KEY` for the first
elicitation and is heavy — run it as a controller step with the reap-safe resumable scorer,
then write the report + LEDGER-style findings. `LC_none`/`LC_oracle`/`LC_pooled`/`ref_*` are
deterministic and reproduce offline once the elicitation cache is warm.

---

## Self-Review

**Spec coverage:** location-shift mechanism (Task 1 `apply_level_shift`); the five arms —
`LC_none`/`ref_*` in the runner, `LC_oracle`/`LC_pooled` (Task 1), `LC_llm` (Task 2),
`LC_hybrid` (Task 1 `hybrid_shifts` + Task 3 `assemble_shifts`); numeric-only via
`numeric_outcomes`; comparability via the cloned face-swap harness (`LC_none`=="carryover",
`ref_oracle_comp`=="marginal-swap" on `b_pool`); scoring protocol identical (`nb.score`,
`restrict_config_dir`, `mean_scores`, 3 seeds/n=3000/B=200). H1/H2/H3 are answered by the
config set. Firewall per arm enforced by what each arm reads.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** shift dicts are `dict[str, float]` throughout; `assemble_shifts` returns
`{"oracle","llm","pooled","hybrid"}` consumed by the runner's `shifts[...]`; `apply_level_shift`
takes `(DataFrame, dict)` -> DataFrame at every call site; `reweighted_pool(sibs, b_pool, cols,
rake_cols, POOL_N, rng)` matches its signature; `select_r2_source(n_siblings, ess)` matches;
`transfer_build(struct, marg, cols, n, seed, mode)` matches; `nb.score(sim, ds, ref, types,
seed=, bootstrap_B=, config_dir=)` matches the face-swap call.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-27-level-correction.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
