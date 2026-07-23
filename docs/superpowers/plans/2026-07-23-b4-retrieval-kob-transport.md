# B4 — Retrieval + KOB transport (Phase 3, slice 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transport the target context's joint Y-structure (T2 associations + T3 covariate-R²) from leave-one-context-out siblings, raked to the target's public X-marginals, through the existing B1/B2 shared-latent machinery — reading none of the target's Y-side joint aggregates — and score it beside B0–B3 to decompose the B2 residual into composition vs mechanism.

**Architecture:** Thin recombination of shipped parts. A new `retrieval.py` builds a reweighted-pooled sibling pseudo-population (`sib_rew`) using the existing `raking_weights`. One keyword-only parameter added to the existing `transfer_build_b2` lets its Step-B R² target come from `sib_rew` (headline, fully firewalled) or the target pool (diagnostic). A new `scripts/transfer_b4.py` orchestrates and scores, mirroring `scripts/transfer_b3.py`. LLM-free, deterministic off microdata.

**Tech Stack:** Python, pandas, numpy; existing modules `ssdataagent.transfer.{decompose,generate,target_aggregates,scoring,pairs}`, `ssdataagent.data.conditional_variance`, `nodonor_bracket`.

## Global Constraints

- **LLM-free / no API key.** B4 is fully deterministic off microdata. No network, no OpenRouter.
- **Comparability — identical to `run_layer2`/`transfer_b3` in every scoring input:** the crosswalk `cols` (source `a` ∩ target pool ∩ reference), `restrict_config_dir(schema.ssdatabench_sim_subdir, set(cols), types, ...)`, reference = `load_schema(pair.target_dataset).real_data_path`, seed offset `1000+s`, `bootstrap_B`, and `mean_scores` aggregation. B4's `cols` MUST equal B2's `cols` — never re-derive them from the sibling pool.
- **B2 stays bit-for-bit unchanged.** The new `transfer_build_b2` parameter is keyword-only with a default that reproduces today's behavior exactly. The existing tests in `tests/test_transfer_generate_b2.py` must stay green untouched.
- **Firewall.** B4 reads only: the target pool's univariate marginals (X and Y, via `_marginal_map`) and the target pool's `CORE_DEMOGRAPHICS` margins (for raking). B4 NEVER reads the target's per-person joint, pairwise associations, covariate-R², or the benchmark reference sample. Siblings contribute microdata (allowed — LOCO holds out only the target wave).
- **Retrieval = same-instrument, leave-one-context-out, uniform raking-pool.** No cross-instrument mixing, no nearest-neighbor distance metric. Report `effective_sample_size/len` (ESS ratio) and the per-sibling R² spread as diagnostics — never silently drop them.
- **`results/` is gitignored.** Durable artifacts are the report, the LEDGER row, and the dashboard rebuild — not the CSVs.
- After landing, regenerate the dashboard per `AGENTS.md` (`.venv/bin/python scripts/build_dashboard.py`).

---

## File Structure

- **Create** `src/ssdataagent/transfer/retrieval.py` — sibling discovery + reweighted-pooled pseudo-population. Pure/testable; filesystem only in `sibling_csvs`.
- **Modify** `src/ssdataagent/transfer/generate.py` — add keyword-only `r2_pool=None` to `transfer_build_b2`; one line changes which frame Step-B reads its R² target from.
- **Create** `scripts/transfer_b4.py` — orchestrator (`b4_columns`, `reweighted_pool`, `per_sibling_r2`, `run_b4`, `main`), mirroring `scripts/transfer_b3.py`.
- **Create** `tests/test_transfer_retrieval.py`, **create** `tests/test_transfer_b4.py`, **extend** `tests/test_transfer_generate_b2.py`.
- **Create** `docs/report/2026-07-23-b4-retrieval-kob-transport.md`; **modify** `docs/experiments/LEDGER.md`; **regenerate** `docs/dashboard/index.html`.

---

### Task 1: Retrieval module — reweighted-pooled sibling pseudo-population

**Files:**
- Create: `src/ssdataagent/transfer/retrieval.py`
- Test: `tests/test_transfer_retrieval.py`

**Interfaces:**
- Consumes: `ssdataagent.transfer.decompose.raking_weights(a, b, covariates, *, bins=10, iters=30) -> np.ndarray` (per-row weights on `a` so its weighted `covariates` marginals match `b`'s), `ssdataagent.transfer.decompose.effective_sample_size(w) -> float` (Kish ESS).
- Produces:
  - `sibling_csvs(pair) -> list[Path]` — same-instrument sibling CSVs (every `*.csv` in the target wave's dataset directory except the target wave), sorted.
  - `reweighted_pool(sib_frames: list[pd.DataFrame], target_pool: pd.DataFrame, cols: list[str], rake_cols: list[str], n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, float]` — returns `(sib_rew, ess_ratio)`: siblings concatenated on `cols`, raked to `target_pool`'s `rake_cols` margins, weighted-resampled to `n` rows.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer_retrieval.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def test_sibling_csvs_loco_excludes_target():
    from ssdataagent.transfer.pairs import PAIRS
    from ssdataagent.transfer import retrieval
    cps = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    sibs = retrieval.sibling_csvs(cps)
    names = {p.name for p in sibs}
    assert cps.target_csv.name not in names          # LOCO: target held out
    assert "cps-asec1970.csv" in names                # designated source is a valid sibling
    assert len(sibs) == 3                             # 1970, 1990, 2000
    gss = [p for p in PAIRS if p.id == "gss_1994_2018"][0]
    assert len(retrieval.sibling_csvs(gss)) == 1       # degenerate: only 1994


def test_reweighted_pool_matches_target_margin_and_reports_ess():
    from ssdataagent.transfer import retrieval
    rng = np.random.default_rng(0)
    # sibling stack skews YOUNG; target skews OLD. After raking on age the resampled
    # pool's age composition must move toward the target's.
    sib = pd.DataFrame({"age": np.r_[np.full(800, 25), np.full(200, 65)],
                        "income": rng.normal(0, 1, 1000)})
    target = pd.DataFrame({"age": np.r_[np.full(200, 25), np.full(800, 65)],
                           "income": rng.normal(0, 1, 1000)})
    sib_rew, ess = retrieval.reweighted_pool([sib], target, ["age", "income"], ["age"],
                                             n=4000, rng=rng)
    assert len(sib_rew) == 4000
    frac_old = (sib_rew["age"] == 65).mean()
    assert frac_old > 0.6                              # raked toward target's 0.8, up from 0.2
    assert 0.0 < ess <= 1.0                             # Kish ratio is a reliability signal


def test_reweighted_pool_is_deterministic():
    from ssdataagent.transfer import retrieval
    sib = pd.DataFrame({"age": [25, 65, 25, 65] * 50, "income": list(range(200))})
    target = pd.DataFrame({"age": [65] * 100, "income": list(range(100))})
    a, _ = retrieval.reweighted_pool([sib], target, ["age", "income"], ["age"], 300,
                                     np.random.default_rng(7))
    b, _ = retrieval.reweighted_pool([sib], target, ["age", "income"], ["age"], 300,
                                     np.random.default_rng(7))
    assert a.equals(b)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: ssdataagent.transfer.retrieval` (module not yet created).

- [ ] **Step 3: Write the implementation**

```python
# src/ssdataagent/transfer/retrieval.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.transfer.decompose import effective_sample_size, raking_weights


def sibling_csvs(pair) -> list[Path]:
    """Same-instrument sibling CSVs under leave-one-context-out: every ``*.csv`` in the
    target wave's dataset directory except the target wave itself. The designated source
    wave IS a sibling (only the target is held out). Sorted for determinism."""
    tgt = pair.target_csv.resolve()
    return sorted(p for p in pair.target_csv.parent.glob("*.csv")
                  if p.resolve() != tgt)


def reweighted_pool(sib_frames: list[pd.DataFrame], target_pool: pd.DataFrame,
                    cols: list[str], rake_cols: list[str], n: int,
                    rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Concatenate siblings on ``cols``, rake to ``target_pool``'s ``rake_cols`` marginals
    (IPF, the KOB composition transport), and draw a weighted resample of ``n`` rows.

    Returns ``(sib_rew, ess_ratio)``. Raking simultaneously corrects each sibling's
    composition toward the target and pools siblings (a composition-nearer sibling gets
    more weight). ``ess_ratio`` is the Kish effective sample size over the stack size: a
    low value means the raking concentrated weight on a few rows -- a thin transport the
    caller must surface. Reads only the target's ``rake_cols`` margins (public X-margins);
    never the target's joint."""
    stack = pd.concat([f[cols] for f in sib_frames], ignore_index=True)
    w = raking_weights(stack, target_pool, rake_cols)
    ess_ratio = effective_sample_size(w) / len(w) if len(w) else 0.0
    idx = rng.choice(len(stack), size=n, replace=True, p=w / w.sum())
    return stack.iloc[idx].reset_index(drop=True), ess_ratio
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_retrieval.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/retrieval.py tests/test_transfer_retrieval.py
git commit -m "feat(transfer): LOCO sibling retrieval + raked pseudo-population for B4"
```

---

### Task 2: `transfer_build_b2` — pluggable R² source

**Files:**
- Modify: `src/ssdataagent/transfer/generate.py:94-170` (`transfer_build_b2`)
- Test: `tests/test_transfer_generate_b2.py` (extend)

**Interfaces:**
- Consumes: existing `ssdataagent.transfer.target_aggregates.target_aggregates(pool, cols, covariates, outcomes) -> dict` (uses `agg["outcome_r2"]` only).
- Produces: `transfer_build_b2(source_pool, target_pool, cols, covariates, outcomes, n, seed, *, r2_pool: pd.DataFrame | None = None) -> pd.DataFrame`. When `r2_pool is None` (default): the Step-B covariate-R² target is read from `target_pool` — **byte-identical to today**. When `r2_pool` is provided: the R² target is read from `r2_pool` instead; marginals still come from `target_pool`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_transfer_generate_b2.py`)

```python
def test_b2_r2_pool_none_is_byte_identical_to_default():
    # The new keyword-only param must not perturb the default path at all.
    import numpy as np
    import pandas as pd
    from ssdataagent.transfer.generate import transfer_build_b2
    rng = np.random.default_rng(3)
    a = pd.DataFrame({"age": rng.integers(20, 70, 500),
                      "education": rng.choice(["HS", "College"], 500),
                      "income": rng.normal(50000, 15000, 500)})
    b = pd.DataFrame({"age": rng.integers(20, 70, 500),
                      "education": rng.choice(["HS", "College"], 500),
                      "income": rng.normal(60000, 15000, 500)})
    cols, cov, out_y = ["age", "education", "income"], ["age", "education"], ["income"]
    base = transfer_build_b2(a, b, cols, cov, out_y, n=2000, seed=11)
    same = transfer_build_b2(a, b, cols, cov, out_y, n=2000, seed=11, r2_pool=None)
    assert base.equals(same)


def test_b2_r2_pool_changes_the_recalibration_target():
    # Pointing Step B at a DIFFERENT R^2 source must change the output (proves the R^2
    # target is sourced from r2_pool, not target_pool). r2_pool has a MUCH stronger
    # age->income signal than the target, so the recalibrated frames must differ.
    import numpy as np
    import pandas as pd
    from ssdataagent.transfer.generate import transfer_build_b2
    rng = np.random.default_rng(5)
    a = pd.DataFrame({"age": rng.integers(20, 70, 800),
                      "income": 300 * rng.integers(20, 70, 800) + rng.normal(0, 20000, 800)})
    target = pd.DataFrame({"age": rng.integers(20, 70, 800),
                           "income": rng.normal(50000, 20000, 800)})   # ~no signal
    strong = pd.DataFrame({"age": rng.integers(20, 70, 800),
                           "income": 1500 * rng.integers(20, 70, 800)})  # very strong signal
    cols, cov, out_y = ["age", "income"], ["age"], ["income"]
    via_target = transfer_build_b2(a, target, cols, cov, out_y, n=3000, seed=9)
    via_strong = transfer_build_b2(a, target, cols, cov, out_y, n=3000, seed=9, r2_pool=strong)
    assert not via_target.equals(via_strong)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_generate_b2.py -v`
Expected: FAIL — `test_b2_r2_pool_none_is_byte_identical_to_default` and `test_b2_r2_pool_changes_the_recalibration_target` error with `unexpected keyword argument 'r2_pool'`.

- [ ] **Step 3: Write the implementation**

In `src/ssdataagent/transfer/generate.py`, change the `transfer_build_b2` signature and the single `target_aggregates` call. Signature line (currently ends `n: int, seed: int) -> pd.DataFrame:`):

```python
def transfer_build_b2(source_pool: pd.DataFrame, target_pool: pd.DataFrame,
                      cols: list[str], covariates: list[str], outcomes: list[str],
                      n: int, seed: int, *,
                      r2_pool: pd.DataFrame | None = None) -> pd.DataFrame:
```

Add to the docstring (after the existing Step B paragraph):

```
    ``r2_pool`` (keyword-only, default ``None``) chooses which frame the Step-B
    covariate-R^2 target is read from. ``None`` -> read it from ``target_pool``
    (byte-identical to the original B2). A supplied frame (e.g. B4's reweighted
    sibling pseudo-population) sources the R^2 target from THAT frame instead, while
    the marginals still come from ``target_pool``. B4_retrieval passes its sib_rew
    here to keep the R^2 target off the target's Y-side aggregates entirely.
```

Then change the aggregates line (currently `agg = target_aggregates(target_pool, cols, covariates, outcomes)`) to:

```python
    r2_frame = target_pool if r2_pool is None else r2_pool
    agg = target_aggregates(r2_frame, cols, covariates, outcomes)
```

Everything else in the function stays exactly as-is (marginals from `target_pool`, the shared-latent draw, `bidirectional_r2_blend`).

- [ ] **Step 4: Run the full B2 test file to verify pass + no regression**

Run: `.venv/bin/python -m pytest tests/test_transfer_generate_b2.py -v`
Expected: PASS — the 2 new tests pass AND all pre-existing B2 tests (`test_b2_matches_target_marginals_and_shape`, `test_b2_recalibrates_outcome_r2_toward_target`, `test_b2_matches_b1_exactly_with_no_outcomes`, `test_b2_does_not_drop_target_nonnumeric_subpopulation`) stay green.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/generate.py tests/test_transfer_generate_b2.py
git commit -m "feat(transfer): transfer_build_b2 pluggable r2_pool (B2 default unchanged)"
```

---

### Task 3: `scripts/transfer_b4.py` orchestrator

**Files:**
- Create: `scripts/transfer_b4.py`
- Test: `tests/test_transfer_b4.py`

**Interfaces:**
- Consumes: `retrieval.{sibling_csvs, reweighted_pool}` (Task 1); `generate.transfer_build_b2(..., r2_pool=...)` (Task 2); `target_aggregates` (for the per-sibling R² diagnostic); `pairs.{PAIRS, load_pair, covariates_outcomes}`; `scoring.{restrict_config_dir, mean_scores}`; `nodonor_bracket` as `nb` (`nb._drop_unnamed`, `nb.carve_pool(ds) -> (pool, guarantee)`, `nb.score(sim, ds, ref, types, seed=, bootstrap_B=, config_dir=) -> dict`, `nb.TYPES`); `transfer_map.composition_covariates`; `load_schema(ds).{real_data_path, ssdatabench_sim_subdir}`.
- Produces:
  - `b4_columns(pair) -> (ds, cols, covs, outs)` — `cols` derived IDENTICALLY to `run_layer2` (source `a` ∩ target pool ∩ reference), so B4 scores on the same variables as B0–B3.
  - `reweighted_pool_for(pair, cols, target_pool, rng) -> (sib_rew, ess_ratio, used_waves, dropped_waves)` — load LOCO siblings, keep those containing all `cols`, build the raked pool.
  - `per_sibling_r2(sib_frames_by_wave, target_pool, cols, covs, outs, rake_cols, rng) -> dict[str, dict]` — each sibling's individually-raked outcome-R² bundle (diagnostic; the spread is the finding).
  - `run_b4(pair, *, seeds, n, bootstrap_B, n_rew=None) -> pd.DataFrame` — build `sib_rew` once → score `B4_retrieval` (r2_pool=sib_rew) and `B4_retrieval_targetR2` (r2_pool=None) over seeds → write `results/transfer_map/b4_<pair>.csv`.
  - `main()` — CLI mirroring `transfer_b3.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer_b4.py
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_b4_columns_match_layer2_cols():
    # B4 must score on the SAME crosswalk variables as B0-B3, or the comparison is invalid.
    import transfer_b4
    import pandas as pd
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema
    from ssdataagent.transfer.pairs import PAIRS, load_pair
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    _, cols_b4, _, _ = transfer_b4.b4_columns(pair)
    # replicate run_layer2's derivation
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    ref = nb._drop_unnamed(pd.read_csv(load_schema(pair.target_dataset).real_data_path,
                                       low_memory=False))
    b_pool, _ = nb.carve_pool(pair.target_dataset)
    _, _, cols = load_pair(pair)
    expected = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    assert cols_b4 == expected


def test_run_b4_smoke_scores_both_configs(tmp_path, monkeypatch):
    # Small but real off the CPS microdata (no API key). Both configs score, bounded.
    import transfer_b4
    from ssdataagent.transfer.pairs import PAIRS
    monkeypatch.setattr(transfer_b4, "OUT", tmp_path)
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    df = transfer_b4.run_b4(pair, seeds=2, n=800, bootstrap_B=50)
    assert list(df["config"]) == ["B4_retrieval", "B4_retrieval_targetR2"]
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all()
        assert (df[col] >= 0).all() and (df[col] <= 1).all()
    assert (tmp_path / "b4_cps_1970_1980.csv").exists()
    assert "ess_ratio" in df.columns and (df["ess_ratio"] > 0).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_b4.py -v`
Expected: FAIL — `ModuleNotFoundError: transfer_b4`.

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python
"""B4 -- retrieval + KOB transport (Phase 3, slice 1). Transport the target's joint
Y-structure (T2 associations + T3 covariate-R^2) from leave-one-context-out same-instrument
siblings, raked to the target's public X-marginals, through the existing B1/B2 shared-latent
machinery. Reads NO target Y-side joint aggregate. Two configs:

  B4_retrieval          -- R^2 target transported from the reweighted siblings (fully
                           firewalled: reads no target covariate-R^2).
  B4_retrieval_targetR2 -- R^2 target kept from the target pool (B2's source); isolates the
                           retrieval/reweighting effect. Diagnostic.

See docs/superpowers/specs/2026-07-23-b4-retrieval-kob-transport-design.md. LLM-free.

    .venv/bin/python scripts/transfer_b4.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
    .venv/bin/python scripts/transfer_b4.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
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

from ssdataagent.transfer.generate import transfer_build_b2  # noqa: E402
from ssdataagent.transfer.pairs import (  # noqa: E402
    PAIRS, covariates_outcomes, load_pair,
)
from ssdataagent.transfer.retrieval import reweighted_pool, sibling_csvs  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402
from ssdataagent.transfer.target_aggregates import target_aggregates  # noqa: E402
from transfer_map import composition_covariates  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def b4_columns(pair):
    """(ds, cols, covs, outs). ``cols`` is derived IDENTICALLY to run_layer2 -- source `a`
    ∩ target pool ∩ reference -- so B4 is scored on the same variables as B0-B3."""
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema
    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    ref = nb._drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    b_pool, _ = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    covs, outs = covariates_outcomes(pair.schema_name, cols)
    return ds, cols, covs, outs


def _load_siblings(pair, cols):
    """LOCO sibling frames restricted to ``cols``; keep only siblings containing every
    crosswalk column (a sibling missing a col cannot carry that variable's structure).
    Returns (frames_by_wave, dropped) -- both dicts wave_name -> frame / reason."""
    import nodonor_bracket as nb
    frames, dropped = {}, {}
    for csv in sibling_csvs(pair):
        f = nb._drop_unnamed(pd.read_csv(csv, low_memory=False))
        missing = [c for c in cols if c not in f.columns]
        if missing:
            dropped[csv.stem] = f"missing {missing}"
        else:
            frames[csv.stem] = f[cols]
    return frames, dropped


def reweighted_pool_for(pair, cols, target_pool, rng, n_rew=None):
    """Build the raked sibling pseudo-population for ``pair``. n_rew defaults to the stack
    size (preserve scale). Returns (sib_rew, ess_ratio, used_waves, dropped)."""
    # composition_covariates(cols) returns {age,gender,race} ∩ cols (the raking axes),
    # falling back to all cols only if none of the demographic core survived the crosswalk.
    rake_cols = composition_covariates(cols)
    frames, dropped = _load_siblings(pair, cols)
    if not frames:
        raise RuntimeError(f"{pair.id}: no usable siblings (dropped: {dropped})")
    sib_list = list(frames.values())
    stack_size = sum(len(f) for f in sib_list)
    sib_rew, ess = reweighted_pool(sib_list, target_pool, cols, rake_cols,
                                   n_rew or stack_size, rng)
    return sib_rew, ess, list(frames), dropped


def per_sibling_r2(pair, cols, covs, outs, target_pool, rng):
    """Each sibling's individually-raked outcome-R^2 (diagnostic spread). A wide spread
    across waves is itself the finding: mechanism drifts -> the learned model is needed."""
    rake_cols = composition_covariates(cols)
    frames, _ = _load_siblings(pair, cols)
    out = {}
    for wave, f in frames.items():
        sr, _ = reweighted_pool([f], target_pool, cols, rake_cols, len(f), rng)
        out[wave] = target_aggregates(sr, cols, covs, outs)["outcome_r2"]
    return out


def run_b4(pair, *, seeds, n, bootstrap_B, n_rew=None):
    """Build sib_rew once -> score B4_retrieval (r2_pool=sib_rew) and B4_retrieval_targetR2
    (r2_pool=None, i.e. target pool) over seeds -> write results CSV. Firewalled: reads the
    target's marginals + CORE_DEMOGRAPHICS margins only; the joint Y-structure is
    transported from siblings."""
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema

    ds, cols, covs, outs = b4_columns(pair)
    target_pool, guarantee = nb.carve_pool(ds)
    ref = nb._drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    types = nb.TYPES.get(ds, (1, 2, 3))

    sib_rew, ess, used, dropped = reweighted_pool_for(
        pair, cols, target_pool, np.random.default_rng(0), n_rew)
    print(f"{pair.id}: siblings used {used}; dropped {dropped}; "
          f"sib_rew {len(sib_rew)} rows; ess_ratio {ess:.3f}")
    spread = per_sibling_r2(pair, cols, covs, outs, target_pool, np.random.default_rng(0))
    print(f"{pair.id}: per-sibling outcome_r2 (spread diagnostic): {spread}")

    configs = {
        "B4_retrieval": dict(r2_pool=sib_rew),        # transported R^2 -- fully firewalled
        "B4_retrieval_targetR2": dict(r2_pool=None),  # keep B2's target R^2 -- diagnostic
    }
    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(load_schema(ds).ssdatabench_sim_subdir,
                                      set(cols), types, Path(cfg_td))
        for name, kw in configs.items():
            recs = []
            for s in range(1, seeds + 1):
                sim = transfer_build_b2(sib_rew, target_pool, cols, covs, outs, n, s, **kw)
                recs.append(nb.score(sim, ds, ref, types, seed=1000 + s,
                                     bootstrap_B=bootstrap_B, config_dir=cfg_dir))
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "ess_ratio": round(ess, 4)}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"b4_{pair.id}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", choices=[p.id for p in PAIRS if p.scored])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    a = ap.parse_args()
    pair = [p for p in PAIRS if p.id == a.pair][0]
    df = run_b4(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'b4_{pair.id}.csv'}")
    print("REGIME: no-donor + stricter. Target supplies marginals + X-margins (raking) only;"
          " joint Y-structure transported from LOCO siblings.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_b4.py -v`
Expected: PASS (2 passed). The smoke test runs the real CPS microdata off-cache; allow ~60–120s.

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_b4.py tests/test_transfer_b4.py
git commit -m "feat(transfer): B4 retrieval+KOB orchestrator (two configs, ESS + spread diagnostics)"
```

---

### Task 4: Full-protocol run + report + LEDGER + dashboard

**Files:**
- Create: `docs/report/2026-07-23-b4-retrieval-kob-transport.md`
- Modify: `docs/experiments/LEDGER.md`
- Regenerate: `docs/dashboard/index.html`

**Interfaces:**
- Consumes: `scripts/transfer_b4.py` (Task 3), the existing B2 numbers (cps 0.663 / gss 0.683) and ceilings (cps 0.816 / gss 0.811) from `docs/report/2026-07-23-b2-aggregate-recalibration.md` and the B3 report.

- [ ] **Step 1: Run both scored pairs at full protocol**

Run (each solo — the box reaps two concurrent heavy jobs; see memory `project_b3_llm_prior`):
```bash
.venv/bin/python scripts/transfer_b4.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_b4.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```
Record: per-config T1/T2/T3/overall, `ess_ratio`, the printed per-sibling R² spread, and `used`/`dropped` siblings. gss will report a single sibling (1994) — expected; note it.

- [ ] **Step 2: Write the report**

Create `docs/report/2026-07-23-b4-retrieval-kob-transport.md` with:
- The decomposition table: B1 / B2 / B4_retrieval_targetR2 / B4_retrieval / ceiling, per pair, per T-type, overall.
- The three deltas from the spec (retrieval+reweighting effect = B4_targetR2 − B2; firewall cost = B4_retrieval − B4_targetR2; headline = B4_retrieval − B2), read against the ~0.054 noise floor.
- ESS ratio per pair and the per-sibling R² spread; state the gss single-sibling limitation.
- **The verdict on the estimand:** is the B2 residual composition-transportable (B4 closes part of it reading zero target Y-aggregates) or genuine mechanism shift (B4 ≤ B2)? State what it implies for Phase 3 slice 2 (the learned model).
- Firewall paragraph: exactly what B4 reads and does not read.

- [ ] **Step 3: Add the LEDGER row**

Append one row to `docs/experiments/LEDGER.md` in the existing column format, with a meaningful `hypothesis` one-liner (the dashboard shows it): e.g. "B4: transporting the target's joint Y-structure from raked LOCO siblings — does the B2 residual close without reading any target Y-aggregate?" Use `git_sha` `_pending_` consistent with the B2/B3 rows.

- [ ] **Step 4: Regenerate the dashboard**

```bash
.venv/bin/python scripts/build_dashboard.py
```

- [ ] **Step 5: Commit**

```bash
git add docs/report/2026-07-23-b4-retrieval-kob-transport.md docs/experiments/LEDGER.md docs/dashboard/index.html
git commit -m "report: B4 retrieval+KOB transport -- decompose the B2 residual"
```

---

## Self-Review

**Spec coverage:** estimand (T2 via re-sourced structure, T3 via transported R²) → Tasks 1–3; two-config decomposition → Task 3 `run_b4`; firewall (marginals + X-margins only) → Task 1 `reweighted_pool` + Task 2 `r2_pool` + Task 3 wiring; comparability → Task 3 `b4_columns` + `test_b4_columns_match_layer2_cols`; ESS + per-sibling spread diagnostics → Task 3; LOCO/same-instrument/gss-degeneracy limits → Task 1 `sibling_csvs` + Task 4 report; deliverable verdict → Task 4. All spec sections covered.

**Placeholder scan:** none — every step has concrete code or exact commands.

**Type consistency:** `reweighted_pool(sib_frames, target_pool, cols, rake_cols, n, rng) -> (DataFrame, float)` consumed by Task 3 exactly as produced in Task 1. `transfer_build_b2(..., *, r2_pool=None)` produced in Task 2, called with `r2_pool=sib_rew`/`r2_pool=None` in Task 3. `b4_columns -> (ds, cols, covs, outs)` matches its Task 3 consumers. `sibling_csvs(pair) -> list[Path]` consumed by `_load_siblings`. Consistent.
