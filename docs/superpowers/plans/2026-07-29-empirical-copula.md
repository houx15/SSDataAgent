# Empirical-copula Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current one-factor generator with a faithful **empirical-copula** transfer that preserves source A's full joint (fixing the categorical dependence the `glat` mechanism loses), then score it to confirm the large T2/T3 gain the diagnostic predicted.

**Architecture:** A new pure function `transfer/empirical_copula.py::empirical_transfer` mirrors `generate.transfer_build` but computes each categorical column's copula value from the resampled row's *actual* category interval (joint-preserving) instead of the shared `glat` ordering. A runner scores it against the current engine and the ladder references.

**Tech Stack:** Python 3, numpy, pandas (no new deps); reuses `generate._is_numeric` / `_marginal_map` and the `nodonor_bracket` scorer.

## Global Constraints

- **Only the categorical latent changes** vs `transfer_build`: numeric columns, the `_marginal_map` inverse-CDF, and the missingness handling are identical, so `EC_carry` (marg == source) reproduces a direct row-resample of A's joint.
- **Category ordering must match `_marginal_map`'s:** both use `value_counts(normalize=True)` (frequency-descending) so the source interval and the target inverse-CDF are on the same monotone scale.
- **Reuse** `generate._is_numeric`, `generate._marginal_map`; scorer via `scripts/transfer_faceswap.py`'s pattern (`nodonor_bracket as nb`, `restrict_config_dir`, `mean_scores`).
- **`results/` gitignored.** **Avoid the literal word "eval"** in commit messages. **No `/dev/null` / out-of-repo redirects.** **Never stage the `ssdatabench` submodule** — stage only the files each task names.
- **Local, no server needed.**

---

### Task 1: `empirical_transfer`

**Files:**
- Create: `src/ssdataagent/transfer/empirical_copula.py`
- Test: `tests/test_transfer_empirical.py`

**Interfaces:**
- Consumes: `generate._is_numeric`, `generate._marginal_map`.
- Produces: `empirical_transfer(source: pd.DataFrame, marg: pd.DataFrame, cols: list[str], n: int, seed: int) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer_empirical.py
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_empirical_preserves_categorical_association():
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    from ssdataagent.transfer.generate import transfer_build
    from ssdataagent.transfer.copula_stability import pair_association
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.choice(["a", "b", "c"], n)
    y = np.where(x == "a", "p", np.where(x == "b", "q", "r")).astype(object)
    flip = rng.random(n) < 0.05
    y[flip] = rng.choice(["p", "q", "r"], int(flip.sum()))
    A = pd.DataFrame({"x": x, "y": y})
    v_true = pair_association(A, "x", "y")[0]
    ec = empirical_transfer(A, A, ["x", "y"], n, 1)
    tb = transfer_build(A, A, ["x", "y"], n, 1, "carryover")
    v_ec = pair_association(ec, "x", "y")[0]
    v_tb = pair_association(tb, "x", "y")[0]
    assert v_ec > 0.8 * v_true, f"EC lost the association: {v_ec} vs true {v_true}"
    assert v_ec >= v_tb - 1e-6, f"EC {v_ec} should be >= transfer_build {v_tb}"


def test_empirical_preserves_numeric_rank_copula():
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    n = 3000
    z = rng.normal(size=n)
    A = pd.DataFrame({"u": z + rng.normal(0, 0.3, n), "v": z + rng.normal(0, 0.3, n)})
    B = pd.DataFrame({"u": 100 + 5 * rng.normal(size=n), "v": rng.exponential(size=n)})
    ec = empirical_transfer(A, B, ["u", "v"], n, 1)
    rho_A = spearmanr(A["u"], A["v"]).statistic
    rho_ec = spearmanr(pd.to_numeric(ec["u"]), pd.to_numeric(ec["v"])).statistic
    assert abs(rho_ec - rho_A) < 0.1, f"rank copula not preserved: {rho_ec} vs {rho_A}"


def test_empirical_installs_target_marginals():
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    A = pd.DataFrame({"cat": ["a"] * 700 + ["b"] * 200 + ["c"] * 100,
                      "num": list(range(1000))})
    B = pd.DataFrame({"cat": ["a"] * 100 + ["b"] * 300 + ["c"] * 600,
                      "num": [x + 1000 for x in range(1000)]})
    ec = empirical_transfer(A, B, ["cat", "num"], 5000, 1)
    p = ec["cat"].value_counts(normalize=True)
    assert abs(p.get("c", 0) - 0.6) < 0.05          # target categorical proportions installed
    assert pd.to_numeric(ec["num"]).mean() > 1000   # target numeric range installed


def test_empirical_carries_missingness_rate():
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    A = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 250})
    B = pd.DataFrame({"x": [1.0] * 600 + [np.nan] * 400})   # 40% NaN target
    ec = empirical_transfer(A, B, ["x"], 5000, 1)
    assert 0.30 <= ec["x"].isna().mean() <= 0.50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_empirical.py -v`
Expected: FAIL with `ModuleNotFoundError` (module not created).

- [ ] **Step 3: Create the module**

```python
# src/ssdataagent/transfer/empirical_copula.py
"""Empirical-copula transfer: source A's full joint (shared-row resample) + target marginals.
Fixes the categorical dependence the one-factor glat mechanism loses. See
docs/superpowers/specs/2026-07-29-empirical-copula-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.generate import _is_numeric, _marginal_map


def empirical_transfer(source: pd.DataFrame, marg: pd.DataFrame, cols: list[str],
                       n: int, seed: int) -> pd.DataFrame:
    """A's empirical joint + ``marg``'s marginals. A single shared row-resample (``base``)
    supplies the copula: numeric columns keep the row's rank (as ``transfer_build``);
    categorical columns take the resampled row's ACTUAL category interval (joint-preserving)
    instead of the one-factor ``glat`` ordering. ``marg`` supplies the inverse-CDF value map
    and each column's missingness RATE. With ``marg is source`` this reproduces a direct
    row-resample of A's joint."""
    rng = np.random.default_rng(seed)
    m = len(source)
    num = {c: _is_numeric(source[c]) for c in cols}
    base = rng.integers(0, m, n)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        if num[c]:
            uf = (pd.to_numeric(source[c], errors="coerce")
                  .rank(pct=True, method="first").to_numpy(dtype=float))
            nan = np.isnan(uf)
            uf[nan] = rng.random(int(nan.sum()))
            u = uf[base]
        else:
            sv = source[c].to_numpy()[base]                          # resampled ACTUAL categories
            freq = source[c].dropna().astype(str).value_counts(normalize=True)  # frequency desc
            edges = np.concatenate([[0.0], np.cumsum(freq.to_numpy())])
            lo = {k: edges[i] for i, k in enumerate(freq.index)}
            wd = {k: edges[i + 1] - edges[i] for i, k in enumerate(freq.index)}
            key = pd.Series(sv).astype(str)                          # NaN -> "nan" (not a key)
            lo_a = key.map(lo).to_numpy(dtype=float)
            wd_a = key.map(wd).to_numpy(dtype=float)
            u = lo_a + rng.random(n) * wd_a
            bad = ~np.isfinite(u)                                    # NaN rows / unseen categories
            u[bad] = rng.random(int(bad.sum()))
        u = np.clip(u, 1e-6, 1 - 1e-6)
        em = _marginal_map(marg[c], u, num[c])
        miss = float(marg[c].isna().mean())
        if miss > 0:
            want = int(round(miss * n))
            mask = source[c].isna().to_numpy()[base].copy()
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_empirical.py -v`
Expected: 4 passed. (If `test_empirical_preserves_categorical_association` shows `v_ec` not clearly above `v_tb`, that is a real signal to report — do NOT loosen the threshold; the whole point is that EC preserves the association the one-factor engine loses.)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/empirical_copula.py tests/test_transfer_empirical.py
git commit -m "feat(empirical): empirical-copula transfer preserving the categorical joint"
```

---

### Task 2: Scored runner

**Files:**
- Create: `scripts/transfer_empirical.py`
- Test: `tests/test_transfer_empirical.py`

**Interfaces:**
- Consumes: `empirical_transfer`; `generate.transfer_build`; `pairs` (`PAIRS`, `covariates_outcomes`, `load_pair`); `scoring` (`restrict_config_dir`, `mean_scores`); `schema.load_schema`; `nodonor_bracket as nb`.
- Produces: `run_empirical(pair, *, seeds, n, bootstrap_B) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transfer_empirical.py

def test_ec_carry_reproduces_source_joint_better_than_engine():
    """On a small mixed frame, EC_carry (empirical_transfer(A, A)) keeps A's numeric+categorical
    association at least as well as transfer_build carryover -- the runner's EC_carry arm."""
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    from ssdataagent.transfer.generate import transfer_build
    from ssdataagent.transfer.copula_stability import pair_association
    rng = np.random.default_rng(1)
    n = 3000
    g = rng.choice(["lo", "hi"], n)
    A = pd.DataFrame({
        "grp": g,
        "cat": np.where(g == "hi", "A", "B").astype(object),
        "val": np.where(g == "hi", 1.0, 0.0) + rng.normal(0, 0.2, n),
    })
    ec = empirical_transfer(A, A, ["grp", "cat", "val"], n, 2)
    tb = transfer_build(A, A, ["grp", "cat", "val"], n, 2, "carryover")
    a_true = pair_association(A, "grp", "cat")[0]
    assert pair_association(ec, "grp", "cat")[0] >= pair_association(tb, "grp", "cat")[0] - 1e-6
    assert pair_association(ec, "grp", "cat")[0] > 0.8 * a_true
```

- [ ] **Step 2: Run test to verify it fails / passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_empirical.py -k reproduces -v`
Expected: PASSES on Task 1's code (it exercises `empirical_transfer` only) — this guards the runner's `EC_carry` invariant. Proceed once green.

- [ ] **Step 3: Create the runner**

Create `scripts/transfer_empirical.py`:

```python
#!/usr/bin/env python
"""Empirical-copula transfer -- a faithful structure model. Scores EC_carry (A's joint + A's
marginals, reads no B) and EC_oracle (A's joint + B's true marginals) against the current
engine and the ladder references. See docs/superpowers/specs/2026-07-29-empirical-copula-design.md.

    .venv/bin/python scripts/transfer_empirical.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.data.schema import load_schema  # noqa: E402
from ssdataagent.transfer.empirical_copula import empirical_transfer  # noqa: E402
from ssdataagent.transfer.generate import transfer_build  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes, load_pair  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def run_empirical(pair, *, seeds, n, bootstrap_B):
    """Score carryover(current engine), EC_carry, EC_oracle, ref_oracle_comp(==B1),
    ref_floor, ref_ceiling -- identical protocol to B0-B6. Only EC_oracle + ref_oracle_comp +
    the scorer touch B's pool; EC_carry reads no B."""
    import nodonor_bracket as nb

    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    schema = load_schema(ds)
    ref = nb._drop_unnamed(pd.read_csv(schema.real_data_path, low_memory=False))
    b_pool, guarantee = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    covariates_outcomes(pair.schema_name, cols)   # (kept for parity with the harness)
    types = nb.TYPES.get(ds, (1, 2, 3))

    configs = {
        "carryover":       lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "EC_carry":        lambda s: empirical_transfer(a, a, cols, n, s),
        "EC_oracle":       lambda s: empirical_transfer(a, b_pool, cols, n, s),
        "ref_oracle_comp": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
        "ref_floor":       lambda s: nb.build(b_pool, cols, n, s, "independence"),
        "ref_ceiling":     lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
    }

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(schema.ssdatabench_sim_subdir, set(cols), types, Path(cfg_td))
        for name, builder in configs.items():
            recs = [nb.score(builder(s), ds, ref, types, seed=1000 + s,
                             bootstrap_B=bootstrap_B, config_dir=cfg_dir)
                    for s in range(1, seeds + 1)]
            row = {"pair": pair.id, "config": name, "guarantee": guarantee}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"empirical_{pair.id}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_id")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    args = ap.parse_args()
    pair = next(p for p in PAIRS if p.id == args.pair_id)
    run_empirical(pair, seeds=args.seeds, n=args.n, bootstrap_B=args.bootstrap_B)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then a tiny real smoke**

Run: `.venv/bin/python -m pytest tests/test_transfer_empirical.py -v`
Expected: all pass (5 total).

Then a fast real-data smoke (1 seed, tiny bootstrap) to confirm the runner executes end-to-end on real data and EC_carry is finite:

Run: `.venv/bin/python scripts/transfer_empirical.py cps_1970_1980 --seeds 1 --n 1000 --bootstrap-B 30`
Expected: prints a 6-row table; every `overall` finite; `EC_carry` and `EC_oracle` rows present. If it raises, report BLOCKED with the traceback — do NOT wrap configs in try/except.

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_empirical.py tests/test_transfer_empirical.py
git commit -m "feat(empirical): scored runner (EC_carry / EC_oracle vs engine + ladder)"
```

**Note for the controller (post-implementation):** the full scored run (both pairs, 3 seeds,
n=3000, B=200) is heavy — run it as a controller step (moves to background past 600s), then write
the report + memory. All configs here are deterministic (no LLM/API).

---

## Self-Review

**Spec coverage:** `empirical_transfer` mechanism — categorical `u` from the resampled row's actual
interval, numeric unchanged (Task 1); the six-config ablation incl. `EC_carry` (feasible, no B) and
`EC_oracle` (Task 2); joint-preservation, rank-copula, marginal-install, and missingness are all
tested; scoring protocol identical to the face-swap harness. Generalizability/sibling-transfer is
explicitly deferred to Approach B (spec).

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `empirical_transfer(source, marg, cols, n, seed) -> DataFrame` at every call
site (both `EC_carry`/`EC_oracle`); `_marginal_map(marg[c], u, num)` and `_is_numeric` match
`generate`'s signatures; `nb.score(sim, ds, ref, types, seed=, bootstrap_B=, config_dir=)` matches
the face-swap call; `transfer_build(a, a|b_pool, cols, n, s, mode)` matches.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-empirical-copula.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
