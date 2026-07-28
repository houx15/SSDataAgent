# Public X-margins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "public X-margins" regime — admit B's census-standard demographic margins (age/gender/race) into the marginal frame while keeping the copula and Y-marginals from A / the LLM description — and score it against the ladder to decompose B1's marginal advantage into an X-part and a Y-part.

**Architecture:** A new pure helper `transfer/publicx.py::with_public_x` swaps only the demographic columns of a marginal frame to B's public distribution. A runner `scripts/transfer_publicx.py` clones the face-swap harness and builds seven configs (2 new: `PX_carry`, `PX_llm`) on top of `transfer_build` + the existing blind components.

**Tech Stack:** Python 3, pandas, numpy; reuses `transfer.generate.transfer_build`, `transfer.blind.build_marg_frame` / `elicit_marginals`, and the `nodonor_bracket` scorer via `scripts/transfer_faceswap.py`'s pattern.

## Global Constraints

- **Firewall per arm:** `PX_carry`/`PX_llm` read only B's **public demographic X-margins** (age/gender/race distributions from `b_pool`); the copula and Y never read B's Y. `PX_llm`'s Y reads only B's **description** (LLM). `ref_oracle_comp` (== B1) reads B's full marginals — a labeled ceiling.
- **`PUBLIC_X = ("age", "gender", "race")`** — the census-standard demographics; `x_cols = [c for c in PUBLIC_X if c in covs]`. Do NOT include other background variables (parent education/occupation are not public census margins).
- **Reuse, don't reinvent:** `transfer_build` (copula+marginals); `build_marg_frame` / `elicit_marginals` from `transfer.blind` (unchanged); the harness setup + scorer from `scripts/transfer_faceswap.py` (`a`/`schema`/`ref`/`b_pool`/`cols`/`covs,outs`/`types`/`restrict_config_dir`/`mean_scores`, `nodonor_bracket as nb`).
- **`OPENROUTER_API_KEY` in `.env`** — never print; the Y elicitation is cached under `results/blind_cache/` (cps/gss caches are already warm, so scoring needs no key).
- **No `/dev/null` / out-of-repo redirects** (hook blocks). **Avoid the literal word "eval"** in commit messages. **`results/` is gitignored.** **Never stage the `ssdatabench` submodule** — stage only the files each task names.

---

### Task 1: `with_public_x` helper

**Files:**
- Create: `src/ssdataagent/transfer/publicx.py`
- Test: `tests/test_transfer_publicx.py`

**Interfaces:**
- Produces:
  - `PUBLIC_X = ("age", "gender", "race")`
  - `with_public_x(base_marg: pd.DataFrame, b_pool: pd.DataFrame, x_cols, *, seed: int = 0) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer_publicx.py
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_with_public_x_swaps_only_x_distribution():
    from ssdataagent.transfer.publicx import with_public_x
    base = pd.DataFrame({"age": [20, 20, 20, 20], "inc": [1, 2, 3, 4]})
    b = pd.DataFrame({"age": [60, 60, 60, 60, 60], "inc": [9, 9, 9, 9, 9]})
    out = with_public_x(base, b, ["age"], seed=0)
    assert set(out["age"]) == {60}                 # age now drawn from b
    assert list(out["inc"]) == [1, 2, 3, 4]        # non-x column untouched
    assert len(out) == len(base)


def test_with_public_x_empty_returns_base_unchanged():
    from ssdataagent.transfer.publicx import with_public_x
    base = pd.DataFrame({"age": [1, 2], "inc": [3, 4]})
    out = with_public_x(base, pd.DataFrame({"age": [9]}), [], seed=0)
    pd.testing.assert_frame_equal(out, base)


def test_with_public_x_preserves_missingness():
    from ssdataagent.transfer.publicx import with_public_x
    base = pd.DataFrame({"race": ["W"] * 12})
    b = pd.DataFrame({"race": ["B", "B", "B", "B", "B", "B", None, None, None, None]})  # 40% NaN
    out = with_public_x(base, b, ["race"], seed=1)
    assert 0.15 <= out["race"].isna().mean() <= 0.65     # ~40% NaN carried (resample tolerance)
    assert set(out["race"].dropna()) == {"B"}


def test_with_public_x_absent_column_unchanged():
    from ssdataagent.transfer.publicx import with_public_x
    base = pd.DataFrame({"age": [1, 2], "inc": [3, 4]})
    out = with_public_x(base, pd.DataFrame({"inc": [9, 9]}), ["age"], seed=0)
    assert list(out["age"]) == [1, 2]              # age absent from b -> left unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_publicx.py -v`
Expected: FAIL with `ModuleNotFoundError` (module not created).

- [ ] **Step 3: Create the module**

```python
# src/ssdataagent/transfer/publicx.py
"""Public X-margins: admit B's census-standard demographic margins (age/gender/race) while
keeping the copula and Y-marginals from A / the description. See
docs/superpowers/specs/2026-07-28-public-x-margins-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PUBLIC_X = ("age", "gender", "race")


def with_public_x(base_marg: pd.DataFrame, b_pool: pd.DataFrame, x_cols, *,
                  seed: int = 0) -> pd.DataFrame:
    """Copy of ``base_marg`` with each column in ``x_cols`` that is present in ``b_pool``
    replaced by a length-preserving resample of ``b_pool``'s column (carrying B's marginal,
    including its missingness rate). Every other column is byte-identical to ``base_marg``;
    a column absent from ``b_pool`` (or an empty ``b_pool`` column) is left unchanged."""
    rng = np.random.default_rng(seed)
    out = base_marg.copy()
    n = len(out)
    for c in x_cols:
        if c not in b_pool.columns:
            continue
        src = b_pool[c].to_numpy()
        if len(src) == 0:
            continue
        out[c] = src[rng.integers(0, len(src), n)]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_publicx.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/publicx.py tests/test_transfer_publicx.py
git commit -m "feat(publicx): with_public_x -- swap demographic margins to B's public distribution"
```

---

### Task 2: Runner + real-data dry-run

**Files:**
- Create: `scripts/transfer_publicx.py`
- Test: `tests/test_transfer_publicx.py`

**Interfaces:**
- Consumes: `with_public_x`, `PUBLIC_X`; `transfer.generate.transfer_build`; `transfer.blind.build_marg_frame` / `elicit_marginals`; `transfer.pairs` (`PAIRS`, `covariates_outcomes`, `load_pair`); `transfer.scoring` (`restrict_config_dir`, `mean_scores`); `schema.load_schema`; `nodonor_bracket as nb`.
- Produces: `run_publicx(pair, *, seeds, n, bootstrap_B, dry_run=False)` and a `--dry-run` CLI flag.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transfer_publicx.py

def test_px_frames_differ_from_base_only_in_public_x():
    """PX_carry's marginal frame == A except in PUBLIC_X; PX_llm's == the LLM frame except in
    PUBLIC_X. This is the invariant the runner relies on to isolate the X-margin fix."""
    from ssdataagent.transfer.publicx import with_public_x, PUBLIC_X
    a = pd.DataFrame({"age": [30] * 6, "gender": ["M"] * 6, "race": ["W"] * 6,
                      "income": [1, 2, 3, 4, 5, 6], "education": list("aabbcc")})
    b_pool = pd.DataFrame({"age": [70] * 6, "gender": ["F"] * 6, "race": ["B"] * 6,
                           "income": [9] * 6, "education": list("cccccc")})
    x_cols = [c for c in PUBLIC_X if c in a.columns]
    px_carry = with_public_x(a, b_pool, x_cols, seed=1)
    changed = [c for c in a.columns if not a[c].equals(px_carry[c])]
    assert set(changed) == set(x_cols)                       # only demographics changed
    assert list(px_carry["income"]) == [1, 2, 3, 4, 5, 6]    # Y untouched (== A carry-over)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_publicx.py -k px_frames -v`
Expected: FAIL only if `with_public_x` is broken — it should PASS already (Task 1 code). If it passes, that is the intended green for this invariant; proceed. (This test guards the runner's core invariant without importing the heavy runner.)

- [ ] **Step 3: Create the runner**

Create `scripts/transfer_publicx.py`:

```python
#!/usr/bin/env python
"""Public X-margins -- admit B's census-standard demographics (age/gender/race) into the
marginal frame; keep the copula from A and Y-marginals from A / the LLM description. Scores
PX_carry / PX_llm alongside FS_* and the ladder references. See
docs/superpowers/specs/2026-07-28-public-x-margins-design.md.

    export OPENROUTER_API_KEY=...     # first run only; Y elicitation is cached (blind cache)
    .venv/bin/python scripts/transfer_publicx.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
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
from ssdataagent.transfer.blind import build_marg_frame, elicit_marginals  # noqa: E402
from ssdataagent.transfer.generate import transfer_build  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes, load_pair  # noqa: E402
from ssdataagent.transfer.publicx import PUBLIC_X, with_public_x  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def _marginal_tv(a_col, b_col):
    pa = a_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    pb = b_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return 0.5 * float((pa.reindex(idx, fill_value=0.0) - pb.reindex(idx, fill_value=0.0)).abs().sum())


def run_publicx(pair, *, seeds, n, bootstrap_B, dry_run=False):
    """Score FS_carryover(==B0), PX_carry, PX_llm, FS_llm(blind), ref_oracle_comp(==B1),
    ref_floor, ref_ceiling. PX_* admit B's public age/gender/race margins only."""
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
    x_cols = [c for c in PUBLIC_X if c in covs]

    if dry_run:
        # Show the composition gap being fixed and confirm the swap, no LLM / no scorer.
        print(f"{pair.id}: PUBLIC_X in play = {x_cols}", flush=True)
        for c in x_cols:
            gap = _marginal_tv(a[c], b_pool[c])
            swapped = with_public_x(a, b_pool, [c], seed=0)
            gap_after = _marginal_tv(swapped[c], b_pool[c])
            print(f"  {c}: TV(A,B)={gap:.3f}  ->  TV(swapped,B)={gap_after:.3f}", flush=True)
        return None

    elicited = elicit_marginals(ds, a, cols)          # cached; reads source A + B's description
    llm_marg = build_marg_frame(elicited, a, cols)
    print(f"{pair.id}: x_cols={x_cols}; elicited {sum(1 for c in cols if c in elicited)}/{len(cols)}",
          flush=True)

    configs = {
        "FS_carryover":    lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "PX_carry":        lambda s: transfer_build(a, with_public_x(a, b_pool, x_cols, seed=s),
                                                    cols, n, s, "marginal-swap"),
        "PX_llm":          lambda s: transfer_build(a, with_public_x(llm_marg, b_pool, x_cols, seed=s),
                                                    cols, n, s, "marginal-swap"),
        "FS_llm":          lambda s: transfer_build(a, llm_marg, cols, n, s, "marginal-swap"),
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
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "x_cols": "|".join(x_cols)}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"publicx_{pair.id}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_id")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the demographic composition gap being fixed (no LLM, no scoring)")
    args = ap.parse_args()
    pair = next(p for p in PAIRS if p.id == args.pair_id)
    run_publicx(pair, seeds=args.seeds, n=args.n, bootstrap_B=args.bootstrap_B, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the real-data dry-run**

Run: `.venv/bin/python -m pytest tests/test_transfer_publicx.py -v`
Expected: all pass (5 total).

Then verify real-data wiring (no API key, no scorer) for both pairs:

Run: `.venv/bin/python scripts/transfer_publicx.py cps_1970_1980 --dry-run`
Run: `.venv/bin/python scripts/transfer_publicx.py gss_1994_2018 --dry-run`
Expected: each prints `PUBLIC_X in play = ['age', 'gender', 'race']` and, per column, a non-trivial `TV(A,B)` that drops to ~0 after the swap (`TV(swapped,B)` near 0). If either raises, report BLOCKED with the traceback — do NOT narrow the pair or wrap in try/except.

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_publicx.py tests/test_transfer_publicx.py
git commit -m "feat(publicx): scored runner (PX_carry / PX_llm) + real-data dry-run"
```

**Note for the controller (post-implementation, not the implementer):** the full scored run
across both pairs (all seven configs) reuses the warm blind elicitation cache, so it needs no API
key, but it is heavy — run it as a controller step (foreground moves to background past 600s), then
write the report + memory. `PX_carry`/`ref_*` are fully deterministic; `PX_llm`/`FS_llm` read the
cached elicitation.

---

## Self-Review

**Spec coverage:** `with_public_x` mechanism (Task 1); the seven configs incl. `PX_carry`/`PX_llm`
(Task 2 runner); `PUBLIC_X = age/gender/race` and `x_cols = PUBLIC_X ∩ covs` (Global Constraints +
runner); the decomposition arms (`PX_carry` vs `FS_carryover`, `PX_llm` vs `FS_llm`, `B1` vs
`PX_carry`) are all present; firewall per arm enforced by what each config reads; scoring protocol
identical to the face-swap (`nb.score`, `restrict_config_dir`, `mean_scores`, 3 seeds/n=3000/B=200).

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `with_public_x(DataFrame, DataFrame, list, *, seed) -> DataFrame` at every
call site; `build_marg_frame(elicited, a, cols)` and `elicit_marginals(ds, a, cols)` match their
signatures; `transfer_build(struct, marg, cols, n, seed, mode)` matches; `nb.score(sim, ds, ref,
types, seed=, bootstrap_B=, config_dir=)` matches the face-swap call; `x_cols` is a `list[str]`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-28-public-x-margins.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
