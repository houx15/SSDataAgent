# B6 — ESS-gated Hybrid Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn B5's two hand-compared R² configs into one autonomous, provenance-tagged transfer generator (`B6_hybrid`) that selects retrieval-blend vs. pooled-prior R² from firewall-clean signals — the roadmap's Phase 4 integration.

**Architecture:** A thin selection layer over B5. `predict_target_r2(pair)` already computes both the `learned` (retrieval-blended) and `prior_only` R² maps, the ESS, and B4's `sib_rew` structure vehicle; B6 adds a pure gate `(n_siblings ≥ 2) AND (ess ≥ τ)` that picks one map per context, tags provenance per outcome, and scores one config through the unchanged `transfer_build_b2` vehicle. No new training; reproduces B5's per-pair winner by construction (cps 0.719 via `learned`, gss 0.728 via `prior_only`).

**Tech Stack:** Python, numpy, pandas. numpy-only for the model logic; existing `nodonor_bracket` scorer for end-to-end scoring; pytest.

## Global Constraints

- **Firewall (stricter than B2):** the target's covariate-R² is computed **nowhere**. B6 reads only the target's public marginals, X-margins (raking), and public outcome features. The two gate inputs are firewall-clean: `n_siblings` is a corpus count of held-out sibling contexts, `ess` is a raking diagnostic over public target margins. **The gate reads no score.**
- **B5 byte-identical:** `predict_target_r2`'s new 5th return value (`n_siblings`) must be unused by B5's scoring path; `run_b5`'s call site unpacks-and-ignores it. B5 output CSVs are unchanged.
- **LLM-free, numpy-only** for all B6 model logic (gate + map selection).
- **τ = 0.3** is a documented default in the unidentified (0.10, 0.65) gap, **not load-bearing** — the `n_siblings ≥ 2` criterion separates the scored pairs, and any τ in the gap gives identical selection.
- **`results/` is gitignored** — scored CSVs are not committed.
- **Do not stage the untracked `ssdatabench` submodule** in any commit.
- **Scoring protocol (unchanged from B0–B5):** 3 seeds, `n=3000`, `bootstrap_B=200`, both benchmark pairs, same scorer and noise floor (~0.054). Heavy scoring is reaped on the box; the controller runs it via the resumable per-(config,seed) scorer, not the implementer subagents.

---

### Task 1: The gate and map-selection functions

**Files:**
- Modify: `src/ssdataagent/transfer/rescue.py` (append two functions after `predict_r2`)
- Test: `tests/test_transfer_rescue.py` (append)

**Interfaces:**
- Consumes: nothing new (pure functions over primitives + the `learned` / `prior_only` dicts that `predict_r2` already produces).
- Produces:
  - `select_r2_source(n_siblings: int, ess: float, *, tau: float = 0.3, min_siblings: int = 2) -> bool`
  - `hybrid_r2_map(learned: dict, prior_only: dict, use_retrieval: bool) -> tuple[dict, dict]` returning `(r2_map, provenance)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transfer_rescue.py`:

```python
def test_select_r2_source_truth_table():
    from ssdataagent.transfer.rescue import select_r2_source
    # cps: plural pool, well-sized -> trust retrieval
    assert select_r2_source(3, 0.65) is True
    # gss: lone thin sibling -> fall back to prior (fails BOTH criteria)
    assert select_r2_source(1, 0.10) is False
    # plural pool but poorly raked -> prior (ESS fails)
    assert select_r2_source(3, 0.10) is False
    # well-raked but only one sibling -> prior (count fails)
    assert select_r2_source(1, 0.65) is False


def test_select_r2_source_boundaries_and_tau():
    from ssdataagent.transfer.rescue import select_r2_source
    assert select_r2_source(2, 0.30) is True          # both at threshold -> eligible
    assert select_r2_source(2, 0.2999) is False        # just below tau
    assert select_r2_source(1, 0.99) is False          # count dominates
    # tau is non-load-bearing: any value across the (0.10, 0.65) gap selects identically
    for tau in (0.15, 0.30, 0.50, 0.60):
        assert select_r2_source(3, 0.65, tau=tau) is True
        assert select_r2_source(1, 0.10, tau=tau) is False


def test_hybrid_r2_map_retrieval_branch_is_truthful():
    from ssdataagent.transfer.rescue import hybrid_r2_map
    learned = {"a": 0.80, "b": 0.20, "c": 0.35}
    prior_only = {"a": 0.30, "b": 0.20, "c": 0.35}   # b, c unmoved by retrieval
    r2_map, prov = hybrid_r2_map(learned, prior_only, use_retrieval=True)
    assert r2_map == learned
    assert prov == {"a": "retrieval-blend", "b": "prior", "c": "prior"}


def test_hybrid_r2_map_prior_branch_all_prior():
    from ssdataagent.transfer.rescue import hybrid_r2_map
    learned = {"a": 0.80, "b": 0.20}
    prior_only = {"a": 0.30, "b": 0.20}
    r2_map, prov = hybrid_r2_map(learned, prior_only, use_retrieval=False)
    assert r2_map == prior_only
    assert prov == {"a": "prior", "b": "prior"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_rescue.py -k "select_r2_source or hybrid_r2_map" -v`
Expected: FAIL with `ImportError` / `AttributeError` (functions not defined).

- [ ] **Step 3: Write the implementation**

Append to `src/ssdataagent/transfer/rescue.py`:

```python
def select_r2_source(n_siblings: int, ess: float, *,
                     tau: float = 0.3, min_siblings: int = 2) -> bool:
    """ESS-gated hybrid gate (B6). Trust retrieval-blend only when the sibling pool
    is both plural (>= min_siblings independent same-instrument contexts) AND
    effectively-sized (ess >= tau) -- i.e. not a lone thin sibling. Returns True to
    use the retrieval-blended R^2 (B5 ``learned``), False to fall back to the pooled
    prior (B5 ``prior_only``). tau is deliberately non-load-bearing: the
    sibling-count criterion separates the scored pairs, and any tau in the
    unidentified (0.10, 0.65) gap gives identical selection. Reads no score."""
    return (n_siblings >= min_siblings) and (ess >= tau)


def hybrid_r2_map(learned: dict, prior_only: dict,
                  use_retrieval: bool) -> tuple[dict, dict]:
    """Select the per-outcome R^2 map and a truthful per-outcome provenance tag.
    use_retrieval True  -> r2_map = learned; provenance[o] = 'retrieval-blend' iff
                           retrieval actually moved it (learned[o] != prior_only[o])
                           else 'prior'.
    use_retrieval False -> r2_map = prior_only; every provenance[o] = 'prior'."""
    if use_retrieval:
        r2_map = dict(learned)
        prov = {o: ("retrieval-blend" if learned.get(o) != prior_only.get(o)
                    else "prior") for o in r2_map}
    else:
        r2_map = dict(prior_only)
        prov = {o: "prior" for o in r2_map}
    return r2_map, prov
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_rescue.py -v`
Expected: PASS (new tests + all existing rescue tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/rescue.py tests/test_transfer_rescue.py
git commit -m "feat(transfer): B6 ESS-gated hybrid gate + map selection"
```

---

### Task 2: Expose `n_siblings` from `predict_target_r2` (B5 byte-identical)

**Files:**
- Modify: `scripts/transfer_b5.py` (`predict_target_r2` return; `run_b5` call site)
- Test: `tests/test_transfer_b5.py` (`test_predict_target_r2_shapes` — update unpack + assert)

**Interfaces:**
- Consumes: `reweighted_pool_for(pair, cols, target_pool, rng)` which returns `(sib_rew, ess, used_waves, dropped)` — `used_waves` is the list of surviving sibling wave frames.
- Produces: `predict_target_r2(pair) -> tuple[dict, dict, float, pd.DataFrame, int]` — appends `n_siblings = len(used_waves)` as the 5th element. cps → 3, gss → 1.

- [ ] **Step 1: Update the shape test to expect the 5-tuple**

In `tests/test_transfer_b5.py`, replace the body of `test_predict_target_r2_shapes` unpack line and add the assertion:

```python
def test_predict_target_r2_shapes(tmp_path):
    import transfer_b5
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    learned, prior_only, ess, sib_rew, n_siblings = transfer_b5.predict_target_r2(pair)
    assert set(learned) == set(prior_only)          # same outcome keys
    assert len(sib_rew) > 0                          # structure vehicle materialized
    assert learned                                   # non-empty
    assert 0.0 < ess <= 1.0
    assert isinstance(n_siblings, int) and n_siblings >= 1   # cps holds out 1980 -> 3 sibs
    for d in (learned, prior_only):
        for v in d.values():
            assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: Change `predict_target_r2` to capture and return `n_siblings`**

In `scripts/transfer_b5.py`, in `predict_target_r2`, change the `reweighted_pool_for` unpack and the return. Replace:

```python
    sib_rew, ess, _, _ = reweighted_pool_for(pair, cols, target_pool,
                                             np.random.default_rng(0))
```
with:
```python
    sib_rew, ess, used_waves, _ = reweighted_pool_for(pair, cols, target_pool,
                                                      np.random.default_rng(0))
```

Replace the final line:
```python
    return learned, prior_only, ess, sib_rew
```
with:
```python
    return learned, prior_only, ess, sib_rew, len(used_waves)
```

Update the docstring's last sentence from
`Returns (learned_r2, prior_only_r2, ess, sib_rew).` to
`Returns (learned_r2, prior_only_r2, ess, sib_rew, n_siblings).`

- [ ] **Step 3: Update `run_b5`'s call site to ignore the new value (byte-identical B5)**

In `scripts/transfer_b5.py`, in `run_b5`, replace:
```python
    learned, prior_only, ess, sib_rew = predict_target_r2(pair)
```
with:
```python
    learned, prior_only, ess, sib_rew, _ = predict_target_r2(pair)
```

- [ ] **Step 4: Verify module import health (fast tests)**

Run: `.venv/bin/python -m pytest tests/test_transfer_b5.py -k "corpus or context_records or training_rows or pseudo_targets or noise_points" -v`
Expected: PASS (these fast tests import `transfer_b5`; a syntax/arity break surfaces here). The heavy `test_predict_target_r2_shapes` and `test_run_b5_smoke_scores_both_configs` are dominated by the ~300s LOCO fit and are verified by the controller in a background run (they get reaped if run inline) — the implementer does NOT run them.

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_b5.py tests/test_transfer_b5.py
git commit -m "feat(transfer): predict_target_r2 exposes n_siblings (B5 byte-identical)"
```

---

### Task 3: The `transfer_b6.py` orchestrator

**Files:**
- Create: `scripts/transfer_b6.py`
- Test: `tests/test_transfer_b6.py`

**Interfaces:**
- Consumes:
  - `predict_target_r2(pair) -> (learned, prior_only, ess, sib_rew, n_siblings)` (Task 2)
  - `select_r2_source`, `hybrid_r2_map` (Task 1)
  - `transfer_build_b2(source_pool, target_pool, cols, covs, outs, n, seed, *, r2_target=...)` (existing)
  - `b4_columns(pair) -> (ds, cols, covs, outs)`; `nodonor_bracket.{carve_pool, score, TYPES, _drop_unnamed}`; `restrict_config_dir`, `mean_scores`; `load_schema`.
- Produces: `run_b6(pair, *, seeds, n, bootstrap_B, tau=0.3) -> pd.DataFrame` with one `B6_hybrid` row carrying columns `pair, config, guarantee, ess_ratio, n_siblings, source, T1, T2, T3, overall`; writes `results/transfer_map/b6_<pair.id>.csv`. Also a helper `select_for(pair, *, tau=0.3) -> (r2_map, provenance, meta, sib_rew, target_pool, cols, covs, outs, ds, guarantee, ref, types)` is NOT required — keep the selection inline in `run_b6` but factor the pure decision through the Task 1 functions.

- [ ] **Step 1: Write the failing wiring test**

Create `tests/test_transfer_b6.py`. The test stubs `predict_target_r2` so it does NOT run the ~300s fit; everything else in `run_b6` (carve_pool, tiny-n scoring) is fast. The stub returns `sib_rew = target_pool` (a valid frame with the right columns) so `transfer_build_b2` works.

```python
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def _stub_predict(pair, *, n_siblings, ess):
    """Build a deterministic (learned, prior_only, ess, sib_rew, n_siblings) stub
    from the pair's real columns, with sib_rew = the carved target pool so the
    downstream draw/score run without the heavy LOCO fit."""
    import transfer_b6
    import nodonor_bracket as nb
    ds, cols, covs, outs = transfer_b6.b4_columns(pair)
    target_pool, _ = nb.carve_pool(ds)
    learned = {o: 0.60 for o in outs}          # retrieval-moved
    prior_only = {o: 0.25 for o in outs}       # distinct -> provenance is 'retrieval-blend'
    return learned, prior_only, float(ess), target_pool[cols].copy(), int(n_siblings)


def test_run_b6_gate_pass_uses_retrieval(tmp_path, monkeypatch):
    import transfer_b6
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    monkeypatch.setattr(transfer_b6, "OUT", tmp_path)
    monkeypatch.setattr(transfer_b6, "predict_target_r2",
                        lambda p: _stub_predict(p, n_siblings=3, ess=0.65))
    df = transfer_b6.run_b6(pair, seeds=1, n=200, bootstrap_B=20)
    assert list(df["config"]) == ["B6_hybrid"]
    assert df.iloc[0]["source"] == "retrieval"
    assert int(df.iloc[0]["n_siblings"]) == 3
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all()
    assert (tmp_path / "b6_cps_1970_1980.csv").exists()


def test_run_b6_gate_fail_uses_prior(tmp_path, monkeypatch):
    import transfer_b6
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "gss_1994_2018"][0]
    monkeypatch.setattr(transfer_b6, "OUT", tmp_path)
    monkeypatch.setattr(transfer_b6, "predict_target_r2",
                        lambda p: _stub_predict(p, n_siblings=1, ess=0.10))
    df = transfer_b6.run_b6(pair, seeds=1, n=200, bootstrap_B=20)
    assert df.iloc[0]["source"] == "prior"
    assert int(df.iloc[0]["n_siblings"]) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_b6.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'transfer_b6'`.

- [ ] **Step 3: Write `scripts/transfer_b6.py`**

```python
#!/usr/bin/env python
"""B6 -- ESS-gated hybrid generator (Phase 4 integration). One autonomous config
that selects B5's retrieval-blended R^2 vs. the pooled-prior R^2 from firewall-clean
signals (n_siblings, ESS), provenance-tagged. LLM-free.

    .venv/bin/python scripts/transfer_b6.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
    .venv/bin/python scripts/transfer_b6.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
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
from ssdataagent.transfer.generate import transfer_build_b2  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS  # noqa: E402
from ssdataagent.transfer.rescue import hybrid_r2_map, select_r2_source  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402
from transfer_b4 import b4_columns  # noqa: E402
from transfer_b5 import _load, predict_target_r2  # noqa: E402

OUT = REPO / "results" / "transfer_map"

TAU = 0.3


def run_b6(pair, *, seeds, n, bootstrap_B, tau: float = TAU):
    """Score the single B6_hybrid config: gate (n_siblings, ess) picks B5's learned
    or prior_only R^2 map, then score through the IDENTICAL B4/B5 sib_rew vehicle via
    the r2_target seam. Writes CSV with source + n_siblings provenance columns."""
    import nodonor_bracket as nb
    ds, cols, covs, outs = b4_columns(pair)
    target_pool, guarantee = nb.carve_pool(ds)
    ref = _load(load_schema(ds).real_data_path)
    types = nb.TYPES.get(ds, (1, 2, 3))

    learned, prior_only, ess, sib_rew, n_siblings = predict_target_r2(pair)
    use_retrieval = select_r2_source(n_siblings, ess, tau=tau)
    r2_map, provenance = hybrid_r2_map(learned, prior_only, use_retrieval)
    source = "retrieval" if use_retrieval else "prior"
    print(f"{pair.id}: n_siblings {n_siblings}, ess {ess:.3f}, tau {tau} -> "
          f"source={source}")
    print(f"{pair.id}: provenance {provenance}")

    recs = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(load_schema(ds).ssdatabench_sim_subdir,
                                      set(cols), types, Path(cfg_td))
        for s in range(1, seeds + 1):
            sim = transfer_build_b2(sib_rew, target_pool, cols, covs, outs,
                                    n, s, r2_target=r2_map)
            recs.append(nb.score(sim, ds, ref, types, seed=1000 + s,
                                 bootstrap_B=bootstrap_B, config_dir=cfg_dir))

    row = {"pair": pair.id, "config": "B6_hybrid", "guarantee": guarantee,
           "ess_ratio": round(ess, 4), "n_siblings": n_siblings, "source": source}
    row.update(mean_scores(pd.DataFrame(recs)))
    df = pd.DataFrame([row])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"b6_{pair.id}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", choices=[p.id for p in PAIRS if p.scored])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--tau", type=float, default=TAU)
    a = ap.parse_args()
    pair = [p for p in PAIRS if p.id == a.pair][0]
    df = run_b6(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B, tau=a.tau)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'b6_{pair.id}.csv'}")
    print("REGIME: no-donor + hybrid. Target supplies marginals + X-margins + public"
          " outcome features only; conditional strength is auto-selected between the"
          " retrieval blend and the cross-context prior by an ESS gate reading no score.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_b6.py -v`
Expected: PASS (both branches). Runtime is seconds — the stub skips the LOCO fit.

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_b6.py tests/test_transfer_b6.py
git commit -m "feat(transfer): B6 orchestrator -- ESS-gated hybrid, provenance-tagged"
```

---

## Post-implementation (controller-run, not a subagent task)

These are done by the controller after Task 3 is reviewed clean, because they depend on heavy scoring that gets reaped on the box:

1. **Score both pairs end-to-end** via the resumable per-(config,seed) scorer (mirror `.superpowers/sdd/b5_incremental.py`, adapted to the single `B6_hybrid` config), protocol 3 seeds / n=3000 / B=200. Expected: cps overall ≈ 0.719 (`source=retrieval`), gss overall ≈ 0.728 (`source=prior`). Sanity-check they match the corresponding B5 config numbers (B6 must equal B5_learned on cps and B5_prior_only on gss by construction).
2. **Write the report** `docs/report/2026-07-24-b6-ess-gated-hybrid.md` — claim is *completed integration* (self-selecting, provenance-tagged), not a new score; τ-sensitivity as the robustness argument; the 2-pair validation caveat; both limitations from the spec.
3. **Add the LEDGER row** `b6_ess_gated_hybrid` (newest on top) with a meaningful one-line `hypothesis`.
4. **Rebuild the dashboard** (`.venv/bin/python scripts/build_dashboard.py`) and commit `docs/dashboard/index.html`.
5. **Update memory:** new `project_b6_ess_gated_hybrid.md`, MEMORY.md index line, and supersede the "Phase-4 aim" note in `project_b5_learned_r2_rescue.md`.

---

## Self-Review

**Spec coverage:** gate rule (Task 1 `select_r2_source`) ✓; per-outcome truthful provenance (Task 1 `hybrid_r2_map`) ✓; `n_siblings` exposure with B5 byte-identical (Task 2) ✓; single `B6_hybrid` config through the unchanged vehicle + CSV provenance columns (Task 3) ✓; firewall unchanged (no target-R² read anywhere in B6 code) ✓; scoring protocol + report/LEDGER/dashboard/memory (post-implementation) ✓; τ non-load-bearing tested (Task 1 boundary test) ✓.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test has assertions.

**Type consistency:** `predict_target_r2` returns a 5-tuple everywhere it is unpacked (Task 2 in `run_b5` + test; Task 3 in `run_b6` + stub). `select_r2_source` returns `bool`; `hybrid_r2_map` returns `(dict, dict)`. CSV columns (`source`, `n_siblings`) match between `run_b6` and both b6 tests. `_load` and `predict_target_r2` are imported from `transfer_b5` in `transfer_b6` — both exist (verified).
