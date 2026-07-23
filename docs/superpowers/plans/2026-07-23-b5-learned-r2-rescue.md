# B5 — Learned R² rescue (Phase 3, slice 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict each target context's per-outcome T3 covariate-R² as an
empirical-Bayes blend of B4's same-instrument retrieval estimate and a
cross-context pooled prior, gated by retrieval reliability (ESS), so gss's
thin-retrieval T3 is rescued without disturbing cps.

**Architecture:** A closed-form, numpy-only empirical-Bayes model
(`rescue.py`): a GLS prior μ(features) over true-R² rows pooled across all
schema-backed contexts, plus a σ²(ESS) retrieval-noise curve fit on cps
pseudo-targets, combined by precision weighting. An orchestrator
(`transfer_b5.py`) builds the LOCO training corpus, fits the model, predicts the
held-out target's R² dict, and feeds it through the existing B2 generator via a
new one-line `r2_target` override — scored identically to B0–B4.

**Tech Stack:** Python, numpy, pandas. No new dependency, no LLM, no MCMC.

## Global Constraints

Every task's requirements implicitly include these (copied from the spec
`docs/superpowers/specs/2026-07-23-b5-learned-r2-rescue-design.md`):

- **Comparability (binding).** B5 is scored on the SAME crosswalk `cols` as
  B0–B4 (derived via `transfer_b4.b4_columns`), the SAME `restrict_config_dir`,
  the SAME reference (`load_schema(target_dataset).real_data_path`), the SAME
  seed offset (`1000 + s`), `seeds = 3`, `n = 3000`, `bootstrap_B = 200`.
- **B2/B4 unchanged bit-for-bit.** The only edit to existing generation code is
  an *optional* keyword-only `r2_target` param on `transfer_build_b2` that
  defaults to `None`; with `None` the function is byte-identical to today.
- **Firewall.** B5 reads only the target's public univariate marginals (X and
  Y), its X-margins for raking, and public structural features of each outcome.
  It NEVER reads the target context's per-person joint, its covariate-R², its
  pairwise associations, or the benchmark reference sample. Training contexts
  contribute microdata under leave-one-context-out (only the target wave is
  held out).
- **LLM-free.** B5 runs deterministically off the microdata; no API key.
- **Minimal prior features.** The prior μ uses exactly three structural features
  — normalized marginal entropy, predictor count, numeric indicator — plus an
  intercept. Do not add more (overfitting guard at ~13 contexts).
- **numpy-only closed form.** No PyMC/statsmodels/scikit dependency; the EB fit
  is `np.linalg.lstsq` and arithmetic.
- **results/ is gitignored.** Result CSVs under `results/transfer_map/` are
  written but never committed.

---

### Task 1: Outcome structural features

**Files:**
- Create: `src/ssdataagent/transfer/rescue.py`
- Test: `tests/test_transfer_rescue.py`

**Interfaces:**
- Consumes: `ssdataagent.transfer.generate._is_numeric`.
- Produces: `outcome_features(pool, outcome, predictors, *, numeric_predictors=frozenset()) -> dict[str, float]` returning keys `{"entropy", "n_predictors", "is_numeric"}` (all floats). Firewall-clean: reads only `pool[outcome]`'s univariate marginal and the predictor count.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_rescue.py
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd
from ssdataagent.transfer.rescue import outcome_features


def test_outcome_features_shape_and_ranges():
    pool = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 70, 80, 90],       # numeric predictor
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],  # predictor
        "balanced": ["a", "b", "a", "b", "a", "b", "a", "b"],  # categorical outcome, max entropy
        "constant": ["z"] * 8,                          # zero entropy
        "income": [1.0, 2, 3, 4, 5, 6, 7, 8],           # numeric outcome
    })
    preds = ["age", "sex"]
    f_bal = outcome_features(pool, "balanced", preds, numeric_predictors=frozenset({"age"}))
    assert set(f_bal) == {"entropy", "n_predictors", "is_numeric"}
    assert f_bal["n_predictors"] == 2.0
    assert f_bal["is_numeric"] == 0.0
    assert f_bal["entropy"] == 1.0                      # perfectly balanced binary
    f_const = outcome_features(pool, "constant", preds)
    assert f_const["entropy"] == 0.0                    # single value -> no diversity
    f_inc = outcome_features(pool, "income", preds, numeric_predictors=frozenset({"age"}))
    assert f_inc["is_numeric"] == 1.0
    assert 0.0 <= f_inc["entropy"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_rescue.py::test_outcome_features_shape_and_ranges -v`
Expected: FAIL with `ImportError` / `cannot import name 'outcome_features'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/transfer/rescue.py
"""B5 -- learned R^2 rescue (Phase 3, slice 2). A closed-form, numpy-only
empirical-Bayes model that predicts a target context's per-outcome covariate-R^2
by shrinking B4's same-instrument retrieval estimate toward a cross-context
pooled prior, weighted by retrieval reliability (ESS). LLM-free.

See docs/superpowers/specs/2026-07-23-b5-learned-r2-rescue-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.generate import _is_numeric

FEATURE_NAMES: tuple[str, ...] = ("entropy", "n_predictors", "is_numeric")


def _normalized_entropy(series: pd.Series, numeric: bool) -> float:
    """Shannon entropy of the univariate marginal, normalized to [0, 1]. Numeric
    columns are decile-binned first so a single 'spread/diversity' feature is
    comparable across numeric and categorical outcomes. Reads only the marginal."""
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    if numeric:
        v = pd.to_numeric(s, errors="coerce").dropna()
        if v.nunique() <= 1:
            return 0.0
        binned = pd.qcut(v, min(10, v.nunique()), duplicates="drop")
        counts = binned.value_counts()
    else:
        counts = s.value_counts()
    p = (counts / counts.sum()).to_numpy()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def outcome_features(pool: pd.DataFrame, outcome: str, predictors: list[str],
                     *, numeric_predictors: frozenset[str] = frozenset()) -> dict[str, float]:
    """Firewall-clean structural features of one outcome, from public marginals +
    crosswalk structure only. Never reads the joint. Keys: entropy (normalized
    marginal diversity), n_predictors (usable predictor count), is_numeric."""
    numeric = _is_numeric(pool[outcome])
    preds = [c for c in predictors if c in pool.columns]
    return {
        "entropy": _normalized_entropy(pool[outcome], numeric),
        "n_predictors": float(len(preds)),
        "is_numeric": 1.0 if numeric else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_rescue.py::test_outcome_features_shape_and_ranges -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/rescue.py tests/test_transfer_rescue.py
git commit -m "feat(transfer): B5 outcome structural features"
```

---

### Task 2: The empirical-Bayes model (fit + predict)

**Files:**
- Modify: `src/ssdataagent/transfer/rescue.py`
- Test: `tests/test_transfer_rescue.py`

**Interfaces:**
- Consumes: `FEATURE_NAMES` from Task 1.
- Produces:
  - `PriorFit` dataclass with `.predict(feats: dict) -> float` and `.tau2: float`.
  - `NoiseFit` dataclass with `.sigma2(ess: float) -> float`.
  - `fit_prior(rows: list[dict], *, tau2_floor=1e-4) -> PriorFit` — rows carry `FEATURE_NAMES` keys + `"true_r2"`.
  - `fit_noise(points: list[tuple[float, float]], *, floor=1e-4) -> NoiseFit` — points are `(ess, squared_error)`.
  - `predict_r2(x_co: float | None, ess: float, feats: dict, prior: PriorFit, noise: NoiseFit, *, clip=(0.0, 1.0)) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_transfer_rescue.py
from ssdataagent.transfer.rescue import (
    FEATURE_NAMES, PriorFit, NoiseFit, fit_prior, fit_noise, predict_r2,
)


def _rows(pairs):
    # pairs: list of (entropy, n_predictors, is_numeric, true_r2)
    return [dict(zip((*FEATURE_NAMES, "true_r2"), p)) for p in pairs]


def test_fit_prior_recovers_linear_signal():
    # true_r2 = 0.1 + 0.5*entropy exactly -> prediction matches at a query point.
    rows = _rows([(e, 3.0, 1.0, 0.1 + 0.5 * e) for e in (0.0, 0.25, 0.5, 0.75, 1.0)])
    prior = fit_prior(rows)
    got = prior.predict({"entropy": 0.4, "n_predictors": 3.0, "is_numeric": 1.0})
    assert abs(got - (0.1 + 0.5 * 0.4)) < 1e-6


def test_predict_limits():
    prior = fit_prior(_rows([(e, 3.0, 1.0, 0.3) for e in (0.0, 0.5, 1.0)]))  # mu == 0.3
    feats = {"entropy": 0.5, "n_predictors": 3.0, "is_numeric": 1.0}
    # sigma^2 -> 0 (huge precision on retrieval): posterior == x_co
    tiny = NoiseFit(a=1e-12, b=0.0)
    assert abs(predict_r2(0.9, ess=0.5, feats=feats, prior=prior, noise=tiny) - 0.9) < 1e-3
    # x_co is None: posterior == mu
    assert abs(predict_r2(None, ess=0.5, feats=feats, prior=prior, noise=tiny) - 0.3) < 1e-9
    # low ess pushes posterior toward mu vs high ess (monotone shrinkage)
    noise = NoiseFit(a=0.0, b=0.02)
    hi = predict_r2(0.9, ess=0.65, feats=feats, prior=prior, noise=noise)
    lo = predict_r2(0.9, ess=0.10, feats=feats, prior=prior, noise=noise)
    assert abs(lo - 0.3) < abs(hi - 0.3)              # lo is closer to the prior
    assert 0.3 < lo < hi < 0.9


def test_predict_clips_to_unit_interval():
    prior = fit_prior(_rows([(e, 3.0, 1.0, 0.3) for e in (0.0, 0.5, 1.0)]))
    feats = {"entropy": 0.5, "n_predictors": 3.0, "is_numeric": 1.0}
    out = predict_r2(5.0, ess=1.0, feats=feats, prior=prior, noise=NoiseFit(a=1e-12, b=0.0))
    assert out == 1.0


def test_fit_noise_single_point_and_curve():
    # single point: all noise in the 1/ess term, sigma2 recovers the point
    nf1 = fit_noise([(0.5, 0.04)])
    assert abs(nf1.sigma2(0.5) - 0.04) < 1e-9
    # two points on sigma2 = 1/ess line: b=1, a=0
    nf2 = fit_noise([(0.5, 2.0), (0.25, 4.0)])
    assert abs(nf2.a) < 1e-6 and abs(nf2.b - 1.0) < 1e-6
    # sigma2 decreases as ess grows
    assert nf2.sigma2(1.0) < nf2.sigma2(0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_rescue.py -k "fit_prior or predict or fit_noise" -v`
Expected: FAIL with `cannot import name 'PriorFit'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/ssdataagent/transfer/rescue.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PriorFit:
    """Cross-context pooled prior: mu(features) = intercept + standardized-feature
    slopes, with tau2 the between-context residual variance (empirical-Bayes)."""
    feature_names: tuple[str, ...]
    coef: np.ndarray        # length 1 + n_features (intercept first)
    mean: np.ndarray        # feature means (centering)
    scale: np.ndarray       # feature stds (scaling; zeros replaced by 1)
    tau2: float

    def predict(self, feats: dict) -> float:
        x = np.array([feats[n] for n in self.feature_names], dtype=float)
        xs = (x - self.mean) / self.scale
        return float(self.coef[0] + xs @ self.coef[1:])


@dataclass(frozen=True)
class NoiseFit:
    """Retrieval-noise curve sigma2(ess) = max(a + b/ess, floor). Fit where the
    per-sibling spread is measurable (cps pseudo-targets); extrapolated to gss."""
    a: float
    b: float
    floor: float = 1e-4

    def sigma2(self, ess: float) -> float:
        return max(self.a + self.b / max(ess, 1e-6), self.floor)


def fit_prior(rows: list[dict], *, tau2_floor: float = 1e-4) -> PriorFit:
    """GLS/OLS of true_r2 on standardized structural features across all training
    (context, outcome) rows. tau2 = residual variance (dof-corrected), floored."""
    F = np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["true_r2"] for r in rows], dtype=float)
    mean = F.mean(axis=0)
    scale = F.std(axis=0)
    scale[scale == 0] = 1.0
    Fs = (F - mean) / scale
    X = np.column_stack([np.ones(len(y)), Fs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    tau2 = max(float(resid @ resid) / dof, tau2_floor)
    return PriorFit(FEATURE_NAMES, coef, mean, scale, tau2)


def fit_noise(points: list[tuple[float, float]], *, floor: float = 1e-4) -> NoiseFit:
    """Fit sigma2(ess) = a + b/ess from (ess, squared_error) calibration points by
    OLS, clamping a, b >= 0. With a single point the fit is underdetermined, so all
    noise is attributed to the 1/ess term (a=0) -- the conservative choice that
    makes sigma2 grow as ess shrinks."""
    if not points:
        return NoiseFit(a=floor, b=0.0, floor=floor)
    if len(points) == 1:
        e0, se0 = points[0]
        return NoiseFit(a=0.0, b=max(se0 * max(e0, 1e-6), 0.0), floor=floor)
    E = np.array([[1.0, 1.0 / max(e, 1e-6)] for e, _ in points], dtype=float)
    y = np.array([se for _, se in points], dtype=float)
    ab, *_ = np.linalg.lstsq(E, y, rcond=None)
    return NoiseFit(a=max(float(ab[0]), 0.0), b=max(float(ab[1]), 0.0), floor=floor)


def predict_r2(x_co: float | None, ess: float, feats: dict,
               prior: PriorFit, noise: NoiseFit, *,
               clip: tuple[float, float] = (0.0, 1.0)) -> float:
    """Empirical-Bayes posterior R^2: precision-weighted blend of the retrieval
    estimate x_co (precision 1/sigma2(ess)) and the pooled prior mu (precision
    1/tau2). x_co None -> pure prior. Clipped to the unit interval."""
    mu = prior.predict(feats)
    if x_co is None:
        post = mu
    else:
        s2 = noise.sigma2(ess)
        t2 = prior.tau2
        post = (x_co / s2 + mu / t2) / (1.0 / s2 + 1.0 / t2)
    return float(np.clip(post, clip[0], clip[1]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_rescue.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/rescue.py tests/test_transfer_rescue.py
git commit -m "feat(transfer): B5 empirical-Bayes prior + noise + predict"
```

---

### Task 3: `r2_target` override seam on `transfer_build_b2`

**Files:**
- Modify: `src/ssdataagent/transfer/generate.py:94-179`
- Test: `tests/test_transfer_generate_b2.py`

**Interfaces:**
- Consumes: existing `transfer_build_b2` signature.
- Produces: `transfer_build_b2(..., *, r2_pool=None, r2_target: dict | None = None)`. When `r2_target` is a dict it is used verbatim as the Step-B R² map (skipping `target_aggregates` for the R² target); `None` (default) preserves today's behavior byte-for-byte. `r2_target` takes precedence over `r2_pool` for the R² map.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transfer_generate_b2.py  (create if absent, with the sys.path
# preamble used by the other transfer tests)
import numpy as np
import pandas as pd
from ssdataagent.transfer.generate import transfer_build_b2


def _toy_pools():
    rng = np.random.default_rng(0)
    n = 400
    src = pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "sex": rng.choice(["M", "F"], n),
        "income": rng.normal(50, 10, n),
    })
    tgt = src.sample(frac=1.0, random_state=1).reset_index(drop=True)
    return src, tgt


def test_r2_target_override_changes_output_and_default_is_unchanged():
    src, tgt = _toy_pools()
    cols = ["age", "sex", "income"]
    covs, outs = ["age", "sex"], ["income"]
    base = transfer_build_b2(src, tgt, cols, covs, outs, 300, 7)
    # A very low R^2 target must pull income's covariate-R^2 down vs the default.
    forced = transfer_build_b2(src, tgt, cols, covs, outs, 300, 7,
                               r2_target={"income": 0.0})
    from ssdataagent.data.conditional_variance import covariate_r2
    r2_base = covariate_r2(base, "income", covs, numeric_predictors=frozenset({"age"}))
    r2_forced = covariate_r2(forced, "income", covs, numeric_predictors=frozenset({"age"}))
    assert r2_forced <= r2_base + 1e-9
    # Default None path is byte-identical to a second default call (determinism gate).
    base2 = transfer_build_b2(src, tgt, cols, covs, outs, 300, 7)
    pd.testing.assert_frame_equal(base, base2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_generate_b2.py::test_r2_target_override_changes_output_and_default_is_unchanged -v`
Expected: FAIL with `unexpected keyword argument 'r2_target'`.

- [ ] **Step 3: Write minimal implementation**

In `src/ssdataagent/transfer/generate.py`, change the signature (line 94-97) to add the keyword-only param:

```python
def transfer_build_b2(source_pool: pd.DataFrame, target_pool: pd.DataFrame,
                      cols: list[str], covariates: list[str], outcomes: list[str],
                      n: int, seed: int, *,
                      r2_pool: pd.DataFrame | None = None,
                      r2_target: dict | None = None) -> pd.DataFrame:
```

Replace the R²-map derivation (currently lines 128-129):

```python
    r2_frame = target_pool if r2_pool is None else r2_pool
    agg = target_aggregates(r2_frame, cols, covariates, outcomes)
```

with:

```python
    # r2_target (a precomputed per-outcome R^2 dict, e.g. B5's EB prediction) wins
    # outright and skips target_aggregates. Otherwise the R^2 target is read from
    # r2_pool (B4) or the target pool (B2), exactly as before.
    if r2_target is not None:
        r2_map = r2_target
    else:
        r2_frame = target_pool if r2_pool is None else r2_pool
        r2_map = target_aggregates(r2_frame, cols, covariates, outcomes)["outcome_r2"]
```

Update the `bidirectional_r2_blend` call (line 178) to pass `r2_map`:

```python
    return bidirectional_r2_blend(frame, outcomes, covariates, r2_map,
                                  numeric_predictors=num_pred, rng=rng)
```

Update the docstring's `r2_pool` paragraph to add one sentence documenting
`r2_target` precedence.

- [ ] **Step 4: Run the seam test AND the existing B2 byte-identical test**

Run: `.venv/bin/python -m pytest tests/test_transfer_generate_b2.py -v`
Expected: PASS, including any pre-existing byte-identical/repro test in that file.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/generate.py tests/test_transfer_generate_b2.py
git commit -m "feat(transfer): r2_target override seam on transfer_build_b2 (B5)"
```

---

### Task 4: Training corpus + noise calibration builders

**Files:**
- Create: `scripts/transfer_b5.py`
- Test: `tests/test_transfer_b5.py`

**Interfaces:**
- Consumes: `rescue.outcome_features`; `transfer_b4.{b4_columns, reweighted_pool_for}`; `retrieval.reweighted_pool`; `pairs.covariates_outcomes`; `data.conditional_variance.covariate_r2`; `data.schema.load_schema`; `config.data_root`; `generate._is_numeric`; `nodonor_bracket._drop_unnamed`.
- Produces (all importable from `transfer_b5`):
  - `corpus_contexts() -> list[tuple[str, Path, str]]` — `(context_id, csv_path, schema_name)` for every schema-backed context on disk.
  - `context_records(context_id, csv, schema_name) -> list[dict]` — one row per resolvable native outcome: `FEATURE_NAMES` keys + `true_r2` + `context`/`outcome` labels.
  - `training_rows(exclude_context_ids) -> list[dict]` — pooled `context_records` across the corpus, excluding the held-out target context(s).
  - `noise_points(exclude_csv) -> list[tuple[float, float]]` — `(ess, squared_error)` from cps waves treated as pseudo-targets (LOCO-safe; gss waves are never used as calibration because their only sibling is the other gss wave).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transfer_b5.py
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_corpus_contexts_include_cps_gss_waves():
    import transfer_b5
    ctx = transfer_b5.corpus_contexts()
    ids = {c[0] for c in ctx}
    # all four cps waves and both gss waves are present as contexts
    assert {"cps_1970", "cps_1980", "cps_1990", "cps_2000"} <= ids
    assert {"gss1994", "gss2018"} <= ids
    for _, csv, _ in ctx:
        assert csv.exists()


def test_context_records_are_wellformed():
    import transfer_b5
    rows = transfer_b5.context_records("cps_1980",
                                       REPO / "real_data" / "cps" / "cps-asec1980.csv",
                                       "cps")
    assert rows, "cps_1980 must resolve at least one outcome R^2"
    for r in rows:
        assert {"entropy", "n_predictors", "is_numeric", "true_r2", "context", "outcome"} <= set(r)
        assert 0.0 <= r["true_r2"] <= 1.0 or r["true_r2"] != r["true_r2"]  # in [0,1] or NaN-guarded


def test_training_rows_exclude_target():
    import transfer_b5
    all_rows = transfer_b5.training_rows(exclude_context_ids=set())
    held = transfer_b5.training_rows(exclude_context_ids={"gss2018"})
    assert all(r["context"] != "gss2018" for r in held)
    assert len(held) < len(all_rows)


def test_noise_points_from_cps_are_positive():
    import transfer_b5
    pts = transfer_b5.noise_points(exclude_csv=REPO / "real_data" / "gss" / "gss2018.csv")
    assert len(pts) >= 2                    # >=2 cps pseudo-targets -> a real curve
    for ess, sq in pts:
        assert 0.0 < ess <= 1.0 and sq >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_b5.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'transfer_b5'`.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python
"""B5 -- learned R^2 rescue (Phase 3, slice 2). Empirical-Bayes shrinkage of B4's
retrieval R^2 toward a cross-context pooled prior, ESS-gated. LLM-free.

    .venv/bin/python scripts/transfer_b5.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
    .venv/bin/python scripts/transfer_b5.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
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

from ssdataagent.config import data_root  # noqa: E402
from ssdataagent.data.conditional_variance import covariate_r2  # noqa: E402
from ssdataagent.data.schema import load_schema  # noqa: E402
from ssdataagent.transfer.generate import _is_numeric, transfer_build_b2  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes  # noqa: E402
from ssdataagent.transfer.retrieval import reweighted_pool  # noqa: E402
from ssdataagent.transfer.rescue import (  # noqa: E402
    fit_noise, fit_prior, outcome_features, predict_r2,
)
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402
from ssdataagent.transfer.target_aggregates import target_aggregates  # noqa: E402
from transfer_b4 import b4_columns, reweighted_pool_for  # noqa: E402
from transfer_map import composition_covariates  # noqa: E402

OUT = REPO / "results" / "transfer_map"

# Single-wave, schema-backed datasets contributing prior rows only (no siblings).
_SINGLE_WAVE = ("acs", "nlsy", "addhealth", "cfps", "us")


def _load(csv: Path) -> pd.DataFrame:
    import nodonor_bracket as nb
    return nb._drop_unnamed(pd.read_csv(csv, low_memory=False))


def corpus_contexts() -> list[tuple[str, Path, str]]:
    """(context_id, csv, schema_name) for every schema-backed context on disk. cps
    and gss enumerate all waves (same-instrument siblings); single-wave datasets use
    their schema real_data_path. Skips datasets whose CSV is absent."""
    out: list[tuple[str, Path, str]] = []
    for csv in sorted((data_root() / "cps").glob("cps-asec*.csv")):
        out.append((f"cps_{csv.stem[-4:]}", csv, "cps"))
    for csv in sorted((data_root() / "gss").glob("gss*.csv")):
        out.append((csv.stem, csv, "gss"))
    for name in _SINGLE_WAVE:
        try:
            p = load_schema(name).real_data_path
        except KeyError:
            continue
        if p.exists():
            out.append((name, p, name))
    return out


def context_records(context_id: str, csv: Path, schema_name: str) -> list[dict]:
    """True covariate-R^2 + structural features for every resolvable native outcome
    of one context. Outcomes whose R^2 is None (too few rows / categorical) are
    skipped with no row. This is the ground truth the prior is fit on."""
    df = _load(csv)
    sch = load_schema(schema_name)
    covs = [c for c in sch.background_variables if c in df.columns]
    outs = [c for c in sch.target_variables if c in df.columns]
    num_pred = frozenset(c for c in covs if _is_numeric(df[c]))
    rows: list[dict] = []
    for o in outs:
        r2 = covariate_r2(df, o, covs, numeric_predictors=num_pred)
        if r2 is None or r2 != r2:                       # None or NaN -> unusable
            continue
        feats = outcome_features(df, o, covs, numeric_predictors=num_pred)
        rows.append({"context": context_id, "outcome": o,
                     "true_r2": float(np.clip(r2, 0.0, 1.0)), **feats})
    return rows


def training_rows(exclude_context_ids: set[str]) -> list[dict]:
    """All context_records across the corpus, minus the held-out context(s)."""
    rows: list[dict] = []
    for cid, csv, sch in corpus_contexts():
        if cid in exclude_context_ids:
            continue
        rows.extend(context_records(cid, csv, sch))
    return rows


def _cps_wave_csvs() -> list[Path]:
    return sorted((data_root() / "cps").glob("cps-asec*.csv"))


def noise_points(exclude_csv: Path) -> list[tuple[float, float]]:
    """(ess, squared_error) calibration points from cps waves as pseudo-targets: for
    each cps wave w (!= exclude_csv), rake the OTHER cps waves to w's margins, and
    compare the transported R^2 against w's TRUE R^2 per shared outcome. cps always
    has >=2 remaining siblings, so every ESS point is well-supported. gss is never a
    pseudo-target (its only sibling is the real target -> would leak)."""
    exclude = exclude_csv.resolve()
    waves = _cps_wave_csvs()
    pts: list[tuple[float, float]] = []
    for w in waves:
        if w.resolve() == exclude:
            continue
        sibs = [s for s in waves if s.resolve() != w.resolve()]
        if len(sibs) < 1:
            continue
        wpool = _load(w)
        sch = load_schema("cps")
        cols = [c for c in (list(sch.background_variables) + list(sch.target_variables))
                if c in wpool.columns and all(c in _load(s).columns for s in sibs)]
        covs, outs = covariates_outcomes("cps", cols)
        rake = composition_covariates(cols)
        sib_frames = [_load(s)[cols] for s in sibs]
        stack_n = sum(len(f) for f in sib_frames)
        sib_rew, ess = reweighted_pool(sib_frames, wpool, cols, rake, stack_n,
                                       np.random.default_rng(0))
        num_pred = frozenset(c for c in covs if _is_numeric(wpool[c]))
        x = target_aggregates(sib_rew, cols, covs, outs)["outcome_r2"]
        for o in outs:
            true = covariate_r2(wpool, o, covs, numeric_predictors=num_pred)
            if true is None or x.get(o) is None:
                continue
            pts.append((ess, float((x[o] - true) ** 2)))
    return pts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_b5.py -v`
Expected: PASS (all four corpus/calibration tests). This is a real-data test off
the CPS/GSS microdata; it may take up to ~120s.

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_b5.py tests/test_transfer_b5.py
git commit -m "feat(transfer): B5 training corpus + noise calibration builders"
```

---

### Task 5: Orchestrator — fit LOCO, predict, score both configs

**Files:**
- Modify: `scripts/transfer_b5.py`
- Test: `tests/test_transfer_b5.py`

**Interfaces:**
- Consumes: everything from Task 4 plus `rescue.{fit_prior, fit_noise, predict_r2}`; `transfer_b4.{b4_columns, reweighted_pool_for}`; `transfer_build_b2` `r2_target` seam (Task 3); `nodonor_bracket.{carve_pool, score, TYPES}`.
- Produces:
  - `predict_target_r2(pair) -> tuple[dict, dict, float, pd.DataFrame]` — returns `(learned_r2, prior_only_r2, ess, sib_rew)`: two per-outcome R² dicts (EB posterior and prior-only) for the scored pair's crosswalk outcomes, the target's retrieval ESS, and B4's raked sibling pseudo-population (the structure vehicle, returned so the scorer reuses the identical draw).
  - `run_b5(pair, *, seeds, n, bootstrap_B) -> pd.DataFrame` — scores `B5_learned` and `B5_prior_only`, writes `results/transfer_map/b5_<pair>.csv`.
  - `main()`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transfer_b5.py
def test_run_b5_smoke_scores_both_configs(tmp_path, monkeypatch):
    import transfer_b5
    from ssdataagent.transfer.pairs import PAIRS
    monkeypatch.setattr(transfer_b5, "OUT", tmp_path)
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    df = transfer_b5.run_b5(pair, seeds=2, n=800, bootstrap_B=50)
    assert list(df["config"]) == ["B5_learned", "B5_prior_only"]
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all()
        assert (df[col] >= 0).all() and (df[col] <= 1).all()
    assert "ess_ratio" in df.columns
    assert (tmp_path / "b5_cps_1970_1980.csv").exists()


def test_predict_target_r2_shapes(tmp_path):
    import transfer_b5
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    learned, prior_only, ess, sib_rew = transfer_b5.predict_target_r2(pair)
    assert set(learned) == set(prior_only)          # same outcome keys
    assert len(sib_rew) > 0                          # structure vehicle materialized
    assert learned                                   # non-empty
    assert 0.0 < ess <= 1.0
    for d in (learned, prior_only):
        for v in d.values():
            assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_b5.py::test_predict_target_r2_shapes -v`
Expected: FAIL with `module 'transfer_b5' has no attribute 'predict_target_r2'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/transfer_b5.py`:

```python
def _target_context_id(pair) -> str:
    """The corpus context_id of the pair's TARGET wave, to hold it out under LOCO."""
    stem = pair.target_csv.stem
    if pair.schema_name == "cps":
        return f"cps_{stem[-4:]}"
    return stem                                       # gss1994 / gss2018


def predict_target_r2(pair):
    """Fit the EB model LOCO (target wave excluded), then predict the scored pair's
    crosswalk-outcome R^2 two ways: full posterior (learned) and prior-only. Also
    returns B4's raked sibling pool (the structure vehicle) so the scorer draws
    through the IDENTICAL vehicle B4 used -- B5 vs B4 then differs ONLY in the R^2
    target. Returns (learned_r2, prior_only_r2, ess, sib_rew)."""
    import nodonor_bracket as nb
    ds, cols, covs, outs = b4_columns(pair)
    target_pool, _ = nb.carve_pool(ds)

    # Retrieval data point x_co + ESS, reusing B4's raked sibling pool (default_rng(0)
    # -> byte-identical to B4's sib_rew).
    sib_rew, ess, _, _ = reweighted_pool_for(pair, cols, target_pool,
                                             np.random.default_rng(0))
    x_co = target_aggregates(sib_rew, cols, covs, outs)["outcome_r2"]

    # Fit prior on all contexts except the held-out target wave; noise on cps waves
    # (excluding the target if it is a cps wave).
    prior = fit_prior(training_rows({_target_context_id(pair)}))
    noise = fit_noise(noise_points(pair.target_csv))
    print(f"{pair.id}: prior tau2 {prior.tau2:.4f} coef {np.round(prior.coef, 3)}; "
          f"noise a {noise.a:.4f} b {noise.b:.4f}; target ess {ess:.3f}")

    num_pred = frozenset(c for c in covs if _is_numeric(target_pool[c]))
    learned, prior_only = {}, {}
    for o in outs:
        feats = outcome_features(target_pool, o, covs, numeric_predictors=num_pred)
        learned[o] = predict_r2(x_co.get(o), ess, feats, prior, noise)
        prior_only[o] = predict_r2(None, ess, feats, prior, noise)
    return learned, prior_only, ess, sib_rew


def run_b5(pair, *, seeds, n, bootstrap_B):
    """Score B5_learned (EB posterior R^2) and B5_prior_only (prior-only R^2) through
    the B2 machinery via the r2_target seam, identically to B0-B4. The structure
    vehicle is B4's raked sibling pool ``sib_rew`` (source_pool), so B5 differs from
    B4_retrieval ONLY in where the per-outcome R^2 comes from. Writes CSV."""
    import nodonor_bracket as nb
    ds, cols, covs, outs = b4_columns(pair)
    target_pool, guarantee = nb.carve_pool(ds)
    ref = _load(load_schema(ds).real_data_path)
    types = nb.TYPES.get(ds, (1, 2, 3))

    learned, prior_only, ess, sib_rew = predict_target_r2(pair)
    configs = {"B5_learned": learned, "B5_prior_only": prior_only}

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(load_schema(ds).ssdatabench_sim_subdir,
                                      set(cols), types, Path(cfg_td))
        for name, r2_map in configs.items():
            recs = []
            for s in range(1, seeds + 1):
                sim = transfer_build_b2(sib_rew, target_pool, cols, covs, outs,
                                        n, s, r2_target=r2_map)
                recs.append(nb.score(sim, ds, ref, types, seed=1000 + s,
                                     bootstrap_B=bootstrap_B, config_dir=cfg_dir))
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "ess_ratio": round(ess, 4)}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"b5_{pair.id}.csv", index=False)
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
    df = run_b5(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'b5_{pair.id}.csv'}")
    print("REGIME: no-donor + learned. Target supplies marginals + X-margins + public"
          " outcome features only; conditional strength is an EB blend of retrieval"
          " and a cross-context prior.")


if __name__ == "__main__":
    main()
```

Note on the generator call: `transfer_build_b2(sib_rew, target_pool, ...)` uses B4's
raked sibling pseudo-population `sib_rew` as the structure source and the target pool
as the marginals — the EXACT vehicle B4_retrieval used. B5 swaps only the R² target
(via `r2_target`), holding the T1/T2 draw fixed. This keeps the headline comparison
clean: `B5_learned − B4_retrieval` isolates learned-vs-transported R², nothing else.
Because `sib_rew` is rebuilt with `default_rng(0)` (as in B4) and the generation seed
`s` is identical, the T1/T2 columns are byte-identical to B4's for the same seed;
only the recalibrated numeric outcomes move.

- [ ] **Step 4: Run the smoke + shape tests**

Run: `.venv/bin/python -m pytest tests/test_transfer_b5.py -v`
Expected: PASS. The smoke test runs the full fit→predict→score path on cps at
small N; it may take a few minutes.

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_b5.py tests/test_transfer_b5.py
git commit -m "feat(transfer): B5 orchestrator -- LOCO fit, predict, score"
```

---

## Post-implementation (controller, after all tasks reviewed)

These are NOT subagent tasks — the controller runs them after the whole-branch
review, mirroring the B4 landing:

1. **Full-protocol runs** (`seeds=3 n=3000 bootstrap_B=200`) for both scored
   pairs. The box reaps heavy scoring jobs even solo
   (`[[project_b4_retrieval_kob]]` env note) — run via a resumable per-(config,
   seed) scorer under `.superpowers/sdd/` that persists each score and resumes,
   numerically identical to `run_b5` (reuse its functions + `default_rng(0)`).
2. **Report** `docs/report/2026-07-23-b5-learned-r2-rescue.md`: the B2 / B4 / B5
   table per T-type, the fitted μ coefficients + τ² + σ²(ESS) curve with gss
   marked off-support, and the verdict (rescue vs honest-null → sizes Phase 4).
3. **LEDGER row** `b5_learned_r2_rescue` (newest-on-top) with a meaningful
   `hypothesis` one-liner, then rebuild the dashboard
   (`.venv/bin/python scripts/build_dashboard.py`) and commit
   `docs/dashboard/index.html`.
4. **Memory**: update `[[project_b4_retrieval_kob]]` with the B5 outcome and add
   a `project_b5_learned_r2_rescue` note + MEMORY.md index line.
