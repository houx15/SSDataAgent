# B2 — aggregate-recalibration baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add B2 to the transfer baseline ladder — a firewalled no-donor generator that keeps the source context's dependence *structure* but recalibrates its *strengths* (per-pair association → T2, per-outcome R² → T3) to the target's published-aggregate statistics.

**Architecture:** An explicit two-step Gaussian-copula generator. Step A fits the source's latent correlation matrix, edits unstable pairs toward the target pool's associations (bidirectional, sign-preserving), PSD-projects, draws, and inverse-CDFs onto target marginals. Step B nudges each numeric outcome's observed R² onto the target pool's covariate-R² by blending its rank toward an independent pole (weaken) or a conditional-mean pole (strengthen). Both steps read only low-order aggregates of the target's disjoint pool — never its joint or test sample.

**Tech Stack:** Python, numpy, pandas, scipy.stats (`norm`, `kendalltau`), pytest. Reuses `src/ssdataagent/transfer/{generate,copula_stability}.py` and `src/ssdataagent/data/conditional_variance.py`.

## Global Constraints

- **LLM-free & deterministic.** B2 uses no LLM. Every generator draw is seeded (`np.random.default_rng(seed)`); tests are deterministic.
- **Firewall.** B2 may read from the target only: per-column marginals, per-pair associations, and per-outcome covariate-R² — all computed on the target's **disjoint pool** (`nb.carve_pool`). It must NEVER read the target's per-person joint or the target reference/test sample. Every recalibration value carries a provenance tag.
- **Stability threshold is `0.10`** — the same `|Δ|` cut the transfer map uses (`copula_stability(..., threshold=0.10)`).
- **Sign is source-owned.** Recalibration changes association *magnitude* only; the sign/structure of each latent correlation entry stays as fitted on the source (Result 1: structure is ~stable). Nominal (Cramér's V) pairs are unsigned → magnitude-only by construction.
- **Do not touch `transfer_build`** (B0/B1 path) or `nodonor_bracket.build` (frozen replication path). B2 is additive.
- **Reuse existing scoring plumbing:** `restrict_config_dir` (crosswalk-only config, `yaml.safe_dump sort_keys=False`) and `mean_scores` (ignores `T{t}_error` string columns). Do not reimplement them.
- **Numeric detection everywhere uses the project idiom:** `s.dropna()` first, then `pd.to_numeric(s, errors="coerce").notna().mean() > 0.9` (the existing `_is_numeric`/`_is_num`). Never gate on completeness fraction before dropna.

---

### Task 1: Gaussian-copula primitives

**Files:**
- Create: `src/ssdataagent/transfer/gaussian_copula.py`
- Test: `tests/test_transfer_gaussian_copula.py`

**Interfaces:**
- Consumes: `generate._latent(pool, c, num, glat, rng)`, `generate._marginal_map(marg_col, u, num)`, `generate._is_numeric(s)` (existing, in `src/ssdataagent/transfer/generate.py`).
- Produces:
  - `fit_latent_correlation(pool: pd.DataFrame, cols: list[str], *, seed: int = 0) -> tuple[np.ndarray, dict[str, bool]]` — returns `(R, num)` where `R` is the `len(cols)×len(cols)` latent correlation matrix and `num[c]` is True when column `c` is numeric.
  - `nearest_correlation(R: np.ndarray, *, eps: float = 1e-6) -> np.ndarray` — nearest positive-definite correlation matrix (symmetric, unit diagonal, eigenvalues ≥ `eps`).
  - `draw_copula(R: np.ndarray, n: int, seed: int) -> np.ndarray` — `n×d` array of uniforms from the Gaussian copula with correlation `R`.
  - `copula_to_frame(u: np.ndarray, marg: pd.DataFrame, cols: list[str], num: dict[str, bool], rng: np.random.Generator) -> pd.DataFrame` — inverse-CDF each uniform column onto `marg[c]`'s marginal (delegating to `_marginal_map`), plus target missingness rate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_gaussian_copula.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ssdataagent.transfer.gaussian_copula import (
    copula_to_frame, draw_copula, fit_latent_correlation, nearest_correlation,
)


def _gauss(n, seed, rho):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return pd.DataFrame({"x": z[:, 0], "y": z[:, 1]})


def test_fit_recovers_known_correlation():
    df = _gauss(4000, 1, rho=0.6)
    R, num = fit_latent_correlation(df, ["x", "y"])
    assert num == {"x": True, "y": True}
    assert R.shape == (2, 2)
    assert abs(R[0, 1] - 0.6) < 0.06          # latent corr ~ generating rho


def test_nearest_correlation_repairs_non_psd():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    good = nearest_correlation(bad)
    w = np.linalg.eigvalsh(good)
    assert w.min() >= 0.0
    assert np.allclose(np.diag(good), 1.0)
    assert np.allclose(good, good.T)


def test_draw_and_map_reproduces_dependence_and_marginal():
    R = np.array([[1.0, 0.7], [0.7, 1.0]])
    u = draw_copula(R, 5000, seed=3)
    assert u.shape == (5000, 2)
    assert u.min() > 0.0 and u.max() < 1.0
    # inverse-CDF onto a target marginal preserves that marginal
    marg = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    frame = copula_to_frame(u, marg, ["a", "b"], {"a": True, "b": True},
                            np.random.default_rng(0))
    assert frame.shape == (5000, 2)
    # dependence survived the copula+map
    tau, _ = kendalltau(pd.to_numeric(frame["a"]), pd.to_numeric(frame["b"]))
    assert tau > 0.3
    # marginal support stays within the target's
    assert frame["a"].min() >= 0.0 and frame["a"].max() <= 99.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_transfer_gaussian_copula.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.transfer.gaussian_copula'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/transfer/gaussian_copula.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ssdataagent.transfer.generate import _is_numeric, _latent, _marginal_map

_EPS = 1e-6


def fit_latent_correlation(pool: pd.DataFrame, cols: list[str], *,
                           seed: int = 0) -> tuple[np.ndarray, dict[str, bool]]:
    """Latent Gaussian-copula correlation of ``cols`` in ``pool``.

    Each column is put through the same latent transform the generator uses
    (``_latent``: rank-copula for numeric, category-ordering for nominal), mapped to
    normal scores, and Pearson-correlated. This IS the source copula the generator draws
    from, so Step A's edits are on the same scale it will sample."""
    rng = np.random.default_rng(seed)
    num = {c: _is_numeric(pool[c]) for c in cols}
    znum = {c: pd.to_numeric(pool[c], errors="coerce").rank(pct=True)
            for c in cols if num[c]}
    glat = (pd.DataFrame(znum).mean(axis=1).fillna(0.5).to_numpy() if znum
            else np.full(len(pool), 0.5))
    z = np.column_stack([
        norm.ppf(np.clip(_latent(pool, c, num[c], glat, rng), _EPS, 1 - _EPS))
        for c in cols
    ])
    R = np.corrcoef(z, rowvar=False)
    if R.ndim == 0:                       # single column
        R = np.array([[1.0]])
    return np.nan_to_num(R, nan=0.0) * (1 - np.eye(len(cols))) + np.eye(len(cols)), num


def nearest_correlation(R: np.ndarray, *, eps: float = _EPS) -> np.ndarray:
    """Nearest PSD correlation matrix: symmetrize, clip eigenvalues to ``eps``, renormalize."""
    A = (R + R.T) / 2.0
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    A = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(A), eps, None))
    A = A / np.outer(d, d)
    np.fill_diagonal(A, 1.0)
    return A


def draw_copula(R: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` uniform draws from the Gaussian copula with correlation ``R``."""
    R = nearest_correlation(R)
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, R.shape[0])) @ L.T
    return np.clip(norm.cdf(z), _EPS, 1 - _EPS)


def copula_to_frame(u: np.ndarray, marg: pd.DataFrame, cols: list[str],
                    num: dict[str, bool], rng: np.random.Generator) -> pd.DataFrame:
    """Inverse-CDF each uniform column onto ``marg``'s marginal, applying its missingness rate."""
    n = u.shape[0]
    out: dict[str, np.ndarray] = {}
    for j, c in enumerate(cols):
        em = _marginal_map(marg[c], u[:, j], num[c])
        miss = float(marg[c].isna().mean())
        if miss > 0:
            mask = rng.random(n) < miss
            em[mask] = np.nan
        out[c] = em
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_transfer_gaussian_copula.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/gaussian_copula.py tests/test_transfer_gaussian_copula.py
git commit -m "feat(transfer): Gaussian-copula primitives for B2"
```

---

### Task 2: Step A — per-pair strength recalibration

**Files:**
- Create: `src/ssdataagent/transfer/recalibrate.py`
- Test: `tests/test_transfer_recalibrate.py`

**Interfaces:**
- Consumes: `gaussian_copula.nearest_correlation` (Task 1).
- Produces:
  - `tau_to_r(tau: float) -> float` — Kendall τ → Gaussian latent correlation, `sin(π·τ/2)`.
  - `recalibrate_matrix(R_source: np.ndarray, cols: list[str], a_src: dict, a_tgt: dict, methods: dict, *, threshold: float = 0.10) -> np.ndarray` — where `a_src`/`a_tgt` map an unordered pair `(v1, v2)` (with `v1`, `v2` in `cols` order) to an association value, and `methods` maps the same key to `"kendall"` | `"cramers_v"` | `"undefined"`. Returns a PSD-projected correlation matrix: stable/undefined/missing pairs keep `R_source`; unstable pairs move toward the target magnitude keeping the source sign.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_recalibrate.py
from __future__ import annotations

import numpy as np

from ssdataagent.transfer.recalibrate import recalibrate_matrix, tau_to_r


def test_tau_to_r_monotone_signed():
    assert tau_to_r(0.0) == 0.0
    assert tau_to_r(1.0) > 0.99
    assert tau_to_r(-0.5) < 0.0


def test_stable_pair_kept_unstable_moved_both_directions():
    cols = ["x", "y", "z"]
    R = np.array([[1.0, 0.50, 0.20],
                  [0.50, 1.0, -0.40],
                  [0.20, -0.40, 1.0]])
    a_src = {("x", "y"): 0.30, ("x", "z"): 0.13, ("y", "z"): -0.26}
    # x,y target ~equal (stable, keep); x,z target much stronger (strengthen, up);
    # y,z target much weaker (weaken toward 0, magnitude down) keeping negative sign
    a_tgt = {("x", "y"): 0.33, ("x", "z"): 0.55, ("y", "z"): -0.05}
    methods = {("x", "y"): "kendall", ("x", "z"): "kendall", ("y", "z"): "kendall"}
    Rp = recalibrate_matrix(R, cols, a_src, a_tgt, methods)
    # stable pair unchanged
    assert abs(Rp[0, 1] - 0.50) < 1e-9
    # strengthened pair moved up, same (positive) sign
    assert Rp[0, 2] > 0.20 and Rp[0, 2] > 0.0
    # weakened pair moved toward zero, still negative
    assert -0.40 < Rp[1, 2] < 0.0
    # symmetric + valid
    assert np.allclose(Rp, Rp.T)
    assert np.linalg.eigvalsh(Rp).min() >= 0.0


def test_undefined_and_missing_pairs_keep_source():
    cols = ["x", "y"]
    R = np.array([[1.0, 0.4], [0.4, 1.0]])
    Rp = recalibrate_matrix(R, cols, {("x", "y"): 0.2}, {("x", "y"): 0.9},
                            {("x", "y"): "undefined"})
    assert abs(Rp[0, 1] - 0.4) < 1e-9      # undefined -> untouched
    Rp2 = recalibrate_matrix(R, cols, {}, {}, {("x", "y"): "kendall"})
    assert abs(Rp2[0, 1] - 0.4) < 1e-9     # missing association -> untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_transfer_recalibrate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.transfer.recalibrate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/transfer/recalibrate.py
from __future__ import annotations

import numpy as np

from ssdataagent.transfer.gaussian_copula import nearest_correlation


def tau_to_r(tau: float) -> float:
    """Kendall's tau -> Gaussian latent correlation (the copula's closed form)."""
    return float(np.sin(np.pi * tau / 2.0))


def _target_entry(r_src: float, s: float, t: float, method: str) -> float:
    """New latent entry matching target magnitude ``t`` while keeping ``r_src``'s sign."""
    sign = 1.0 if r_src >= 0 else -1.0
    if method == "kendall":
        mag = abs(tau_to_r(t))
    else:  # cramers_v is unsigned: scale the current magnitude by the target/source ratio
        mag = abs(r_src) * (t / s) if s > 1e-9 else abs(r_src)
    return float(np.clip(sign * mag, -0.999, 0.999))


def recalibrate_matrix(R_source: np.ndarray, cols: list[str], a_src: dict,
                       a_tgt: dict, methods: dict, *,
                       threshold: float = 0.10) -> np.ndarray:
    """Edit unstable pairs of ``R_source`` toward the target association, then PSD-project.

    Stable (|a_tgt-a_src| < threshold), ``undefined``, and missing pairs keep the source
    entry. Unstable pairs move to the target magnitude, source sign preserved."""
    idx = {c: i for i, c in enumerate(cols)}
    R = np.array(R_source, dtype=float).copy()
    for key, method in methods.items():
        v1, v2 = key
        if v1 not in idx or v2 not in idx:
            continue
        s, t = a_src.get(key), a_tgt.get(key)
        if (method == "undefined" or s is None or t is None
                or not np.isfinite(s) or not np.isfinite(t)):
            continue
        if abs(t - s) < threshold:
            continue
        i, j = idx[v1], idx[v2]
        new = _target_entry(R[i, j], float(s), float(t), method)
        R[i, j] = R[j, i] = new
    return nearest_correlation(R)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_transfer_recalibrate.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/recalibrate.py tests/test_transfer_recalibrate.py
git commit -m "feat(transfer): Step-A per-pair strength recalibration"
```

---

### Task 3: Step B — bidirectional R² blend

**Files:**
- Modify: `src/ssdataagent/transfer/recalibrate.py` (add `bidirectional_r2_blend`)
- Test: `tests/test_transfer_recalibrate.py` (add cases)

**Interfaces:**
- Consumes: `conditional_variance.covariate_r2` (existing), `generate._is_numeric`, `generate._marginal_map`.
- Produces:
  - `bidirectional_r2_blend(frame: pd.DataFrame, outcomes: list[str], predictors: list[str], r2_target: dict[str, float | None], *, numeric_predictors: frozenset[str], rng: np.random.Generator, iters: int = 16) -> pd.DataFrame` — returns a copy of `frame` where each **numeric** outcome's observed covariate-R² is nudged onto `r2_target[outcome]` by blending its rank toward an independent pole (target below own) or a conditional-mean pole (target above own); the outcome's marginal is preserved by inverse-CDF back onto its own column. Non-numeric outcomes, and outcomes with unknown own/target R², are left unchanged.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transfer_recalibrate.py
import pandas as pd

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.recalibrate import bidirectional_r2_blend


def _linear(n, seed, beta):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": beta * x + rng.normal(0, 1, n)})


def test_blend_weakens_toward_low_target():
    df = _linear(4000, 1, beta=2.0)           # strong R^2 (~0.8)
    own = covariate_r2(df, "y", ["x"], numeric_predictors=frozenset({"x"}))
    out = bidirectional_r2_blend(df, ["y"], ["x"], {"y": 0.2},
                                 numeric_predictors=frozenset({"x"}),
                                 rng=np.random.default_rng(0))
    new = covariate_r2(out, "y", ["x"], numeric_predictors=frozenset({"x"}))
    assert own > 0.5 and new < own and abs(new - 0.2) < 0.12
    # marginal preserved: every output value is drawn from the source column's support
    # (_marginal_map is a quantile map, not a permutation) and the mean is close
    assert set(np.round(out["y"].astype(float), 6)).issubset(
        set(np.round(df["y"].astype(float), 6)))
    assert abs(pd.to_numeric(out["y"]).mean() - df["y"].mean()) < 0.15


def test_blend_strengthens_toward_high_target():
    df = _linear(4000, 2, beta=0.3)           # weak R^2 (~0.08)
    own = covariate_r2(df, "y", ["x"], numeric_predictors=frozenset({"x"}))
    out = bidirectional_r2_blend(df, ["y"], ["x"], {"y": 0.4},
                                 numeric_predictors=frozenset({"x"}),
                                 rng=np.random.default_rng(0))
    new = covariate_r2(out, "y", ["x"], numeric_predictors=frozenset({"x"}))
    assert new > own + 0.1                     # moved up toward the higher target


def test_blend_skips_unknown_and_nonnumeric():
    df = _linear(500, 3, beta=1.0)
    df["cat"] = np.where(df["x"] > 0, "hi", "lo")
    out = bidirectional_r2_blend(df, ["y", "cat"], ["x"], {"y": None},
                                 numeric_predictors=frozenset({"x"}),
                                 rng=np.random.default_rng(0))
    # y target unknown -> unchanged; cat non-numeric -> unchanged
    assert out["y"].equals(df["y"]) and out["cat"].equals(df["cat"])


def test_blend_leaves_covariate_columns_untouched():
    # The copula/R^2 seam: Step B only rewrites outcome columns, so covariate columns
    # (and hence covariate-covariate associations Step A set) are unchanged by construction.
    df = _linear(3000, 4, beta=2.0)
    df["age"] = df["x"]                              # a covariate that is NOT an outcome
    out = bidirectional_r2_blend(df, ["y"], ["x", "age"], {"y": 0.2},
                                 numeric_predictors=frozenset({"x", "age"}),
                                 rng=np.random.default_rng(0))
    assert out["x"].equals(df["x"]) and out["age"].equals(df["age"])
    assert not out["y"].equals(df["y"])             # the outcome DID change
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_transfer_recalibrate.py -q`
Expected: FAIL with `ImportError: cannot import name 'bidirectional_r2_blend'`

- [ ] **Step 3: Write minimal implementation**

Move these imports to the **top** of `recalibrate.py` (with the existing `import numpy as np`), not mid-file; the function body follows below.

```python
# add to the imports at the top of src/ssdataagent/transfer/recalibrate.py
import pandas as pd

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.generate import _is_numeric, _marginal_map

# module constant (top of file)
_EPS = 1e-9


def _rank_pct(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True, method="first").to_numpy()


def _ols_prediction(frame: pd.DataFrame, y: str, predictors: list[str],
                    num_pred: frozenset[str]) -> np.ndarray | None:
    """Fitted E[y|X] with a one-hot / numeric design (mirrors covariate_r2's design)."""
    yv = pd.to_numeric(frame[y], errors="coerce")
    blocks = []
    for p in predictors:
        if p not in frame.columns:
            continue
        if p in num_pred or _is_numeric(frame[p]):
            v = pd.to_numeric(frame[p], errors="coerce")
            blocks.append(v.fillna(v.mean()).to_numpy().reshape(-1, 1))
        else:
            d = pd.get_dummies(frame[p].astype("string"), dummy_na=True, dtype=float)
            blocks.append(d.to_numpy())
    ok = yv.notna().to_numpy()
    if not blocks or int(ok.sum()) < 20:
        return None
    design = np.column_stack(blocks)
    X = np.column_stack([np.ones(len(frame)), design])
    beta, *_ = np.linalg.lstsq(X[ok], yv.to_numpy()[ok], rcond=None)
    return X @ beta


def bidirectional_r2_blend(frame: pd.DataFrame, outcomes: list[str],
                           predictors: list[str], r2_target: dict,
                           *, numeric_predictors: frozenset[str],
                           rng: np.random.Generator, iters: int = 16) -> pd.DataFrame:
    """Nudge each numeric outcome's covariate-R^2 onto its target by blending toward an
    independent pole (weaken) or the conditional-mean pole (strengthen), preserving the
    outcome's marginal via inverse-CDF back onto its own column."""
    frame = frame.copy()
    num_pred = frozenset(numeric_predictors)
    preds = [p for p in predictors if p in frame.columns]
    for y in outcomes:
        if y not in frame.columns or not _is_numeric(frame[y]):
            continue
        target = r2_target.get(y)
        own = covariate_r2(frame, y, preds, numeric_predictors=num_pred)
        if target is None or own is None or own <= _EPS or abs(target - own) < 1e-3:
            continue
        yv = pd.to_numeric(frame[y], errors="coerce").to_numpy()
        u_cur = _rank_pct(yv)
        strengthen = target > own
        if strengthen:
            yhat = _ols_prediction(frame, y, preds, num_pred)
            if yhat is None:
                continue
            pole = _rank_pct(yhat)
        else:
            pole = _rank_pct(rng.permutation(len(yv)).astype(float))
        r = rng.random(len(yv))                         # fixed thresholds -> monotone in g
        marg = frame[y]
        lo, hi, best = 0.0, 1.0, frame[y].to_numpy()
        for _ in range(iters):
            g = (lo + hi) / 2.0
            u = np.where(r < g, pole, u_cur)
            remapped = _marginal_map(marg, np.clip(u, 1e-6, 1 - 1e-6), True)
            tmp = frame.copy()
            tmp[y] = remapped
            r2 = covariate_r2(tmp, y, preds, numeric_predictors=num_pred)
            if r2 is None:
                break
            best = remapped
            need_more_pole = (r2 < target) if strengthen else (r2 > target)
            if need_more_pole:
                lo = g
            else:
                hi = g
        frame[y] = best
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_transfer_recalibrate.py -q`
Expected: PASS (7 passed — 3 from Task 2 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/recalibrate.py tests/test_transfer_recalibrate.py
git commit -m "feat(transfer): Step-B bidirectional R^2 blend"
```

---

### Task 4: Firewalled target-aggregate reader

**Files:**
- Create: `src/ssdataagent/transfer/target_aggregates.py`
- Modify: `src/ssdataagent/transfer/copula_stability.py` (add `pairwise_associations`)
- Test: `tests/test_transfer_target_aggregates.py`
- Test: `tests/test_transfer_copula_stability.py` (add one case for the new helper)

**Interfaces:**
- Consumes: `copula_stability.pair_association` (existing), `conditional_variance.covariate_r2` (existing), `generate._is_numeric`.
- Produces:
  - `copula_stability.pairwise_associations(frame: pd.DataFrame, cols: list[str]) -> dict[tuple[str, str], tuple[float, str]]` — every unordered pair (in `cols` order) → `(association, method)` via `pair_association`.
  - `target_aggregates.target_aggregates(pool: pd.DataFrame, cols: list[str], covariates: list[str], outcomes: list[str]) -> dict` — returns `{"pairwise_assoc": {pair: value}, "pairwise_method": {pair: method}, "outcome_r2": {outcome: value|None}, "provenance": {...}}`. Computes associations over all pairs of `cols`, and covariate-R² of each numeric outcome on `covariates`. Reads only `pool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_target_aggregates.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.target_aggregates import target_aggregates


def _pool(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.normal(45, 15, n)
    income = 1000 * age + rng.normal(0, 5000, n)
    gender = np.where(rng.random(n) < 0.5, "m", "f")
    return pd.DataFrame({"age": age, "gender": gender, "income": income})


def test_target_aggregates_shape_and_firewall():
    pool = _pool(2000, 1)
    agg = target_aggregates(pool, ["age", "gender", "income"], ["age", "gender"], ["income"])
    # a value + method for every unordered pair (3 choose 2 = 3)
    assert len(agg["pairwise_assoc"]) == 3
    assert set(agg["pairwise_method"].values()) <= {"kendall", "cramers_v"}
    # income R^2 on age/gender is high (income ~ age)
    assert agg["outcome_r2"]["income"] > 0.5
    # provenance names the pool source, not any test/reference sample
    assert agg["provenance"]["source"] == "target_pool"
    assert "n_rows" in agg["provenance"]


def test_target_aggregates_reads_only_pool():
    # The function signature exposes no test/reference frame — a structural firewall check.
    import inspect
    sig = inspect.signature(target_aggregates)
    assert set(sig.parameters) == {"pool", "cols", "covariates", "outcomes"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_transfer_target_aggregates.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.transfer.target_aggregates'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/ssdataagent/transfer/copula_stability.py
def pairwise_associations(frame: pd.DataFrame, cols: list[str]) -> dict:
    """Every unordered pair (in ``cols`` order) -> (association, method) via pair_association."""
    out = {}
    for v1, v2 in itertools.combinations(cols, 2):
        out[(v1, v2)] = pair_association(frame, v1, v2)
    return out
```

```python
# src/ssdataagent/transfer/target_aggregates.py
from __future__ import annotations

import pandas as pd

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.copula_stability import pairwise_associations
from ssdataagent.transfer.generate import _is_numeric


def target_aggregates(pool: pd.DataFrame, cols: list[str], covariates: list[str],
                      outcomes: list[str]) -> dict:
    """Firewalled low-order aggregates of the TARGET's disjoint pool: per-pair
    associations and per-outcome covariate-R^2. Reads only ``pool`` — never a test or
    reference sample. Provenance-tagged so the firewall is auditable per cell."""
    assoc = pairwise_associations(pool, cols)
    pairwise_assoc = {k: v[0] for k, v in assoc.items()}
    pairwise_method = {k: v[1] for k, v in assoc.items()}
    num_pred = frozenset(c for c in covariates if _is_numeric(pool[c]))
    preds = [c for c in covariates if c in pool.columns]
    outcome_r2 = {
        y: (covariate_r2(pool, y, preds, numeric_predictors=num_pred)
            if y in pool.columns and _is_numeric(pool[y]) else None)
        for y in outcomes
    }
    return {
        "pairwise_assoc": pairwise_assoc,
        "pairwise_method": pairwise_method,
        "outcome_r2": outcome_r2,
        "provenance": {"source": "target_pool", "n_rows": int(len(pool)),
                       "reads": "marginals+pairwise_assoc+covariate_r2"},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_transfer_target_aggregates.py tests/test_transfer_copula_stability.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/target_aggregates.py src/ssdataagent/transfer/copula_stability.py tests/test_transfer_target_aggregates.py
git commit -m "feat(transfer): firewalled target-aggregate reader"
```

---

### Task 5: `transfer_build_b2` — orchestrate Steps A+B

**Files:**
- Modify: `src/ssdataagent/transfer/generate.py` (add `transfer_build_b2`)
- Test: `tests/test_transfer_generate_b2.py`

**Interfaces:**
- Consumes: `gaussian_copula.{fit_latent_correlation, draw_copula, copula_to_frame}` (Task 1); `recalibrate.{recalibrate_matrix, bidirectional_r2_blend}` (Tasks 2–3); `copula_stability.pairwise_associations` (Task 4); `target_aggregates.target_aggregates` (Task 4).
- Produces:
  - `transfer_build_b2(source_pool: pd.DataFrame, target_pool: pd.DataFrame, cols: list[str], covariates: list[str], outcomes: list[str], n: int, seed: int) -> pd.DataFrame` — the B2 generator. Source copula recalibrated to target aggregates, drawn onto target marginals, R²-adjusted per outcome.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer_generate_b2.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.generate import transfer_build, transfer_build_b2


def _ctx(n, seed, xmean, beta):
    rng = np.random.default_rng(seed)
    age = rng.normal(xmean, 1, n)
    edu = np.where(age > xmean, "hi", "lo")
    income = beta * age + rng.normal(0, 0.5, n)
    return pd.DataFrame({"age": age, "education": edu, "income": income})


def test_b2_matches_target_marginals_and_shape():
    a = _ctx(3000, 1, xmean=0.0, beta=2.0)      # source: strong age->income
    b = _ctx(3000, 2, xmean=3.0, beta=0.5)      # target: shifted marginals, weaker mechanism
    out = transfer_build_b2(a, b, ["age", "education", "income"],
                            ["age", "education"], ["income"], n=3000, seed=7)
    assert list(out.columns) == ["age", "education", "income"]
    assert len(out) == 3000
    # T1: target marginal recovered (age mean ~ 3, not source's 0)
    assert abs(pd.to_numeric(out["age"]).mean() - 3.0) < 0.4


def test_b2_recalibrates_outcome_r2_toward_target():
    # source mechanism strong, target weak -> B2 should pull income R^2 DOWN toward target,
    # closer than B1 (which keeps the source's strong mechanism).
    a = _ctx(4000, 1, xmean=0.0, beta=2.5)
    b = _ctx(4000, 2, xmean=0.0, beta=0.4)
    cols, cov, out_y = ["age", "education", "income"], ["age", "education"], ["income"]
    np_ = frozenset({"age", "income"})
    tgt_r2 = covariate_r2(b, "income", ["age", "education"], numeric_predictors=np_)
    b1 = transfer_build(a, b, cols, 4000, 7, "marginal-swap")
    b2 = transfer_build_b2(a, b, cols, cov, out_y, n=4000, seed=7)
    r2_b1 = covariate_r2(b1, "income", ["age", "education"], numeric_predictors=np_)
    r2_b2 = covariate_r2(b2, "income", ["age", "education"], numeric_predictors=np_)
    assert abs(r2_b2 - tgt_r2) < abs(r2_b1 - tgt_r2)     # B2 closer to target than B1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_transfer_generate_b2.py -q`
Expected: FAIL with `ImportError: cannot import name 'transfer_build_b2'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/ssdataagent/transfer/generate.py
def transfer_build_b2(source_pool: pd.DataFrame, target_pool: pd.DataFrame,
                      cols: list[str], covariates: list[str], outcomes: list[str],
                      n: int, seed: int) -> pd.DataFrame:
    """B2 — source copula recalibrated to the target's published aggregates.

    Step A: fit the source latent correlation, edit unstable pairs toward the target
    pool's associations (bidirectional, sign preserved), draw onto target marginals.
    Step B: nudge each numeric outcome's covariate-R^2 onto the target pool's R^2.
    Reads from the target only low-order aggregates of ``target_pool`` (never its joint
    or a test sample)."""
    from ssdataagent.transfer.copula_stability import pairwise_associations
    from ssdataagent.transfer.gaussian_copula import (
        copula_to_frame, draw_copula, fit_latent_correlation,
    )
    from ssdataagent.transfer.recalibrate import (
        bidirectional_r2_blend, recalibrate_matrix,
    )
    from ssdataagent.transfer.target_aggregates import target_aggregates

    R_source, num = fit_latent_correlation(source_pool, cols)
    src_assoc = pairwise_associations(source_pool, cols)
    agg = target_aggregates(target_pool, cols, covariates, outcomes)

    a_src = {k: v[0] for k, v in src_assoc.items()}
    a_tgt = agg["pairwise_assoc"]
    # a pair is comparable only if source and target used the same association method
    methods = {
        k: (src_assoc[k][1] if src_assoc[k][1] == agg["pairwise_method"].get(k)
            else "undefined")
        for k in src_assoc
    }
    R_prime = recalibrate_matrix(R_source, cols, a_src, a_tgt, methods)

    u = draw_copula(R_prime, n, seed)
    rng = np.random.default_rng(seed)
    frame = copula_to_frame(u, target_pool, cols, num, rng)

    num_pred = frozenset(c for c in covariates if num.get(c, False))
    return bidirectional_r2_blend(frame, outcomes, covariates, agg["outcome_r2"],
                                  numeric_predictors=num_pred, rng=rng)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_transfer_generate_b2.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/generate.py tests/test_transfer_generate_b2.py
git commit -m "feat(transfer): transfer_build_b2 orchestrating Steps A+B"
```

---

### Task 6: Widen the scored-pair registry

**Files:**
- Modify: `src/ssdataagent/transfer/pairs.py:43-51` (flip three cps pairs to scored)
- Test: `tests/test_transfer_pairs.py:13-23` (update the registry-shape assertion)

**Interfaces:**
- Consumes: nothing new.
- Produces: `cps_1980_1990`, `cps_1990_2000`, `cps_1970_2000` become `scored=True, target_dataset="cps"`.

- [ ] **Step 1: Update the failing test**

Replace the `scored` assertion block in `tests/test_transfer_pairs.py::test_pairs_registry_shape`:

```python
    scored = {p.id for p in PAIRS if p.scored}
    assert scored == {"gss_1994_2018", "cps_1970_1980", "cps_1980_1990",
                      "cps_1990_2000", "cps_1970_2000"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_transfer_pairs.py::test_pairs_registry_shape -q`
Expected: FAIL (assertion: current scored set has only the two original ids)

- [ ] **Step 3: Flip the pairs to scored**

In `src/ssdataagent/transfer/pairs.py`, edit the `PAIRS` list so these three rows carry `"cps", True, "cps"` (was `"cps", False, None`):

```python
    TransferPair("cps_1980_1990", _cps("cps-asec1980.csv"), _cps("cps-asec1990.csv"), "cps", True, "cps"),
    TransferPair("cps_1970_2000", _cps("cps-asec1970.csv"), _cps("cps-asec2000.csv"), "cps", True, "cps"),
    TransferPair("cps_1990_2000", _cps("cps-asec1990.csv"), _cps("cps-asec2000.csv"), "cps", True, "cps"),
```

Leave `cps_1970_1990` and `cps_1980_2000` as `False, None` (not in the chosen eval set).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_transfer_pairs.py -q`
Expected: PASS (all — note `test_pairs_registry_shape` also asserts `(p.target_dataset is not None) == p.scored`, which now holds)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/pairs.py tests/test_transfer_pairs.py
git commit -m "feat(transfer): mark cps year-ladder pairs scored for B2 eval"
```

---

### Task 7: Wire B2 into the Layer-2 ladder

**Files:**
- Modify: `scripts/transfer_map.py:25` (imports) and `scripts/transfer_map.py:150-155` (configs dict)
- Test: `tests/test_transfer_map.py` (add a B2-wiring case)

**Interfaces:**
- Consumes: `transfer_build_b2` (Task 5), `covariates_outcomes` (existing in `pairs.py`).
- Produces: a `B2_recalibrated` row in each scored pair's Layer-2 output.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_transfer_map.py
def test_run_layer2_configs_include_b2(monkeypatch):
    # run_layer2 builds a dict of named builders; B2 must be one of them and must call
    # transfer_build_b2 with the pair's covariates/outcomes. We probe the builder table
    # by capturing what configs run_layer2 constructs, without scoring.
    import transfer_map as tm
    names = tm.LAYER2_CONFIG_NAMES
    assert "B2_recalibrated" in names
    assert names.index("B1_marginal_swap") < names.index("B2_recalibrated")  # ladder order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_transfer_map.py::test_run_layer2_configs_include_b2 -q`
Expected: FAIL with `AttributeError: module 'transfer_map' has no attribute 'LAYER2_CONFIG_NAMES'`

- [ ] **Step 3: Wire B2 into the ladder**

In `scripts/transfer_map.py`, extend the import at line 25:

```python
from ssdataagent.transfer.generate import transfer_build, transfer_build_b2  # noqa: E402
```

and add to the existing `pairs` import (the `from ssdataagent.transfer.pairs import ...` line):

```python
from ssdataagent.transfer.pairs import covariates_outcomes, load_pair  # noqa: E402
```

Add a module-level constant near the top of the file (after imports), so the ladder order is testable without scoring:

```python
LAYER2_CONFIG_NAMES = [
    "B0_carryover", "B1_marginal_swap", "B2_recalibrated",
    "within_B_floor", "within_B_ceiling",
]
```

In `run_layer2`, compute the X/Y split and add the B2 builder to the `configs` dict (keep insertion order matching `LAYER2_CONFIG_NAMES`):

```python
        covs, outs = covariates_outcomes(pair.schema_name, cols)

        configs = {
            "B0_carryover": lambda s: transfer_build(a, a, cols, n, s, "carryover"),
            "B1_marginal_swap": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
            "B2_recalibrated": lambda s: transfer_build_b2(a, b_pool, cols, covs, outs, n, s),
            "within_B_floor": lambda s: nb.build(b_pool, cols, n, s, "independence"),
            "within_B_ceiling": lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
        }
        assert list(configs.keys()) == LAYER2_CONFIG_NAMES  # single source of truth for order
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_transfer_map.py -q`
Expected: PASS (all)

- [ ] **Step 5: Smoke-run one scored pair at small scale**

Run: `.venv/bin/python scripts/transfer_map.py --pairs cps_1970_1980 --seeds 1 --n 800 --bootstrap-B 30`
Expected: completes; `results/transfer_map/baselines_cps_1970_1980.csv` contains a `B2_recalibrated` row with numeric `T1`/`T2`/`T3`/`overall`.

- [ ] **Step 6: Commit**

```bash
git add scripts/transfer_map.py tests/test_transfer_map.py
git commit -m "feat(transfer): wire B2 into the Layer-2 ladder"
```

---

### Task 8: Run the wide eval + report

**Files:**
- Create: `docs/report/2026-07-22-b2-aggregate-recalibration.md`
- Modify: `docs/experiments/LEDGER.md` (new `b2_recalibration` row)
- Modify: `docs/dashboard/index.html` (regenerated)

**Interfaces:**
- Consumes: `scripts/transfer_map.py` (Tasks 6–7), `scripts/build_dashboard.py` (existing).
- Produces: the ladder table across the five scored pairs and the copula-stability-vs-gap-closing finding.

- [ ] **Step 1: Run the ladder for the fast pairs (foreground)**

Run each cps pair at the publication protocol; capture into `results/transfer_map/`:

```bash
.venv/bin/python scripts/transfer_map.py --pairs cps_1970_1980 cps_1980_1990 cps_1990_2000 cps_1970_2000 --seeds 3 --n 3000 --bootstrap-B 200
```
Expected: four `baselines_<pair>.csv` files, each with all five ladder rows and numeric scores. If a pair is killed for wall-clock, re-run it alone and record the achieved scale.

- [ ] **Step 2: Run the gss pair (may need reduced scale)**

```bash
.venv/bin/python scripts/transfer_map.py --pairs gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```
Expected: `baselines_gss_1994_2018.csv` with five rows. If killed, fall back to `--seeds 1 --n 800 --bootstrap-B 30` and label it preliminary in the report (as Phase 1 did). Record the actual scale used.

- [ ] **Step 3: Write the report**

Create `docs/report/2026-07-22-b2-aggregate-recalibration.md` following the structure of `docs/report/2026-07-22-transfer-map.md`. Populate from the `results/transfer_map/baselines_*.csv` files produced above. It MUST contain, with real numbers pulled from the CSVs:

- a one-line status header linking the roadmap Phase-2 rung and the spec;
- the **ladder table per pair** (B0 / B1 / **B2** / floor / ceiling) with T1/T2/T3/overall;
- the **headline finding**: for each pair, B2's gap-closing measured as `(B2_overall − B1_overall) / (ceiling_overall − B1_overall)`, laid beside that pair's Phase-1 copula-stable fraction (1.00→0.78), stating whether gap-closing tracks stability;
- a per-type note (does B2's lift land on T2/T3, with T1 ≈ B1?);
- the achieved scale per pair (seeds / n / bootstrap_B) and any pair left preliminary;
- an honest-limits section reusing the spec's limits (nominal approximation; copula/R² seam; row-level firewall; aggregate-bounded);
- the decision-gate read: does B2 close most of the residual on most cells → whether Phase 3 is yet justified;
- a Replication block with the exact commands from Steps 1–2.

- [ ] **Step 4: Add the LEDGER row**

Append a row to `docs/experiments/LEDGER.md` matching the existing column schema (open the file, copy the header, fill every column). Set `git_sha` to `_pending_`, and write a meaningful one-line `hypothesis` (e.g. "target-aggregate recalibration of the source copula closes the B1 mechanism residual in proportion to copula stability").

- [ ] **Step 5: Rebuild the dashboard**

```bash
.venv/bin/python scripts/build_dashboard.py
```
Expected: `docs/dashboard/index.html` regenerated with no error.

- [ ] **Step 6: Commit**

```bash
git add docs/report/2026-07-22-b2-aggregate-recalibration.md docs/experiments/LEDGER.md docs/dashboard/index.html results/transfer_map/
git commit -m "report: B2 aggregate-recalibration ladder + copula-stability finding"
```

---

## Notes for the executor

- **Run tests with `.venv/bin/pytest`** (the project's venv). The full transfer suite is `tests/test_transfer_*.py`.
- **Do not modify** `transfer_build` (B0/B1) or `nodonor_bracket.build` — B2 is strictly additive. If a change to either seems necessary, stop and escalate.
- **Scale/compute:** the B=200 five-pair sweep is heavy and gss runs have been killed for wall-clock before. Run pairs individually, record the achieved scale, and label any reduced-scale pair preliminary in the report — do not silently drop a pair.
- **Firewall discipline is the review's top gate:** B2 must never read `target_pool`'s joint beyond the aggregates in `target_aggregates`, and never the reference/test sample. `ref` in `run_layer2` is the scorer's business, never an input to a builder.
