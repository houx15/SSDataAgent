# Empirical-copula transfer — a faithful structure model

**Date:** 2026-07-29
**Branch:** `empirical-copula` (off `main` @ 68f0d1b)
**Motivated by:** a diagnostic (scratch) showing the current generator is badly lossy on the
dependence structure. On `cps_1970_1980`, resampling A's *true* joint scores T2/T3 =
**0.757 / 0.785** vs the current engine's **0.620 / 0.658**, and A's-joint-with-A's-marginals
(reads no B) overall **0.665 already beats B1's 0.646**. The structure — not the marginals —
is our biggest leak.

## Why the current engine leaks

`generate.transfer_build` resamples source rows through a **single shared latent factor**
(`glat` = mean of the numeric columns' percentile ranks). For *numeric* columns the shared-row
resample preserves the joint ranks; but for *categorical* columns, `_latent` orders the
categories along that one factor (`groupby(cat).mean(glat).sort_values()`), so two categoricals'
**co-occurrence is only captured through their shared correlation with one axis**, never their
true pairwise joint. Most outcomes are categorical (marital_status, education, laborforce,
occupation, poverty_status), so this one-factor approximation is where T2/T3 leak.

## The fix

Keep the shared-row resample (which already preserves the numeric joint), and replace the
**categorical** latent with a **joint-preserving** one: take each resampled row's *actual*
category and place `u` inside that category's cumulative-frequency interval (with jitter),
rather than re-deriving it from `glat`. Because the row index (`base`) is shared across all
columns, the copula vector `(u_1,…,u_d)` then comes from **one real source row** → A's empirical
copula is preserved exactly. The marginal is still installed by the existing
`_marginal_map` inverse-CDF, so structure and marginals stay cleanly separated.

## Scope

- **Situation 2 (time transfer)**, both scored pairs: `cps_1970_1980`, `gss_1994_2018`.
  3 seeds, `n=3000`, `bootstrap_B=200` — identical protocol/scorer to B0–B6.
- **Local, no new dependencies** (numpy/pandas only). This is the faithful-copula step; the
  richer parametric/ML models (vine copula, tree-based conditional generation) are the
  **named follow-on (Approach B)**, for the residual and for the cross-country case where
  regularization matters.

## Mechanism

New pure function `empirical_transfer(source, marg, cols, n, seed) -> pd.DataFrame`:

```
base = rng.integers(0, m, n)                      # shared resample -> A's joint (empirical copula)
for c in cols:
    if numeric(source[c]):
        u = source[c].rank(pct)[base]             # row's own rank (joint-preserving, as today)
    else:
        # joint-preserving categorical latent: the resampled row's ACTUAL category interval
        cats, freq = source[c] value_counts (fixed order: frequency desc, ties by label)
        edges = cumulative(freq)                  # [F_{k-1}, F_k) per category
        cat_i = source[c][base]                   # the resampled row's real category
        u = edges[cat_i-1] + rng.random() * (edges[cat_i] - edges[cat_i-1])   # jitter within mass
    em = _marginal_map(marg[c], u, numeric)       # inverse-CDF to the TARGET marginal
    apply marg[c]'s missingness rate              # same as transfer_build
```

`_marginal_map`'s categorical branch already maps `u` to the target category by cumulative
frequency; using the **same frequency ordering** on both sides makes the swap rank-consistent.
The only change from `transfer_build` is the categorical `u` source (actual-category interval
vs `glat`), so numeric behaviour and the marginal/missingness handling are unchanged.

## Configs (the ablation)

| config | copula | marginals | reads B? | role |
|---|---|---|---|---|
| `carryover` (== current engine) | one-factor | A | no | current baseline (~0.556) |
| **`EC_carry`** | **empirical (A joint)** | A | **no** | faithful engine, *feasible* — expect ≈ 0.665, already > B1 |
| **`EC_oracle`** | **empirical (A joint)** | B true | yes | ceiling of the method (A structure + B marginals) |
| `ref_oracle_comp` (== B1) | one-factor | B true | yes | marginal-swap ceiling (~0.646) |
| `ref_floor` / `ref_ceiling` | — | — | — | independence floor / microdata ceiling (~0.805) |

## What it decides

- **`EC_carry` vs `carryover`** — the pure engine fix (both A marginals): how much T2/T3 the
  one-factor approximation was leaking.
- **`EC_carry` vs `B1`** — does the fixed engine with **A's own marginals** (no B data) beat the
  current marginal-swap that *reads* B? The diagnostic says yes; confirming it means a faithful
  copula is worth more than correct marginals here.
- **`EC_oracle` vs `B1`** — the copula improvement holding marginals fixed (both B's true).
- **`EC_oracle` vs `ref_ceiling`** — how close a transferred structure gets to the achievable max.

If `EC_oracle` lands near the ceiling, the recipe is settled: **faithful A structure + imported
target marginals** — and the marginal-import work (public re-indexing / stats, already scoped)
becomes the feasible substitute for the oracle marginals.

## Generalizability guard (per the ML discussion)

The diagnostic already shows A's structure transfers for *time* (its T3 0.785 even exceeds B's
own resample) — so a faithful, un-regularized empirical copula generalizes essentially for free
here, consistent with Q3's 95–100% stability on adjacent waves. For **cross-country / Approach B**,
where structure is less stable (group pairs 69–74%), the model must be selected by
**sibling-transfer validation** (fit on one wave, test the structure on a *held-out* sibling),
not by in-domain fit. v1 scores the two time pairs; the sibling-transfer harness is a named
follow-on for Approach B.

## Architecture

```
   A (source) ──shared-row resample──▶ A's empirical joint (copula)
                                       │  per column: joint-preserving u
   marg (A's own | B's true) ─────────▶ _marginal_map (inverse-CDF)
                                       ▼
              synthetic B ──▶ nb.score T1–T5   (restrict_config_dir, mean_scores)
```

### Components (files)

1. **`src/ssdataagent/transfer/empirical_copula.py`** (new; pure): `empirical_transfer(...)`
   (above); imports `_is_numeric`, `_marginal_map` from `generate`.
2. **`scripts/transfer_empirical.py`** (new): clones the level-correct/face-swap harness
   (`a`/`schema`/`ref`/`b_pool`/`cols`/`types`/`restrict_config_dir`/`mean_scores`,
   `nodonor_bracket as nb`); builds `carryover`, `EC_carry`, `EC_oracle`, `ref_oracle_comp`,
   `ref_floor`, `ref_ceiling`; writes `results/transfer_map/empirical_<pair>.csv`.

## Testing (TDD)

- **Joint preservation:** on constructed data with a strong categorical↔categorical association,
  `empirical_transfer(A, A, …)` reproduces the association (Cramér's V) far closer than
  `transfer_build(A, A, …, "carryover")` — the core claim.
- **Numeric rank-copula preserved:** two correlated numeric columns keep their Spearman
  correlation under an `EC` marginal swap (monotone map).
- **Marginals installed:** `EC_carry` output marginals ≈ A's; `EC_oracle` output marginals ≈ B's
  (per column, categorical proportions / numeric quantiles).
- **Missingness:** each column's NaN rate ≈ `marg`'s rate.
- **Integration smoke (stubbed tiny):** the runner builds all six configs and produces a CSV.

## Limitations (stated in the report)

- Categorical marginal swap assumes a **consistent frequency ordering** across contexts; a
  category present in B but absent in A can't be produced (measurement non-invariance — e.g. the
  race recoding, though demographics aren't scored).
- **Two scored time pairs**; cross-context generalization (and the sibling-transfer guard) is
  Approach B.
- `EC_oracle` reads B's marginals — a labeled ceiling, not a feasible method; the feasible arm is
  `EC_carry` (no B) plus the separate marginal-import track.

## Reproduce

```
.venv/bin/python scripts/transfer_empirical.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_empirical.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```
