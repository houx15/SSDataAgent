# Public X-margins — grounding demographic composition in public data

**Date:** 2026-07-28
**Branch:** `public-x-margins` (off `main` @ 68f0d1b)
**Motivated by:** the blind face-swap (`2026-07-26-blind-faceswap.md`) — its T1 collapsed
because it *guesses* the demographic X-margins; and this week's characterization + level-correction,
which showed the score gap lives in T1 (marginals), and `B1` (A's copula + B's *full* true
marginals) reaches T1 ≈ 0.81 vs the carry-over baseline's 0.40.

## The idea

The demographic marginals — age, gender, race — are **genuinely public** (census tables exist
for almost any country-year). So instead of carrying A's demographic mix or guessing it from a
description, **admit B's public demographic X-margins** while keeping the dependence structure
(copula) from A and the outcome (Y) marginals from A or the LLM. This attacks T1 exactly where
the blind regime is weak, and needs no B microdata.

## Scope

- **v1 uses B's own public demographic margins** (age/gender/race distributions computed from
  B's public pool) — the same "public aggregate" firewall class as the B0–B6 ladder. External
  curated census tables are a purity/generalization upgrade, deferred.
- **`PUBLIC_X = ("age", "gender", "race")`** — the census-standard demographics, present in both
  scored pairs. gss's other background variables (parent education/occupation) are NOT public
  census margins and are excluded from the public-X set (they stay carried from A).
- **Two scored pairs** (`cps_1970_1980`, `gss_1994_2018`), 3 seeds, `n=3000`, `bootstrap_B=200` —
  identical protocol/scorer to B0–B6 / the face-swap.

## Firewall

- `PX_carry`, `PX_llm` read **only B's public demographic X-margins** (age/gender/race
  distributions) — an allowed public aggregate. The copula and Y-marginals never read B's Y.
- `PX_llm`'s Y-marginals read only B's **description** (LLM), same firewall as the blind face-swap.
- `ref_oracle_comp` (== B1) reads B's full marginals — a labeled ceiling, not a method arm.

## Mechanism

Reuses the existing `transfer_build` (copula from `struct=A`, marginals from `marg`) plus one new
helper. Build the `marg` frame so the **demographic X columns carry B's public distribution** and
everything else carries its base source:

```
with_public_x(base_marg, b_pool, PUBLIC_X) -> marg
   # returns a copy of base_marg with each PUBLIC_X column replaced by a resample of b_pool's
   # column (same length as base_marg; B's marginal incl. its missingness rate); all other
   # columns byte-identical to base_marg.
synthetic_B = transfer_build(struct=A, marg, cols, n, seed, "marginal-swap")
```

- **`PX_carry`** = `with_public_x(a, b_pool, PUBLIC_X)` — base is A (Y = A carry-over), X replaced
  by B's public margins. Isolates the X-margin fix: differs from `FS_carryover` *only* in X.
- **`PX_llm`** = `with_public_x(build_marg_frame(elicited, a, cols), b_pool, PUBLIC_X)` — base is
  the blind LLM marginal frame (Y from the description), X replaced by B's public margins.
  Differs from `FS_llm` *only* in X.

`build_marg_frame` and `elicit_marginals` are the existing blind-face-swap components (unchanged).

## Configs (the ablation)

All scored T1–T5 on both pairs, identical protocol.

| config | copula | X-margins | Y-margins | reads of B |
|---|---|---|---|---|
| `FS_carryover` (ref, == B0) | A | A | A | none |
| **`PX_carry`** | A | **B public** | A | public demographics |
| **`PX_llm`** | A | **B public** | LLM (description) | public demographics + description |
| `FS_llm` (ref, blind) | A | LLM | LLM | description |
| `ref_oracle_comp` (== B1) | A | B true | B true | B's full marginals (ceiling) |
| `ref_floor` / `ref_ceiling` | — | — | — | independence floor / microdata ceiling |

## What it decides

Decompose B1's marginal advantage over carry-over into an **X-part** and a **Y-part**:

- **`PX_carry` − `FS_carryover`** = the value of fixing the demographic X-margins alone (the core
  question). If large, public-X is the lever.
- **`ref_oracle_comp` (B1) − `PX_carry`** = the residual value of also getting Y marginals right.
- **`PX_llm` − `FS_llm`** = the X-margin fix inside the fully-blind regime (does admitting public
  demographics rescue the blind score?).
- **`PX_llm` − `PX_carry`** = whether LLM-elicited Y adds anything over just carrying A's Y, once
  X is fixed.

Per-type reading: the fix targets **T1** (marginals); T2/T3 move only as a second-order effect of
different marginal values feeding the fixed copula (reported as such, consistent with the
marginal-swap T3 behavior already observed).

## Architecture

```
   A (source) ──copula──────────────────────────────────┐
   b_pool ──(age/gender/race public margins)──▶ X cols   │
   A's Y  /  build_marg_frame(elicited) ───────▶ Y cols   ├─ with_public_x ─▶ marg
                                                          │
   transfer_build(struct=A, marg, cols, n, seed, "marginal-swap") ─▶ synthetic B ─▶ score T1–T5
```

### Components (files)

1. **`src/ssdataagent/transfer/publicx.py`** (new; pure):
   - `PUBLIC_X = ("age", "gender", "race")`.
   - `with_public_x(base_marg: pd.DataFrame, b_pool: pd.DataFrame, x_cols, *, seed=0) -> pd.DataFrame`
     — copy of `base_marg` with each `x_cols` column present in `b_pool` replaced by a length-preserving
     resample of `b_pool`'s column (carries B's marginal incl. missingness); other columns untouched.
2. **`scripts/transfer_publicx.py`** (new): clones `scripts/transfer_faceswap.py`'s setup
   (`a`/`schema`/`ref`/`b_pool`/`cols`/`covs,outs`/`types`/`restrict_config_dir`/`mean_scores`,
   and `elicit_marginals`/`build_marg_frame`). Builds the seven configs above (`x_cols =
   [c for c in PUBLIC_X if c in covs]`), scores, writes `results/transfer_map/publicx_<pair>.csv`.
   LLM-free at score time (elicitation cached under `results/blind_cache/`).

## Testing (TDD)

- `with_public_x`: the X columns' value distribution matches `b_pool`'s (not the base's); every
  non-X column is byte-identical to `base_marg`; output length == `len(base_marg)`; a column absent
  from `b_pool` is left unchanged.
- `with_public_x` preserves B's missingness in X: an X column that is e.g. 20% NaN in `b_pool`
  comes out ~20% NaN.
- Comparability smoke: `with_public_x(a, b_pool, [])` (empty x_cols) returns `a` unchanged, so
  `PX_carry` with no public columns reduces to `FS_carryover`.
- Integration (stubbed LLM): the runner builds all seven configs and `PX_carry`/`PX_llm` differ
  from `FS_carryover`/`FS_llm` only in the `PUBLIC_X` columns of the marginal frame.

## Limitations (stated in the report)

- **B's own public margins**, not externally-sourced census tables — v1 tests the mechanism; the
  external-census version is the honest-public upgrade.
- **age/gender/race only** — the three census-standard demographics; other background variables
  stay carried from A.
- **Two scored pairs**; same-instrument time transfer only.
- Marginal-swapping carries a small T3 cost (already observed this week); reported alongside the
  T1 gain, not hidden.

## Reproduce

```
export OPENROUTER_API_KEY=...        # first run only; Y elicitation is cached (shared blind cache)
.venv/bin/python scripts/transfer_publicx.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_publicx.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```
