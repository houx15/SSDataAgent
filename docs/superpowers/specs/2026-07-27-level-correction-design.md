# Level-correction channel — location-shifted marginals for cross-context transfer

**Date:** 2026-07-27
**Branch:** `level-correction` (off `main` @ 68f0d1b)
**Motivated by:** the characterization study (`2026-07-27-transfer-characterization-study-design.md`):
Y-heterogeneity is ~90% mechanism, the copula is 78–100% stable, and mechanism moves are
~half **level-shift** (Q4). This builds the cheap, copula-preserving half.

## Why this exists

The blind face-swap failed on T1 because eliciting a full Y-marginal is hard. But the
characterization says the between-context heterogeneity lives largely in a **location**
shift of Y|X — a **1-parameter-per-outcome** target, far easier to estimate from thin
signal than a whole distribution. The hypothesis: **keep A's marginal shape (and copula),
correct only the location for B.** If a *location-only* estimate is supplyable where the
full marginal wasn't, that is a real win for the description-only regime.

## Regime & firewall

This spans both regimes as ablation arms, all scored on the same axis as B0–B6:
- **Blind arm** (`LC_llm`) reads only B's audited textual description (same firewall as the
  face-swap / `blind_specs.py`). Eliciting an *average* value is closer to public knowledge
  than a sample statistic; the audited-description discipline is unchanged.
- **Public-margin arms** (`LC_pooled`, `LC_hybrid`) read only B's **public X-margins**
  (raking siblings), never B's Y — the B4/retrieval firewall.
- **Oracle arm** (`LC_oracle`) reads B's true Y (labeled ceiling, like `ref_oracle_comp`).

## Scope

- **Numeric outcomes only** (location shift = additive, matching Q4's mean-gap definition).
  Ordinal/nominal "level" is a separate design (named follow-on).
- **Two scored pairs**: `cps_1970_1980`, `gss_1994_2018`. Same protocol as B0–B6 /
  face-swap: 3 seeds, `n=3000`, `bootstrap_B=200`.
- **Full ladder**: `LC_none`, `LC_oracle`, `LC_llm`, `LC_pooled`, `LC_hybrid` + references.

## Mechanism — location-corrected marginals

For each numeric outcome `y`, generate B with A's copula (`struct=a`) and **A's marginal
shape shifted by a per-outcome location** `Δ_y`:

```
marg_shifted = a.copy();  marg_shifted[y] = to_numeric(a[y]) + Δ_y   # numeric outcomes only
synthetic_B  = transfer_build(struct=a, marg=marg_shifted, cols, n, seed, "marginal-swap")
```

`transfer_build` already takes the pattern/copula from `struct` and the inverse-CDF value
map + missingness RATE from `marg` (`generate._marginal_map`). Shifting a `marg` column by a
constant shifts the generated column by that constant, preserving variance/skew/copula and
A's missingness rate. **Covariates and non-numeric outcomes keep A's marginals** (carryover),
so `LC_*` differs from `LC_none`/B0 **only** in the numeric-Y locations — a clean isolation
of the level-correction. `Δ_y = L̂_y − mean_A(y)`, where `mean_A` is the NaN-aware mean and
`L̂_y` is the arm's estimate of B's level (the location statistic is `mean`, matching Q4;
`median` is a parametrized robustness option).

## The arms

| config | `L̂_y` (B's level estimate) | reads of B | reuses |
|---|---|---|---|
| `LC_none` (==B0 carryover) | Δ_y = 0 (no shift) | none | — |
| `LC_oracle` | `mean_B(y)` on `b_pool` | B's Y (ceiling) | — |
| `LC_llm` | LLM elicits B's mean per numeric Y from B's **description** | description only | `blind` elicit+cache pattern; model `anthropic/claude-sonnet-4.5` |
| `LC_pooled` | `mean(sib_rew[y])`, LOCO siblings raked to B's public X-margins | public X-margins | `retrieval.sibling_csvs` + `reweighted_pool` |
| `LC_hybrid` | **ESS-gated fuse** of `LC_pooled` and `LC_llm`, per outcome | public X-margins | `rescue.select_r2_source(n_siblings, ess)` |

**References** (reuse the face-swap harness, byte-identical comparability):
`ref_floor` / `ref_ceiling` (bracket), and **`ref_oracle_comp` (==B1)** — A's copula + B's
**full** marginal (shape+location). `LC_none==B0` and `ref_oracle_comp==B1` reproduce ladder
rows exactly, keeping everything on one axis.

**`LC_hybrid` gate.** Per outcome, `select_r2_source(n_siblings, ess)` (already in `rescue.py`,
already tested: `n_siblings>=2 and ess>=τ`, τ=0.3) chooses the `LC_pooled` shift when siblings
are rich and effectively-sized, else falls back to the `LC_llm` shift. This is the B6
construction applied to level, and it embodies the project's "combine algorithms + LLM":
retrieval carries the rich-sibling case (cps, 3 siblings), the LLM description carries the
thin case (gss, 1 sibling). A continuous precision-weighted blend is a named extension; v1
uses the tested ESS-gated selection. (Caveat, per B6: with 2 pairs the gate boundary is
unidentified — reported honestly, not oversold.)

## Architecture

```
                         a (source A microdata)
                               │  copula/shape
  ┌──────────── L̂_y estimate per numeric outcome ────────────┐
  │  oracle: mean(b_pool[y])        (reads B — ceiling)        │
  │  llm:    elicit mean | description   (blind)               │
  │  pooled: mean(reweighted_pool(siblings → B public Xmargins))│
  │  hybrid: select_r2_source(n_sib, ess) ? pooled : llm       │
  └───────────────────────────┬───────────────────────────────┘
             Δ_y = L̂_y − mean_A(y)   │
                                     ▼
       apply_level_shift(a[cols], {y: Δ_y})  →  marg_shifted
                                     ▼
   transfer_build(a, marg_shifted, cols, n, seed, "marginal-swap") → synthetic B
                                     ▼
                 nb.score(...) T1–T5  (restrict_config_dir, mean_scores)
```

### Components (files)

1. **`src/ssdataagent/transfer/levelcorrect.py`** (new; pure + one cached LLM call):
   - `numeric_outcomes(a, b, outs) -> list[str]` — outcomes numeric in BOTH, via the
     **scorer's** numeric test (`nodonor_bracket._is_numeric`; `decompose._is_num` mirrors it)
     so the shifted set matches what the scorer grades as numeric.
   - `outcome_mean(frame, y) -> float` — NaN-aware mean of a numeric column.
   - `oracle_shifts(a, b, ys) -> dict[str, float]` — `mean_b − mean_a`.
   - `pooled_shifts(a, sib_rew, ys) -> dict[str, float]` — `mean(sib_rew) − mean_a`.
   - `llm_shifts(a, ds, ys, *, client=None, cache_dir=None, regenerate=False) -> dict[str,float]`
     — elicit B's mean per numeric Y from `BLIND_SPECS[ds]` description, cached to
     `results/levelcorrect_cache/<ds>_levels.json`; `Δ = llm_mean − mean_a`. Reads no B data.
   - `hybrid_shifts(pooled, llm, n_siblings, ess) -> dict[str, float]` — per outcome,
     `select_r2_source(n_siblings, ess)` picks `pooled[y]` else `llm[y]`.
   - `apply_level_shift(marg: pd.DataFrame, shifts: dict[str,float]) -> pd.DataFrame` — pure;
     returns a copy with each numeric `shifts` column shifted (NaN preserved), others untouched.
2. **`scripts/transfer_levelcorrect.py`** (new): clones `scripts/transfer_faceswap.py`
   (`a`/`schema`/`ref`/`b_pool`/`cols`/`covs,outs`/`types`/`restrict_config_dir`/`mean_scores`).
   Computes `numeric_ys`, the four shift dicts (oracle/llm/pooled/hybrid), builds the five
   `transfer_build`-based configs + three references, scores, writes
   `results/transfer_map/levelcorrect_<pair>.csv`. LLM-free at score time (cache).
3. **LLM location prompt** in `levelcorrect.py` (or a thin helper reused from `blind`): asks,
   for each numeric Y, a single expected **average** value for the described population; JSON
   `{var: number}`; parsed defensively (junk → outcome dropped → that outcome's Δ=0, i.e.
   carryover for that Y). Same brace-balanced parse discipline as `blind._last_json_object`.

## Evaluation — the three hypotheses

- **H1 (ceiling):** `LC_oracle` overall vs `LC_none`. A positive gap = pure location-correction
  recovers real score (validates Q4's "level is a big, cheap half").
- **H2 (decomposition):** `LC_oracle` vs `ref_oracle_comp` (B1). The remaining gap is the
  **shape** contribution to the full-marginal benefit — how much is *not* location.
- **H3 (feasibility — the crux):** do `LC_llm` / `LC_pooled` / `LC_hybrid` approach
  `LC_oracle`? A yes means the location shift is **supplyable** (blind and/or from public
  margins) even though the full marginal was not — the contribution over the face-swap.
  Per-type reading: the shift targets T1 (marginals); T2/T3 should move only second-order
  (copula unchanged), reported as such.

## Testing (TDD)

- `apply_level_shift`: shifting a column by Δ moves its mean by exactly Δ, leaves variance and
  NaN positions unchanged, and leaves non-listed columns byte-identical.
- `oracle_shifts` / `pooled_shifts`: on constructed frames, Δ equals the mean difference;
  numeric-only (a categorical outcome is skipped).
- `numeric_outcomes`: returns outcomes numeric in both frames, excludes a categorical one.
- `hybrid_shifts`: with `n_siblings>=2, ess>=0.3` picks pooled per outcome; with `n_siblings=1`
  (or low ess) picks llm. (Gate reuses already-tested `select_r2_source`.)
- `llm_shifts`: monkeypatched client returning a JSON level map yields `Δ = llm − mean_a`;
  malformed entries drop to Δ=0 (carryover) rather than raise.
- Integration (comparability): `LC_none` config output scores byte-identically to a direct
  `transfer_build(a, a, cols, n, s, "carryover")` (== B0) for a fixed seed on a tiny `n`.

## Limitations (stated in the report)

- **Additive location** may under-serve income (multiplicative/log inflation); flagged, an
  affine shift+scale is a named extension. (Scale is "shape", deliberately out of a *level*
  channel.)
- **2 scored pairs**; gss has **1 LOCO sibling**, so `LC_pooled`/`LC_hybrid` are ESS-thin on
  gss (inherits B4/B5's corpus limit). The `LC_llm` arm is sibling-free and still applies.
- **Numeric Y only**; ordinal/nominal level is a separate design.
- **Hybrid gate** boundary is unidentified from 2 pairs (per B6); reported, not tuned.
- Location statistic is the **mean** (Q4-consistent); heavy-tailed outcomes may prefer the
  median — parametrized, not swept in v1.

## Reproduce

```
export OPENROUTER_API_KEY=...        # first run only; level elicitation is cached durably
.venv/bin/python scripts/transfer_levelcorrect.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_levelcorrect.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```
