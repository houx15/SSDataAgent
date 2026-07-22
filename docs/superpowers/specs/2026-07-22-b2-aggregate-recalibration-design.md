# B2 — aggregate-recalibration baseline (design)

**Date:** 2026-07-22
**Roadmap slice:** Phase 2, rung B2 of `docs/2026-07-22-transfer-roadmap.md`
("skeleton + aggregate recalibration").
**Builds on:** the Phase-1 transfer map (`docs/report/2026-07-22-transfer-map.md`);
the transfer modules `src/ssdataagent/transfer/{pairs,generate,decompose,copula_stability}.py`;
the existing variance-repair `src/ssdataagent/data/conditional_variance.py`.
**Status:** design, pre-implementation.

## The question B2 answers

The Phase-1 ladder ends at B1 (marginal-swap): it keeps the **source** context's
dependence structure wholesale and swaps in the **target's marginals**. B1 beat the
within-target independence floor, but it left a residual on both T2 (pairwise
association) and T3 (regression R²) — e.g. cps 1970→1980 B1 `T2 0.56 / T3 0.58` vs
the microdata ceiling `0.84 / 0.78`. B1 leaves that gap because it transplants the
source's *strengths* unchanged; Result 2 of the transfer map showed the levels move by
genuine mechanism, not composition.

**B2** keeps the source's dependence *structure* (which edges exist, their signs — the
part Result 1 showed is 78–100% stable) but **recalibrates its strengths** (per-pair
association, per-outcome R²/dispersion) to the **target's published aggregates**. It is
the direct generalization of the two tricks that already work within-context
(variance-repair, event-order): *source/prior structure, target-aggregate calibration.*

B2 is **LLM-free** and **deterministic** (the roadmap's "Phase 2, mostly LLM-free"). The
LLM-driven rung is B3, out of scope here.

## Firewall (what makes B2 no-donor for the target)

B2 reads from the target context, beyond the marginals B1 already uses, **only two
low-order aggregates of the target's disjoint pool**:

- per-pair associations (Kendall τ for numeric/ordinal, Cramér's V for nominal);
- per-outcome covariate-R².

These sit on the **same footing** as the pool covariate-R² the existing variance-repair
"pool" rung already consumes (a higher-order aggregate than a 2-way cross-tab), so B2
stays inside the established no-donor firewall. B2 **never** reads the target's per-person
joint or the target test/reference sample. Every recalibration value carries a provenance
tag (which aggregate, from which pool) so the firewall is auditable per cell, as the
roadmap's Phase-4 sketch requires.

The pool is **row-disjoint, not person-disjoint** (no person key), so the firewall is
row-level — the same caveat the Phase-1 report states for B1.

## Architecture — the two-step Gaussian-copula generator

The Phase-1 `transfer_build` correlates all columns through a single shared latent
(`glat`); that structure cannot express or edit *per-pair* dependence strength. B2 therefore
uses an explicit **Gaussian-copula** construction, run in two calibration steps that own
disjoint tests.

### Step A — copula strengths (owns T2)

1. **Fit `R_source`.** Transform each source column to normal scores (`Φ⁻¹` of the
   column's fractional rank), then take the Pearson correlation among the normal-scored
   columns. This is the standard Gaussian-copula fit; it is exact for numeric/ordinal
   columns. Nominal categoricals (no natural order) are placed on a latent scale by the
   existing category-ordering heuristic from `generate._latent` (order categories by their
   mean shared-latent value); their pairwise strength is measured with Cramér's V, as
   `copula_stability` already does. **Documented approximation** — see Limits.
2. **Recalibrate toward the target.** For each unordered pair `(i, j)` compare the source
   association `a_src(i,j)` to the target-pool association `a_tgt(i,j)`, both on the same
   association scale:
   - **stable** (`|a_tgt − a_src| < 0.10`, the transfer-map threshold) → keep
     `R_source[i,j]`;
   - **unstable** → move the latent entry toward the magnitude implied by `a_tgt`,
     **keeping the source's sign** (bidirectional: the entry can move up or down).
   The mapping from a target association back to a latent Gaussian correlation uses the
   copula's closed form where available (numeric/ordinal: `r = sin(π·τ/2)` for Kendall τ)
   and a monotone calibration otherwise.
3. **Draw & map.** Project the edited matrix `R'` to the nearest valid correlation matrix
   (symmetric eigenvalue-clip: clip negative eigenvalues to a small floor, renormalize the
   diagonal to 1). Draw `z ~ N(0, R')` for `n` rows, map `u = Φ(z)`, and inverse-CDF each
   column onto the **target marginals** (reuse `generate._marginal_map`), including the
   target's per-column missingness rate (as B1 does). Output: T1 = target marginals,
   T2 = recalibrated associations.

### Step B — outcome R²/dispersion (owns T3), bidirectional

Measure `R²_own` per outcome on Step-A's output, then nudge the **observed** covariate-R²
onto the target pool's covariate-R². This is the symmetric generalization of
variance-repair, which today only weakens (`alpha = clip(sqrt(R2_target/R2_own), 0, 1)`):

- target **weaker** than own (`alpha < 1`) → blend the outcome toward an **independent**
  draw (existing behavior);
- target **stronger** than own (`alpha > 1`) → blend the outcome toward its **conditional
  mean** `E[Y | X]` (the coherent pole `conditional_variance` already computes as the
  mean-collapse it repairs *away* from).

Each outcome thus slides on the line **independent ← current → conditional-mean**, with the
R² target choosing the point. It is applied as a **residual observed-scale correction after
the copula step**, so it is small when Step A already landed the outcome's R².

### The copula/R² seam (known care-point)

Step B perturbs the outcome-involving pairs Step A set. The ordering is fixed — **copula
first, R² as a residual nudge** — and the implementation plan pins the exact blend math and
verifies with tests that (a) Step B moves observed R² onto target in both directions and
(b) it does not materially undo Step A's covariate–covariate associations. This seam is
called out here rather than hidden.

## Components

New modules (small, single-responsibility, unit-tested):

- `src/ssdataagent/transfer/gaussian_copula.py` — `fit_latent_correlation(pool, cols)`
  (normal-scores → correlation), `nearest_correlation(R)` (PSD projection),
  `draw_copula(R, n, seed) → uniforms`, and inverse-CDF onto target marginals (delegating
  to `generate._marginal_map`).
- `src/ssdataagent/transfer/recalibrate.py` — `recalibrate_matrix(R_source, a_src, a_tgt,
  threshold) → R'` (Step A edit, sign-preserving, stable/unstable branch) and
  `bidirectional_r2_blend(frame, outcomes, predictors, r2_target) → frame` (Step B).
- `src/ssdataagent/transfer/target_aggregates.py` — `target_aggregates(pool, cols,
  covariates, outcomes) → {pairwise_assoc, outcome_r2, provenance}`; the firewalled reader,
  reusing `copula_stability.pair_association` and `conditional_variance.covariate_r2`.

Modified:

- `src/ssdataagent/transfer/generate.py` — add `transfer_build_b2(source_pool,
  target_pool, cols, covariates, outcomes, n, seed)` orchestrating Steps A+B; reuses
  `_marginal_map`, `_is_numeric`. `transfer_build` (B0/B1) is left untouched.
- `src/ssdataagent/transfer/pairs.py` — flip `cps_1980_1990`, `cps_1990_2000`,
  `cps_1970_2000` to `scored=True, target_dataset="cps"` (they already share the cps
  scoring config). Update `test_pairs_registry_shape` accordingly.
- `scripts/transfer_map.py` — add the **B2** config to the Layer-2 ladder alongside
  B0/B1/floor/ceiling; run the widened scored-pair set; default the scored runs to
  **B=200**.

## Data flow

```
source pool ─► fit R_source (normal-scores rank correlation)
target pool ─► target_aggregates: {pairwise assoc a_tgt, outcome R²_tgt}   [FIREWALL: aggregates only]
target pool ─► target marginals (per column, with missingness rate)        [as B1]

R_source + a_src + a_tgt + threshold  ─► recalibrate_matrix ─► R'  (PSD-projected)
draw z ~ N(0, R') ─► u = Φ(z) ─► inverse-CDF onto target marginals ─► X_b2   (Step A)
X_b2 + R²_tgt ─► bidirectional_r2_blend per outcome ─► X_b2_final            (Step B)
score X_b2_final vs target test sample  (SSDataBench, restricted crosswalk config, B=200)
```

## Evaluation

The ladder (**B0 carry-over, B1 marginal-swap, B2, within-target independence floor,
within-target microdata ceiling**) scored on five pairs spanning time-gap and copula
stability:

| pair | time gap | copula-stable frac (Phase 1) |
|---|---:|---:|
| cps_1980_1990 | 10y | 1.00 |
| cps_1990_2000 | 10y | 1.00 |
| cps_1970_1980 | 10y | 0.95 |
| cps_1970_2000 | 30y | 0.78 |
| gss_1994_2018 | 24y | 0.95 |

Protocol: uniform **B=200**, multi-seed, scored on the crosswalk (transferable) variables
via the existing `restrict_config_dir`. This closes the Phase-1 gss `B=200` loose end (gss
was 1-seed / B=30 preliminary).

**Headline scientific test:** does B2's gap-closing (B2 vs ceiling, relative to B1 vs
ceiling) **track copula stability** — largest where the copula is 1.00-stable, smallest at
0.78? A yes ties the method result back to the Phase-1 map. Secondary: per-type
attribution (does B2's lift land on T2 and T3 as designed, leaving T1 at B1's level?).

**Feeds the roadmap decision gate:** if B2 closes most of the residual on most cells, the
project is a statistics+agent paper and Phase 3 (learned model) is not yet justified; a
large surviving residual is what would justify it.

## Error handling

- Non-PSD `R'` after editing → `nearest_correlation` (eigenvalue-clip) always yields a
  valid draw.
- Missing/sparse target aggregate for a pair (empty cross-tab cell) → keep `R_source` for
  that pair, labeled and logged; never crash.
- Source/target association-method mismatch for a pair (numeric in one, nominal in the
  other) → "undefined", keep source, mirroring the `copula_stability` guard.
- Reuse the Phase-1 `restrict_config_dir` (crosswalk-only scoring, `yaml.safe_dump
  sort_keys=False` for T3 model_type pairing) and `mean_scores` (ignores `T{t}_error`
  string columns).

## Testing

All deterministic (seeded), LLM-free.

- **gaussian_copula:** `fit_latent_correlation` recovers a known correlation on synthetic
  normal data (within tolerance); `nearest_correlation` repairs a hand-crafted non-PSD
  matrix (min eigenvalue ≥ 0, diagonal 1); inverse-CDF output matches the target marginal
  (KS, per column).
- **recalibrate:** a stable pair is kept unchanged; an unstable pair is moved toward the
  target **in both directions** (target stronger → entry up; target weaker → entry down);
  sign is preserved; `bidirectional_r2_blend` drives observed `R²_own` onto the target for
  an outcome that is initially too weak (blend toward conditional mean) and one initially
  too strong (blend toward independent).
- **target_aggregates:** firewall — the reader touches only the pool argument (no test
  frame in scope); provenance tags present for every returned statistic.
- **integration:** synthetic A/B where B differs from A by a **known strength shift**
  (a pair's association and an outcome's R² both changed); assert B2 recovers the target
  associations and R² closer than B1, and B2's ladder overall ≥ B1's on the fixture.

## Limits (written in advance)

- **Nominal categoricals** enter the Gaussian copula through a heuristic latent ordering;
  their strength recalibration is approximate (Cramér's V is unsigned, so "sign" is
  undefined for nominal pairs — those are recalibrated in magnitude only). Numeric/ordinal
  pairs are exact.
- **Same-country time transfer only** — country transfer (CPS↔CFPS) and measurement
  non-invariance remain out of scope, as in Phase 1.
- **Row-level, not person-level firewall** (no person key in the pools).
- **The copula/R² seam** is a residual-correction approximation, not a joint optimization;
  the plan verifies it does not materially undo Step A, but exact simultaneous matching of
  all pairwise associations and all outcome R² is not guaranteed.
- B2 is bounded by what target *aggregates* carry: structure the aggregates cannot see
  (higher-order interactions) is not recovered — the residual that would, if large, justify
  Phase 3.

## Replication (target shape)

```bash
# B2 in the Layer-2 ladder for one scored pair, publication protocol
.venv/bin/python scripts/transfer_map.py --pairs cps_1970_1980 --seeds 3 --bootstrap-B 200

# all five scored pairs
.venv/bin/python scripts/transfer_map.py --pairs \
  cps_1970_1980 cps_1980_1990 cps_1990_2000 cps_1970_2000 gss_1994_2018 \
  --seeds 3 --bootstrap-B 200
```

Outputs extend `results/transfer_map/baselines_<pair>.csv` with a `B2` row per pair.
```
