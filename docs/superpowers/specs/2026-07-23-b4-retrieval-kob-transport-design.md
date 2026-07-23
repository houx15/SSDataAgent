# B4 — Retrieval + KOB transport (Phase 3, slice 1) — design

**Date:** 2026-07-23
**Status:** design, pending user review
**Roadmap:** `docs/2026-07-22-transfer-roadmap.md` Phase 3 (learned statistics
generator), zero-training slice. Predecessors: B0–B3 (Phase 2 baseline ladder).

## Goal

One sentence: predict the target context's **joint Y-structure** (pairwise
associations *and* per-outcome conditional strength) from **sibling contexts we
hold microdata for**, reading none of the target's Y-side joint aggregates — and
use it to answer whether the residual B2 leaves is *composition* (closeable by
reweighting) or *genuine mechanism shift* (needs the full learned model).

This is the roadmap's mandated zero-training ablation ("retrieval of
nearest-context statistics with a KOB-style adjustment"). It must be built before
the learned model (Phase 3, slice 2) can claim to beat anything.

## Why now / what it decides

B2 leaves ~60% of the B1→ceiling gap unclosed (cps 0.663 vs ceiling 0.816;
gss 0.683 vs 0.811), and B3 (LLM prior pointed at target) is worse than B1/B2 on
both pairs. The roadmap's decision gate is therefore open: a large residual
survives B2 AND B3. But the gate does not say *what* the residual is. This slice
performs a **Kitagawa–Oaxaca–Blinder decomposition of the B2 residual**:

- If reweighting sibling microdata to the target's public X-marginals closes part
  of the residual → that part was **composition contamination**, transportable
  with zero target Y-data → transfer is easier than B2 implied.
- If it does not → the residual is **genuine mechanism shift** ("same X, different
  Y") → only borrowing a nearer context's mechanism can help → the full learned
  model (slice 2) is justified, and this slice built its evaluation harness.

Either outcome is a publishable finding and directly sizes slice 2.

## The estimand

For a target context `c`, the parts of the statistics bundle that B2 does NOT get
from the target's public univariate marginals:

1. **Pairwise associations** (drives T2) — carried, as in B1/B2, by the
   shared-latent draw's structure source. B2's Step-A experiments proved you
   cannot *inject* a target-specific association matrix into a fixed generator
   (Gaussian-copula fit→redraw leaks dependence; per-column coherence rates are
   directionally incapable). So we do not inject — we **change which microdata the
   shared-latent draw reads its structure from**.
2. **Per-outcome conditional strength** (drives T3) — the covariate-R² target that
   B2's Step B recalibrates toward. B2 reads this from the *target* pool (a Y-side
   conditional aggregate). B4 instead **transports** it from siblings.

Marginals (T1) are unchanged: B4 reads the target's public univariate marginals
for X and Y exactly as B0–B3 do. The firewall forbids the target's *joint* — its
pairwise associations and its covariate-R² — not its toplines.

## Mechanism

### 1. Reweighted-pooled sibling pseudo-population (`sib_rew`)

- **Retrieval set:** same-instrument siblings we hold microdata for, under
  **leave-one-context-out**: hold out the target wave, take all others.
  - cps target 1980 ← {1970, 1990, 2000}
  - gss target 2018 ← {1994}  (single sibling — see Limitations)
- **KOB transport = raking.** Concatenate the sibling frames; compute per-row
  raking weights (`ssdataagent.transfer.decompose.raking_weights`) so the stack's
  weighted marginals on `CORE_DEMOGRAPHICS` (age, gender, race — the benchmark's
  `input: true` demographic core) match the **target pool's** margins. Raking
  simultaneously (a) corrects each sibling's composition toward the target and
  (b) pools siblings, letting a composition-nearer sibling contribute more.
  This reads only target **X-margins** — public.
- **Materialize:** draw a weighted resample of the stack into a fixed-size
  pseudo-population `sib_rew` (size = the stack size, or a fixed 20k, whichever is
  set in the plan) using a seeded RNG. `sib_rew` is the single reweighted material
  from which BOTH the structure and the transported R² are read, so they stay
  mutually consistent.
- **Reliability signal:** record `effective_sample_size(weights)/n` (Kish ESS
  ratio). Low ESS = the raking concentrated weight on a few rows = the transport
  is thin; carry it into the report, never silently.

### 2. Generate through the existing B2 machinery, re-sourced

Run the B1/B2 shared-latent construction (`generate.transfer_build` /
`transfer_build_b2`) with the source swapped to `sib_rew`:

- **Structure source (`struct`) = `sib_rew`** (was: the pair's designated source
  `a`). This is how the transported T2 lands — through the *same* faithful draw
  that already reproduces its source's associations (B1 assoc error 0.067). No
  copula surgery, so Step A's failure mode is never re-entered.
- **Marginals (`marg`) = target pool** (X and Y univariate — public), exactly as
  B1/B2.
- **Step-B R² target** (the covariate-R² fed to `bidirectional_r2_blend`):
  - **B4_retrieval (headline, fully firewalled):** `covariate_r2(sib_rew, …)` —
    transported from siblings. Reads NO target Y-aggregate.
  - **B4_retrieval_targetR2 (diagnostic):** `covariate_r2(target_pool, …)` — B2's
    own R² source. Isolates the reweighting/retrieval effect while holding the R²
    source fixed at B2's.

### 3. The two-config decomposition

| Config | Structure source | R² target | Firewall vs B2 |
|---|---|---|---|
| B2_recalibrated (existing) | designated source `a` | target pool | baseline |
| B4_retrieval_targetR2 (diag) | reweighted sibling pool | target pool | same |
| B4_retrieval (headline) | reweighted sibling pool | transported (sibling) | **stricter** |

- **B4_retrieval_targetR2 − B2** = the pure retrieval+reweighting effect (does
  raking a LOCO sibling pool to target margins beat the single designated source,
  holding R² fixed?).
- **B4_retrieval − B4_retrieval_targetR2** = the cost of giving up the target R²
  aggregate (replacing it with a transported one) — the price of the tighter
  firewall.
- **B4_retrieval − B2** = the headline: can a strictly-more-firewalled estimator
  match the aggregate-reading B2?

## Firewall (stated for day one)

- **B4 reads:** target public univariate marginals (X and Y), target X-margins for
  raking. All public toplines/census-style aggregates.
- **B4 never reads:** the target's per-person joint, its pairwise associations, its
  covariate-R², or the benchmark reference sample.
- **Siblings contribute microdata** (allowed — they are training contexts, held out
  under LOCO), used only to compute their raked structure and R².
- Provenance: every transported statistic is computed from `sib_rew` and is
  printable/checkable against any published sibling tabulation — the firewall stays
  auditable, per the roadmap's Phase-3 principle.

## Scoring / comparability

Scored identically to B0–B3: same crosswalk `cols`, same `restrict_config_dir`,
same reference (`load_schema(target_dataset).real_data_path`), same seed offset
(`1000+s`), same `bootstrap_B`. B4 configs slot into the Layer-2 ladder beside
B0–B3 and are reported in the same CSV/LEDGER/dashboard.

## Components / files (for the plan to detail)

- **New:** `src/ssdataagent/transfer/retrieval.py` — `sibling_contexts(pair)` (LOCO
  same-instrument set), `reweighted_siblings(...)` → (`sib_rew`, ess_ratio),
  reusing `raking_weights` / `effective_sample_size`.
- **New:** `scripts/transfer_b4.py` — orchestrator mirroring `transfer_b3.py`:
  build `sib_rew`, run the two B4 configs through the B2 machinery, score, write
  `results/transfer_map/b4_<pair>.csv`.
- **Reused verbatim:** `generate.transfer_build_b2` (structure + Step-B blend),
  `target_aggregates`, `covariate_r2`, `scoring.{restrict_config_dir,mean_scores}`,
  `nodonor_bracket.{carve_pool, score, TYPES}`.
- **Possibly touched:** `transfer_build_b2` currently hardcodes reading the R²
  target from its `target_pool` argument via `target_aggregates`. B4 needs the R²
  target to come from `sib_rew` (headline) or `target_pool` (diagnostic) while the
  marginals always come from `target_pool`. The plan will decide between (a)
  passing an explicit `r2_source` frame, or (b) a thin B4-local generate function
  that calls `bidirectional_r2_blend` directly. Preference: minimal, reuse-heavy —
  likely (a), an optional parameter defaulting to today's behavior so B2 is
  untouched bit-for-bit.

## Limitations (on the record now)

1. **gss retrieval is degenerate.** Only one same-instrument sibling (1994 = the
   designated source), so for gss, `sib_rew` = reweighted 1994 and the slice tests
   ONLY the reweighting/KOB effect, not retrieval. cps (3 siblings) tests both.
   State plainly in the report.
2. **LOCO uses future waves** (1990/2000 to hit 1980). Temporally odd but
   firewall-clean and standard LOCO; it is the exact protocol slice 2 needs.
3. **Small N.** No nearest-neighbor distance metric (underpowered at N=3/N=1);
   uniform raking-pool instead. Per-sibling transported bundles are computed and
   their **spread reported as a diagnostic** — a large spread is itself the finding
   ("mechanism drifts across waves → learned model needed") and never enters the
   headline estimator.
4. **Same-instrument only.** No cross-instrument (cps↔gss) retrieval — different
   codebooks/wording make it indefensible for one fragile data point.

## Success criteria

- The two B4 configs run and score on both scored pairs on the same footing as
  B0–B3, deterministically off the microdata (no API key — B4 is LLM-free).
- The decomposition table (B2 / B4_targetR2 / B4_retrieval, per T-type) is produced
  with ESS reliability and per-sibling spread.
- A clear verdict on the estimand: is the B2 residual composition-transportable or
  mechanism-shifted? This sizes Phase 3 slice 2.

## Non-goals

- No training, no learned model (that is slice 2, gated on this slice's verdict).
- No dispersion or event-order transport (T4/T5) — deferred; the scored pairs are
  T1/T2/T3.
- No new datasets / corpus expansion.
