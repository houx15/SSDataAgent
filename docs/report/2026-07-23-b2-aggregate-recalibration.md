# B2: recalibrating a borrowed generator to the target's published aggregates

**Date:** 2026-07-23
**Scope:** Phase-2 rung **B2** of `docs/2026-07-22-transfer-roadmap.md`, on the same
same-country **time-transfer** axis as the Phase-1 map (GSS 1994→2018, CPS ASEC 1970→1980).
No LLM, no training.
**Spec:** `docs/superpowers/specs/2026-07-22-b2-aggregate-recalibration-design.md`
(read Amendments 1 and 2 — the original Step-A design was retired mid-flight).
**Replication:** `scripts/transfer_map.py --pairs <pair> --seeds 3 --n 3000 --bootstrap-B 200`

## The question

Phase 1 ended at **B1** (marginal-swap): keep the source context's dependence structure
wholesale, swap in the target's marginals. B1 beat the within-target independence floor but
left a wide residual against the microdata ceiling. B1 leaves that gap because it transplants
the source's *strengths* unchanged.

**B2** asks whether that residual closes by **recalibrating those strengths to the target's
published aggregates** — staying inside the no-donor firewall. It is the direct
generalization of the two tricks that already work within-context (variance repair,
event-order): *source structure, target-aggregate calibration.*

## What B2 ended up being

**B2 = B1's shared-latent marginal-swap draw + per-outcome R²/dispersion recalibration
toward the target pool's covariate-R².**

That is narrower than the spec originally proposed. The spec also called for **per-pair
association recalibration** ("Step A"). Step A was implemented twice, on two different
vehicles, and **dropped both times as a measured negative result** (see *Step A: a negative
result* below). What survives is the R²/dispersion half — which is precisely what the
roadmap's own wording for B2 asks for ("recalibrate θ and dispersion from B's published
aggregates only").

**Firewall.** B2 reads from the target *only* per-column marginals, per-pair associations,
and per-outcome covariate-R², all computed on the target's **disjoint pool** — never the
target's per-person joint, never the reference/test sample. Every statistic is
provenance-tagged. The pool is **row-disjoint, not person-disjoint** (no person key), so the
firewall is row-level, as in Phase 1.

## Result 1 — B2 improves on B1, and the gain is concentrated in T3

Both benchmark-backed pairs, publication protocol (3 seeds, n=3000, bootstrap_B=200):

**cps 1970→1980** (10-year gap)

| config | T1 | T2 | T3 | overall |
|---|---:|---:|---:|---:|
| B0 carry-over | 0.401 | 0.619 | 0.656 | 0.558 |
| B1 marginal-swap | 0.810 | 0.554 | 0.573 | 0.646 |
| **B2 recalibrated** | 0.801 | 0.559 | **0.629** | **0.663** |
| within-target independence floor | 0.856 | 0.378 | 0.004 | 0.413 |
| within-target microdata ceiling | 0.849 | 0.840 | 0.761 | 0.816 |

**gss 1994→2018** (24-year gap)

| config | T1 | T2 | T3 | overall |
|---|---:|---:|---:|---:|
| B0 carry-over | 0.395 | 0.821 | 0.619 | 0.612 |
| B1 marginal-swap | 0.641 | 0.816 | 0.494 | 0.651 |
| **B2 recalibrated** | 0.642 | 0.825 | **0.583** | **0.683** |
| within-target independence floor | 0.800 | 0.720 | 0.003 | 0.508 |
| within-target microdata ceiling | 0.806 | 0.900 | 0.727 | 0.811 |

B2 > B1 on both pairs: **+0.017** (cps) and **+0.033** (gss) overall. Both gaps sit at or
below the ~0.054 overall noise floor individually, so neither alone is a confident win.

The **per-type** picture is where the signal is: the gain is almost entirely **T3**
(**+0.056** cps, **+0.089** gss), the test B2's mechanism actually targets, while T1 and T2
move by ≲0.01. That is the designed behavior showing up in the designed place, on two
different instruments 24 years and one survey program apart.

## Result 2 — the direction is consistent across four source waves

Layer-2 scoring resolves its reference from the *dataset* (see Limits), so three additional
CPS runs are really the same 1980 reference viewed from different **source waves**. Read that
way they form an unplanned but usable ablation (`results/transfer_map/aux_source_wave/`):

| source wave | distance to CPS-1980 reference | B1 overall | B2 overall | Δ overall | Δ T3 |
|---|---|---:|---:|---:|---:|
| CPS 1970 | 10y forward | 0.646 | 0.663 | **+0.017** | +0.056 |
| CPS 1980 | 0y (source == target) | 0.696 | 0.729 | **+0.034** | +0.105 |
| CPS 1990 | 10y backward | 0.627 | 0.683 | **+0.056** | +0.214 |
| GSS 1994 | 24y forward (own reference) | 0.651 | 0.683 | **+0.033** | +0.089 |

**4 of 4 positive on both metrics**, with T3 gains of +0.056 to +0.214. No configuration was
made worse. Individually most overall gaps are sub-noise; the consistency of sign across four
distinct sources and two instruments is the stronger evidence.

Note the 0-year row is a **control**, not transfer: source and target are the same wave, so
B0 == B1 exactly (0.6955 both) by construction. B2 still gains **+0.034** there — meaning the
R² recalibration is doing real work *even at zero transfer distance*. The gain is therefore
about correcting the generator's conditional strength in general, **not** about repairing
transfer-induced drift specifically. That materially narrows what B2 can be claimed to do.

## Step A: a negative result worth reporting

The spec's per-pair association recalibration was built twice and failed twice.

**Vehicle 1 — explicit Gaussian copula.** Fit the source's latent correlation matrix, edit
unstable pairs toward the target's associations, redraw via Cholesky, inverse-CDF onto target
marginals. Result: **B2 scored *below* B1** (0.601 vs 0.646). Diagnosis showed the cause was
the *vehicle*, not the hypothesis — on the ~95% of pairs the edit never touched, mean
association error was **0.067 (B1) vs 0.114 (B2)**, i.e. the fit→redraw round trip leaks
dependence, worst on nominal pairs (`age × marital_status`: target 0.61, B1 0.36, B2 0.17).
Retired in Amendment 1.

**Vehicle 2 — per-column coherence rates on B1's shared-latent draw.** Give each column a
rate `alpha_c ∈ [0,1]`; pairwise association then scales as `alpha_c · alpha_d · a_src`.
Because `alpha == 1` for all columns provably reproduces B1 byte-for-byte, forcing alphas to
1 isolates Step B exactly. Ablation on cps 1970→1980, publication protocol:

| config | T1 | T2 | T3 | overall |
|---|---:|---:|---:|---:|
| B1 (neither step) | 0.810 | 0.554 | 0.573 | 0.646 |
| **B2 with alpha ≡ 1 (Step B only)** | 0.801 | 0.559 | 0.629 | **0.663** |
| B2 with fitted alpha (Step A + B) | 0.777 | 0.570 | 0.596 | 0.648 |

**Step A costs −0.015 overall** (T1 −0.024, T3 −0.033, T2 +0.012). Two measurements explain
why it cannot work on this vehicle:

1. **Too coarse.** 7 of 11 columns fit to exactly `alpha = 1.0`; the rest only reach
   0.92–0.99. The pair is ~95% copula-stable, so any weakening damages the many pairs that
   were already correct in order to chase the few that drifted. A per-column knob cannot
   localize a per-pair correction.
2. **Directionally incapable.** `n_pairs_understrength = 27 of 55` — on half the pairs the
   target wants *more* dependence than the source has, which `alpha ≤ 1` structurally cannot
   supply.

The two vehicles are complementary in exactly the wrong way: the copula offers **per-pair**
control but leaks dependence; the shared latent **doesn't leak** but offers only **per-column**
control. Neither gives both. That is the concrete obstacle any future association-recalibration
attempt has to clear.

## Result 3 — how much of the gain was a bug fix

Honesty requires separating the hypothesis from the housekeeping. A large share of B2's T3
gain came from fixing a **data-semantics bug**, not from the recalibration idea:

`age_first_childbirth` carries the sentinel string `"No Child"`. That pushed the column past
`_is_numeric`'s threshold, so the R² repair **silently skipped it** — only 2 of 8 outcomes
were ever repaired. It took three stacked fixes to correct: the consumer gate, the *producer*
gate in `target_aggregates` (which left the first fix inert), and a dtype widening at the
write-back site (which had been crashing the run outright). With those fixed, T3 on cps
1970→1980 moved 0.515 → 0.629.

This is the same class of trap documented for `child_number`. **A sentinel is not missing
data and not a measurement**; it is a category that must be preserved. Sentinel-bearing
outcomes are now repaired on their coercible subpopulation with sentinel rows returned
byte-identical.

## Decision gate

The roadmap's gate: *"if B1/B2 close most of the gap on most cells → statistics + agent
paper; only if a large mechanism-shift residual survives B2/B3 is learned adaptation (Phase 3)
justified."*

**A large residual survives B2.** Best B2 sits at **0.663 (cps)** and **0.683 (gss)** against
ceilings of **0.816 / 0.811** — roughly **60% of the B1→ceiling gap remains unclosed**, almost
all of it in T2 and T3.

**But the gate is not yet closed, because B3 has not been run.** The gate reads "survives
B2/**B3**", and B3 — the LLM prior pointed at the target context — is unbuilt. If B3 closes
residual that B2 cannot, Phase 3 stays unjustified and this remains a statistics+agent paper.
**B3 is the next slice, and it is what actually decides Phase 3.**

## Limits and honesty

- **The wide five-pair evaluation this work planned was not achievable, and three of its runs
  were invalid.** Layer-2 scoring resolves its target pool and reference from
  `TransferPair.target_dataset` (a dataset *name*), **not** from `target_csv`. SSDataBench
  defines exactly one reference per dataset — `samples_cps.csv` is the **1980** wave. Marking
  extra CPS pairs `scored` therefore re-scored them against the 1980 reference:
  `cps_1970_2000` came out **byte-identical** to `cps_1970_1980` (same source, same
  reference), and `cps_1980_1990` was silently a zero-gap control. This was an error
  introduced by this work and has been reverted; `pairs.py` now documents the constraint and
  the test asserts scored pairs' `target_csv` really is the benchmark wave. Scoring a genuine
  CPS 1970→2000 transfer requires building a new per-wave reference and config, which does
  not exist.
- **Only two genuine transfer pairs**, the same two Phase 1 had. The extra CPS runs are a
  source-wave ablation against a fixed reference, not additional transfer distances, and are
  reported as such.
- **The planned headline test — does B2's gain track copula stability? — could not be run.**
  It needed the time-gap ladder that the point above shows is unavailable.
- **Most individual overall gaps are within the ~0.054 noise floor.** The claim rests on
  consistency of direction across four sources and the concentration of gains in T3, not on
  any single pair clearing noise.
- **The zero-gap control shows the gain is not transfer-specific** — B2 helps as much when
  source and target are the same wave. B2 corrects conditional strength generally.
- **A substantial part of the improvement was a sentinel bug fix**, not the recalibration
  hypothesis (Result 3).
- **Same-country time transfer only.** Country transfer (CPS↔CFPS) and measurement
  non-invariance remain untouched. Nothing here claims attitudes transfer across cultures.
- **Row-level firewall**, not person-level (no person key in the pools).

## Replication

```bash
# the two benchmark-backed transfer pairs, publication protocol
.venv/bin/python scripts/transfer_map.py --pairs cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_map.py --pairs gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```

Outputs: `results/transfer_map/baselines_<pair>.csv` (B0 / B1 / B2 / floor / ceiling);
source-wave ablation in `results/transfer_map/aux_source_wave/`.
Modules: `src/ssdataagent/transfer/{pairs,generate,recalibrate,target_aggregates,
copula_stability,gaussian_copula}.py`; orchestrator `scripts/transfer_map.py`.
`gaussian_copula.py`, `recalibrate_matrix`, and `fit_coherence_alphas` remain in the tree
with tests but are **not** in B2's path (Amendment 2).
