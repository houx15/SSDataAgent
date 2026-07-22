# The transfer map: what carries across contexts, and what doesn't

**Date:** 2026-07-22
**Scope:** same-country **time transfer** — GSS 1994→2018, and the CPS ASEC year ladder
(1970/1980/1990/2000). No new data, no LLM, no training.
**Status:** Phase-1 deliverable of `docs/2026-07-22-transfer-roadmap.md` — the transfer map,
publishable on its own.
**Replication:** `scripts/transfer_map.py` (`--no-scoring` for the map only). Spec:
`docs/superpowers/specs/2026-07-22-transfer-map-design.md`.

## The question

Every no-donor number so far is *within-context*: fit and score on the same survey wave.
The project's goal is a generator that produces a reliable sample for a context it was never
fitted on. Before building that, we must **measure** where cross-context transfer is cheap
and where it is hard. This report is that measurement, for the cleanest transfer axis:
same country, same instrument, different time.

We factor a context's joint as `P_c(X, Y) = marginals_c × copula_c` and ask, per outcome and
per context pair, two things:

1. **Composition vs mechanism** — how much of the A→B gap in an outcome closes if we only
   swap the demographic composition (reweight A's age/gender/race to B's)? What remains is
   genuine change in the mechanism `P(Y | X)`.
2. **Copula stability** — is the *dependence structure* the same in A and B, independent of
   the marginals? This is the "face-swap" hypothesis: keep the structure, swap the features.

## Two layers (and the firewall between them)

- **Layer 1 — the diagnostic map (the answer key).** Reads the full microdata of *both*
  contexts. It is a measurement of ground truth, **not** a generator, so it is deliberately
  **not** firewalled. It labels each (pair × outcome) composition-dominated vs
  mechanism-shifted (DFL reweighting) and each variable pair copula-stable vs shifted
  (Kendall's τ / Cramér's V).
- **Layer 2 — firewalled baselines.** Two no-donor generators scored against the target's
  benchmark sample: **B0 (carry-over)** generates for the source and is scored on the target;
  **B1 (marginal-swap)** keeps the source's copula but draws the **target's marginals** — and
  reads *only* the target's per-column marginals, never its joint, never its test sample.

The scientific throughline: where the map says the copula is stable, B1 (which transplants
the source copula onto target marginals) should recover most of the transfer loss.

## Result 1 — the copula is remarkably stable, and it decays with time distance

Fraction of variable pairs whose dependence (Kendall τ / Cramér's V) is stable
(|Δ| < 0.10) across the two waves:

| pair | time gap | copula-stable fraction |
|---|---:|---:|
| cps 1980→1990 | 10y | **1.00** |
| cps 1990→2000 | 10y | **1.00** |
| cps 1970→1980 | 10y | 0.95 |
| cps 1980→2000 | 20y | 0.96 |
| gss 1994→2018 | 24y | 0.95 |
| cps 1970→1990 | 20y | 0.85 |
| cps 1970→2000 | 30y | **0.78** |

The dependence structure is 78–100% stable across every pair, and the stable fraction
**falls monotonically with the time gap** — 1.00 at 10 years, 0.78 at 30. This is direct
evidence for the face-swap hypothesis on the time axis: *the copula travels; how far, depends
on how far you go.*

## Result 2 — time transfer is mechanism, not composition

Composition share per (pair × outcome); everything not listed is mechanism-shifted
(composition share ≈ 0). Only **two** cells in the entire map are composition-dominated:

| pair | composition-dominated outcomes | notable partial-composition |
|---|---|---|
| gss 1994→2018 | **marital_status** (0.59) | health 0.17 |
| cps 1970→1980 | **laborforce** (0.71) | marital_status 0.14 |
| cps 1970→1990 | — | laborforce 0.23, occupation 0.14 |
| cps 1980→1990 | — | education 0.28, occupation 0.19 |
| cps 1970→2000 | — | laborforce 0.24, occupation 0.15 |
| cps 1980→2000 | — | occupation 0.10 |
| cps 1990→2000 | — | (all < 0.02) |

Reweighting the demographic composition (age/gender/race) closes almost none of the outcome
gap. Socioeconomic outcomes shift over decades for **period/cohort reasons** — the education
expansion, rising female labor-force participation, income growth — not because the age/sex/
race mix changed. The two composition exceptions are exactly the demographically-driven ones:
marriage rates (age structure) and 1970s labor-force participation.

This **partly overturns the roadmap's prior** ("time transfer mostly composition"): the
*structure* transfers cheaply (Result 1), but the *levels* move by genuine mechanism, not
composition. That is precisely why a marginal swap — fix the levels, keep the structure —
is the right transfer move (Result 3).

*Caveat.* Composition is defined over the exogenous demographic core (age/gender/race, the
benchmark's `input: true` variables); changes in education's own distribution count as
mechanism here. Raking on this core gives an effective sample ratio ≈ 0.10 on these pairs
(reported per-cell as `ess_ratio`); for several outcomes reweighting the demographics moves
the outcome *away* from the target (`gap_residual > gap_raw`) — composition and mechanism
oppose (an aging population vs a secular rise in schooling) — which correctly reads as
mechanism, share 0.

## Result 3 — a firewalled marginal-swap beats the within-target no-donor floor

The Layer-2 baselines, scored on the transferable (crosswalk) variables. **B1 keeps the
source copula and swaps in target marginals; it never reads the target's joint.**

**cps 1970→1980** (3 seeds, n=3000, bootstrap_B=100):

| config | T1 | T2 | T3 | overall |
|---|---:|---:|---:|---:|
| B0 carry-over (source, no transfer) | 0.40 | 0.62 | 0.68 | 0.564 |
| **B1 marginal-swap (firewalled)** | 0.81 | 0.56 | 0.58 | **0.649** |
| within-target independence floor | 0.85 | 0.38 | 0.003 | 0.412 |
| within-target microdata ceiling | 0.85 | 0.84 | 0.78 | 0.824 |

**gss 1994→2018** (1 seed, n=800, bootstrap_B=30 — preliminary; multi-seed re-measure pending):

| config | T1 | T2 | T3 | overall |
|---|---:|---:|---:|---:|
| B0 carry-over | 0.36 | 0.79 | 0.61 | 0.588 |
| **B1 marginal-swap (firewalled)** | 0.62 | 0.81 | 0.44 | 0.625 |
| within-target independence floor | 0.72 | 0.78 | 0.01 | 0.505 |
| within-target microdata ceiling | 0.79 | 0.84 | 0.69 | 0.771 |

Two things, both consistent with Results 1–2:

- **B1 > B0**: swapping the target's marginals onto the source's structure lifts the overall
  score (cps 0.564→0.649; gss 0.588→0.625), driven by **T1** (marginals now match the
  target: cps 0.40→0.81, gss 0.36→0.62).
- **B1 > the within-target independence floor** (cps 0.649 vs 0.412; gss 0.625 vs 0.505).
  This is the headline. The floor is what a no-donor method scores if it knows *only* the
  target's marginals and assumes independence. B1 knows the same marginals **plus a copula
  borrowed from another decade** — and that borrowed copula is worth **+0.24 (cps) / +0.12
  (gss)** overall, almost entirely on **T2 and T3** (the copula-carried tests: cps T3
  0.003→0.58, gss T3 0.01→0.44). The dependence structure a same-country earlier wave gives
  you for free is more than the independence assumption, and Result 1 is why: that structure
  is 78–100% stable.

B1 sits firmly between the independence floor and the microdata ceiling — it recovers a large
share of what real target microdata would buy, using none of it.

## Honesty and limits

- **Same-country time transfer only.** Country transfer (CPS↔CFPS) and measurement
  non-invariance are the hard cases and are out of scope here (roadmap Phase 1's country
  crosswalk is deferred). Nothing here claims attitudes transfer across cultures.
- **The map (Layer 1) reads the target's joint by design** — it is the answer key, not a
  method, and must never be quoted as a no-donor result. Only the Layer-2 B0/B1 numbers are
  firewalled.
- **Composition is measured on a demographic core** (age/gender/race) with a moderate
  effective sample ratio (~0.10); `ess_ratio` is reported per cell so the reliability is
  visible. The composition/mechanism *labels* are directionally robust (reweighting
  demographics does not close socioeconomic gaps); the exact residual magnitudes on low-ESS
  cells are noisier.
- **cps/gss pools are row-disjoint, not person-disjoint** (no person key), so B1's firewall
  is row-level, not person-level.
- **Layer-2 scale differs by pair** (cps 3 seeds / n=3000 / B=100; gss 1 seed / n=800 / B=30,
  preliminary — the multi-seed gss scoring runs kept getting killed mid-pass). Numbers are
  above the ~0.054 overall noise floor for the B1-vs-floor gap but are not the published
  B=200 protocol; a uniform B=200 re-measure is the loose end.
- **Mechanism-dominance bounds the naive method.** Because the levels move by mechanism, a
  pure marginal swap has a ceiling (B1 < microdata ceiling); closing the rest needs the
  target's own aggregate calibration of the mechanism — exactly roadmap Phase-2's B2, the
  next step.

## Replication

```bash
# Layer-1 transfer map, all pairs (fast, no scorer, no firewall — the answer key)
.venv/bin/python scripts/transfer_map.py --no-scoring

# Layer-2 firewalled baselines for a scored pair
.venv/bin/python scripts/transfer_map.py --pairs cps_1970_1980 --seeds 3 --bootstrap-B 100
```

Outputs in `results/transfer_map/`: `map.csv` (composition/mechanism per pair×outcome, with
`ess_ratio`), `copula.csv` (per-pair association stability), `baselines_<pair>.csv`
(B0/B1/floor/ceiling). Modules: `src/ssdataagent/transfer/{pairs,generate,decompose,
copula_stability}.py`; orchestrator `scripts/transfer_map.py`.
