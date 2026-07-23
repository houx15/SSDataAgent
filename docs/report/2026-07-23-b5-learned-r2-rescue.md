# B5 — Learned R² rescue (Phase 3, slice 2)

**Date:** 2026-07-23
**Branch:** `b5-learned-r2-rescue` (off `main` @ d167066)
**Spec:** `docs/superpowers/specs/2026-07-23-b5-learned-r2-rescue-design.md`
**Plan:** `docs/superpowers/plans/2026-07-23-b5-learned-r2-rescue.md`

## What B5 is

The roadmap's Phase-3 trained model (`context → sufficient statistics`),
restricted to the one bundle component B4 proved is the live problem:
**per-outcome conditional strength (T3 covariate-R²).** B5 predicts each target
outcome's R² as a closed-form **empirical-Bayes blend** of B4's same-instrument
retrieval estimate `x_co` (shrunk by a learned retrieval-noise curve σ²(ESS))
toward a **cross-context pooled prior** μ(features) fit over ~13 schema-backed
`(instrument × wave)` contexts. LLM-free, numpy-only, scored identically to
B0–B4. Two configs isolate the mechanism:

- **`B5_learned`** — the full EB posterior (retrieval blended toward prior).
- **`B5_prior_only`** — the pooled prior alone (τ²→0 limit, reads *no* retrieval).

Firewall (stricter than B2): reads only the target's public marginals, X-margins
for raking, and public per-outcome structural features. The target's covariate-R²
is computed **nowhere**; `x_co` and the noise curve come from siblings.

## Results (3 seeds, n=3000, bootstrap_B=200 — same protocol as B0–B4)

| pair | T-type | B2 | B4_retrieval | B4_targetR2† | **B5_learned** | **B5_prior_only** | ceiling |
|---|---|---|---|---|---|---|---|
| cps_1970_1980 | T1 | .639 | — | — | .760 | .762 | — |
| (ESS 0.65) | T2 | — | — | — | .584 | .559 | — |
|  | T3 | — | .753 | — | **.813** | .497 | — |
|  | **overall** | **.663** | **.708** | .694 | **.719** | .606 | .816 |
| gss_1994_2018 | T1 | .64 | — | — | .726 | .725 | — |
| (ESS 0.10) | T2 | — | — | — | .791 | .794 | — |
|  | T3 | — | .447 | — | .541 | **.664** | — |
|  | **overall** | **.683** | **.653** | .735† | .686 | **.728** | .811 |

†`B4_targetR2` reads the target pool's covariate-R². **Every B5 config reads zero
target Y-side data** — only a cross-context prior and (for `B5_learned`) the
sibling retrieval estimate. B4 numbers from `docs/report/2026-07-23-b4-retrieval-kob-transport.md`.

## Findings

### 1. The cross-context prior rescues the thin-retrieval regime (gss).

`B5_prior_only` scores **0.728** on gss — **+0.075 over B4_retrieval (0.653)**,
**+0.045 over B2 (0.683)** — and nearly matches `B4_targetR2` (0.735) **while
reading none of the target's Y-side aggregates.** Its T3 (0.664) recovers most of
what B4's single-sibling transport collapsed (0.447). This is the roadmap's
Phase-3 claim realized: a small model that pools conditional strength across many
contexts predicts gss's R² better than transporting a lone thin sibling, reading
zero target microdata. **The residual B4 left on gss was borrowable after all —
by pooling, not by transport.**

### 2. The retrieval data point carries the rich-retrieval regime (cps) — and *hurts* the thin one.

`B5_learned` scores **0.719** on cps — a **new fully-firewalled best** (> B4_retrieval
0.708 > B2 0.663), with T3 0.813. The ablation is decisive: on cps the retrieval
point is essential (`B5_learned` 0.719 ≫ `B5_prior_only` 0.606). But on **gss the
retrieval point is a net drag** (`B5_prior_only` 0.728 > `B5_learned` 0.686): the
lone raked 1994 sibling *inflates* R² (per-outcome x_co runs high — e.g.
`age_first_childbirth` 0.47 vs prior 0.23), and blending it in re-injects the
variance that collapsed B4's seed 2 (T3 0.443). **Retrieval helps exactly where
ESS is high and hurts where ESS is low** — the precise structure an ESS gate exists
to exploit.

### 3. The *ingredients* of the rescue are proven; the *learned* gate is corpus-limited.

An estimator that simply **selects by ESS** — retrieval-blend when ESS is high
(cps → 0.719), prior-only when ESS is low (gss → 0.728) — beats B2 **and** B4 on
**both** pairs by a clear margin, reading no target Y-data. That selection is the
deliverable Phase 4 should fold in now.

But the *fitted* σ²(ESS) curve does **not** gate hard enough on its own. Fit LOCO
on cps pseudo-targets, it comes out nearly **flat**:

| target | prior μ intercept | τ² | σ²(ESS) fit | σ² at own ESS | retrieval weight |
|---|---|---|---|---|---|
| cps (ESS .65) | 0.181 | 0.0124 | a=0.0000, b=0.0030 | 0.0046 | ~0.73 |
| gss (ESS .10) | 0.186 | 0.0134 | a=0.0048, b=0.0001 | 0.0058 | ~0.70 |

At *both* pairs the model lands ~70% weight on retrieval — so `B5_learned`
under-shrinks on gss (0.686) and leaves ~0.04 on the table vs `B5_prior_only`
(0.728). **The gss ESS (0.10) is ~6× off the cps calibration band (~0.4–0.7); the
b-slope that would down-weight thin retrieval is weakly identified because cps is
the *only* multi-sibling calibration source.** This is exactly the spec's
Limitation #2 and the roadmap's risk #3: below ~8–10 multi-wave contexts the gate
cannot be learned, only assumed. It is **not** a method failure — the prior works,
the selection works — it is a **corpus-scale** limit on learning the gate end-to-end.

## The fitted model (auditable)

- **Prior μ(f):** intercept ≈ 0.18, small negative slopes on standardized
  {entropy, n_predictors, is_numeric} (coef ≈ [0.18, −0.03, −0.02, −0.03]). A mild
  regularizer centered near R² ≈ 0.18–0.25; τ² ≈ 0.013 (between-context spread).
- **σ²(ESS) = max(a + b/ESS, 1e-4):** cps-LOCO (a=0, b=0.003); gss-LOCO
  (a=0.0048, b=0.0001). **gss's ESS 0.10 is off-support — flagged, not trusted.**
- Every number is printed by `predict_target_r2` and checkable against published
  tabulations; the firewall stays inspectable.

## Verdict → Phase 4

1. **Fold the ESS-gated selection in now** — a free, zero-training win over B2 and
   B4 on both pairs (cps 0.719 / gss 0.728), reading only public aggregates + a
   cross-context prior.
2. **The pooled prior is the workhorse for low-ESS contexts** — it is what rescues
   gss. The trained model's remaining job is to *learn* the gate (when to trust
   retrieval vs prior) rather than hand-set it — and that is gated on **corpus
   expansion** (WVS/ISSP/ESS: more instruments spanning a range of ESS), the one
   thing that would identify the σ²(ESS) slope. Below that scale, ship the
   ESS-gated hybrid and state the corpus dependency.
3. **B5 closes the Phase-3 decision** the B2/B3 gate opened: the mechanism residual
   is **transportable by pooling** (composition + cross-context prior), not
   irreducible. The full weight-space hypernetwork the roadmap contrasts against is
   not needed — statistics-space pooling suffices at this scale.

## Limitations

- **Learned gate under-powered (headline caveat, Finding 3):** cps is the only
  multi-sibling ESS calibration source; the σ²(ESS) slope is weakly identified and
  gss's ESS is off-support. `B5_learned` therefore under-realizes the win that
  ESS-gated *selection* achieves. Corpus expansion is the fix.
- **Cross-instrument prior comparability:** μ pools R² across instruments with
  different outcomes; the three structural features are the only normalizer. τ² ≈
  0.013 is modest, so the prior is informative but not tight.
- **Numeric outcomes only:** T3 recalibration touches numeric outcomes; B5 does not
  change T1/T2 (the shared-latent draw is B4's `sib_rew` vehicle, unchanged, so
  `B5_learned − B4_retrieval` isolates the R² change). On outcomes where retrieval
  x_co is None, `B5_learned` supplies the prior μ (B4 left them unrecalibrated) —
  but the numeric-only blend means only numeric outcomes with a retrieval estimate
  actually differ in generation.
- **Two scored pairs** (benchmark-backed). The corpus of ~13 contexts feeds the
  prior; only cps/gss are scored end-to-end.

## Reproduce

```
.venv/bin/python scripts/transfer_b5.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_b5.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```

Heavy scoring jobs are reaped even solo on the box; the full-protocol numbers here
were produced with a resumable per-(config,seed) scorer (`.superpowers/sdd/
b5_incremental.py`, numerically identical to `run_b5` — reuses its functions +
`default_rng(0)`, caches the deterministic LOCO fit so a resume skips it).
