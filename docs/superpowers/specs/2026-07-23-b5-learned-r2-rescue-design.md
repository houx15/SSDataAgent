# B5 — Learned R² rescue (Phase 3, slice 2) — design

**Date:** 2026-07-23
**Status:** design, pending user review
**Roadmap:** `docs/2026-07-22-transfer-roadmap.md` Phase 3 (learned statistics
generator), the trained-model slice. Predecessor: B4 (Phase 3 slice 1, the
zero-training retrieval + KOB ablation). Baseline ladder: B0–B4.

## Goal

One sentence: a small, fitted model that predicts the target context's
**per-outcome conditional strength** (T3 covariate-R²) by borrowing strength
across *all* available contexts, so the estimate stays reliable **where B4's
same-instrument retrieval is too thin to transport it** — reading none of the
target's Y-side joint aggregates.

This is the roadmap's Phase-3 trained model (`context → statistics bundle`),
restricted to the single bundle component B4 proved is the live problem:
conditional strength. It must beat B2 **and** B4 to earn its complexity
(roadmap #8).

## Thesis / what it decides

B4 delivered a sharp, asymmetric result:

- **cps** (3 same-instrument siblings, ESS 0.65): transported R² already lands
  T3 at 0.753 against a 0.761 ceiling — near-ceiling, nothing left to add.
- **gss** (1 sibling, ESS 0.10): raking the lone 1994 sibling to 2018's margins
  over-concentrates weight and the transported R² **collapses** T3 0.687→0.447,
  dragging B4_retrieval (0.653) under B2 (0.683).

So the residual is *not* mechanism-invariance failure — for gss the mechanism is
there to borrow (same instrument, same outcome semantics, one wave apart); the
failure is **data scarcity in the retrieval step**. B5 asks the precise question
that leaves open:

> Can a model that pools conditional-strength information across *all* contexts
> rescue the thin-retrieval regime (gss) **without disturbing** the rich one
> (cps)?

- If yes (B5 > B2 and B5 > B4 on gss, B5 ≈ B4 on cps) → learned adaptation in
  statistics space earns its place; Phase 4 integration is justified.
- If no (B5 ≈ B4 at this corpus size) → the binding constraint is **corpus
  scale, not method** (roadmap risk #3); the deliverable becomes the fitted
  ESS→noise curve plus an evidence-backed "expand the corpus before training"
  verdict.

Either outcome is publishable and directly sizes Phase 4. This is the ablation
the roadmap mandates before any trained model can claim to beat B2 or retrieval.

## The estimand

For a target context `c` and each **numeric** outcome `o` (the ones
`bidirectional_r2_blend` recalibrates; categoricals carry no R² target), predict

```
R²(c, o) = covariate-R² of outcome o given the crosswalk covariates, in context c
```

reading only public marginals of `c` (never `c`'s per-person joint, its
covariate-R², or a reference sample). B5 changes *only* the R² target fed to the
existing Step-B recalibration. Marginals (T1) and the shared-latent structure
draw (T2) are untouched — identical to B1/B2/B4. The firewall forbids the
target's *joint*, not its public univariate toplines.

## Architecture — empirical-Bayes shrinkage

For each `(c, o)`, form the R² estimate as a precision-weighted blend of a
retrieval data point and a cross-context prior:

```
                 x_co / σ²_co   +   μ(f_o) / τ²
   R²̂(c, o)  =  ─────────────────────────────────
                    1 / σ²_co    +    1 / τ²
```

Three fitted pieces:

1. **x_co — the retrieval data point.** B4's transported R²: the raked
   same-instrument sibling pool's `covariate_r2` for outcome `o`. Semantically
   matched to the target, but high-variance when retrieval is thin.

2. **μ(f_o) — the cross-context pooled prior.** A small GLS regression of true
   R² on **minimal, firewall-clean structural features** `f_o` of the outcome,
   fit across every training `(context, outcome)` row. Feature set (2–4,
   deliberately minimal to avoid overfitting at ~13 contexts):
   - outcome marginal spread (variance for numeric / normalized entropy),
   - number of covariates (predictors) available for that outcome,
   - a numeric-vs-ordinal indicator.
   μ carries "how explainable is an outcome *like this*", borrowed from **all**
   contexts including cross-instrument ones — the only signal available for gss,
   whose sole same-instrument sibling is the held-out-adjacent 1994 wave.

3. **σ²_co — retrieval noise, a learned function of retrieval breadth.** Modeled
   as `σ²_co = a + b / ESS_c` (or an equivalent monotone-decreasing-in-ESS
   form), with `(a, b)` fit on the contexts that have ≥2 siblings — where the
   **per-sibling R² spread** (already computed by B4's `per_sibling_r2`
   diagnostic) is a direct empirical handle on how noisy the transport is. cps
   (3 siblings) teaches the curve; it is **extrapolated to gss** (1 sibling,
   where spread is unmeasurable). This extrapolation *is* the cross-context
   borrowing that makes the rescue possible, and it is stated as the load-bearing
   assumption (see Limitations).

4. **τ² — between-context variance.** Fit by empirical Bayes
   (method-of-moments / REML) across training contexts: how much true R² varies
   around μ(f_o) beyond retrieval noise.

**Behaviour.** When retrieval is rich (σ²_co small, cps) the posterior sits on
x_co → cps is unchanged and stays near ceiling. When retrieval is thin (σ²_co
large, gss) the posterior shrinks toward μ(f_o) → gss is rescued by the pooled
prior instead of the collapsed single-sibling transport. **B4's discrete
ESS-gate is the limiting case of this smooth, fitted shrinkage.**

Closed-form, **numpy-only** — no new dependency, no MCMC, no LLM.

## Firewall (unchanged from B4, still auditable)

- **B5 reads:** target public univariate marginals (X and Y), target X-margins
  for raking, the outcome's public structural features (all derived from public
  marginals / the crosswalk). All public toplines.
- **B5 never reads:** the target context's per-person joint, its pairwise
  associations, its covariate-R², or the benchmark reference sample.
- **Training contexts contribute microdata** — allowed, they are training
  contexts held out under **leave-one-context-out** (only the target *wave* is
  held out; sibling waves and other instruments are training data, exactly as in
  B4 / roadmap #7).
- Provenance stays printable: μ's coefficients, τ², and the σ²(ESS) parameters
  are a handful of numbers checkable against any published tabulation.

## Fit & evaluation

- **Training corpus.** Every `(instrument × wave)` CSV on disk is one context
  (~13: cps ×4, gss ×2, nlsy ×2, and acs/cfps/charls/ecls/addhealth/
  understandingsociety ×1). Cross-instrument pooling is **forced**, not
  optional: same-instrument gss offers a single sibling, so the prior can only
  come from other instruments. Compute each context's true `covariate_r2` once
  as the fit's ground truth.
- **Leave-one-context-out.** To score target wave `c`, fit μ, τ², σ²(ESS) on all
  contexts **except `c`**, then predict `c`'s R² dict. The target wave's own true
  R² never enters its own fit.
- **Plug-in seam.** Feed the predicted dict through the existing generator via a
  new keyword-only override on `transfer_build_b2` (`r2_target: dict | None =
  None`) that, when supplied, replaces `agg["outcome_r2"]` directly and skips
  `target_aggregates` for the R² target. Default `None` keeps B2/B4 behaviour
  **byte-identical** (asserted by test).
- **Scoring / comparability.** Scored **identically** to B0–B4: same crosswalk
  `cols` (derived exactly as `run_layer2` / `b4_columns`), same
  `restrict_config_dir`, same reference (`load_schema(target_dataset).
  real_data_path`), same seed offset (`1000 + s`), `seeds = 3`, `n = 3000`,
  `bootstrap_B = 200`. B5 slots into the Layer-2 ladder beside B0–B4 in the same
  CSV / LEDGER / dashboard.

## Configs (the ablation, roadmap #8)

| Config | R² target source | Purpose |
|---|---|---|
| B2 (existing) | target pool's covariate-R² | aggregate-recalibration baseline |
| B4_retrieval (existing) | raked sibling pool (x_co) | zero-training transport |
| **B5_learned** (headline) | EB posterior R²̂(c,o) | full fitted shrinkage |
| **B5_prior_only** (diagnostic) | μ(f_o) only (τ²→0, no retrieval) | isolates the prior's contribution vs the retrieval data point |

- **B5_learned − B4_retrieval** = what the learned shrinkage adds over raw
  transport (the headline; must be ≥ 0 on cps and > 0 on gss to earn its keep).
- **B5_learned − B5_prior_only** = how much the retrieval data point still
  contributes once the prior exists (should be ~0 on gss where retrieval is
  worthless, positive on cps where it is near-ceiling).
- **B5_learned − B2** = the Phase-3 headline: does a strictly-firewalled fitted
  estimator beat the aggregate-reading B2 on both pairs?

## Components / files (for the plan to detail)

- **New:** `src/ssdataagent/transfer/rescue.py` — the EB model. Pure functions:
  outcome structural features `f_o` from public marginals; fit `(μ-coefficients,
  τ², a, b)` from a list of per-context training records; predict an R² dict for
  a held-out context given its retrieval x_co, ESS, and features. numpy-only.
- **New:** `scripts/transfer_b5.py` — orchestrator mirroring `transfer_b4.py`:
  build each context's true-R² training record + the target's retrieval x_co
  (reusing `transfer_b4.reweighted_pool_for` / `per_sibling_r2`), fit LOCO,
  predict, run `B5_learned` + `B5_prior_only` through the B2 machinery, score,
  write `results/transfer_map/b5_<pair>.csv`.
- **Modify:** `transfer_build_b2` — add keyword-only `r2_target: dict | None =
  None` override (default byte-identical). This is the only change to existing
  generation code.
- **Reused verbatim:** `transfer_b4.{b4_columns, reweighted_pool_for,
  per_sibling_r2}`, `conditional_variance.covariate_r2`,
  `target_aggregates`, `recalibrate.bidirectional_r2_blend`,
  `scoring.{restrict_config_dir, mean_scores}`,
  `nodonor_bracket.{carve_pool, score, TYPES, _drop_unnamed}`.

## Limitations (on the record now)

1. **Underpowered corpus.** ~13 cross-instrument contexts is at/below the
   roadmap's ~8–10 floor (risk #3). B5 is explicitly a *measurement* of whether
   learned shrinkage helps at this scale, not a bet that it will. The honest null
   (B5 ≈ B4 → corpus is the gate) is a first-class outcome, stated in the report.
2. **σ²(ESS) is extrapolated to gss.** The noise curve is fit where sibling
   spread is measurable (cps, ≥2 siblings) and applied where it is not (gss, 1
   sibling). This single extrapolation is what carries the rescue; the report
   must show the fitted curve and flag the gss point as off-support.
3. **Cross-instrument prior comparability.** μ pools R² across instruments with
   different outcomes and predictor sets; the structural features are the only
   normalizer. A large residual variance τ² (weak prior) is itself an
   informative finding.
4. **Numeric outcomes only.** T3 recalibration touches numeric outcomes;
   categorical outcomes are untouched (as in B2/B4). B5 does not change T1/T2.
5. **Same estimand as B4, one bundle component.** No association (T2) or
   dispersion / event-order (T4/T5) transport — deferred; the scored pairs are
   T1/T2/T3 and T2 injection is B2's proven dead-end.

## Success criteria

- The two B5 configs run and score on both scored pairs on the same footing as
  B0–B4, deterministically off the microdata (no API key — B5 is LLM-free).
- The fitted model is inspectable: μ coefficients, τ², and the σ²(ESS) curve are
  printed, with the gss point marked off-support.
- A clear verdict on the thesis: does pooled shrinkage rescue gss's T3 without
  hurting cps, and does B5 beat B2 and B4? This sizes Phase 4 (integrate) vs a
  corpus-expansion detour.

## Non-goals

- No corpus expansion (WVS/ISSP/ESS ingestion is its own project — roadmap
  risk #1).
- No neural net, no MCMC, no hierarchical-Bayes sampler — closed-form empirical
  Bayes only (roadmap #4 "start classical").
- No LLM (no elicited-prior base; the residual-on-LLM-prior variant of roadmap #3
  is explicitly deferred — the project's B3 evidence shows the LLM prior is a
  weak structure source, so the LLM-free pooling path is the headline).
- No T2 / association or T4/T5 transport.
- No change to T1 marginals or the shared-latent draw.
