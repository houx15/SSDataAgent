# Technical roadmap: from no-donor generation to cross-context transfer

**Date:** 2026-07-22
**Goal:** a generator `G(context, public aggregates) → synthetic sample` that produces
reliable survey samples for a context it was never fitted on — e.g. US 2020 survey →
China analog, wave 2020 → wave 2022 — while keeping the no-donor firewall for the
*target* context (target microdata is never read; only its public aggregates are).

**Core formalization.** A context is `c = (population, time, instrument)`. Factorize the
joint:

```
P_c(X, Y) = P_c(X) · P_c(Y | X)
            ─────    ────────────
            composition   mechanism
```

and further split the mechanism into three transferable layers:

```
P_c(Y | X) = skeleton S        (which X→Y edges exist, their signs)   — candidate invariant
           × parameters θ_c    (strengths, intercepts, thresholds)    — context-modulated
           × dispersion D_c    (residual variance / conditional shape) — the "same X, different Y" term
```

Equivalent copula view: `P_c(X,Y) = marginals_c × copula_c`. The face-swap hypothesis is
**copula ≈ stable, marginals ≈ context-specific** — keep the structure, swap the features.
Everything below is organized around measuring how true that is, then patching where it
is false.

---

## Phase 1 — Measurement: build the transfer benchmark (~3–4 weeks)

**Do this first. No method work until this exists.** Right now every score is
within-context; the final goal is a cross-context claim, so the scorer must be pointed
across contexts before any method can be said to work.

1. **Context pairs.**
   - *Time transfer:* GSS wave t → wave t+k (same instrument, same country — cleanest
     pair); CPS year → year.
   - *Country transfer:* CPS ↔ CFPS on a harmonized variable crosswalk (the crosswalk
     table is real labor — expect ~15–25 matchable variables; document every semantic
     compromise the way `data_audit.py` documents traps).
2. **Adapt the SSDataBench scorer to transfer mode:** everything fitted on context A +
   aggregates of B; scored against reference sample of B. Same seeding discipline,
   `bootstrap_B = 200`, same noise floor (~0.054).
3. **Kitagawa–Oaxaca–Blinder decomposition** per (context pair, Y variable): how much of
   the gap in Y's distribution closes by reweighting X alone (composition share) vs.
   remains (mechanism shift)?
4. **Copula stability test:** fit dependence structure on A, marginals from B, score
   against B. Per variable pair, per context pair.

**Deliverable: the transfer map** — a table of (context-pair × variable) cells labeled
composition-dominated (cheap to transfer) vs. mechanism-shifted (hard). Expected pattern:
time transfer mostly composition; country transfer OK for demographics→behavior, breaks
for demographics→attitudes (measurement invariance).

**This deliverable is publishable on its own** even if no method follows.

## Phase 2 — Baseline ladder (~2–3 weeks, mostly LLM-free)

Score four baselines on the Phase 1 benchmark, cheapest first:

- **B0 — naive carry-over:** generate for A, score against B. The floor.
- **B1 — marginal swap (MRP-style):** mechanism estimated on A, X drawn from B's
  marginals. Tests pure composition transfer. No-donor for B by construction.
- **B2 — skeleton + aggregate recalibration:** keep A's (or the LLM prior's) skeleton;
  recalibrate θ and dispersion from B's *published aggregates only*. This generalizes the
  two tricks that already work (variance repair, event-order module) into a pattern:
  "LLM/source structure, target-aggregate calibration."
- **B3 — current no-donor method pointed at B:** LLM prior prompted with B's context and
  codebook. Measures how much context adaptation the prior already does for free.

**Decision gate:** if B1/B2 close most of the gap on most cells → the paper is a
statistics + agent paper; skip to Phase 4 and only revisit training if reviewers demand
it. Only if a large mechanism-shift residual survives B2/B3 is learned adaptation
(Phase 3) justified. Do not train models to close gaps that reweighting closes.

## Phase 3 — Learned statistics generator: context → sufficient statistics
*(only if the gate opens; ~2–3 months)*

**Design principle.** Modulate the generator in **statistics space, not weight space**.
Hypernetwork/hyper-LoRA approaches (Text-to-LoRA, SHINE, Zhyper — cite as related work,
contrast in the paper) map context → adapter *weights*: opaque, LLM-specific, and
data-hungry (millions of output dimensions). Here instead a small model maps context →
the **statistics bundle** the agent pipeline already consumes. This continues the line
already in the 2026-07-21 report ("deep learning model → generate the statistics that
normally require real data → guide the LLM"). Nothing else in the pipeline changes:
today the bundle is read from the disjoint pool's aggregates; here it is *predicted*
for the target context.

Advantages over weight-space: **auditable** (predicted statistics can be printed and
checked against any published tabulation — the firewall stays inspectable),
**low-dimensional** (hundreds of numbers, so tens of training contexts can suffice where
a hypernetwork starves), **LLM-agnostic**, and naturally **uncertainty-carrying**.

1. **The statistics bundle θ_c** (what the model predicts for target context c):
   - X composition: demographic marginals and key cross-tabs (usually partly public
     from census — the model fills gaps, it doesn't predict from scratch);
   - X–Y dependence: correlation/copula parameters or standardized effect sizes per
     (X, Y) pair;
   - conditional strength: **R² targets per outcome** — the variance-repair alpha
     becomes a predicted quantity instead of a pool-derived one;
   - dispersion: residual variance / conditional shape per Y (the "same X, different Y"
     term, explicit);
   - event-order distributions where the instrument is longitudinal.
2. **Input asymmetry.** X-side aggregates (census marginals) are public for almost any
   (country, year); the Y-side and X–Y dependence are what's missing. So inputs are:
   context descriptors (country, year, topic, codebook variable semantics — embeddable
   from codebook text) **plus the known X-side aggregates**; outputs are the Y-side of
   the bundle.
3. **Residual design (recommended).** Don't predict raw statistics — predict the
   **correction between the LLM prior's elicited statistic and the truth**, learned
   across training contexts. The LLM prior carries the semantics ("education–income
   positive everywhere"); the small model learns the prior's systematic biases (e.g.
   "attitude correlations overestimated by ~0.15 in context family Z"). Lower-variance
   target, and the learned bias map is a publishable finding in itself.
4. **Model class, ranked by data appetite.** Start classical: **hierarchical Bayes /
   partial pooling** across contexts (interpretable shrinkage, posterior uncertainty
   for free, works at tens of contexts). Next: GP or gradient boosting per statistic
   type over (context features × variable-pair embedding). A small neural net only if
   the corpus grows large.
5. **Uncertainty gating.** Generalize the existing rule ("variance repair can hurt when
   well-calibrated — apply only where measurably over-strong"): apply each predicted
   correction only where the posterior is confident; otherwise fall back to the LLM
   prior untouched.
6. **Task corpus.** Every (country × wave × instrument) is one training context:
   GSS waves + CPS years + CFPS waves, extended with harmonized cross-national programs
   (**WVS, ISSP, ESS**) as capacity allows. Compute each context's true bundle once;
   train on descriptor → bundle; evaluate strictly **leave-one-context-out**.
7. **Firewall (amended claim):** training contexts contribute microdata (only to compute
   their aggregate statistics); the **target context contributes only public aggregates,
   never microdata**. "No donor from the target context" — state it from day one.
8. **Ablation:** learned predictor vs. B2 (zero-training aggregate recalibration — the
   special case of this architecture) vs. retrieval of nearest-context statistics with a
   KOB-style adjustment. The trained model must beat both to earn its complexity.

## Phase 4 — Integration: the hybrid pipeline (~3–4 weeks)

The agent skeleton stays; the statistics feeding it change source:

```
audit(codebook_B, public aggregates_B)               # test-blind, as today
→ θ̂_B = statistics model(context_B, X-aggregates_B) # predicted bundle + uncertainty
→ X ~ marginals from θ̂_B                            # composition (T1)
→ Y | X from LLM prior elicited as today             # mechanism skeleton
→ recalibrate strengths, dispersion to θ̂_B          # uncertainty-gated; subsumes variance repair
→ event-order module from θ̂_B                       # T4/T5 where applicable
```

Each module keeps a provenance tag (which aggregate, which prior) so the firewall stays
auditable per context.

## Phase 5 — Validation & claims (~3–4 weeks)

- **Held-out contexts, strictly:** train with China excluded → test CFPS; train with
  post-2020 waves excluded → test the newest wave. Never tune anything on a test context.
- **Per-type ablations** (as in the current report): which component moves T1…T5 in the
  *transfer* setting.
- **Honest-limits section written in advance:** attitude variables under measurement
  non-invariance; crosswalk compromises; cells where transfer fails and why.

---

## Risks, ordered by how likely they are to actually hurt

1. **Harmonization labor, not modeling, is the bottleneck.** The CPS↔CFPS crosswalk and
   WVS/ISSP ingestion will consume more time than any training run. Budget accordingly.
2. **Measurement non-invariance** on attitudes: no method can transfer a mechanism for a
   construct that changes meaning across cultures. The transfer map (Phase 1) bounds the
   claim; don't fight it.
3. **Task-corpus scale.** The statistics-space design is far less data-hungry than
   weight-space alternatives, but hierarchical pooling still needs enough contexts to
   estimate between-context variation — below ~8–10 contexts, stay with B2 + retrieval
   and skip training entirely.
4. **Scorer transferability:** SSDataBench tests assume within-context references; check
   each test still means something when the reference is a different context's sample.

## Sequence at a glance

```
Phase 1  transfer benchmark + KOB + copula test     ← the immediate next step
Phase 2  B0–B3 baseline ladder → decision gate
Phase 3  (conditional) small statistics model: context → θ̂ (residual on LLM prior)
Phase 4  hybrid agent+adapter pipeline
Phase 5  held-out-context validation
```

Two publishable checkpoints regardless of how far it goes: the **transfer map**
(Phase 1–2, no training needed) and the **full transfer method** (Phase 3–5).
