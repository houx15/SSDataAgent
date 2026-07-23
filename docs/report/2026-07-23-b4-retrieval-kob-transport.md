# B4: transporting the target's joint structure from raked sibling contexts

**Date:** 2026-07-23
**Scope:** Phase-3 **slice 1** (the zero-training retrieval + KOB ablation) of
`docs/2026-07-22-transfer-roadmap.md`, on the same two scored pairs as B0–B3.
**Spec / plan:** `docs/superpowers/specs/2026-07-23-b4-retrieval-kob-transport-design.md`,
`docs/superpowers/plans/2026-07-23-b4-retrieval-kob-transport.md`.

## The question

B2 left ~60% of the B1→ceiling gap unclosed, and B3 (the LLM prior pointed at the
target) was worse than B1/B2. The roadmap's decision gate is open, but the gate does
not say **what** the residual is. B4 performs a Kitagawa–Oaxaca–Blinder decomposition
of it: is the residual **composition** — closeable by reweighting a sibling context to
the target's public X-marginals, reading zero target Y-data — or **genuine mechanism
shift**, which only a learned model can borrow across contexts?

## What B4 is

Take the target's joint Y-structure (pairwise associations → T2, per-outcome
conditional strength → T3) not from the pair's single designated source but from a
**leave-one-context-out pool of same-instrument siblings**, raked to the target's
public X-marginals (IPF on age/gender/race), and run it through the *existing* B1/B2
shared-latent draw. No copula surgery — the transported structure rides the same
faithful draw B1 already uses (which is why B2's Step-A association-injection dead end
is never re-entered). Two configs isolate the two moving parts:

| config | structure source | R² (T3) target | firewall vs B2 |
|---|---|---|---|
| **B2** (baseline) | single designated source | target pool | — |
| **B4_retrieval_targetR2** | raked LOCO sibling pool | target pool | **same** as B2 |
| **B4_retrieval** | raked LOCO sibling pool | transported (sibling) | **stricter** — reads no target R² |

- `B4_retrieval_targetR2 − B2` = the pure **retrieval + reweighting** effect (KOB
  composition transport), holding the R² source at B2's.
- `B4_retrieval − B4_retrieval_targetR2` = the cost/benefit of **giving up the target
  R² aggregate** and transporting it from siblings instead.

Both configs are LLM-free, deterministic off the microdata, and scored on the identical
footing as B0–B3 (same crosswalk, `restrict_config_dir`, reference, seed offset,
`bootstrap_B=200`, 3 seeds).

## Results

**cps 1970→1980** (3 siblings {1970, 1990, 2000}, ESS ratio **0.65**):

| config | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| B1 marginal-swap | 0.810 | 0.554 | 0.573 | 0.646 |
| B2 recalibrated | 0.801 | 0.559 | 0.629 | 0.663 |
| B4_retrieval_targetR2 | 0.773 | 0.582 | 0.726 | 0.694 |
| **B4_retrieval** (firewalled) | 0.759 | 0.611 | **0.753** | **0.708** |
| within-target microdata ceiling | 0.849 | 0.840 | 0.761 | 0.816 |

**gss 1994→2018** (1 sibling {1994} — retrieval degenerate, ESS ratio **0.10**):

| config | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| B1 marginal-swap | 0.641 | 0.816 | 0.494 | 0.651 |
| B2 recalibrated | 0.642 | 0.825 | 0.583 | 0.683 |
| B4_retrieval (firewalled) | 0.726 | 0.787 | 0.447 | 0.653 |
| **B4_retrieval_targetR2** | 0.724 | 0.794 | **0.687** | **0.735** |
| within-target microdata ceiling | 0.806 | 0.900 | 0.727 | 0.811 |

## Two findings, one mechanism

**Finding 1 — the KOB composition transport beats B2 on both pairs, at B2's own
firewall.** `B4_retrieval_targetR2` (raked sibling pool, still reading the target R²)
> B2 on both: cps **+0.031**, gss **+0.052** overall. The single designated source,
used unraked as B0–B3 do, leaves composition signal on the table; raking a sibling pool
to the target's public X-margins recovers it. On cps the gain is in T3 (+0.097) and T2
(+0.023); on gss it is in T1 (+0.082) and T3 (+0.104). This half reads only X-margins —
public everywhere. **It is a free, strong win and should be folded into the pipeline
regardless of what slice 2 does.**

**Finding 2 — transporting the R² from siblings works when the pool is rich and fails
when it is thin, exactly as ESS predicts.**
- cps (3 siblings, ESS 0.65): the fully-firewalled `B4_retrieval` **beats** even the
  target-R² config — 0.708 vs 0.694, T3 0.753 vs 0.726. The transported sibling R² is a
  *better* target than the target pool's own R², and B4_retrieval closes **94% of the
  B2→ceiling T3 residual** (0.629→0.753, ceiling 0.761) reading zero target Y-data.
- gss (1 sibling, ESS 0.10): switching to the transported R² **collapses T3** from
  0.687 to 0.447 (−0.240) — below even B1 — dragging `B4_retrieval` (0.653) under B2.
  A single sibling raked to a very different 2018 composition (ESS 0.10 = a handful of
  1994 rows carry most of the weight) is too thin to estimate a conditional strength.

The ESS ratio is not a footnote — it is the dial. High retrieval breadth → the target's
conditional structure transports from siblings alone; low breadth → the transported R²
is noise and you must keep the target aggregate.

## Verdict on the estimand

**A large part of the B2 residual is composition-transportable, not irreducible
mechanism shift** — but the reach depends on retrieval breadth:

- The **composition** component (raking to X-margins) closes a real chunk of the
  residual on **both** pairs (cps +0.031, gss +0.052), reading only public X-margins.
- The **conditional-strength (T3)** component transports from siblings **when enough
  siblings exist** (cps: near-ceiling T3, fully firewalled) and **not when they don't**
  (gss: the single thin sibling gives a bad R²).

So "the residual is genuine mechanism shift" is **rejected for the parts that transport**
(most of cps's T3, the composition on both pairs) and **confirmed as a data-scarcity
problem, not a mechanism-invariance problem, for gss**: the mechanism is *there* to
borrow — a rich pool would supply it — there simply is only one gss sibling on disk.

## What this sharpens for Phase 3 slice 2 (the learned model)

The gate stays open, and slice 2 is now aimed:

1. **Fold the KOB composition transport (raking) into the pipeline now.** It is a
   zero-training, X-margin-only win on both pairs and does not need the learned model.
2. **Slice 2's real job is robust conditional-strength transport at low retrieval
   breadth** — precisely the gss regime. A learned predictor that pools the R² /
   dependence across *many* contexts is exactly what makes the estimate reliable where a
   single raked sibling (ESS 0.10) is too thin. cps shows the ceiling this can reach;
   gss shows why pooling is needed to reach it everywhere.
3. **Uncertainty gating has a concrete, measured signal: ESS.** The roadmap's
   "apply the correction only where the posterior is confident" maps directly onto the
   retrieval ESS ratio. An ESS-gated hybrid — transported R² when ESS is high (cps),
   fall back to the target aggregate (or B2) when ESS is low (gss) — would score
   ≈0.708 (cps) and ≈0.735 (gss), **beating B2 on both**. That hybrid is the natural
   slice-2 baseline the learned model must then beat.

## Firewall

B4 reads from the target **only** its public univariate marginals (X and Y, via the
inverse-CDF value map) and its age/gender/race margins (for raking). It never reads the
target's per-person joint, its pairwise associations, its covariate-R² (the
`B4_retrieval` config), or the benchmark reference sample. Siblings contribute microdata
(allowed — LOCO holds out only the target wave). `B4_retrieval` is therefore a
*stricter* firewall than B2, which reads the target's covariate-R²; `B4_retrieval_targetR2`
matches B2's firewall exactly and isolates the reweighting gain.

## Limitations

- **gss retrieval is degenerate** (one same-instrument sibling), so the gss slice tests
  only the reweighting effect and a single-sibling R² transport — the ESS 0.10 makes its
  `B4_retrieval` number a cautionary data point, not a method verdict. cps (3 siblings,
  ESS 0.65) is where the retrieval story is real.
- **LOCO uses future waves** (1990/2000 to hit 1980): temporally odd but firewall-clean,
  and the exact protocol slice 2 needs.
- **Per-sibling R² spread** (diagnostic): cps `age_first_childbirth` 0.33–0.50,
  `income` 0.08–0.13 across waves — moderate mechanism drift the pool averages over; the
  spread being bounded is why pooling works on cps.
- Numbers scored via a resumable per-(config,seed) scorer (the box reaps heavy jobs);
  numerically identical to `transfer_b4.run_b4` since it reuses its functions and the
  same `default_rng(0)` for `sib_rew`.
