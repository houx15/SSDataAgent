# B6 — ESS-gated hybrid generator (Phase 4, integration)

**Date:** 2026-07-24
**Branch:** `b6-ess-gated-hybrid` (off `main` @ 2fd6efb)
**Spec:** `docs/superpowers/specs/2026-07-24-b6-ess-gated-hybrid-design.md`
**Plan:** `docs/superpowers/plans/2026-07-24-b6-ess-gated-hybrid.md`
**Roadmap:** `docs/2026-07-22-transfer-roadmap.md` Phase 4

## What B6 is

The roadmap's **Phase 4 — Integration**: "the agent skeleton stays; the statistics
feeding it change source." B6 realizes it for the one bundle component B4/B5 proved
is the live problem (T3 conditional strength): a **single autonomous generator**,
`B6_hybrid`, that decides — from firewall-clean signals and **never from scores** —
whether to feed the pipeline B5's retrieval-blended R² (`learned`) or its
cross-context pooled-prior R² (`prior_only`), then tags provenance per outcome.

B5 emitted two configs and a human eyeballed which won per pair. **B6 is that
decision made autonomous and auditable.** The claim here is not a new score — it is
**completed integration**: the transfer pipeline now self-selects its R² source,
provenance-tagged, and the selection beats every prior baseline on both pairs.

## The gate

```
select_r2_source(n_siblings, ess, *, tau=0.3, min_siblings=2) -> bool
    return (n_siblings >= min_siblings) and (ess >= tau)
```

Trust the retrieval blend only when the sibling pool is both **plural** (≥2 raked
same-instrument contexts survived the crosswalk) **and effectively-sized** (raking
ESS ≥ τ) — i.e. *not a lone thin sibling*. Otherwise fall back to the pooled prior.

The two inputs are firewall-clean: `n_siblings` is a corpus count of held-out
sibling contexts; `ess` is a raking diagnostic over the target's public X-margins.
The gate reads no benchmark score.

## Results (3 seeds, n=3000, bootstrap_B=200 — same protocol as B0–B5)

| pair | n_sib | ESS | gate → source | T1 | T2 | T3 | **B6_hybrid** | B2 | B4_retr | B5_learned | B5_prior |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cps_1970_1980 | 3 | 0.65 | both pass → **retrieval** | .760 | .584 | .813 | **0.719** | .663 | .708 | .719 | .606 |
| gss_1994_2018 | 1 | 0.10 | both fail → **prior** | .725 | .794 | .664 | **0.728** | .683 | .653 | .686 | .728 |

**`B6_hybrid` picks the per-pair winner and beats B2 AND B4 on both pairs**, reading
zero target Y-data. By construction it equals the selected B5 config; the scorer's
per-seed identity check confirms this **exactly** — B6 vs the selected B5 config is
`|Δ| = 0.00` on all six seed scores (cps: 0.7143 / 0.7268 / 0.7153 == B5_learned;
gss: 0.7010 / 0.7032 / 0.7788 == B5_prior_only). B6 adds no numerical noise; it adds
the autonomous, firewall-clean, provenance-tagged *decision*.

## Findings

### 1. Integration is complete: the pipeline self-selects, and the selection wins both pairs.

The transfer generator no longer needs a human to pick retrieval-vs-prior per
context. `select_r2_source` reads `(n_siblings, ess)` — both computed before any
scoring — and routes cps to the retrieval blend (0.719) and gss to the pooled prior
(0.728). This is the ESS-gated selection B5's Finding 3 flagged as "the deliverable
Phase 4 should fold in now," now folded in: **one config, beating B2 (+0.056 / +0.045)
and B4_retrieval (+0.011 / +0.075) on both pairs at B2's own firewall.**

### 2. τ is deliberately not load-bearing — the sibling-count criterion carries the decision.

The AND-gate separates the two scored pairs on the **structural** criterion alone:
cps has 3 siblings, gss has 1, so `n_siblings ≥ 2` already routes them correctly
regardless of τ. The ESS term only reinforces it (cps ESS 0.65 ≫ gss 0.10). Concretely,
**any τ ∈ (0.10, 0.65) yields identical selection**, and the sibling-count criterion
alone yields the same answer — so no magic number carries the result. τ = 0.3 is a
documented default in the unidentified gap, not a tuned quantity. (Unit-tested:
`test_select_r2_source_boundaries_and_tau` sweeps τ ∈ {0.15, 0.30, 0.50, 0.60}.)

### 3. Provenance is per-outcome truthful and auditable.

Each outcome carries a source tag. On gss (prior branch) all 16 outcomes are
`prior`. On cps (retrieval branch) the tag is `retrieval-blend` **only where retrieval
actually moved the estimate** (`child_number`, `age_first_childbirth`, `income`) and
`prior` for the five outcomes whose retrieval `x_co` was None and fell to the prior μ
inside B5 (`marital_status`, `education`, `laborforce`, `occupation`, `poverty_status`).
The firewall stays inspectable per context: the gate record (`n_siblings`, `ess`, τ,
decision) and per-outcome provenance are printed and the context-level `source` +
`n_siblings` are written to the results CSV.

## Firewall (unchanged from B5, stricter than B2)

B6 reads only the target's **public marginals** (T1 draw), **X-margins** (raking the
sibling pool), and **public per-outcome structural features** (prior μ). The target's
covariate-R² is computed **nowhere**. The whole-branch review verified the gate
consumes only `n_siblings` + `ess` and that scoring output never feeds back into the
decision. The structure vehicle is B4's exact `sib_rew`, so B6 differs from the
selected B5 config in nothing but which R² map is chosen — confirmed by the exact
per-seed identity above.

## Limitations

1. **Two scored pairs ⇒ the gate is "validated" only in that it auto-picks the known
   per-pair winner.** With ESS at 0.10 and 0.65 and sibling counts 1 and 3, the
   decision boundary is not identified from the data — many rules separate these two
   points. A genuine held-out test of the *rule* (does it route an unseen context's
   ESS correctly?) needs **corpus expansion** (WVS/ISSP/ESS — roadmap risk #3). This
   is stated plainly; B6 does not claim a learned or cross-validated gate.
2. **No new score.** `B6_hybrid` reproduces B5's per-pair winner by construction; the
   contribution is integration (autonomy + provenance + a firewall-clean rule), not a
   higher number. The residual vs the microdata ceilings (cps 0.816, gss 0.811)
   survives — closing it is a corpus/learning problem, not an integration one.
3. **Numeric-outcome R² only** (inherits B5 scope). T1/T2 (and T4/T5 where applicable)
   draw through B4's `sib_rew`, unchanged. B6 does not expand the predicted bundle to
   marginals/dependence/event-order — those layers already transfer well under B4.

## Verdict → what's next

Phase 4's integration goal is met: the transfer pipeline is now a single autonomous
generator that routes its conditional-strength source by a stated, firewall-clean,
provenance-tagged ESS gate, beating B2 and B4 on both benchmark pairs. The two open
threads are both **corpus-scale**, not method-scale:

- **Learn/validate the gate** (Phase 5-adjacent): ingest cross-national programs so
  the σ²(ESS) slope and the decision boundary are *identified and held-out tested*,
  not assumed — the one thing that would turn "auto-picks the known winner" into
  "generalizes to unseen contexts."
- **Expand the predicted bundle** beyond T3 if a later gate opens for T1/T2/T4 — not
  justified today (B4 transfers them well).

## Reproduce

```
.venv/bin/python scripts/transfer_b6.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_b6.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```

Heavy scoring is reaped on the box even solo; the numbers here were produced with a
resumable per-seed scorer (`.superpowers/sdd/b6_incremental.py`, numerically identical
to `run_b6` — reuses its functions + `default_rng(0)`, reuses the cached B5 LOCO
prediction so a resume skips the ~300s fit, and cross-checks each seed's bit-identity
against the corresponding B5 config).
