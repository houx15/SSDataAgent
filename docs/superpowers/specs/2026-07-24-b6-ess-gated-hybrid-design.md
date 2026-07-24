# B6 — ESS-gated hybrid generator (Phase 4 integration)

**Date:** 2026-07-24
**Branch:** `b6-ess-gated-hybrid` (off `main` @ 2fd6efb)
**Roadmap:** `docs/2026-07-22-transfer-roadmap.md` Phase 4 (Integration: the hybrid pipeline)
**Predecessors:** B4 (`docs/report/2026-07-23-b4-retrieval-kob-transport.md`),
B5 (`docs/report/2026-07-23-b5-learned-r2-rescue.md`)

## What B6 is

The roadmap's **Phase 4** — "the agent skeleton stays; the statistics feeding it
change source." B6 realizes it for the one bundle component B4/B5 proved is the
live problem (T3 conditional strength): a **single autonomous generator** that
decides, from firewall-clean signals and **never from scores**, whether to feed
the pipeline the retrieval-blended R² (B5's `learned`) or the pooled-prior R²
(B5's `prior_only`).

B5 emitted two configs and a human eyeballed which won per pair. B6 is the
integration deliverable: one config, `B6_hybrid`, that self-selects and carries a
provenance tag per outcome. It closes Phase 4's integration goal without any new
training — it is a thin, auditable selection layer over B5.

## Thesis

B5 Finding 3: an estimator that **selects by ESS** — retrieval-blend when the
sibling pool is reliable (cps → 0.719), prior-only when it is thin (gss → 0.728)
— beats B2 **and** B4 on **both** scored pairs, reading zero target Y-data. B6 is
that estimator, packaged as the canonical transfer generator with a decision rule
that is stated, firewall-clean, and robust to its one free parameter.

## Architecture

A thin layer on B5. `predict_target_r2(pair)` already computes everything B6
needs; B6 adds only a selection step.

```
learned, prior_only, ess, sib_rew, n_siblings = predict_target_r2(pair)   # B5, +n_siblings
use_retrieval = (n_siblings >= 2) and (ess >= TAU)          # TAU = 0.3
r2_map[o]     = learned[o] if use_retrieval else prior_only[o]   # per outcome
sim           = transfer_build_b2(sib_rew, target_pool, cols, covs, outs,
                                  n, seed, r2_target=r2_map)     # B4/B5 vehicle, unchanged
```

The structure vehicle is B4's exact `sib_rew` (the raked LOCO sibling pool), and
only the R² source is switched. Therefore **B6 reproduces the per-pair winning
config by construction**:

| pair | n_siblings | ESS | gate | B6 == | overall |
|---|---|---|---|---|---|
| cps_1970_1980 | 3 | 0.65 | both pass → retrieval | B5_learned | **0.719** |
| gss_1994_2018 | 1 | 0.10 | both fail → prior | B5_prior_only | **0.728** |

Beats B2 (.663 / .683) and B4_retrieval (.708 / .653) on both pairs, as one
self-selecting config, zero training.

## The gate

```
select_r2_source(n_siblings, ess, *, tau=0.3, min_siblings=2) -> bool
    return (n_siblings >= min_siblings) and (ess >= tau)
```

**Two criteria, AND-combined**, encoding "trust retrieval only when the sibling
pool is both *plural* and *effectively-sized* — i.e. not a lone thin sibling":

- `n_siblings >= 2` — a **structural** fact (how many independent same-instrument
  contexts survived the crosswalk to retrieve from). cps has 3 (1970/1990/2000,
  target 1980 held out); gss has 1 (1994, target 2018 held out).
- `ess >= tau` — the raking effective-sample-size ratio, the reliability signal
  B5 built its σ²(ESS) curve around.

**τ is deliberately not load-bearing.** The `n_siblings >= 2` criterion is what
structurally separates the two scored pairs; τ can sit anywhere in the
unidentified gap (0.10, 0.65) for identical selection, *and* the sibling-count
criterion alone yields the same decision. The report states this sensitivity as
the robustness argument: no magic number carries the result. τ = 0.3 is a
documented default in the gap, not a tuned quantity.

## Firewall

Unchanged from B5 (stricter than B2). The generator reads only:

- the target's **public marginals** (T1 draw),
- the target's **X-margins** (raking `sib_rew`),
- **public per-outcome structural features** (prior μ),

and the target's covariate-R² is computed **nowhere**. The two gate inputs are
firewall-clean: `n_siblings` is a corpus fact (a count of held-out sibling
contexts), and `ess` is a raking diagnostic computed over public target margins.
No score is consulted by the gate.

## Provenance (roadmap §152)

Each module keeps a provenance tag so the firewall stays auditable per context.
B6 emits:

- **Per-outcome source tag** — `"retrieval-blend"` or `"prior"` for every native
  outcome, **truthful per outcome**: in the retrieval branch an outcome is tagged
  `"retrieval-blend"` only when retrieval actually moved it
  (`learned[o] != prior_only[o]`); outcomes whose retrieval `x_co` was None have
  `learned[o] == prior_only[o]` (pure prior μ inside B5) and are tagged `"prior"`.
  In the prior branch every outcome is `"prior"`.
- **Context-level gate record** — `n_siblings`, `ess`, `tau`, and the boolean
  decision.

Written to the output CSV (`source`, `n_siblings` columns beside `ess_ratio`) and
printed by the orchestrator. A sidecar is not required — the CSV row plus stdout
carry the full provenance.

## Components and files

- **`src/ssdataagent/transfer/rescue.py`** (extend): add two pure, unit-testable
  functions —
  - `select_r2_source(n_siblings, ess, *, tau=0.3, min_siblings=2) -> bool`
  - `hybrid_r2_map(learned, prior_only, use_retrieval) -> tuple[dict, dict]`
    returning `(r2_map, provenance)`. When `use_retrieval` is True, `r2_map =
    learned` and `provenance[o] = "retrieval-blend"` iff `learned[o] !=
    prior_only[o]` else `"prior"`. When False, `r2_map = prior_only` and every
    `provenance[o] = "prior"`.
- **`scripts/transfer_b5.py`** (minimal change): `predict_target_r2` captures
  `used_waves` (already returned by `reweighted_pool_for`) and returns
  `n_siblings = len(used_waves)` as a 5th tuple element. B5's own call site is
  updated to unpack-and-ignore it; B5 output is **byte-identical** (the extra
  return value is never used by B5's scoring).
- **`scripts/transfer_b6.py`** (create): thin orchestrator — `predict_target_r2`
  → `select_r2_source` → `hybrid_r2_map` → score one `B6_hybrid` config through
  `transfer_build_b2(sib_rew, …, r2_target=r2_map)` identically to B0–B5 → write
  CSV with `source` / `n_siblings` columns → print the gate decision and
  per-outcome provenance.
- **Tests**:
  - `tests/test_transfer_rescue.py` (extend): the gate truth table —
    (3 sib, .65) → True; (1 sib, .10) → False; (3 sib, .10) → False (ESS fails);
    (1 sib, .65) → False (count fails); τ-boundary behavior; and `hybrid_r2_map`
    selecting the right map + emitting the right provenance for both branches.
  - `tests/test_transfer_b6.py` (create): fast wiring test — with a stubbed/small
    `predict_target_r2`, assert the selected `r2_map` equals `learned` when the
    gate passes and `prior_only` when it fails, and that provenance tags match.
    No full scoring in the unit test.

## Scoring protocol

Identical to B0–B5: 3 seeds, `n=3000`, `bootstrap_B=200`, same scorer, same
noise floor (~0.054), scored end-to-end on both benchmark pairs. Heavy scoring
jobs are reaped on the box even solo, so reuse the resumable per-(config,seed)
scorer pattern from B5 (`.superpowers/sdd/` scratch, gitignored), which caches the
deterministic LOCO fit so a resume rebuilds only the fast `sib_rew`.

## Expected result

`B6_hybrid`: cps **0.719**, gss **0.728** — reproducing B5's per-pair winner as a
single autonomous config. The scientific claim is **not** a new score; it is
**completed integration**: the transfer pipeline now self-selects its R² source
from firewall-clean signals, provenance-tagged, beating every prior baseline on
both pairs.

## Limitations (stated up front in the report)

1. **τ is not load-bearing** (headline robustness point): the `≥2 siblings`
   criterion structurally separates the pairs; any τ ∈ (0.10, 0.65) — and the
   count criterion alone — gives identical selection. Reported as sensitivity.
2. **Two scored pairs ⇒ the gate is "validated" only in that it auto-picks the
   known per-pair winner.** A genuine held-out test of the gate (does the rule
   generalize to an *unseen* context's ESS?) needs corpus expansion (WVS/ISSP/ESS
   — Phase 5 / roadmap risk #3). Not a new finding here; stated plainly.
3. **Numeric-outcome R² only** (inherits B5 scope). T1/T2 (and T4/T5 where
   applicable) draw through B4's `sib_rew`, unchanged — so `B6_hybrid` differs
   from the selected B5 config in nothing at all on these two pairs; its value is
   the autonomous, provenance-tagged decision rule, not a different draw.
4. **B6 does not expand the predicted bundle.** Marginals (T1), dependence (T2),
   and event-order (T4/T5) are still drawn from the disjoint pool / B4 vehicle,
   not predicted. Extending prediction to those layers is a later, larger scope
   the roadmap contemplates but B4 already transfers them well.

## Reproduce

```
.venv/bin/python scripts/transfer_b6.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_b6.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```
