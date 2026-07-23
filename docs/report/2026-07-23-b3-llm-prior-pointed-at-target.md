# B3: the no-donor LLM prior pointed at the target — and why Phase 3 is now justified

**Date:** 2026-07-23
**Scope:** Phase-2 rung **B3** of `docs/2026-07-22-transfer-roadmap.md`, the final baseline
before the decision gate. Same-country **time transfer** (CPS ASEC 1970→1980, GSS
1994→2018), the two benchmark-backed pairs.
**Spec:** `docs/superpowers/specs/2026-07-23-b3-llm-prior-pointed-at-target-design.md`
**Replication:** `.venv/bin/python scripts/transfer_b3.py <pair> --seeds 3 --n 3000 --bootstrap-B 200`

## The question

B1 (marginal-swap) and B2 (aggregate recalibration) both take the conditional structure —
the skeleton and copula — from the **source** context A. **B3 asks whether the structure
is better supplied by the LLM prior about the target B instead.** It is the current
no-donor full method (`scripts/nodonor_fullmethod.py`): draw demographic seeds from B's
marginals, have the LLM complete each person coherently from B's context and codebook, then
recalibrate conditional variance. Pointed at the target, restricted to the crosswalk
columns, scored identically to B0/B1/B2. No source context A is used at all.

Three rungs vary how the numeric outcomes' strength is calibrated (categorical outcomes and
the copula always come straight from the LLM):

- **B3_raw** — the LLM's own conditional strengths, no repair.
- **B3_elicited** — numeric strengths recalibrated toward the LLM's **elicited** R²
  (fully firewalled: no target aggregate touched for strength).
- **B3_pool_R2** — numeric strengths recalibrated toward the target pool's **covariate-R²**
  (the same aggregate footing as B2).

The decision gate (roadmap): *"only if a large mechanism-shift residual survives B2/B3 is
learned adaptation (Phase 3) justified."* B3 is what decides it.

## Result 1 — the LLM prior is a *weaker* structure source than transfer, on both pairs

Publication protocol (3 seeds, n=3000, bootstrap_B=200):

**cps 1970→1980** (10-year gap)

| config | T1 | T2 | T3 | overall |
|---|---:|---:|---:|---:|
| B0 carry-over | 0.401 | 0.619 | 0.656 | 0.558 |
| B1 marginal-swap | 0.810 | 0.554 | 0.573 | 0.646 |
| B2 recalibrated | 0.801 | 0.559 | 0.629 | **0.663** |
| **B3_raw** | 0.806 | 0.606 | 0.386 | **0.599** |
| B3_elicited | 0.802 | 0.579 | 0.326 | 0.569 |
| B3_pool_R2 | 0.804 | 0.518 | 0.199 | 0.507 |
| within-B independence floor | 0.856 | 0.378 | 0.004 | 0.413 |
| within-B microdata ceiling | 0.849 | 0.840 | 0.761 | 0.816 |

**gss 1994→2018** (24-year gap)

| config | T1 | T2 | T3 | overall |
|---|---:|---:|---:|---:|
| B0 carry-over | 0.395 | 0.821 | 0.619 | 0.612 |
| B1 marginal-swap | 0.641 | 0.816 | 0.494 | 0.651 |
| B2 recalibrated | 0.642 | 0.825 | 0.583 | **0.683** |
| B3_raw | 0.800 | 0.434 | 0.199 | 0.478 |
| B3_elicited | 0.802 | 0.483 | 0.554 | 0.613 |
| **B3_pool_R2** | 0.803 | 0.490 | 0.551 | **0.615** |
| within-B independence floor | 0.800 | 0.720 | 0.003 | 0.508 |
| within-B microdata ceiling | 0.806 | 0.900 | 0.727 | 0.811 |

**On both pairs, the best B3 rung sits below B1, which sits below B2:**
`cps 0.599 < 0.646 < 0.663`, `gss 0.615 < 0.651 < 0.683`. The LLM prior pointed at the
target — the "current no-donor method" — does **not** beat transplanting the source
structure, let alone recalibrating it. B3 is worse than B2 by **0.064 (cps)** and
**0.068 (gss)**, both beyond the ~0.054 noise floor.

## Result 2 — *how* the prior fails differs sharply by instrument

The overall numbers hide two opposite failure modes, both visible in the per-type split.

**cps: the prior gets dependence roughly right but the repair hurts.** B3_raw's T2 (0.606)
actually edges out B1/B2, and its T3 (0.386) is respectable — the LLM knows 1980 US
demographic structure well. But recalibrating the numeric outcomes *lowers* T3
monotonically (raw 0.386 → elicited 0.326 → pool 0.199). The LLM's raw conditional
strengths are already reasonable; shrinking them toward the R² targets moves them the wrong
way. This is not a B3 artifact — the durable `nodonor_fullmethod` shows the identical
pattern on cps (raw 0.596 → elicited 0.546 → pool 0.507; see Validation).

**gss: the prior invents dependence, and the repair rescues T3.** B3 *wins* T1 (0.80 vs
B1/B2's 0.64 — the marginal-driven generation is excellent) but **collapses on T2**
(0.43–0.49 vs B1/B2's 0.82). The LLM manufactures pairwise associations the real 2018 data
does not have: even the within-B independence floor scores T2 0.720, higher than any B3
rung, because independence at least matches the many genuinely-insignificant pairs. On T3
the story flips — B3_raw is catastrophic (0.199, the LLM wildly over-predicts conditional
strength) and the R² repair rescues it to 0.55 (elicited and pool tie), comparable to B2's
0.583. So on gss the prior's numeric strengths are badly over-confident and *need*
recalibration, the opposite of cps.

The two rungs that recalibrate:
- **`B3_pool_R2` vs `B2`** (same pool-R² aggregate, structure source differs): B2 wins on
  both (cps 0.663 vs 0.507; gss 0.683 vs 0.615). Transferred structure beats LLM structure
  even with strength held to the same target.
- **`B3_elicited` vs `B2`** (fully firewalled prior): B2 still wins (cps 0.663 vs 0.569;
  gss 0.683 vs 0.613). The prior does not adapt "for free" enough to matter.

## Validation — B3_raw reproduces the durable no-donor path

`B3_raw` is architecturally the durable no-donor headline method restricted to the crosswalk.
Run head-to-head on cps at full protocol:

| | T1 | T2 | T3 | overall |
|---|---:|---:|---:|---:|
| `nodonor_fullmethod` raw (full config, 12 cols) | 0.782 | 0.618 | 0.387 | 0.596 |
| B3_raw (crosswalk config, 11 cols) | 0.806 | 0.606 | 0.386 | 0.599 |

Overall matches within **0.003** (T3 identical to 0.001); the small T1 gap is `birth_year`,
a marginal the full config scores and the crosswalk drops as non-transferable. The pool-R²
rung matches exactly (0.507 both). The restricted-scoring path is faithful.

## Decision gate — a large mechanism residual survives B2 *and* B3 → Phase 3 is justified

The gate: *"if B1/B2 close most of the gap on most cells → statistics+agent paper; only if a
large mechanism-shift residual survives B2/B3 is learned adaptation (Phase 3) justified."*

The best no-training baseline is **B2**, at **0.663 (cps)** and **0.683 (gss)** against
ceilings of **0.816 / 0.811** — roughly **55–60% of the B1→ceiling gap remains unclosed**.
**B3 does not close it; B3 does not even reach B1.** Neither the LLM prior alone, nor
aggregate recalibration, nor their comparison, closes the residual. Every no-training route
to the target's conditional structure — transplant (B1), recalibrate (B2), elicit from the
prior (B3) — leaves a large, consistent gap, concentrated in T2/T3 (the dependence and
regression structure), the mechanism terms.

**This is the condition the roadmap names for justifying Phase 3.** The gate is now closed
in favor of **learned adaptation in statistics space** (context → predicted statistics
bundle, residual on the LLM prior). B3 also sharpens *what* Phase 3 must predict: not
marginals (B3 already nails T1) but the **X–Y dependence** (T2, where the prior invents
structure) and **per-outcome conditional strength** (T3, where the prior is miscalibrated in
*instrument-specific directions* — too weak on cps, too strong on gss). A learned corrector
on the elicited statistics is exactly aimed at that residual.

## Methods notes

- **`default_alpha = 1.0` in every rung.** The variance repair weakens a column's covariate
  dependence toward an R² target; categorical outcomes carry no R² target
  (`covariate_r2 → None`), so the repair has no basis to touch them. B3 leaves them at full
  LLM coherence in all rungs. The durable `nodonor_fullmethod` passes `default_alpha=0.5`,
  which additionally half-shrinks every categorical outcome — a heuristic dropped here
  because it both depresses T2/T3 and confounds the `B3_pool_R2` vs `B2` comparison (B2
  recalibrates only numeric R²). This was found and fixed during validation: the first run
  hardcoded 0.5 and produced a corrupted `B3_pool_R2`.
- **Firewall (row-level).** B3 reads from the target only per-column marginals (seeds +
  prompt blob), per-outcome covariate-R² (`B3_pool_R2` rung only), and public
  codebook/context. Never the target's joint, never the reference/test sample. Every rung is
  provenance-tagged.
- **GSS scored via a resumable per-rung scorer.** At full protocol the GSS 16-outcome
  bootstrap exceeded this environment's memory/wall-clock repeatedly. It was scored one rung
  at a time with each rung persisted on completion (numerically identical to a single
  `run_b3` call — each seed uses a fresh `default_rng(s)`, so rung order is irrelevant). The
  ~13-minute GSS LLM generation is cached under `results/nodonor_cache/gss_cond_raw.csv`.

## Limits and honesty

- **Only two genuine transfer pairs** (Layer-2 resolves the reference from the dataset name;
  see the B2 report). B3 is scored on exactly those two.
- **`mental_health` absent from GSS B3** — a real GSS-2018 T3 outcome the crosswalk drops
  because the 1994 source lacks the column. An honest crosswalk gap, not a choice.
- **Attitude variables under measurement non-invariance** (`gender_role_attitude`,
  `political_view`, `trust`, `work_hard`): B3's T2 collapse on GSS is partly the LLM
  imposing present-day association structure on these; no method is claimed to transfer a
  construct whose meaning shifts.
- **The prior is one model, one temperature.** B3 uses `anthropic/claude-sonnet-4.5` at
  temperature 0.8. A different or larger model might carry stronger structure; the claim is
  about the *current* no-donor method, not about LLM priors in general.
- **Same-country time transfer only.** No country transfer, no cross-cultural claim.
- **Row-level firewall**, not person-level (no person key in the pools).

## Replication

```bash
export OPENROUTER_API_KEY=...   # first run only; GSS generation caches under results/nodonor_cache/
.venv/bin/python scripts/transfer_b3.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_b3.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
# validation:
.venv/bin/python scripts/nodonor_fullmethod.py cps --seeds 3 --n 3000 --bootstrap-B 200
```

Outputs: `results/transfer_map/b3_<pair>.csv` (three rungs each). Modules:
`src/ssdataagent/transfer/{scoring,b3_specs}.py`; orchestrator `scripts/transfer_b3.py`;
reused stages `scripts/nodonor_fullmethod.py`.
