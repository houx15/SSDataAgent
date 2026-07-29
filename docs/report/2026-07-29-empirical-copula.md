# Empirical-copula transfer — the engine was the leak

**Date:** 2026-07-29
**Branch:** `empirical-copula` (off `main` @ 68f0d1b)
**Spec/plan:** `docs/superpowers/specs/2026-07-29-empirical-copula-design.md`,
`docs/superpowers/plans/2026-07-29-empirical-copula.md`

## What this is

The first result this week that **moves the score**. A faithful empirical-copula transfer
replaces the generator's one-factor `glat` mechanism — which collapsed all categorical
dependence onto a single latent axis — with a joint-preserving one: each resampled row's
*actual* category interval. Source A's full joint is preserved exactly; the marginals are still
installed by the existing inverse-CDF, so structure and marginals stay separable.

## Results (identical scorer/protocol to B0–B6)

### cps_1970_1980 — full (3 seeds, n=3000, B=200)

| config | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| carryover (current engine) | 0.401 | 0.619 | 0.656 | 0.558 |
| **EC_carry** (A joint + A marginals, *reads no B*) | 0.449 | **0.758** | **0.751** | **0.653** |
| **EC_oracle** (A joint + B marginals) | 0.856 | 0.623 | 0.578 | **0.686** |
| ref_oracle_comp (== B1, old engine + B marginals) | 0.810 | 0.554 | 0.573 | 0.646 |
| ref_floor / ref_ceiling | 0.856 / 0.849 | 0.378 / 0.840 | 0.004 / 0.761 | 0.413 / 0.816 |

### gss_1994_2018 — reduced precision (1 seed, n=3000, B=50; heavier scoring, see Limitations)

| config | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| carryover | 0.420 | 0.815 | 0.620 | 0.618 |
| **EC_carry** (reads no B) | 0.434 | 0.866 | **0.920** | **0.740** |
| **EC_oracle** | 0.789 | 0.835 | 0.520 | 0.714 |
| ref_oracle_comp (== B1) | 0.639 | 0.816 | 0.447 | 0.634 |
| ref_floor / ref_ceiling | 0.819 / 0.824 | 0.719 / 0.896 | 0.000 / 0.633 | 0.513 / 0.784 |

## Findings

### 1. The generator, not the marginals, was the biggest leak.
`EC_carry` beats the current `carryover` engine by **+0.095 (cps) / +0.122 (gss)**, driven by
structure: T2 **+0.14 / +0.05**, T3 **+0.10 / +0.30**. Both use *A's own marginals* — the only
change is a faithful joint. The one-factor `glat` mechanism was discarding most of the categorical
dependence.

### 2. A feasible arm beats the oracle-marginal baseline.
`EC_carry` reads **no B data** (A's joint + A's marginals) yet beats `B1` (old engine + B's *true*
marginals): **0.653 vs 0.646 (cps), 0.740 vs 0.634 (gss)**. Getting the structure right is worth
more than getting the marginals right — a reversal of the working assumption.

### 3. Copula improvement holds with marginals fixed.
Same B marginals, better copula: `EC_oracle` vs `B1` = **+0.040 (cps) / +0.080 (gss)**. On cps,
`EC_oracle` (0.686) closes ~50% of the carry-over→ceiling gap.

### 4. Swapping marginals costs structure — the next constraint.
`EC_carry`'s T2/T3 exceed `EC_oracle`'s (cps 0.758/0.751 vs 0.623/0.578; gss even more), because
mapping A's rank structure onto B's *different* marginal values perturbs the realized
associations. On gss the structure cost of the swap outweighs the T1 gain, so `EC_carry` (0.740) >
`EC_oracle` (0.714); on cps the T1 gain wins. Either way, the ideal endpoint is **A's structure +
marginals imported in a structure-preserving way**.

## Interpretation → what this resets

This is the new baseline for the whole program: the faithful copula recovers a large,
free (no-B) chunk of the score that was being thrown away. It also sharpens the marginal-import
track (public re-indexing / stats): its job is to shift the outcome marginals **without disturbing
the copula** — because we now know the copula is worth ~0.10 and the swap can cost it back. The
combined target is `EC_carry`'s structure + structure-preserving target marginals, aiming past
`EC_oracle` toward the ~0.79–0.82 ceiling.

## Limitations

- **gss at reduced precision** (1 seed, B=50): its 24-outcome scoring exceeded the local
  background-run lifetime at full settings; the effect is large and directionally identical to cps,
  but the gss point estimates are noisier. A full-precision gss rerun belongs on the server.
- **Two scored time pairs**, same-instrument transfer only. Cross-context generalization (and the
  sibling-transfer model-selection guard) is the named Approach B (richer/parametric models).
- **Categorical swap** assumes a consistent frequency ordering across contexts; a category present
  in B but absent in A can't be produced (measurement non-invariance; demographics aren't scored).
- `EC_oracle` reads B's marginals — a labeled ceiling. The feasible arm is `EC_carry` (no B) plus
  the separate marginal-import track.

## Reproduce

```
.venv/bin/python scripts/transfer_empirical.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_empirical.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```
