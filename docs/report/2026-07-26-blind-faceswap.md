# Blind face-swap — description-only cross-context generation (Approach A)

**Date:** 2026-07-26
**Branch:** `blind-faceswap` (off `main` @ ffd5470)
**Spec:** `docs/superpowers/specs/2026-07-26-blind-faceswap-design.md`
**Plan:** `docs/superpowers/plans/2026-07-26-blind-faceswap.md`

## What this is

The first result in the **truly-blind regime**: the generator sees only a *textual
description* of the target context B (population + variable definitions) and **never
any numeric aggregate of B**. It transfers source context A's copula structure and
gets B's marginals ("features") from an LLM (`anthropic/claude-sonnet-4.5`) that reads
only B's audited description. B's real microdata is a held-out yardstick used only to
score. This is the literal form of the face-swap hypothesis — **same structure,
swapped features** — under the strictest firewall the project has run.

**Configs** (3 seeds, n=3000, bootstrap_B=200 — identical protocol/scorer to B0–B6):

- **`FS_carryover`** — A's copula + A's own marginals (blind; reads no B at all).
- **`FS_llm`** — A's copula + **LLM-elicited** marginals from B's description (the
  face-swap; our method).
- **`ref_oracle_comp`** — A's copula + **B's true marginals** (== the ladder's B1;
  reads B — a labeled *upper bound* for the composition axis, not our method).
- **`ref_floor` / `ref_ceiling`** — within-target independence floor / microdata ceiling.

The LLM elicits marginals only (never the joint); category labels are constrained to
A's codebook universe; the descriptions were firewall-audited to remove every
target-sample statistic (see spec "Firewall audit").

## Results

### cps_1970_1980 (floor 0.413, ceiling 0.816)

| config | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| `FS_carryover` (==B0) | 0.401 | 0.619 | 0.656 | **0.558** |
| **`FS_llm`** | 0.427 | 0.587 | 0.682 | **0.565** |
| `ref_oracle_comp` (==B1) | 0.810 | 0.554 | 0.573 | **0.646** |

### gss_1994_2018 (floor 0.508, ceiling 0.811)

| config | T1 | T2 | T3 | overall |
|---|---|---|---|---|
| `FS_carryover` (==B0) | 0.395 | 0.821 | 0.619 | **0.612** |
| **`FS_llm`** | 0.399 | 0.823 | 0.586 | **0.603** |
| `ref_oracle_comp` (==B1) | 0.641 | 0.816 | 0.494 | **0.651** |

**Comparability confirmed (bit-identical):** `FS_carryover` reproduces the ladder's
`B0_carryover` (cps 0.558, gss 0.612) and `ref_oracle_comp` reproduces `B1_marginal_swap`
(cps 0.646, gss 0.651) exactly — the blind pipeline is scored on the same variables,
seeds, and scorer as B0–B6.

## Findings

### 1. The transferred *structure* works; blind *composition* is the bottleneck — entirely in T1.

On both pairs `FS_llm`'s conditional structure is healthy: **T2/T3 match or beat the
oracle-composition B1** (cps T3 0.682 > 0.573; gss T3 0.586 > 0.494; T2 comparable).
The whole shortfall is **T1 (marginals)**: `FS_llm` T1 is **0.427 (cps) / 0.399 (gss)**
against oracle-composition's **0.810 / 0.641**. The LLM cannot estimate the target's
demographic marginals accurately enough from a description — the copula transfers, the
composition does not.

### 2. LLM-elicited composition is no better than a naive source carry-over.

`FS_llm` ≈ `FS_carryover` on both pairs (cps **0.565 vs 0.558**, +0.007; gss **0.603 vs
0.612**, −0.009 — both inside the ~0.054 noise floor). Eliciting 1980/2018 marginals
from the LLM buys essentially nothing over just reusing the 1970/1994 source marginals:
the LLM's blind demographic estimates are about as far from B's truth as the previous
wave is. The "features" half of the face-swap does not yet earn its LLM call.

### 3. The blind method still beats independence, but falls short of knowing B's marginals.

`FS_llm` clears the within-target independence floor on both pairs (cps 0.565 vs 0.413;
gss 0.603 vs 0.508) — the copula borrowed from another wave is worth more than
independence even when the marginals are blind. But it lands **below oracle-composition
B1** (cps −0.081, gss −0.048): the price of "knowing nothing about B" is real and, per
Finding 1, is paid almost entirely on the demographic marginals.

### 4. Encouraging qualitative signals in the elicited marginals.

The LLM read the *semantics* correctly from the descriptions alone: it distinguished the
two fertility definitions (cps `child_number` median 0 = household-resident children;
gss median 2 = lifetime children ever born) and inferred gss's modal income bracket
without being told (that fact had been scrubbed from the description as a sample
statistic). The failure is not semantic understanding but **quantitative accuracy** of
the demographic composition.

## Interpretation → what this points to

The mechanism/copula is reliably transferable from a description-only starting point;
the demographic **composition** is the weak link, and the LLM does not beat a naive
carry-over on it. Crucially, **the bottleneck (X marginals) is exactly the part that is
genuinely public** — census demographic margins exist for almost any (country, year).
So the sharpest next step is the regime that *admits public X-margins* (description +
census marginals): ground the composition in public data — where the LLM is weak and the
data is cheap — and let the transfer + LLM carry the mechanism, which already works. The
alternative within the pure-blind regime is a *better composition estimator* (richer
elicitation, or a small learned composition model across contexts); on this evidence a
single-shot LLM elicitation is not it.

## Limitations

- **LLM-dependent & single-shot.** One model (`anthropic/claude-sonnet-4.5`), one
  elicitation call per context, cached for reproducibility. A different model/prompt may
  move `FS_llm`; the finding "blind composition ≈ carry-over" is what to re-test.
- **Two scored pairs**, same-instrument time transfer only (situations 1/3 not covered).
- **Numeric marginals via quantiles** approximate the true shape; and composition is
  elicited as independent marginals (no elicited X cross-tabs), which caps T1.
- **Cache caveat:** the elicitation cache key is `<ds>_marginals.json` and does **not**
  hash the prompt/description — edit a description and re-run with `regenerate=True`, or
  the stale cache is served.

## Reproduce

```
export OPENROUTER_API_KEY=...        # first run only; elicitation is cached durably
.venv/bin/python scripts/transfer_faceswap.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_faceswap.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```

Heavy scoring is reaped on the box; the numbers here were produced with a resumable
per-(config,seed) scorer (`.superpowers/sdd/faceswap_incremental.py`, numerically
identical to `run_faceswap`, reading the durable elicitation cache so scoring needs no
API key).
