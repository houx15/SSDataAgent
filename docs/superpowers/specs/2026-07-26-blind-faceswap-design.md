# Blind face-swap — description-only cross-context generation (Approach A)

**Date:** 2026-07-26
**Branch:** `blind-faceswap` (off `main` @ ffd5470)
**Supersedes the firewall of:** the B0–B6 ladder (which read the target's public
aggregates). This is a **stricter regime**: the generator sees only a *textual
description* of the target context and never any numeric aggregate of it.

## Why this exists (the correction)

The B0–B6 ladder fed the target's public aggregates *into the generator* — T1 used
B's marginals, retrieval raked siblings to B's X-margins. That answers "given A and
B's features, reproduce B" — a calibration game. **The real target** is: given only
a *description* of a context we have no data for, can an LLM (with algorithms)
generate reliable microdata? B's real microdata becomes a **held-out yardstick used
only to score, never an input.**

This is the literal form of the project's face-swap hypothesis — **same structure,
swapped features** — and Phase 1 already measured the premise: the dependence
structure is 78–100% copula-stable across time. Approach A puts each component where
the evidence says it belongs:

- **Structure (the X→Y copula / skeleton)** — transferred from source context A,
  where it is measurably stable. (B3 showed the LLM *invents* bad dependence — T2
  below the independence floor — so the LLM must NOT own this.)
- **Features (the marginals / composition)** — **elicited from the LLM** out of B's
  textual description. (Translating a described context into per-variable
  distributions is the LLM's strength.)

## Scope of this spec

- **Situation 2 (same-instrument time transfer)**, current data: `cps 1970→1980`,
  `gss 1994→2018`. Situations 1 (cross-country) and 3 (survey merge) are out of scope
  here — 1 needs a variable crosswalk, 3 is constructed later. One method first.
- **First cut = the face-swap core** (composition ablation), scored T1–T5. Mechanism
  *deltas* (the LLM adjusting conditional strengths) are a **named follow-on**, not in
  this build.

## The firewall (stricter than every prior rung)

At generation time the generator may read:

- **B's textual context description** — population, instrument, era, variable
  semantics (glosses / coherence rules). Authored from **public / general knowledge
  only**; audited to contain **no number derived from B's sample** (see Firewall
  audit below).
- **The fixed variable schema** — the set of variables we choose to model (codebook
  names, types, categories). No free-form variable discovery in step 1.
- **Source context A** — full microdata (our existing datasets).

The generator may **not** read: B's marginals, X-margins, covariate-R², reference
sample, or any other numeric aggregate of B. B's microdata is loaded **only by the
scorer**, after generation, to compute T1–T5.

### Firewall audit (a required, explicit step)

The B3 context prose (`b3_specs.SPECS`) is the starting text but contains a few
sample-derived numbers (e.g. "mean child_number 0.66", "mean ~1.8"). Every context
description used here is **audited to strip any B-sample statistic**, keeping only
qualitative structure and public knowledge (e.g. "US lifetime fertility ~2–3 children"
is public; "pool mean 1.8" is not). The audited descriptions live in a new
`blind_specs.py` (or an audited field on the existing spec), with a one-line
provenance note per removed number. This audit is itself a firewall guarantee and is
reviewed.

## Architecture

```
description(B) + schema ──LLM──▶ elicited marginals θ̂_X, θ̂_Y   (features)
source A ─────────────────────▶ copula / shared-latent structure (structure)
                     │
                     ▼
   transfer_build(struct = A, marg = build_marg_frame(θ̂), cols, n, seed, "marginal-swap")
                     │
                     ▼
             synthetic B microdata ──▶ SSDataBench T1–T5 vs held-out B
```

The engine is the **existing** `transfer_build` (`generate.py`): `struct=A` supplies
the copula via the shared-latent draw; `marg` supplies each variable's marginal via
inverse-CDF (`_marginal_map`) and its missingness rate. Approach A's only new move is
that `marg` is built from **LLM-elicited distributions**, not from B's data.

### Components

1. **`elicit_marginals(description, schema) → {var: dist}`** (new; mirrors the
   `nodonor_fullmethod` LLM-call + durable-JSON-cache pattern, OpenRouter client,
   cached under `results/blind_cache/` — gitignored but durable, LLM-free at
   score time). Output per variable:
   - **categorical** → `{category: probability}` over the schema's categories;
   - **numeric** → a small set of **quantiles** (e.g. deciles) so the LLM can express
     level and skew.
   The prompt carries the audited description + the variable glosses/rules + the
   schema; it asks ONLY for marginal distributions (never a joint, never per-person
   rows).
2. **`build_marg_frame({var: dist}, source_A, cols) → pd.DataFrame`** (new; pure).
   Synthesizes a representative column per variable whose empirical distribution
   matches the elicited `dist`: categorical → categories repeated to the elicited
   proportions; numeric → values interpolated across the elicited quantiles. Each
   column's **NaN rate is set from source A** (missingness is a design/structure
   property, transferred — not something the LLM is asked to guess).
3. **Assembly + scoring** — `transfer_build(A, marg_frame, cols, n, seed,
   "marginal-swap")` → score with the existing `nodonor_bracket` scorer, restricted
   config, same protocol as B0–B6.

## Configs (the ablation)

All scored T1–T5 on both pairs, 3 seeds, `n=3000`, `bootstrap_B=200`.

| config | copula | marginals | reads B? | role |
|---|---|---|---|---|
| **`FS_carryover`** | A | **A's own** | no | blind floor — no B info at all (`transfer_build` "carryover") |
| **`FS_llm`** | A | **LLM-elicited from B's description** | no (description only) | **the face-swap — our method** |
| `ref_oracle_comp` | A | B's *true* marginals | yes (upper bound only) | = B1; the composition-given ceiling for this axis |
| `ref_floor` / `ref_ceiling` | — | — | — | independence floor / microdata ceiling (from the bracket) |

What the cells decide:

- **`FS_llm` vs `FS_carryover`** — does LLM-elicited composition add real signal over
  just carrying A's composition? (Isolates the LLM's value on the *features* axis.)
- **`FS_llm` vs `ref_oracle_comp` (B1)** — the exact **price of blind composition**:
  how much score is lost by eliciting marginals instead of being handed them.
- **`FS_llm` vs `ref_floor`/`ref_ceiling`** — where a truly-blind generator lands in
  the achievable range.

## Evaluation & expected reading

- The scientific question is **reliability**: does `FS_llm` stay close to the
  composition-given `ref_oracle_comp`, or does blind elicitation collapse it? A small
  `FS_llm` → `ref_oracle_comp` gap means "the LLM reliably supplies composition from a
  description"; a large gap localizes the failure to the features axis (→ motivates a
  learned composition model or public census margins).
- **Per-type diagnosis:** T1 (marginals) directly grades the elicited features; T2/T3
  grade the transferred structure. Because the structure is identical to B1's, any
  T2/T3 gap between `FS_llm` and `ref_oracle_comp` is a *second-order* effect of wrong
  marginals feeding the copula, not a structure change — reported as such.

## Known evaluation subtlety (flagged, not resolved here)

Some SSDataBench T2/T3 configs mark `age/gender/race` as `input:true` — the scorer
*hands* the system those demographics. That partly conflicts with "fully blind." This
spec **runs the benchmark as-is** (so scores stay comparable to the B0–B6 ladder) and
reports where the given-demographics regime applies; whether to run a fully-blind
variant is a later decision, per situation.

## Follow-on (NOT in this build)

- **`FS_llm_delta`** — the LLM also emits X→Y *strength deltas* (e.g. R² adjustments)
  from the description, applied via the existing `bidirectional_r2_blend`. This is the
  mechanism-shift channel; added only if `FS_llm`'s T2/T3 show a systematic,
  LLM-correctable gap.
- **Learned bias-correction** across A-contexts (the roadmap's residual idea, honestly
  held-out) — deferred until an LLM bias worth correcting is measured.

## Limitations

- **LLM-dependent & stochastic.** Elicitation is cached for reproducibility; the model
  and prompt are pinned and recorded. A different model may move `FS_llm`.
- **Two scored pairs**, same-instrument time transfer only. No cross-country or fusion
  yet.
- **Composition only in this build.** Mechanism shift is transferred-as-is (A's
  structure); the delta channel that would adapt it is a follow-on.
- **Numeric marginals via quantiles** are an approximation of the true shape; the
  fidelity ceiling is `ref_oracle_comp`, reported alongside.

## Reproduce

```
export OPENROUTER_API_KEY=...        # first run only; elicitation is cached
.venv/bin/python scripts/transfer_faceswap.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_faceswap.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```
