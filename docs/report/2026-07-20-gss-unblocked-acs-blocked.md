# gss unblocked (same-year wave), acs blocked (no population)

**Date:** 2026-07-20
**Scope:** gss, acs. No test data used.
**Code landed:** `b818210` (gss wired into `nodonor_bracket.py`).
**Replication:** `scripts/nodonor_bracket.py gss --bootstrap-B 200`; `scripts/data_audit.py gss`.

## Summary

- **gss is unblocked.** The standing blocker was the wrong source wave. The same-year
  `gss2018.csv` (2,348 rows) — the wave the 1,000-row reference was drawn from — has every
  reference column including `wealth`/`mental_health`, giving a 1,348-row row-disjoint pool.
- **acs is genuinely blocked**, confirmed by inspection: the only "full" file is the same
  1,000-person sample as the reference, not a population. It needs real ACS microdata.
- gss's no-donor **floor is much higher than cps's (0.546 vs 0.407)**, entirely on T2 —
  its attitudinal variables have weak pairwise associations, so independence loses little.

## gss: the blocker was a stale wave, again

Recorded as blocked because the `gss1994` transfer wave lacks `wealth` (T1/T2) and
`mental_health` (T3). But the benchmark's reference is the **2018** wave (`gss_2018` scoring
config), and `real_data/gss/gss2018.csv` was sitting unused — the same pattern as the cps
stale-1970 fix. It carries all 33 reference columns with real data (`wealth` 56% non-null,
`mental_health` 60%), and 100% of reference rows match, so a row-disjoint carve yields a
1,348-row pool. Wired as gss's `FULL_SOURCE`.

**Caveat:** the pool is small (1,348 rows), and gss has no person key, so it is
`row-disjoint`, not `person-disjoint` — the microdata ceiling resamples a small pool with
repetition. The numbers are sound but the ceiling is less independent than cfps's 57k-row
`pid`-disjoint pool.

### Bracket (B=200, seeded, 5 seeds)

| config | T1 | T2 | T3 | overall | regime |
|---|---|---|---|---|---|
| independence (floor) | 0.805 | 0.689 | 0.142 | **0.546** ±.005 | no-donor |
| copula-old (buggy) | 0.810 | 0.784 | 0.398 | 0.664 ±.006 | microdata |
| copula-fixed | 0.663 | 0.839 | 0.543 | 0.682 ±.026 | microdata |
| **rowresample (ceiling)** | 0.802 | 0.893 | 0.742 | **0.812** ±.007 | microdata |

The bracket is well-ordered (adjacent gaps +0.118, +0.018, +0.130), so the structure is
sound even on the small pool. `copula-fixed` T1 dips to 0.663 — the person-linked
missingness top-up is noisier on 1,348 rows than on cps's 180k.

### gss vs cps — the floor is where they differ

Both are cross-sectional (T1–T3 only), so their overalls are directly comparable:

| | floor (no-donor) | ceiling (microdata) | T2 floor | T3 floor |
|---|---|---|---|---|
| cps | 0.407 | 0.816 | 0.353 | 0.004 |
| gss | **0.546** | 0.812 | **0.689** | 0.142 |

The ceilings match (~0.81), but gss's **floor is 0.14 higher**, almost entirely from T2
(0.689 vs 0.353). gss is dominated by attitudes (religion, political view, trust, happiness,
job satisfaction) whose pairwise associations are genuinely weak, so drawing every column
independently forfeits little — the joint carries less of the signal than in cps, where
demographic→SES couplings are strong. This is a property of the *data*, not the method: the
no-donor floor is high exactly where the real joint is close to independent.

### Audit runs clean on gss

`scripts/data_audit.py gss` (test-blind) flags the `age_first_childbirth` `No Child`
sentinel and the `age + birth_year = 2018` identity — the same two corrections cps needed.
Notably it does **not** flag gss `child_number`: GSS asks lifetime births (CHILDS), which
rises monotonically with age (0.34 → 2.47, mean 1.84), so the cumulative-monotonicity check
correctly clears it — the same check that flags cps's household-roster `child_number`
(mean 0.66, falls to 0.16). Same variable name, opposite semantics, both judged correctly
from pool data alone.

## acs: a data wall, not a wiring gap

The only candidate full source is `acs_clean.csv`, and it is **1,000 rows — the same sample
as the reference**: identical means and SDs on age/income/child_number, and pooling the two
1,000-row files yields 1,665 unique rows (no meaningful set of additional people). A
row-disjoint carve leaves ~0 pool. acs cannot be run in any pool-based regime here.

Unblocking acs requires real ACS microdata — public via IPUMS-USA for the same year and
geography as the benchmark's `acs_1980` reference. Until that is added, acs stays out of
the bracket and the no-donor method.

## Status after this

| dataset | pool | bracket | note |
|---|---|---|---|
| cfps | 57,474 person-disjoint | ✅ | full 5-type; the headline dataset |
| cps | 180,488 row-disjoint | ✅ | fertility-semantics fixed |
| addhealth | 5,504 person-disjoint | ✅ | T4 unwinnable (real data scores 0.000) |
| **gss** | **1,348 row-disjoint** | ✅ | **unblocked here** |
| acs | — | ❌ | blocked pending IPUMS microdata |

Next for gss: the *full method* (LLM conditional generation + variance repair) needs a gss
`Spec` in `nodonor_fullmethod.py` and API calls — the online follow-on. The bracket above is
the LLM-free reference it will be positioned against.
