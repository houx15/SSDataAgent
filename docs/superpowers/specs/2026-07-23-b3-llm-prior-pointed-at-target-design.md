# B3: the no-donor LLM prior pointed at the target context

**Date:** 2026-07-23
**Scope:** Phase-2 rung **B3** of `docs/2026-07-22-transfer-roadmap.md`. The final baseline
before the decision gate. Same-country **time transfer** axis as Phase 1 / B2
(GSS 1994→2018, CPS ASEC 1970→1980).
**Predecessor:** `docs/superpowers/specs/2026-07-22-b2-aggregate-recalibration-design.md`
and its report `docs/report/2026-07-23-b2-aggregate-recalibration.md`.

## The question

B1 transplants the source context's dependence structure and swaps in the target's
marginals. B2 keeps that transplanted structure and recalibrates each numeric outcome's
covariate-R² toward the target's published aggregate. Both take the *skeleton + copula*
from the **source** context A.

**B3 asks a different question: what if the structure comes from the LLM prior about the
target B instead of from A?** The roadmap's wording: *"current no-donor method pointed at
B — LLM prior prompted with B's context and codebook. Measures how much context adaptation
the prior already does for free."* B3 uses **no source context at all**. It is the existing
no-donor full method (`scripts/nodonor_fullmethod.py`) — draw demographic seeds from B's
marginals, have the LLM complete each person's downstream traits coherently, then repair
conditional variance toward an R² target — run against the **target** context and scored on
the identical footing as B0/B1/B2.

If the prior closes the residual B2 leaves, Phase 3 (learned adaptation) stays unjustified.
If a large mechanism-shift residual survives **both** B2 and B3, Phase 3 is finally earned.

## Firewall

Row-level, identical in spirit to B1/B2. B3 reads from the target **only**:

- per-column **marginals** of the disjoint pool (seeds are drawn from them; the prompt
  carries a marginal blob) — the same aggregate B1/B2 consume;
- per-outcome **covariate-R²** of the pool (the `B3_pool_R2` rung only; a labeled low-order
  aggregate, on the same footing as B2);
- the **codebook / public context** — the population name and coherence rules in the prompt.

It never reads the target's per-person joint and never reads the reference/test sample.
The elicited-R² rung reads *no* target aggregate for strength at all (R² comes from the
LLM reasoning over outcome names). Every rung is provenance-tagged. The pool is
row-disjoint, not person-disjoint (no person key), so the firewall is row-level, as in
Phase 1.

## Architecture (Approach A)

A thin new orchestrator reusing the existing durable stages; the diagnostic transfer map
stays **LLM-free / no-API-key** (a property its header advertises and the offline run
depends on).

**New / changed files:**

- **`src/ssdataagent/transfer/scoring.py`** (new) — `restrict_config_dir` and `mean_scores`
  lifted **verbatim** out of `scripts/transfer_map.py` into a tested `src` module.
  `transfer_map.py` imports them back; behavior is unchanged (parity test pins this).
- **`src/ssdataagent/transfer/b3_specs.py`** (new) — the `Spec` dataclass and per-dataset
  specs. The `cps` spec **moves here verbatim** from `nodonor_fullmethod.py`, which imports
  it back, so the durable no-donor path is untouched and B3 shares one source of truth for
  the cps prompt. The `gss` spec is new (below).
- **`scripts/transfer_b3.py`** (new) — the orchestrator. Imports the LLM stages
  (`generate`, `elicit`, `sample_seeds`, `marginal_blob`, `complete_batch`) from
  `nodonor_fullmethod.py`, the specs from `b3_specs.py`, and the scoring helpers from
  `scoring.py`.

**Data flow for one scored pair (target = B):**

```
carve_pool(B) -> b_pool                      # disjoint pool, marginals only (no-donor)
cols = crosswalk(source, B)                  # identical set B0/B1/B2 use
sample_seeds(b_pool, spec, n_people, seed=42)   # age/gender/race ~ B marginals, independent
  -> LLM completes downstream traits; prompt = B population + rules + B marginal blob
  -> cache results/nodonor_cache/<B>_cond_raw.csv     # cps warm; gss cold
elicit(numeric outcomes) -> cache <B>_elicit.json     # R^2 from outcome names only (honest)
for rung in {raw, elicited-R2, pool-R2}:
  for s in 1..seeds:
    sim = sample_variance_repaired(raw, b_pool, cols, predictors, n, rng=s, alpha=rung)
    score(sim, B, ref, types, seed=1000+s, bootstrap_B=B, config_dir=restrict_config_dir(cols))
  -> mean_scores across seeds
write results/transfer_map/b3_<pair>.csv     # pair, config(rung), guarantee, T1, T2, T3, overall
```

**Comparability guarantees.** B3 uses the identical `cols` crosswalk, the identical
`restrict_config_dir` output, the identical `ref` (`schema.real_data_path`), the identical
seed offset (`1000+s`), and the B2 publication protocol (`seeds=3`, `n=3000`,
`bootstrap_B=200`). The only thing that differs from B1/B2 is where the conditional
structure comes from. This is the exact class of comparability the previous slice got wrong
(scoring against the wrong reference); it is nailed down here by reusing the same helpers.

## The GSS 2018 Spec

`cps` moves over unchanged. `gss` is new. Key semantic findings, verified against the pool:

- **GSS `child_number` is lifetime "children ever born"** (pool mean 1.8; the stock
  `type3.yaml` description confirms it) — the **opposite** of the CPS household-roster trap,
  where `child_number` is resident children under 18. So the age relationship runs the other
  way: in GSS it *rises then plateaus* with age; in CPS it *peaks and falls*.
- GSS is **adults only** (age 18–89): no child carve-outs.
- GSS `income` is a categorical **bracket** (`$10000 OR MORE` dominates), not a dollar
  amount, so it is not numeric and gets no R² repair.

```
population = "the 2018 US General Social Survey (GSS), adults 18 and older"
seeds              = [age, gender, race]        # drawn independently from B's marginals
derived            = {}                          # no birth_year identity in the crosswalk
predictors         = [age, gender, race, education]   # mirrors the restricted gss type3.yaml
numeric_predictors = {age}
log_vars           = {}                           # income is categorical brackets
types              = (1, 2, 3)
```

**Numeric outcomes that get R² repair** — the T3 responses surviving crosswalk restriction
(`mental_health` drops: the 1994 source has no such column). `glosses` is scoped to exactly
these three, since `elicit()` sends `glosses` for R² elicitation and the repair skips
categorical outcomes anyway:

```
child_number         : "total number of children EVER BORN in the respondent's lifetime
                        (GSS lifetime fertility, NOT resident children); 0..8, mean ~1.8"
age_first_childbirth : "age at the respondent's FIRST live birth over their lifetime
                        ('No Child' if they never had a child)"
vocabulary_test      : "number of correct words (0-10) on the GSS WORDSUM vocabulary test,
                        an indicator of verbal/cognitive skill"
```

**Coherence rules (the prompt's `rules` block):**

```
- Every respondent is an ADULT (18-89). There are no children in this sample: everyone
  can have completed education, a labor-force status, and attitudes.
- FERTILITY IS LIFETIME here, the opposite of a household roster. child_number counts
  ALL children the person has EVER had, so it RISES with age and then plateaus: near 0 in
  the late teens/early 20s, climbing to ~2 by the 40s and staying there into old age. Do
  NOT let it fall for older people.
    * child_number 0 <=> age_first_childbirth 'No Child'. A nonzero count needs a real
      first-birth age, always >= 12 and < the person's own age.
    * age_first_childbirth is the LIFETIME first birth, typically 18-30, only weakly
      related to current age; more-educated people tend to start later.
- Education (Less than high school / High school / College and above) is bounded by age
  only at the young end (a 19-year-old is rarely 'College and above' yet).
- vocabulary_test (0-10) rises with education and is roughly flat in age.
- income is a categorical BRACKET ('$10000 OR MORE' dominates; 'Unemployed' when not
  earning), not a dollar amount. occupation is a broad census category; 'Unemployed' or
  'Military Occupations' are valid. spouse_occupation is 'No spouse' for the unmarried.
- Attitudes (gender_role_attitude, political_view, trust, happy, satisfy_job, work_hard)
  and health should read as one coherent person, plausible against the marginals below.
- immigrant_status and parental background (mother/father education & occupation) are
  inferred coherently, kept plausible against the marginals.
```

## Variable set is pinned to the benchmark

B3 generates **all** crosswalk columns (24 for GSS, 11 for CPS); T1 (marginals) and T2
(pairwise association) score all of them, including every attitude. The "three variables"
count is only which outcomes get the **T3 R² repair**, and the restricted T3 scores exactly
those numeric outcomes. Two invariants forbid widening the set:

1. B3 must use the **identical `cols`** as B0/B1/B2 — widening it breaks the ladder
   comparison that is B3's whole purpose.
2. The repair predictors must **mirror `type3.yaml` exactly** — adding predictors is the
   "silently mis-sizes every alpha" trap the durable script warns about.

The one genuinely missing variable is `mental_health` (a real GSS-2018 T3 outcome the
crosswalk drops because the 1994 source lacks the column) — an honest crosswalk gap, noted
in the report, not a choice. "More variables" belongs to Phase 3 (the learned statistics
model spanning more contexts), not this baseline rung.

## The three rungs and decision-gate reporting

Written to `results/transfer_map/b3_<pair>.csv`, each scored identically to B0/B1/B2:

| rung | conditional strength from | firewall role |
|---|---|---|
| `B3_raw` | LLM structure, no repair | pure prior — how good is the raw completion |
| `B3_elicited` | LLM structure + LLM-**elicited** R² | fully firewalled from target aggregates ("prior does it for free") |
| `B3_pool_R2` | LLM structure + **pool** covariate-R² | same aggregate footing as B2 — isolates *structure source* |

**Two decision-gate comparisons:**

- **`B3_pool_R2` vs `B2`** — both consume the target's pool R²; the only difference is where
  the skeleton + copula come from (LLM prior vs transplanted source A). The clean "is the
  prior a better structure than transfer?" test.
- **`B3_elicited` vs `B2`** — can the prior, given only marginals and no target R² aggregate
  at all, match aggregate recalibration? If yes, context adaptation really is "for free."

**Gate outcome.** If `B3_*` closes the residual B2 left (~60% of the B1→ceiling gap), the
paper stays a statistics+agent paper and Phase 3 is skipped. If a large mechanism-shift
residual survives both B2 and B3, the learned adaptation of Phase 3 is justified.

## Testing

LLM stages are non-deterministic, so tests target the deterministic seams; the warm cps
cache carries the one true end-to-end check.

**Unit / deterministic:**

- **Extraction parity** — `transfer_map` imports `restrict_config_dir` / `mean_scores` from
  `scoring.py`; a fixture test pins identical output (restricted config YAML + mean over a
  synthetic score frame) to pre-extraction behavior.
- **cps spec identity** — `nodonor_fullmethod.SPECS["cps"] is b3_specs.SPECS["cps"]`.
- **GSS spec invariants** — `seeds ⊆ crosswalk`; `predictors` exactly mirror the restricted
  gss `type3.yaml`; `glosses.keys()` are exactly the numeric T3 outcomes surviving
  restriction; `log_vars` empty; `income` not numeric.
- **Column restriction** — `downstream = cols − seeds − derived`; `predictors ⊆ cols`;
  nothing scored that isn't in the crosswalk.
- **Repair wiring** — on a tiny synthetic `raw` frame with fixed R² targets, the three rungs
  each return `n` rows with the crosswalk columns and finite scores (reuses tested `cv`
  functions).

**End-to-end (the real gate, per the sentinel-bug lesson):**

- **cps off the warm cache** — `cps_cond_raw.csv` + `cps_elicit.json` already exist, so
  `transfer_b3.py cps_1970_1980` runs with **zero API calls** and is fully deterministic.
  Assert it writes `b3_cps_1970_1980.csv` with all three rungs and finite scores, **and**
  that `B3_raw` reproduces the existing `nodonor_fullmethod` "raw (no repair)" number within
  noise — validating the restricted-scoring path against a known quantity. Check the output
  CSV **mtime** so a stale file cannot fake a pass.
- **GSS** needs live API (cold cache) — an execution run, not a unit test; it produces the
  cache, then re-scores deterministically.

## Replication

```bash
export OPENROUTER_API_KEY=...   # first run only; stages cache under results/nodonor_cache/
.venv/bin/python scripts/transfer_b3.py cps_1970_1980 --seeds 3 --n 3000 --bootstrap-B 200
.venv/bin/python scripts/transfer_b3.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
```

Outputs: `results/transfer_map/b3_<pair>.csv` (three rungs each). Modules:
`src/ssdataagent/transfer/{scoring,b3_specs}.py`; orchestrator `scripts/transfer_b3.py`;
reused stages `scripts/nodonor_fullmethod.py`.

## Limits (carried forward, stated in advance)

- **Only two genuine transfer pairs** (Layer-2 resolves the reference from the dataset
  name; see the B2 report). B3 is scored on exactly those two.
- **`mental_health` is absent** from GSS B3 (crosswalk gap, above).
- **Attitude variables under measurement non-invariance** (gender_role_attitude,
  political_view, trust, work_hard): the prior may transfer their marginals but their
  cross-context mechanism is exactly what no method is claimed to fix.
- **Same-country time transfer only.** No country transfer, no cross-cultural claim.
- **Row-level firewall**, not person-level (no person key in the pools).
```
