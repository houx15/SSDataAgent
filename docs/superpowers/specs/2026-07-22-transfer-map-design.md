# Transfer map (same-country time transfer) — design

**Date:** 2026-07-22
**Roadmap:** `docs/2026-07-22-transfer-roadmap.md` — this is **Phase 1 + the LLM-free
half of Phase 2** for the same-country **time-transfer** pairs only.
**Scope decision (user, 2026-07-22):** build the time-transfer map for GSS + CPS;
no new data, no LLM, no training. Country transfer (CPS↔CFPS), B2/B3, and the learned
statistics model are explicitly **out of scope** here.

---

## 1. Goal

Produce the roadmap's **transfer map**: a table of `(context-pair × variable)` cells
labeled **composition-dominated** (the cross-context gap closes by swapping marginals —
cheap to transfer) vs **mechanism-shifted** (the gap survives a marginal swap — hard),
plus the two LLM-free no-donor baselines (**B0** naive carry-over, **B1** marginal swap)
scored against the target context. The map and the baselines validate each other.

This deliverable is **publishable on its own** (roadmap Phase 1) and is the measurement
substrate every later phase is judged against. No method work beyond B0/B1 happens here.

## 2. What already exists (and what we reuse vs build)

Two disconnected halves are already in the repo:

- **Donor-era transfer scaffolding** — `src/ssdataagent/data/transfer.py`
  (`TRANSFER_PAIRS = {"gss":"gss1994","cps":"cps1970"}`, `load_source_wave`,
  `compute_crosswalk`), transfer *conditions* in `experiments/conditions.py`, and
  completed runs in `results/ship_transfer_cheap/`. The conditions are **donor-regime**
  (e.g. `block_donor_transfer` copies real rows) and are **not reused** — they violate
  the no-donor firewall. `transfer.py`'s crosswalk + source-wave loading **is reused**
  (firewall-neutral).
- **No-donor pipeline** — `scripts/nodonor_bracket.py` (`build`, `score`, `carve_pool`),
  `src/ssdataagent/data/conditional_variance.py` (`covariate_r2`, `_dummy_design`). The
  scorer `score(sim, ds, ref, types, seed, bootstrap_B)` **recomputes every T1–T5
  statistic live from the `ref` you pass** — nothing is baked in from the fitting
  context — so transfer-mode scoring is "pass context B's reference." No scorer change.

**The one real code change** is that `build()` (bracket.py:137) uses a single `pool` for
*both* the copula/latent fit *and* the marginal inverse-CDF map. Transfer needs those two
frames split: **structure from A, marginals from B.**

## 3. Scientific design — two layers that validate each other

### Layer 1 — the diagnostic map (ground truth; NOT firewalled)

A *measurement* of where transfer is easy vs hard, computed from the **full microdata of
both A and B**. This deliberately reads B's joint — it is the truth the method is later
judged against, not a generator. Per `(pair, outcome Y)`:

- **Composition vs mechanism share** via reweighting (DiNardo–Fortin–Lemieux): rake A's
  rows so A's covariate marginals match B's, then measure how much of the A→B gap in Y
  closes. `composition_share = (gap_raw − gap_residual) / gap_raw`.
- **Copula stability** per covariate–outcome and outcome–outcome pair: is the *dependence*
  (rank association) the same in A and B, independent of marginals? This is the face-swap
  hypothesis in its cleanest form.

### Layer 2 — the firewalled baselines (scored against B's reference)

Two no-donor generators, scored with the real SSDataBench scorer against B's benchmark
sample:

- **B0 — naive carry-over:** generate a population matching **A** (structure and marginals
  from A), score against B. The floor: raw untransferred gap.
- **B1 — marginal swap:** structure (copula) from **A**, marginals from **B's public
  aggregates**. Tests pure composition transfer. **Firewalled:** reads only B's per-column
  marginals (T1-level info), never B's joint, never B's test sample.

### The throughline

`B1 − B0` is how much a marginal swap recovers. The Layer-1 map **predicts** it: where the
map says *composition-dominated*, B1 should recover most of B0's loss and approach the
within-B no-donor floor; where *mechanism-shifted*, B1 should stay near B0. If map and
baselines agree, the map is validated as a transfer predictor — the paper's core claim.

## 4. Context pairs

Same-country, same-instrument time pairs, using wave CSVs already on disk.

| pair id | source A | target B | B reference for scoring | layer |
|---|---|---|---|---|
| `gss_1994_2018` | `real_data/gss/gss1994.csv` | `real_data/gss/gss2018.csv` | benchmark `sampled_gss.csv` | 1 + 2 |
| `cps_1970_1980` | `real_data/cps/cps-asec1970.csv` | `real_data/cps/cps-asec1980.csv` | benchmark `samples_cps.csv` | 1 + 2 |
| `cps_1970_1990` | 1970 | `cps-asec1990.csv` | — | 1 only |
| `cps_1980_1990` | 1980 | 1990 | — | 1 only |
| `cps_1970_2000` | 1970 | `cps-asec2000.csv` | — | 1 only |
| `cps_1980_2000` | 1980 | 2000 | — | 1 only |
| `cps_1990_2000` | 1990 | 2000 | — | 1 only |

**Why two tiers.** Layer-2 scoring needs a benchmark reference + wired scoring config,
which only the two benchmark targets (gss2018, cps1980) have. Layer-1 diagnostics need
only two microdata frames, so the full CPS **time-distance ladder** (10/20/30-year gaps)
runs on the map even where we don't score — that ladder ("gap grows with time distance,
composition share falls") is a headline finding, free.

**Crosswalk.** Each pair restricts to variables present in both waves. GSS/CPS waves share
one instrument, so we intersect the target wave's `background_variables` + `target_variables`
with the source frame's columns (same logic as `compute_crosswalk`, but tolerant of extra
waves not in `datasets.yaml`). **Every dropped variable is logged**, the way `data_audit.py`
documents traps.

**Covariates (X) vs outcomes (Y).** `X = crosswalk ∩ schema.background_variables`;
`Y = crosswalk ∩ schema.target_variables`. Extra CPS waves use the `cps` schema's split
(same instrument).

## 5. Component designs

### 5a. `src/ssdataagent/transfer/generate.py` — the build() split

```python
def transfer_build(struct: pd.DataFrame, marg: pd.DataFrame, cols: list[str],
                   n: int, seed: int, mode: str) -> pd.DataFrame
```

Generalizes `nodonor_bracket.build`. The correlated per-row uniform latent (the copula:
shared `base` index + `glat` + `_latent` ranks + missingness *pattern*) is computed on
**`struct`**; the inverse-CDF value map (sorted numeric values / categorical `value_counts`)
and the missingness *rate* come from **`marg`**.

- `mode="carryover"` (B0): caller passes `struct == marg == A_pool`. Reduces to today's
  `build(A_pool, ..., "copula-fixed")` — verified by a test asserting frame equality.
- `mode="marginal-swap"` (B1): `struct = A_pool`, `marg = B_pool`. A's dependence, B's
  marginals. This is the face-swap.

`bracket.build()` stays **untouched** (it is the frozen no-donor replication path). Shared
primitives (`_latent`, `_is_numeric`, the inverse-CDF map) are lifted into this module;
`bracket.py` keeps its own copies. ~40 lines duplicated, isolated and independently tested
— acceptable per the project's "don't gate refactors on bit-for-bit reproduction" rule,
while not risking the frozen path.

### 5b. `src/ssdataagent/transfer/decompose.py` — composition vs mechanism

```python
def raking_weights(frame: pd.DataFrame, targets: dict[str, pd.Series],
                   covariates: list[str], *, bins: int = 10) -> np.ndarray
def kob_decompose(a: pd.DataFrame, b: pd.DataFrame, response: str,
                  covariates: list[str]) -> dict   # {composition_share, mechanism_share,
                                                    #  gap_raw, gap_residual, label, method}
def oaxaca_blinder(a, b, response, covariates) -> dict   # numeric-Y cross-check
```

- **Raking:** iterative proportional fitting of per-row weights on `a` so its weighted
  covariate marginals match `b`'s. Numeric covariates binned to `bins` quantile edges
  (edges from the pooled A∪B values so both are comparable). Reuse
  `conditional_variance._dummy_design` for the categorical encoding where helpful.
- **DFL decomposition (primary, all Y types):**
  `gap_raw = dist(P_a(Y), P_b(Y))`, `gap_residual = dist(P_a^w(Y), P_b(Y))`,
  `composition_share = clip((gap_raw − gap_residual)/gap_raw, 0, 1)`.
  `dist` = 1-Wasserstein on pooled-std-standardized values for numeric Y; total-variation
  distance for categorical Y. `label = composition-dominated if share ≥ 0.5 else
  mechanism-shifted`; `NaN` when `gap_raw < eps` (contexts already aligned on Y).
- **Oaxaca–Blinder (secondary, numeric Y only):** OLS `Y ~ dummy(X)` on A and B; twofold
  split `endowment = (X̄_b−X̄_a)·β_a`, `coefficient = X̄_b·(β_b−β_a)`;
  `composition_share_ob = |endowment|/(|endowment|+|coefficient|)`. Reported beside DFL as
  a cross-check, not the label.

### 5c. `src/ssdataagent/transfer/copula_stability.py`

```python
def pair_association(frame: pd.DataFrame, v1: str, v2: str) -> tuple[float, str]
def copula_stability(a, b, cols: list[str]) -> pd.DataFrame   # per-pair τ_a, τ_b, |Δ|, label
```

Per unordered variable pair in the crosswalk:
- both numeric/ordinal → **Kendall τ** on each context (rank-based ⇒ marginal-free ⇒ a
  clean copula probe); `stability = |τ_a − τ_b|`; also record sign agreement.
- any nominal member → **Cramér's V** on each; `stability = |V_a − V_b|`.
- `label = stable if metric < 0.10 else shifted` (threshold documented, tunable).

Aggregate to a per-pair-context mean stability. Connection: B1's T2 should pass exactly on
the *stable* pairs.

### 5d. `src/ssdataagent/transfer/pairs.py` — pair registry

```python
@dataclass(frozen=True)
class TransferPair:
    id: str; source_csv: Path; target_csv: Path
    schema_name: str            # "gss" | "cps" — X/Y split + scoring config source
    scored: bool                # True only for benchmark-backed targets
    target_dataset: str | None  # ds name to pass to score() when scored

PAIRS: list[TransferPair]       # the 7 rows of §4
def load_pair(p) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]  # A, B, crosswalk cols
```

`load_pair` reads both CSVs (via existing `_drop_unnamed`), computes the crosswalk against
the schema, logs drops. For scored pairs, B's disjoint marginal pool comes from
`carve_pool(target_dataset)` (already excludes B's benchmark rows), keeping B1 firewalled.

### 5e. `scripts/transfer_map.py` — orchestrator

CLI: `transfer_map.py [--pairs ...] [--seeds N] [--bootstrap-B 200] [--n 5000]`.

1. **Layer 1** over all pairs: for each `(pair, Y)` run `kob_decompose` (+ OB cross-check);
   for each pair run `copula_stability`. Write `results/transfer_map/map_<pair>.csv` and a
   combined `map.csv`.
2. **Layer 2** over scored pairs: B0 `transfer_build(A,A,carryover)` and B1
   `transfer_build(A, B_pool, marginal-swap)`, each `score(...)` against B's reference over
   `types=(1,2,3)`, `seeds` replicates, `bootstrap_B=200`, seeded. Also print the within-B
   no-donor floor and microdata ceiling (from `bracket.build(B_pool, independence|rowresample)`)
   as reference anchors. Write `results/transfer_map/baselines.csv`.
3. Print the map table and the baseline table; note NaN/skip reasons explicitly (no silent
   caps).

## 6. Firewall discipline (the central honesty point)

- **Layer 1 (map) is intentionally not firewalled** — it reads both contexts' microdata to
  define ground truth. The report must state this plainly: the map is the *answer key*, not
  a method.
- **Layer 2 baselines are firewalled.** B1 reads only B's **per-column marginals** (the
  same public-aggregate footing as handing over T1 tabulations) and never B's joint or B's
  test sample. B0 reads no B data at all. Each generated frame carries no B-joint provenance.
- Standing measurement discipline: seed the scorer, `bootstrap_B=200`, never quote an
  overall gap below the ~0.054 noise floor (`project_scorer_noise_floor`).

## 7. Deliverables

- `src/ssdataagent/transfer/{__init__,generate,decompose,copula_stability,pairs}.py`
- `scripts/transfer_map.py`
- `tests/test_transfer_{generate,decompose,copula_stability,pairs}.py`
- `results/transfer_map/{map.csv, map_<pair>.csv, baselines.csv}`
- `docs/report/2026-07-22-transfer-map.md` — the map, methods, composition/mechanism table,
  B0/B1 scores, map↔baseline validation, honest limits.
- `docs/experiments/LEDGER.md` row + `scripts/build_dashboard.py` rebuild (per AGENTS.md).

## 8. Testing strategy

Unit tests use small synthetic frames with **known answers**:
- `transfer_build`: (a) `carryover` mode equals `bracket.build(A, "copula-fixed")` on a
  fixed seed; (b) `marginal-swap` output's per-column marginals match `marg`, not `struct`
  (KS / value-count check); (c) a strong rank correlation injected in `struct` survives into
  the swapped output (copula preserved).
- `kob_decompose`: construct A,B where the Y-gap is 100% composition (same mechanism, X
  shifted) → `composition_share ≈ 1`; and where it is 100% mechanism (same X, β flipped) →
  `≈ 0`. OB and DFL agree on the numeric-Y linear case.
- `copula_stability`: A,B sharing a Gaussian copula with different marginals → `|Δτ| ≈ 0`
  (stable); A,B with opposite-sign dependence → large, `shifted`.
- `pairs`: crosswalk drops a source-only/target-only column and logs it; `scored` flags
  correct.

Everything is LLM-free and does not shell out to the scorer, so no `live_*` markers.
`transfer_map.py`'s real scored run is executed by the controller after the code tasks
(it needs the ssdatabench submodule + data), not asserted in a unit test.

## 9. Honesty / limits (write in the report)

- Same-country time transfer only; measurement non-invariance (country transfer) untouched.
- CPS extra-wave references (1990/2000) are Layer-1 only — no scored baseline there.
- cps/gss pools are row-disjoint (no person key), so B1's firewall is row-level, not
  person-level (`project_nodonor_regime`).
- Layer-1 map uses B's joint by design; it is not a no-donor result and must never be
  quoted as one.

## 10. Out of scope (later phases)

B2 (skeleton + aggregate recalibration / LLM), B3 (LLM prior pointed at B), country
transfer + crosswalk labor, longitudinal within-file wave slicing (CFPS/addhealth/US),
and the Phase-3 learned statistics model. This slice is the LLM-free, training-free
measurement floor they all build on.
