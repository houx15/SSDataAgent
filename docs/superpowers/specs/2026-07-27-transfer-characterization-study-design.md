# Transfer characterization study — how heterogeneous are contexts, and why?

**Date:** 2026-07-27
**Branch:** `transfer-characterization` (off `main` @ b7f70a0)
**Kind:** analyst-side **characterization** (measurement, not a generator). Reads A *and*
B microdata freely — this is separate from, and does not touch, the generator firewall.

## Why this exists

The blind face-swap localized the error (T1 / composition) but did not tell us *why
composition matters* or *how fixable the mechanism is*. Before building more modeling
machinery (census X-margins, a mechanism-delta channel, a learned composition model), we
measure the transfer problem directly on the real data and produce statistics +
visualizations that **speak the conclusions plainly**. The study answers five questions
the user posed:

1. **Y-heterogeneity split** — of the Y difference between two contexts, how much is
   *different-X* (composition) vs *same-X-different-Y* (mechanism)?
2. **X-composition distance** — how different is the demographic composition between
   contexts?
3. **Mechanism difference** — how different are the X→Y association structures?
4. **Shape vs level** — when the mechanism *does* move, does the conditional curve shift
   in **level** (fixable by a cheap offset) or change **shape** (needs real adaptation)?
5. **Learned composition model** — is a small model that captures composition feasible?
   *(Characterized, not built — see "Q5 disposition".)*

## Firewall note (important, and settled)

Understanding the data is a different activity from generating it. The generator firewall
binds the *generator* (it may never read B's aggregates). This study is us, as analysts,
characterizing difficulty — it reads both contexts' microdata by design. Any method the
study points to still runs blind at generation time. No firewall concern here.

## Scope

- **Datasets:** cps, gss, cfps.
- **Situations covered:** situation 2 (period / time) and the group (subgroup) contrast.
  Cross-country cps-vs-cfps (situation 1) is **out of v1** — it needs a variable crosswalk.
- **X / Y** come from the schema: `X = background_variables`, `Y = target_variables`
  (via `load_schema`). `birth_year` stays dropped (`pairs.NON_TRANSFERABLE`).

## Contexts and pair families

A **context** is a labeled row-subset: `(dataset, csv, label, group_filter?)` where
`group_filter` is an optional `(column, value)` restriction. A **pair** is
`(context_A, context_B)` tagged with a `family`.

### Time / period family (repeated cross-sections)

| pair id | A | B |
|---|---|---|
| `cps_1970_1980` | cps 1970 | cps 1980 |
| `cps_1980_1990` | cps 1980 | cps 1990 |
| `cps_1990_2000` | cps 1990 | cps 2000 |
| `cps_1970_2000` | cps 1970 | cps 2000 |
| `gss_1994_2018` | gss 1994 | gss 2018 |

**CFPS is excluded from the time family** — it is a single life-course panel (per-age
income/education/occupation sequences), not repeated cross-sections, so it has no clean
period axis. This is stated in the report, not faked.

### Group family (ethnicity: minority vs majority; A = majority, B = minority)

Race coding is **not wave-invariant** (cps1970/gss1994 = `White/Black/Other`;
cps1980/gss2018 = `Non-Black,Non-Hispanic / Black / Hispanic`). The one category stable
across every US wave is **Black**, so the universal split is **majority vs minority**:

| pair id | A (majority) | B (minority) |
|---|---|---|
| `cps_1980_race` | cps 1980, `race != "Black"` | cps 1980, `race == "Black"` |
| `gss_2018_race` | gss 2018, `race != "Black"` | gss 2018, `race == "Black"` |
| `cfps_minzu` | cfps, `minzu == "han"` | cfps, `minzu == "minority"` |

CFPS `minzu` NaN rows (≈8k) are dropped from its group contrast. Group sizes are all
adequate (Black: cps ≈18k, gss ≈385; minority ≈4.4k).

v1 uses one representative wave per US dataset; adding more waves is a one-line registry
edit and is left as a noted extension.

## Per-schema constants

Raking on many high-cardinality margins concentrates weight (see `decompose.raking_weights`
/ ESS), so Q1 rakes on a small **core** set; Q2 reports *every* X.

| schema | core demographics (raking, Q1) | focal covariate (Q4) |
|---|---|---|
| cps | `age, gender, race` | `age` |
| gss | `age, gender, race` | `age` |
| cfps | `gender, sib_number` | `birth_year` |

For a **group** pair the grouping variable is removed from **both** the Q1 core **and** the
Q2 X-sweep: it is constant within each subgroup, so raking on it is a no-op and its Q2
marginal distance would be 1.0 by construction (it *is* the split, not a finding).
`cps/gss` core reuses the existing `transfer_map.CORE_DEMOGRAPHICS = ("age","gender","race")`.

## Metrics → questions

All reuse existing, tested primitives where they exist. New code is thin.

- **Q1 — composition vs mechanism.** `decompose.kob_decompose(a, b, response=y, covariates=core)`
  per Y → `composition_share`, `mechanism_share`, `ess_ratio`, `label`. Numeric Y also gets
  `decompose.oaxaca_blinder` as a cross-check (`endowment`/`coefficient`/`composition_share_ob`).
  **Headline figure:** distribution of `composition_share` across all Y, faceted by family
  (time vs group) and dataset.
- **Q2 — X-composition distance.** New pure helper `marginal_distance(a_col, b_col)`:
  numeric → standardized 1-Wasserstein (divide by pooled SD), categorical → total-variation
  (½ Σ|p−q|), NaN bucketed as its own category. Computed for every `X` (for group pairs,
  excluding the grouping column — see "Per-schema constants"). **Figure:** per-pair
  composition-gap bars per X.
- **Q3 — mechanism difference.** `copula_stability.copula_stability(a, b, cols)` over the
  crosswalk columns → summarize `%stable / %shifted / %undefined` and `median(abs_delta)`.
  **Figure:** stability bars per pair; the association-shift table backs it.
- **Q4 — shape vs level (numeric Y only).** New pure helper
  `shape_level_split(a, b, response, focal, *, bins=10)`: bin `focal` on pooled quantile
  edges; per bin compute the conditional-mean gap `g(x)=E_B[Y|x]−E_A[Y|x]`; then
  `level = mean_x g(x)`, `shape = rms_x(g(x) − level)`,
  `shape_ratio = shape / (|level| + shape + eps)`. `shape_ratio ≈ 0` ⇒ pure level shift
  (a cheap offset fixes it); `≈ 1` ⇒ the gradient itself changed. Categorical Y are out of
  Q4's scope (they are covered by Q3). **Figure:** level vs shape stacked bars per numeric Y.
- **Q5 — learned composition model (disposition, not build).** The study's by-product is
  the corpus of ~8 context-pairs with their composition gaps and ESS. The report includes a
  short, honest note: with ~8 pairs we remain below the roadmap's ≥8–10-*context* threshold
  for learning transport in any stable way, so a learned composition model stays a
  corpus-gated follow-on; the near-term composition fix is public census X-margins. No model
  is trained here.

## Architecture

```
config/registry ──▶ CONTEXTS (row-subsets) ──▶ PAIRS (family-tagged)
                              │
        per pair: load A, B, crosswalk cols, core, focal
                              │
   ┌──────────────┬───────────┴───────────┬──────────────┐
  Q1 kob/oaxaca  Q2 marginal_distance   Q3 copula_stab   Q4 shape_level_split
   └──────────────┴───────────┬───────────┴──────────────┘
                              ▼
              tidy long-format results DataFrame
                              │
              ┌───────────────┴───────────────┐
        results/characterization/*.csv    docs/report/*.html (base64 figs)
```

### Components (files)

1. **`src/ssdataagent/transfer/characterize.py`** (new). Pure/analytical core:
   - `@dataclass Context(dataset, csv, label, group_col=None, group_val=None)` and
     `load_context(ctx) -> pd.DataFrame` (reads csv, drops `Unnamed:`, applies the group
     filter). Reuses `pairs._drop_unnamed`.
   - `@dataclass Pair(id, family, a: Context, b: Context, schema_name)`.
   - `CONTEXTS` / `PAIRS` registries encoding the tables above; `CORE_DEMOGRAPHICS` and
     `FOCAL` per schema.
   - `marginal_distance(a_col, b_col) -> (distance, kind)` — pure.
   - `shape_level_split(a, b, response, focal, *, bins=10) -> dict` — pure.
   - `pair_records(pair) -> list[dict]` — runs Q1–Q4 for one pair, emitting tidy rows
     `{pair, family, dataset, question, metric, key, value, extra...}`.
   - `run_characterization(pairs=PAIRS) -> pd.DataFrame` — concatenates all `pair_records`.
2. **`scripts/characterize.py`** (new). CLI: `run_characterization()` → write tidy CSV to
   `results/characterization/characterization.csv` (gitignored) and a committed copy to
   `docs/report/2026-07-27-characterization-data.csv`.
3. **`scripts/characterize_report.py`** (new). Reads the tidy CSV, renders the Q1–Q4
   matplotlib figures, embeds them as base64 `<img>` in one **self-contained HTML** at
   `docs/report/2026-07-27-transfer-characterization.html` (git-pull-and-open, like the
   dashboard). No external assets.

`decompose._is_num` is the single numeric/categorical authority (import it) so Q1/Q2/Q4
agree on which columns are numeric.

## Testing (TDD)

- `marginal_distance`: identical columns → 0; disjoint categoricals → 1.0 (TV); a known
  numeric shift → expected standardized Wasserstein.
- `shape_level_split`: constructed data with a pure vertical offset → `shape_ratio ≈ 0`;
  data with a pure slope change and zero mean gap → `level ≈ 0`, `shape_ratio ≈ 1`.
- `load_context`: a group filter yields only rows with the given value and the expected n;
  `Unnamed:` columns are dropped.
- Group-pair covariate honesty: for a group pair, the core passed to `kob_decompose`
  excludes the grouping column.
- `run_characterization` on a tiny 2-pair synthetic registry returns the tidy schema
  (expected columns, one row per (pair, question, key)) with finite values.
- Real-data smoke (marked slow / optional): `PAIRS` all load and `pair_records` returns
  non-empty for each without raising.

## Deliverable

- `docs/report/2026-07-27-transfer-characterization.html` — the self-contained report
  (committed).
- `docs/report/2026-07-27-characterization-data.csv` — the tidy table behind every figure
  (committed, small).
- A short findings section in the report answering Q1–Q4 in plain language, plus the Q5
  disposition note.

## Limitations (stated up front in the report)

- **n = ~8 pairs.** Every "how different across contexts" statement is a handful of data
  points — it characterizes *these* transfers, not a distribution over contexts.
- **Marginal raking (Q1).** `kob_decompose` matches covariate *marginals*, not the joint;
  a composition difference living purely in covariate interactions is attributed to
  mechanism (documented in `decompose.kob_decompose`). The copula map (Q3) is the
  complementary probe.
- **Q4 is numeric-Y only** and uses a single focal covariate; it is a directional
  shape-vs-level read, not a full functional decomposition.
- **CFPS** contributes only a group contrast (no period axis) and uses a smaller raking
  core (no `age`), so its ESS/shares are not directly comparable in magnitude to the US
  period pairs — compared qualitatively.

## Reproduce

```
.venv/bin/python scripts/characterize.py
.venv/bin/python scripts/characterize_report.py
open docs/report/2026-07-27-transfer-characterization.html
```
