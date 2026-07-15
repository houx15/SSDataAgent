# Production strategies

Curated registry of the generation strategies we have **vetted and endorse**, with
their verified scores. Unlike `results/` (gitignored, every raw run), this folder
is committed and holds only what is ready.

## Promotion criteria

A strategy earns a row here only when **all** hold:

1. **In the registry, not scratchpad** — the code lives in
   `src/ssdataagent/strategies/` (or an equivalent importable module), not in a
   one-off scratchpad script.
2. **Verified over multiple seeds** against the paper's exact 1000-row reference
   (`real_data/used_dataset/sampled_cfps.csv`), with the honest protocol: training
   / aggregate stats drawn from the **disjoint pool** (`load_disjoint_train`,
   0 `pid` overlap), never tuned on the test reference.
3. **Score we stand behind** — reported with seed count and standard error, and
   reproducible from the named entry point.

Anything not meeting all three stays in `docs/report/` (findings) or scratchpad
(exploration) until it does.

## Reference points (not strategies — the goalposts)

| name | overall | note |
|---|---|---|
| PNAS best of 15 LLMs | 0.30 | the published bar |
| marginal-draw floor (no-donor) | 0.477 | draw each column from the pool marginal; zero joint |
| whole-row resample (achievable ceiling) | 0.806 ±.008 | fresh real people vs the benchmark; unbeatable by any copy-from-pool method |
| sim = reference (unreachable) | 0.913 | benchmark scored against itself; eval bootstrap noise only |

## Strategies

| strategy | regime | overall | T1 | T2 | T3 | T4 | T5 | seeds | entry point |
|---|---|---|---|---|---|---|---|---|---|
| block-donor (domain) | microdata | **0.793** ±.007 | 0.816 | 0.774 | 0.701 | 0.867 | 0.809 | 12 | `strategies/block_donor.py` (granularity=`domain`) |
| block-donor (mega) | microdata | 0.784 ±.014 | 0.829 | 0.813 | 0.698 | 0.758 | 0.821 | 12 | `strategies/block_donor.py` (granularity=`mega`) |

### Notes per strategy

- **block-donor** — copies coherent column *blocks* from covariate-matched real
  donors. Ties whole-row resampling on fidelity (the microdata regime is
  saturated) but its value is **privacy**: copy_rate 0.001 vs resampling's 1.000
  (99.7% uniquely re-identifiable). It is a diagnostic + privacy method, **not** an
  answer to PNAS's no-data question. See
  `docs/report/2026-07-15-benchmark-saturation-and-disclosure.md`.

## Candidates (not yet promoted)

| candidate | regime | overall | why not promoted yet |
|---|---|---|---|
| no-donor conditioned generator | no-donor | ~0.49 | generator pipeline lives in scratchpad (`nodonor_conditioned.py`); needs to move into the registry with a test (roadmap task 0) |
| + event-order module (A) | no-donor + aggregate ordering | **0.64** ±.03 (T4 0.70) | module is in the registry + tested (`data/event_order_knowledge.py`); the generator it wraps is still scratchpad. Report the **bracket** (0.49 knowledge-only / 0.64 +aggregate-ordering / 0.79 microdata), never 0.64 bare — T4 is aggregate-ordering access, train/test-clean but not a knowledge result. See `docs/report/2026-07-15-no-donor-event-order-result.md`. |

**Honest note on A:** knowledge alone leaves T4 at 0.000 (the LLM's ordering prior
is too diffuse); the aggregate ordering distribution from the *disjoint* pool
unlocks it (same footing as T1 calibration). On the benchmarks we supply nothing
for (T2 0.53, T3 0.32) the generator is *below* published PNAS — genuine joint
knowledge is the real gap, targeted by roadmap possibility B.

See `docs/report/2026-07-15-no-donor-roadmap.md` for the no-donor build order.
