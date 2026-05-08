# Run EXP-006e on the cloud box

> **For the user.** What to run on the GCP box once you've pulled the EXP-006e
> commit. EXP-006e tests whether **hard-gating `commit_generator` on
> `score_event_order`** (plus a corrected family recipe for many-valued
> event-age targets) closes the T4 gap that EXP-006c left open. Local smoke
> on nlsy already confirmed the gate fires and the agent self-corrects —
> see "Local smoke result" below.

## What changed since EXP-006c

- `commit.py` — chains with ≥2 event-age columns refuse commit until
  `score_event_order` has covered ≥2 of them. The error message names the
  detected event-age cols so the agent self-corrects without guessing.
- `state.py` — `RuntimeState.event_order_calls` audit trail, populated by
  successful `score_event_order` calls only.
- `verify.py` — `score_overall` adds a `chronology_hint` field when the
  chain has event-age cols, telling the agent that score_overall is
  T1-only and pointing at score_event_order.
- `prompt_templates.py` — `rubric_tools_v3` variant. Two changes vs v2:
  (a) **family recipe corrected**: many-valued event-age targets use
  `empirical_lookup` conditioned on prior events, NOT `linear_regression`.
  v2's regression preference is what caused nlsy T3 to regress -0.196.
  (b) **gate surfaced**: recipe explicitly says commit will be refused
  without score_event_order, so the agent doesn't loop.

## Local smoke result (already done — context only)

Ran `smoke_nlsy_tools_v3` (n=100) on the laptop. Sequence observed:

```
turn 15: commit_generator           → ERROR: missing_event_order_check
turn 16: score_event_order events=  → compliance_rate=0.52 (failed threshold,
         [age_finished_education,     but logged in the audit trail)
          age_started_work]
turn 18: commit_generator           → committed=True
```

Final chain has every event-age column fit as `empirical_lookup`
conditional on the prior event-age + demographics — exactly what the
recipe tells the agent to do.

T4 on n=100 was only 0.033 (small-sample noise; empirical_lookup tables
are sparse with ~50 train rows). The cloud run with n=1000 is where T4
should actually move.

## On the cloud box

```bash
cd ~/SSDataAgent
git pull                              # picks up rubric_tools_v3 + new yaml + commit.py gate
conda activate ssda
pip install -r requirements.txt       # noop unless deps changed
```

The submodule is unchanged.

## Smoke first (~$0.05, 2 min)

The local smoke already confirmed the gate works on nlsy. The cloud smoke
is just to confirm the deployed image matches.

```bash
tmux new -s ssda
conda activate ssda
python scripts/run_experiment.py --experiment smoke_nlsy_tools_v3 | tee smoke_v3.log
```

Sanity check before main run:

```bash
python -c "
import json, os
base = 'results/smoke_nlsy_tools_v3/full_agent/nlsy'
sub = sorted(os.listdir(base))[-1]
tc = json.load(open(f'{base}/{sub}/workspace/tool_calls.json'))
seo = [c for c in tc if c.get('tool') == 'score_event_order' and 'compliance_rate' in (c.get('result') or {})]
print(f'score_event_order successful calls: {len(seo)}')
assert len(seo) >= 1, 'gate did not fire — investigate'
print('OK: gate fired.')
"
```

If that prints `OK: gate fired.`, proceed.

## Main run — EXP-006e (~12-15 min, ~$5-8)

```bash
tmux new -s exp006e
conda activate ssda
python scripts/run_batch.py exp006e_tools_long exp006e_tools_cross | tee batch_006e.log

# Detach: Ctrl-b d  — disconnect SSH freely; the box keeps running.
# Re-attach: tmux attach -t exp006e
```

Order matters: `exp006e_tools_long` is the load-bearing run (where the
gate fix applies). `exp006e_tools_cross` is a re-test of the v2 gss/cps
T3 wins to make sure the v3 family-recipe correction doesn't regress
them.

Check status from a separate shell:

```bash
conda activate ssda
python scripts/status.py exp006e_tools_long exp006e_tools_cross
tail -f results/exp006e_tools_long/run.log
```

When all `✓ done`:

```bash
python scripts/generate_exp_report.py exp006e_tools_long --baseline pilot_paper_longitudinal_gpt54
python scripts/generate_exp_report.py exp006e_tools_cross --baseline pilot_paper_agents_gpt54
```

## What to look for

**vs EXP-006c (v2 → v3, longitudinal):**

| Dataset | EXP-006c T4 | EXP-006e target |
|---|---:|---:|
| nlsy    | 0.022 | ≥ 0.10 |
| addhealth | 0.000 | ≥ 0.05 |
| cfps    | 0.000 | ≥ 0.05 |
| us      | 0.230 | ≥ 0.20 (hold the line) |

**vs EXP-006c (v2 → v3, longitudinal T3 — regression check):**

| Dataset | EXP-006c T3 | EXP-006e expected |
|---|---:|---:|
| nlsy    | 0.298 | ≥ 0.45 (recover toward EXP-006b's 0.494) |
| addhealth | 0.282 | ≥ 0.28 (no regression) |
| cfps    | —     | — |
| us      | 0.291 | ≥ 0.29 (no regression) |

**vs EXP-006c (v2 → v3, cross-sectional — sanity):**

The family-recipe correction only changed advice for many-valued
event-age targets. Cross-sectional T3 wins should hold:

| Dataset | EXP-006c T3 | EXP-006e expected |
|---|---:|---:|
| gss | 0.485 | ≥ 0.45 |
| cps | 0.537 | ≥ 0.50 |
| acs | 0.025 | ≥ 0.02 (won't fix here — needs missingness work) |

If long T4 rises and cross T3 holds, the gate + recipe-correction
worked and we can deprecate v2. If long T4 doesn't move even with the
gate, the bottleneck is the empirical_lookup table sparsity — queue an
EXP-006f that swaps to a parametric event-time model.

## Pulling results back to the laptop

```bash
# from your laptop, inside the project dir:
rsync -av --progress \
    user@cloud-box:~/SSDataAgent/results/exp006e_tools_long/ \
    results/exp006e_tools_long/
rsync -av --progress \
    user@cloud-box:~/SSDataAgent/results/exp006e_tools_cross/ \
    results/exp006e_tools_cross/
rsync -av --progress \
    user@cloud-box:~/SSDataAgent/docs/experiments/ \
    docs/experiments/

git add docs/experiments/2026-05-*-exp006e_*.md
git commit -m "EXP-006e results"
git push
```

## Pitfalls

- **Don't run `smoke_nlsy_tools_v2` or v1** — they're the un-gated prompts.
- **Costs roughly:** smoke ≈ $0.05; long batch ≈ $4-5; cross batch ≈ $1.50.
- **gpt-5.4 stochasticity is large.** As before — single-run T4 can swing
  ±0.05. If a single dataset looks bad, re-run once before concluding.
