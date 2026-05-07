# Run EXP-006c on the cloud box

> **For the user.** What to run on the GCP box once you've pulled `a988000…`
> or later. EXP-006c tests whether a tighter system prompt (family-selection
> recipe + longitudinal chronology recipe) closes the T3/T4 gap that EXP-006b
> showed. EXP-006d (optional) is the no_semantic / no_data ablation on the
> same prompt.

## On the cloud box

```bash
cd ~/SSDataAgent
git pull                        # picks up rubric_tools_v2 + new yaml
.venv/bin/pip install -r requirements.txt   # noop unless deps changed
```

The submodule is unchanged, no need to re-init.

## Smoke first (2 min, ~$0.05)

A quick end-to-end check that `rubric_tools_v2` runs cleanly before spending
the full ~$5–10 on the ACS+long batches. Same shape as Stage A's smoke.

```bash
nohup .venv/bin/python scripts/run_experiment.py \
    --experiment smoke_acs_tools_v2 \
    > smoke_v2.log 2>&1 &

tail -f smoke_v2.log              # watch turns; Ctrl-C tail when DONE
.venv/bin/python scripts/status.py smoke_acs_tools_v2
```

If smoke shows `done.flag` and an `overall` ≥ 0.30 in summary.csv, proceed.
If it crashed, check `results/smoke_acs_tools_v2/full_agent/acs/*/error.txt`.

## Main run — EXP-006c (~12–15 min, ~$5–10)

Sequential batch of cross + long, full_agent only on each. 7 datasets.

```bash
tmux new -s exp006c
cd ~/SSDataAgent
nohup .venv/bin/python scripts/run_batch.py \
    exp006c_tools_cross exp006c_tools_long \
    > batch_006c.log 2>&1 &

# Detach: Ctrl-b d  — disconnect SSH freely; the box keeps running.
```

Check from anywhere:

```bash
.venv/bin/python scripts/status.py exp006c_tools_cross exp006c_tools_long
tail -f results/exp006c_tools_cross/run.log    # current dataset's stream
```

When all `✓ done`, generate reports:

```bash
.venv/bin/python scripts/generate_exp_report.py \
    exp006c_tools_cross --baseline pilot_paper_agents_gpt54
.venv/bin/python scripts/generate_exp_report.py \
    exp006c_tools_long --baseline pilot_paper_longitudinal_gpt54
```

## Optional — EXP-006d ablation (~25–35 min, ~$15–25)

Triples the cross-sectional run because it includes `agent_no_semantic` and
`agent_no_data`. Only do this if the main EXP-006c result looks promising
*and* you want to confirm whether stripping semantic context still helps the
tool-using path the way it helped the legacy code-block path in EXP-001.

```bash
nohup .venv/bin/python scripts/run_batch.py exp006d_tools_ablation_cross \
    > batch_006d.log 2>&1 &
```

Report it same way (no longitudinal counterpart yet — rationale in the yaml
comment).

## What I'm looking for in the results

vs `exp006b_tools_cross` / `exp006b_tools_long` (same model, old prompt):

| Dataset | EXP-006b T3 | EXP-006c target |
|---|---:|---:|
| gss | 0.207 | ≥ 0.30 |
| cps | 0.197 | ≥ 0.30 |
| acs | 0.003 | ≥ 0.20 |
| nlsy T4 | 0.010 | ≥ 0.10 |
| addhealth T4 | 0.000 | ≥ 0.05 |
| cfps T4 | 0.000 | ≥ 0.05 |
| us T4 | 0.135 | ≥ 0.15 |

If most cross T3s and most longitudinal T4s rise vs EXP-006b, the prompt
recipes worked and we can deprecate the legacy code-block path. If they
don't, queue an EXP-006e that goes harder on guidance (or considers a
different family — e.g. add gradient_boosting for numeric targets).

## Pulling results back to the laptop

```bash
# from your laptop, inside the project dir:
rsync -av --progress \
    user@cloud-box:~/SSDataAgent/results/exp006c_tools_cross/ \
    results/exp006c_tools_cross/
rsync -av --progress \
    user@cloud-box:~/SSDataAgent/results/exp006c_tools_long/ \
    results/exp006c_tools_long/
rsync -av --progress \
    user@cloud-box:~/SSDataAgent/docs/experiments/ \
    docs/experiments/

git add docs/experiments/2026-05-*-exp006c_*.md
git commit -m "EXP-006c results"
git push
```

## Pitfalls

- **Don't run `smoke_acs_tools` (v1) again** — it's the EXP-006b prompt and
  doesn't test the new recipes.
- **Costs roughly:** v1 smoke ≈ $0.05; cross batch ≈ $1.50; long batch ≈ $4;
  ablation ≈ $5. Add ~30% for retries / variability. Cap your billing alarm.
- **gpt-5.4 stochasticity is large.** Stage A scored ACS at 0.498, Stage B
  at 0.402 — same code, same prompt, different overall score. If a single
  EXP-006c dataset looks bad, re-running once will tell you whether it's
  noise or a real prompt-induced regression.
