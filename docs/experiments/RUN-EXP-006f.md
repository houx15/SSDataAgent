# Run EXP-006f diagnostic on the cloud box

> **For the user.** What to run once you've pulled commit `b604880`
> (EXP-006f: persist-on-error + soft-fail chronology gate) plus the
> new `exp006f_tools_diag` config. This is a **diagnostic** re-run of
> cfps + addhealth alone — the two datasets that crashed on EXP-006e and
> lost their transcripts. ~$0.50, ~5 min.

## What changed since EXP-006e

No prompt or model change. Only the orchestrator plumbing:

- `orchestrator.py` — force-commit + sampling now run inside a single
  `try/finally`, so `tool_calls.json` / `transcript.json` / `chain.json`
  always land on disk even when the forced commit raises. On EXP-006e
  these two files were missing on the crashed cfps/addhealth runs, which
  is why we have no forensics to work with.
- `commit.py` + `state.py` — when `max_turns` is reached and the
  chronology gate is the only blocker, the orchestrator sets
  `state.t4_unverified=True` and retries `commit_generator`. The flag
  bypasses the gate, the run completes (penalized on T4), and the result
  carries `t4_unverified=True` + a `warning` so reporting can flag it
  rather than crash. The flag is **orchestrator-only** — never in the
  LLM-facing tool schema, so the agent can't abuse it.

Net effect: if cfps or addhealth hits the same gate again, this time we
get a full transcript + a generated CSV with the run flagged unverified,
instead of a crashed run with no data.

## On the cloud box

```bash
cd ~/SSDataAgent
git pull                              # picks up commit b604880 + exp006f config
conda activate ssda
# pip install -r requirements.txt    # only if deps changed (they didn't)
```

The submodule is unchanged.

## Run (~5 min, ~$0.50)

```bash
tmux new -s exp006f
conda activate ssda
python scripts/run_experiment.py --experiment exp006f_tools_diag | tee diag_006f.log

# Detach: Ctrl-b d  — disconnect SSH freely; the box keeps running.
# Re-attach: tmux attach -t exp006f
```

Smaller than EXP-006e so a single `run_experiment` call is fine — no
need for `run_batch`.

Check status from another shell if you want:

```bash
conda activate ssda
python scripts/status.py exp006f_tools_diag
tail -f results/exp006f_tools_diag/run.log
```

When you see `done.flag`:

```bash
python scripts/generate_exp_report.py exp006f_tools_diag \
    --baseline pilot_paper_longitudinal_gpt54
```

## What to look for

This is **diagnostic**, not a benchmark. Three possible outcomes — each
points at a different next step.

### Outcome A — both runs commit cleanly (no max_turns, no t4_unverified)

The cfps/addhealth crashes were caused by some incidental error path
the EXP-006f try/finally happens to also paper over (e.g. a sandbox
timeout, a YAML race, a transient API blip). Read the tool_calls.json
to confirm the agent did call `score_event_order` and committed
normally. If T4 is in line with nlsy (~0.10+), declare victory and move
on.

### Outcome B — runs complete but with `t4_unverified=True` in chain.json

The agent really did hit the chronology gate and never called
`score_event_order` in 40 turns. Two follow-ups, in order:

1. **Read `results/exp006f_tools_diag/full_agent/{cfps,addhealth}/<latest>/workspace/tool_calls.json`**
   to see what the agent did spend its turns on. Looking for: was it
   stuck in a fit-loop, did it misread a column as non-event-age, or did
   it run out of turns mid-recipe?
2. Decide between (a) bumping `DEFAULT_MAX_TURNS` from 40 to 60 in
   `orchestrator.py` if it ran out of room mid-correct-behavior, vs.
   (b) widening the `_event_age_columns` heuristic in `commit.py` if
   the gate misfired on a column that doesn't actually need ordering
   (e.g. `age_at_death` is terminal — strictly true T4 doesn't require
   it to come last vs. another event).

### Outcome C — still crashes

Then the bug isn't what we thought it was. Read the new `tool_calls.json`
+ `transcript.json` (which are now guaranteed to land) and diagnose
from there.

## Pulling results back to the laptop

```bash
# from your laptop, inside the project dir:
rsync -av --progress \
    user@cloud-box:~/SSDataAgent/results/exp006f_tools_diag/ \
    results/exp006f_tools_diag/
```

No corresponding `docs/experiments/2026-*-exp006f*.md` is written by
the run itself — write a short retro report after reading the artifacts.

## Pitfalls

- **Don't run `exp006e_tools_long` again** — that batch retests all four
  longitudinal datasets and costs ~$5. This diag is targeted at the two
  that failed.
- **The `t4_unverified` flag changes downstream reporting.** If the
  current `generate_exp_report.py` does not yet read
  `chain_meta["t4_unverified"]`, the T4 number it prints for an
  unverified chain will be whatever the verifier computed against the
  unverified-by-design generation order — i.e., low. That's the
  intended "penalize, don't crash" semantics; if you want the report to
  flag it explicitly, that's a small post-run patch, not part of this
  diag.
- **gpt-5.4 stochasticity:** if exactly one of the two datasets soft-fails
  and the other doesn't, don't over-read it — single-run variance is
  large. If you see Outcome B, consider a second run of the soft-failing
  dataset alone before changing the gate or turn cap.
