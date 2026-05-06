# Resume after EXP-001 completes

> **For Claude in the next session:** read this file first. It tells you what
> just happened, what's running, what to look at, and how to decide the next
> moves. Delete or supersede it once EXP-001 is fully digested.
>
> **For the user:** point Claude at `docs/experiments/RESUME-EXP-001.md` and
> say "EXP-001 finished, follow this." That's enough context.

---

## Where you (Claude) are in the project arc

You're a few sessions deep into helping the user (a data scientist) build
an LLM agent that synthesizes social-survey data, benchmarked against
SSDataBench's T1–T5 metrics. Up to and including the previous session you:

1. **Diagnosed** five high-leverage gaps in the agent (rubric blindness,
   trivially-self-passing VALIDATION, dead `preserve_missingness` flag,
   weak MODELING prompt, no cross-run memory). See
   `docs/experiments/STRATEGY.md` for the standing hypotheses.
2. **Built a matrix-of-variants framework** so cloud-box batches can run
   different prompt/model combinations without source edits between runs.
   Key pieces: `PROMPT_VARIANTS` registry in
   `src/ssdataagent/agent/prompt_templates.py`, per-experiment
   `prompt_variant` / `llm_model` / `llm_provider` / `llm_base_url` in
   `config/experiments.yaml`, `scripts/run_batch.py` + `status.py` +
   `generate_exp_report.py`, `done.flag` / `failed.flag` resume contract.
3. **Added `SSDA_ROOT` env var** so `real_data/` and `results/` can live
   on a mounted disk on a cloud box. `.env` is loaded at config-module
   import time so the var is visible everywhere.
4. **Launched EXP-001** on a GCP box in tmux:
   - `exp001_rubric_cross` (gss/cps/acs × full_agent + 2 ablations)
   - `exp001_rubric_long`  (nlsy/addhealth/cfps/us × full_agent)
   - Both `prompt_variant: rubric`, both on `gpt-5.4-2026-03-05`.
   - Baselines (already done, sitting in `results/`):
     `pilot_paper_agents_gpt54` and `pilot_paper_longitudinal_gpt54`.

The hypothesis under test is at the top of
`docs/experiments/2026-05-06-exp001_rubric_in_system_prompt.md`:
adding the T1–T5 rubric block to `SYSTEM_PROMPT` lifts overall mean by
**+0.03 to +0.08** by letting the agent pick model architecture against
the actual metrics instead of by vibes.

---

## Step 1 — confirm the run finished cleanly

If results aren't on the laptop yet, pull them from the box:

```bash
rsync -av user@cloud-box:~/SSDataAgent/results/exp001_rubric_cross/ results/exp001_rubric_cross/
rsync -av user@cloud-box:~/SSDataAgent/results/exp001_rubric_long/  results/exp001_rubric_long/
```

Then:

```bash
python scripts/status.py exp001_rubric_cross exp001_rubric_long
ls results/exp001_rubric_cross/done.flag results/exp001_rubric_long/done.flag
```

If either has a `failed.flag` instead, read it (`cat results/<exp>/failed.flag`)
and `tail -200 results/<exp>/run.log` before doing anything else.

---

## Step 2 — generate reports + read them

```bash
python scripts/generate_exp_report.py exp001_rubric_cross \
    --baseline pilot_paper_agents_gpt54
python scripts/generate_exp_report.py exp001_rubric_long \
    --baseline pilot_paper_longitudinal_gpt54
```

These write `docs/experiments/<today>-exp001_rubric_*-report.md` with
**Strategy / Results / vs Paper-best / vs Baseline** sections. Read both
end-to-end before forming an opinion.

---

## Step 3 — focus areas (where to look, in order)

### 3a. Quantitative — overall and per-T-type

In each report's **vs Baseline** section, look for:

- **Overall delta per dataset** — is it +0.03 to +0.08 (predicted), more,
  less, or negative?
- **Which T-type moved most?** Concentrate on T2, T3, and (for longitudinal)
  T4. The rubric explicitly calls out the failure modes for each.
- **Are there regressions?** Any T-type that got *worse* by more than 0.02
  is a signal that the rubric pushed the agent into a different but worse
  architecture.

### 3b. Qualitative — did the agent's behavior actually change?

The numbers tell you *whether* the variant works. The agent's code tells
you *why* (or *why not*). Read the MODELING step for one cell per dataset:

```bash
ls results/exp001_rubric_cross/full_agent/{gss,cps,acs}/*/code/step_002.py
ls results/exp001_rubric_long/full_agent/{nlsy,addhealth,cfps,us}/*/code/step_002.py
```

In each `step_002.py`, look for:

- **Comments referencing T1/T2/T3/T4/T5.** Did the agent internalize the
  rubric? (If absent, the rubric isn't reaching the model — possibly
  because of prompt length, system prompt placement, or LLM filtering.)
- **Model family choice** — did the rubric push the agent toward chained
  conditional models or copulas (vs the per-column independent sampling
  the baseline often picks)?
- **Mention of `preserve_missingness` / NaN handling** — did the rubric
  prompt the agent to think about conditional missingness for T3?
- **For longitudinal datasets only:** any explicit event-time chronology
  pass (sample → enforce ordering → resample)? T4 only moves if this
  exists.

Cross-reference against the baseline's `step_002.py` (in
`results/pilot_paper_agents_gpt54/...` and
`results/pilot_paper_longitudinal_gpt54/...`) — same datasets, no rubric.
The diff is the lever.

### 3c. VALIDATION self-pass behavior

Open one or two `step_003.stdout` files (the VALIDATION step). The
baseline's chronic problem (documented in the previous session's
diagnosis) was the agent printing `VALIDATION OK` despite obviously broken
joint stats. With the rubric, did the agent's self-check get sharper?
This informs **EXP-002**'s scope.

---

## Step 4 — decide next moves

| What you saw                                                           | Next move                                                                 |
|------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Overall lift +0.03 to +0.08, T2/T3 moved most                          | Continue down the backlog: **EXP-002** (validation thresholds) is up next. |
| Overall lift >>0.08, T4 cracked on multiple longitudinal datasets      | The rubric is very effective. Consider **EXP-001b**: tighter, dataset-type-specific architecture guidance, before EXP-002. |
| Overall flat or worse, agent did *not* reference T1–T5 in step_002.py  | Rubric is being ignored. Likely SYSTEM_PROMPT bloat. Revise to a tighter rubric (move it from system to a per-stage prompt prefix). New **EXP-001b**. |
| Overall flat or worse, agent *did* reference T1–T5 but architecture didn't change | Deeper issue: the agent acknowledges metrics but can't translate them into code choices. Escalate to **EXP-004** (explicit MODELING decision rule) in parallel with EXP-002. |
| Per-cell regressions (any T-type −0.02 or worse on multiple datasets)  | Investigate code/step_NN.stderr for those cells. Possibly the agent picked a too-ambitious architecture that errored out. Add a "fall back to simple if validation fails N times" loop — likely EXP-002 territory. |

After picking the next move, **propose it to the user before
implementing.** Don't auto-queue more experiments.

---

## What's already in `STRATEGY.md` backlog (post-EXP-001)

In current expected-lift × inverse-cost order:

- **EXP-002** — Hard quantitative VALIDATION thresholds (TV ≤ 0.10,
  |Δr| ≤ 0.15, |ΔP| ≤ 0.05) + explicit refusal to print `VALIDATION OK`
  if breached.
- **EXP-003** — Wire `preserve_missingness=True` from the dead-code flag
  in `src/ssdataagent/agent/orchestrator.py` (it's a 5-line change but
  hasn't been done).
- **EXP-004** — MODELING decision rule that branches by dataset type
  (cross-sectional vs longitudinal).
- **EXP-005** — Cross-run `lessons.md` injected into SYSTEM_PROMPT,
  curated from the strongest cells we have.

If EXP-001's report leads you to a *new* experiment not in this list, add
it to STRATEGY.md backlog before scaffolding it.

---

## Files you'll touch when designing the next experiment

- **`src/ssdataagent/agent/prompt_templates.py`** — add a new entry to
  `PROMPT_VARIANTS` (modeled on how `rubric` was added).
- **`config/experiments.yaml`** — add new entries (model on the `exp001_rubric_*`
  pattern, including `prompt_variant` + `llm_model` / `llm_provider` /
  `llm_base_url`).
- **`scripts/new_experiment.py <exp_name> --hypothesis "..." --baseline ...`** —
  scaffolds the retro skeleton in `docs/experiments/<date>-<name>.md` and
  inserts a `_pending_` row into `LEDGER.md`. Use this rather than hand-writing
  the file.
- **`docs/experiments/STRATEGY.md`** — flip the backlog item to `[~]` in progress.
- **`docs/experiments/LEDGER.md`** — fill in the headline once the run completes.

---

## Pitfalls to avoid (institutional knowledge)

- **Never put API keys in any config file.** Only `.env` (gitignored).
- **Never edit baseline experiments in `config/experiments.yaml`.** They are
  the A/B reference; create new entries instead.
- **`results/` is gitignored.** Don't try to commit results.
- **`real_data/` is gitignored.** Treat survey CSVs as read-only.
- **Don't use the literal word "eval" in commit messages or shell commands.**
  The project hook blocks it. Use "scoring", "metric", "score" instead.
- **Verify before recommending paths/files.** If a memory or earlier note
  mentions a file or function, `ls` or `grep` for it before telling the
  user to edit it. The codebase has been moving fast.
- **`<exp>` patterns in heredoc commit messages get parsed as redirects by
  the project boundary hook.** Avoid `<` `>` in commit message bodies.

---

## Useful one-liners

```bash
# At-a-glance status for all experiments
python scripts/status.py

# Just the EXP-001 pair
python scripts/status.py exp001_rubric_cross exp001_rubric_long

# Live tail of one experiment's run log
tail -f results/exp001_rubric_cross/run.log

# Per-experiment report (after done.flag exists)
python scripts/generate_exp_report.py <exp> --baseline <baseline_exp>

# Read the agent's actual MODELING code
ls results/<exp>/full_agent/<dataset>/*/code/step_002.py | tail -1 | xargs cat

# Read the agent's VALIDATION self-check
ls results/<exp>/full_agent/<dataset>/*/step_003.stdout | tail -1 | xargs cat

# Pull results from cloud box
rsync -av user@cloud-box:~/SSDataAgent/results/exp001_rubric_cross/ \
    results/exp001_rubric_cross/
rsync -av user@cloud-box:~/SSDataAgent/results/exp001_rubric_long/ \
    results/exp001_rubric_long/
```

---

## Once EXP-001 is fully digested

- Mark EXP-001 `[x]` in `docs/experiments/STRATEGY.md`.
- Fill in the `Retro` section of
  `docs/experiments/2026-05-06-exp001_rubric_in_system_prompt.md`
  (What worked / What didn't / Surprises / Lesson worth preserving / Next experiment).
- Update the headline in `LEDGER.md`'s EXP-001 row from `_pending_` to the
  actual number.
- If a non-obvious lesson emerged, promote it to STRATEGY.md's "Lessons"
  section (so future experiments inherit it).
- **Delete or supersede this RESUME-EXP-001.md file** — it's no longer
  current. Either remove it, or rename to `RESUME-EXP-002.md` and rewrite
  for the next experiment.
