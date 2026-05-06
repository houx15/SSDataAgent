# Running experiments on the GCP cloud box

The framework is designed for the workflow: **launch a batch of experiments
in tmux, walk away, come back to a finished comparison report.** Local
network drops don't matter; the box keeps running.

## What to upload (not in git)

Two paths must be `rsync`'d from your laptop to the cloud box. Both are in
`.gitignore` precisely so they're never accidentally committed.

| Path                  | Size  | Why                                                     |
|-----------------------|-------|---------------------------------------------------------|
| `real_data/`          | ~6 MB | Cleaned SSDataBench survey CSVs the agent and metrics consume. |
| `ssdatabench/`        | ~35 MB | Third-party eval suite (referenced by `config/datasets.yaml`). |

```bash
# from your laptop, inside the project dir
rsync -av --progress \
    real_data/ \
    user@cloud-box:~/SSDataAgent/real_data/

rsync -av --progress \
    ssdatabench/ \
    user@cloud-box:~/SSDataAgent/ssdatabench/
```

You can prune `real_data/` further if you want — only `real_data/used_dataset/`
and `real_data/dataset_meta.json` are required. The other subdirectories
(`real_data/addhealth/`, `real_data/cfps/`, etc.) are raw source data not
read by any experiment.

### Putting `real_data/` on a separate disk

If the cloud box has a mounted persistent disk for survey data, you don't
have to put it under the repo. Set `SSDA_DATA_ROOT` in `.env` to the
absolute path of the data directory:

```bash
# in .env on the cloud box
SSDA_DATA_ROOT=/mnt/disk2/survey_data
```

The directory at `$SSDA_DATA_ROOT` should then contain `used_dataset/`,
`dataset_meta.json`, etc. — i.e. the same subtree you'd otherwise put
under `real_data/`. The leading `real_data/` in `config/datasets.yaml` is
stripped at lookup time when this var is set. Unset = the repo's
`real_data/` (existing behavior).

`ssdatabench/` is currently expected under the repo root only. Ping if you
also want an env knob for that — same pattern.

## What to create on the box

The API key is **never** in any config file. Create `.env` at the project root
on the cloud box yourself:

```bash
# on the cloud box
cat > ~/SSDataAgent/.env <<'EOF'
LLM_API_KEY=sk-...your-real-key...
EOF
chmod 600 ~/SSDataAgent/.env
```

That's the *only* required env var. `LLM_PROVIDER`, `LLM_BASE_URL`, and
`LLM_MODEL` come from `config/experiments.yaml` per-experiment now (see
`exp001_rubric_cross` for an example), so one `.env` covers every experiment
in the batch even if they target different models.

If you do want a fallback default in `.env` (in case some yaml entry omits
`llm_model`), the format matches `.env.example`:

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.4-2026-03-05
LLM_API_KEY=sk-...
```

## Setup on a fresh box

```bash
cd ~/SSDataAgent
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
# upload data + ssdatabench (see above)
# create .env (see above)
```

## Running a batch

```bash
# in tmux on the cloud box
tmux new -s ssda
cd ~/SSDataAgent
source .venv/bin/activate

# kick off the batch — sequential, resumable, one log per experiment
nohup python scripts/run_batch.py \
    exp001_rubric_cross exp001_rubric_long \
    > batch.log 2>&1 &

# detach: Ctrl-b d
# disconnect SSH freely; the box keeps running.
```

## Checking on it from anywhere

```bash
# at-a-glance status for all experiments
python scripts/status.py

# only the ones you care about
python scripts/status.py exp001_rubric_cross exp001_rubric_long

# live log for the currently-running experiment
tail -f results/exp001_rubric_cross/run.log

# raw "what's done" — single source of truth
ls results/*/done.flag
```

`results/<exp>/done.flag` is the contract: if it exists, the experiment
finished and `summary.csv` is on disk. The batch runner skips experiments
whose `done.flag` exists, so re-running the same `run_batch.py` command
after a reboot picks up from the next undone one — no separate `--resume`
needed for the batch level.

If a single experiment fails, `failed.flag` is written with the traceback
and the batch continues with the next one. Inspect with
`cat results/<exp>/failed.flag`. Re-running clears the flag and retries.

## Generating the report

After `status.py` shows all `✓ done`:

```bash
python scripts/generate_exp_report.py exp001_rubric_cross \
    --baseline pilot_paper_agents_gpt54
python scripts/generate_exp_report.py exp001_rubric_long \
    --baseline pilot_paper_longitudinal_gpt54
```

Each command writes `docs/experiments/<date>-<exp>-report.md` with three
sections: Strategy, Results (T1-T5 per dataset), and **vs Paper-best** (Δ
against the per-cell strongest paper LLM from `config/paper_baselines.json`).
Pull those reports back to your laptop with `rsync` (or just `git add` them —
they don't contain data, only numbers).

## Pulling results back

```bash
# results CSVs + per-experiment reports — small, ~MB at most
rsync -av --progress \
    user@cloud-box:~/SSDataAgent/results/ \
    results/

rsync -av --progress \
    user@cloud-box:~/SSDataAgent/docs/experiments/ \
    docs/experiments/
```
