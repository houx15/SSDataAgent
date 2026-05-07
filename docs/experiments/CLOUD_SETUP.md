# Running experiments on the GCP cloud box

The framework is designed for the workflow: **launch a batch of experiments
in tmux, walk away, come back to a finished comparison report.** Local
network drops don't matter; the box keeps running.

## What to fetch and what to upload

`ssdatabench/` is a **git submodule** (third-party scoring suite,
~35 MB). Don't rsync it — `git submodule update` populates it from the
upstream repo, and that's the source of truth for which files are pinned.

`real_data/` is the only thing that has to be **rsync'd** from your
laptop because the survey CSVs aren't redistributed via git.

```bash
# on the cloud box, after `git clone`:
cd ~/SSDataAgent
git submodule update --init --recursive

# from your laptop, inside the project dir:
rsync -av --progress real_data/ user@cloud-box:~/SSDataAgent/real_data/
```

You can prune `real_data/` further if you want — only `real_data/used_dataset/`
and `real_data/dataset_meta.json` are required. The other subdirectories
(`real_data/addhealth/`, `real_data/cfps/`, etc.) are raw source data not
read by any experiment.

### Putting data + results on a mounted disk

If the cloud box has a persistent disk and you want the uploaded survey
CSVs *and* the per-experiment outputs on it (boot disk stays small), set
`SSDA_ROOT` in `.env`:

```bash
# in .env on the cloud box
SSDA_ROOT=/mnt/disk2/ssda
```

That directory should contain `real_data/` (you upload it) and
`results/` (auto-created on first run). The rsync target becomes:

```bash
rsync -av --progress real_data/ user@cloud-box:/mnt/disk2/ssda/real_data/
```

With `SSDA_ROOT` set, `scripts/run_batch.py` reads CSVs from
`$SSDA_ROOT/real_data/` and writes outputs to `$SSDA_ROOT/results/`;
`scripts/status.py` reads `$SSDA_ROOT/results/` automatically. Unset =
both stay under the repo. `ssdatabench/` is not affected by `SSDA_ROOT`
— as a submodule it always lives under the repo.

#### Splitting data and results across disks (optional)

Each tree has an independent override:

```bash
SSDA_DATA_ROOT=/mnt/fast/real_data
SSDA_RESULTS_ROOT=/mnt/large/results
```

Any unset override falls back to `$SSDA_ROOT/<tree>/`, then to the repo.

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
git clone <repo-url> ~/SSDataAgent
cd ~/SSDataAgent
git submodule update --init --recursive   # populates ssdatabench/
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
# rsync real_data/ from your laptop (see above)
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
