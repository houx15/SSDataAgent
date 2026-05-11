# Experiment Dashboard

Self-contained HTML dashboard at `docs/dashboard/index.html`. Double-click
the file to open it — no server, no install, no network needed.

## What it shows

- **Champion card** — the non-pilot experiment with the highest
  `overall_mean_full_agent` (mean of all `full_agent` cells in
  `summary.csv`, across both halves of a multi-experiment row).
  Ties broken by most recent date.
- **All-experiments table** — one row per `LEDGER.md` row, sortable on
  every column. Click a row to expand a detail panel with config,
  per-condition × dataset × T-type heatmap, hypothesis, and lesson.
- Filter box (matches exp_name, hypothesis, prompt_variant) and a
  "show pilots" toggle.

The champion's mean can favor narrow-but-strong spikes over
broad-and-mixed runs (a single-dataset spike has fewer cells than a
7-dataset run, so one good cell weighs more). Read the table, not just
the card.

## Rebuilding

After landing a new experiment (new `done.flag`, new LEDGER row, new
retro), regenerate the HTML:

```bash
.venv/bin/python scripts/build_dashboard.py
```

Verbose mode prints which rows had missing results or retros:

```bash
.venv/bin/python scripts/build_dashboard.py --verbose
```

Strict mode exits non-zero if anything is missing (useful for CI):

```bash
.venv/bin/python scripts/build_dashboard.py --strict
```

Commit the regenerated HTML alongside the experiment artifacts so
teammates can `git pull && open docs/dashboard/index.html`:

```bash
git add docs/dashboard/index.html
git commit -m "dashboard: rebuild after <exp_name>"
```

## What it reads

| Source | What for |
|---|---|
| `docs/experiments/LEDGER.md` | The canonical list of experiments. One markdown table row → one dashboard row. Hypothesis and headline columns are the curated one-liners shown in the UI. |
| `results/<exp_name>/summary.csv` | Per-cell pass rates (`condition,dataset,type,pass_rate`). Source of truth for scores. |
| `results/<exp_name>/done.flag` | What actually ran (`prompt_variant`, `llm_model`, `llm_provider`, `finished_at`). Overrides yaml if they disagree. |
| `config/experiments.yaml` | Per-experiment config (datasets, conditions, n_rows). |
| `docs/experiments/<date>-<exp>-report.md` and `docs/report/<date>-*.md` | Retro narrative. Parsed heading-agnostically; bullets like `- **Lesson worth preserving:** ...` are extracted as KV pairs. |

LEDGER rows that name two experiments (e.g.
`exp006c_tools_cross + exp006c_tools_long`) are merged into one
dashboard row with combined scores.

Rows whose `exp_name` starts with `pilot_` are tagged and excluded
from the champion selection.

## Troubleshooting

- **`WARNING no results for <exp>`** — the LEDGER references an
  experiment with no `results/<exp>/done.flag`. Either the experiment
  hasn't been pulled back from the cloud box yet, or its results live
  under a different directory name. Add it or remove the LEDGER row.
- **`WARNING retro not found: <path>`** — the LEDGER's retro link
  doesn't resolve. Older pilots use `../report/...`; newer experiments
  use bare filenames relative to `docs/experiments/`. Fix the link in
  `LEDGER.md`.
- **Champion is a spike, not a workhorse** — see "What it shows"
  above. The mean rewards few-good-cells over many-mixed-cells. Use
  the table to see the broader runs in context.
- **Hypothesis text looks wrong** — the dashboard prefers the retro's
  YAML frontmatter `hypothesis` if present, then `## Hypothesis` first
  line, then the LEDGER `hypothesis` column. Real retros mostly fall
  through to LEDGER, so write a good one-liner there.

## What it does NOT do

- No live monitoring — it's a static artifact rebuilt on demand.
- No editing config from the UI — read-only.
- No diff-two-experiments view — detail panel shows one at a time.
- No timeline / scoreline chart — the sortable table is the comparison
  surface.
- No auto-rebuild on file change — run the script yourself after a new
  experiment finishes.

## Where the code lives

- `scripts/build_dashboard.py` — CLI entry point.
- `src/ssdataagent/dashboard/` — parsers + merger + renderer.
- `src/ssdataagent/dashboard/templates/index.html.jinja` — page template.
- `tests/test_dashboard_*.py` — 43 unit tests covering the full pipeline.

Design spec: `docs/superpowers/specs/2026-05-11-experiments-dashboard-design.md`.
