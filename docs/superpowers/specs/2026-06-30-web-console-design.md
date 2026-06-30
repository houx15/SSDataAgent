# Local web console (experiment control plane) — design

**Date:** 2026-06-30
**Status:** approved (design)
**Part:** 7 of the strategy-seam refactor (`docs/handoff/design-reference.md` §14; `docs/handoff/delta-plan.md` P5 — optional, last)
**Precedes:** implementation plan `docs/superpowers/plans/2026-06-30-web-console.md`

## 1. Goal

A **localhost, single-user web console** that is the operational surface of the
iteration loop: launch/manage runs, browse results, compare strategies, export
reports, and keep the lab notebook. Localhost only; no auth in v1.

The console's success criterion (from §14) is making **one loop fast**: *spot a
failure in a run (which pattern type / which variable) → fork its config with a
tweak → launch → compare against the parent → log the result.* If forking +
launching + comparing takes more than a few clicks, the design has failed.

Part 7 is explicitly **optional and last** in the delta-plan: it must not block or
drive the rest of the refactor, and it must not modify the runner or the existing
flat-file store. It sits *on top of* what Parts 1–6 produced.

## 2. Locked design decisions

- **Scope: full six views** — leaderboard, run launcher + status board, run detail,
  compare, report export, lab notebook. (User chose the full §14 surface over a thin
  MVP.)
- **Backend: FastAPI** (imports the project's runner/config directly — same language
  as the harness). New optional dependency group `console = ["fastapi",
  "uvicorn[standard]"]`. `jinja2` and `markdown-it-py` are already deps.
- **Store: SQLite as a rebuildable index + queue state; flat files remain the source
  of truth.** DB lives at `results/_console.db` (the `_` prefix means `status.py`'s
  scan ignores it). The runner is **never modified**. Disk wins on any conflict; the
  index is droppable and rebuildable from `results/` at any time. The only
  non-rebuildable state in the DB is the `notebook` table and the queue/launch status
  of console-started runs that have not yet written flags to disk.
- **DB engine: SQLite** (stdlib `sqlite3`, zero new dependency, transactional for
  queue-status writes; single-user scale).
- **Frontend: React + Vite SPA** (TypeScript), `react-router` for the six views,
  **Plotly.js** (`react-plotly.js`) for charts. Lives in `web/` at the repo root,
  separate from `src/`. Dev: `vite dev` proxies `/api` → `localhost:8000`. Local/prod:
  `vite build` → `web/dist`, served by FastAPI `StaticFiles`.
- **Job execution: in-process worker thread, subprocess-backed**, sequential by default
  with a configurable concurrency cap (mirrors `run_batch.py`). Each job shells out to
  `scripts/run_experiment.py --experiment NAME`; status polled from flags + `run.log`
  into the DB. No Celery/Redis. No MLflow/W&B.
- **Report export reuses `scripts/generate_exp_report.py`** — refactor its core into an
  importable module (`ssdataagent.reports`) shared by the script and the console.
- **Notion mirror is best-effort/optional in v1** — the FastAPI backend has no MCP
  access, so Notion sync is a configured-token-only feature; absent a token it is a
  no-op. The lab notebook's primary durable output is an appended `LEDGER.md` row
  (7-column format, so the existing dashboard keeps working).
- **Determinism / safety:** the runner and the byte-stable artifact path are untouched;
  backend tests mock the subprocess (no real runner, no LLM); the SQLite index is built
  from a fabricated `results/` fixture.

## 3. Two granularities (experiment vs run)

The existing layout has two levels, and the console models both:

- **Experiment** = the launch unit. One `config/experiments.yaml` entry →
  `run_experiment.py` → all conditions × datasets. Experiment-level artifacts:
  `results/<exp>/done.flag` | `failed.flag` | `summary.csv` | `run.log`.
- **Run** = one condition × dataset execution:
  `results/<exp>/<condition>/<dataset>/<run_id>/` with `meta.json`, `eval.json`,
  `generated.csv`, `prompts.jsonl`, `responses.jsonl`, `workspace/`. (There are **no**
  run-level flag files; flags live only at experiment level.)

The **leaderboard** is row-per-run (condition × dataset), columns = pass rate by
pattern type + over-determination gap + cost. The **launcher** operates on experiments.

## 4. SQLite schema (`results/_console.db`)

`CREATE TABLE IF NOT EXISTS` only — no migration framework. All tables except
`notebook` (and the live queue status of console-started experiments) are rebuildable
from `results/`.

```sql
experiments(
  name TEXT PRIMARY KEY,
  status TEXT,            -- queued | running | done | failed | interrupted | pending
  prompt_variant TEXT,
  model TEXT, provider TEXT,
  git_sha TEXT,
  finished_at TEXT,
  config_json TEXT,       -- the launched experiments.yaml entry
  config_hash TEXT,
  source TEXT             -- 'disk' | 'console'
)

runs(
  experiment TEXT, condition TEXT, dataset TEXT, run_id TEXT,
  run_dir TEXT,
  by_type_json TEXT,      -- {"type1": 0.57, ...} from eval.json / summary.csv
  overall_average REAL,
  overdetermination_gap REAL,   -- nullable; absent on historical runs
  cost REAL,
  finished_at TEXT,
  PRIMARY KEY (experiment, condition, dataset, run_id)
)

notebook(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  hypothesis TEXT, change TEXT, result TEXT, interpretation TEXT, next TEXT,
  linked_experiments_json TEXT,
  ledger_synced INTEGER DEFAULT 0,
  notion_page_id TEXT          -- nullable
)
```

`status` mirrors `status.py`'s `_state()`: `done.flag`→done, `failed.flag`→failed,
`run.log` present but no flag→interrupted/running, none→pending. Console-launched
experiments carry `queued`/`running` until a flag appears, then `sync` reconciles.

## 5. Backend layout (`src/ssdataagent/console/`)

| Module | Responsibility |
|---|---|
| `db.py` | SQLite connect + schema (`CREATE TABLE IF NOT EXISTS`); `connect(results_root)`. |
| `sync.py` | `sync_index(conn, results_root)`: scan `results/` (skip `_`-prefixed), compute experiment status like `status.py._state()`, parse `done.flag`/`summary.csv`, descend run dirs reading `meta.json`/`eval.json`; upsert. Idempotent. Only overwrites `source='disk'` rows; transitions console rows when flags appear. Disk wins. |
| `queue.py` | `JobQueue(results_root, concurrency=1)`: `enqueue(name)`, worker thread, subprocess `run_experiment.py --experiment NAME`, `cancel(name)` (terminate), status → DB. |
| `forking.py` | `fork_experiment(base, new_name, overrides)`: read `config/experiments.yaml[base]`, apply overrides, write a new entry preserving siblings (round-trip YAML), return entry. |
| `reports.py` | wrap the refactored `ssdataagent.reports` core: `render_report(experiments, fmt)` → HTML + Markdown. |
| `notebook.py` | entry CRUD; `append_to_ledger(entry)` (7-col row); `mirror_to_notion(entry)` (no-op without token). |
| `app.py` | FastAPI factory; mounts `api/` router + `StaticFiles` for `web/dist`. |
| `api/runs.py`, `api/leaderboard.py`, `api/compare.py`, `api/reports.py`, `api/notebook.py` | routers. |
| `models.py` | pydantic request/response schemas. |
| `__main__.py` | `python -m ssdataagent.console` → uvicorn (host=127.0.0.1). |

## 6. Endpoints (the six views)

1. **Leaderboard** — `GET /api/leaderboard?condition&dataset&model` → rows
   (experiment, condition, dataset, pass-rate-by-type, over-determination gap, cost);
   `is_champion` per (condition × dataset) = the row with the highest `overall_average`
   across **all** strategies/experiments for that cell, excluding pilots (experiment
   name starts with `pilot_`). This generalizes the dashboard's `_mark_champion`, which
   was restricted to `full_agent`; the console ranks across strategies, so the champion
   is the best strategy in each cell. Triggers a `sync_index` first.
2. **Run launcher + status board** — `POST /api/runs` {`name` | `fork_from`+`overrides`+
   `new_name`} enqueue; `GET /api/runs` list with status; `GET /api/runs/{name}` one;
   `POST /api/runs/{name}/cancel`; `GET /api/runs/{name}/log` (tail `run.log`, poll;
   optional SSE stream).
3. **Run detail** — `GET /api/runs/{name}/detail` → full metrics (`by_type`,
   `by_variable`, `by_domain`, `overdetermination`), artifact links (generated.csv,
   `workspace/` code, raw synthetic data, **logged LLM I/O** = `prompts.jsonl` /
   `responses.jsonl`), exact `config_json` + `git_sha`.
4. **Compare** — `POST /api/compare` {selectors: list of (experiment, condition,
   dataset)} → metric diff table, strategy × pattern-type matrix (heatmap data), and
   real-vs-sim distribution data (real eval subset vs `generated.csv`) for client-side
   Plotly.
5. **Report export** — `POST /api/reports` {experiments, format: `html`|`md`} → rendered
   report (methods, leaderboard, over-determination gap, key real-vs-sim plots, linked
   notebook interpretation); downloadable.
6. **Lab notebook** — `GET /api/notebook`, `POST /api/notebook` (hypothesis → change →
   result → interpretation → next, linked to experiments); on create, append a
   `LEDGER.md` row and best-effort Notion mirror.

## 7. Frontend (`web/`)

Vite + React + TypeScript. Routes: `/` leaderboard, `/runs` launcher+status,
`/runs/:name` detail, `/compare`, `/reports`, `/notebook`. Charts via
`react-plotly.js` (heatmap + distribution overlays). API client hits `/api/*`. Dev
server proxies `/api` to uvicorn; production build is served by FastAPI `StaticFiles`.
A `scripts/console` helper builds the SPA (if `web/dist` missing) then launches uvicorn.

## 8. Plan phasing

One spec; the plan runs in three independently-testable phases:

- **Phase 1 — backend read-only core:** `db` + `sync` + leaderboard & run-detail API
  over the existing `results/`. Deliverable: a usable read-only console (no launching).
- **Phase 2 — control:** `JobQueue` + launcher endpoints + `forking` + compare API.
  Deliverable: fork → launch → compare loop server-side.
- **Phase 3 — outputs + UI:** report export (`ssdataagent.reports` refactor) + lab
  notebook (+ LEDGER/Notion) + the React SPA wiring all six views.

## 9. Determinism, safety, leakage

- **Runner untouched.** No change to `runner.py`, `conditions.py`, strategies, or the
  byte-stable artifact path. The two P0 byte-stable tests stay green.
- **No drift.** SQLite is a cache; `sync_index` rebuilds it from disk and disk always
  wins. Dropping `results/_console.db` loses nothing but `notebook` + in-flight queue
  status.
- **Subprocess isolation.** Launches go through the existing `run_experiment.py` CLI;
  the console adds no new model-execution path. Tests mock `subprocess`.
- **Localhost only**, no auth (v1, single-user). Bind `127.0.0.1`.

## 10. Out of scope (v1)

- Multi-user / auth / remote hosting.
- Celery/Redis, MLflow/W&B (self-contained store only).
- Modifying the runner to write to the DB (DB stays a rebuildable index).
- A full JS test suite (frontend gets a build smoke + minimal component tests).
- Real-time Notion bidirectional sync (best-effort one-way mirror, token-gated).
- HTMX/Jinja server-rendered alternative (React SPA chosen).

## 11. Testing strategy

- **Backend (full pytest):**
  - `tests/test_console_db.py` — schema creation, connect, idempotent re-create.
  - `tests/test_console_sync.py` — fabricate a `results/` tree (done.flag, summary.csv,
    run dirs with eval.json incl. and excl. overdetermination) → `sync_index` → assert
    `experiments`/`runs` rows, status mapping mirrors `status.py._state()`, disk-wins on
    re-sync, `_`-prefixed dirs skipped.
  - `tests/test_console_queue.py` — `JobQueue` with **mocked subprocess**: enqueue →
    running → done/failed transitions, cancel terminates, concurrency cap respected.
  - `tests/test_console_forking.py` — fork writes a new YAML entry, preserves siblings,
    applies overrides, round-trips.
  - `tests/test_console_api.py` — FastAPI `TestClient` over a temp results fixture +
    temp SQLite: leaderboard rows + champion flag, run detail metrics + artifact links,
    compare diff + heatmap shape, reports HTML+MD render, notebook create → LEDGER row.
  - `tests/test_console_notebook.py` — entry → LEDGER 7-col append (dashboard parser
    accepts it); Notion mirror no-op without token.
- **Frontend (light):** `web/` `npm run build` succeeds (smoke); `vitest` component
  tests for the leaderboard table and compare view.

## 12. Gate (per `feedback_refactor_gate_philosophy`)

Our tests pass + no NEW failures vs. the 4 pre-existing `autograd`-missing failures
(`tests/test_config.py::test_unknown_provider_raises` + 3
`tests/test_ssdatabench_integration.py *_legacy`). No bit-for-bit reproduction gate.
The two P0 byte-stable runner tests stay green (runner untouched).
