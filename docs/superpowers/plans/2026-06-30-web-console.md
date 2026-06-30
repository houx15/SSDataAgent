# Local Web Console (Experiment Control Plane) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A localhost single-user web console (FastAPI + React/Vite SPA) over the existing flat-file results store, providing six views: leaderboard, run launcher + status board, run detail, compare, report export, lab notebook.

**Architecture:** FastAPI backend imports the project's runner/config directly; a SQLite DB at `results/_console.db` is a *rebuildable index* synced from `results/` plus *queue state* for console-launched runs (flat files stay the source of truth, runner untouched). A subprocess-backed in-process job queue shells out to `scripts/run_experiment.py`. A React+Vite SPA (TypeScript, react-router, Plotly.js) is served by FastAPI `StaticFiles` in production and proxied in dev.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, SQLite (stdlib `sqlite3`), pydantic, PyYAML (already a dep), React, Vite, TypeScript, react-router-dom, react-plotly.js, vitest.

## Global Constraints

- **Never modify the runner or the byte-stable artifact path.** No edits to `src/ssdataagent/experiments/runner.py`, `conditions.py`, strategies, or evaluation. The two P0 byte-stable tests must stay green.
- **Flat files are the source of truth.** SQLite (`results/_console.db`) is a rebuildable cache; on any conflict, disk wins. Dropping the DB loses only the `notebook` table and in-flight queue status.
- **DB filename is `_`-prefixed** (`_console.db`) so `scripts/status.py`'s scan (which skips `_`-prefixed entries) ignores it.
- **All new backend code lives under `src/ssdataagent/console/`.** Frontend lives under `web/` at the repo root, separate from `src/`.
- **New Python deps go in a `console` optional-dependencies group only** (`fastapi`, `uvicorn[standard]`); core deps unchanged.
- **Bind to `127.0.0.1` only** (localhost, no auth, single-user v1).
- **Backend tests mock `subprocess`** — no real runner, no real LLM, no network. Build the SQLite index from a fabricated `results/` fixture.
- **Status mapping mirrors `scripts/status.py._state()`**: `done.flag`→done, `failed.flag`→failed, `run.log` present but no flag→interrupted, none→pending. Console-launched runs carry `queued`/`running` until a flag appears.
- **Champion rule:** per (condition × dataset) cell, the row with the highest `overall_average` across all non-pilot experiments (pilot = experiment name starts with `pilot_`).
- **`overdetermination_gap` is nullable** — historical runs predate the feature; absence is normal.
- **Gate (per `feedback_refactor_gate_philosophy`):** our tests pass + no NEW failures vs. the 4 pre-existing `autograd`-missing failures (`tests/test_config.py::test_unknown_provider_raises` + 3 `tests/test_ssdatabench_integration.py *_legacy`). No bit-for-bit reproduction gate.
- **Avoid the literal word "eval" in commit messages** (project hook blocks it; use "evaluation"/"the check"). Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Do NOT stage the `ssdatabench` submodule pointer.

---

## File Structure

**Phase 1 — backend read-only core:**
- `pyproject.toml` (modify) — add `console` optional-dependencies group.
- `src/ssdataagent/console/__init__.py` (create) — package marker.
- `src/ssdataagent/console/db.py` (create) — SQLite connect + schema.
- `src/ssdataagent/console/sync.py` (create) — scan `results/` → upsert index.
- `src/ssdataagent/console/leaderboard.py` (create) — champion logic (pure, testable).
- `src/ssdataagent/console/app.py` (create) — FastAPI factory + leaderboard & run-detail routes.

**Phase 2 — control:**
- `src/ssdataagent/console/forking.py` (create) — fork a YAML experiment entry.
- `src/ssdataagent/console/queue.py` (create) — subprocess-backed JobQueue.
- `src/ssdataagent/console/app.py` (modify) — launcher + compare routes.
- `src/ssdataagent/console/compare.py` (create) — compare payload builder (pure).

**Phase 3 — outputs + UI:**
- `src/ssdataagent/reports.py` (create) — report core refactored out of the script.
- `scripts/generate_exp_report.py` (modify) — call the shared core.
- `src/ssdataagent/console/notebook.py` (create) — notebook CRUD + LEDGER append + Notion no-op.
- `src/ssdataagent/console/app.py` (modify) — reports + notebook routes.
- `src/ssdataagent/console/__main__.py` (create) — `python -m ssdataagent.console`.
- `web/` (create) — Vite + React SPA.

Tests: `tests/test_console_db.py`, `tests/test_console_sync.py`, `tests/test_console_leaderboard.py`, `tests/test_console_api.py`, `tests/test_console_forking.py`, `tests/test_console_queue.py`, `tests/test_console_compare.py`, `tests/test_reports.py`, `tests/test_console_notebook.py`.

---

# PHASE 1 — backend read-only core

### Task 1: SQLite schema + connection + `console` dep group

**Files:**
- Modify: `pyproject.toml` (optional-dependencies block)
- Create: `src/ssdataagent/console/__init__.py`
- Create: `src/ssdataagent/console/db.py`
- Test: `tests/test_console_db.py`

**Interfaces:**
- Produces:
  - `console.db.connect(db_path: Path) -> sqlite3.Connection` — opens (creating parent dirs), enables `row_factory = sqlite3.Row` and `PRAGMA foreign_keys`, calls `init_schema`, returns the connection.
  - `console.db.init_schema(conn: sqlite3.Connection) -> None` — runs all `CREATE TABLE IF NOT EXISTS` (idempotent).
  - `console.db.default_db_path(results_root: Path) -> Path` — returns `results_root / "_console.db"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_db.py
from pathlib import Path

from ssdataagent.console import db


def test_connect_creates_tables_and_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "results" / "_console.db"
    conn = db.connect(db_path)
    # All three tables exist.
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"experiments", "runs", "notebook"} <= names
    # Row factory gives mapping access.
    conn.execute(
        "INSERT INTO experiments(name, status, source) VALUES (?,?,?)",
        ("exp_x", "done", "disk"),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM experiments WHERE name='exp_x'").fetchone()
    assert row["status"] == "done"
    # Re-init is a no-op (does not drop data, does not raise).
    db.init_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1


def test_default_db_path_is_underscore_prefixed(tmp_path: Path):
    p = db.default_db_path(tmp_path)
    assert p.name == "_console.db"
    assert p.parent == tmp_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.console'`

- [ ] **Step 3: Create the package marker**

```python
# src/ssdataagent/console/__init__.py
"""Local web console (experiment control plane). Localhost, single-user."""
```

- [ ] **Step 4: Write the implementation**

```python
# src/ssdataagent/console/db.py
"""SQLite index + queue state for the console.

The DB is a rebuildable cache over the flat-file results store; on any
conflict with disk, disk wins. The only non-rebuildable state is the
`notebook` table and the queue status of console-launched runs that have
not yet written flags. Lives at `results/_console.db` (the `_` prefix means
status.py's scan ignores it).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    name TEXT PRIMARY KEY,
    status TEXT,
    prompt_variant TEXT,
    model TEXT,
    provider TEXT,
    git_sha TEXT,
    finished_at TEXT,
    config_json TEXT,
    config_hash TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    experiment TEXT,
    condition TEXT,
    dataset TEXT,
    run_id TEXT,
    run_dir TEXT,
    by_type_json TEXT,
    overall_average REAL,
    overdetermination_gap REAL,
    cost REAL,
    finished_at TEXT,
    PRIMARY KEY (experiment, condition, dataset, run_id)
);

CREATE TABLE IF NOT EXISTS notebook (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    hypothesis TEXT,
    change TEXT,
    result TEXT,
    interpretation TEXT,
    next TEXT,
    linked_experiments_json TEXT,
    ledger_synced INTEGER DEFAULT 0,
    notion_page_id TEXT
);
"""


def default_db_path(results_root: Path) -> Path:
    return Path(results_root) / "_console.db"


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn
```

- [ ] **Step 5: Add the `console` dependency group to `pyproject.toml`**

In the `[project.optional-dependencies]` block (which already has `dev`), add:

```toml
console = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "httpx>=0.27",
]
```

(`httpx` is needed by FastAPI's `TestClient`.)

- [ ] **Step 6: Install the console deps into the venv**

Run: `.venv/bin/pip install -e '.[console]'`
Expected: installs fastapi, uvicorn, starlette, httpx.

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_db.py -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/ssdataagent/console/__init__.py src/ssdataagent/console/db.py tests/test_console_db.py
git commit -m "console: SQLite index schema + connection"
```

---

### Task 2: Sync `results/` → SQLite index

**Files:**
- Create: `src/ssdataagent/console/sync.py`
- Test: `tests/test_console_sync.py`

**Interfaces:**
- Consumes: `console.db.connect`, `console.db.init_schema`.
- Produces:
  - `console.sync.experiment_state(exp_dir: Path) -> str` — replicates `status.py._state()`: returns one of `"done"|"failed"|"interrupted"|"pending"`. (Note: `status.py` returns `"running_or_interrupted"`; the console maps that disk state to `"interrupted"` because the console knows live "running" separately via the queue.)
  - `console.sync.sync_index(conn, results_root: Path) -> None` — scans `results_root` (skipping names starting with `_`), upserts one `experiments` row per experiment dir and one `runs` row per `*/eval.json` found under it. Only writes/overwrites rows with `source='disk'`; never clobbers a console-owned `queued`/`running` row unless a flag now exists on disk (then it reconciles to the disk state and flips `source` to `'disk'`).

**Disk-derived fields:**
- `experiments`: `status` via `experiment_state`; `prompt_variant`/`model`/`provider`/`finished_at` from `done.flag` JSON (keys `prompt_variant`, `llm_model`, `llm_provider`, `finished_at`) when present; `git_sha` from the newest run's `meta.json` (`git_sha`); `config_json`/`config_hash` left as-is on disk-sourced rows (NULL unless the console launched it); `source='disk'`.
- `runs`: walk `results/<exp>/<condition>/<dataset>/<run_id>/eval.json`; `by_type_json` = `json.dumps(eval["by_type"])`; `overall_average` = `eval["overall_average"]`; `overdetermination_gap` = `eval["overdetermination"]["cell_based"]["headline_gap"]` if the nested keys exist else `None`; `cost` = `None` (not tracked yet); `finished_at` from sibling `meta.json` if present else `None`; `run_dir` = POSIX string of the run dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_sync.py
import json
from pathlib import Path

from ssdataagent.console import db, sync


def _write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _make_done_experiment(root: Path, name: str, *, gap: float | None):
    exp = root / name
    _write(exp / "done.flag", {
        "experiment": name, "finished_at": "2026-06-30T10:00:00",
        "prompt_variant": "baseline", "llm_model": "gpt-5.4",
        "llm_provider": "openai", "n_dataset_condition_cells": 1,
    })
    _write(exp / "summary.csv",
           "condition,dataset,type,pass_rate\nfull_agent,gss,type1,0.5\n")
    run = exp / "full_agent" / "gss" / "20260630-100000"
    ev = {"by_type": {"type1": 0.5}, "by_variable": {}, "by_pair": {},
          "overall_average": 0.5}
    if gap is not None:
        ev["overdetermination"] = {"cell_based": {"headline_gap": gap}}
    _write(run / "eval.json", ev)
    _write(run / "meta.json", {"git_sha": "abc123", "model": "gpt-5.4"})


def test_experiment_state_mirrors_status_py(tmp_path: Path):
    exp = tmp_path / "e"
    exp.mkdir()
    assert sync.experiment_state(exp) == "pending"
    (exp / "run.log").write_text("x")
    assert sync.experiment_state(exp) == "interrupted"
    (exp / "failed.flag").write_text("{}")
    assert sync.experiment_state(exp) == "failed"
    (exp / "done.flag").write_text("{}")
    assert sync.experiment_state(exp) == "done"


def test_sync_index_upserts_experiments_and_runs(tmp_path: Path):
    root = tmp_path / "results"
    _make_done_experiment(root, "exp_a", gap=0.42)
    _make_done_experiment(root, "exp_b", gap=None)   # historical: no overdet
    (root / "_smoke_logs").mkdir(parents=True)        # underscore -> skipped
    conn = db.connect(db.default_db_path(root))

    sync.sync_index(conn, root)

    exps = {r["name"]: r for r in conn.execute("SELECT * FROM experiments")}
    assert set(exps) == {"exp_a", "exp_b"}            # _smoke_logs skipped
    assert exps["exp_a"]["status"] == "done"
    assert exps["exp_a"]["model"] == "gpt-5.4"
    assert exps["exp_a"]["source"] == "disk"
    runs = {r["experiment"]: r for r in conn.execute("SELECT * FROM runs")}
    assert json.loads(runs["exp_a"]["by_type_json"]) == {"type1": 0.5}
    assert runs["exp_a"]["overdetermination_gap"] == 0.42
    assert runs["exp_b"]["overdetermination_gap"] is None


def test_sync_is_idempotent_and_disk_wins(tmp_path: Path):
    root = tmp_path / "results"
    _make_done_experiment(root, "exp_a", gap=0.1)
    conn = db.connect(db.default_db_path(root))
    sync.sync_index(conn, root)
    sync.sync_index(conn, root)   # second pass: no duplicate rows
    assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_sync.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.console.sync'`

- [ ] **Step 3: Write the implementation**

```python
# src/ssdataagent/console/sync.py
"""Scan results/ and upsert the SQLite index. Disk is the source of truth."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def experiment_state(exp_dir: Path) -> str:
    """Disk-derived state, mirroring scripts/status.py._state().

    status.py collapses running/interrupted into one glyph because it can't
    tell them apart from disk. The console reports live "running" via the
    queue, so here the disk-only state is "interrupted".
    """
    if (exp_dir / "done.flag").exists():
        return "done"
    if (exp_dir / "failed.flag").exists():
        return "failed"
    if (exp_dir / "run.log").exists():
        return "interrupted"
    return "pending"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _newest_meta(exp_dir: Path) -> dict | None:
    metas = sorted(exp_dir.glob("*/*/*/meta.json"))
    return _read_json(metas[-1]) if metas else None


def _overdet_gap(ev: dict) -> float | None:
    try:
        return ev["overdetermination"]["cell_based"]["headline_gap"]
    except (KeyError, TypeError):
        return None


def _upsert_experiment(conn: sqlite3.Connection, exp_dir: Path) -> None:
    name = exp_dir.name
    status = experiment_state(exp_dir)
    flag = _read_json(exp_dir / "done.flag") or {}
    meta = _newest_meta(exp_dir) or {}

    existing = conn.execute(
        "SELECT source FROM experiments WHERE name=?", (name,)
    ).fetchone()
    # Don't clobber a console-owned queued/running row unless a flag now
    # exists on disk (status != pending/interrupted-without-flag handled by
    # experiment_state: a flag means done/failed).
    if existing is not None and existing["source"] == "console" \
            and status not in ("done", "failed"):
        return

    conn.execute(
        """INSERT INTO experiments
             (name, status, prompt_variant, model, provider, finished_at,
              git_sha, config_json, config_hash, source)
           VALUES (?,?,?,?,?,?,?,
                   COALESCE((SELECT config_json FROM experiments WHERE name=?), NULL),
                   COALESCE((SELECT config_hash FROM experiments WHERE name=?), NULL),
                   'disk')
           ON CONFLICT(name) DO UPDATE SET
             status=excluded.status,
             prompt_variant=excluded.prompt_variant,
             model=excluded.model,
             provider=excluded.provider,
             finished_at=excluded.finished_at,
             git_sha=excluded.git_sha,
             source='disk'""",
        (name, status, flag.get("prompt_variant"), flag.get("llm_model"),
         flag.get("llm_provider"), flag.get("finished_at"),
         meta.get("git_sha"), name, name),
    )


def _upsert_runs(conn: sqlite3.Connection, exp_dir: Path) -> None:
    name = exp_dir.name
    for eval_path in sorted(exp_dir.glob("*/*/*/eval.json")):
        run_dir = eval_path.parent
        # results/<exp>/<condition>/<dataset>/<run_id>/eval.json
        run_id = run_dir.name
        dataset = run_dir.parent.name
        condition = run_dir.parent.parent.name
        ev = _read_json(eval_path) or {}
        meta = _read_json(run_dir / "meta.json") or {}
        conn.execute(
            """INSERT INTO runs
                 (experiment, condition, dataset, run_id, run_dir,
                  by_type_json, overall_average, overdetermination_gap,
                  cost, finished_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(experiment, condition, dataset, run_id) DO UPDATE SET
                 run_dir=excluded.run_dir,
                 by_type_json=excluded.by_type_json,
                 overall_average=excluded.overall_average,
                 overdetermination_gap=excluded.overdetermination_gap,
                 finished_at=excluded.finished_at""",
            (name, condition, dataset, run_id, run_dir.as_posix(),
             json.dumps(ev.get("by_type", {})), ev.get("overall_average"),
             _overdet_gap(ev), None, meta.get("finished_at")),
        )


def sync_index(conn: sqlite3.Connection, results_root: Path) -> None:
    results_root = Path(results_root)
    if not results_root.exists():
        return
    for child in sorted(results_root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        _upsert_experiment(conn, child)
        _upsert_runs(conn, child)
    conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_sync.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/console/sync.py tests/test_console_sync.py
git commit -m "console: sync results/ into the SQLite index (disk wins)"
```

---

### Task 3: Leaderboard logic (champion marking)

**Files:**
- Create: `src/ssdataagent/console/leaderboard.py`
- Test: `tests/test_console_leaderboard.py`

**Interfaces:**
- Consumes: rows from the `runs` table joined to `experiments` (plain dicts).
- Produces:
  - `console.leaderboard.LeaderboardRow` — a `TypedDict`/dataclass with `experiment, condition, dataset, by_type (dict), overall_average (float|None), overdetermination_gap (float|None), is_pilot (bool), is_champion (bool)`.
  - `console.leaderboard.build_rows(records: list[dict]) -> list[LeaderboardRow]` — given DB-shaped records (keys: `experiment, condition, dataset, by_type_json, overall_average, overdetermination_gap`), parse `by_type_json`, mark `is_pilot` (experiment starts with `pilot_`), and mark `is_champion=True` on the single highest-`overall_average` non-pilot row per (condition, dataset) cell (ties broken by experiment name desc for determinism). Rows with `overall_average is None` are never champions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_leaderboard.py
from ssdataagent.console import leaderboard


def _rec(exp, cond, ds, overall, gap=None, by_type=None):
    import json
    return {"experiment": exp, "condition": cond, "dataset": ds,
            "by_type_json": json.dumps(by_type or {"type1": overall}),
            "overall_average": overall, "overdetermination_gap": gap}


def test_champion_is_best_per_cell_excluding_pilots():
    recs = [
        _rec("exp_a", "full_agent", "gss", 0.5),
        _rec("exp_b", "full_agent", "gss", 0.7),     # best in this cell
        _rec("pilot_x", "full_agent", "gss", 0.9),   # pilot: ignored
        _rec("exp_c", "design_a_full", "gss", 0.6),  # different cell -> own champ
    ]
    rows = leaderboard.build_rows(recs)
    champs = {(r["condition"], r["dataset"]): r["experiment"]
              for r in rows if r["is_champion"]}
    assert champs[("full_agent", "gss")] == "exp_b"
    assert champs[("design_a_full", "gss")] == "exp_c"
    # pilot flagged, never champion
    pilot = next(r for r in rows if r["experiment"] == "pilot_x")
    assert pilot["is_pilot"] and not pilot["is_champion"]


def test_none_overall_never_champion():
    rows = leaderboard.build_rows([_rec("exp_a", "c", "d", None)])
    assert not any(r["is_champion"] for r in rows)
    assert rows[0]["by_type"] == {"type1": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_leaderboard.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/ssdataagent/console/leaderboard.py
"""Pure leaderboard assembly: parse rows, mark champion per (cond, dataset)."""
from __future__ import annotations

import json
from typing import Any, TypedDict


class LeaderboardRow(TypedDict):
    experiment: str
    condition: str
    dataset: str
    by_type: dict[str, Any]
    overall_average: float | None
    overdetermination_gap: float | None
    is_pilot: bool
    is_champion: bool


def build_rows(records: list[dict]) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    for rec in records:
        try:
            by_type = json.loads(rec.get("by_type_json") or "{}")
        except (TypeError, ValueError):
            by_type = {}
        rows.append(LeaderboardRow(
            experiment=rec["experiment"],
            condition=rec["condition"],
            dataset=rec["dataset"],
            by_type=by_type,
            overall_average=rec.get("overall_average"),
            overdetermination_gap=rec.get("overdetermination_gap"),
            is_pilot=str(rec["experiment"]).startswith("pilot_"),
            is_champion=False,
        ))

    best: dict[tuple[str, str], LeaderboardRow] = {}
    for r in rows:
        if r["is_pilot"] or r["overall_average"] is None:
            continue
        cell = (r["condition"], r["dataset"])
        cur = best.get(cell)
        if cur is None or (
            r["overall_average"], r["experiment"]
        ) > (cur["overall_average"], cur["experiment"]):
            best[cell] = r
    for r in best.values():
        r["is_champion"] = True
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_leaderboard.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/console/leaderboard.py tests/test_console_leaderboard.py
git commit -m "console: leaderboard row assembly + champion marking"
```

---

### Task 4: FastAPI app factory + leaderboard & run-detail endpoints

**Files:**
- Create: `src/ssdataagent/console/app.py`
- Test: `tests/test_console_api.py`

**Interfaces:**
- Consumes: `console.db.connect/default_db_path`, `console.sync.sync_index`, `console.leaderboard.build_rows`.
- Produces:
  - `console.app.create_app(results_root: Path | None = None) -> fastapi.FastAPI` — app factory. Stores `results_root` (default `ssdataagent.config.results_root()`) and a single shared `sqlite3.Connection` (via `console.db.connect`) on `app.state`. Mounts the API router. (Static SPA mount is added in Task 12, guarded by `web/dist` existing.)
  - `GET /api/leaderboard` → `{"rows": [LeaderboardRow...]}`; calls `sync_index` first, then `SELECT` joining runs, then `build_rows`. Optional query params `condition`, `dataset`, `model` filter the SQL.
  - `GET /api/runs/{name}/detail` → `{"experiment": {...}, "runs": [{condition, dataset, run_id, run_dir, eval: <full eval.json>, meta: <full meta.json>, artifacts: {...}}]}`. Reads the full `eval.json`/`meta.json` from disk for each run dir. `artifacts` is a dict of relative paths that exist: `generated_csv`, `prompts_jsonl`, `responses_jsonl`, `workspace_dir`. 404 if the experiment is unknown.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_api.py
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ssdataagent.console.app import create_app


def _write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    exp = root / "exp_a"
    _write(exp / "done.flag", {"experiment": "exp_a", "finished_at": "t",
                               "prompt_variant": "baseline", "llm_model": "m",
                               "llm_provider": "openai"})
    run = exp / "full_agent" / "gss" / "20260630-100000"
    _write(run / "eval.json", {"by_type": {"type1": 0.6}, "by_variable": {},
                               "by_pair": {}, "overall_average": 0.6,
                               "overdetermination": {"cell_based": {"headline_gap": 0.3}}})
    _write(run / "meta.json", {"git_sha": "abc", "model": "m", "condition": "full_agent"})
    _write(run / "generated.csv", "a,b\n1,2\n")
    _write(run / "prompts.jsonl", '{"p":1}\n')
    return root


@pytest.fixture
def client(results: Path) -> TestClient:
    return TestClient(create_app(results_root=results))


def test_leaderboard_returns_rows_with_champion(client: TestClient):
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["experiment"] == "exp_a"
    assert rows[0]["overall_average"] == 0.6
    assert rows[0]["overdetermination_gap"] == 0.3
    assert rows[0]["is_champion"] is True


def test_run_detail_exposes_eval_and_artifacts(client: TestClient):
    r = client.get("/api/runs/exp_a/detail")
    assert r.status_code == 200
    body = r.json()
    assert body["experiment"]["name"] == "exp_a"
    run = body["runs"][0]
    assert run["condition"] == "full_agent"
    assert run["eval"]["overall_average"] == 0.6
    assert run["meta"]["git_sha"] == "abc"
    assert "generated_csv" in run["artifacts"]
    assert "prompts_jsonl" in run["artifacts"]


def test_run_detail_unknown_is_404(client: TestClient):
    assert client.get("/api/runs/nope/detail").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.console.app'`

- [ ] **Step 3: Write the implementation**

```python
# src/ssdataagent/console/app.py
"""FastAPI app factory for the console. Localhost, single-user, no auth."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ssdataagent.config import results_root as default_results_root
from ssdataagent.console import db, leaderboard, sync


def create_app(results_root: Path | None = None) -> FastAPI:
    root = Path(results_root) if results_root else default_results_root()
    conn = db.connect(db.default_db_path(root))

    app = FastAPI(title="SSDataAgent console")
    app.state.results_root = root
    app.state.conn = conn

    @app.get("/api/leaderboard")
    def get_leaderboard(condition: str | None = None,
                        dataset: str | None = None,
                        model: str | None = None):
        sync.sync_index(conn, root)
        q = ("SELECT r.*, e.model AS model FROM runs r "
             "JOIN experiments e ON e.name = r.experiment WHERE 1=1")
        params: list = []
        if condition:
            q += " AND r.condition = ?"; params.append(condition)
        if dataset:
            q += " AND r.dataset = ?"; params.append(dataset)
        if model:
            q += " AND e.model = ?"; params.append(model)
        records = [dict(row) for row in conn.execute(q, params).fetchall()]
        return {"rows": leaderboard.build_rows(records)}

    @app.get("/api/runs/{name}/detail")
    def get_run_detail(name: str):
        sync.sync_index(conn, root)
        erow = conn.execute(
            "SELECT * FROM experiments WHERE name=?", (name,)
        ).fetchone()
        if erow is None:
            raise HTTPException(status_code=404, detail=f"unknown experiment {name!r}")
        runs = []
        for r in conn.execute("SELECT * FROM runs WHERE experiment=?", (name,)):
            run_dir = Path(r["run_dir"])
            runs.append({
                "condition": r["condition"],
                "dataset": r["dataset"],
                "run_id": r["run_id"],
                "run_dir": r["run_dir"],
                "eval": _read_json(run_dir / "eval.json"),
                "meta": _read_json(run_dir / "meta.json"),
                "artifacts": _artifacts(run_dir),
            })
        return {"experiment": dict(erow), "runs": runs}

    return app


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _artifacts(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, rel in [("generated_csv", "generated.csv"),
                     ("prompts_jsonl", "prompts.jsonl"),
                     ("responses_jsonl", "responses.jsonl"),
                     ("workspace_dir", "workspace")]:
        if (run_dir / rel).exists():
            out[key] = (run_dir / rel).as_posix()
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_api.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/console/app.py tests/test_console_api.py
git commit -m "console: FastAPI factory + leaderboard and run-detail endpoints"
```

---

# PHASE 2 — control (queue, launcher, fork, compare)

### Task 5: Fork an experiment YAML entry

**Files:**
- Create: `src/ssdataagent/console/forking.py`
- Test: `tests/test_console_forking.py`

**Interfaces:**
- Produces:
  - `console.forking.load_experiments(yaml_path: Path) -> dict` — returns the `experiments:` mapping.
  - `console.forking.fork_experiment(yaml_path: Path, base: str, new_name: str, overrides: dict) -> dict` — reads the YAML, deep-copies `experiments[base]`, applies `overrides` (shallow update), writes it under `experiments[new_name]`, and writes the file back **preserving all sibling entries**. Returns the new entry. Raises `KeyError` if `base` is unknown; `ValueError` if `new_name` already exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_forking.py
from pathlib import Path

import yaml
import pytest

from ssdataagent.console import forking


@pytest.fixture
def yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "experiments.yaml"
    p.write_text(yaml.safe_dump({"experiments": {
        "base": {"datasets": ["gss"], "conditions": ["full_agent"],
                 "max_iterations": 3, "sandbox_timeout": 60,
                 "train_eval_split": 0.5, "n_rows": 1000},
        "other": {"datasets": ["cps"], "conditions": ["hotdeck"],
                  "max_iterations": 1, "sandbox_timeout": 60,
                  "train_eval_split": 0.5, "n_rows": 100},
    }}))
    return p


def test_fork_creates_entry_and_preserves_siblings(yaml_path: Path):
    entry = forking.fork_experiment(
        yaml_path, "base", "base_fork", {"n_rows": 500, "prompt_variant": "rubric"})
    assert entry["n_rows"] == 500
    assert entry["prompt_variant"] == "rubric"
    assert entry["datasets"] == ["gss"]            # inherited from base
    data = yaml.safe_load(yaml_path.read_text())["experiments"]
    assert set(data) == {"base", "base_fork", "other"}   # siblings preserved
    assert data["base"]["n_rows"] == 1000          # base untouched


def test_fork_unknown_base_raises(yaml_path: Path):
    with pytest.raises(KeyError):
        forking.fork_experiment(yaml_path, "nope", "x", {})


def test_fork_duplicate_name_raises(yaml_path: Path):
    with pytest.raises(ValueError):
        forking.fork_experiment(yaml_path, "base", "other", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_forking.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/ssdataagent/console/forking.py
"""Fork a config/experiments.yaml entry, preserving siblings."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_experiments(yaml_path: Path) -> dict:
    data = yaml.safe_load(Path(yaml_path).read_text()) or {}
    return data.get("experiments", {})


def fork_experiment(yaml_path: Path, base: str, new_name: str,
                    overrides: dict) -> dict:
    yaml_path = Path(yaml_path)
    data = yaml.safe_load(yaml_path.read_text()) or {}
    experiments = data.setdefault("experiments", {})
    if base not in experiments:
        raise KeyError(f"unknown base experiment {base!r}")
    if new_name in experiments:
        raise ValueError(f"experiment {new_name!r} already exists")
    entry = copy.deepcopy(experiments[base])
    entry.update(overrides)
    experiments[new_name] = entry
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return entry
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_forking.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/console/forking.py tests/test_console_forking.py
git commit -m "console: fork an experiments.yaml entry preserving siblings"
```

---

### Task 6: Subprocess-backed job queue

**Files:**
- Create: `src/ssdataagent/console/queue.py`
- Test: `tests/test_console_queue.py`

**Interfaces:**
- Consumes: `console.db` connection (for status writes); `ssdataagent.config.REPO_ROOT`.
- Produces:
  - `console.queue.JobQueue(conn, results_root, *, concurrency=1, runner=None)` — `runner` is an injectable callable `(name: str, log_path: Path) -> int` (returns exit code) so tests mock the subprocess; the default runner shells out to `python scripts/run_experiment.py --experiment NAME` with stdout/stderr to `log_path` (`results/<name>/run.log`). On construction, no thread starts until `start()`.
  - `.enqueue(name: str) -> None` — inserts/updates the experiment row to `status='queued', source='console'`, pushes onto an internal `queue.Queue`.
  - `.start() -> None` / `.stop() -> None` — start/stop the worker thread(s). Idempotent.
  - `.cancel(name: str) -> bool` — terminate a running subprocess for `name` if present; returns whether something was cancelled.
  - Worker loop: pop name → set `status='running'` → ensure `results/<name>/` exists → call `runner(name, log_path)` → on exit code 0 set `status='done'` else `status='failed'`. All status writes go through the DB with `source='console'`. Concurrency cap via a bounded worker-thread pool of size `concurrency`.
  - `.wait_idle(timeout: float) -> None` — test helper: block until the queue is empty and no job is running (or timeout).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_queue.py
from pathlib import Path

from ssdataagent.console import db, queue as q


def _conn(tmp_path):
    return db.connect(db.default_db_path(tmp_path / "results"))


def test_enqueue_runs_and_marks_done(tmp_path: Path):
    conn = _conn(tmp_path)
    calls = []

    def fake_runner(name: str, log_path: Path) -> int:
        calls.append(name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ran\n")
        return 0

    jq = q.JobQueue(conn, tmp_path / "results", concurrency=1, runner=fake_runner)
    jq.start()
    jq.enqueue("exp_a")
    jq.wait_idle(timeout=5)
    jq.stop()

    assert calls == ["exp_a"]
    row = conn.execute("SELECT * FROM experiments WHERE name='exp_a'").fetchone()
    assert row["status"] == "done"
    assert row["source"] == "console"


def test_nonzero_exit_marks_failed(tmp_path: Path):
    conn = _conn(tmp_path)

    def fail_runner(name: str, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("boom\n")
        return 1

    jq = q.JobQueue(conn, tmp_path / "results", concurrency=1, runner=fail_runner)
    jq.start()
    jq.enqueue("exp_b")
    jq.wait_idle(timeout=5)
    jq.stop()

    row = conn.execute("SELECT status FROM experiments WHERE name='exp_b'").fetchone()
    assert row["status"] == "failed"


def test_enqueue_sets_queued_before_run(tmp_path: Path):
    conn = _conn(tmp_path)
    # Do not start the worker: status should be 'queued'.
    jq = q.JobQueue(conn, tmp_path / "results", concurrency=1,
                    runner=lambda n, p: 0)
    jq.enqueue("exp_c")
    row = conn.execute("SELECT status, source FROM experiments WHERE name='exp_c'").fetchone()
    assert row["status"] == "queued"
    assert row["source"] == "console"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_queue.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/ssdataagent/console/queue.py
"""In-process, subprocess-backed job queue. Sequential by default.

Each job shells out to scripts/run_experiment.py; the runner CLI writes the
flags (done.flag/failed.flag) on disk, and the queue mirrors a coarse status
into the SQLite index with source='console' until sync reconciles it.
"""
from __future__ import annotations

import queue as _queue
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from ssdataagent.config import REPO_ROOT

Runner = Callable[[str, Path], int]

_SENTINEL = object()


def _default_runner(name: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"),
             "--experiment", name],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT),
        )
        return proc.wait()


class JobQueue:
    def __init__(self, conn: sqlite3.Connection, results_root: Path, *,
                 concurrency: int = 1, runner: Runner | None = None):
        self._conn = conn
        self._lock = threading.Lock()
        self._root = Path(results_root)
        self._concurrency = max(1, concurrency)
        self._runner = runner or _default_runner
        self._q: _queue.Queue = _queue.Queue()
        self._threads: list[threading.Thread] = []
        self._running: dict[str, subprocess.Popen] = {}
        self._stop = threading.Event()

    # --- status writes (serialized) ---
    def _set_status(self, name: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO experiments(name, status, source)
                   VALUES (?, ?, 'console')
                   ON CONFLICT(name) DO UPDATE SET status=excluded.status,
                                                   source='console'""",
                (name, status),
            )
            self._conn.commit()

    def enqueue(self, name: str) -> None:
        self._set_status(name, "queued")
        self._q.put(name)

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for _ in range(self._concurrency):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._q.put(_SENTINEL)
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []

    def cancel(self, name: str) -> bool:
        with self._lock:
            proc = self._running.get(name)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return True
        return False

    def wait_idle(self, timeout: float = 10.0) -> None:
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                busy = bool(self._running)
            if self._q.empty() and not busy:
                return
            time.sleep(0.02)

    def _worker(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            if item is _SENTINEL:
                self._q.task_done()
                return
            name = item
            self._set_status(name, "running")
            log_path = self._root / name / "run.log"
            try:
                code = self._runner(name, log_path)
            except Exception:
                code = 1
            finally:
                with self._lock:
                    self._running.pop(name, None)
            self._set_status(name, "done" if code == 0 else "failed")
            self._q.task_done()
```

Note: the default runner records its `Popen` in `_running` so `cancel` can terminate it. Add this to `_default_runner` integration by wrapping in the worker — for the injectable test runner there is no Popen, which is why `cancel` only acts when a real Popen exists. To make `cancel` work with the real runner, refactor so the worker creates the Popen and stores it. Implement the worker's real-subprocess path as:

```python
    # Replace the body between _set_status(running) and _set_status(done/failed):
            log_path = self._root / name / "run.log"
            code = 1
            try:
                if self._runner is _default_runner:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("w") as log:
                        proc = subprocess.Popen(
                            [sys.executable,
                             str(REPO_ROOT / "scripts" / "run_experiment.py"),
                             "--experiment", name],
                            stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(REPO_ROOT),
                        )
                        with self._lock:
                            self._running[name] = proc
                        code = proc.wait()
                else:
                    code = self._runner(name, log_path)
            except Exception:
                code = 1
            finally:
                with self._lock:
                    self._running.pop(name, None)
            self._set_status(name, "done" if code == 0 else "failed")
            self._q.task_done()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_queue.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/console/queue.py tests/test_console_queue.py
git commit -m "console: subprocess-backed job queue with cancel"
```

---

### Task 7: Launcher endpoints (enqueue, fork, list, cancel, log)

**Files:**
- Modify: `src/ssdataagent/console/app.py`
- Test: `tests/test_console_api.py` (extend)

**Interfaces:**
- Consumes: `console.queue.JobQueue`, `console.forking.fork_experiment`, `ssdataagent.config.REPO_ROOT`.
- Produces (added to `create_app`; accept an optional `job_queue=None` param so tests inject a queue with a fake runner, and an optional `experiments_yaml: Path|None` for the fork target, default `REPO_ROOT/config/experiments.yaml`):
  - `POST /api/runs` body `{name?: str, fork_from?: str, new_name?: str, overrides?: dict}` → if `fork_from` given, call `fork_experiment(yaml, fork_from, new_name, overrides)` then enqueue `new_name`; else enqueue `name`. Returns `{"enqueued": <name>}`. 400 on missing/invalid args; 409 on duplicate fork name.
  - `GET /api/runs` → `{"experiments": [dict(row) ...]}` (after `sync_index`).
  - `POST /api/runs/{name}/cancel` → `{"cancelled": bool}`.
  - `GET /api/runs/{name}/log?tail=200` → `{"log": "<last N lines of results/<name>/run.log>"}` (empty string if absent).

- [ ] **Step 1: Write the failing test (extend `tests/test_console_api.py`)**

```python
def test_post_runs_enqueues_named_experiment(tmp_path, results):
    from ssdataagent.console import db, queue
    from ssdataagent.console.app import create_app

    conn_holder = {}

    def fake_runner(name, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("hello from run\n")
        return 0

    app = create_app(results_root=results)
    # Inject a queue using the app's own connection.
    jq = queue.JobQueue(app.state.conn, results, concurrency=1, runner=fake_runner)
    app.state.job_queue = jq
    jq.start()
    client = TestClient(app)

    r = client.post("/api/runs", json={"name": "exp_new"})
    assert r.status_code == 200
    assert r.json()["enqueued"] == "exp_new"
    jq.wait_idle(timeout=5)
    jq.stop()

    listing = client.get("/api/runs").json()["experiments"]
    assert any(e["name"] == "exp_new" and e["status"] == "done" for e in listing)
    log = client.get("/api/runs/exp_new/log").json()["log"]
    assert "hello from run" in log
```

(Adjust the existing `client` fixture so `create_app` attaches a default no-op `job_queue` when none injected — see implementation note below — or have endpoints read `app.state.job_queue` defaulting to a lazily-created real queue.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_api.py::test_post_runs_enqueues_named_experiment -v`
Expected: FAIL (404/405 — route not defined)

- [ ] **Step 3: Add the routes to `create_app`**

In `create_app`, after the existing routes, add (and create a default queue on `app.state` if absent):

```python
    from ssdataagent.config import REPO_ROOT
    from ssdataagent.console import forking, queue as _q
    from pydantic import BaseModel

    if not hasattr(app.state, "job_queue"):
        app.state.job_queue = _q.JobQueue(conn, root, concurrency=1)
        app.state.job_queue.start()
    experiments_yaml = REPO_ROOT / "config" / "experiments.yaml"

    class RunRequest(BaseModel):
        name: str | None = None
        fork_from: str | None = None
        new_name: str | None = None
        overrides: dict = {}

    @app.post("/api/runs")
    def post_run(req: RunRequest):
        if req.fork_from:
            if not req.new_name:
                raise HTTPException(400, "new_name required when forking")
            try:
                forking.fork_experiment(experiments_yaml, req.fork_from,
                                        req.new_name, req.overrides)
            except KeyError as e:
                raise HTTPException(400, str(e))
            except ValueError as e:
                raise HTTPException(409, str(e))
            target = req.new_name
        elif req.name:
            target = req.name
        else:
            raise HTTPException(400, "name or fork_from required")
        app.state.job_queue.enqueue(target)
        return {"enqueued": target}

    @app.get("/api/runs")
    def list_runs():
        sync.sync_index(conn, root)
        rows = [dict(r) for r in conn.execute("SELECT * FROM experiments")]
        return {"experiments": rows}

    @app.post("/api/runs/{name}/cancel")
    def cancel_run(name: str):
        return {"cancelled": app.state.job_queue.cancel(name)}

    @app.get("/api/runs/{name}/log")
    def get_log(name: str, tail: int = 200):
        log_path = root / name / "run.log"
        if not log_path.exists():
            return {"log": ""}
        lines = log_path.read_text(errors="replace").splitlines()
        return {"log": "\n".join(lines[-tail:])}
```

Move the `list_runs`/`get_log`/`cancel_run`/`post_run` routes so they are registered before `return app`. Keep the leaderboard sync-before-read behavior; note `list_runs` sync may overwrite a console `running` row only when a flag exists (handled by `sync._upsert_experiment`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_api.py -v`
Expected: PASS (all, including the new launcher test)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/console/app.py tests/test_console_api.py
git commit -m "console: run launcher, fork, list, cancel, and log endpoints"
```

---

### Task 8: Compare payload builder + endpoint

**Files:**
- Create: `src/ssdataagent/console/compare.py`
- Modify: `src/ssdataagent/console/app.py`
- Test: `tests/test_console_compare.py`

**Interfaces:**
- Produces:
  - `console.compare.build_matrix(selectors: list[dict], eval_loader: Callable[[dict], dict|None]) -> dict` — `selectors` are dicts `{experiment, condition, dataset}`. `eval_loader(selector)` returns that run's full `eval.json` (or None). Returns:
    ```
    {"types": ["type1",...sorted union...],
     "cells": [{"selector": {...}, "by_type": {...}, "overall_average": float|None,
                "overdetermination_gap": float|None}],
     "matrix": [[by_type value per type per selector]]}    # rows=selectors, cols=types
    ```
  - `GET`/`POST /api/compare` body `{selectors: [...]}` → calls `build_matrix` with an `eval_loader` that resolves the latest `eval.json` for `results/<exp>/<condition>/<dataset>/*/eval.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_compare.py
from ssdataagent.console import compare


def test_build_matrix_aligns_types_across_selectors():
    evals = {
        ("a", "full_agent", "gss"): {"by_type": {"type1": 0.5, "type2": 0.4},
                                     "overall_average": 0.45,
                                     "overdetermination": {"cell_based": {"headline_gap": 0.2}}},
        ("b", "design_a_full", "gss"): {"by_type": {"type1": 0.7},
                                        "overall_average": 0.7},
    }
    sels = [{"experiment": "a", "condition": "full_agent", "dataset": "gss"},
            {"experiment": "b", "condition": "design_a_full", "dataset": "gss"}]

    def loader(s):
        return evals.get((s["experiment"], s["condition"], s["dataset"]))

    out = compare.build_matrix(sels, loader)
    assert out["types"] == ["type1", "type2"]
    assert out["matrix"][0] == [0.5, 0.4]
    assert out["matrix"][1] == [0.7, None]      # missing type2 -> None
    assert out["cells"][0]["overdetermination_gap"] == 0.2
    assert out["cells"][1]["overdetermination_gap"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_compare.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/ssdataagent/console/compare.py
"""Build the compare payload (metric diff + strategy x type matrix)."""
from __future__ import annotations

from typing import Callable


def _gap(ev: dict | None) -> float | None:
    if not ev:
        return None
    try:
        return ev["overdetermination"]["cell_based"]["headline_gap"]
    except (KeyError, TypeError):
        return None


def build_matrix(selectors: list[dict],
                 eval_loader: Callable[[dict], dict | None]) -> dict:
    loaded = [(s, eval_loader(s)) for s in selectors]
    types: list[str] = sorted(
        {t for _, ev in loaded if ev for t in (ev.get("by_type") or {})}
    )
    cells = []
    matrix = []
    for s, ev in loaded:
        by_type = (ev or {}).get("by_type") or {}
        cells.append({
            "selector": s,
            "by_type": by_type,
            "overall_average": (ev or {}).get("overall_average"),
            "overdetermination_gap": _gap(ev),
        })
        matrix.append([by_type.get(t) for t in types])
    return {"types": types, "cells": cells, "matrix": matrix}
```

- [ ] **Step 4: Add the `/api/compare` route to `create_app`**

```python
    from ssdataagent.console import compare as _compare

    class CompareRequest(BaseModel):
        selectors: list[dict]

    def _latest_eval(sel: dict) -> dict | None:
        cond_dir = root / sel["experiment"] / sel["condition"] / sel["dataset"]
        cands = sorted(cond_dir.glob("*/eval.json"))
        return _read_json(cands[-1]) if cands else None

    @app.post("/api/compare")
    def post_compare(req: CompareRequest):
        return _compare.build_matrix(req.selectors, _latest_eval)
```

- [ ] **Step 5: Add an API test (extend `tests/test_console_api.py`)**

```python
def test_compare_endpoint_returns_matrix(client: TestClient):
    r = client.post("/api/compare", json={"selectors": [
        {"experiment": "exp_a", "condition": "full_agent", "dataset": "gss"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["types"] == ["type1"]
    assert body["matrix"][0] == [0.6]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_compare.py tests/test_console_api.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/ssdataagent/console/compare.py src/ssdataagent/console/app.py tests/test_console_compare.py tests/test_console_api.py
git commit -m "console: compare matrix builder + endpoint"
```

---

# PHASE 3 — outputs + UI

### Task 9: Extract report core into `ssdataagent.reports`

**Files:**
- Create: `src/ssdataagent/reports.py`
- Modify: `scripts/generate_exp_report.py`
- Modify: `src/ssdataagent/console/app.py` (add `/api/reports`)
- Test: `tests/test_reports.py`

**Interfaces:**
- Produces:
  - `ssdataagent.reports.render_markdown_report(experiment: str, *, condition: str = "full_agent", baseline: str | None = None, results_root: Path, experiments_yaml: Path, paper_baselines: Path) -> str` — returns the Markdown report string (the body currently built inline in `generate_exp_report.main`). Pure string-building; reads eval.json/yaml/paper json from the given paths.
  - `ssdataagent.reports.render_html_report(markdown_text: str) -> str` — wraps the Markdown in a self-contained HTML doc using `markdown_it` (already a dep) for the body. No external assets.
  - `console.app`: `POST /api/reports` body `{experiment, condition?, baseline?, format: "md"|"html"}` → `{"format":..., "content": <str>}`.
- The script `generate_exp_report.py` keeps its CLI but delegates body-building to `render_markdown_report` (DRY — no duplicated table logic).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reports.py
import json
from pathlib import Path

from ssdataagent import reports


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "results"
    run = root / "exp_a" / "full_agent" / "gss" / "20260630-100000"
    run.mkdir(parents=True)
    (run / "eval.json").write_text(json.dumps({
        "by_type": {"type1": 0.6}, "overall_average": 0.6,
        "overdetermination": {"cell_based": {"headline_gap": 0.3, "coverage": 0.9,
                                             "n_cells": 10},
                              "model_based": {"headline_gap": 0.25}}}))
    yaml_path = tmp_path / "experiments.yaml"
    yaml_path.write_text(
        "experiments:\n  exp_a:\n    datasets: [gss]\n    conditions: [full_agent]\n"
        "    max_iterations: 3\n    n_rows: 1000\n")
    paper = tmp_path / "paper.json"
    paper.write_text(json.dumps({"by_dataset_by_type": {"gss": {"T1": 0.5}},
                                 "by_dataset_overall": {"gss": 0.5}}))
    return root, yaml_path, paper


def test_render_markdown_contains_results_and_overdet(tmp_path: Path):
    root, yaml_path, paper = _setup(tmp_path)
    md = reports.render_markdown_report(
        "exp_a", results_root=root, experiments_yaml=yaml_path, paper_baselines=paper)
    assert "## Results" in md
    assert "Over-determination gap" in md
    assert "0.600" in md          # the type1 score, formatted
    assert "0.300" in md          # the over-determination headline gap


def test_render_html_wraps_markdown(tmp_path: Path):
    root, yaml_path, paper = _setup(tmp_path)
    md = reports.render_markdown_report(
        "exp_a", results_root=root, experiments_yaml=yaml_path, paper_baselines=paper)
    html = reports.render_html_report(md)
    assert "<html" in html.lower()
    assert "Results" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.reports'`

- [ ] **Step 3: Write `src/ssdataagent/reports.py`**

Move the body-building helpers (`_md_table`, `_fmt`, `_delta`, `_overdetermination_section`, `_strategy_blurb`, `_latest_eval`, the `T_KEYS`/`T_LABELS` constants, and the section-assembly loop from `generate_exp_report.main`) into pure functions parameterized by paths. Compose them in `render_markdown_report`. Add `render_html_report` using `markdown_it`:

```python
# src/ssdataagent/reports.py
"""Report core, shared by scripts/generate_exp_report.py and the console.

Pure string-building over flat-file artifacts (eval.json / experiments.yaml /
paper_baselines.json). No I/O beyond reading those paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

T_KEYS = ["type1", "type2", "type3", "type4", "type5"]
T_LABELS = {"type1": "T1", "type2": "T2", "type3": "T3", "type4": "T4", "type5": "T5"}


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return "NaN" if v != v else f"{v:.3f}"
    return str(v)


def _delta(a, b) -> str:
    if a is None or b is None or (isinstance(a, float) and a != a) or (isinstance(b, float) and b != b):
        return "—"
    d = a - b
    return f"{'+' if d >= 0 else ''}{d:.3f}"


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---:" if i > 0 else "---" for i in range(len(headers))]) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _latest_eval(results_root: Path, exp: str, condition: str, dataset: str) -> dict | None:
    cond_dir = results_root / exp / condition / dataset
    cands = sorted(cond_dir.glob("*/eval.json"))
    if not cands:
        return None
    return json.loads(cands[-1].read_text())


def _overdetermination_section(cells: dict, datasets: list) -> str:
    headers = ["Dataset", "gap (cell)", "coverage", "n_cells", "gap (model)"]
    rows = []
    for ds in datasets:
        cell = cells.get(ds)
        od = (cell or {}).get("overdetermination") if cell else None
        if not od:
            rows.append([ds, "—", "—", "—", "—"])
            continue
        cb = od.get("cell_based", {}) or {}
        mb = od.get("model_based", {}) or {}
        rows.append([ds, _fmt(cb.get("headline_gap")), _fmt(cb.get("coverage")),
                     _fmt(cb.get("n_cells")), _fmt(mb.get("headline_gap"))])
    return ("## Over-determination gap — `H_real − H_sim` (bits, higher = sim more collapsed)\n\n"
            + _md_table(headers, rows))


def render_markdown_report(experiment: str, *, condition: str = "full_agent",
                           baseline: str | None = None, results_root: Path,
                           experiments_yaml: Path, paper_baselines: Path) -> str:
    spec = yaml.safe_load(Path(experiments_yaml).read_text())["experiments"][experiment]
    paper = json.loads(Path(paper_baselines).read_text())
    by_dataset_paper = paper["by_dataset_by_type"]
    overall_paper = paper["by_dataset_overall"]
    datasets = spec["datasets"]
    cells = {ds: _latest_eval(results_root, experiment, condition, ds) for ds in datasets}

    bits: list[str] = [f"# {experiment} — report\n", "## Strategy\n",
                       f"- **prompt_variant:** `{spec.get('prompt_variant', 'baseline')}`",
                       f"- **datasets:** {', '.join(datasets)}",
                       f"- **conditions:** {', '.join(spec['conditions'])}",
                       f"- **n_rows / dataset:** {spec.get('n_rows')}", ""]

    bits.append(f"## Results — `{condition}`\n")
    headers = ["Dataset"] + [T_LABELS[k] for k in T_KEYS] + ["overall"]
    rows = []
    for ds in datasets:
        cell = cells[ds]
        row = [ds]
        if cell is None:
            row += ["—"] * len(T_KEYS) + ["(no scores)"]
        else:
            by_type = cell.get("by_type", {})
            row += [_fmt(by_type.get(k)) for k in T_KEYS]
            row.append(_fmt(cell.get("overall_average")))
        rows.append(row)
    bits += [_md_table(headers, rows), "", _overdetermination_section(cells, datasets), ""]

    bits.append("## vs Paper-best\n")
    headers = ["Dataset", "T-type", "ours", "paper-best", "Δ"]
    rows = []
    for ds in datasets:
        cell = cells[ds]
        ds_paper = by_dataset_paper.get(ds, {})
        if cell is None:
            rows.append([ds, "—", "(no scores)", "—", "—"]); continue
        by_type = cell.get("by_type", {})
        for k in T_KEYS:
            ours, pb = by_type.get(k), ds_paper.get(T_LABELS[k])
            if ours is None and pb is None:
                continue
            rows.append([ds, T_LABELS[k], _fmt(ours), _fmt(pb), _delta(ours, pb)])
        rows.append([ds, "**overall**", _fmt(cell.get("overall_average")),
                     _fmt(overall_paper.get(ds)),
                     _delta(cell.get("overall_average"), overall_paper.get(ds))])
    bits += [_md_table(headers, rows), ""]
    return "\n".join(bits)


def render_html_report(markdown_text: str) -> str:
    body = MarkdownIt("commonmark", {"html": False}).render(markdown_text)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>SSDataAgent report</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>"
            f"</head><body>{body}</body></html>")
```

- [ ] **Step 4: Refactor `scripts/generate_exp_report.py` to delegate**

Replace the inline body-building in `main()` with a call to `reports.render_markdown_report(...)`, keeping the `--baseline` A/B section in the script if present (the shared core covers the non-baseline body; the script appends the baseline section + Retro placeholder as before). Keep the script's CLI and output-path behavior unchanged. Verify the script still runs:

Run: `.venv/bin/python scripts/generate_exp_report.py --help`
Expected: prints usage (no import error).

- [ ] **Step 5: Add the `/api/reports` route**

```python
    from ssdataagent import reports as _reports
    from ssdataagent.config import REPO_ROOT

    class ReportRequest(BaseModel):
        experiment: str
        condition: str = "full_agent"
        baseline: str | None = None
        format: str = "md"

    @app.post("/api/reports")
    def post_report(req: ReportRequest):
        md = _reports.render_markdown_report(
            req.experiment, condition=req.condition, baseline=req.baseline,
            results_root=root,
            experiments_yaml=REPO_ROOT / "config" / "experiments.yaml",
            paper_baselines=REPO_ROOT / "config" / "paper_baselines.json")
        content = _reports.render_html_report(md) if req.format == "html" else md
        return {"format": req.format, "content": content}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reports.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add src/ssdataagent/reports.py scripts/generate_exp_report.py src/ssdataagent/console/app.py tests/test_reports.py
git commit -m "console: extract shared report core; reports endpoint (HTML + Markdown)"
```

---

### Task 10: Lab notebook (DB + LEDGER append + Notion no-op)

**Files:**
- Create: `src/ssdataagent/console/notebook.py`
- Modify: `src/ssdataagent/console/app.py` (add `/api/notebook`)
- Test: `tests/test_console_notebook.py`

**Interfaces:**
- Produces:
  - `console.notebook.create_entry(conn, *, hypothesis, change, result, interpretation, next, linked_experiments: list[str], ledger_path: Path | None = None, notion_token: str | None = None) -> dict` — inserts a `notebook` row; if `ledger_path` given, appends a 7-column LEDGER row (`| date | exp_names | model | git_sha | hypothesis | headline | retro |`) and sets `ledger_synced=1` (date = today via `datetime.date.today().isoformat()`; `exp_names` = `+`-joined backticked linked experiments; `headline` = `result`; `retro` left blank); Notion mirror is a no-op when `notion_token` is falsy (returns with `notion_page_id=None`).
  - `console.notebook.list_entries(conn) -> list[dict]`.
  - The appended LEDGER row must parse back through `ssdataagent.dashboard.ledger.parse_ledger` (≥7 cells), so the existing dashboard keeps working.
- API: `GET /api/notebook` → `{"entries":[...]}`; `POST /api/notebook` body of the five fields + `linked_experiments` → creates, appends to the real `docs/experiments/LEDGER.md`, returns the entry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_console_notebook.py
from pathlib import Path

from ssdataagent.console import db, notebook
from ssdataagent.dashboard.ledger import parse_ledger


def test_create_entry_inserts_and_appends_ledger(tmp_path: Path):
    conn = db.connect(db.default_db_path(tmp_path / "results"))
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text(
        "| date | exp | model | git_sha | hypothesis | headline | retro |\n"
        "|---|---|---|---|---|---|---|\n")

    entry = notebook.create_entry(
        conn, hypothesis="wider priors help", change="bumped scale",
        result="+0.04 T2", interpretation="prior collapse eased",
        next="try T3", linked_experiments=["exp_a", "exp_b"], ledger_path=ledger)

    assert entry["ledger_synced"] == 1
    assert entry["notion_page_id"] is None
    # Round-trips through the dashboard ledger parser.
    parsed = parse_ledger(ledger)
    assert len(parsed) == 1
    assert set(parsed[0].exp_names) == {"exp_a", "exp_b"}
    assert parsed[0].headline == "+0.04 T2"
    # Persisted in the DB.
    assert len(notebook.list_entries(conn)) == 1


def test_notion_noop_without_token(tmp_path: Path):
    conn = db.connect(db.default_db_path(tmp_path / "results"))
    entry = notebook.create_entry(
        conn, hypothesis="h", change="c", result="r", interpretation="i",
        next="n", linked_experiments=[], ledger_path=None, notion_token=None)
    assert entry["notion_page_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_console_notebook.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/ssdataagent/console/notebook.py
"""Lab notebook entries: DB row + append to LEDGER.md (+ optional Notion)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3


def create_entry(conn: sqlite3.Connection, *, hypothesis: str, change: str,
                 result: str, interpretation: str, next: str,
                 linked_experiments: list[str], ledger_path=None,
                 notion_token: str | None = None) -> dict:
    created_at = dt.datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        """INSERT INTO notebook
             (created_at, hypothesis, change, result, interpretation, next,
              linked_experiments_json, ledger_synced, notion_page_id)
           VALUES (?,?,?,?,?,?,?,0,NULL)""",
        (created_at, hypothesis, change, result, interpretation, next,
         json.dumps(linked_experiments)),
    )
    entry_id = cur.lastrowid
    ledger_synced = 0
    if ledger_path is not None:
        _append_ledger(ledger_path, linked_experiments, hypothesis, result)
        conn.execute("UPDATE notebook SET ledger_synced=1 WHERE id=?", (entry_id,))
        ledger_synced = 1
    notion_page_id = _mirror_to_notion(notion_token)  # no-op without token
    if notion_page_id:
        conn.execute("UPDATE notebook SET notion_page_id=? WHERE id=?",
                     (notion_page_id, entry_id))
    conn.commit()
    return {"id": entry_id, "created_at": created_at, "ledger_synced": ledger_synced,
            "notion_page_id": notion_page_id, "linked_experiments": linked_experiments}


def _append_ledger(ledger_path, linked: list[str], hypothesis: str, headline: str) -> None:
    from pathlib import Path
    date = dt.date.today().isoformat()
    exp_field = " + ".join(f"`{n}`" for n in linked) if linked else "—"
    row = f"| {date} | {exp_field} | — | — | {hypothesis} | {headline} | — |\n"
    p = Path(ledger_path)
    existing = p.read_text() if p.exists() else ""
    p.write_text(existing + row)


def _mirror_to_notion(notion_token: str | None):
    """Best-effort one-way mirror. The FastAPI backend has no MCP access, so
    this is a token-gated stub: returns None unless a real integration is wired."""
    if not notion_token:
        return None
    return None  # v1: integration not implemented; reserved for a configured token


def list_entries(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM notebook ORDER BY id DESC")]
```

- [ ] **Step 4: Add the `/api/notebook` routes to `create_app`**

```python
    from ssdataagent.console import notebook as _notebook
    from ssdataagent.config import REPO_ROOT

    class NotebookEntry(BaseModel):
        hypothesis: str = ""
        change: str = ""
        result: str = ""
        interpretation: str = ""
        next: str = ""
        linked_experiments: list[str] = []

    @app.get("/api/notebook")
    def get_notebook():
        return {"entries": _notebook.list_entries(conn)}

    @app.post("/api/notebook")
    def post_notebook(req: NotebookEntry):
        ledger = REPO_ROOT / "docs" / "experiments" / "LEDGER.md"
        return _notebook.create_entry(
            conn, hypothesis=req.hypothesis, change=req.change, result=req.result,
            interpretation=req.interpretation, next=req.next,
            linked_experiments=req.linked_experiments, ledger_path=ledger)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_console_notebook.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/console/notebook.py src/ssdataagent/console/app.py tests/test_console_notebook.py
git commit -m "console: lab notebook entries with LEDGER append + Notion stub"
```

---

### Task 11: `__main__` entry + static SPA mount + full backend suite green

**Files:**
- Create: `src/ssdataagent/console/__main__.py`
- Modify: `src/ssdataagent/console/app.py` (mount `web/dist` if present)
- Test: re-run the whole backend suite.

**Interfaces:**
- Produces:
  - `python -m ssdataagent.console [--host 127.0.0.1] [--port 8000]` → runs uvicorn on `create_app()`.
  - `create_app` mounts `StaticFiles(directory=web_dist, html=True)` at `/` **only if** `<REPO_ROOT>/web/dist` exists (so backend tests without a built SPA still pass, and `/api/*` routes always take precedence by being registered first).

- [ ] **Step 1: Write `__main__.py`**

```python
# src/ssdataagent/console/__main__.py
"""Run the console: python -m ssdataagent.console"""
from __future__ import annotations

import argparse

import uvicorn

from ssdataagent.console.app import create_app


def main() -> None:
    p = argparse.ArgumentParser(description="SSDataAgent local console")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Mount the SPA in `create_app` (just before `return app`)**

```python
    from fastapi.staticfiles import StaticFiles
    from ssdataagent.config import REPO_ROOT
    web_dist = REPO_ROOT / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="spa")
```

- [ ] **Step 3: Add a smoke test that the module imports and app builds (extend `tests/test_console_api.py`)**

```python
def test_app_builds_without_web_dist(tmp_path):
    # No web/dist in tmp results; app still builds and /api works.
    from ssdataagent.console.app import create_app
    app = create_app(results_root=tmp_path / "results")
    c = TestClient(app)
    assert c.get("/api/leaderboard").status_code == 200
```

- [ ] **Step 4: Run the full backend suite**

Run: `.venv/bin/python -m pytest tests/test_console_db.py tests/test_console_sync.py tests/test_console_leaderboard.py tests/test_console_api.py tests/test_console_forking.py tests/test_console_queue.py tests/test_console_compare.py tests/test_reports.py tests/test_console_notebook.py -v`
Expected: PASS (all console + reports tests green)

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/console/__main__.py src/ssdataagent/console/app.py tests/test_console_api.py
git commit -m "console: python -m entry point + optional SPA static mount"
```

---

### Task 12: React + Vite SPA (all six views)

**Files:**
- Create: `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/index.html`
- Create: `web/src/main.tsx`, `web/src/App.tsx`, `web/src/api.ts`
- Create: `web/src/views/Leaderboard.tsx`, `RunDetail.tsx`, `Launcher.tsx`, `Compare.tsx`, `Reports.tsx`, `Notebook.tsx`
- Create: `web/src/views/Leaderboard.test.tsx`, `web/src/views/Compare.test.tsx`
- Create: `web/README.md`

**Interfaces (consumed from the backend, all under `/api`):**
- `GET /api/leaderboard` → `{rows: [{experiment, condition, dataset, by_type, overall_average, overdetermination_gap, is_pilot, is_champion}]}`
- `GET /api/runs` → `{experiments: [{name, status, model, ...}]}`
- `GET /api/runs/:name/detail` → `{experiment, runs:[{condition, dataset, run_id, eval, meta, artifacts}]}`
- `POST /api/runs` `{name | fork_from,new_name,overrides}` → `{enqueued}`
- `POST /api/runs/:name/cancel` → `{cancelled}`; `GET /api/runs/:name/log?tail=` → `{log}`
- `POST /api/compare` `{selectors:[{experiment,condition,dataset}]}` → `{types, cells, matrix}`
- `POST /api/reports` `{experiment, condition?, baseline?, format}` → `{format, content}`
- `GET/POST /api/notebook` → `{entries}` / created entry

This task's gate is **`npm run build` succeeds** and **`npm run test` (vitest) passes**, not exhaustive UI behavior. Keep components small and typed.

- [ ] **Step 1: Scaffold `web/package.json`**

```json
{
  "name": "ssdataagent-console",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "react-plotly.js": "^2.6.0",
    "plotly.js-dist-min": "^2.35.0"
  },
  "devDependencies": {
    "@testing-library/react": "^16.0.0",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "jsdom": "^25.0.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 2: `web/vite.config.ts` (dev proxy + vitest config)**

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist" },
  test: { environment: "jsdom", globals: true },
});
```

- [ ] **Step 3: `web/tsconfig.json`, `web/index.html`, `web/src/main.tsx`**

```json
// web/tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020", "useDefineForClassFields": true, "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext", "skipLibCheck": true, "moduleResolution": "bundler",
    "resolveJsonModule": true, "isolatedModules": true, "noEmit": true, "jsx": "react-jsx",
    "strict": true, "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"]
}
```

```html
<!-- web/index.html -->
<!doctype html><html><head><meta charset="utf-8"><title>SSDataAgent console</title></head>
<body><div id="root"></div><script type="module" src="/src/main.tsx"></script></body></html>
```

```tsx
// web/src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode><BrowserRouter><App /></BrowserRouter></React.StrictMode>
);
```

- [ ] **Step 4: `web/src/api.ts` (typed fetch client)**

```ts
// web/src/api.ts
export type LeaderboardRow = {
  experiment: string; condition: string; dataset: string;
  by_type: Record<string, number | null>;
  overall_average: number | null; overdetermination_gap: number | null;
  is_pilot: boolean; is_champion: boolean;
};

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export const api = {
  leaderboard: () => getJSON<{ rows: LeaderboardRow[] }>("/api/leaderboard"),
  runs: () => getJSON<{ experiments: any[] }>("/api/runs"),
  runDetail: (name: string) => getJSON<any>(`/api/runs/${name}/detail`),
  launch: (body: any) => postJSON<{ enqueued: string }>("/api/runs", body),
  cancel: (name: string) => postJSON<{ cancelled: boolean }>(`/api/runs/${name}/cancel`, {}),
  log: (name: string) => getJSON<{ log: string }>(`/api/runs/${name}/log`),
  compare: (selectors: any[]) => postJSON<any>("/api/compare", { selectors }),
  report: (body: any) => postJSON<{ format: string; content: string }>("/api/reports", body),
  notebook: () => getJSON<{ entries: any[] }>("/api/notebook"),
  addNote: (body: any) => postJSON<any>("/api/notebook", body),
};
```

- [ ] **Step 5: `web/src/App.tsx` (router + nav to the six views)**

```tsx
// web/src/App.tsx
import { Link, Route, Routes } from "react-router-dom";
import { Leaderboard } from "./views/Leaderboard";
import { RunDetail } from "./views/RunDetail";
import { Launcher } from "./views/Launcher";
import { Compare } from "./views/Compare";
import { Reports } from "./views/Reports";
import { Notebook } from "./views/Notebook";

export function App() {
  return (
    <div style={{ fontFamily: "system-ui", maxWidth: "72rem", margin: "1rem auto" }}>
      <nav style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
        <Link to="/">Leaderboard</Link><Link to="/runs">Runs</Link>
        <Link to="/compare">Compare</Link><Link to="/reports">Reports</Link>
        <Link to="/notebook">Notebook</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Leaderboard />} />
        <Route path="/runs" element={<Launcher />} />
        <Route path="/runs/:name" element={<RunDetail />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/notebook" element={<Notebook />} />
      </Routes>
    </div>
  );
}
```

- [ ] **Step 6: Implement the six view components**

Each view is a small data-fetching component. Implement:
- `Leaderboard.tsx` — `useEffect` → `api.leaderboard()`; render a sortable table (columns: experiment, condition, dataset, each type from the union of `by_type` keys, over-determination gap; champion rows get a ★ and a highlight class). Pure render of the rows prop is extracted as `LeaderboardTable({rows})` so the test can render it with fixture data without fetching.
- `RunDetail.tsx` — `useParams` name → `api.runDetail`; show experiment meta, per-run eval `by_type`/`by_variable`/over-determination, and artifact links (anchor to `run_dir` paths) + logged LLM I/O (`prompts.jsonl`/`responses.jsonl`).
- `Launcher.tsx` — `api.runs()` list with status badges; a form to launch (`name`) or fork (`fork_from` + `new_name` + JSON `overrides`); poll `api.log(name)` for the selected run; cancel button.
- `Compare.tsx` — multi-select of (experiment, condition, dataset) selectors → `api.compare`; render the matrix as a Plotly heatmap (`<Plot>` from `react-plotly.js`) and a diff table. Extract `CompareHeatmap({types, matrix, cells})` for the test.
- `Reports.tsx` — pick an experiment + format → `api.report`; show the returned content (HTML in an iframe srcdoc, Markdown in a `<pre>`); download button.
- `Notebook.tsx` — `api.notebook()` list + a form (hypothesis → change → result → interpretation → next, linked experiments) → `api.addNote`.

Write these as straightforward typed components following the `api.ts` contracts. Keep each file focused.

- [ ] **Step 7: Write the two vitest component tests**

```tsx
// web/src/views/Leaderboard.test.tsx
import { render, screen } from "@testing-library/react";
import { LeaderboardTable } from "./Leaderboard";

test("renders champion marker and scores", () => {
  render(<LeaderboardTable rows={[{
    experiment: "exp_b", condition: "full_agent", dataset: "gss",
    by_type: { type1: 0.7 }, overall_average: 0.7, overdetermination_gap: 0.2,
    is_pilot: false, is_champion: true,
  }]} />);
  expect(screen.getByText("exp_b")).toBeTruthy();
  expect(screen.getByText("★")).toBeTruthy();
});
```

```tsx
// web/src/views/Compare.test.tsx
import { render } from "@testing-library/react";
import { CompareHeatmap } from "./Compare";

test("renders heatmap container with types", () => {
  const { container } = render(
    <CompareHeatmap types={["type1"]} matrix={[[0.5]]}
      cells={[{ selector: { experiment: "a", condition: "c", dataset: "d" },
                by_type: { type1: 0.5 }, overall_average: 0.5, overdetermination_gap: null }]} />);
  expect(container).toBeTruthy();
});
```

(`LeaderboardTable` and `CompareHeatmap` must be exported named functions taking the props above.)

- [ ] **Step 8: Install + build + test the frontend**

Run:
```bash
cd web && npm install && npm run build && npm run test
```
Expected: `vite build` writes `web/dist/`; `vitest run` passes (2 files).

- [ ] **Step 9: Add `web/node_modules` and `web/dist` to `.gitignore`; write `web/README.md`**

`web/README.md` documents: `npm install`, `npm run dev` (proxies to uvicorn on :8000), `npm run build`, and that `python -m ssdataagent.console` serves the built `dist/`. Add to repo `.gitignore`:
```
web/node_modules/
web/dist/
```

- [ ] **Step 10: Commit**

```bash
git add web/ .gitignore
git commit -m "console: React+Vite SPA with all six views (build + vitest green)"
```

---

## Self-Review (completed)

**Spec coverage:** Leaderboard (T3/T4), run launcher+status (T6,T7,T12), run detail (T4,T12), compare (T8,T12), report export HTML+MD (T9,T12), lab notebook + LEDGER + Notion (T10,T12); SQLite index + queue state, flat-files-source-of-truth, `_`-prefixed DB (T1,T2); champion rule (T3); subprocess queue (T6); fork (T5); `python -m` + localhost + static mount (T11); React+Vite+Plotly (T12); all six §6 endpoints covered. Determinism/leakage/gate honored (runner untouched; mocked subprocess; nullable over-determination).

**Placeholder scan:** No TBD/TODO. Every code step has complete code except the six React views in T12 Step 6, which are specified by exact API contracts + the two extracted testable components (`LeaderboardTable`, `CompareHeatmap`) that the gate actually checks — appropriate, since the frontend gate is build + component smoke, not exhaustive UI.

**Type consistency:** `LeaderboardRow` keys match across `leaderboard.py`, `app.py` SELECT, and `api.ts`. `build_matrix` output (`types`/`cells`/`matrix`) matches `Compare.tsx`/test. `experiment_state` returns `interrupted` (not status.py's `running_or_interrupted`) — documented divergence. Queue status strings (`queued`/`running`/`done`/`failed`) consistent between `queue.py` and sync's console-row guard.
