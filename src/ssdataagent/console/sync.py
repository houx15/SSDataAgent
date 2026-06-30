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
