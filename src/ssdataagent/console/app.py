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
                "artifacts": _artifacts(run_dir, root),
            })
        return {"experiment": dict(erow), "runs": runs}

    return app


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _artifacts(run_dir: Path, root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, rel in [("generated_csv", "generated.csv"),
                     ("prompts_jsonl", "prompts.jsonl"),
                     ("responses_jsonl", "responses.jsonl"),
                     ("workspace_dir", "workspace")]:
        if (run_dir / rel).exists():
            out[key] = (run_dir / rel).relative_to(root).as_posix()
    return out
