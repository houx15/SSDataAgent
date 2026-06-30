# src/ssdataagent/console/app.py
"""FastAPI app factory for the console. Localhost, single-user, no auth."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ssdataagent.config import REPO_ROOT, results_root as default_results_root
from ssdataagent.console import compare as _compare
from ssdataagent.console import db, forking, leaderboard, notebook as _notebook, queue as _q, sync
from ssdataagent import reports as _reports


class RunRequest(BaseModel):
    name: str | None = None
    fork_from: str | None = None
    new_name: str | None = None
    overrides: dict = {}


class CompareRequest(BaseModel):
    selectors: list[dict]


class ReportRequest(BaseModel):
    experiment: str
    condition: str = "full_agent"
    baseline: str | None = None
    format: str = "md"


class NotebookEntry(BaseModel):
    hypothesis: str = ""
    change: str = ""
    result: str = ""
    interpretation: str = ""
    next: str = ""
    linked_experiments: list[str] = []


def create_app(
    results_root: Path | None = None,
    *,
    job_queue=None,
    experiments_yaml: Path | None = None,
) -> FastAPI:
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

    # --- Launcher routes ---
    _experiments_yaml = experiments_yaml if experiments_yaml is not None else REPO_ROOT / "config" / "experiments.yaml"

    if job_queue is not None:
        app.state.job_queue = job_queue
    else:
        app.state.job_queue = _q.JobQueue(conn, root, concurrency=1)
        app.state.job_queue.start()

    @app.post("/api/runs")
    def post_run(req: RunRequest):
        if req.fork_from:
            if not req.new_name:
                raise HTTPException(400, "new_name required when forking")
            try:
                forking.fork_experiment(_experiments_yaml, req.fork_from,
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

    def _latest_eval(sel: dict) -> dict | None:
        cond_dir = root / sel["experiment"] / sel["condition"] / sel["dataset"]
        cands = sorted(cond_dir.glob("*/eval.json"))
        return _read_json(cands[-1]) if cands else None

    @app.post("/api/compare")
    def post_compare(req: CompareRequest):
        return _compare.build_matrix(req.selectors, _latest_eval)

    @app.post("/api/reports")
    def post_report(req: ReportRequest):
        md = _reports.render_markdown_report(
            req.experiment, condition=req.condition, baseline=req.baseline,
            results_root=root,
            experiments_yaml=REPO_ROOT / "config" / "experiments.yaml",
            paper_baselines=REPO_ROOT / "config" / "paper_baselines.json")
        content = _reports.render_html_report(md) if req.format == "html" else md
        return {"format": req.format, "content": content}

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

    web_dist = REPO_ROOT / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="spa")

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
