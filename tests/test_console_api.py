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
    assert run["artifacts"]["generated_csv"] == "exp_a/full_agent/gss/20260630-100000/generated.csv"
    assert not run["artifacts"]["generated_csv"].startswith("/")


def test_run_detail_unknown_is_404(client: TestClient):
    assert client.get("/api/runs/nope/detail").status_code == 404


def test_post_runs_enqueues_named_experiment(tmp_path, results):
    from ssdataagent.console import db, queue
    from ssdataagent.console.app import create_app

    def fake_runner(name, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("hello from run\n")
        return 0

    conn = db.connect(db.default_db_path(results))
    jq = queue.JobQueue(conn, results, concurrency=1, runner=fake_runner)
    app = create_app(results_root=results, job_queue=jq)
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


def test_compare_endpoint_returns_matrix(client: TestClient):
    r = client.post("/api/compare", json={"selectors": [
        {"experiment": "exp_a", "condition": "full_agent", "dataset": "gss"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["types"] == ["type1"]
    assert body["matrix"][0] == [0.6]
