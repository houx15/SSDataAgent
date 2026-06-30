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
