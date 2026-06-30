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


def test_sync_protects_console_row_until_flag_appears(tmp_path: Path):
    root = tmp_path / "results"
    conn = db.connect(db.default_db_path(root))

    # Insert a console-owned row directly
    conn.execute(
        "INSERT INTO experiments(name, status, source) VALUES ('exp_live','queued','console')"
    )
    conn.commit()

    # Create experiment dir on disk WITHOUT any flag
    (root / "exp_live").mkdir(parents=True)

    # First sync: row should be PROTECTED
    sync.sync_index(conn, root)
    row = conn.execute("SELECT status, source FROM experiments WHERE name = 'exp_live'").fetchone()
    assert row["status"] == "queued", "Console-owned row status should remain unchanged"
    assert row["source"] == "console", "Console-owned row source should remain unchanged"

    # Now write a terminating flag
    _write(root / "exp_live" / "done.flag", {
        "experiment": "exp_live", "finished_at": "2026-06-30T10:00:00",
        "prompt_variant": "baseline", "llm_model": "gpt-5.4",
        "llm_provider": "openai", "n_dataset_condition_cells": 1,
    })

    # Second sync: row should be RECONCILED
    sync.sync_index(conn, root)
    row = conn.execute("SELECT status, source FROM experiments WHERE name = 'exp_live'").fetchone()
    assert row["status"] == "done", "Console-owned row should reconcile to disk state after flag"
    assert row["source"] == "disk", "Console-owned row source should update to disk after flag"
