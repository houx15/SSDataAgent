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
