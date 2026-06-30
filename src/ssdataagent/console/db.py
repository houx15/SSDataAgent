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
