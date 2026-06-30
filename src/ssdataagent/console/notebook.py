# src/ssdataagent/console/notebook.py
"""Lab notebook entries: DB row + append to LEDGER.md (+ optional Notion)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path


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
    date = dt.date.today().isoformat()
    exp_field = " + ".join(f"`{n}`" for n in linked) if linked else ""
    row = f"| {date} | {exp_field} | — | — | {hypothesis} | {headline} | — |\n"
    p = Path(ledger_path)
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    p.write_text(existing + row, encoding="utf-8")


def _mirror_to_notion(notion_token: str | None):
    """Best-effort one-way mirror. The FastAPI backend has no MCP access, so
    this is a token-gated stub: returns None unless a real integration is wired."""
    if not notion_token:
        return None
    return None  # v1: integration not implemented; reserved for a configured token


def list_entries(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM notebook ORDER BY id DESC")]
