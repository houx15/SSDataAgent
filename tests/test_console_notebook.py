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


def test_empty_linked_experiments_roundtrips_to_empty(tmp_path: Path):
    conn = db.connect(db.default_db_path(tmp_path / "results"))
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text(
        "| date | exp | model | git_sha | hypothesis | headline | retro |\n"
        "|---|---|---|---|---|---|---|\n")
    notebook.create_entry(
        conn, hypothesis="h", change="c", result="r", interpretation="i",
        next="n", linked_experiments=[], ledger_path=ledger)
    parsed = parse_ledger(ledger)
    assert len(parsed) == 1
    assert parsed[0].exp_names == []
