from pathlib import Path

import pytest

from ssdataagent.dashboard.ledger import LedgerEntry, parse_ledger

FIXTURES = Path(__file__).parent / "fixtures" / "dashboard"


def test_parse_ledger_returns_one_entry_per_row():
    entries = parse_ledger(FIXTURES / "LEDGER.md")
    assert len(entries) == 3


def test_parse_ledger_extracts_basic_fields():
    entries = parse_ledger(FIXTURES / "LEDGER.md")
    a = next(e for e in entries if "exp_demo_a" in e.exp_names)
    assert a.date == "2026-05-10"
    assert a.exp_names == ["exp_demo_a"]
    assert a.model == "gpt-5.4-2026-03-05"
    assert a.git_sha == "def5678"
    assert "demo experiment" in a.hypothesis
    assert "0.42" in a.headline
    assert a.retro_link.endswith("2026-05-10-exp_demo_a-report.md")


def test_parse_ledger_splits_multi_exp_rows():
    entries = parse_ledger(FIXTURES / "LEDGER.md")
    b = next(e for e in entries if "exp_demo_b_cross" in e.exp_names)
    assert b.exp_names == ["exp_demo_b_cross", "exp_demo_b_long"]


def test_parse_ledger_flags_pilots():
    entries = parse_ledger(FIXTURES / "LEDGER.md")
    pilot = next(e for e in entries if "pilot_demo" in e.exp_names)
    assert pilot.is_pilot is True
    non_pilot = next(e for e in entries if "exp_demo_a" in e.exp_names)
    assert non_pilot.is_pilot is False


def test_parse_ledger_strips_backticks_from_exp_names():
    entries = parse_ledger(FIXTURES / "LEDGER.md")
    for e in entries:
        for name in e.exp_names:
            assert "`" not in name


def test_parse_ledger_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_ledger(FIXTURES / "DOES_NOT_EXIST.md")


def test_parse_ledger_splits_slash_separated_exp_names(tmp_path):
    p = tmp_path / "LEDGER.md"
    p.write_text(
        "| date | exp_name | model | git_sha | hypothesis | headline | retro |\n"
        "|------|----------|-------|---------|------------|----------|-------|\n"
        "| 2026-04-30 | `pilot_a` / `pilot_b` / `pilot_c` | m | sha | h | hl | [r](x.md) |\n",
        encoding="utf-8",
    )
    entries = parse_ledger(p)
    assert len(entries) == 1
    assert entries[0].exp_names == ["pilot_a", "pilot_b", "pilot_c"]
    for name in entries[0].exp_names:
        assert "`" not in name


def test_parse_ledger_strips_embedded_backticks_after_split(tmp_path):
    """Real LEDGER has `name1` + `name2` (suffix) — both names must be clean."""
    p = tmp_path / "LEDGER.md"
    p.write_text(
        "| date | exp_name | model | git_sha | hypothesis | headline | retro |\n"
        "|------|----------|-------|---------|------------|----------|-------|\n"
        "| 2026-05-07 | `exp_a` + `exp_b` (Stage B) | m | sha | h | hl | [r](x.md) |\n",
        encoding="utf-8",
    )
    entries = parse_ledger(p)
    assert entries[0].exp_names == ["exp_a", "exp_b"]
    for name in entries[0].exp_names:
        assert "`" not in name
        assert "(" not in name


def test_parse_ledger_strips_parenthetical_suffix_from_single_name(tmp_path):
    """Some pilot rows have a single name with a parenthetical: `pilot_x` (full paper compare)."""
    p = tmp_path / "LEDGER.md"
    p.write_text(
        "| date | exp_name | model | git_sha | hypothesis | headline | retro |\n"
        "|------|----------|-------|---------|------------|----------|-------|\n"
        "| 2026-05-03 | `pilot_paper_agents` (full paper compare) | m | sha | h | hl | [r](x.md) |\n",
        encoding="utf-8",
    )
    entries = parse_ledger(p)
    assert entries[0].exp_names == ["pilot_paper_agents"]
    assert entries[0].is_pilot is True
