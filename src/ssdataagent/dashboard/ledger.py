"""Parse docs/experiments/LEDGER.md into structured LedgerEntry records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LedgerEntry:
    date: str
    exp_names: list[str]
    model: str
    git_sha: str
    hypothesis: str
    headline: str
    retro_link: str
    is_pilot: bool = field(default=False)


_TABLE_ROW_RE = re.compile(r"^\|(?P<cells>.+)\|\s*$")
_SEPARATOR_RE = re.compile(r"^\|[-:\s|]+\|\s*$")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def parse_ledger(path: Path) -> list[LedgerEntry]:
    text = Path(path).read_text(encoding="utf-8")
    rows = _extract_table_rows(text)
    return [_row_to_entry(r) for r in rows]


def _extract_table_rows(text: str) -> list[list[str]]:
    """Return data rows (header and separator stripped) as list-of-cells."""
    table_lines: list[str] = []
    for line in text.splitlines():
        if _TABLE_ROW_RE.match(line):
            table_lines.append(line)
    if len(table_lines) < 3:
        return []
    # Drop header (row 0) and separator (row 1). Keep data rows.
    rows = []
    for line in table_lines[1:]:
        if _SEPARATOR_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 7:
            rows.append(cells)
    return rows


def _row_to_entry(cells: list[str]) -> LedgerEntry:
    date, exp_field, model, git_sha, hypothesis, headline, retro_field = cells[:7]
    exp_names = _split_exp_names(exp_field)
    retro_link = _extract_first_link(retro_field)
    is_pilot = any(name.startswith("pilot_") for name in exp_names)
    return LedgerEntry(
        date=date,
        exp_names=exp_names,
        model=model,
        git_sha=git_sha,
        hypothesis=hypothesis,
        headline=headline,
        retro_link=retro_link,
        is_pilot=is_pilot,
    )


def _split_exp_names(field: str) -> list[str]:
    parts = re.split(r"\s*[+,]\s*", field)
    return [p.strip().strip("`").strip() for p in parts if p.strip()]


def _extract_first_link(field: str) -> str:
    m = _MD_LINK_RE.search(field)
    return m.group(1) if m else ""
