"""Parse retro markdown reports into structured sections + bullets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RetroSections:
    path: Path
    frontmatter: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    bullets: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    raw_text: str = ""


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_BULLET_KV_RE = re.compile(
    r"^\s*-\s+\*\*(?P<label>[^*]+?):\*\*\s*(?P<value>.*?)\s*$",
    re.MULTILINE,
)


def parse_retro(path: Path) -> RetroSections | None:
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    sections = _split_sections(body)
    bullets = {h: _extract_bullet_kv(b) for h, b in sections.items()}
    return RetroSections(
        path=path,
        frontmatter=frontmatter,
        sections=sections,
        bullets=bullets,
        raw_text=text,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    fm = {str(k): "" if v is None else str(v) for k, v in data.items()}
    return fm, text[m.end():]


def _split_sections(body: str) -> dict[str, str]:
    """Split body on `## ` headings; return `{heading: body_text}`."""
    headings = list(_H2_RE.finditer(body))
    out: dict[str, str] = {}
    for i, match in enumerate(headings):
        name = _normalize_heading(match.group(1))
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        out[name] = body[start:end].strip()
    return out


def _normalize_heading(text: str) -> str:
    """Strip trailing modifiers like `Results — full_agent` → `Results`."""
    for sep in (" — ", " - ", " – "):
        if sep in text:
            return text.split(sep, 1)[0].strip()
    return text.strip()


def _extract_bullet_kv(section_body: str) -> list[tuple[str, str]]:
    return [
        (m.group("label").strip(), m.group("value").strip())
        for m in _BULLET_KV_RE.finditer(section_body)
    ]
