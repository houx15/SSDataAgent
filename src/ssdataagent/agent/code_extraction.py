from __future__ import annotations

import re

_PATTERN = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def extract_python_block(text: str) -> str | None:
    """Return the first fenced Python code block in *text*, or None."""
    m = _PATTERN.search(text)
    if not m:
        return None
    return m.group(1).rstrip("\n")
