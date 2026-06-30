"""Build the compare payload (metric diff + strategy x type matrix)."""
from __future__ import annotations

from typing import Callable


def _gap(ev: dict | None) -> float | None:
    if not ev:
        return None
    try:
        return ev["overdetermination"]["cell_based"]["headline_gap"]
    except (KeyError, TypeError):
        return None


def build_matrix(selectors: list[dict],
                 eval_loader: Callable[[dict], dict | None]) -> dict:
    loaded = [(s, eval_loader(s)) for s in selectors]
    types: list[str] = sorted(
        {t for _, ev in loaded if ev for t in (ev.get("by_type") or {})}
    )
    cells = []
    matrix = []
    for s, ev in loaded:
        by_type = (ev or {}).get("by_type") or {}
        cells.append({
            "selector": s,
            "by_type": by_type,
            "overall_average": (ev or {}).get("overall_average"),
            "overdetermination_gap": _gap(ev),
        })
        matrix.append([by_type.get(t) for t in types])
    return {"types": types, "cells": cells, "matrix": matrix}
