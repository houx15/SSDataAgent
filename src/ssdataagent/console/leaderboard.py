"""Pure leaderboard assembly: parse rows, mark champion per (cond, dataset)."""
from __future__ import annotations

import json
from typing import Any, TypedDict


class LeaderboardRow(TypedDict):
    experiment: str
    condition: str
    dataset: str
    by_type: dict[str, Any]
    overall_average: float | None
    overdetermination_gap: float | None
    is_pilot: bool
    is_champion: bool


def build_rows(records: list[dict]) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    for rec in records:
        try:
            by_type = json.loads(rec.get("by_type_json") or "{}")
        except (TypeError, ValueError):
            by_type = {}
        rows.append(LeaderboardRow(
            experiment=rec["experiment"],
            condition=rec["condition"],
            dataset=rec["dataset"],
            by_type=by_type,
            overall_average=rec.get("overall_average"),
            overdetermination_gap=rec.get("overdetermination_gap"),
            is_pilot=str(rec["experiment"]).startswith("pilot_"),
            is_champion=False,
        ))

    best: dict[tuple[str, str], LeaderboardRow] = {}
    for r in rows:
        if r["is_pilot"] or r["overall_average"] is None:
            continue
        cell = (r["condition"], r["dataset"])
        cur = best.get(cell)
        if cur is None or (
            r["overall_average"], r["experiment"]
        ) > (cur["overall_average"], cur["experiment"]):
            best[cell] = r
    for r in best.values():
        r["is_champion"] = True
    return rows
