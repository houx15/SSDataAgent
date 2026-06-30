# tests/test_console_leaderboard.py
from ssdataagent.console import leaderboard


def _rec(exp, cond, ds, overall, gap=None, by_type=None):
    import json
    return {"experiment": exp, "condition": cond, "dataset": ds,
            "by_type_json": json.dumps(by_type or {"type1": overall}),
            "overall_average": overall, "overdetermination_gap": gap}


def test_champion_is_best_per_cell_excluding_pilots():
    recs = [
        _rec("exp_a", "full_agent", "gss", 0.5),
        _rec("exp_b", "full_agent", "gss", 0.7),     # best in this cell
        _rec("pilot_x", "full_agent", "gss", 0.9),   # pilot: ignored
        _rec("exp_c", "design_a_full", "gss", 0.6),  # different cell -> own champ
    ]
    rows = leaderboard.build_rows(recs)
    champs = {(r["condition"], r["dataset"]): r["experiment"]
              for r in rows if r["is_champion"]}
    assert champs[("full_agent", "gss")] == "exp_b"
    assert champs[("design_a_full", "gss")] == "exp_c"
    # pilot flagged, never champion
    pilot = next(r for r in rows if r["experiment"] == "pilot_x")
    assert pilot["is_pilot"] and not pilot["is_champion"]


def test_none_overall_never_champion():
    rows = leaderboard.build_rows([_rec("exp_a", "c", "d", None)])
    assert not any(r["is_champion"] for r in rows)
    assert rows[0]["by_type"] == {"type1": None}


def test_tie_broken_by_experiment_name_desc():
    recs = [_rec("exp_a", "c", "d", 0.5), _rec("exp_z", "c", "d", 0.5)]
    rows = leaderboard.build_rows(recs)
    champ = next(r for r in rows if r["is_champion"])
    assert champ["experiment"] == "exp_z"


def test_malformed_by_type_json_degrades_to_empty_dict():
    rec = {"experiment": "exp_a", "condition": "c", "dataset": "d",
           "by_type_json": "not-json", "overall_average": 0.5,
           "overdetermination_gap": None}
    rows = leaderboard.build_rows([rec])
    assert rows[0]["by_type"] == {}
