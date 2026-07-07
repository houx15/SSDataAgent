import json

from ssdataagent.console import strategies


def _rec(exp, cond, ds, overall, by_type):
    return {"experiment": exp, "condition": cond, "dataset": ds,
            "run_id": f"rid-{cond}-{ds}", "model": "gpt-5.4",
            "by_type_json": json.dumps(by_type), "overall_average": overall}


PAPER = {
    "by_dataset_overall": {"gss": 0.39, "cps": 0.40, "acs": 0.40},
    "by_dataset_by_type": {
        "gss": {"T1": 0.13, "T2": 0.71, "T3": 0.55},
        "cps": {"T1": 0.10, "T2": 0.71, "T3": 0.57},
    },
}


def test_classify_maps_conditions_to_family_and_mode():
    assert strategies.classify("design_a_full")[0] == "Design A"
    assert strategies.classify("design_a_aggregate")[1] == "published marginals only"
    assert strategies.classify("s1_raked_transfer")[0] == "S1"
    assert strategies.classify("hotdeck") == ("Hotdeck", "full training microdata")
    # unknown condition degrades to itself
    assert strategies.classify("mystery")[0] == "mystery"


def test_board_picks_best_run_per_family_with_paper_delta():
    recs = [
        _rec("ship_a", "design_a_full", "gss", 0.61, {"type1": 0.40, "type2": 0.77, "type3": 0.67}),
        _rec("ship_a", "design_a_aggregate", "gss", 0.33, {"type1": 0.20, "type2": 0.40, "type3": 0.39}),
        _rec("ship_h", "hotdeck", "acs", 0.74, {"type1": 0.57, "type2": 0.82, "type3": 0.85}),
    ]
    board = strategies.build_board(recs, PAPER)
    fams = {b["family"]: b for b in board}
    # Design A: best run is the full one (0.61), not the aggregate (0.33)
    assert fams["Design A"]["overall_average"] == 0.61
    assert fams["Design A"]["condition"] == "design_a_full"
    assert abs(fams["Design A"]["delta_overall"] - (0.61 - 0.39)) < 1e-9
    # per-type comparison carries the paper's numbers
    t1 = next(t for t in fams["Design A"]["types"] if t["t"] == "T1")
    assert t1["ours"] == 0.40 and t1["paper"] == 0.13
    # board is sorted by overall desc -> hotdeck (0.74) first
    assert board[0]["family"] == "Hotdeck"


def test_board_excludes_pilots_and_null_overall():
    recs = [
        _rec("pilot_x", "design_a_full", "gss", 0.99, {"type1": 0.9}),
        _rec("ship_a", "design_a_full", "gss", None, {"type1": 0.5}),
    ]
    assert strategies.build_board(recs, PAPER) == []


def test_detail_groups_runs_with_blurb_and_modes():
    recs = [
        _rec("ship_a", "design_a_full", "gss", 0.61, {"type1": 0.40}),
        _rec("ship_a", "design_a_transfer", "cps", 0.45, {"type1": 0.20}),
    ]
    d = strategies.build_detail("Design A", recs, PAPER)
    assert d["family"] == "Design A" and d["blurb"]
    assert d["datasets"] == ["cps", "gss"]
    assert set(d["data_modes"]) == {"full training microdata", "earlier survey wave (transfer)"}
    assert len(d["rows"]) == 2
    gss_row = next(r for r in d["rows"] if r["dataset"] == "gss")
    assert abs(gss_row["delta_overall"] - (0.61 - 0.39)) < 1e-9
