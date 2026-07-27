import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_marginal_distance_identical_is_zero():
    from ssdataagent.transfer.characterize import marginal_distance
    num = pd.Series([1.0, 2.0, 3.0, 4.0])
    d, kind = marginal_distance(num, num.copy())
    assert kind == "wasserstein" and abs(d) < 1e-9
    cat = pd.Series(["a", "b", "a", "c"])
    d, kind = marginal_distance(cat, cat.copy())
    assert kind == "tv" and abs(d) < 1e-9


def test_marginal_distance_disjoint_categoricals_is_one():
    from ssdataagent.transfer.characterize import marginal_distance
    a = pd.Series(["x", "x", "x"])
    b = pd.Series(["y", "y", "y"])
    d, kind = marginal_distance(a, b)
    assert kind == "tv" and abs(d - 1.0) < 1e-9


def test_marginal_distance_known_numeric_shift():
    from ssdataagent.transfer.characterize import marginal_distance
    # a all 0, b all 1 -> pooled SD 0.5 -> standardized values 0 vs 2 -> Wasserstein 2.0
    a = pd.Series([0.0] * 100)
    b = pd.Series([1.0] * 100)
    d, kind = marginal_distance(a, b)
    assert kind == "wasserstein" and abs(d - 2.0) < 1e-6


def test_shape_level_split_pure_level_shift():
    from ssdataagent.transfer.characterize import shape_level_split
    focal = np.repeat(np.arange(10), 20).astype(float)
    a = pd.DataFrame({"f": focal, "y": focal.copy()})
    b = pd.DataFrame({"f": focal, "y": focal + 5.0})
    r = shape_level_split(a, b, "y", "f", bins=10)
    assert abs(r["level"] - 5.0) < 1e-6
    assert r["shape"] < 1e-6
    assert r["shape_ratio"] < 1e-3


def test_shape_level_split_pure_shape_change():
    from ssdataagent.transfer.characterize import shape_level_split
    # 10 distinct values symmetric about 0 (skip 0) -> one per bin, so `level` is exactly 0
    # and the whole gap is shape. (arange(-5,6) has 11 values into 10 bins, which merges the
    # top bin and biases `level` to -0.45; skipping 0 keeps the binning symmetric.)
    focal = np.repeat(np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]), 20).astype(float)
    a = pd.DataFrame({"f": focal, "y": np.zeros(len(focal))})
    b = pd.DataFrame({"f": focal, "y": focal.copy()})        # gradient changes, mean gap = 0
    r = shape_level_split(a, b, "y", "f", bins=10)
    assert abs(r["level"]) < 1e-6
    assert r["shape"] > 3.0
    assert r["shape_ratio"] > 0.99


def test_load_context_group_filter_and_negate(tmp_path):
    from ssdataagent.transfer.characterize import Context, load_context
    csv = tmp_path / "toy.csv"
    pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3],
        "race": ["Black", "White", None, "Black"],
        "x": [1, 2, 3, 4],
    }).to_csv(csv, index=False)
    minority = Context("toy", csv, "black", "race", "Black")
    df = load_context(minority)
    assert list(df.columns) == ["race", "x"]      # Unnamed dropped
    assert len(df) == 2 and set(df["race"]) == {"Black"}
    majority = Context("toy", csv, "rest", "race", "Black", negate=True)
    dfm = load_context(majority)
    assert len(dfm) == 1 and set(dfm["race"]) == {"White"}   # NaN excluded from both


def test_pairs_registry_shape_and_paths():
    from ssdataagent.transfer.characterize import PAIRS, CORE_DEMOGRAPHICS, FOCAL, GROUP_COL
    ids = [p.id for p in PAIRS]
    assert ids == [
        "cps_1970_1980", "cps_1980_1990", "cps_1990_2000", "cps_1970_2000",
        "gss_1994_2018", "cps_1980_race", "gss_2018_race", "cfps_minzu",
    ]
    fams = {p.id: p.family for p in PAIRS}
    assert fams["gss_1994_2018"] == "time" and fams["cfps_minzu"] == "group"
    for p in PAIRS:
        assert p.a.csv.exists() and p.b.csv.exists(), f"missing csv for {p.id}"
    assert CORE_DEMOGRAPHICS["cfps"] == ("gender", "sib_number")
    assert FOCAL["cfps"] == "birth_year" and GROUP_COL["gss"] == "race"


def test_resolve_columns_group_excludes_grouping_var():
    from ssdataagent.transfer.characterize import PAIRS, resolve_columns
    pair = next(p for p in PAIRS if p.id == "gss_2018_race")
    r = resolve_columns(pair)
    assert "race" not in r["core"], "grouping var must be excluded from Q1 core"
    assert "race" not in r["x_sweep"], "grouping var must be excluded from Q2 sweep"
    assert r["core"] == ["age", "gender"]
    assert r["focal"] == "age"


def test_run_characterization_tidy_schema_on_one_pair():
    from ssdataagent.transfer.characterize import PAIRS, run_characterization
    pair = next(p for p in PAIRS if p.id == "gss_2018_race")   # smallest (single gss wave)
    df = run_characterization([pair])
    for col in ("pair", "family", "dataset", "question", "metric", "key", "value"):
        assert col in df.columns
    assert set(df["question"]) >= {"Q1", "Q2", "Q3", "Q4"}
    q1 = df[(df["question"] == "Q1") & (df["metric"] == "composition_share")]
    assert len(q1) > 0 and q1["value"].notna().any()
    assert (df["pair"] == "gss_2018_race").all()


def test_write_outputs_writes_both_copies(tmp_path):
    from ssdataagent.transfer.characterize import write_outputs
    df = pd.DataFrame({"pair": ["p"], "question": ["Q1"], "value": [0.5]})
    results_csv, committed_csv = write_outputs(df, tmp_path)
    assert results_csv.exists() and committed_csv.exists()
    assert results_csv == tmp_path / "results" / "characterization" / "characterization.csv"
    assert committed_csv == tmp_path / "docs" / "report" / "2026-07-27-characterization-data.csv"
    back = pd.read_csv(committed_csv)
    assert list(back["pair"]) == ["p"] and float(back["value"].iloc[0]) == 0.5
