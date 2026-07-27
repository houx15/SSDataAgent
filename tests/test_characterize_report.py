import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import pandas as pd


def _toy_df():
    rows = []
    for pair, fam in [("cps_1970_1980", "time"), ("gss_2018_race", "group")]:
        ds = "cps" if pair.startswith("cps") else "gss"
        rows += [
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q1",
             "metric": "composition_share", "key": "income", "value": 0.6},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q2",
             "metric": "marginal_distance", "key": "age", "value": 0.3, "kind": "wasserstein"},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q3",
             "metric": "pct_stable", "key": "all_pairs", "value": 0.7},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q3",
             "metric": "pct_shifted", "key": "all_pairs", "value": 0.2},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q3",
             "metric": "pct_undefined", "key": "all_pairs", "value": 0.1},
            {"pair": pair, "family": fam, "dataset": ds, "question": "Q4",
             "metric": "shape_ratio", "key": "income", "value": 0.4, "level": 2.0, "shape": 1.3},
        ]
    return pd.DataFrame(rows)


def test_build_report_html_is_self_contained():
    from characterize_report import build_report_html
    html = build_report_html(_toy_df())
    assert html.lstrip().startswith("<!doctype html>")
    assert "</html>" in html
    # figures embedded, no external asset references
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html
    assert "src=\"http" not in html
    # all four questions titled
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert q in html
