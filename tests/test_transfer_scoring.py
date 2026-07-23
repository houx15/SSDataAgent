import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_mean_scores_selects_only_numeric_type_and_overall():
    from ssdataagent.transfer.scoring import mean_scores
    df = pd.DataFrame([
        {"T1": 0.8, "T2": 0.5, "T3": 0.6, "overall": 0.63, "T3_error": "boom"},
        {"T1": 0.7, "T2": 0.6, "T3": 0.4, "overall": 0.57, "T3_error": "boom"},
    ])
    out = mean_scores(df)
    assert out == {"T1": 0.75, "T2": 0.55, "T3": 0.5, "overall": 0.6}
    assert "T3_error" not in out  # string column starting with 'T' must be excluded


def test_transfer_map_reexports_helpers_from_scoring():
    # transfer_map must import the helpers, not redefine them (single source of truth).
    import transfer_map
    from ssdataagent.transfer import scoring
    assert transfer_map.mean_scores is scoring.mean_scores
    assert transfer_map.restrict_config_dir is scoring.restrict_config_dir
