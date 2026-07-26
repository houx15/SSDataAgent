import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_synth_numeric_matches_quantiles():
    from ssdataagent.transfer.blind import _synth_numeric
    col = _synth_numeric([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], L=5000)
    assert len(col) == 5000
    assert abs(np.quantile(col, 0.5) - 5.0) < 0.2
    assert col.min() >= -0.1 and col.max() <= 10.1


def test_synth_categorical_matches_probs_and_length():
    from ssdataagent.transfer.blind import _synth_categorical
    col = _synth_categorical({"a": 0.5, "b": 0.3, "c": 0.2}, L=1000)
    assert len(col) == 1000
    vc = pd.Series(col).value_counts(normalize=True)
    assert abs(vc["a"] - 0.5) < 0.01 and abs(vc["b"] - 0.3) < 0.01


def test_build_marg_frame_uses_elicited_and_carries_A_missingness():
    from ssdataagent.transfer.blind import build_marg_frame
    a = pd.DataFrame({
        "age": [20, 30, 40, 50, np.nan, 60, 70, 80, 25, 35],   # numeric, 10% missing
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],  # categorical, 0% missing
    })
    elicited = {
        "age": {"quantiles": [18, 22, 30, 40, 50, 60, 65, 70, 75, 80, 90]},
        "sex": {"probs": {"M": 0.7, "F": 0.3}},
    }
    frame = build_marg_frame(elicited, a, ["age", "sex"], L=2000, seed=0)
    # elicited proportions win for sex
    vc = frame["sex"].dropna().astype(str).value_counts(normalize=True)
    assert abs(vc["M"] - 0.7) < 0.02
    # A's missingness RATE is carried (age ~10%, sex ~0%)
    assert abs(frame["age"].isna().mean() - 0.1) < 0.02
    assert frame["sex"].isna().mean() < 0.001
    # elicited numeric level wins (median ~60 from the quantiles, not A's ~40)
    assert abs(np.nanmedian(pd.to_numeric(frame["age"])) - 60) < 5


def test_build_marg_frame_falls_back_to_A_when_missing():
    from ssdataagent.transfer.blind import build_marg_frame
    a = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    frame = build_marg_frame({}, a, ["x"], L=1000, seed=0)   # nothing elicited -> carry A
    assert len(frame) == 1000
    assert 1 <= np.nanmedian(pd.to_numeric(frame["x"])) <= 10
