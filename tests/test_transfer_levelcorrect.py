import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_numeric_outcomes_picks_numeric_in_both():
    from ssdataagent.transfer.levelcorrect import numeric_outcomes
    a = pd.DataFrame({"inc": [1, 2, 3], "occ": ["x", "y", "z"], "kid": [0, 1, 2]})
    b = pd.DataFrame({"inc": [4, 5, 6], "occ": ["p", "q", "r"], "kid": [1, 2, 3]})
    assert numeric_outcomes(a, b, ["inc", "occ", "kid"]) == ["inc", "kid"]


def test_oracle_and_pooled_shifts_are_mean_diffs():
    from ssdataagent.transfer.levelcorrect import oracle_shifts, pooled_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0, 0.0, 0.0]})
    b = pd.DataFrame({"inc": [5.0, 5.0, 5.0, 5.0]})
    assert abs(oracle_shifts(a, b, ["inc"])["inc"] - 5.0) < 1e-9
    assert abs(pooled_shifts(a, b, ["inc"])["inc"] - 5.0) < 1e-9


def test_apply_level_shift_moves_mean_preserves_shape_and_others():
    from ssdataagent.transfer.levelcorrect import apply_level_shift, outcome_mean
    a = pd.DataFrame({"inc": [1.0, 2.0, 3.0, np.nan], "occ": ["x", "y", "z", "w"]})
    out = apply_level_shift(a, {"inc": 10.0})
    assert abs(outcome_mean(out, "inc") - (outcome_mean(a, "inc") + 10.0)) < 1e-9
    assert abs(pd.to_numeric(out["inc"]).var() - pd.to_numeric(a["inc"]).var()) < 1e-9
    assert out["inc"].isna().sum() == 1            # NaN preserved
    assert list(out["occ"]) == list(a["occ"])      # other columns untouched


def test_apply_level_shift_zero_nonfinite_and_missing_are_noops():
    from ssdataagent.transfer.levelcorrect import apply_level_shift
    a = pd.DataFrame({"inc": [1.0, 2.0]})
    assert list(apply_level_shift(a, {"inc": 0.0})["inc"]) == [1.0, 2.0]
    assert list(apply_level_shift(a, {"inc": float("nan")})["inc"]) == [1.0, 2.0]
    assert list(apply_level_shift(a, {"missing": 5.0}).columns) == ["inc"]


def test_hybrid_shifts_gate_routes_pooled_vs_llm():
    from ssdataagent.transfer.levelcorrect import hybrid_shifts
    pooled, llm = {"inc": 3.0}, {"inc": 7.0}
    assert hybrid_shifts(pooled, llm, n_siblings=3, ess=0.6)["inc"] == 3.0   # plural+effective
    assert hybrid_shifts(pooled, llm, n_siblings=1, ess=0.6)["inc"] == 7.0   # thin -> llm
    # an outcome absent from the chosen arm falls back to the other
    assert hybrid_shifts({}, {"inc": 7.0}, n_siblings=3, ess=0.6)["inc"] == 7.0
