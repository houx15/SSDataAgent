import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd
from ssdataagent.transfer.rescue import outcome_features


def test_outcome_features_shape_and_ranges():
    pool = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 70, 80, 90],       # numeric predictor
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],  # predictor
        "balanced": ["a", "b", "a", "b", "a", "b", "a", "b"],  # categorical outcome, max entropy
        "constant": ["z"] * 8,                          # zero entropy
        "income": [1.0, 2, 3, 4, 5, 6, 7, 8],           # numeric outcome
    })
    preds = ["age", "sex"]
    f_bal = outcome_features(pool, "balanced", preds, numeric_predictors=frozenset({"age"}))
    assert set(f_bal) == {"entropy", "n_predictors", "is_numeric"}
    assert f_bal["n_predictors"] == 2.0
    assert f_bal["is_numeric"] == 0.0
    assert f_bal["entropy"] == 1.0                      # perfectly balanced binary
    f_const = outcome_features(pool, "constant", preds)
    assert f_const["entropy"] == 0.0                    # single value -> no diversity
    f_inc = outcome_features(pool, "income", preds, numeric_predictors=frozenset({"age"}))
    assert f_inc["is_numeric"] == 1.0
    assert 0.0 <= f_inc["entropy"] <= 1.0
