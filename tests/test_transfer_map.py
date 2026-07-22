# tests/test_transfer_map.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from transfer_map import mean_scores, run_layer1


def _frame(n, seed, xmean, beta):
    rng = np.random.default_rng(seed)
    x = rng.normal(xmean, 1, n)
    edu = np.where(x > xmean, "hi", "lo")
    return pd.DataFrame({"age": x, "education": edu, "income": beta * x + rng.normal(0, .3, n)})


def test_run_layer1_returns_map_and_copula():
    a = _frame(1500, 1, xmean=0.0, beta=1.0)
    b = _frame(1500, 2, xmean=3.0, beta=1.0)   # composition shift on outcomes
    covariates, outcomes = ["age", "education"], ["income"]
    kob, cop = run_layer1(a, b, ["age", "education", "income"], covariates, outcomes)
    # KOB has one row per outcome, with a share and a label
    assert set(kob["response"]) == {"income"}
    assert {"composition_share", "label", "gap_raw"}.issubset(kob.columns)
    inc = kob[kob["response"] == "income"].iloc[0]
    assert inc["label"] in {"composition-dominated", "mechanism-shifted", "aligned"}
    # copula table covers all unordered pairs of the 3 columns
    assert len(cop) == 3
    assert {"v1", "v2", "abs_delta", "label"}.issubset(cop.columns)


def test_mean_scores_ignores_error_columns():
    # nb.score emits string T{t}_error columns on a failed type-eval; averaging must
    # not crash on them (regression: startswith('T') swept up 'T1_error').
    df = pd.DataFrame([
        {"T1": 0.8, "T2": 0.6, "T3": None, "T3_error": "KeyError: education", "overall": 0.7},
        {"T1": 0.7, "T2": 0.5, "T3": None, "T3_error": "KeyError: education", "overall": 0.6},
    ])
    out = mean_scores(df)
    assert out["T1"] == 0.75 and out["T2"] == 0.55
    assert "T3" not in out            # all-None -> dropped
    assert "T3_error" not in out      # string error column never averaged
    assert abs(out["overall"] - 0.65) < 1e-9
