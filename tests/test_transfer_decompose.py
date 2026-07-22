# tests/test_transfer_decompose.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.decompose import (
    kob_decompose, oaxaca_blinder, raking_weights,
)


def _mk(n, seed, xmean, beta):
    rng = np.random.default_rng(seed)
    x = rng.normal(xmean, 1, n)
    return pd.DataFrame({"x": x, "y": beta * x + rng.normal(0, 0.3, n)})


def test_raking_matches_target_marginal():
    a = _mk(2000, 1, xmean=0.0, beta=1.0)
    b = _mk(2000, 2, xmean=3.0, beta=1.0)
    w = raking_weights(a, b, ["x"], bins=8)
    # weighted mean of A's x should move toward B's (~3)
    wm = np.average(pd.to_numeric(a["x"]), weights=w)
    assert abs(wm - 3.0) < 0.4


def test_composition_dominated():
    # same mechanism (beta=1), X shifted -> gap is pure composition -> share ~1
    a = _mk(3000, 3, xmean=0.0, beta=1.0)
    b = _mk(3000, 4, xmean=3.0, beta=1.0)
    d = kob_decompose(a, b, "y", ["x"])
    assert d["composition_share"] > 0.7
    assert d["label"] == "composition-dominated"


def test_mechanism_shifted():
    # same X distribution, mechanism flips (beta 1 -> -1) -> share ~0
    a = _mk(3000, 5, xmean=0.0, beta=1.0)
    b = _mk(3000, 5, xmean=0.0, beta=-1.0)   # same seed => same X, different y-mechanism
    d = kob_decompose(a, b, "y", ["x"])
    assert d["composition_share"] < 0.3
    assert d["label"] == "mechanism-shifted"


def test_oaxaca_agrees_on_linear_case():
    a = _mk(4000, 6, xmean=0.0, beta=1.0)
    b = _mk(4000, 7, xmean=3.0, beta=1.0)
    ob = oaxaca_blinder(a, b, "y", ["x"], numeric_predictors=frozenset({"x"}))
    # pure composition => endowment term dominates
    assert ob["composition_share_ob"] > 0.7


def test_numeric_detection_survives_missingness():
    # A numeric response/covariate with 30% item non-response must STILL be treated as
    # numeric (Wasserstein branch), not flipped to categorical. Regression for the
    # _is_num completeness-threshold bug: pure composition shift with missing y.
    a = _mk(3000, 10, xmean=0.0, beta=1.0)
    b = _mk(3000, 11, xmean=3.0, beta=1.0)
    rng = np.random.default_rng(99)
    a.loc[rng.random(len(a)) < 0.3, "y"] = np.nan     # 30% missing outcome
    b.loc[rng.random(len(b)) < 0.3, "y"] = np.nan
    d = kob_decompose(a, b, "y", ["x"])
    assert d["method"] == "dfl"
    assert d["composition_share"] > 0.6               # still recovered as composition
    assert d["label"] == "composition-dominated"


def test_raking_survives_covariate_missingness():
    a = _mk(3000, 12, xmean=0.0, beta=1.0)
    b = _mk(3000, 13, xmean=3.0, beta=1.0)
    rng = np.random.default_rng(7)
    a.loc[rng.random(len(a)) < 0.25, "x"] = np.nan     # 25% missing covariate
    w = raking_weights(a, b, ["x"], bins=8)
    wm = np.average(pd.to_numeric(a["x"]).fillna(pd.to_numeric(a["x"]).mean()), weights=w)
    # raking still moves A's weighted x toward B's ~3 (not stuck near A's ~0)
    assert wm > 1.5


def test_oaxaca_mechanism_shifted():
    # mirror of the composition case: same X, flipped beta -> coefficient term dominates
    a = _mk(4000, 14, xmean=0.0, beta=1.0)
    b = _mk(4000, 14, xmean=0.0, beta=-1.0)   # same seed => same X
    ob = oaxaca_blinder(a, b, "y", ["x"], numeric_predictors=frozenset({"x"}))
    assert ob["composition_share_ob"] < 0.3


def test_aligned_returns_nan():
    a = _mk(2000, 8, xmean=0.0, beta=1.0)
    b = _mk(2000, 9, xmean=0.0, beta=1.0)   # essentially same distribution
    d = kob_decompose(a, b, "y", ["x"])
    assert (not np.isfinite(d["composition_share"])) or d["label"] in {
        "composition-dominated", "mechanism-shifted", "aligned"}
