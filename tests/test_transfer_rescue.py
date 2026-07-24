import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd
from ssdataagent.transfer.rescue import outcome_features
from ssdataagent.transfer.rescue import (
    FEATURE_NAMES, PriorFit, NoiseFit, fit_prior, fit_noise, predict_r2,
)


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


def _rows(pairs):
    # pairs: list of (entropy, n_predictors, is_numeric, true_r2)
    return [dict(zip((*FEATURE_NAMES, "true_r2"), p)) for p in pairs]


def test_fit_prior_recovers_linear_signal():
    # true_r2 = 0.1 + 0.5*entropy exactly -> prediction matches at a query point.
    rows = _rows([(e, 3.0, 1.0, 0.1 + 0.5 * e) for e in (0.0, 0.25, 0.5, 0.75, 1.0)])
    prior = fit_prior(rows)
    got = prior.predict({"entropy": 0.4, "n_predictors": 3.0, "is_numeric": 1.0})
    assert abs(got - (0.1 + 0.5 * 0.4)) < 1e-6


def test_predict_limits():
    prior = fit_prior(_rows([(e, 3.0, 1.0, 0.3) for e in (0.0, 0.5, 1.0)]))  # mu == 0.3
    feats = {"entropy": 0.5, "n_predictors": 3.0, "is_numeric": 1.0}
    # sigma^2 -> 0 (huge precision on retrieval): posterior == x_co
    tiny = NoiseFit(a=1e-12, b=0.0, floor=0.0)
    assert abs(predict_r2(0.9, ess=0.5, feats=feats, prior=prior, noise=tiny) - 0.9) < 1e-3
    # x_co is None: posterior == mu
    assert abs(predict_r2(None, ess=0.5, feats=feats, prior=prior, noise=tiny) - 0.3) < 1e-9
    # low ess pushes posterior toward mu vs high ess (monotone shrinkage)
    noise = NoiseFit(a=0.0, b=0.02)
    hi = predict_r2(0.9, ess=0.65, feats=feats, prior=prior, noise=noise)
    lo = predict_r2(0.9, ess=0.10, feats=feats, prior=prior, noise=noise)
    assert abs(lo - 0.3) < abs(hi - 0.3)              # lo is closer to the prior
    assert 0.3 < lo < hi < 0.9


def test_predict_clips_to_unit_interval():
    prior = fit_prior(_rows([(e, 3.0, 1.0, 0.3) for e in (0.0, 0.5, 1.0)]))
    feats = {"entropy": 0.5, "n_predictors": 3.0, "is_numeric": 1.0}
    out = predict_r2(5.0, ess=1.0, feats=feats, prior=prior, noise=NoiseFit(a=1e-12, b=0.0))
    assert out == 1.0


def test_fit_noise_single_point_and_curve():
    # single point: all noise in the 1/ess term, sigma2 recovers the point
    nf1 = fit_noise([(0.5, 0.04)])
    assert abs(nf1.sigma2(0.5) - 0.04) < 1e-9
    # two points on sigma2 = 1/ess line: b=1, a=0
    nf2 = fit_noise([(0.5, 2.0), (0.25, 4.0)])
    assert abs(nf2.a) < 1e-6 and abs(nf2.b - 1.0) < 1e-6
    # sigma2 decreases as ess grows
    assert nf2.sigma2(1.0) < nf2.sigma2(0.1)


def test_select_r2_source_truth_table():
    from ssdataagent.transfer.rescue import select_r2_source
    # cps: plural pool, well-sized -> trust retrieval
    assert select_r2_source(3, 0.65) is True
    # gss: lone thin sibling -> fall back to prior (fails BOTH criteria)
    assert select_r2_source(1, 0.10) is False
    # plural pool but poorly raked -> prior (ESS fails)
    assert select_r2_source(3, 0.10) is False
    # well-raked but only one sibling -> prior (count fails)
    assert select_r2_source(1, 0.65) is False


def test_select_r2_source_boundaries_and_tau():
    from ssdataagent.transfer.rescue import select_r2_source
    assert select_r2_source(2, 0.30) is True          # both at threshold -> eligible
    assert select_r2_source(2, 0.2999) is False        # just below tau
    assert select_r2_source(1, 0.99) is False          # count dominates
    # tau is non-load-bearing: any value across the (0.10, 0.65) gap selects identically
    for tau in (0.15, 0.30, 0.50, 0.60):
        assert select_r2_source(3, 0.65, tau=tau) is True
        assert select_r2_source(1, 0.10, tau=tau) is False


def test_hybrid_r2_map_retrieval_branch_is_truthful():
    from ssdataagent.transfer.rescue import hybrid_r2_map
    learned = {"a": 0.80, "b": 0.20, "c": 0.35}
    prior_only = {"a": 0.30, "b": 0.20, "c": 0.35}   # b, c unmoved by retrieval
    r2_map, prov = hybrid_r2_map(learned, prior_only, use_retrieval=True)
    assert r2_map == learned
    assert prov == {"a": "retrieval-blend", "b": "prior", "c": "prior"}


def test_hybrid_r2_map_prior_branch_all_prior():
    from ssdataagent.transfer.rescue import hybrid_r2_map
    learned = {"a": 0.80, "b": 0.20}
    prior_only = {"a": 0.30, "b": 0.20}
    r2_map, prov = hybrid_r2_map(learned, prior_only, use_retrieval=False)
    assert r2_map == prior_only
    assert prov == {"a": "prior", "b": "prior"}
