# tests/test_transfer_generate_b2.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.generate import transfer_build, transfer_build_b2


def _ctx(n, seed, xmean, beta):
    rng = np.random.default_rng(seed)
    age = rng.normal(xmean, 1, n)
    edu = np.where(age > xmean, "hi", "lo")
    income = beta * age + rng.normal(0, 0.5, n)
    return pd.DataFrame({"age": age, "education": edu, "income": income})


def test_b2_matches_target_marginals_and_shape():
    a = _ctx(3000, 1, xmean=0.0, beta=2.0)      # source: strong age->income
    b = _ctx(3000, 2, xmean=3.0, beta=0.5)      # target: shifted marginals, weaker mechanism
    out = transfer_build_b2(a, b, ["age", "education", "income"],
                            ["age", "education"], ["income"], n=3000, seed=7)
    assert list(out.columns) == ["age", "education", "income"]
    assert len(out) == 3000
    # T1: target marginal recovered (age mean ~ 3, not source's 0)
    assert abs(pd.to_numeric(out["age"]).mean() - 3.0) < 0.4


def test_b2_recalibrates_outcome_r2_toward_target():
    # source mechanism strong, target weak -> B2 should pull income R^2 DOWN toward target,
    # closer than B1 (which keeps the source's strong mechanism).
    a = _ctx(4000, 1, xmean=0.0, beta=2.5)
    b = _ctx(4000, 2, xmean=0.0, beta=0.4)
    cols, cov, out_y = ["age", "education", "income"], ["age", "education"], ["income"]
    np_ = frozenset({"age", "income"})
    tgt_r2 = covariate_r2(b, "income", ["age", "education"], numeric_predictors=np_)
    b1 = transfer_build(a, b, cols, 4000, 7, "marginal-swap")
    b2 = transfer_build_b2(a, b, cols, cov, out_y, n=4000, seed=7)
    r2_b1 = covariate_r2(b1, "income", ["age", "education"], numeric_predictors=np_)
    r2_b2 = covariate_r2(b2, "income", ["age", "education"], numeric_predictors=np_)
    assert abs(r2_b2 - tgt_r2) < abs(r2_b1 - tgt_r2)     # B2 closer to target than B1
