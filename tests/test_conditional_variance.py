"""Conditional-variance repair (the T2/T3 fix) — mean-collapse diagnosis + repair.

These pin the core promise: an outcome the raw generation makes near-deterministic
in the covariates (collapsed residual variance -> inflated covariate-R^2) is pulled
back toward a target R^2 by the per-outcome blend, while the marginal (T1) is held
on the pool and the predictor block stays coherent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.data.conditional_variance import (
    covariate_r2,
    elicit_r2_targets,
    repair_alpha,
    sample_variance_repaired,
    variance_repair_alphas,
)

PREDICTORS = ["highest_education", "gender"]
_EDU = ["low", "mid", "high", "top"]
_EDU_VAL = {"low": 0.0, "mid": 1.0, "high": 2.0, "top": 3.0}


def _population(seed, resid_sd, n=3000):
    """Income is a fixed covariate effect (2 * education) plus residual noise.
    Small ``resid_sd`` == mean-collapse (near-deterministic, R^2 -> 1); large
    ``resid_sd`` == a realistic population (modest R^2)."""
    rng = np.random.default_rng(seed)
    edu = rng.choice(_EDU, n)
    signal = 2.0 * np.array([_EDU_VAL[e] for e in edu])
    income = signal + rng.normal(0.0, resid_sd, n)
    return pd.DataFrame({
        "highest_education": edu,
        "gender": rng.choice(["M", "F"], n),
        "income": income,
    })


def test_covariate_r2_recovers_known_r2():
    # signal var = 4 * var(edu_val) = 4 * 1.25 = 5 ; resid var = 3^2 = 9 ; R^2 ~ 0.36
    pop = _population(0, resid_sd=3.0, n=20000)
    r2 = covariate_r2(pop, "income", PREDICTORS)
    assert 0.30 < r2 < 0.42, f"expected R^2 ~ 0.36, got {r2:.3f}"
    # a near-deterministic (collapsed) population has R^2 close to 1
    collapsed = _population(0, resid_sd=0.3, n=20000)
    assert covariate_r2(collapsed, "income", PREDICTORS) > 0.9


def test_repair_alpha_is_one_directional():
    # deflate 0.98 -> 0.36 : alpha = sqrt(0.36/0.98) ~ 0.606
    assert abs(repair_alpha(0.98, 0.36) - np.sqrt(0.36 / 0.98)) < 1e-9
    # target above own -> no inflation (failure is one-directional)
    assert repair_alpha(0.10, 0.50) == 1.0
    # degenerate inputs -> no repair
    assert repair_alpha(None, 0.3) == 1.0
    assert repair_alpha(0.0, 0.3) == 1.0


def test_variance_repair_alphas_from_own_generation():
    raw = _population(1, resid_sd=0.3)          # collapsed: R^2_own ~ 0.98
    alphas = variance_repair_alphas(raw, PREDICTORS, {"income": 0.36})
    assert 0.4 < alphas["income"] < 0.8, alphas


def test_repair_deflates_covariate_r2_and_locks_marginal():
    raw = _population(1, resid_sd=0.3)          # over-strong joint (mean-collapse)
    pool = _population(2, resid_sd=3.0)         # honest target: modest R^2
    target = covariate_r2(pool, "income", PREDICTORS)
    alpha = variance_repair_alphas(raw, PREDICTORS, {"income": target})

    coherent = sample_variance_repaired(
        raw, pool, ["highest_education", "gender", "income"], PREDICTORS,
        5000, np.random.default_rng(4), alpha={"income": 1.0})
    repaired = sample_variance_repaired(
        raw, pool, ["highest_education", "gender", "income"], PREDICTORS,
        5000, np.random.default_rng(4), alpha=alpha)

    r2_coh = covariate_r2(coherent, "income", PREDICTORS)
    r2_rep = covariate_r2(repaired, "income", PREDICTORS)
    # keeping the raw joint reproduces the over-strong association ...
    assert r2_coh > target + 0.2, f"coherent R^2 {r2_coh:.3f} should stay inflated"
    # ... and the per-outcome blend pulls it down toward the target
    assert r2_rep < r2_coh - 0.1, f"repair should deflate: {r2_rep:.3f} !< {r2_coh:.3f}"
    assert r2_rep < target + 0.2, f"repaired R^2 {r2_rep:.3f} should approach target {target:.3f}"

    # T1 lock: the income marginal is the pool's regardless of alpha
    assert abs(repaired["income"].std() - pool["income"].std()) < 0.5 * pool["income"].std()


def test_missingness_is_person_linked_and_hits_pool_rate():
    """Which rows are missing must track the sampled respondent's OWN missingness --
    a person with no children is the one whose age-at-first-child is blank. The
    original code scattered missingness at random, severing that link and destroying
    the covariate structure of every missing-heavy column (measured on real data:
    age_first_childbirth covariate-R^2 0.51 -> 0.03). The pool's missing *rate* is a
    supplied aggregate and must still be reproduced (that is what locks T1)."""
    rng = np.random.default_rng(0)
    n = 4000
    # `raw`: income is missing exactly for the low-education half -- a perfectly
    # person-linked (covariate-predictable) missingness pattern.
    edu = np.where(rng.random(3000) < 0.5, "low", "high")
    raw = pd.DataFrame({
        "highest_education": edu,
        "gender": rng.choice(["M", "F"], 3000),
        "income": np.where(edu == "low", np.nan, rng.normal(10.0, 2.0, 3000)),
    })
    # pool marginal carries the same 50% missing rate
    pool = raw.copy()

    out = sample_variance_repaired(
        raw, pool, ["highest_education", "gender", "income"], PREDICTORS,
        n, np.random.default_rng(7), alpha={"income": 1.0})

    rate = out["income"].isna().mean()
    assert abs(rate - 0.5) < 0.05, f"pool missing rate not reproduced: {rate:.3f}"

    # the link: among emitted rows, missingness must concentrate on 'low' education.
    miss_by_edu = out.groupby("highest_education")["income"].apply(lambda s: s.isna().mean())
    assert miss_by_edu["low"] > 0.9, f"low-edu rows should be ~all missing: {miss_by_edu.to_dict()}"
    assert miss_by_edu["high"] < 0.1, f"high-edu rows should be ~none missing: {miss_by_edu.to_dict()}"


def test_elicit_r2_targets_reads_cache(tmp_path):
    cache = tmp_path / "r2.txt"
    cache.write_text('reasoning about effect sizes...\n{"income": 0.3, "mood": 0.05}')
    out = elicit_r2_targets(["income", "mood"], PREDICTORS, client=None, model="x",
                            cache_path=str(cache))
    assert out == {"income": 0.3, "mood": 0.05}
