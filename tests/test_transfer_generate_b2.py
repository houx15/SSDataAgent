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


def test_b2_alpha_all_one_matches_b1_exactly():
    # Amendment 1's critical regression test: with every column's fitted alpha == 1.0,
    # the shared-latent B2 vehicle IS B1 plus a knob -- it must reproduce
    # transfer_build(..., "marginal-swap") exactly (same values, same RNG order).
    #
    # target == source.copy() forces every pairwise association to be BIT-IDENTICAL
    # between source and target (same computation on identical data, no floating-point
    # drift), so fit_coherence_alphas' coordinate descent lands on alpha_c == 1.0
    # exactly for every column (not just approximately -- floating-point exactness
    # matters here because the generation loop branches on `alpha >= 1.0` and any
    # value even a hair below 1.0 would take the Bernoulli-mixing branch instead,
    # consuming extra RNG draws and breaking the byte-exact match).
    # outcomes=[] makes Step B (bidirectional_r2_blend) a true no-op: its per-outcome
    # loop never executes, so it consumes no RNG and changes no values either.
    a = _ctx(800, 1, xmean=0.0, beta=1.8)
    source = a
    target = a.copy()
    cols = ["age", "education", "income"]
    ref = transfer_build(source, target, cols, n=1500, seed=42, mode="marginal-swap")
    got = transfer_build_b2(source, target, cols, ["age", "education"], [],
                            n=1500, seed=42)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref.reset_index(drop=True))


def test_b2_shrinks_association_toward_weaker_target():
    # The core Amendment 1 mechanism at the generate.py level (not just via Step B's
    # R^2 blend): when the target pool's age<->income association is much weaker than
    # the source's, B2's fitted alpha should pull the OUTPUT association closer to the
    # target than B1's raw carryover, on the same pair Step A is meant to calibrate.
    a = _ctx(4000, 1, xmean=0.0, beta=2.5)      # source: strong age->income
    b = _ctx(4000, 2, xmean=0.0, beta=0.1)      # target: much weaker mechanism
    cols, cov, out_y = ["age", "education", "income"], ["age", "education"], ["income"]
    tgt_tau = abs(kendalltau(b["age"], pd.to_numeric(b["income"]))[0])
    b1 = transfer_build(a, b, cols, 4000, 7, "marginal-swap")
    b2 = transfer_build_b2(a, b, cols, cov, out_y, n=4000, seed=7)
    tau_b1 = abs(kendalltau(pd.to_numeric(b1["age"]), pd.to_numeric(b1["income"]))[0])
    tau_b2 = abs(kendalltau(pd.to_numeric(b2["age"]), pd.to_numeric(b2["income"]))[0])
    assert abs(tau_b2 - tgt_tau) < abs(tau_b1 - tgt_tau)


def test_b2_does_not_drop_target_nonnumeric_subpopulation():
    # Shared column "x" is fully numeric in the source but 20% non-numeric strings in
    # the target. The numeric-ness map used for target-side operations (marginal
    # mapping) must come from the TARGET pool, not the source: using the source's
    # verdict would build a numeric marginal and dropna() away the target's
    # non-numeric subpopulation, silently violating the "target marginal recovered"
    # guarantee.
    rng = np.random.default_rng(0)
    n = 600
    source = pd.DataFrame({"x": rng.normal(0, 1, n)})
    tgt_vals = rng.normal(0, 1, n).astype(object)
    non_numeric_idx = rng.choice(n, int(0.2 * n), replace=False)
    tgt_vals[non_numeric_idx] = "NA_CODE"
    target = pd.DataFrame({"x": tgt_vals})
    out = transfer_build_b2(source, target, ["x"], [], [], n=n, seed=7)
    assert (out["x"] == "NA_CODE").any()
