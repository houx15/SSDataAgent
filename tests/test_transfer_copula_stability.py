from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.copula_stability import (
    copula_stability,
    pair_association,
    pairwise_associations,
)


def _gauss_copula(n, seed, rho, xmean=0.0, ymean=0.0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return pd.DataFrame({"x": z[:, 0] + xmean, "y": z[:, 1] + ymean})


def test_stable_copula_different_marginals():
    a = _gauss_copula(2000, 1, rho=0.7, xmean=0.0, ymean=0.0)
    b = _gauss_copula(2000, 2, rho=0.7, xmean=5.0, ymean=9.0)  # same dependence, shifted
    tau_a, method = pair_association(a, "x", "y")
    tau_b, _ = pair_association(b, "x", "y")
    assert method == "kendall"
    assert abs(tau_a - tau_b) < 0.1        # rank-based => marginal-invariant


def test_shifted_copula_flagged():
    a = _gauss_copula(2000, 3, rho=0.7)
    b = _gauss_copula(2000, 4, rho=-0.7)   # opposite dependence
    df = copula_stability(a, b, ["x", "y"])
    row = df.iloc[0]
    assert row["abs_delta"] > 0.5
    assert row["label"] == "shifted"


def test_copula_stability_frame_shape():
    a = _gauss_copula(500, 5, rho=0.5)
    b = _gauss_copula(500, 6, rho=0.5, xmean=2.0)
    df = copula_stability(a, b, ["x", "y"])
    assert list(df.columns) == ["v1", "v2", "method", "assoc_a", "assoc_b",
                                "abs_delta", "label"]
    assert len(df) == 1                    # one unordered pair from 2 cols
    assert df.iloc[0]["label"] == "stable"


def test_method_mismatch_marks_undefined():
    # If a pair is numeric (kendall) in A but nominal (cramers_v) in B, the two
    # metrics are not comparable -> the pair must be labeled "undefined", not diffed.
    a = _gauss_copula(400, 8, rho=0.6)                     # x,y both numeric -> kendall
    b = pd.DataFrame({"x": ["p", "q", "r", "s"] * 100,      # x,y nominal -> cramers_v
                      "y": ["a", "b", "c", "d"] * 100})
    df = copula_stability(a, b, ["x", "y"])
    row = df.iloc[0]
    assert row["label"] == "undefined"
    assert "kendall" in row["method"] and "cramers_v" in row["method"]


def test_numeric_pair_with_missing_values_still_uses_kendall():
    # A numeric column with item non-response (NaN) should still be detected
    # as numeric (dropna before numeric-coercibility check), not miscategorized
    # as nominal.
    a = _gauss_copula(500, 7, rho=0.6)
    a.loc[a.sample(frac=0.2, random_state=0).index, "x"] = np.nan
    tau, method = pair_association(a, "x", "y")
    assert method == "kendall"
    assert tau == tau  # not NaN


def test_pairwise_associations_covers_every_unordered_pair():
    a = _gauss_copula(500, 9, rho=0.5)
    a["z"] = pd.Series(["p", "q"] * 250)
    out = pairwise_associations(a, ["x", "y", "z"])
    assert set(out.keys()) == {("x", "y"), ("x", "z"), ("y", "z")}
    for value, method in out.values():
        assert method in {"kendall", "cramers_v"}
    assert out[("x", "y")] == pair_association(a, "x", "y")
