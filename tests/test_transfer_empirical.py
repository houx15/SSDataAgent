import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_empirical_preserves_categorical_association():
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    from ssdataagent.transfer.generate import transfer_build
    from ssdataagent.transfer.copula_stability import pair_association
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.choice(["a", "b", "c"], n)
    y = np.where(x == "a", "p", np.where(x == "b", "q", "r")).astype(object)
    flip = rng.random(n) < 0.05
    y[flip] = rng.choice(["p", "q", "r"], int(flip.sum()))
    A = pd.DataFrame({"x": x, "y": y})
    v_true = pair_association(A, "x", "y")[0]
    ec = empirical_transfer(A, A, ["x", "y"], n, 1)
    tb = transfer_build(A, A, ["x", "y"], n, 1, "carryover")
    v_ec = pair_association(ec, "x", "y")[0]
    v_tb = pair_association(tb, "x", "y")[0]
    assert v_ec > 0.8 * v_true, f"EC lost the association: {v_ec} vs true {v_true}"
    assert v_ec >= v_tb - 1e-6, f"EC {v_ec} should be >= transfer_build {v_tb}"


def test_empirical_preserves_numeric_rank_copula():
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    n = 3000
    z = rng.normal(size=n)
    A = pd.DataFrame({"u": z + rng.normal(0, 0.3, n), "v": z + rng.normal(0, 0.3, n)})
    B = pd.DataFrame({"u": 100 + 5 * rng.normal(size=n), "v": rng.exponential(size=n)})
    ec = empirical_transfer(A, B, ["u", "v"], n, 1)
    rho_A = spearmanr(A["u"], A["v"]).statistic
    rho_ec = spearmanr(pd.to_numeric(ec["u"]), pd.to_numeric(ec["v"])).statistic
    assert abs(rho_ec - rho_A) < 0.1, f"rank copula not preserved: {rho_ec} vs {rho_A}"


def test_empirical_installs_target_marginals():
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    A = pd.DataFrame({"cat": ["a"] * 700 + ["b"] * 200 + ["c"] * 100,
                      "num": list(range(1000))})
    B = pd.DataFrame({"cat": ["a"] * 100 + ["b"] * 300 + ["c"] * 600,
                      "num": [x + 1000 for x in range(1000)]})
    ec = empirical_transfer(A, B, ["cat", "num"], 5000, 1)
    p = ec["cat"].value_counts(normalize=True)
    assert abs(p.get("c", 0) - 0.6) < 0.05          # target categorical proportions installed
    assert pd.to_numeric(ec["num"]).mean() > 1000   # target numeric range installed


def test_empirical_carries_missingness_rate():
    from ssdataagent.transfer.empirical_copula import empirical_transfer
    A = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 250})
    B = pd.DataFrame({"x": [1.0] * 600 + [np.nan] * 400})   # 40% NaN target
    ec = empirical_transfer(A, B, ["x"], 5000, 1)
    assert 0.30 <= ec["x"].isna().mean() <= 0.50
