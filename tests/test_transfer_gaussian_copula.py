# tests/test_transfer_gaussian_copula.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ssdataagent.transfer.gaussian_copula import (
    copula_to_frame, draw_copula, fit_latent_correlation, nearest_correlation,
)


def _gauss(n, seed, rho):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return pd.DataFrame({"x": z[:, 0], "y": z[:, 1]})


def test_fit_recovers_known_correlation():
    df = _gauss(4000, 1, rho=0.6)
    R, num = fit_latent_correlation(df, ["x", "y"])
    assert num == {"x": True, "y": True}
    assert R.shape == (2, 2)
    assert abs(R[0, 1] - 0.6) < 0.06          # latent corr ~ generating rho


def test_nearest_correlation_repairs_non_psd():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    good = nearest_correlation(bad)
    w = np.linalg.eigvalsh(good)
    assert w.min() >= 0.0
    assert np.allclose(np.diag(good), 1.0)
    assert np.allclose(good, good.T)


def test_draw_and_map_reproduces_dependence_and_marginal():
    R = np.array([[1.0, 0.7], [0.7, 1.0]])
    u = draw_copula(R, 5000, seed=3)
    assert u.shape == (5000, 2)
    assert u.min() > 0.0 and u.max() < 1.0
    # inverse-CDF onto a target marginal preserves that marginal
    marg = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    frame = copula_to_frame(u, marg, ["a", "b"], {"a": True, "b": True},
                            np.random.default_rng(0))
    assert frame.shape == (5000, 2)
    # dependence survived the copula+map
    tau, _ = kendalltau(pd.to_numeric(frame["a"]), pd.to_numeric(frame["b"]))
    assert tau > 0.3
    # marginal support stays within the target's
    assert frame["a"].min() >= 0.0 and frame["a"].max() <= 99.0
