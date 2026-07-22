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


def test_nearest_correlation_holds_eigenvalue_floor_under_adversarial_sweep():
    """Regression for the eigenvalue-floor bug: the old implementation clipped
    eigenvalues to `eps`, reconstructed, then renormalized the diagonal to 1
    via `A / outer(d, d)`. That renormalization is a congruence transform --
    it keeps the sign of positive-definiteness but does NOT preserve the
    eigenvalue floor, so post-renormalization minimum eigenvalues could fall
    back below `eps` (observed as low as ~7.1e-7 against eps=1e-6 in an
    empirical sweep). This asserts the floor itself (>= eps), not merely
    >= 0, across a deterministic sweep of random near-singular / indefinite
    matrices spanning several dimensions -- so it fails against the old
    implementation.
    """
    eps = 1e-6
    rng = np.random.default_rng(20260722)
    for dim in range(2, 13):
        for trial in range(30):
            # Build an adversarial symmetric matrix: a random Gaussian matrix,
            # symmetrized, with unit diagonal forced -- generically indefinite
            # and, for larger dims, often numerically near-singular.
            M = rng.standard_normal((dim, dim))
            A = (M + M.T) / 2.0
            np.fill_diagonal(A, 1.0)
            # Occasionally scale off-diagonals up to push further from PSD.
            if trial % 3 == 0:
                A = A * (1.2 if trial % 2 == 0 else 1.0)
                np.fill_diagonal(A, 1.0)

            good = nearest_correlation(A, eps=eps)
            w = np.linalg.eigvalsh(good)

            assert w.min() >= eps, (
                f"dim={dim} trial={trial}: min eigenvalue {w.min():.3e} "
                f"below floor eps={eps:.1e}"
            )
            assert np.allclose(good, good.T)
            assert np.allclose(np.diag(good), 1.0)


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
