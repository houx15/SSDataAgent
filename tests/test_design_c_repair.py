import numpy as np

from ssdataagent.strategies.design_c import repair_weights


def _induced_marginal(neighbor_idx, codes_t, w, n_bins):
    cand_codes = codes_t[neighbor_idx]
    cand_w = w[neighbor_idx]
    row_sum = cand_w.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    sel = cand_w / row_sum
    m = np.bincount(cand_codes.ravel(), weights=sel.ravel(), minlength=n_bins)[:n_bins]
    return m / len(neighbor_idx)


def test_repair_reduces_marginal_distance():
    # 4 donors, target with 2 bins; donors 0,1 -> bin 0, donors 2,3 -> bin 1
    codes = {"t": np.array([0, 0, 1, 1])}
    # every eval row sees all 4 donors as candidates
    neighbor_idx = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    supports = {"t": {"kind": "cat", "support": ["a", "b"]}}
    goal = {"t": np.array([0.25, 0.75])}  # want bin 1 dominant
    w = repair_weights(neighbor_idx, codes, goal, supports, ["t"])
    m_before = _induced_marginal(neighbor_idx, codes["t"], np.ones(4), 2)
    m_after = _induced_marginal(neighbor_idx, codes["t"], w, 2)
    d_before = np.abs(m_before - goal["t"]).sum()
    d_after = np.abs(m_after - goal["t"]).sum()
    assert d_after < d_before
    assert d_after < 1e-3  # converges on this separable case


def test_repair_handles_zero_mass_bin():
    codes = {"t": np.array([0, 0, 0])}  # no donor in bin 1
    neighbor_idx = np.array([[0, 1, 2]])
    supports = {"t": {"kind": "cat", "support": ["a", "b"]}}
    goal = {"t": np.array([0.5, 0.5])}
    w = repair_weights(neighbor_idx, codes, goal, supports, ["t"])
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)


def test_repair_no_targets_returns_unit_weights():
    w = repair_weights(np.zeros((0, 0), dtype=int), {}, {}, {}, [])
    assert w.shape == (0,)
