# tests/test_transfer_recalibrate.py
from __future__ import annotations

import numpy as np

from ssdataagent.transfer.recalibrate import recalibrate_matrix, tau_to_r


def test_tau_to_r_monotone_signed():
    assert tau_to_r(0.0) == 0.0
    assert tau_to_r(1.0) > 0.99
    assert tau_to_r(-0.5) < 0.0


def test_stable_pair_kept_unstable_moved_both_directions():
    cols = ["x", "y", "z"]
    R = np.array([[1.0, 0.50, 0.20],
                  [0.50, 1.0, -0.40],
                  [0.20, -0.40, 1.0]])
    a_src = {("x", "y"): 0.30, ("x", "z"): 0.13, ("y", "z"): -0.26}
    # x,y target ~equal (stable, keep); x,z target much stronger (strengthen, up);
    # y,z target much weaker (weaken toward 0, magnitude down) keeping negative sign
    a_tgt = {("x", "y"): 0.33, ("x", "z"): 0.55, ("y", "z"): -0.05}
    methods = {("x", "y"): "kendall", ("x", "z"): "kendall", ("y", "z"): "kendall"}
    Rp = recalibrate_matrix(R, cols, a_src, a_tgt, methods)
    # stable pair unchanged
    assert abs(Rp[0, 1] - 0.50) < 1e-9
    # strengthened pair moved up, same (positive) sign
    assert Rp[0, 2] > 0.20 and Rp[0, 2] > 0.0
    # weakened pair moved toward zero, still negative
    assert -0.40 < Rp[1, 2] < 0.0
    # symmetric + valid
    assert np.allclose(Rp, Rp.T)
    assert np.linalg.eigvalsh(Rp).min() >= 0.0


def test_undefined_and_missing_pairs_keep_source():
    cols = ["x", "y"]
    R = np.array([[1.0, 0.4], [0.4, 1.0]])
    Rp = recalibrate_matrix(R, cols, {("x", "y"): 0.2}, {("x", "y"): 0.9},
                            {("x", "y"): "undefined"})
    assert abs(Rp[0, 1] - 0.4) < 1e-9      # undefined -> untouched
    Rp2 = recalibrate_matrix(R, cols, {}, {}, {("x", "y"): "kendall"})
    assert abs(Rp2[0, 1] - 0.4) < 1e-9     # missing association -> untouched
