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


def test_zero_source_entry_keeps_source_sign_not_forced_positive():
    # Finding 1: a source latent entry of exactly 0.0 carries no sign to preserve
    # (this happens for real via gaussian_copula.fit_latent_correlation's
    # nan_to_num(nan=0.0) on degenerate/zero-variance columns). Recalibrating it
    # toward a positive target must NOT silently force a positive sign -- it must
    # be left untouched.
    cols = ["x", "y"]
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    a_src = {("x", "y"): 0.0}
    a_tgt = {("x", "y"): 0.8}  # large enough to blow past the 0.10 stability threshold
    methods = {("x", "y"): "kendall"}
    Rp = recalibrate_matrix(R, cols, a_src, a_tgt, methods)
    assert Rp[0, 1] == 0.0


def test_cramers_v_target_stronger_than_source_scales_up_and_clamps():
    # Finding 2: cramers_v branch, target-stronger direction. s=0.3, t=1.0 gives a
    # 3.33x ratio on a source magnitude of 0.3 -> naive 1.0, which must clamp to
    # 0.95 -- NOT saturate at the shared 0.999 clip (that would be indistinguishable
    # from an unbounded blow-up and hides the bug this finding is about).
    cols = ["x", "y"]
    R = np.array([[1.0, 0.3], [0.3, 1.0]])
    a_src = {("x", "y"): 0.3}
    a_tgt = {("x", "y"): 1.0}
    methods = {("x", "y"): "cramers_v"}
    Rp = recalibrate_matrix(R, cols, a_src, a_tgt, methods)
    assert Rp[0, 1] > 0.3            # moved toward target strength
    assert abs(Rp[0, 1] - 0.95) < 1e-9  # clamped exactly at 0.95, not 0.999


def test_cramers_v_target_weaker_than_source_scales_down():
    # Finding 2: cramers_v branch, target-weaker direction.
    cols = ["x", "y"]
    R = np.array([[1.0, 0.6], [0.6, 1.0]])
    a_src = {("x", "y"): 0.6}
    a_tgt = {("x", "y"): 0.1}
    methods = {("x", "y"): "cramers_v"}
    Rp = recalibrate_matrix(R, cols, a_src, a_tgt, methods)
    assert 0.0 < Rp[0, 1] < 0.6


def test_cramers_v_small_source_skips_recalibration():
    # Finding 2: s < 0.05 is too weak a base to scale by -- without the guard,
    # s=0.02 -> t=0.5 is a 25x ratio that would (pre-clamp-fix) saturate at 0.999,
    # and even with the clamp would jump the entry to 0.95. The fix must instead
    # leave the source entry untouched.
    cols = ["x", "y"]
    R = np.array([[1.0, 0.02], [0.02, 1.0]])
    a_src = {("x", "y"): 0.02}
    a_tgt = {("x", "y"): 0.5}
    methods = {("x", "y"): "cramers_v"}
    Rp = recalibrate_matrix(R, cols, a_src, a_tgt, methods)
    assert abs(Rp[0, 1] - 0.02) < 1e-9


def test_concatenated_mismatch_method_string_keeps_source():
    # Finding 3: copula_stability.copula_stability encodes a source/target method
    # mismatch as e.g. "kendall/cramers_v". This must be treated as undefined (not
    # just the literal string "undefined"), or it could get diffed and routed into
    # the wrong branch.
    cols = ["x", "y"]
    R = np.array([[1.0, 0.4], [0.4, 1.0]])
    Rp = recalibrate_matrix(R, cols, {("x", "y"): 0.2}, {("x", "y"): 0.9},
                            {("x", "y"): "kendall/cramers_v"})
    assert abs(Rp[0, 1] - 0.4) < 1e-9


def test_nonfinite_association_value_keeps_source():
    # Finding 4: a NaN association VALUE (as opposed to a missing dict key) must
    # also be treated as "keep the source entry", not fed into the diff/threshold
    # math (where it would silently propagate NaN into R).
    cols = ["x", "y"]
    R = np.array([[1.0, 0.4], [0.4, 1.0]])
    Rp = recalibrate_matrix(R, cols, {("x", "y"): float("nan")}, {("x", "y"): 0.9},
                            {("x", "y"): "kendall"})
    assert abs(Rp[0, 1] - 0.4) < 1e-9
    assert np.isfinite(Rp[0, 1])
