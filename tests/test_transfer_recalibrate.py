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


# --- Task 3: Step B -- bidirectional R^2 blend ---

import pandas as pd

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.recalibrate import bidirectional_r2_blend


def _linear(n, seed, beta):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": beta * x + rng.normal(0, 1, n)})


def test_blend_weakens_toward_low_target():
    df = _linear(4000, 1, beta=2.0)           # strong R^2 (~0.8)
    own = covariate_r2(df, "y", ["x"], numeric_predictors=frozenset({"x"}))
    out = bidirectional_r2_blend(df, ["y"], ["x"], {"y": 0.2},
                                 numeric_predictors=frozenset({"x"}),
                                 rng=np.random.default_rng(0))
    new = covariate_r2(out, "y", ["x"], numeric_predictors=frozenset({"x"}))
    assert own > 0.5 and new < own and abs(new - 0.2) < 0.12
    # marginal preserved: every output value is drawn from the source column's support
    # (_marginal_map is a quantile map, not a permutation) and the mean is close
    assert set(np.round(out["y"].astype(float), 6)).issubset(
        set(np.round(df["y"].astype(float), 6)))
    assert abs(pd.to_numeric(out["y"]).mean() - df["y"].mean()) < 0.15


def test_blend_strengthens_toward_high_target():
    df = _linear(4000, 2, beta=0.3)           # weak R^2 (~0.08)
    own = covariate_r2(df, "y", ["x"], numeric_predictors=frozenset({"x"}))
    out = bidirectional_r2_blend(df, ["y"], ["x"], {"y": 0.4},
                                 numeric_predictors=frozenset({"x"}),
                                 rng=np.random.default_rng(0))
    new = covariate_r2(out, "y", ["x"], numeric_predictors=frozenset({"x"}))
    assert new > own + 0.1                     # moved up toward the higher target


def test_blend_skips_unknown_and_nonnumeric():
    df = _linear(500, 3, beta=1.0)
    df["cat"] = np.where(df["x"] > 0, "hi", "lo")
    out = bidirectional_r2_blend(df, ["y", "cat"], ["x"], {"y": None},
                                 numeric_predictors=frozenset({"x"}),
                                 rng=np.random.default_rng(0))
    # y target unknown -> unchanged; cat non-numeric -> unchanged
    assert out["y"].equals(df["y"]) and out["cat"].equals(df["cat"])


def test_blend_leaves_covariate_columns_untouched():
    # The copula/R^2 seam: Step B only rewrites outcome columns, so covariate columns
    # (and hence covariate-covariate associations Step A set) are unchanged by construction.
    df = _linear(3000, 4, beta=2.0)
    df["age"] = df["x"]                              # a covariate that is NOT an outcome
    out = bidirectional_r2_blend(df, ["y"], ["x", "age"], {"y": 0.2},
                                 numeric_predictors=frozenset({"x", "age"}),
                                 rng=np.random.default_rng(0))
    assert out["x"].equals(df["x"]) and out["age"].equals(df["age"])
    assert not out["y"].equals(df["y"])             # the outcome DID change


def test_blend_preserves_outcome_missingness_exactly():
    # Finding 1: originally-missing outcome rows must come back missing -- same
    # positions, same count -- never filled from the pole and never silently pinned
    # to the marginal's minimum value by an undefined-index cast.
    df = _linear(2000, 5, beta=2.0)                  # strong R^2 (~0.8) -> weaken path
    miss_mask = np.random.default_rng(42).random(len(df)) < 0.2
    df.loc[miss_mask, "y"] = np.nan
    own = covariate_r2(df, "y", ["x"], numeric_predictors=frozenset({"x"}))
    out = bidirectional_r2_blend(df, ["y"], ["x"], {"y": 0.2},
                                 numeric_predictors=frozenset({"x"}),
                                 rng=np.random.default_rng(0))
    out_missing = out["y"].isna()
    assert int(out_missing.sum()) == int(miss_mask.sum())
    assert (out_missing.to_numpy() == miss_mask).all()
    new = covariate_r2(out, "y", ["x"], numeric_predictors=frozenset({"x"}))
    assert new < own                                 # non-missing values still moved toward target


# --- Amendment 1: fit_coherence_alphas (shared-latent Step A knob) ---

from ssdataagent.transfer.recalibrate import fit_coherence_alphas


def test_alphas_stay_in_unit_interval():
    cols = ["a", "b", "c"]
    a_src = {("a", "b"): 0.9, ("a", "c"): 0.8, ("b", "c"): 0.7}
    a_tgt = {("a", "b"): 0.05, ("a", "c"): 0.05, ("b", "c"): 0.05}
    methods = {k: "kendall" for k in a_src}
    alphas, _ = fit_coherence_alphas(cols, a_src, a_tgt, methods)
    assert all(0.0 <= v <= 1.0 for v in alphas.values())


def test_uniform_half_target_drives_alphas_to_sqrt_half():
    cols = ["a", "b", "c"]
    a_src = {("a", "b"): 0.5, ("a", "c"): 0.5, ("b", "c"): 0.5}
    a_tgt = {("a", "b"): 0.25, ("a", "c"): 0.25, ("b", "c"): 0.25}
    methods = {k: "kendall" for k in a_src}
    alphas, n_under = fit_coherence_alphas(cols, a_src, a_tgt, methods)
    for c in cols:
        assert abs(alphas[c] - np.sqrt(0.5)) < 0.02
    assert n_under == 0


def test_all_stable_target_equals_source_leaves_alphas_at_one():
    cols = ["a", "b", "c"]
    a_src = {("a", "b"): 0.4, ("a", "c"): -0.3, ("b", "c"): 0.6}
    a_tgt = dict(a_src)
    methods = {k: "kendall" for k in a_src}
    alphas, n_under = fit_coherence_alphas(cols, a_src, a_tgt, methods)
    for c in cols:
        assert abs(alphas[c] - 1.0) < 1e-9
    assert n_under == 0


def test_unusable_method_mismatch_pairs_excluded():
    cols = ["a", "b"]
    a_src = {("a", "b"): 0.5}
    a_tgt = {("a", "b"): 0.1}
    methods = {("a", "b"): "undefined"}   # not kendall/cramers_v -> excluded
    alphas, n_under = fit_coherence_alphas(cols, a_src, a_tgt, methods)
    assert alphas["a"] == 1.0 and alphas["b"] == 1.0
    assert n_under == 0     # never entered the candidate set -- not comparable


def test_n_pairs_understrength_counts_correctly():
    cols = ["a", "b", "c"]
    a_src = {("a", "b"): 0.5, ("a", "c"): 0.1, ("b", "c"): 0.3}
    a_tgt = {("a", "b"): 0.2, ("a", "c"): 0.4, ("b", "c"): 0.3}  # a,c wants MORE than src
    methods = {k: "kendall" for k in a_src}
    alphas, n_under = fit_coherence_alphas(cols, a_src, a_tgt, methods)
    assert n_under == 1
    # the understrength pair is left as-is: alpha can only shrink, never invent strength
    assert alphas["a"] * alphas["c"] * 0.1 <= 0.1 + 1e-9


def test_missing_or_nonfinite_association_excluded():
    cols = ["a", "b", "c"]
    a_src = {("a", "b"): 0.5, ("b", "c"): float("nan")}
    a_tgt = {("a", "b"): 0.25}   # ("b", "c") missing target entirely
    methods = {("a", "b"): "kendall", ("b", "c"): "kendall"}
    alphas, n_under = fit_coherence_alphas(cols, a_src, a_tgt, methods)
    assert alphas["c"] == 1.0   # no usable pair touches c
    assert n_under == 0


def test_negligible_source_association_excluded_from_fit():
    # source magnitude below the 0.05 floor is too weak to scale by reliably
    cols = ["a", "b"]
    a_src = {("a", "b"): 0.01}
    a_tgt = {("a", "b"): 0.5}
    methods = {("a", "b"): "cramers_v"}
    alphas, n_under = fit_coherence_alphas(cols, a_src, a_tgt, methods)
    assert alphas["a"] == 1.0 and alphas["b"] == 1.0
    assert n_under == 1   # still counted as understrength even though excluded from the fit


def test_blend_constant_outcome_left_completely_unchanged():
    # Finding 2: covariate_r2 returns nan (not None) for a constant-valued outcome, so
    # a guard that only checks `own <= _EPS` never fires on nan and the column falls
    # through into the blend, flipping its dtype even though the values can't move.
    df = _linear(300, 6, beta=1.0)
    df["y"] = 5.0
    out = bidirectional_r2_blend(df, ["y"], ["x"], {"y": 0.3},
                                 numeric_predictors=frozenset({"x"}),
                                 rng=np.random.default_rng(0))
    assert out["y"].equals(df["y"])
