import numpy as np
import pandas as pd

from ssdataagent.data.event_timing import (
    conditional_joint_repair,
    event_timing_variables,
)

EV = ["w", "s", "m"]


def _rng():
    return np.random.default_rng(0)


def test_repair_replaces_event_cols_with_donor_values():
    donor = pd.DataFrame({"g": ["A"] * 40, "w": [1] * 40, "s": [2] * 40, "m": [3] * 40})
    gen = pd.DataFrame({"g": ["A"] * 10, "w": [9] * 10, "s": [9] * 10, "m": [9] * 10})
    out = conditional_joint_repair(gen, donor, EV, condition_cols=["g"], rng=_rng())
    assert list(out["w"]) == [1] * 10
    assert list(out["s"]) == [2] * 10
    assert list(out["m"]) == [3] * 10


def test_repair_copies_whole_tuple_jointly():
    # two distinct joint tuples; every output row must be one of them intact
    donor = pd.DataFrame({
        "g": ["A"] * 50,
        "w": [1, 3] * 25, "s": [2, 2] * 25, "m": [3, 1] * 25,
    })
    gen = pd.DataFrame({"g": ["A"] * 30, "w": [0] * 30, "s": [0] * 30, "m": [0] * 30})
    out = conditional_joint_repair(gen, donor, EV, condition_cols=["g"], rng=_rng())
    tuples = set(map(tuple, out[EV].to_numpy()))
    assert tuples <= {(1, 2, 3), (3, 2, 1)}


def test_conditional_matches_on_covariate():
    # group A donors are all 10s, group B all 20s
    donor = pd.DataFrame({
        "g": ["A"] * 40 + ["B"] * 40,
        "w": [10] * 40 + [20] * 40, "s": [10] * 40 + [20] * 40, "m": [10] * 40 + [20] * 40,
    })
    gen = pd.DataFrame({"g": ["A"] * 15 + ["B"] * 15, "w": [0] * 30, "s": [0] * 30, "m": [0] * 30})
    out = conditional_joint_repair(gen, donor, EV, condition_cols=["g"], rng=_rng(), min_cell=20)
    assert (out.loc[out["g"] == "A", EV].to_numpy() == 10).all()
    assert (out.loc[out["g"] == "B", EV].to_numpy() == 20).all()


def test_continuous_covariate_is_binned_not_matched_exactly():
    """A continuous covariate (here birth_year) almost never matches on its exact
    value. Without binning every row collapses to the unconditional fallback and
    the covariate association is lost; quantile-binning must preserve it.

    Donors: the young cohort's events are all 20, the old cohort's all 40, and
    every birth_year is DISTINCT so an exact-match key would find no cell."""
    rng = np.random.default_rng(0)
    # two cohorts far apart with a wide empty gap (1960-1999) so any bin edge in
    # the gap separates them cleanly; every birth_year is distinct
    old_years = list(range(1940, 1960))    # 20 distinct, events = 40
    young_years = list(range(2000, 2020))  # 20 distinct, events = 20
    donor = pd.DataFrame({
        "birth_year": old_years + young_years,
        "w": [40] * 20 + [20] * 20, "s": [40] * 20 + [20] * 20, "m": [40] * 20 + [20] * 20,
    })
    # generated rows carry birth years never seen verbatim in the donor
    gen = pd.DataFrame({"birth_year": [2010] * 20 + [1945] * 20,
                        "w": [0] * 40, "s": [0] * 40, "m": [0] * 40})
    out = conditional_joint_repair(gen, donor, EV, condition_cols=["birth_year"],
                                   rng=rng, min_cell=15, n_bins=2)
    young = out["birth_year"] == 2010
    old = out["birth_year"] == 1945
    # binning must route each cohort to its own donor pool despite no exact match
    assert out.loc[young, "w"].mean() < 30, "young cohort should draw the ~20 donors"
    assert out.loc[old, "w"].mean() > 30, "old cohort should draw the ~40 donors"


def test_falls_back_when_cell_too_small():
    # generated row's covariate 'C' has no donors -> unconditional fallback
    donor = pd.DataFrame({"g": ["A"] * 40, "w": [5] * 40, "s": [6] * 40, "m": [7] * 40})
    gen = pd.DataFrame({"g": ["C"] * 10, "w": [0] * 10, "s": [0] * 10, "m": [0] * 10})
    out = conditional_joint_repair(gen, donor, EV, condition_cols=["g"], rng=_rng())
    # matched to the only donor tuple despite covariate mismatch
    assert list(out["w"]) == [5] * 10


def test_preserves_non_event_columns_and_missingness():
    donor = pd.DataFrame({"g": ["A"] * 40, "w": [1] * 40, "s": [np.nan] * 40, "m": [3] * 40})
    gen = pd.DataFrame({"g": ["A"] * 5, "keep": [7, 8, 9, 10, 11],
                        "w": [0] * 5, "s": [0] * 5, "m": [0] * 5})
    out = conditional_joint_repair(gen, donor, EV, condition_cols=["g"], rng=_rng())
    assert list(out["keep"]) == [7, 8, 9, 10, 11]  # untouched
    assert out["s"].isna().all()  # donor NaN carried over verbatim


def test_no_event_vars_returns_copy_unchanged():
    gen = pd.DataFrame({"g": ["A"], "x": [1]})
    out = conditional_joint_repair(gen, gen, [], condition_cols=["g"], rng=_rng())
    pd.testing.assert_frame_equal(out, gen)


def test_event_timing_variables_addhealth_and_cross_sectional():
    assert event_timing_variables("addhealth") == [
        "age_started_work", "age_at_first_sex", "age_at_first_marriage",
    ]
    # gss is cross-sectional: no type4 event config
    assert event_timing_variables("gss") == []
