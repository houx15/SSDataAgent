"""No-donor event-order module (A) — knowledge-based life-course event ages.

These pin the core promise: the LLM (here, a hand-authored fixture) supplies the
ordering *distribution* and gaps; the sampler turns that into per-person event
ages whose order holds by construction, with occurrence calibrated to the pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssdataagent.data.event_order_knowledge import (
    StratumEventSpec,
    apply_event_order,
    calibrate_event_block,
    fixture_specs,
    sample_event_block,
)
from ssdataagent.data.event_timing import event_timing_variables

CFPS_EVENTS = ["age_finished_education", "age_at_first_marriage", "age_at_first_child"]


def _synthetic_pool(n=3000, seed=0):
    """A stand-in disjoint pool: the 3 cfps event ages with realistic occurrence
    rates + sentinels, plus the two demographic covariates the strata key on."""
    rng = np.random.default_rng(seed)
    edu = rng.normal(18, 3, n).round()
    marr = rng.normal(25, 4, n).round()
    child = rng.normal(28, 4, n).round()
    marr = np.where(rng.random(n) < 0.85, marr, None)          # 15% never married
    child = np.where(rng.random(n) < 0.82, child, None)        # 18% never had child
    edu = np.where(rng.random(n) < 0.98, edu, None)
    return pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "highest_education": rng.choice(
            ["primary school or below", "middle school", "high school",
             "college and above"], n),
        "age_finished_education": [x if x is not None else None for x in edu],
        "age_at_first_marriage": [x if x is not None else "never married" for x in marr],
        "age_at_first_child": [x if x is not None else "never had child" for x in child],
    })


def _numeric(s):
    return pd.to_numeric(s, errors="coerce")


def test_fixture_specs_wellformed():
    specs = fixture_specs("cfps")
    assert len(specs) >= 4, "expect >=4 strata (2 gender x 2 education buckets)"
    for key, spec in specs.items():
        assert isinstance(spec, StratumEventSpec)
        assert abs(sum(spec.ordering.values()) - 1.0) < 1e-6, f"{key} ordering !~ 1"
        assert all(m > 0 for (m, _sd) in spec.gaps.values()), f"{key} has non-positive gap"
        assert all(0.0 <= p <= 1.0 for p in spec.occurrence.values()), f"{key} bad occ"
        # every ordering label is a permutation of the cfps event columns
        for label in spec.ordering:
            assert sorted(label.split("<")) == sorted(CFPS_EVENTS), f"bad label {label}"


def test_sampler_order_by_construction_and_occurrence():
    pool = _synthetic_pool()
    specs = fixture_specs("cfps")
    demo = pool[["gender", "highest_education"]].sample(
        n=4000, replace=True, random_state=1).reset_index(drop=True)
    rng = np.random.default_rng(7)
    block = sample_event_block(demo, specs, pool, CFPS_EVENTS, rng)

    # occurrence rate per event reproduces the pool's aggregate rate
    for c in CFPS_EVENTS:
        pool_rate = _numeric(pool[c]).notna().mean()
        got_rate = _numeric(block[c]).notna().mean()
        assert abs(got_rate - pool_rate) < 0.05, f"{c}: {got_rate:.3f} vs {pool_rate:.3f}"

    # among people for whom all 3 events occurred, ages are strictly increasing
    # along the drawn order -> the realized-order distribution matches the spec:
    # ~93% land in the canonical edu<marriage<child order, none tie.
    ages = pd.DataFrame({c: _numeric(block[c]) for c in CFPS_EVENTS})
    full = ages.dropna()
    assert len(full) > 1500, "expected many fully-occurring rows"
    e, m, c = CFPS_EVENTS
    distinct = ((full[e] != full[m]) & (full[m] != full[c]) & (full[e] != full[c]))
    assert distinct.all(), "constructed ages must be strictly distinct (positive gaps)"
    canonical = ((full[e] < full[m]) & (full[m] < full[c])).mean()
    assert abs(canonical - 0.93) < 0.05, f"canonical-order fraction {canonical:.3f} != ~0.93"


def _canonical_fraction(df):
    e, m, c = CFPS_EVENTS
    a = pd.DataFrame({k: _numeric(df[k]) for k in CFPS_EVENTS}).dropna()
    return ((a[e] < a[m]) & (a[m] < a[c])).mean()


def test_calibration_preserves_order_and_tightens_marginals():
    pool = _synthetic_pool()
    specs = fixture_specs("cfps")
    demo = pool[["gender", "highest_education"]].sample(
        n=4000, replace=True, random_state=2).reset_index(drop=True)
    rng = np.random.default_rng(3)
    block = sample_event_block(demo, specs, pool, CFPS_EVENTS, rng)
    cal = calibrate_event_block(block, pool, CFPS_EVENTS)

    # order is preserved: the realized-order distribution barely moves
    assert abs(_canonical_fraction(cal) - _canonical_fraction(block)) < 0.03

    # the marriage marginal (constructed as anchor+gap, so biased young) is pulled
    # toward the pool's aggregate marriage-age marginal
    pool_marr = _numeric(pool["age_at_first_marriage"]).mean()
    before = abs(_numeric(block["age_at_first_marriage"]).mean() - pool_marr)
    after = abs(_numeric(cal["age_at_first_marriage"]).mean() - pool_marr)
    assert after < before, f"calibration should tighten marriage marginal: {after} !< {before}"
    assert after < 1.5, f"calibrated marriage mean off by {after:.2f} yr"

    # sentinels survive calibration untouched
    assert (cal["age_at_first_marriage"] == "never married").sum() > 0


def test_apply_replaces_only_event_columns_and_guards_boundary():
    pool = _synthetic_pool()
    specs = fixture_specs("cfps")
    frame = pool.sample(n=600, replace=True, random_state=5).reset_index(drop=True).copy()
    frame["mean_income_30_40"] = np.arange(len(frame))  # a non-event column
    rng = np.random.default_rng(9)
    out = apply_event_order(frame, "cfps", specs, pool, rng)

    ev = event_timing_variables("cfps")
    for c in frame.columns:
        if c not in ev:
            assert out[c].equals(frame[c]), f"non-event column {c} must be untouched"
    assert any(not out[c].equals(frame[c]) for c in ev), "event columns should change"

    # honest-boundary guard: handing it the forbidden reference as the pool raises
    with pytest.raises(ValueError):
        apply_event_order(frame, "cfps", specs, pool, rng, forbid_ref=pool)
