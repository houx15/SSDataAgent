"""Disclosure metrics — the privacy axis the fidelity benchmark omits.

The point these pin: whole-row resampling and a recombining generator can be
indistinguishable on fidelity yet sit at opposite ends of re-identification risk,
and the metric must separate them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.evaluation.disclosure import disclosure_metrics


def _pool(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.integers(0, 50, n),
        "b": rng.choice(["x", "y", "z"], n),
        "c": rng.integers(0, 50, n),
    })


def test_whole_row_resample_is_near_total_disclosure():
    pool = _pool()
    resample = pool.sample(n=1000, replace=True, random_state=1).reset_index(drop=True)
    m = disclosure_metrics(resample, pool)
    assert m["copy_rate"] == 1.0, "every resampled row IS a real pool row"


def test_recombination_lowers_copy_rate():
    """Build synthetic rows by taking each column from a DIFFERENT pool row — a
    caricature of block-donor. Almost no full row should equal a real one."""
    pool = _pool()
    rng = np.random.default_rng(2)
    n = 1000
    synth = pd.DataFrame({
        c: pool[c].to_numpy()[rng.integers(0, len(pool), n)] for c in pool.columns
    })
    m = disclosure_metrics(synth, pool)
    assert m["copy_rate"] < 0.2, f"recombined rows should rarely be a real person: {m}"


def test_unique_copy_rate_flags_the_reidentifiable_ones():
    pool = _pool()
    resample = pool.sample(n=500, replace=True, random_state=3).reset_index(drop=True)
    m = disclosure_metrics(resample, pool)
    # some pool rows are unique on (a,b,c); copying them is a re-identification
    assert 0.0 < m["unique_copy_rate"] <= m["copy_rate"]


def test_excess_subtracts_the_coincidence_floor():
    pool = _pool()
    # a fresh independent real-like sample collides with the pool only by chance
    baseline = _pool(seed=99)
    synth = pool.sample(n=300, replace=True, random_state=4).reset_index(drop=True)
    m = disclosure_metrics(synth, pool, baseline=baseline)
    assert "coincidence_floor" in m
    assert m["copy_rate_excess"] == max(0.0, m["copy_rate"] - m["coincidence_floor"])
    assert m["copy_rate_excess"] > 0.5, "resampling memorises far above the floor"


def test_columns_default_to_the_shared_set():
    pool = _pool()
    synth = pool.sample(n=100, replace=True, random_state=5).copy()
    synth["extra"] = 1  # a column the pool lacks must be ignored, not crash
    m = disclosure_metrics(synth, pool)
    assert m["n_columns"] == 3
