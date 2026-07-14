"""The no-leakage guarantee.

`load_disjoint_train` exists so we can score against the paper's benchmark sample
WHOLE (comparable numbers, less reference noise) while still training on microdata
the benchmark never saw. That is only sound if the exclusion is *proven*, so these
tests pin the refusals as hard as the happy path: a function that silently trains
on the rows it is about to be graded against is worse than no function.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ssdataagent.data.loader import LeakageError, load_disjoint_train, load_real_data
from ssdataagent.data.schema import load_schema


pytestmark = pytest.mark.skipif(
    load_schema("cfps").full_source_path is None
    or not load_schema("cfps").full_source_path.exists(),
    reason="cfps full source not present",
)


def test_training_pool_shares_no_row_with_the_benchmark():
    ref = load_real_data("cfps")
    train = load_disjoint_train("cfps", n_sample=5000, seed=7)
    key = load_schema("cfps").full_source_key
    assert len(train) == 5000
    assert not (set(train[key]) & set(ref[key])), "training rows leaked into the benchmark"


def test_benchmark_sample_is_scored_whole():
    """The point of the disjoint pool: we no longer have to cut the reference in
    half to manufacture held-out training rows."""
    ref = load_real_data("cfps")
    assert len(ref) == 1000, "cfps benchmark reference must stay at the paper's N"


def test_pool_is_deterministic_for_a_seed():
    key = load_schema("cfps").full_source_key
    a = load_disjoint_train("cfps", n_sample=1000, seed=3)
    b = load_disjoint_train("cfps", n_sample=1000, seed=3)
    c = load_disjoint_train("cfps", n_sample=1000, seed=4)
    assert list(a[key]) == list(b[key])
    assert list(a[key]) != list(c[key])


def test_profile_id_is_synthesized_for_the_eval():
    """SSDataBench's matched bootstrap bails out without profile_id; the raw source
    doesn't carry one (it's added during the paper's cleaning)."""
    train = load_disjoint_train("cfps", n_sample=200, seed=1)
    assert "profile_id" in train.columns
    assert train["profile_id"].notna().all()


def test_refuses_a_dataset_with_no_full_source():
    with pytest.raises(LeakageError, match="no full_source_path"):
        load_disjoint_train("gss", n_sample=100)


def test_refuses_a_dataset_with_no_declared_row_key(monkeypatch):
    """No unique key means we cannot identify the benchmark rows to exclude. Refuse
    rather than hand back a pool that might contain them."""
    real = load_schema

    def keyless(name):
        s = real(name)
        return type(s)(**{**s.__dict__, "full_source_key": None})

    monkeypatch.setattr("ssdataagent.data.loader.load_schema", keyless)
    with pytest.raises(LeakageError, match="full_source_key"):
        load_disjoint_train("cfps", n_sample=100)


def test_asking_for_more_rows_than_exist_returns_the_whole_pool():
    train = load_disjoint_train("cfps", n_sample=10_000_000, seed=1)
    ref = load_real_data("cfps")
    full = pd.read_csv(load_schema("cfps").full_source_path, low_memory=False)
    assert len(train) == len(full) - len(ref)
