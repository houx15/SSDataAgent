import pandas as pd
import pytest

from ssdataagent.data.loader import load_real_data
from ssdataagent.data.splitter import split_train_eval


def test_split_sizes():
    df = load_real_data("gss")
    train, eval_ = split_train_eval(df, ratio=0.5, seed=42)
    assert len(train) + len(eval_) == len(df)
    assert abs(len(train) - 500) <= 1


def test_split_reproducibility():
    df = load_real_data("gss")
    a1, b1 = split_train_eval(df, ratio=0.5, seed=42)
    a2, b2 = split_train_eval(df, ratio=0.5, seed=42)
    pd.testing.assert_frame_equal(a1.reset_index(drop=True), a2.reset_index(drop=True))
    pd.testing.assert_frame_equal(b1.reset_index(drop=True), b2.reset_index(drop=True))


def test_no_overlap():
    df = load_real_data("gss")
    train, eval_ = split_train_eval(df, ratio=0.5, seed=42)
    assert set(train["profile_id"]).isdisjoint(set(eval_["profile_id"]))


def test_invalid_ratio_raises():
    df = pd.DataFrame({"profile_id": range(10)})
    with pytest.raises(ValueError):
        split_train_eval(df, ratio=1.5, seed=0)


def test_different_seed_different_split():
    df = load_real_data("gss")
    a, _ = split_train_eval(df, ratio=0.5, seed=1)
    b, _ = split_train_eval(df, ratio=0.5, seed=2)
    # extremely unlikely to coincide
    assert not a.reset_index(drop=True).equals(b.reset_index(drop=True))
