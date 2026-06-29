from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import HotDeckStrategy, hotdeck_generate


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["income", "vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def _train(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n).astype(float)
    region = rng.choice(["N", "S"], n)
    income = age * 1.5 + rng.normal(0, 5, n)
    vote = np.where(region == "N", "A", "B")
    return pd.DataFrame({"age": age, "region": region,
                         "income": income.clip(0, 200), "vote": vote})


def test_hotdeck_output_targets_are_real_train_vectors():
    s = toy_schema()
    train = _train()
    bg = train[["age", "region"]].iloc[:10].reset_index(drop=True)
    out = hotdeck_generate(train, bg, s, k=5, seed=42)
    assert len(out) == 10
    train_pairs = set(zip(train["income"].round(6), train["vote"]))
    for inc, vote in zip(out["income"].round(6), out["vote"]):
        assert (inc, vote) in train_pairs  # whole target vector is a real one


def test_hotdeck_is_deterministic():
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:10].reset_index(drop=True)
    a = hotdeck_generate(train, bg, s, k=5, seed=42)
    b = hotdeck_generate(train, bg, s, k=5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_hotdeck_no_target_leakage_and_profile_id():
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:5].reset_index(drop=True)
    out = hotdeck_generate(train, bg, s, k=5, seed=42)
    assert "profile_id" in out.columns
    assert set(["age", "region", "income", "vote"]).issubset(out.columns)


def test_hotdeck_strategy_requires_microdata():
    import pytest

    class _Gate:
        dataset_name = "toy"
        def fit_microdata(self): return None
        def background(self): return pd.DataFrame()
    with pytest.raises(ValueError):
        HotDeckStrategy().generate(_Gate(), Path("/tmp"), None)
