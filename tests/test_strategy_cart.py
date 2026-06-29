from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import CartStrategy, cart_generate


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["income", "vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def _train(n=300, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n).astype(float)
    region = rng.choice(["N", "S"], n)
    income = (age * 1.2 + rng.normal(0, 20, n)).clip(0, 200)
    # vote varies WITHIN every region -> a non-collapsed conditional
    vote = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame({"age": age, "region": region, "income": income, "vote": vote})


def test_cart_respects_allowed_and_ranges():
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:50].reset_index(drop=True)
    out = cart_generate(train, bg, s, seed=42)
    assert len(out) == 50
    assert set(out["vote"].unique()).issubset({"A", "B", "C"})
    assert out["income"].between(0, 200).all()


def test_cart_is_deterministic():
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:50].reset_index(drop=True)
    a = cart_generate(train, bg, s, seed=42)
    b = cart_generate(train, bg, s, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_cart_does_not_collapse_variance():
    # vote is genuinely diverse given background -> sampled output must not be constant
    s, train = toy_schema(), _train()
    bg = train[["age", "region"]].iloc[:100].reset_index(drop=True)
    out = cart_generate(train, bg, s, seed=42)
    assert out["vote"].nunique() > 1
