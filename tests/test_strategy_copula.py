from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import CopulaStrategy, copula_generate


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["income", "vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def _train(n=400, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 90, n)
    region = rng.choice(["N", "S"], n)
    income = (age * 1.8 + rng.normal(0, 8, n)).clip(0, 200)  # strong age->income
    vote = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame({"age": age, "region": region, "income": income, "vote": vote})


def test_copula_respects_allowed_and_ranges():
    s, train = toy_schema(), _train()
    bg = pd.DataFrame({"age": np.linspace(20, 88, 60), "region": ["N", "S"] * 30})
    out = copula_generate(train, bg, s, seed=42)
    assert len(out) == 60
    assert set(out["vote"].unique()).issubset({"A", "B", "C"})
    assert out["income"].between(0, 200).all()


def test_copula_is_deterministic():
    s, train = toy_schema(), _train()
    bg = pd.DataFrame({"age": np.linspace(20, 88, 60), "region": ["N", "S"] * 30})
    a = copula_generate(train, bg, s, seed=42)
    b = copula_generate(train, bg, s, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_copula_conditions_on_background():
    # generated income should track the strong age->income relationship
    s, train = toy_schema(), _train()
    bg = pd.DataFrame({"age": np.linspace(20, 88, 120), "region": ["N", "S"] * 60})
    out = copula_generate(train, bg, s, seed=42)
    r = np.corrcoef(bg["age"].to_numpy(), out["income"].to_numpy())[0, 1]
    assert r > 0.5  # positive conditioning preserved
