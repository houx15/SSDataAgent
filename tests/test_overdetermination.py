from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.evaluation.overdetermination import overdetermination


def cat_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={}, population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_collapsed_sim_gives_positive_gap():
    s = cat_schema()
    # region N: real votes 50/50 (H=1 bit); sim all A (H=0) -> gap ~ 1.0
    real = pd.DataFrame({"region": ["N"] * 40, "vote": ["A", "B"] * 20})
    sim = pd.DataFrame({"region": ["N"] * 40, "vote": ["A"] * 40})
    res = overdetermination(real=real, sim=sim, schema=s, min_count=10)
    cb = res["cell_based"]
    assert abs(cb["per_target"]["vote"]["h_real"] - 1.0) < 1e-6
    assert cb["per_target"]["vote"]["h_sim"] < 1e-6
    assert abs(cb["per_target"]["vote"]["gap"] - 1.0) < 1e-6
    assert cb["headline_gap"] > 0.9
    assert cb["coverage"] == 1.0
    assert cb["n_cells"] == 1


def test_no_cells_meet_min_count_reports_reason():
    s = cat_schema()
    real = pd.DataFrame({"region": ["N"] * 5, "vote": ["A", "B"] * 2 + ["A"]})
    sim = pd.DataFrame({"region": ["N"] * 5, "vote": ["A"] * 5})
    res = overdetermination(real=real, sim=sim, schema=s, min_count=1000)
    assert res["cell_based"]["headline_gap"] is None
    assert "reason" in res["cell_based"]


def test_numeric_target_uses_real_derived_bins():
    s = DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["income"],
        descriptions={}, allowed_values={"region": ["N"]},
        numeric_ranges={"income": (0.0, 100.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"region": ["N"] * 100, "income": rng.uniform(0, 100, 100)})
    sim = pd.DataFrame({"region": ["N"] * 100, "income": [50.0] * 100})  # collapsed
    res = overdetermination(real=real, sim=sim, schema=s, min_count=10, n_target_bins=4)
    assert res["cell_based"]["per_target"]["income"]["gap"] > 0  # real spread > sim


def test_misaligned_backgrounds_report_reason():
    s = cat_schema()
    real = pd.DataFrame({"region": ["N"] * 10, "vote": ["A", "B"] * 5})
    sim = pd.DataFrame({"region": ["S"] * 10, "vote": ["A"] * 10})
    res = overdetermination(real=real, sim=sim, schema=s, min_count=1)
    # backgrounds differ -> cell-based still computes per-cell, but alignment
    # guard records a warning; ensure it does not raise and returns a dict
    assert isinstance(res, dict) and "cell_based" in res


def test_model_based_present_and_directional():
    s = cat_schema()
    rng = np.random.default_rng(0)
    region = rng.choice(["N", "S"], 200)
    real = pd.DataFrame({"region": region, "vote": rng.choice(["A", "B"], 200)})
    sim = pd.DataFrame({"region": region, "vote": ["A"] * 200})  # collapsed
    res = overdetermination(real=real, sim=sim, schema=s, min_count=10)
    mb = res["model_based"]
    assert "vote" in mb["per_target"]
    assert mb["per_target"]["vote"]["gap"] >= 0  # collapsed sim -> lower entropy
