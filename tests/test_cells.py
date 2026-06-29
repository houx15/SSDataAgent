from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.data import cells


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={"age": (18.0, 90.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_fit_and_assign_consistent_across_frames():
    s = toy_schema()
    train = pd.DataFrame({"age": np.linspace(20, 80, 40), "region": ["N", "S"] * 20})
    scheme = cells.fit_scheme(train, ["age", "region"], s, n_bins=4)
    a = cells.assign(train, scheme)
    # same scheme applied to a second frame yields keys drawn from the same space
    other = pd.DataFrame({"age": [25.0, 75.0], "region": ["N", "S"]})
    b = cells.assign(other, scheme)
    assert len(a) == 40 and len(b) == 2
    assert all("|" in k for k in a)            # composite cell keys (age_bin|region)


def test_bin_edges_and_discretize():
    edges = cells.bin_edges(pd.Series([0.0, 25.0, 50.0, 75.0, 100.0]), 4)
    idx = cells.discretize(pd.Series([0.0, 100.0]), edges)
    assert idx[0] == 0 and idx[1] == len(edges) - 2


def test_metric_still_works_after_refactor():
    # the over-determination metric consumes cells.* now; smoke it
    from ssdataagent.evaluation.overdetermination import overdetermination
    s = toy_schema()
    real = pd.DataFrame({"region": ["N"] * 40, "age": np.linspace(20, 80, 40),
                         "vote": ["A", "B"] * 20})
    sim = pd.DataFrame({"region": ["N"] * 40, "age": np.linspace(20, 80, 40),
                        "vote": ["A"] * 40})
    res = overdetermination(real=real, sim=sim, schema=s, min_count=10)
    assert "cell_based" in res
