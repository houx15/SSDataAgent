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


def test_assign_handles_string_dtype_missing():
    # regression: pandas StringDtype keeps missing as NA, which `.astype(str)`
    # leaves as a float rather than the literal "nan" — so the "|".join in
    # assign used to raise "expected str instance, float found". Real GSS/NLSY/
    # AddHealth/CFPS/US frames load categorical backgrounds as StringDtype.
    s = toy_schema()
    df = pd.DataFrame({
        "age": [25.0, 40.0, 55.0, 70.0],
        "region": pd.array(["N", pd.NA, "S", pd.NA], dtype="string"),
    })
    scheme = cells.fit_scheme(df, ["age", "region"], s, n_bins=2)
    keys = cells.assign(df, scheme)
    assert len(keys) == 4
    assert all(isinstance(k, str) and "|" in k for k in keys)
    # the two NA rows land in the same (age-permitting) missing bucket
    assert keys.iloc[1].split("|")[1] == keys.iloc[3].split("|")[1]


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


def test_describe_cell_maps_numeric_index_to_range():
    s = toy_schema()
    train = pd.DataFrame({"age": np.linspace(20, 80, 40), "region": ["N", "S"] * 20})
    scheme = cells.fit_scheme(train, ["age", "region"], s, n_bins=4)
    key = cells.assign(train, scheme).iloc[0]   # e.g. "0|N"
    desc = cells.describe_cell(scheme, key)
    assert desc["region"] in ("N", "S")          # categorical passthrough
    assert desc["age"].startswith("[") and "," in desc["age"]   # numeric -> range, not a bare index
