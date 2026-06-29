from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.baselines import (
    background_frame,
    classify_columns,
    clip_decode,
    encode_numeric,
    ordinal_encode,
)


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy",
        real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"],
        target_variables=["income", "vote"],
        descriptions={},
        allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="",
        ssdatabench_sim_subdir="toy",
        evaluation_script="x.py",
        domains={},
    )


def test_classify_columns_splits_by_schema():
    num, cat = classify_columns(toy_schema(), ["age", "region", "income", "vote"])
    assert num == ["age", "income"]
    assert cat == ["region", "vote"]


def test_encode_numeric_standardizes_and_one_hots():
    s = toy_schema()
    df = pd.DataFrame({"age": [20.0, 40.0, 60.0], "region": ["N", "S", "N"]})
    X, stats = encode_numeric(df, ["age", "region"], s)
    # age column standardized to mean 0
    assert abs(X[:, 0].mean()) < 1e-9
    # region one-hot: 2 columns (N, S)
    assert X.shape == (3, 3)
    # reusing stats on new data keeps the same scaling origin
    X2, _ = encode_numeric(pd.DataFrame({"age": [40.0], "region": ["S"]}),
                           ["age", "region"], s, stats=stats)
    assert abs(X2[0, 0]) < 1e-9  # 40 is the train mean -> 0


def test_ordinal_encode_codes_categoricals():
    s = toy_schema()
    df = pd.DataFrame({"region": ["N", "S", "Z"], "age": [20.0, 30.0, 40.0]})
    X = ordinal_encode(df, ["region", "age"], s)
    assert X[0, 0] == 0.0 and X[1, 0] == 1.0 and X[2, 0] == -1.0  # Z unknown
    assert X[2, 1] == 40.0


def test_clip_decode_clips_numeric_to_range():
    s = toy_schema()
    df = pd.DataFrame({"income": [-5.0, 250.0, 50.0], "vote": ["A", "B", "C"]})
    out = clip_decode(df, s)
    assert list(out["income"]) == [0.0, 200.0, 50.0]
    assert list(out["vote"]) == ["A", "B", "C"]


def test_background_frame_drops_targets_adds_profile_id():
    s = toy_schema()
    ev = pd.DataFrame({"age": [20.0], "region": ["N"], "income": [99.0], "vote": ["A"]})
    out = background_frame(ev, s)
    assert "income" not in out.columns and "vote" not in out.columns
    assert "profile_id" in out.columns
    assert list(out["age"]) == [20.0]
