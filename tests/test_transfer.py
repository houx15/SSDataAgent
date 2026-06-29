from pathlib import Path

import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.data.transfer import TRANSFER_PAIRS, compute_crosswalk


def _schema(bg, tgt, name="s") -> DatasetSchema:
    return DatasetSchema(
        name=name, real_data_path=Path("/nonexistent.csv"),
        background_variables=bg, target_variables=tgt,
        descriptions={}, allowed_values={}, numeric_ranges={},
        population_context="", ssdatabench_sim_subdir="x",
        evaluation_script="x.py", domains={},
    )


def test_transfer_pairs_mapping():
    assert TRANSFER_PAIRS["gss"] == "gss1994"
    assert TRANSFER_PAIRS["cps"] == "cps1970"


def test_compute_crosswalk_intersection_and_order():
    target = _schema(["age", "region"], ["vote", "income", "wealth"])
    source = _schema(["age", "region"], ["vote", "income"])  # source lacks 'wealth'
    target_df = pd.DataFrame(columns=["age", "region", "vote", "income", "wealth"])
    source_df = pd.DataFrame(columns=["age", "region", "vote", "income"])  # lacks wealth col
    cw = compute_crosswalk(target, source, source_df, target_df)
    assert cw == ["age", "region", "vote", "income"]      # target-schema order, wealth dropped


def test_compute_crosswalk_excludes_column_absent_in_a_frame():
    target = _schema(["age"], ["vote"])
    source = _schema(["age"], ["vote"])
    target_df = pd.DataFrame(columns=["age", "vote"])
    source_df = pd.DataFrame(columns=["age"])              # 'vote' column missing in source
    assert compute_crosswalk(target, source, source_df, target_df) == ["age"]
