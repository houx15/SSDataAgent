from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.base import InfoGate


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["age", "region"], target_variables=["vote", "income"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={"age": (18.0, 90.0), "income": (0.0, 200.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def _gate(condition, **kw):
    train = pd.DataFrame({"age": [20.0, 40.0], "region": ["N", "S"],
                          "vote": ["A", "B"], "income": [10.0, 90.0]})
    ev = pd.DataFrame({"age": [30.0], "region": ["N"], "vote": ["A"], "income": [50.0]})
    base = dict(condition=condition, dataset_name="toy", workspace=Path("/tmp"),
                client=None, train=train, eval_rows=ev)
    base.update(kw)
    return InfoGate(**base)


@patch("ssdataagent.strategies.base.load_schema", side_effect=lambda n: toy_schema())
def test_fit_microdata_per_condition(_ls):
    assert _gate(Condition.FULL).fit_microdata() is not None
    assert _gate(Condition.NO_DATA).fit_microdata() is None
    assert _gate(Condition.DIRECT).fit_microdata() is None
    src = pd.DataFrame({"age": [25.0], "region": ["S"], "vote": ["B"], "income": [70.0]})
    g = _gate(Condition.TRANSFER, source=src, source_name="toy_src",
              crosswalk=("age", "region", "vote"))
    fm = g.fit_microdata()
    assert list(fm.columns) == ["age", "region", "vote"]   # crosswalk cols only
    assert "income" not in fm.columns                       # non-crosswalk target excluded


@patch("ssdataagent.strategies.base.load_schema", side_effect=lambda n: toy_schema())
def test_known_marginals_and_associations_sources(_ls):
    # A/C compute from train; DIRECT -> None
    assert _gate(Condition.FULL).known_marginals() is not None
    assert _gate(Condition.NO_DATA).known_marginals() is not None   # C: aggregates w/o rows
    assert _gate(Condition.NO_DATA).known_associations() is not None
    assert _gate(Condition.DIRECT).known_marginals() is None
    assert _gate(Condition.DIRECT).known_associations() is None
    # B: from source, and a target var absent from crosswalk is not in the marginals
    src = pd.DataFrame({"age": [25.0, 30.0], "region": ["S", "N"], "vote": ["B", "A"],
                        "income": [70.0, 20.0]})
    g = _gate(Condition.TRANSFER, source=src, crosswalk=("age", "region", "vote"))
    km = g.known_marginals()
    assert "vote" in km and "income" not in km   # only crosswalk targets


@patch("ssdataagent.strategies.base.load_schema", side_effect=lambda n: toy_schema())
def test_transfer_no_target_leakage(_ls):
    # source rows are the ONLY microdata exposed; target eval targets never appear
    src = pd.DataFrame({"age": [25.0], "region": ["S"], "vote": ["B"], "income": [70.0]})
    g = _gate(Condition.TRANSFER, source=src, crosswalk=("age", "region", "vote"))
    fm = g.fit_microdata()
    assert fm.equals(src[["age", "region", "vote"]])
