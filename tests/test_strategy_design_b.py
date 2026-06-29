import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.design_b import DesignBStrategy


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["vote"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={}, population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


class FixedClient:
    cfg = type("C", (), {"model": "m"})()
    def chat(self, messages, system=None):
        # always emit vote distribution [0.5, 0.5] over support ["A","B"]
        return json.dumps({"vote": [0.5, 0.5]})


def _gate(condition, train, ev, **kw):
    base = dict(condition=condition, dataset_name="toy", workspace=Path("/tmp"),
                client=FixedClient(), train=train, eval_rows=ev)
    base.update(kw)
    return InfoGate(**base)


def test_design_b_full_calibrates_to_known_marginal(tmp_path):
    with patch("ssdataagent.strategies.base.load_schema", side_effect=lambda n: toy_schema()), \
         patch("ssdataagent.strategies.design_b.load_schema", side_effect=lambda n: toy_schema()), \
         patch("ssdataagent.strategies.elicitation.DatasetSchema", DatasetSchema):
        s = toy_schema()
        # train marginal heavily favors A (0.8/0.2)
        train = pd.DataFrame({"region": ["N", "S"] * 50,
                              "vote": (["A"] * 80 + ["B"] * 20)})
        ev = pd.DataFrame({"region": ["N", "S"] * 50})
        gate = _gate(Condition.FULL, train, ev)
        cfg = type("Cfg", (), {"results_root": tmp_path})()
        res = DesignBStrategy().generate(gate, tmp_path, cfg)
        assert len(res.generated) == 100
        assert set(res.generated["vote"].unique()).issubset({"A", "B"})
        # elicited [0.5,0.5] raked toward the train marginal 0.8 -> majority A
        share_A = (res.generated["vote"] == "A").mean()
        assert share_A > 0.6
        assert res.meta_extras["backend"] == "design_b"


def test_design_b_aggregate_condition_no_microdata(tmp_path):
    with patch("ssdataagent.strategies.base.load_schema", side_effect=lambda n: toy_schema()), \
         patch("ssdataagent.strategies.design_b.load_schema", side_effect=lambda n: toy_schema()):
        s = toy_schema()
        train = pd.DataFrame({"region": ["N", "S"] * 50, "vote": (["A"] * 60 + ["B"] * 40)})
        ev = pd.DataFrame({"region": ["N", "S"] * 50})
        gate = _gate(Condition.NO_DATA, train, ev)   # fit_microdata() is None; known_marginals from train
        cfg = type("Cfg", (), {"results_root": tmp_path})()
        res = DesignBStrategy().generate(gate, tmp_path, cfg)   # must NOT raise
        assert len(res.generated) == 100
        assert (tmp_path / "fit_summary.json").exists()
