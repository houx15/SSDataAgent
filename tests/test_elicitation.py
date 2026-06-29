import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies import elicitation as E


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["vote", "income"],
        descriptions={"vote": "party"}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={"income": (0.0, 100.0)},
        population_context="ctx", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
    def chat(self, messages, system=None):
        self.calls += 1
        self.last_messages = messages
        return self.responses.pop(0)


def test_target_support_and_known_vector():
    s = toy_schema()
    sup_v = E.target_support(s, "vote")
    assert sup_v == {"kind": "cat", "support": ["A", "B"]}
    sup_i = E.target_support(s, "income", n_numeric_bins=4)
    assert sup_i["kind"] == "num" and len(sup_i["edges"]) == 5
    kv = E.known_vector({"kind": "categorical", "probs": {"A": 0.75, "B": 0.25}}, sup_v)
    assert np.allclose(kv, [0.75, 0.25])


def test_parse_renormalizes_and_caches(tmp_path):
    s = toy_schema()
    sup = {"vote": E.target_support(s, "vote"), "income": E.target_support(s, "income", n_numeric_bins=4)}
    kv = {"vote": np.array([0.5, 0.5]), "income": np.full(4, 0.25)}
    resp = json.dumps({"vote": [3, 1], "income": [1, 1, 1, 1]})  # unnormalized
    client = FakeClient([resp])
    out = E.elicit_cell_distributions(
        client, dataset="toy", condition="full_agent",
        cell_descs={"N": {"region": "N"}}, schema=s,
        targets=["vote", "income"], supports=sup, known_vectors=kv,
        run_dir=tmp_path, cache_dir=tmp_path / "cache",
    )
    assert np.allclose(out["N"]["vote"], [0.75, 0.25])      # renormalized
    assert abs(out["N"]["income"].sum() - 1.0) < 1e-9
    # cache hit: a second call with the same args makes NO new client call
    client2 = FakeClient([])  # would IndexError if called
    out2 = E.elicit_cell_distributions(
        client2, dataset="toy", condition="full_agent",
        cell_descs={"N": {"region": "N"}}, schema=s,
        targets=["vote", "income"], supports=sup, known_vectors=kv,
        run_dir=tmp_path, cache_dir=tmp_path / "cache",
    )
    assert client2.calls == 0
    assert np.allclose(out2["N"]["vote"], [0.75, 0.25])
    # raw I/O logged
    assert (tmp_path / "elicitation").exists()


def test_malformed_json_falls_back_to_known(tmp_path):
    s = toy_schema()
    sup = {"vote": E.target_support(s, "vote")}
    kv = {"vote": np.array([0.6, 0.4])}
    client = FakeClient(["not json", "still not json", "{bad", "{bad}"])  # all retries fail
    out = E.elicit_cell_distributions(
        client, dataset="toy", condition="full_agent",
        cell_descs={"N": {"region": "N"}}, schema=s,
        targets=["vote"], supports=sup, known_vectors=kv,
        run_dir=tmp_path, cache_dir=tmp_path / "cache", max_retries=3,
    )
    assert np.allclose(out["N"]["vote"], [0.6, 0.4])        # fell back to known marginal
