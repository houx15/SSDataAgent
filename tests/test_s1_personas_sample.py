import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.s1 import sample_personas, S1PersonasStrategy


def test_sample_personas_respects_weights_and_support():
    # one categorical target, two subtypes: subtype 0 -> always "a", subtype 1 -> always "b"
    sup = {"kind": "cat", "support": ["a", "b"]}
    cell_personas = {"c0": [
        {"weight": 0.8, "dists": {"t": np.array([1.0, 0.0])}},
        {"weight": 0.2, "dists": {"t": np.array([0.0, 1.0])}},
    ]}
    keys = ["c0"] * 4000
    out = sample_personas(keys, cell_personas, {"t": sup}, ["t"], seed=0)
    vals = np.array(out["t"], dtype=object)
    assert set(np.unique(vals)).issubset({"a", "b"})
    assert abs((vals == "a").mean() - 0.8) < 0.03            # weight 0.8 on the "a" subtype


def test_sample_personas_deterministic():
    sup = {"kind": "num", "edges": np.linspace(0.0, 10.0, 11)}
    cp = {"c0": [{"weight": 1.0, "dists": {"t": np.full(10, 0.1)}}]}
    a = sample_personas(["c0", "c0"], cp, {"t": sup}, ["t"], seed=5)
    b = sample_personas(["c0", "c0"], cp, {"t": sup}, ["t"], seed=5)
    assert np.array_equal(np.array(a["t"]), np.array(b["t"]))


class _PersonaClient:
    def __init__(self, supports):
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()
        self._lens = {t: (len(s["support"]) if s["kind"] == "cat" else len(s["edges"]) - 1)
                      for t, s in supports.items()}

    def chat(self, messages, system=None):
        self.calls += 1
        dists = {t: [1.0 / L] * L for t, L in self._lens.items()}
        return json.dumps({"subtypes": [{"weight": 0.6, "dists": dists},
                                        {"weight": 0.4, "dists": dists}]})


def _frame(schema, n, seed):
    rng = np.random.default_rng(seed)
    data = {}
    for c in list(schema.background_variables) + list(schema.target_variables):
        if c in schema.numeric_ranges:
            lo, hi = schema.numeric_ranges[c]
            data[c] = rng.uniform(lo, hi, n)
        else:
            data[c] = rng.choice(schema.allowed_values.get(c) or ["a", "b"], n)
    return pd.DataFrame(data)


def test_personas_strategy_end_to_end(tmp_path):
    schema = load_schema("gss")
    targets = list(schema.target_variables)
    supports = {t: E.target_support(schema, t, n_numeric_bins=10) for t in targets}
    g = InfoGate(condition=Condition.NO_DATA, dataset_name="gss", workspace=tmp_path,
                 client=_PersonaClient(supports), train=_frame(schema, 300, 0),
                 eval_rows=_frame(schema, 40, 1))
    res = S1PersonasStrategy().generate(g, tmp_path, type("Cfg", (), {"results_root": tmp_path})())
    for t in targets:
        assert t in res.generated.columns
    assert len(res.generated) == 40
    fs = json.loads(Path(tmp_path, "fit_summary.json").read_text())
    assert fs["variant"] == "personas" and fs["n_personas"] == 3
