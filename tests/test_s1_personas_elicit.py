# tests/test_s1_personas_elicit.py
import json

import numpy as np

from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.s1 import _validate_personas, elicit_cell_personas


def _sup(schema, t):
    return E.target_support(schema, t, n_numeric_bins=10)


def test_validate_normalizes_weights_and_dists():
    schema = load_schema("gss")
    t = next(c for c in schema.target_variables if c not in schema.numeric_ranges)
    sup = _sup(schema, t)
    L = len(sup["support"])
    kv = {t: np.full(L, 1.0 / L)}
    obj = {"subtypes": [
        {"weight": 3.0, "dists": {t: [1.0] + [0.0] * (L - 1)}},
        {"weight": 1.0, "dists": {t: [0.0] * (L - 1) + [2.0]}},   # unnormalized dist
    ]}
    subs = _validate_personas(obj, [t], {t: sup}, kv, n_personas=3)
    assert len(subs) == 2
    assert abs(sum(s["weight"] for s in subs) - 1.0) < 1e-9
    assert subs[0]["weight"] == 0.75 and subs[1]["weight"] == 0.25
    for s in subs:
        assert abs(s["dists"][t].sum() - 1.0) < 1e-9


def test_validate_fallback_to_single_known_subtype():
    schema = load_schema("gss")
    t = next(c for c in schema.target_variables if c not in schema.numeric_ranges)
    sup = _sup(schema, t)
    L = len(sup["support"])
    kv = {t: np.full(L, 1.0 / L)}
    subs = _validate_personas({"garbage": 1}, [t], {t: sup}, kv, n_personas=3)
    assert len(subs) == 1 and subs[0]["weight"] == 1.0
    assert np.allclose(subs[0]["dists"][t], kv[t])


class _PersonaClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()

    def chat(self, messages, system=None):
        self.calls += 1
        return self.payload


def test_elicit_personas_caches(tmp_path):
    schema = load_schema("gss")
    t = next(c for c in schema.target_variables if c not in schema.numeric_ranges)
    sup = _sup(schema, t)
    L = len(sup["support"])
    kv = {t: np.full(L, 1.0 / L)}
    payload = json.dumps({"subtypes": [{"weight": 1.0, "dists": {t: [1.0 / L] * L}}]})
    c = _PersonaClient(payload)
    kw = dict(dataset="gss", condition="no_data", cell_descs={"c0": {"x": "y"}},
              schema=schema, targets=[t], supports={t: sup}, known_vectors=kv,
              run_dir=tmp_path, cache_dir=tmp_path / "cache", n_personas=3)
    r1 = elicit_cell_personas(c, **kw)
    assert c.calls == 1 and "c0" in r1 and len(r1["c0"]) >= 1
    elicit_cell_personas(c, **kw)
    assert c.calls == 1            # cache hit
