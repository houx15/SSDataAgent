import json
from pathlib import Path

from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.design_a import Structure, elicit_structure, _validate_structure


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()

    def chat(self, messages, system=None):
        self.calls += 1
        return self.payload


def test_validate_prunes_illegal_parents():
    targets = ["t1", "t2"]
    backgrounds = ["age", "educ"]
    obj = {"order": ["t1", "t2"],
           "parents": {"t1": ["age", "t2", "bogus"], "t2": ["t1", "educ"]},
           "prior_scale": {"t1": 2.0}, "offsets": {}}
    s = _validate_structure(obj, targets, backgrounds, transport=False)
    # t1 cannot have t2 (later) or bogus (unknown) as parent; t2 may have t1 (earlier)
    assert s.parents["t1"] == ["age"]
    assert s.parents["t2"] == ["t1", "educ"]
    assert s.prior_scale["t1"] == 2.0
    assert s.prior_scale["t2"] == 1.0          # default
    assert s.offsets["t1"] == 0.0              # transport=False zeros offsets


def test_validate_completes_partial_order():
    s = _validate_structure({"order": ["t2"]}, ["t1", "t2"], ["age"], transport=False)
    assert set(s.order) == {"t1", "t2"} and len(s.order) == 2


def test_elicit_fallback_on_bad_json(tmp_path):
    schema = load_schema("gss")
    targets = list(schema.target_variables)[:2]
    bgs = list(schema.background_variables)
    c = _Client("not json at all")
    s = elicit_structure(c, dataset="gss", condition="full", schema=schema, targets=targets,
                         backgrounds=bgs, run_dir=tmp_path, cache_dir=tmp_path / "cache")
    assert set(s.order) == set(targets)
    assert all(s.parents[t] == bgs for t in targets)   # default = all backgrounds


def test_elicit_caches(tmp_path):
    schema = load_schema("gss")
    targets = list(schema.target_variables)[:2]
    bgs = list(schema.background_variables)
    payload = json.dumps({"order": targets, "parents": {t: bgs[:1] for t in targets},
                          "prior_scale": {}, "offsets": {}})
    c = _Client(payload)
    kw = dict(dataset="gss", condition="full", schema=schema, targets=targets,
              backgrounds=bgs, run_dir=tmp_path, cache_dir=tmp_path / "cache")
    elicit_structure(c, **kw)
    assert c.calls == 1
    elicit_structure(c, **kw)            # cache hit
    assert c.calls == 1
