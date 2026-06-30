# tests/test_strategy_design_c.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.design_c import DesignCStrategy


class _FakeClient:
    """Returns empty JSON so elicit_cell_distributions falls back to known_vector."""
    def __init__(self):
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()

    def chat(self, messages, system=None):
        self.calls += 1
        return "{}"


def _train(schema, n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for c in list(schema.background_variables) + list(schema.target_variables):
        if c in schema.numeric_ranges:
            lo, hi = schema.numeric_ranges[c]
            data[c] = rng.uniform(lo, hi, n)
        else:
            cats = schema.allowed_values.get(c) or ["a", "b"]
            data[c] = rng.choice(cats, n)
    return pd.DataFrame(data)


def _gate(condition, schema, tmp_path, source=None, crosswalk=()):
    train = _train(schema)
    bg = _train(schema, n=30, seed=1)
    return InfoGate(condition=condition, dataset_name="gss", workspace=tmp_path,
                    client=_FakeClient(), train=train, eval_rows=bg,
                    source=source, source_name="gss1994" if source is not None else None,
                    crosswalk=crosswalk)


def test_full_condition_generates_all_targets(tmp_path):
    schema = load_schema("gss")
    gate = _gate(Condition.FULL, schema, tmp_path)
    res = DesignCStrategy().generate(gate, tmp_path, cfg=type("Cfg", (), {"results_root": tmp_path})())
    for t in schema.target_variables:
        assert t in res.generated.columns
    assert len(res.generated) == 30
    assert res.meta_extras["backend"] == "design_c"
    assert json.loads(Path(tmp_path, "fit_summary.json").read_text())["transport"] is False


def test_aggregate_condition_does_not_raise(tmp_path):
    schema = load_schema("gss")
    gate = _gate(Condition.NO_DATA, schema, tmp_path)
    res = DesignCStrategy().generate(gate, tmp_path, cfg=type("Cfg", (), {"results_root": tmp_path})())
    assert len(res.generated) == 30
    assert res.meta_extras["transport"] is False


def test_full_condition_is_deterministic(tmp_path):
    schema = load_schema("gss")
    g1 = _gate(Condition.FULL, schema, tmp_path / "a")
    g2 = _gate(Condition.FULL, schema, tmp_path / "b")
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    r1 = DesignCStrategy().generate(g1, tmp_path / "a", cfg=type("Cfg", (), {"results_root": tmp_path})())
    r2 = DesignCStrategy().generate(g2, tmp_path / "b", cfg=type("Cfg", (), {"results_root": tmp_path})())
    # same train/bg seeds -> identical output
    pd.testing.assert_frame_equal(r1.generated, r2.generated)


def test_transfer_condition_end_to_end_no_leakage(tmp_path):
    schema = load_schema("gss")
    source = _train(schema, n=200, seed=2)
    crosswalk = tuple(list(schema.background_variables) + list(schema.target_variables))
    n_eval = 30  # _gate uses _train(schema, n=30, seed=1) for eval_rows

    # --- first run ---
    g1 = _gate(Condition.TRANSFER, schema, tmp_path / "a",
               source=source, crosswalk=crosswalk)
    (tmp_path / "a").mkdir()
    cfg = type("Cfg", (), {"results_root": tmp_path})()
    res = DesignCStrategy().generate(g1, tmp_path / "a", cfg=cfg)

    # (a) all target columns present, correct row count
    for t in schema.target_variables:
        assert t in res.generated.columns
    assert len(res.generated) == n_eval

    # (b) transport flag in meta_extras
    assert res.meta_extras["transport"] is True

    # (c) transport flag in written fit_summary.json
    fit_summary = json.loads(Path(tmp_path / "a", "fit_summary.json").read_text())
    assert fit_summary["transport"] is True

    # (d) no-leakage: every emitted value traces to a source donor
    generated = res.generated
    for t in schema.target_variables:
        if t not in generated.columns:
            continue
        if t in schema.numeric_ranges:
            gen_vals = set(round(float(v), 6) for v in generated[t])
            src_vals = set(round(float(v), 6) for v in source[t])
        else:
            gen_vals = set(generated[t])
            src_vals = set(source[t])
        assert gen_vals <= src_vals, (
            f"Leakage detected in target '{t}': values not in source: {gen_vals - src_vals}"
        )

    # (e) determinism: two identically-built gates yield identical output
    g2 = _gate(Condition.TRANSFER, schema, tmp_path / "b",
               source=source, crosswalk=crosswalk)
    (tmp_path / "b").mkdir()
    res2 = DesignCStrategy().generate(g2, tmp_path / "b", cfg=cfg)
    pd.testing.assert_frame_equal(res.generated, res2.generated)
