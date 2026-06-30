# tests/test_strategy_design_a.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.design_a import DesignAStrategy


class _FakeClient:
    """Returns "{}" — valid JSON with no keys → empty parents via _validate_structure → intercept-only GLMs."""
    def __init__(self):
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()

    def chat(self, messages, system=None):
        self.calls += 1
        return "{}"


class _StructuredClient:
    """Returns a caller-supplied JSON string — exercises a real non-empty parent
    structure through elicit_structure (non-empty parents → regression nodes)."""
    def __init__(self, response_json: str):
        self.response_json = response_json
        self.cfg = type("C", (), {"model": "structured-fake"})()

    def chat(self, messages, system=None):
        return self.response_json


def _frame(schema, n, seed):
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


def _cfg(tmp_path):
    return type("Cfg", (), {"results_root": tmp_path})()


def _gate(condition, schema, tmp_path, *, train=None, source=None, crosswalk=()):
    if train is None:
        train = _frame(schema, 200, 0)
    bg = _frame(schema, 30, 1)
    return InfoGate(condition=condition, dataset_name="gss", workspace=tmp_path,
                    client=_FakeClient(), train=train, eval_rows=bg,
                    source=source, source_name="gss1994" if source is not None else None,
                    crosswalk=crosswalk)


def test_full_generates_all_targets(tmp_path):
    schema = load_schema("gss")
    g = _gate(Condition.FULL, schema, tmp_path)
    res = DesignAStrategy().generate(g, tmp_path, _cfg(tmp_path))
    for t in schema.target_variables:
        assert t in res.generated.columns
    assert len(res.generated) == 30
    fs = json.loads(Path(tmp_path, "fit_summary.json").read_text())
    assert fs["backend"] == "design_a" and fs["transport"] is False


def test_aggregate_does_not_raise(tmp_path):
    schema = load_schema("gss")
    g = _gate(Condition.NO_DATA, schema, tmp_path)
    res = DesignAStrategy().generate(g, tmp_path, _cfg(tmp_path))
    assert len(res.generated) == 30
    for t in schema.target_variables:
        assert t in res.generated.columns


def test_full_is_deterministic(tmp_path):
    schema = load_schema("gss")
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    g1 = _gate(Condition.FULL, schema, tmp_path, train=_frame(schema, 200, 0))
    g2 = _gate(Condition.FULL, schema, tmp_path, train=_frame(schema, 200, 0))
    # identical eval rows: rebuild the same bg seed by reusing the same gate construction
    r1 = DesignAStrategy().generate(g1, tmp_path / "a", _cfg(tmp_path))
    r2 = DesignAStrategy().generate(g2, tmp_path / "b", _cfg(tmp_path))
    pd.testing.assert_frame_equal(r1.generated, r2.generated)


def test_transfer_end_to_end_no_leakage(tmp_path):
    """B fits on source, not target train. Poison the target-survey target with a
    high mean; source has a low mean. Generated must track SOURCE, proving no leak."""
    schema = load_schema("gss")
    num_t = next(t for t in schema.target_variables if t in schema.numeric_ranges)
    lo, hi = schema.numeric_ranges[num_t]
    train = _frame(schema, 200, 0); train[num_t] = hi - 0.01      # target survey: high
    source = _frame(schema, 200, 2); source[num_t] = lo + 0.01    # source: low
    crosswalk = tuple(list(schema.background_variables) + list(schema.target_variables))
    g = _gate(Condition.TRANSFER, schema, tmp_path, train=train, source=source, crosswalk=crosswalk)
    res = DesignAStrategy().generate(g, tmp_path, _cfg(tmp_path))
    fs = json.loads(Path(tmp_path, "fit_summary.json").read_text())
    assert fs["transport"] is True
    mid = (lo + hi) / 2
    assert res.generated[num_t].mean() < mid     # tracks source (low), not target train (high)


def test_structured_parents_condition_on_parent(tmp_path):
    """End-to-end: num_t conditioned on a background (num_bg); t2 conditioned on
    num_bg AND the already-generated num_t — exercising the accumulating-sampled path.

    Asserts (a) correctness: all targets generated, correct length, categoricals valid;
    (b) parent dependence: eval rows with high num_bg get higher mean generated num_t
    than rows with low num_bg, confirming the background-parent regression runs and
    not an intercept-only fallback."""
    schema = load_schema("gss")
    num_bg = next(b for b in schema.background_variables if b in schema.numeric_ranges)
    num_t = next(t for t in schema.target_variables if t in schema.numeric_ranges)
    t2 = next(t for t in schema.target_variables if t != num_t)

    # Build structure JSON: num_t <- num_bg; t2 <- num_bg + num_t (earlier target)
    response_json = json.dumps({
        "order": [num_t, t2],
        "parents": {num_t: [num_bg], t2: [num_bg, num_t]},
        "prior_scale": {},
        "offsets": {},
    })

    # Training data: num_t is an exact linear rescaling of num_bg (strong signal)
    train = _frame(schema, 400, 0)
    lo_bg, hi_bg = schema.numeric_ranges[num_bg]
    lo_t, hi_t = schema.numeric_ranges[num_t]
    train[num_t] = ((train[num_bg] - lo_bg) / (hi_bg - lo_bg)) * (hi_t - lo_t) + lo_t

    # Large eval frame (200 rows) for a stable median split (~100 rows each half)
    eval_bg = _frame(schema, 200, 1)

    gate = InfoGate(
        condition=Condition.FULL,
        dataset_name="gss",
        workspace=tmp_path,
        client=_StructuredClient(response_json),
        train=train,
        eval_rows=eval_bg,
    )

    res = DesignAStrategy().generate(gate, tmp_path, _cfg(tmp_path))

    # (a) run succeeds; every target present; correct length
    for t in schema.target_variables:
        assert t in res.generated.columns
    assert len(res.generated) == 200

    # (b) categorical validity: generated values are within schema.allowed_values
    for t in schema.target_variables:
        if t in schema.allowed_values:
            allowed = set(str(v) for v in schema.allowed_values[t])
            generated_vals = set(str(v) for v in res.generated[t].dropna())
            assert generated_vals <= allowed, f"{t}: out-of-range values {generated_vals - allowed}"

    # (c) parent dependence: high-num_bg rows must yield higher mean generated num_t
    # Both eval_bg and res.generated have reset 0-based integer index (same row order)
    median_bg = eval_bg[num_bg].median()
    lo_mask = eval_bg[num_bg] < median_bg
    hi_mask = eval_bg[num_bg] >= median_bg
    mean_lo = res.generated.loc[lo_mask, num_t].mean()
    mean_hi = res.generated.loc[hi_mask, num_t].mean()
    assert mean_hi > mean_lo + 1.0, (
        f"Parent dependence failed: mean_hi={mean_hi:.3f}, mean_lo={mean_lo:.3f}; "
        f"expected mean_hi > mean_lo + 1.0 "
        f"(num_bg={num_bg!r}, num_t={num_t!r}, t2={t2!r})"
    )
