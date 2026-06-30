# tests/test_s1_raw_raked.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.s1 import S1RawStrategy, S1RakedStrategy


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


class _SkewClient:
    """Returns a fixed, NON-degenerate skewed prob vector per target (mass on every
    bin, decaying). Skewed away from the train marginal so raking visibly moves it."""
    def __init__(self, supports):
        self.calls = 0
        self.cfg = type("C", (), {"model": "fake"})()
        self._lens = {t: (len(s["support"]) if s["kind"] == "cat" else len(s["edges"]) - 1)
                      for t, s in supports.items()}

    def chat(self, messages, system=None):
        self.calls += 1
        obj = {}
        for t, L in self._lens.items():
            v = np.array([0.5 ** i for i in range(L)], float)
            obj[t] = (v / v.sum()).tolist()
        return json.dumps(obj)


def _gate(condition, schema, tmp_path, *, client, train=None, source=None, crosswalk=(), n_eval=300):
    if train is None:
        train = _frame(schema, 400, 0)
    bg = _frame(schema, n_eval, 1)
    return InfoGate(condition=condition, dataset_name="gss", workspace=tmp_path,
                    client=client, train=train, eval_rows=bg,
                    source=source, source_name="gss1994" if source is not None else None,
                    crosswalk=crosswalk)


def _supports(schema, targets):
    return {t: E.target_support(schema, t, n_numeric_bins=10) for t in targets}


def test_raw_generates_all_targets(tmp_path):
    schema = load_schema("gss")
    targets = list(schema.target_variables)
    g = _gate(Condition.FULL, schema, tmp_path, client=_SkewClient(_supports(schema, targets)))
    res = S1RawStrategy().generate(g, tmp_path, _cfg(tmp_path))
    for t in targets:
        assert t in res.generated.columns
    fs = json.loads(Path(tmp_path, "fit_summary.json").read_text())
    assert fs["backend"] == "s1" and fs["variant"] == "raw" and fs["raked"] is False


def test_raked_marginal_closer_than_raw(tmp_path):
    schema = load_schema("gss")
    # pick a categorical target with a stable known marginal
    t = next(c for c in schema.target_variables if c not in schema.numeric_ranges)
    sup = E.target_support(schema, t, n_numeric_bins=10)
    order = sup["support"]
    train = _frame(schema, 600, 0)
    client = _SkewClient(_supports(schema, list(schema.target_variables)))
    known = E.known_vector({"probs": pd.Series(train[t]).value_counts(normalize=True).to_dict()}, sup)

    def marg(df):
        vc = pd.Series(df[t]).value_counts(normalize=True)
        return np.array([float(vc.get(v, 0.0)) for v in order])

    g_raw = _gate(Condition.FULL, schema, tmp_path / "r", client=client, train=train)
    g_rake = _gate(Condition.FULL, schema, tmp_path / "k", client=client, train=train)
    (tmp_path / "r").mkdir(); (tmp_path / "k").mkdir()
    raw = S1RawStrategy().generate(g_raw, tmp_path / "r", _cfg(tmp_path))
    rake = S1RakedStrategy().generate(g_rake, tmp_path / "k", _cfg(tmp_path))
    d_raw = np.abs(marg(raw.generated) - known).sum()
    d_rake = np.abs(marg(rake.generated) - known).sum()
    assert d_rake < d_raw


def test_raw_deterministic_and_cache(tmp_path):
    schema = load_schema("gss")
    client = _SkewClient(_supports(schema, list(schema.target_variables)))
    g = _gate(Condition.FULL, schema, tmp_path, client=client)
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    r1 = S1RawStrategy().generate(g, tmp_path / "a", _cfg(tmp_path))
    calls_after_first = client.calls
    r2 = S1RawStrategy().generate(g, tmp_path / "b", _cfg(tmp_path))
    assert client.calls == calls_after_first      # cache hit -> zero new calls
    pd.testing.assert_frame_equal(r1.generated, r2.generated)


def test_raked_transfer_no_leakage(tmp_path):
    schema = load_schema("gss")
    num_t = next(t for t in schema.target_variables if t in schema.numeric_ranges)
    lo, hi = schema.numeric_ranges[num_t]
    train = _frame(schema, 300, 0); train[num_t] = hi - 0.01      # target survey: high
    source = _frame(schema, 300, 2); source[num_t] = lo + 0.01    # source: low
    crosswalk = tuple(list(schema.background_variables) + list(schema.target_variables))
    client = _SkewClient(_supports(schema, list(schema.target_variables)))
    g = _gate(Condition.TRANSFER, schema, tmp_path, client=client, train=train,
              source=source, crosswalk=crosswalk)
    res = S1RakedStrategy().generate(g, tmp_path, _cfg(tmp_path))
    assert res.generated[num_t].mean() < (lo + hi) / 2            # tracks source, not target train
