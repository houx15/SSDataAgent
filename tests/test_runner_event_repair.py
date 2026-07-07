from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ssdataagent.data.loader import load_real_data
from ssdataagent.experiments.runner import _maybe_repair_event_timing

ADDHEALTH_EV = ["age_started_work", "age_at_first_sex", "age_at_first_marriage"]


def _gate(donor):
    return SimpleNamespace(fit_microdata=lambda: donor)


def _cfg(**kw):
    base = dict(event_timing_repair=True, event_timing_condition_cols={}, seed=0)
    base.update(kw)
    return SimpleNamespace(**base)


def _frames():
    n = 60
    donor = pd.DataFrame({
        "gender": ["male"] * 30 + ["female"] * 30,
        "race": ["black"] * 60,
        "age_started_work": [18] * 60,
        "age_at_first_sex": [15] * 60,
        "age_at_first_marriage": [25] * 60,
    })
    gen = pd.DataFrame({
        "gender": ["male"] * 30 + ["female"] * 30,
        "race": ["black"] * 60,
        "age_started_work": [0] * n,
        "age_at_first_sex": [0] * n,
        "age_at_first_marriage": [0] * n,
    })
    return donor, gen


def test_repair_disabled_is_noop():
    donor, gen = _frames()
    out, meta = _maybe_repair_event_timing(gen, gate=_gate(donor), dataset="addhealth",
                                           cfg=_cfg(event_timing_repair=False))
    assert meta == {}
    pd.testing.assert_frame_equal(out, gen)


def test_repair_noop_when_no_donor():
    _, gen = _frames()
    out, meta = _maybe_repair_event_timing(gen, gate=_gate(None), dataset="addhealth", cfg=_cfg())
    assert meta == {}
    pd.testing.assert_frame_equal(out, gen)


def test_repair_replaces_event_cols_when_enabled():
    donor, gen = _frames()
    out, meta = _maybe_repair_event_timing(gen, gate=_gate(donor), dataset="addhealth", cfg=_cfg())
    assert meta["event_timing_repair"]["event_vars"] == ADDHEALTH_EV
    assert list(out["age_started_work"].unique()) == [18]
    assert list(out["age_at_first_marriage"].unique()) == [25]


def test_repair_noop_for_cross_sectional_dataset():
    donor, gen = _frames()
    out, meta = _maybe_repair_event_timing(gen, gate=_gate(donor), dataset="gss", cfg=_cfg())
    assert meta == {}  # gss has no T4 event config


def test_load_real_data_full_source_sample():
    df = load_real_data("addhealth", n_sample=200, seed=1)
    assert len(df) == 200
    assert "age_at_first_sex" in df.columns


def test_load_real_data_sample_requires_full_source():
    with pytest.raises(ValueError, match="no full_source_path"):
        load_real_data("cps", n_sample=100)
