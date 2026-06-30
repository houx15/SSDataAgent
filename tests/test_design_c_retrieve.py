import numpy as np
import pandas as pd

from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.design_c import retrieve_candidates, encode_to_codes


def _toy_schema():
    s = load_schema("gss")
    return s


def test_retrieve_returns_k_neighbors():
    s = _toy_schema()
    bgv = list(s.background_variables)
    donors = pd.DataFrame({c: list(range(20)) for c in bgv})
    bg = pd.DataFrame({c: [0, 19] for c in bgv})
    idx = retrieve_candidates(donors, bg, s, k=5)
    assert idx.shape == (2, 5)
    # nearest neighbor of row 0 (all-zeros) should include donor 0
    assert 0 in idx[0]


def test_retrieve_keff_clamped_to_donor_count():
    s = _toy_schema()
    bgv = list(s.background_variables)
    donors = pd.DataFrame({c: [0, 1, 2] for c in bgv})
    bg = pd.DataFrame({c: [0] for c in bgv})
    idx = retrieve_candidates(donors, bg, s, k=10)
    assert idx.shape == (1, 3)


def test_encode_to_codes_categorical_and_numeric():
    s = load_schema("gss")
    targets = list(s.target_variables)[:2]
    supports = {}
    from ssdataagent.strategies import elicitation as E
    for t in targets:
        supports[t] = E.target_support(s, t, n_numeric_bins=10)
    donors = pd.DataFrame({t: [list((s.allowed_values.get(t) or [0]))[0]] * 3
                           if supports[t]["kind"] == "cat"
                           else [float(s.numeric_ranges[t][0])] * 3 for t in targets})
    codes = encode_to_codes(donors, targets, supports)
    for t in targets:
        assert codes[t].shape == (3,)
        n_bins = len(supports[t]["support"]) if supports[t]["kind"] == "cat" else len(supports[t]["edges"]) - 1
        assert codes[t].min() >= 0 and codes[t].max() < n_bins
