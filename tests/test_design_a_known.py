import numpy as np

from ssdataagent.strategies.design_a import sample_from_known


def test_known_categorical_matches_marginal():
    sup = {"kind": "cat", "support": ["a", "b"]}
    vals = sample_from_known(sup, np.array([0.8, 0.2]), 5000, np.random.default_rng(0))
    assert abs((vals == "a").mean() - 0.8) < 0.03


def test_known_numeric_within_range_and_deterministic():
    sup = {"kind": "num", "edges": np.linspace(0.0, 10.0, 11)}
    vec = np.full(10, 0.1)
    a = sample_from_known(sup, vec, 200, np.random.default_rng(7))
    b = sample_from_known(sup, vec, 200, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() <= 10.0
