# tests/test_design_a_nodes.py
import numpy as np

from ssdataagent.strategies.design_a import (
    _design_matrix, fit_numeric_node, fit_categorical_node, sample_node,
)


class _Schema:
    numeric_ranges = {"x": (0.0, 10.0)}
    allowed_values = {"c": ["a", "b"]}


def test_design_matrix_empty_parents_is_ones():
    import pandas as pd
    X, stats = _design_matrix(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), [], _Schema())
    assert X.shape == (3, 1) and np.allclose(X, 1.0)


def test_numeric_node_samples_with_spread():
    rng = np.random.default_rng(0)
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    y = 2.0 * X[:, 0] + rng.normal(0, 0.5, 50)
    m = fit_numeric_node(X, y, prior_scale=1.0)
    sup = {"kind": "num", "edges": np.linspace(0, 10, 11)}
    vals = sample_node(m, X[:5], sup, offset=0.0, rng=np.random.default_rng(1))
    assert vals.shape == (5,)
    # two draws from the same fitted model differ (full-conditional sampling, not the mean)
    a = sample_node(m, X[:5], sup, offset=0.0, rng=np.random.default_rng(1))
    b = sample_node(m, X[:5], sup, offset=0.0, rng=np.random.default_rng(2))
    assert not np.allclose(a, b)


def test_numeric_offset_shifts_mean():
    rng = np.random.default_rng(0)
    X = np.zeros((100, 1))
    y = rng.normal(5.0, 0.1, 100)
    m = fit_numeric_node(X, y, prior_scale=1.0)
    sup = {"kind": "num", "edges": np.linspace(0, 100, 11)}
    base = sample_node(m, np.zeros((100, 1)), sup, offset=0.0, rng=np.random.default_rng(3))
    shifted = sample_node(m, np.zeros((100, 1)), sup, offset=20.0, rng=np.random.default_rng(3))
    assert shifted.mean() - base.mean() > 15.0


def test_categorical_node_valid_classes_and_constant_fallback():
    rng = np.random.default_rng(0)
    X = np.random.default_rng(0).normal(size=(60, 2))
    y = np.array(["a", "b"] * 30)
    m = fit_categorical_node(X, y, prior_scale=1.0, classes=["a", "b"])
    sup = {"kind": "cat", "support": ["a", "b"]}
    vals = sample_node(m, X[:10], sup, offset=0.0, rng=rng)
    assert set(np.unique(vals)).issubset({"a", "b"})
    # single-class training -> constant model
    mc = fit_categorical_node(X, np.array(["a"] * 60), prior_scale=1.0, classes=["a", "b"])
    cv = sample_node(mc, X[:4], sup, offset=0.0, rng=rng)
    assert list(cv) == ["a", "a", "a", "a"]
