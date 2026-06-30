import numpy as np
import pandas as pd

from ssdataagent.strategies import elicitation as E
from ssdataagent.strategies.design_c import bootstrap_pool, draw_targets


def test_bootstrap_pool_matches_marginal_in_expectation():
    # one categorical target, known marginal 70/30 over two categories
    schema = type("S", (), {"background_variables": ["age"],
                            "allowed_values": {"x": ["a", "b"]},
                            "numeric_ranges": {}})()
    bg = pd.DataFrame({"age": list(range(2000))})
    supports = {"x": {"kind": "cat", "support": ["a", "b"]}}
    known = {"x": {"probs": {"a": 0.7, "b": 0.3}}}
    pool = bootstrap_pool(bg, known, supports, schema, seed=1)
    assert len(pool) == 2000
    assert list(pool["age"]) == list(bg["age"])  # backgrounds preserved
    frac_a = (pool["x"] == "a").mean()
    assert abs(frac_a - 0.7) < 0.05


def test_draw_targets_emits_real_donor_values():
    donors = pd.DataFrame({"t": ["a", "b", "c"]})
    neighbor_idx = np.array([[0, 1, 2], [0, 1, 2]])
    # weight donor 2 ('c') overwhelmingly
    weights = np.array([1e-9, 1e-9, 1.0])
    drawn = draw_targets(neighbor_idx, weights, donors, ["t"], seed=0)
    assert list(drawn["t"]) == ["c", "c"]


def test_draw_targets_deterministic():
    donors = pd.DataFrame({"t": [10.0, 20.0, 30.0, 40.0]})
    neighbor_idx = np.array([[0, 1], [2, 3], [0, 3]])
    weights = np.array([1.0, 2.0, 1.5, 0.5])
    a = draw_targets(neighbor_idx, weights, donors, ["t"], seed=7)
    b = draw_targets(neighbor_idx, weights, donors, ["t"], seed=7)
    assert np.array_equal(a["t"], b["t"])
