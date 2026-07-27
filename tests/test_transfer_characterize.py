import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_marginal_distance_identical_is_zero():
    from ssdataagent.transfer.characterize import marginal_distance
    num = pd.Series([1.0, 2.0, 3.0, 4.0])
    d, kind = marginal_distance(num, num.copy())
    assert kind == "wasserstein" and abs(d) < 1e-9
    cat = pd.Series(["a", "b", "a", "c"])
    d, kind = marginal_distance(cat, cat.copy())
    assert kind == "tv" and abs(d) < 1e-9


def test_marginal_distance_disjoint_categoricals_is_one():
    from ssdataagent.transfer.characterize import marginal_distance
    a = pd.Series(["x", "x", "x"])
    b = pd.Series(["y", "y", "y"])
    d, kind = marginal_distance(a, b)
    assert kind == "tv" and abs(d - 1.0) < 1e-9


def test_marginal_distance_known_numeric_shift():
    from ssdataagent.transfer.characterize import marginal_distance
    # a all 0, b all 1 -> pooled SD 0.5 -> standardized values 0 vs 2 -> Wasserstein 2.0
    a = pd.Series([0.0] * 100)
    b = pd.Series([1.0] * 100)
    d, kind = marginal_distance(a, b)
    assert kind == "wasserstein" and abs(d - 2.0) < 1e-6


def test_shape_level_split_pure_level_shift():
    from ssdataagent.transfer.characterize import shape_level_split
    focal = np.repeat(np.arange(10), 20).astype(float)
    a = pd.DataFrame({"f": focal, "y": focal.copy()})
    b = pd.DataFrame({"f": focal, "y": focal + 5.0})
    r = shape_level_split(a, b, "y", "f", bins=10)
    assert abs(r["level"] - 5.0) < 1e-6
    assert r["shape"] < 1e-6
    assert r["shape_ratio"] < 1e-3


def test_shape_level_split_pure_shape_change():
    from ssdataagent.transfer.characterize import shape_level_split
    focal = np.repeat(np.arange(-5, 6), 20).astype(float)   # symmetric about 0
    a = pd.DataFrame({"f": focal, "y": np.zeros(len(focal))})
    b = pd.DataFrame({"f": focal, "y": focal.copy()})        # gradient changes, mean gap ~ 0
    r = shape_level_split(a, b, "y", "f", bins=10)
    assert abs(r["level"]) < 0.5
    assert r["shape"] > 1.0
    assert r["shape_ratio"] > 0.8
