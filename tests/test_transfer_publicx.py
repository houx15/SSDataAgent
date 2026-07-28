import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_with_public_x_swaps_only_x_distribution():
    from ssdataagent.transfer.publicx import with_public_x
    base = pd.DataFrame({"age": [20, 20, 20, 20], "inc": [1, 2, 3, 4]})
    b = pd.DataFrame({"age": [60, 60, 60, 60, 60], "inc": [9, 9, 9, 9, 9]})
    out = with_public_x(base, b, ["age"], seed=0)
    assert set(out["age"]) == {60}                 # age now drawn from b
    assert list(out["inc"]) == [1, 2, 3, 4]        # non-x column untouched
    assert len(out) == len(base)


def test_with_public_x_empty_returns_base_unchanged():
    from ssdataagent.transfer.publicx import with_public_x
    base = pd.DataFrame({"age": [1, 2], "inc": [3, 4]})
    out = with_public_x(base, pd.DataFrame({"age": [9]}), [], seed=0)
    pd.testing.assert_frame_equal(out, base)


def test_with_public_x_preserves_missingness():
    from ssdataagent.transfer.publicx import with_public_x
    base = pd.DataFrame({"race": ["W"] * 12})
    b = pd.DataFrame({"race": ["B", "B", "B", "B", "B", "B", None, None, None, None]})  # 40% NaN
    out = with_public_x(base, b, ["race"], seed=1)
    assert 0.15 <= out["race"].isna().mean() <= 0.65     # ~40% NaN carried (resample tolerance)
    assert set(out["race"].dropna()) == {"B"}


def test_with_public_x_absent_column_unchanged():
    from ssdataagent.transfer.publicx import with_public_x
    base = pd.DataFrame({"age": [1, 2], "inc": [3, 4]})
    out = with_public_x(base, pd.DataFrame({"inc": [9, 9]}), ["age"], seed=0)
    assert list(out["age"]) == [1, 2]              # age absent from b -> left unchanged
