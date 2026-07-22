# tests/test_transfer_generate.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.transfer.generate import transfer_build


def _frame(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1, n)
    return pd.DataFrame({
        "x": x,
        "y": x * 1.5 + rng.normal(0, 0.5, n),          # numeric, correlated with x
        "g": np.where(x > shift, "hi", "lo"),           # categorical, tied to x
    })


def test_carryover_matches_bracket_copula_fixed():
    import nodonor_bracket as nb
    a = _frame(400, 1)
    cols = ["x", "y", "g"]
    got = transfer_build(a, a, cols, n=300, seed=7, mode="carryover")
    ref = nb.build(a, cols, n=300, seed=7, mode="copula-fixed")
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref.reset_index(drop=True))


def test_marginal_swap_takes_target_marginals():
    a = _frame(500, 2, shift=0.0)          # x ~ N(0,1)
    b = _frame(500, 3, shift=5.0)          # x ~ N(5,1) -- clearly different marginal
    out = transfer_build(a, b, ["x", "y", "g"], n=2000, seed=9, mode="marginal-swap")
    # swapped output's x marginal follows B (mean ~5), not A (mean ~0)
    assert abs(pd.to_numeric(out["x"]).mean() - 5.0) < 0.5
    assert abs(pd.to_numeric(out["x"]).mean() - 0.0) > 3.0


def test_marginal_swap_preserves_copula():
    # A has strong x~y rank dependence; after swapping B's marginals the dependence survives
    a = _frame(600, 4)
    b = _frame(600, 5, shift=5.0)
    out = transfer_build(a, b, ["x", "y", "g"], n=3000, seed=11, mode="marginal-swap")
    r = pd.to_numeric(out["x"]).corr(pd.to_numeric(out["y"]), method="spearman")
    assert r > 0.6          # positive dependence carried over from A's copula


def test_rejects_unknown_mode():
    a = _frame(50, 1)
    with pytest.raises(ValueError):
        transfer_build(a, a, ["x"], n=10, seed=1, mode="nonsense")
