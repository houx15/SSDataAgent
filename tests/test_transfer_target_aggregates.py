from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.transfer.target_aggregates import target_aggregates


def _pool(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.normal(45, 15, n)
    income = 1000 * age + rng.normal(0, 5000, n)
    gender = np.where(rng.random(n) < 0.5, "m", "f")
    return pd.DataFrame({"age": age, "gender": gender, "income": income})


def test_target_aggregates_shape_and_firewall():
    pool = _pool(2000, 1)
    agg = target_aggregates(pool, ["age", "gender", "income"], ["age", "gender"], ["income"])
    # a value + method for every unordered pair (3 choose 2 = 3)
    assert len(agg["pairwise_assoc"]) == 3
    assert set(agg["pairwise_method"].values()) <= {"kendall", "cramers_v"}
    # income R^2 on age/gender is high (income ~ age)
    assert agg["outcome_r2"]["income"] > 0.5
    # provenance names the pool source, not any test/reference sample
    assert agg["provenance"]["source"] == "target_pool"
    assert "n_rows" in agg["provenance"]


def test_target_aggregates_reads_only_pool():
    # The function signature exposes no test/reference frame — a structural firewall check.
    import inspect
    sig = inspect.signature(target_aggregates)
    assert set(sig.parameters) == {"pool", "cols", "covariates", "outcomes"}
