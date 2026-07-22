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


def test_target_aggregates_missing_covariate():
    # Regression test: covariates not in pool should not raise KeyError.
    # Variable crosswalks routinely drop variables absent from one context,
    # so a missing covariate is a realistic input that should be gracefully skipped.
    pool = _pool(2000, 1)
    # Request "education" as a covariate, but pool only has age/gender/income
    agg = target_aggregates(pool, ["age", "gender", "income"],
                           covariates=["age", "education"],  # "education" not in pool
                           outcomes=["income"])
    # Should return normally with expected structure, not raise KeyError
    assert "pairwise_assoc" in agg
    assert "outcome_r2" in agg
    assert "provenance" in agg
    # missing covariate filtered out gracefully; outcome_r2 computed on only age (available)
    assert isinstance(agg["outcome_r2"]["income"], (float, type(None)))
