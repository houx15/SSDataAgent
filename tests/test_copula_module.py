from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies import copula


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=[], target_variables=["vote", "income"],
        descriptions={}, allowed_values={"vote": ["A", "B"]},
        numeric_ranges={"income": (0.0, 100.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_build_cuts_and_roundtrip():
    s = toy_schema()
    df = pd.DataFrame({"vote": ["A", "B", "A", "B"], "income": [10.0, 90.0, 20.0, 80.0]})
    cuts = copula.build_cuts(df, ["vote", "income"], s)
    assert cuts["vote"]["kind"] == "cat" and cuts["income"]["kind"] == "num"
    # latent then invert returns valid support members / in-range numerics
    z = copula.latent_matrix(df, ["vote", "income"], cuts)
    assert z.shape == (4, 2)
    inv_vote = copula.invert(z[:, 0], cuts["vote"])
    assert set(inv_vote).issubset({"A", "B"})


def test_make_pd_is_positive_definite():
    M = np.array([[1.0, 0.9], [0.9, 1.0]])
    pd_M = copula.make_pd(M, 1e-6)
    assert np.all(np.linalg.eigvalsh(pd_M) > 0)


def test_correlated_normal_reproduces_correlation():
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    rng = np.random.default_rng(0)
    samples = copula.correlated_normal(Sigma, 20000, rng)
    assert samples.shape == (20000, 2)
    emp = np.corrcoef(samples, rowvar=False)[0, 1]
    assert abs(emp - 0.8) < 0.05


def test_baselines_still_imports_helpers():
    # behavior-preserving extraction: copula strategy path still works
    from ssdataagent.strategies.baselines import copula_generate
    s = toy_schema()
    train = pd.DataFrame({"vote": ["A", "B"] * 20, "income": list(np.linspace(0, 100, 40))})
    out = copula_generate(train, train[[]].assign(profile_id=range(40)), s, seed=1)
    assert set(out["vote"].unique()).issubset({"A", "B"})
