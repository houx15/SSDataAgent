from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.strategies.design_b import build_target_copula, sample_targets
from ssdataagent.strategies import elicitation as E


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"], target_variables=["vote", "income"],
        descriptions={}, allowed_values={"region": ["N", "S"], "vote": ["A", "B"]},
        numeric_ranges={"income": (0.0, 100.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_build_target_copula_identity_when_no_ref():
    s = toy_schema()
    Sig = build_target_copula(None, ["vote", "income"], s)
    assert np.allclose(Sig, np.eye(2))


def test_build_target_copula_from_ref_is_correlation():
    s = toy_schema()
    ref = pd.DataFrame({"vote": ["A", "B"] * 50, "income": list(np.linspace(0, 100, 100))})
    Sig = build_target_copula(ref, ["vote", "income"], s)
    assert Sig.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(Sig) > 0)


def test_sample_respects_support_and_ranges():
    s = toy_schema()
    supports = {"vote": E.target_support(s, "vote"),
                "income": E.target_support(s, "income", n_numeric_bins=4)}
    calibrated = {"N": {"vote": np.array([1.0, 0.0]),          # always A
                        "income": np.array([1.0, 0.0, 0.0, 0.0])}}  # lowest bin
    eval_cells = ["N", "N", "N"]
    out = sample_targets(eval_cells, calibrated, supports, np.eye(2),
                         ["vote", "income"], seed=1)
    assert out["vote"] == ["A", "A", "A"]                       # degenerate marginal honored
    assert all(0.0 <= x <= 25.0 for x in out["income"])         # lowest of 4 even bins of [0,100]


def test_sample_is_deterministic():
    s = toy_schema()
    supports = {"vote": E.target_support(s, "vote"),
                "income": E.target_support(s, "income", n_numeric_bins=4)}
    calibrated = {"N": {"vote": np.array([0.5, 0.5]), "income": np.full(4, 0.25)}}
    ev = ["N"] * 20
    a = sample_targets(ev, calibrated, supports, np.eye(2), ["vote", "income"], seed=7)
    b = sample_targets(ev, calibrated, supports, np.eye(2), ["vote", "income"], seed=7)
    assert a == b
