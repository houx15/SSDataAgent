from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.schema import DatasetSchema
from ssdataagent.data.aggregates import marginals, associations


def toy_schema() -> DatasetSchema:
    return DatasetSchema(
        name="toy", real_data_path=Path("/nonexistent.csv"),
        background_variables=["region"],
        target_variables=["vote", "income", "age"],
        descriptions={},
        allowed_values={"region": ["N", "S"], "vote": ["A", "B", "C"]},
        numeric_ranges={"income": (0.0, 200.0), "age": (18.0, 90.0)},
        population_context="", ssdatabench_sim_subdir="toy",
        evaluation_script="x.py", domains={},
    )


def test_marginals_categorical_normalizes_over_allowed():
    s = toy_schema()
    df = pd.DataFrame({"vote": ["A", "A", "B"]})  # no C present
    m = marginals(df, ["vote"], s)
    assert m["vote"]["kind"] == "categorical"
    probs = m["vote"]["probs"]
    assert set(probs) == {"A", "B", "C"}          # missing C present at 0
    assert abs(probs["A"] - 2 / 3) < 1e-9 and probs["C"] == 0.0
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_marginals_categorical_sums_to_one_with_out_of_domain_value():
    s = toy_schema()
    df = pd.DataFrame({"vote": ["A", "B", "Z"]})  # 'Z' not in allowed_values ["A","B","C"]
    m = marginals(df, ["vote"], s)
    probs = m["vote"]["probs"]
    assert set(probs) == {"A", "B", "C"}            # only allowed cats
    assert abs(sum(probs.values()) - 1.0) < 1e-9     # normalized over in-domain count
    assert "Z" not in probs


def test_marginals_numeric_quantiles_and_moments():
    s = toy_schema()
    df = pd.DataFrame({"income": [0.0, 50.0, 100.0, 150.0, 200.0]})
    m = marginals(df, ["income"], s, n_bins=4)
    assert m["income"]["kind"] == "numeric"
    assert m["income"]["quantiles"][1.0] == 200.0 and m["income"]["quantiles"][0.0] == 0.0
    assert abs(m["income"]["mean"] - 100.0) < 1e-9


def test_associations_perfect_and_independent():
    s = toy_schema()
    # vote perfectly determined by region -> Cramer's V ~ 1
    perfect = pd.DataFrame({"region": ["N", "N", "S", "S"] * 10,
                            "vote": ["A", "A", "B", "B"] * 10})
    a = associations(perfect, ["region", "vote"], s)
    assert a["region"]["vote"] > 0.9
    assert a["vote"]["region"] == a["region"]["vote"]  # symmetric
    # region independent of vote -> ~0
    indep = pd.DataFrame({"region": (["N", "S"] * 20),
                          "vote": (["A", "B"] * 20)})
    indep["vote"] = (["A", "A", "B", "B"] * 10)  # uncorrelated with alternating region
    a2 = associations(indep, ["region", "vote"], s)
    assert a2.get("region", {}).get("vote", 0.0) < 0.4


def test_associations_mixed_and_numnum_and_degenerate():
    s = toy_schema()
    df = pd.DataFrame({"region": ["N", "S"] * 20,
                       "income": list(np.linspace(0, 200, 40)),
                       "age": list(np.linspace(18, 90, 40))})
    a = associations(df, ["region", "income", "age"], s)
    assert 0.0 <= a["region"]["income"] <= 1.0      # cat x num -> eta
    assert a["income"]["age"] > 0.9                 # num x num -> |r| (both linear)
    # degenerate: constant column produces no entry, no raise
    dgn = pd.DataFrame({"vote": ["A"] * 10, "income": [5.0] * 10})
    a3 = associations(dgn, ["vote", "income"], s)
    assert a3 == {}
