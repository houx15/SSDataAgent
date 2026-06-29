import numpy as np

from ssdataagent.strategies.design_b import rake


def test_rake_matches_known_marginal():
    # two cells, equal weight; known marginal [0.5, 0.5]; LLM gave skewed cells
    cell_vectors = {"c0": np.array([0.9, 0.1]), "c1": np.array([0.3, 0.7])}
    cell_weights = {"c0": 0.5, "c1": 0.5}
    known = np.array([0.5, 0.5])
    out = rake(cell_vectors, cell_weights, known)
    mix = 0.5 * out["c0"] + 0.5 * out["c1"]
    assert np.allclose(mix, known, atol=1e-4)
    # relative ordering within each cell preserved (c0 still favors index 0)
    assert out["c0"][0] > out["c0"][1]


def test_rake_each_cell_sums_to_one():
    out = rake({"c0": np.array([0.2, 0.8])}, {"c0": 1.0}, np.array([0.5, 0.5]))
    assert abs(out["c0"].sum() - 1.0) < 1e-9
