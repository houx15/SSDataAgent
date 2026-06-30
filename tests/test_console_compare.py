# tests/test_console_compare.py
from ssdataagent.console import compare


def test_build_matrix_aligns_types_across_selectors():
    evals = {
        ("a", "full_agent", "gss"): {"by_type": {"type1": 0.5, "type2": 0.4},
                                     "overall_average": 0.45,
                                     "overdetermination": {"cell_based": {"headline_gap": 0.2}}},
        ("b", "design_a_full", "gss"): {"by_type": {"type1": 0.7},
                                        "overall_average": 0.7},
    }
    sels = [{"experiment": "a", "condition": "full_agent", "dataset": "gss"},
            {"experiment": "b", "condition": "design_a_full", "dataset": "gss"}]

    def loader(s):
        return evals.get((s["experiment"], s["condition"], s["dataset"]))

    out = compare.build_matrix(sels, loader)
    assert out["types"] == ["type1", "type2"]
    assert out["matrix"][0] == [0.5, 0.4]
    assert out["matrix"][1] == [0.7, None]      # missing type2 -> None
    assert out["cells"][0]["overdetermination_gap"] == 0.2
    assert out["cells"][1]["overdetermination_gap"] is None
