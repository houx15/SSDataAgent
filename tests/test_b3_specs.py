import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_cps_spec_is_single_source_of_truth():
    import nodonor_fullmethod as nf
    from ssdataagent.transfer import b3_specs
    assert nf.SPECS["cps"] is b3_specs.SPECS["cps"]  # moved, not duplicated


def test_cps_spec_fields_intact():
    from ssdataagent.transfer.b3_specs import SPECS
    cps = SPECS["cps"]
    assert cps.seeds == ["age", "gender", "race"]
    assert cps.predictors == ["age", "gender", "race", "education"]
    assert cps.numeric_predictors == frozenset({"age"})
    assert cps.log_vars == frozenset({"income"})
    assert cps.types == (1, 2, 3)
    assert "1980" in cps.population
    assert "child_number" in cps.glosses
