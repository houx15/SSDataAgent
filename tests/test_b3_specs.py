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


def test_gss_spec_invariants():
    from ssdataagent.transfer.b3_specs import SPECS
    gss = SPECS["gss"]
    assert gss.seeds == ["age", "gender", "race"]
    assert gss.predictors == ["age", "gender", "race", "education"]
    assert gss.numeric_predictors == frozenset({"age"})
    assert gss.log_vars == frozenset()          # income is categorical brackets in GSS
    assert gss.derived == {}
    assert gss.types == (1, 2, 3)
    assert "2018" in gss.population and "GSS" in gss.population
    # glosses scoped to exactly the numeric T3 outcomes surviving crosswalk restriction
    assert set(gss.glosses) == {"child_number", "age_first_childbirth", "vocabulary_test"}
    # lifetime-fertility rule present (opposite of the CPS household-roster trap)
    assert "EVER BORN" in gss.glosses["child_number"]
    assert "LIFETIME" in gss.rules or "lifetime" in gss.rules


def test_gss_seeds_and_predictors_are_crosswalk_columns():
    from ssdataagent.transfer.b3_specs import SPECS
    from ssdataagent.transfer.pairs import PAIRS, load_pair
    pair = [p for p in PAIRS if p.id == "gss_1994_2018"][0]
    _, _, cols = load_pair(pair)
    gss = SPECS["gss"]
    assert set(gss.seeds) <= set(cols)
    assert set(gss.predictors) <= set(cols)
    assert set(gss.glosses) <= set(cols)
