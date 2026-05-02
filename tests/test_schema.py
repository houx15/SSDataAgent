import pytest

from ssdataagent.data.schema import DatasetSchema, load_schema


def test_load_gss_schema():
    s = load_schema("gss")
    assert isinstance(s, DatasetSchema)
    assert s.name == "gss"
    assert s.background_variables, "must have background variables"
    assert s.target_variables, "must have at least one target"


def test_schema_has_descriptions_for_targets():
    s = load_schema("gss")
    for var in s.target_variables:
        assert s.descriptions.get(var), f"missing description for {var}"


def test_schema_has_allowed_values_for_categoricals():
    s = load_schema("gss")
    assert "gender" in s.allowed_values
    assert set(s.allowed_values["gender"]) == {"Female", "Male"}


def test_schema_has_numeric_ranges():
    s = load_schema("gss")
    assert "age" in s.numeric_ranges
    lo, hi = s.numeric_ranges["age"]
    assert lo == 18 and hi == 89


def test_schema_population_context_nonempty():
    s = load_schema("gss")
    assert s.population_context.strip()


def test_schema_real_data_path_exists():
    s = load_schema("gss")
    assert s.real_data_path.exists()


def test_schema_evaluation_script_path():
    s = load_schema("gss")
    assert s.evaluation_script.endswith("gss_2018.py")


def test_schema_domains_populated_for_gss():
    """GSS YAML tags every variable with a domain (Demography, SES, Health,
    Marriage, Attitude, Ability) — used for per-domain reporting rollup."""
    s = load_schema("gss")
    assert s.domains.get("age") == "Demography"
    assert s.domains.get("marital_status") == "Marriage"
    assert s.domains.get("income") == "SES"
    assert s.domains.get("health") == "Health"
    assert s.domains.get("political_view") == "Attitude"
    assert len(s.domains) >= 25  # all 30 GSS vars have a domain


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_schema("nope")


def test_load_cps_schema():
    s = load_schema("cps")
    assert s.name == "cps"


def test_load_acs_schema():
    s = load_schema("acs")
    assert s.name == "acs"
