import pandas as pd

from ssdataagent.data.loader import load_real_data
from ssdataagent.data.schema import load_schema


def test_load_real_data_gss_shape():
    df = load_real_data("gss")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "gender" in df.columns
    assert "profile_id" in df.columns


def test_categorical_values_within_allowed():
    df = load_real_data("gss")
    s = load_schema("gss")
    for var, allowed in s.allowed_values.items():
        if var not in df.columns:
            continue
        observed = set(df[var].dropna().unique())
        unknown = observed - set(allowed)
        assert not unknown, f"{var} has values outside allowed: {unknown}"


def test_load_cps_and_acs():
    assert len(load_real_data("cps")) == 1000
    assert len(load_real_data("acs")) == 1000
