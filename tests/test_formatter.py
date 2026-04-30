import pandas as pd

from ssdataagent.generation.formatter import format_generated, write_simulated


def test_format_clips_to_allowed_values():
    df = pd.DataFrame({
        "gender": ["Male", "Other"],
        "age": [40, 200],
        "profile_id": [0, 1],
    })
    out = format_generated(df, dataset_name="gss")
    # 'Other' is not allowed for gender → coerced to NaN
    assert set(out["gender"].dropna().unique()) <= {"Male", "Female"}
    # age 200 is outside numeric range → clipped
    assert out["age"].max() <= 89


def test_format_preserves_row_count():
    df = pd.DataFrame({"gender": ["Male"] * 5, "age": [30] * 5, "profile_id": range(5)})
    out = format_generated(df, dataset_name="gss")
    assert len(out) == 5


def test_format_adds_profile_id_when_missing():
    df = pd.DataFrame({"gender": ["Male"] * 3, "age": [30] * 3})
    out = format_generated(df, dataset_name="gss")
    assert "profile_id" in out.columns


def test_format_replaces_all_nan_profile_id():
    """If the agent emits profile_id but every value is NaN, replace with a
    fresh 0..N. SSDataBench's bootstrap does dropna() on [profile_id, var]
    so all-NaN profile_id wipes every row before the test runs."""
    import numpy as np
    df = pd.DataFrame({
        "gender": ["Male"] * 3, "age": [30] * 3,
        "profile_id": [np.nan, np.nan, np.nan],
    })
    out = format_generated(df, dataset_name="gss")
    assert out["profile_id"].notna().all()


def test_format_adds_missing_schema_columns_with_baseline_values():
    """Unseen-variable runs omit a target column. The eval bootstrap can't
    sample from an all-NaN series, so format_generated fills missing schema
    vars with random-uniform draws within the schema range/allowed values —
    a "no informed prediction" baseline that scores low but doesn't crash eval."""
    df = pd.DataFrame({"gender": ["Male"] * 100, "age": [30] * 100})
    out = format_generated(df, dataset_name="gss")
    # gss target schema includes age_first_childbirth (numeric)
    assert "age_first_childbirth" in out.columns
    assert not out["age_first_childbirth"].isna().any(), "should be filled, not NaN"
    # And marital_status (categorical with allowed values)
    assert "marital_status" in out.columns
    from ssdataagent.data.schema import load_schema
    allowed = set(load_schema("gss").allowed_values["marital_status"])
    assert set(out["marital_status"].unique()).issubset(allowed)


def test_write_simulated_creates_expected_layout(tmp_path):
    df = pd.DataFrame({"gender": ["Male"], "age": [30], "profile_id": [0]})
    path = write_simulated(
        df, dataset_name="gss", run_id="test123",
        ssdatabench_root=tmp_path / "ssdb",
    )
    assert path.exists()
    assert path.name.startswith("sim_profiles")
    assert "gss_2018" in str(path)
    assert "test123" in str(path)


def test_write_simulated_also_writes_sampled_pair(tmp_path):
    """SSDataBench batch_eval expects both sampled_*.csv and sim_*.csv in the
    same folder. The formatter writes the sim file; a paired sampled file is
    also written from the eval split if provided."""
    sim_df = pd.DataFrame({"gender": ["Male"], "age": [30], "profile_id": [0]})
    sampled_df = pd.DataFrame({"gender": ["Female"], "age": [25], "profile_id": [0]})
    sim_path = write_simulated(
        sim_df, dataset_name="gss", run_id="paired",
        ssdatabench_root=tmp_path / "ssdb",
        sampled_df=sampled_df,
    )
    sampled_path = sim_path.parent / "sampled_paired.csv"
    assert sampled_path.exists()
