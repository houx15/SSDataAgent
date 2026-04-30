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


def test_format_adds_missing_schema_columns_as_nan():
    """Unseen-variable runs may legitimately omit a target column. The eval
    must still be runnable; format_generated fills missing schema vars with NaN
    so the bootstrap test scores them at zero rather than crashing on KeyError."""
    df = pd.DataFrame({"gender": ["Male"] * 3, "age": [30] * 3})
    out = format_generated(df, dataset_name="gss")
    # gss target schema includes age_first_childbirth — should appear, all NaN
    assert "age_first_childbirth" in out.columns
    assert out["age_first_childbirth"].isna().all()


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
