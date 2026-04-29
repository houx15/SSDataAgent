from unittest.mock import MagicMock

import pandas as pd

from ssdataagent.experiments.direct_generation import generate_direct


def test_generate_direct_returns_dataframe_with_all_targets():
    """Each row of the eval split is fed to the LLM; the LLM returns a JSON
    object with the target variables; the result is a DataFrame with columns
    matching the eval split + the LLM-produced targets."""
    sampled = pd.DataFrame({
        "gender": ["Male", "Female"],
        "age": [30, 40],
        "profile_id": [0, 1],
    })
    fake_client = MagicMock()
    fake_client.chat.side_effect = [
        '{"marital_status": "Married", "education": "College and above"}',
        '{"marital_status": "Single", "education": "High school"}',
    ]
    out = generate_direct(
        client=fake_client,
        sampled=sampled,
        dataset_name="gss",
    )
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2
    assert "marital_status" in out.columns
    assert "education" in out.columns
    assert "gender" in out.columns  # background passed through


def test_generate_direct_handles_invalid_json():
    """If the LLM returns garbage, that row's targets become NaN but the
    pipeline continues."""
    sampled = pd.DataFrame({"gender": ["Male"], "age": [30], "profile_id": [0]})
    fake_client = MagicMock()
    fake_client.chat.return_value = "not json"
    out = generate_direct(
        client=fake_client, sampled=sampled, dataset_name="gss",
    )
    assert len(out) == 1
    # at least one target column should be present (NaN-filled)
    assert any(out[c].isna().all() for c in out.columns if c != "gender" and c != "age")
