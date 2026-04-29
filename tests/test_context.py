import json

import pandas as pd

from ssdataagent.agent.context import (
    AgentContext,
    Condition,
    build_context,
)


def _df():
    return pd.DataFrame({"gender": ["Male", "Female"], "age": [30, 40], "income": [10, 20]})


def test_full_context_stages_data_and_descriptions(tmp_path):
    ctx: AgentContext = build_context(
        condition=Condition.FULL,
        dataset_name="gss",
        train_df=_df(),
        workspace=tmp_path,
    )
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "descriptions.json").exists()
    payload = json.loads((tmp_path / "descriptions.json").read_text())
    assert "descriptions" in payload
    assert payload["descriptions"]
    assert ctx.has_data is True
    assert ctx.has_descriptions is True


def test_no_semantic_context(tmp_path):
    ctx = build_context(
        condition=Condition.NO_SEMANTIC,
        dataset_name="gss",
        train_df=_df(),
        workspace=tmp_path,
    )
    assert (tmp_path / "train.csv").exists()
    assert not (tmp_path / "descriptions.json").exists()
    assert ctx.has_data is True
    assert ctx.has_descriptions is False


def test_no_data_context(tmp_path):
    ctx = build_context(
        condition=Condition.NO_DATA,
        dataset_name="gss",
        train_df=_df(),
        workspace=tmp_path,
    )
    assert not (tmp_path / "train.csv").exists()
    assert (tmp_path / "descriptions.json").exists()
    assert ctx.has_data is False
    assert ctx.has_descriptions is True


def test_unseen_context_hides_columns(tmp_path):
    df = _df()
    ctx = build_context(
        condition=Condition.UNSEEN,
        dataset_name="gss",
        train_df=df,
        workspace=tmp_path,
        unseen_variables=["income"],
    )
    staged = pd.read_csv(tmp_path / "train.csv")
    assert "income" not in staged.columns
    assert "gender" in staged.columns
    desc = json.loads((tmp_path / "descriptions.json").read_text())
    assert "income" in desc["descriptions"]
    assert ctx.unseen_variables == ("income",)


def test_descriptions_payload_has_population_context(tmp_path):
    build_context(
        condition=Condition.FULL,
        dataset_name="gss",
        train_df=_df(),
        workspace=tmp_path,
    )
    payload = json.loads((tmp_path / "descriptions.json").read_text())
    assert payload.get("context"), "population context must be present"
    assert payload.get("background_variables")
    assert payload.get("target_variables")
