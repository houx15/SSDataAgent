"""Phase-6 regression tests: unseen-variable behavior end-to-end."""
import json

import pandas as pd

from ssdataagent.agent.context import Condition, build_context
from ssdataagent.evaluation.runner import PassRates, split_by_seen_unseen
from ssdataagent.experiments.conditions import get_condition


def test_unseen_excluded_from_data(tmp_path):
    df = pd.DataFrame({"gender": ["M"], "age": [30], "income": [50000]})
    build_context(
        condition=Condition.UNSEEN, dataset_name="gss",
        train_df=df, workspace=tmp_path, unseen_variables=["income"],
    )
    staged = pd.read_csv(tmp_path / "train.csv")
    assert "income" not in staged.columns


def test_unseen_descriptions_present(tmp_path):
    df = pd.DataFrame({"gender": ["M"], "age": [30], "income": [50000]})
    build_context(
        condition=Condition.UNSEEN, dataset_name="gss",
        train_df=df, workspace=tmp_path, unseen_variables=["income"],
    )
    desc = json.loads((tmp_path / "descriptions.json").read_text())
    assert "income" in desc["descriptions"]


def test_unseen_condition_is_registered():
    spec = get_condition("full_agent_unseen")
    assert spec.context_condition is Condition.UNSEEN
    assert spec.strategy == "agent"


def test_seen_unseen_split_partitions_correctly():
    rates = PassRates(by_variable={
        "type1": {"gender": 0.8, "age_first_childbirth": 0.3},
    })
    seen, unseen = split_by_seen_unseen(rates, unseen_vars=["age_first_childbirth"])
    assert seen.by_variable["type1"] == {"gender": 0.8}
    assert unseen.by_variable["type1"] == {"age_first_childbirth": 0.3}
