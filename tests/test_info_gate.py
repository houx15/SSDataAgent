from pathlib import Path

import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.strategies.base import InfoGate, StrategyResult


def _gate(condition):
    train = pd.DataFrame({"a": [1, 2]})
    eval_rows = pd.DataFrame({"a": [3]})
    return InfoGate(
        condition=condition, dataset_name="gss", workspace=Path("/tmp/ws"),
        client=object(), train=train, eval_rows=eval_rows,
    )


def test_background_returns_eval_rows():
    gate = _gate(Condition.FULL)
    assert gate.background().equals(pd.DataFrame({"a": [3]}))


def test_fit_microdata_returns_train_for_data_conditions():
    for cond in (Condition.FULL, Condition.NO_SEMANTIC, Condition.UNSEEN):
        assert _gate(cond).fit_microdata().equals(pd.DataFrame({"a": [1, 2]}))


def test_fit_microdata_none_when_data_hidden():
    for cond in (Condition.NO_DATA, Condition.DIRECT):
        assert _gate(cond).fit_microdata() is None


def test_strategy_result_holds_frame_and_extras():
    r = StrategyResult(generated=pd.DataFrame({"x": [1]}), meta_extras={"k": 1})
    assert list(r.generated.columns) == ["x"]
    assert r.meta_extras == {"k": 1}
