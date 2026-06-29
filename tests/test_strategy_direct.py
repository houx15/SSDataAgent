import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.direct_strategy import DirectGenerationStrategy


def _fake_direct(*, client, sampled, dataset_name, transcript_out=None):
    if transcript_out is not None:
        transcript_out.append({"row": 0, "prompt": "P", "response": "R"})
    return pd.DataFrame({"profile_id": [0], "gender": ["Male"]})


def _gate():
    return InfoGate(
        condition=Condition.DIRECT, dataset_name="gss", workspace=Path("/tmp/ws"),
        client=object(), train=pd.DataFrame(),
        eval_rows=pd.DataFrame({"profile_id": [0], "age": [30]}),
    )


@patch("ssdataagent.strategies.direct_strategy.generate_direct", side_effect=_fake_direct)
def test_direct_strategy_writes_artifacts_and_returns_result(_d, tmp_path):
    result = DirectGenerationStrategy().generate(_gate(), tmp_path, cfg=None)
    assert result.generated.equals(pd.DataFrame({"profile_id": [0], "gender": ["Male"]}))
    assert result.meta_extras == {"n_individuals": 1}
    assert (tmp_path / "prompts.jsonl").read_text() == \
        json.dumps({"row": 0, "role": "user", "content": "P"}) + "\n"
    assert (tmp_path / "responses.jsonl").read_text() == \
        json.dumps({"row": 0, "role": "assistant", "content": "R"}) + "\n"


def test_direct_strategy_name():
    assert DirectGenerationStrategy().name == "direct"
