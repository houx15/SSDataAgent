from unittest.mock import MagicMock

import pandas as pd
import pytest

from ssdataagent.agent.context import Condition
from ssdataagent.agent.orchestrator import Orchestrator, RunResult


SCRIPTED = [
    # exploration
    "```python\nimport pandas as pd; print(pd.read_csv('train.csv').describe())\n```",
    # modeling — stash sample seed; defer model to a re-readable training file
    """```python
import json
json.dump({'seed': 42}, open('model.json', 'w'))
print('MODEL OK')
```""",
    # validation
    "```python\nprint('VALIDATION OK')\n```",
    # generation — bootstrap from train.csv with stored seed
    """```python
import json, pandas as pd
cfg = json.load(open('model.json'))
df = pd.read_csv('train.csv')
df.sample(50, replace=True, random_state=cfg['seed']).reset_index(drop=True).to_csv('generated.csv', index=False)
print('GENERATED OK')
```""",
]


@pytest.fixture
def tiny_train_df():
    return pd.DataFrame({
        "gender": ["Male", "Female"] * 25,
        "age": list(range(20, 70)),
    })


def _setup_workspace(tmp_path, df):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    df.to_csv(workspace / "train.csv", index=False)
    return workspace


def test_orchestrator_runs_all_stages(tmp_path, tiny_train_df):
    workspace = _setup_workspace(tmp_path, tiny_train_df)
    fake_client = MagicMock()
    fake_client.chat.side_effect = SCRIPTED

    orch = Orchestrator(client=fake_client, n_rows=50, max_validation_iters=1)
    result: RunResult = orch.run(
        condition=Condition.FULL,
        dataset_name="gss",
        workspace=workspace,
        has_data=True,
        has_descriptions=False,
    )
    assert isinstance(result.generated, pd.DataFrame)
    # generation step writes 50 rows regardless of n_rows arg (test fixture is fixed)
    assert len(result.generated) == 50
    assert fake_client.chat.call_count == 4
    assert len(result.code_steps) == 4
    assert len(result.sandbox_results) == 4


def test_orchestrator_validation_loop_iterates(tmp_path, tiny_train_df):
    workspace = _setup_workspace(tmp_path, tiny_train_df)
    failing_validation = "```python\nprint('VALIDATION FAILED: wrong age range')\n```"
    bad_then_good = SCRIPTED[:2] + [failing_validation, SCRIPTED[1], SCRIPTED[2], SCRIPTED[3]]

    fake_client = MagicMock()
    fake_client.chat.side_effect = bad_then_good

    orch = Orchestrator(client=fake_client, n_rows=50, max_validation_iters=2)
    result = orch.run(
        condition=Condition.FULL,
        dataset_name="gss",
        workspace=workspace,
        has_data=True,
        has_descriptions=False,
    )
    assert len(result.generated) == 50
    # 1 explore + 1 model + (1 fail + 1 remodel + 1 ok) + 1 generate = 6
    assert fake_client.chat.call_count == 6


def test_orchestrator_raises_when_no_code_block(tmp_path, tiny_train_df):
    workspace = _setup_workspace(tmp_path, tiny_train_df)
    fake_client = MagicMock()
    fake_client.chat.return_value = "I don't have code for you."
    orch = Orchestrator(client=fake_client, n_rows=10, max_validation_iters=1)
    with pytest.raises(RuntimeError, match="no code block"):
        orch.run(
            condition=Condition.FULL,
            dataset_name="gss",
            workspace=workspace,
            has_data=True,
            has_descriptions=False,
        )


def test_orchestrator_raises_when_generation_missing(tmp_path, tiny_train_df):
    workspace = _setup_workspace(tmp_path, tiny_train_df)
    bad_generation = "```python\nprint('forgot to write csv')\n```"
    scripted = SCRIPTED[:3] + [bad_generation]
    fake_client = MagicMock()
    fake_client.chat.side_effect = scripted
    orch = Orchestrator(client=fake_client, n_rows=10, max_validation_iters=1)
    with pytest.raises(RuntimeError, match="generated.csv"):
        orch.run(
            condition=Condition.FULL,
            dataset_name="gss",
            workspace=workspace,
            has_data=True,
            has_descriptions=False,
        )


def test_orchestrator_persists_sandbox_output_on_failure(tmp_path, tiny_train_df):
    """When run() raises, step_NNN.stdout/stderr/exit must be on disk so
    failures are debuggable without re-running."""
    workspace = _setup_workspace(tmp_path, tiny_train_df)
    bad_generation = "```python\nprint('forgot to write csv')\n```"
    scripted = SCRIPTED[:3] + [bad_generation]
    fake_client = MagicMock()
    fake_client.chat.side_effect = scripted
    orch = Orchestrator(client=fake_client, n_rows=10, max_validation_iters=1)
    with pytest.raises(RuntimeError):
        orch.run(
            condition=Condition.FULL,
            dataset_name="gss",
            workspace=workspace,
            has_data=True,
            has_descriptions=False,
        )
    # 4 steps ran (explore, model, validate, bad_generation) before the raise
    for i in range(1, 5):
        assert (workspace / f"step_{i:03d}.stdout").exists(), f"missing stdout for step {i}"
        assert (workspace / f"step_{i:03d}.stderr").exists()
        assert (workspace / f"step_{i:03d}.exit").exists()
    # The forgot-to-write script should have stdout containing its print
    assert "forgot to write csv" in (workspace / "step_004.stdout").read_text()
