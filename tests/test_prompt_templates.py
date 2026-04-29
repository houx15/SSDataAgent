from ssdataagent.agent.prompt_templates import (
    SYSTEM_PROMPT,
    exploration_prompt,
    generation_prompt,
    modeling_prompt,
    validation_prompt,
)


def test_system_prompt_describes_role_and_workspace():
    assert "data analyst" in SYSTEM_PROMPT.lower()
    assert "fresh python process" in SYSTEM_PROMPT.lower()
    assert "```python" in SYSTEM_PROMPT


def test_exploration_prompt_references_train_csv_when_data_present():
    p = exploration_prompt(has_data=True, has_descriptions=True)
    assert "train.csv" in p
    assert "descriptions.json" in p


def test_exploration_prompt_no_data():
    p = exploration_prompt(has_data=False, has_descriptions=True)
    assert "train.csv" not in p
    assert "descriptions.json" in p


def test_modeling_prompt_includes_findings():
    p = modeling_prompt(findings_summary="median age = 47")
    assert "median age = 47" in p


def test_validation_prompt_mentions_validation():
    p = validation_prompt()
    assert "validation" in p.lower() or "validate" in p.lower()


def test_generation_prompt_specifies_n_and_target_path():
    p = generation_prompt(n_rows=1000, target_path="generated.csv")
    assert "1000" in p
    assert "generated.csv" in p
