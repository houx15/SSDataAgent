import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        skip_live = pytest.mark.skip(reason="live LLM tests gated by RUN_LIVE_LLM_TESTS=1")
        for item in items:
            if "live_llm" in item.keywords:
                item.add_marker(skip_live)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
