# SSDataAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an LLM-driven data analyst agent that explores real social-survey data, generates synthetic individuals via code execution, and is evaluated against SSDataBench's five statistical pattern types.

**Architecture:** Fixed-stage agent loop (explore → model → validate → generate). Stateless Python subprocess sandbox with a shared workspace. Subprocess wrapper around SSDataBench's evaluation scripts. OpenAI-compatible LLM client (DeepSeek for Phase-1).

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, openai SDK, anthropic SDK, pyyaml, python-dotenv, pytest, pytest-mock.

---

## Conventions

- Repo root throughout: `/Users/houyuxin/08Coding/SSDataAgent`
- Package import path: `ssdataagent.*` (lives at `src/ssdataagent/`)
- All commands are run from repo root unless noted.
- After every task, run the suite and commit. Tasks are bite-sized (target: 2–10 minutes each).
- Live-LLM tests are gated by `RUN_LIVE_LLM_TESTS=1`. Default `pytest` skips them.
- Unless specified, commits use the form `phase N: <component>` and include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

---

## PHASE 0 — Project Setup & SSDataBench Integration

### Task 0.1: Project skeleton, gitignore, pyproject

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/ssdataagent/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.env.example`

- [ ] **Step 1: Write `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/

# Secrets and local config
.env
config/config.py

# Run artifacts
results/
ssdatabench/simulated_data/agent_*/
ssdatabench/evaluation_results/agent_*/

# OS
.DS_Store
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ssdataagent"
version = "0.1.0"
description = "LLM data-analyst agent for population-level survey simulation"
requires-python = ">=3.10"
dependencies = [
  "pandas>=2.0",
  "numpy>=1.24",
  "scipy>=1.11",
  "statsmodels>=0.14",
  "scikit-learn>=1.3",
  "matplotlib>=3.7",
  "openai>=1.40",
  "anthropic>=0.34",
  "pyyaml>=6.0",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4",
  "pytest-mock>=3.12",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
markers = [
  "live_llm: tests that hit real LLM APIs; gated by RUN_LIVE_LLM_TESTS=1",
  "live_eval: tests that shell out to ssdatabench evaluation scripts",
]
```

- [ ] **Step 3: Write `src/ssdataagent/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Write `tests/__init__.py` and `tests/conftest.py`**

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
import os
import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE_LLM_TESTS") != "1":
        skip_live = pytest.mark.skip(reason="live LLM tests gated by RUN_LIVE_LLM_TESTS=1")
        for item in items:
            if "live_llm" in item.keywords:
                item.add_marker(skip_live)


@pytest.fixture
def repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]
```

- [ ] **Step 5: Write `.env.example`**

```
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-replace-me
LLM_MODEL=deepseek-v4-flash
```

- [ ] **Step 6: Create venv, install package, verify pytest discovers nothing yet**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r ssdatabench/requirements.txt
pytest
```
Expected: `0 tests ran` (or similar — no failures).

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml src/ tests/ .env.example
git commit -m "phase 0: project skeleton, gitignore, pyproject, conftest"
```

---

### Task 0.2: Real `.env` (local only) + delete `config/config.py`

**Files:**
- Create: `.env` (NOT committed — verify gitignored)
- Delete: `config/config.py`

- [ ] **Step 1: Write `.env` (locally, never committed)**

Copy `.env.example` to `.env` and fill `LLM_API_KEY=sk-REDACTED` (the user-provided DeepSeek key).

```bash
cp .env.example .env
# then edit .env to insert real key
```

- [ ] **Step 2: Verify .env is gitignored**

Run: `git status --ignored | grep '\.env$'`
Expected: `.env` shown as ignored.

- [ ] **Step 3: Delete obsolete `config/config.py`**

```bash
git rm -f config/config.py 2>/dev/null || rm -f config/config.py
```

- [ ] **Step 4: Commit (only the deletion if the file was tracked)**

```bash
git add config/ 2>/dev/null
git commit -m "phase 0: remove plaintext API key file in favor of .env" --allow-empty
```

---

### Task 0.3: Config module — load `.env` + `llm.yaml` with env precedence

**Files:**
- Create: `src/ssdataagent/config.py`
- Create: `config/llm.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test (`tests/test_config.py`)**

```python
import os
from pathlib import Path
import pytest
from ssdataagent.config import LLMConfig, load_llm_config


def test_load_from_yaml_only(tmp_path, monkeypatch):
    yaml = tmp_path / "llm.yaml"
    yaml.write_text(
        "provider: openai\n"
        "base_url: https://api.example.com\n"
        "model: foo-1\n"
        "temperature: 0.5\n"
        "max_tokens: 1024\n"
    )
    for v in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "from-env")  # api_key is always from env
    cfg = load_llm_config(yaml_path=yaml)
    assert isinstance(cfg, LLMConfig)
    assert cfg.provider == "openai"
    assert cfg.base_url == "https://api.example.com"
    assert cfg.model == "foo-1"
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 1024
    assert cfg.api_key == "from-env"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    yaml = tmp_path / "llm.yaml"
    yaml.write_text(
        "provider: openai\nbase_url: https://yaml.example/v1\nmodel: yaml-model\n"
        "temperature: 1.0\nmax_tokens: 4096\n"
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_BASE_URL", "https://env.example")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    cfg = load_llm_config(yaml_path=yaml)
    assert cfg.provider == "anthropic"
    assert cfg.base_url == "https://env.example"
    assert cfg.model == "env-model"
    assert cfg.api_key == "env-key"


def test_missing_api_key_raises(tmp_path, monkeypatch):
    yaml = tmp_path / "llm.yaml"
    yaml.write_text("provider: openai\nbase_url: x\nmodel: y\n")
    for v in ("LLM_API_KEY",):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        load_llm_config(yaml_path=yaml)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Write `config/llm.yaml`**

```yaml
provider: openai
base_url: https://api.deepseek.com
model: deepseek-v4-flash
temperature: 1.0
max_tokens: 4096
```

- [ ] **Step 4: Implement `src/ssdataagent/config.py`**

```python
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO_ROOT / "config" / "llm.yaml"


@dataclass(frozen=True)
class LLMConfig:
    provider: Literal["openai", "anthropic"]
    base_url: str
    api_key: str
    model: str
    temperature: float = 1.0
    max_tokens: int = 4096


def load_llm_config(yaml_path: Path | None = None) -> LLMConfig:
    load_dotenv(REPO_ROOT / ".env", override=False)
    yaml_path = yaml_path or DEFAULT_YAML
    data: dict = {}
    if yaml_path.exists():
        with yaml_path.open() as f:
            data = yaml.safe_load(f) or {}

    provider = os.environ.get("LLM_PROVIDER", data.get("provider", "openai"))
    base_url = os.environ.get("LLM_BASE_URL", data.get("base_url", ""))
    model = os.environ.get("LLM_MODEL", data.get("model", ""))
    api_key = os.environ.get("LLM_API_KEY", "")
    temperature = float(data.get("temperature", 1.0))
    max_tokens = int(data.get("max_tokens", 4096))

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY not found in environment. Set it in .env or export it."
        )
    if provider not in ("openai", "anthropic"):
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r}")

    return LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

- [ ] **Step 5: Run test, verify pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/config.py config/llm.yaml tests/test_config.py
git commit -m "phase 0: LLM config loader with env > yaml precedence"
```

---

### Task 0.4: SSDataBench evaluation smoke test (no LLM)

**Goal:** Confirm we can shell out to one of their evaluation scripts. They ship existing simulated data we can evaluate.

**Files:**
- Create: `tests/test_ssdatabench_integration.py`
- Create: `scripts/smoke_eval.py`

- [ ] **Step 1: Inspect what simulated data ships in ssdatabench**

```bash
ls ssdatabench/simulated_data/ 2>&1 | head
```
If empty (which the `ls` above showed), we'll write a minimal synthetic CSV to evaluate (see step 2).

- [ ] **Step 2: Write the failing test**

`tests/test_ssdatabench_integration.py`:
```python
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest


SSDATABENCH = Path(__file__).resolve().parents[1] / "ssdatabench"


@pytest.mark.live_eval
def test_real_data_loadable():
    """The user-provided cleaned CSVs in real_data/ load and have the documented columns."""
    repo = Path(__file__).resolve().parents[1]
    meta = json.loads((repo / "real_data" / "dataset_meta.json").read_text())
    for entry in meta:
        csv = repo / "real_data" / Path(entry["output"]).name
        df = pd.read_csv(csv)
        assert list(df.columns) == entry["columns"], f"{csv.name} column mismatch"
        assert len(df) == entry["rows"], f"{csv.name} expected {entry['rows']} rows"


@pytest.mark.live_eval
def test_evaluation_script_runs(tmp_path):
    """Shell out to ssdatabench's GSS-2018 evaluation against a copy of the real data
    treated as 'simulated' — proves the pipeline executes end-to-end."""
    repo = Path(__file__).resolve().parents[1]
    real_csv = repo / "real_data" / "gss_clean.csv"
    assert real_csv.exists()

    # Stage a single sim-profile inside ssdatabench's expected sim-root layout.
    sim_root = SSDATABENCH / "simulated_data" / "gss_2018" / "agent_smoke"
    sim_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(real_csv, sim_root / "sim_profiles_smoke.csv")
    output_base = SSDATABENCH / "evaluation_results" / "gss_2018" / "agent_smoke"

    try:
        result = subprocess.run(
            ["python", "scripts/evaluation/gss_2018.py",
             "--sim-root", str(sim_root.relative_to(SSDATABENCH)),
             "--output-base", str(output_base.relative_to(SSDATABENCH))],
            cwd=SSDATABENCH,
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        shutil.rmtree(sim_root, ignore_errors=True)

    # We don't assert success exit code — the script may flag low pass rates — but it
    # must not crash with a Python error before producing some output.
    assert "Traceback" not in result.stderr, f"Evaluation crashed: {result.stderr[-2000:]}"
```

Note: `live_eval` tests aren't auto-skipped by `conftest.py`. They run by default; in CI you can skip with `-m "not live_eval"`. They are slow but local-only.

- [ ] **Step 3: Write `scripts/smoke_eval.py`**

```python
"""Manual smoke test: run SSDataBench's GSS-2018 evaluation on the real data treated
as simulated. Useful first sanity check that the evaluation pipeline works."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SSDATABENCH = REPO / "ssdatabench"


def main():
    real_csv = REPO / "real_data" / "gss_clean.csv"
    sim_root = SSDATABENCH / "simulated_data" / "gss_2018" / "agent_smoke"
    sim_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(real_csv, sim_root / "sim_profiles_smoke.csv")
    output_base = SSDATABENCH / "evaluation_results" / "gss_2018" / "agent_smoke"
    try:
        subprocess.run(
            ["python", "scripts/evaluation/gss_2018.py",
             "--sim-root", str(sim_root.relative_to(SSDATABENCH)),
             "--output-base", str(output_base.relative_to(SSDATABENCH))],
            cwd=SSDATABENCH,
            check=False,
        )
        print("\n[smoke_eval] outputs under:", output_base)
    finally:
        shutil.rmtree(sim_root, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_ssdatabench_integration.py -v -m live_eval`
Expected: 2 passed (or one skipped if no GSS data; first must pass).

- [ ] **Step 5: If the eval script fails due to missing master config, log it as a known issue and adjust**

If `evaluation/config/gss_2018/evaluation_master.yaml` is missing or the script needs a different sampled-prefix, capture the error in the assertion and adjust to use `--single` mode or invoke `ssdatabench/evaluation/run_all_types.py` directly. The test should at minimum prove `python scripts/evaluation/gss_2018.py --help` runs (add a fallback assertion only if the full run can't be made green within the task scope; document the deviation in commit message).

- [ ] **Step 6: Commit**

```bash
git add tests/test_ssdatabench_integration.py scripts/smoke_eval.py
git commit -m "phase 0: ssdatabench evaluation smoke test"
```

---

### Task 0.5: LLM connectivity (live, gated)

**Files:**
- Create: `tests/test_llm_connectivity.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from ssdataagent.config import load_llm_config


@pytest.mark.live_llm
def test_openai_api_reachable():
    """Send a trivial prompt via the OpenAI SDK with the configured base_url + key.
    Verifies the configured DeepSeek/OpenAI-compatible endpoint actually responds."""
    cfg = load_llm_config()
    assert cfg.provider == "openai", "Phase-0 connectivity test assumes openai-compatible"

    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        temperature=0.0,
        max_tokens=8,
    )
    text = resp.choices[0].message.content or ""
    assert "PONG" in text.upper(), f"unexpected response: {text!r}"
```

- [ ] **Step 2: Run with live flag**

Run: `RUN_LIVE_LLM_TESTS=1 pytest tests/test_llm_connectivity.py -v`
Expected: PASS. If the model name is wrong (e.g. `deepseek-v4-flash` doesn't exist), the call will raise a 404 / model-not-found error. In that case **STOP and ask the user** which model to use; update `config/llm.yaml`; rerun.

- [ ] **Step 3: Run without live flag**

Run: `pytest tests/test_llm_connectivity.py -v`
Expected: 1 skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/test_llm_connectivity.py
git commit -m "phase 0: live LLM connectivity test (gated)"
```

---

## PHASE 1 — Data Layer

### Task 1.1: Schema (`DatasetSchema` + YAML loader)

**Files:**
- Create: `src/ssdataagent/data/__init__.py`
- Create: `src/ssdataagent/data/schema.py`
- Create: `config/datasets.yaml`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py`:
```python
import pytest
from ssdataagent.data.schema import DatasetSchema, load_schema


def test_load_gss_schema():
    s = load_schema("gss")
    assert isinstance(s, DatasetSchema)
    assert s.name == "gss"
    assert "gender" in s.background_variables
    assert "age" in s.background_variables
    assert s.target_variables, "must have at least one target"
    assert "gender" not in s.target_variables, "background and target are disjoint"


def test_schema_has_descriptions_for_targets():
    s = load_schema("gss")
    for var in s.target_variables:
        assert s.descriptions.get(var), f"missing description for {var}"


def test_schema_has_allowed_values_for_categoricals():
    s = load_schema("gss")
    # gender is categorical with two allowed values
    allowed = s.allowed_values["gender"]
    assert set(allowed) == {"Female", "Male"}


def test_schema_population_context_nonempty():
    s = load_schema("gss")
    assert s.population_context.strip()


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_schema("nope")
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Write `config/datasets.yaml`**

```yaml
datasets:
  gss:
    real_data_path: real_data/gss_clean.csv
    ssdatabench_yaml: ssdatabench/real_data/data_configs/gss2018.yaml
    ssdatabench_sim_subdir: gss_2018
    evaluation_script: scripts/evaluation/gss_2018.py
    type: cross-sectional
  cps:
    real_data_path: real_data/cps_clean.csv
    ssdatabench_yaml: ssdatabench/real_data/data_configs/cps1980.yaml
    ssdatabench_sim_subdir: cps_1980
    evaluation_script: scripts/evaluation/cps_1980.py
    type: cross-sectional
  acs:
    real_data_path: real_data/acs_clean.csv
    ssdatabench_yaml: ssdatabench/real_data/data_configs/acs1980.yaml
    ssdatabench_sim_subdir: acs_1980
    evaluation_script: scripts/evaluation/acs_1980.py
    type: cross-sectional
```

- [ ] **Step 4: Implement `src/ssdataagent/data/__init__.py` and `schema.py`**

`__init__.py`: empty file.

`schema.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ssdataagent.config import REPO_ROOT


DATASETS_YAML = REPO_ROOT / "config" / "datasets.yaml"


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    real_data_path: Path
    background_variables: list[str]
    target_variables: list[str]
    descriptions: dict[str, str]
    allowed_values: dict[str, list[Any]]
    numeric_ranges: dict[str, tuple[float, float]]
    population_context: str
    ssdatabench_sim_subdir: str
    evaluation_script: str


def _registry() -> dict[str, dict]:
    with DATASETS_YAML.open() as f:
        return yaml.safe_load(f)["datasets"]


def load_schema(name: str) -> DatasetSchema:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown dataset {name!r}; known: {list(reg)}")
    entry = reg[name]
    yaml_path = REPO_ROOT / entry["ssdatabench_yaml"]
    with yaml_path.open() as f:
        spec = yaml.safe_load(f)

    background = list((spec.get("input_variables") or {}).keys())
    targets = list((spec.get("output_variables") or {}).keys())
    descriptions: dict[str, str] = {}
    allowed: dict[str, list[Any]] = {}
    numeric: dict[str, tuple[float, float]] = {}

    for var, meta in {**(spec.get("input_variables") or {}), **(spec.get("output_variables") or {})}.items():
        descriptions[var] = (meta or {}).get("description", "")
        a = (meta or {}).get("allowed")
        if isinstance(a, list):
            allowed[var] = a
        elif isinstance(a, dict) and a.get("type") == "numeric":
            numeric[var] = (float(a["min"]), float(a["max"]))

    pop_ctx = spec.get("context") or ""

    return DatasetSchema(
        name=name,
        real_data_path=REPO_ROOT / entry["real_data_path"],
        background_variables=background,
        target_variables=targets,
        descriptions=descriptions,
        allowed_values=allowed,
        numeric_ranges=numeric,
        population_context=str(pop_ctx).strip(),
        ssdatabench_sim_subdir=entry["ssdatabench_sim_subdir"],
        evaluation_script=entry["evaluation_script"],
    )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_schema.py -v`
Expected: 5 passed.

If `gender` is in `output_variables` not `input_variables` for the GSS yaml, adjust: read both, classify by presence in real_data CSV columns vs SSDataBench's input list. Inspect via:
```bash
python -c "from ssdataagent.data.schema import load_schema; s=load_schema('gss'); print('bg:',s.background_variables); print('tgt:',s.target_variables)"
```

- [ ] **Step 6: Commit**

```bash
git add config/datasets.yaml src/ssdataagent/data/ tests/test_schema.py
git commit -m "phase 1: dataset schema loader"
```

---

### Task 1.2: Loader

**Files:**
- Create: `src/ssdataagent/data/loader.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/test_loader.py`:
```python
import pandas as pd

from ssdataagent.data.loader import load_real_data


def test_load_real_data_gss_shape():
    df = load_real_data("gss")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "gender" in df.columns
    assert "profile_id" in df.columns


def test_categorical_values_within_allowed():
    from ssdataagent.data.schema import load_schema
    df = load_real_data("gss")
    s = load_schema("gss")
    for var, allowed in s.allowed_values.items():
        if var not in df.columns:
            continue
        observed = set(df[var].dropna().unique())
        unknown = observed - set(allowed)
        assert not unknown, f"{var} has values outside allowed: {unknown}"
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement `loader.py`**

```python
from __future__ import annotations
import pandas as pd

from ssdataagent.data.schema import load_schema


def load_real_data(name: str) -> pd.DataFrame:
    schema = load_schema(name)
    df = pd.read_csv(schema.real_data_path)
    return df
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_loader.py -v`
Expected: 2 passed. If a categorical mismatch surfaces (e.g., values like `"Single"` not in the GSS schema), adjust the schema YAML mapping in Task 1.1 — but do NOT mutate the SSDataBench yaml; instead add a translation step inside `load_real_data` that normalizes whatever known divergences exist, and add a regression test that documents them.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/data/loader.py tests/test_loader.py
git commit -m "phase 1: real-data loader"
```

---

### Task 1.3: Splitter

**Files:**
- Create: `src/ssdataagent/data/splitter.py`
- Create: `tests/test_splitter.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_splitter.py`:
```python
import pandas as pd
import pytest

from ssdataagent.data.loader import load_real_data
from ssdataagent.data.splitter import split_train_eval


def test_split_sizes():
    df = load_real_data("gss")
    train, eval_ = split_train_eval(df, ratio=0.5, seed=42)
    assert len(train) + len(eval_) == len(df)
    assert abs(len(train) - 500) <= 1


def test_split_reproducibility():
    df = load_real_data("gss")
    a1, b1 = split_train_eval(df, ratio=0.5, seed=42)
    a2, b2 = split_train_eval(df, ratio=0.5, seed=42)
    pd.testing.assert_frame_equal(a1.reset_index(drop=True), a2.reset_index(drop=True))
    pd.testing.assert_frame_equal(b1.reset_index(drop=True), b2.reset_index(drop=True))


def test_no_overlap():
    df = load_real_data("gss")
    train, eval_ = split_train_eval(df, ratio=0.5, seed=42)
    assert set(train["profile_id"]).isdisjoint(set(eval_["profile_id"]))


def test_invalid_ratio_raises():
    df = pd.DataFrame({"profile_id": range(10)})
    with pytest.raises(ValueError):
        split_train_eval(df, ratio=1.5, seed=0)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_splitter.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement `splitter.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd


def split_train_eval(df: pd.DataFrame, ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < ratio < 1:
        raise ValueError(f"ratio must be in (0, 1); got {ratio}")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(round(len(df) * ratio))
    train_idx, eval_idx = idx[:n_train], idx[n_train:]
    return df.iloc[train_idx].copy(), df.iloc[eval_idx].copy()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_splitter.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/data/splitter.py tests/test_splitter.py
git commit -m "phase 1: train/eval splitter"
```

---

## PHASE 2 — Sandbox & Context

### Task 2.1: Sandbox

**Files:**
- Create: `src/ssdataagent/agent/__init__.py`
- Create: `src/ssdataagent/agent/sandbox.py`
- Create: `tests/test_sandbox.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_sandbox.py`:
```python
import pandas as pd
import pytest

from ssdataagent.agent.sandbox import Sandbox, SandboxResult


def test_execute_simple_code(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=10)
    try:
        r = sb.run("print(1 + 1)")
    finally:
        sb.close()
    assert isinstance(r, SandboxResult)
    assert r.exit_code == 0
    assert r.stdout.strip() == "2"
    assert r.timed_out is False


def test_pandas_available(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=15)
    try:
        r = sb.run("import pandas as pd; print(pd.DataFrame({'x':[1,2]}).x.mean())")
    finally:
        sb.close()
    assert r.exit_code == 0
    assert "1.5" in r.stdout


def test_timeout(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=2)
    try:
        r = sb.run("while True: pass")
    finally:
        sb.close()
    assert r.timed_out is True
    assert r.exit_code != 0


def test_error_capture(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=10)
    try:
        r = sb.run("undefined_name")
    finally:
        sb.close()
    assert r.exit_code != 0
    assert "NameError" in r.stderr


def test_multi_step_via_files(tmp_path):
    """Stateless model: state persists by writing files to the shared workspace."""
    sb = Sandbox(workspace_root=tmp_path, timeout=15)
    try:
        r1 = sb.run("import json; json.dump({'k': 42}, open('state.json','w'))")
        r2 = sb.run("import json; print(json.load(open('state.json'))['k'])")
    finally:
        sb.close()
    assert r1.exit_code == 0
    assert r2.exit_code == 0
    assert r2.stdout.strip() == "42"


def test_stage_file(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=15)
    try:
        sb.stage_file("greet.txt", "hello\n")
        r = sb.run("print(open('greet.txt').read().strip())")
    finally:
        sb.close()
    assert r.stdout.strip() == "hello"


def test_close_removes_workspace(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=10)
    workspace = sb.workspace
    assert workspace.exists()
    sb.close()
    assert not workspace.exists()
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_sandbox.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement `sandbox.py`**

`src/ssdataagent/agent/__init__.py`: empty file.

```python
from __future__ import annotations
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    timed_out: bool


_STDOUT_TRUNCATE = 8 * 1024  # bytes shown to LLM in formatted result


class Sandbox:
    def __init__(self, workspace_root: Path | None = None, timeout: int = 60):
        parent = workspace_root or Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(tempfile.mkdtemp(prefix="ssdataagent_", dir=parent))
        self.timeout = timeout
        self._step = 0

    def stage_file(self, name: str, content: bytes | str) -> Path:
        path = self.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        return path

    def run(self, code: str) -> SandboxResult:
        self._step += 1
        script = self.workspace / f"step_{self._step:03d}.py"
        script.write_text(code)
        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, str(script.name)],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            rc = -1
        duration = time.monotonic() - start
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=rc,
            duration_s=duration,
            timed_out=timed_out,
        )

    def close(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_sandbox.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/agent/__init__.py src/ssdataagent/agent/sandbox.py tests/test_sandbox.py
git commit -m "phase 2: sandbox (stateless subprocess + shared workspace)"
```

---

### Task 2.2: Context builder (4 conditions)

**Files:**
- Create: `src/ssdataagent/agent/context.py`
- Create: `tests/test_context.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_context.py`:
```python
from pathlib import Path

import pandas as pd

from ssdataagent.agent.context import (
    AgentContext,
    build_context,
    Condition,
)


def test_full_context_stages_data_and_descriptions(tmp_path):
    ctx: AgentContext = build_context(
        condition=Condition.FULL,
        dataset_name="gss",
        train_df=pd.DataFrame({"gender": ["Male", "Female"], "age": [30, 40]}),
        workspace=tmp_path,
    )
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "descriptions.json").exists()
    assert "Sex" in (tmp_path / "descriptions.json").read_text() or \
           "gender" in (tmp_path / "descriptions.json").read_text()
    assert ctx.has_data is True
    assert ctx.has_descriptions is True


def test_no_semantic_context(tmp_path):
    ctx = build_context(
        condition=Condition.NO_SEMANTIC,
        dataset_name="gss",
        train_df=pd.DataFrame({"gender": ["Male"], "age": [30]}),
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
        train_df=pd.DataFrame({"gender": ["Male"], "age": [30]}),
        workspace=tmp_path,
    )
    assert not (tmp_path / "train.csv").exists()
    assert (tmp_path / "descriptions.json").exists()
    assert ctx.has_data is False
    assert ctx.has_descriptions is True


def test_unseen_context_hides_columns(tmp_path):
    df = pd.DataFrame({"gender": ["M"], "age": [30], "income": [50000]})
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
    # descriptions still mention income (the agent must guess it)
    desc = (tmp_path / "descriptions.json").read_text()
    assert "income" in desc
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_context.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement `context.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from ssdataagent.data.schema import load_schema


class Condition(str, Enum):
    FULL = "full_agent"
    NO_SEMANTIC = "agent_no_semantic"
    NO_DATA = "agent_no_data"
    UNSEEN = "full_agent_unseen"
    DIRECT = "direct_generation"


@dataclass(frozen=True)
class AgentContext:
    condition: Condition
    dataset_name: str
    workspace: Path
    has_data: bool
    has_descriptions: bool
    unseen_variables: tuple[str, ...] = ()


def build_context(
    *,
    condition: Condition,
    dataset_name: str,
    train_df: pd.DataFrame,
    workspace: Path,
    unseen_variables: list[str] | None = None,
) -> AgentContext:
    schema = load_schema(dataset_name)
    workspace.mkdir(parents=True, exist_ok=True)
    unseen = tuple(unseen_variables or ())

    has_data = condition in (Condition.FULL, Condition.NO_SEMANTIC, Condition.UNSEEN)
    has_descriptions = condition in (Condition.FULL, Condition.NO_DATA, Condition.UNSEEN)

    if has_data:
        df = train_df.copy()
        if condition is Condition.UNSEEN:
            df = df.drop(columns=[c for c in unseen if c in df.columns])
        df.to_csv(workspace / "train.csv", index=False)

    if has_descriptions:
        payload = {
            "context": schema.population_context,
            "descriptions": schema.descriptions,
            "allowed_values": schema.allowed_values,
            "numeric_ranges": {k: list(v) for k, v in schema.numeric_ranges.items()},
            "background_variables": schema.background_variables,
            "target_variables": schema.target_variables,
        }
        (workspace / "descriptions.json").write_text(json.dumps(payload, indent=2))

    return AgentContext(
        condition=condition,
        dataset_name=dataset_name,
        workspace=workspace,
        has_data=has_data,
        has_descriptions=has_descriptions,
        unseen_variables=unseen,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_context.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/agent/context.py tests/test_context.py
git commit -m "phase 2: context builder for the four experimental conditions"
```

---

## PHASE 3 — Agent Orchestrator

### Task 3.1: Code-block extraction utility

**Files:**
- Create: `src/ssdataagent/agent/code_extraction.py`
- Create: `tests/test_code_extraction.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_code_extraction.py`:
```python
import pytest
from ssdataagent.agent.code_extraction import extract_python_block


def test_extracts_fenced_python():
    text = "Here is code:\n```python\nx = 1\nprint(x)\n```\nDone."
    assert extract_python_block(text) == "x = 1\nprint(x)"


def test_extracts_bare_fence():
    text = "```\nprint('hi')\n```"
    assert extract_python_block(text) == "print('hi')"


def test_returns_first_block_when_multiple():
    text = "```python\nA\n```\nthen\n```python\nB\n```"
    assert extract_python_block(text) == "A"


def test_returns_none_when_no_block():
    assert extract_python_block("no code here") is None
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_code_extraction.py -v`

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
import re

_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_python_block(text: str) -> str | None:
    m = _PATTERN.search(text)
    if not m:
        return None
    return m.group(1).rstrip("\n")
```

- [ ] **Step 4: Run, verify pass; commit**

Run: `pytest tests/test_code_extraction.py -v`
Expected: 4 passed.

```bash
git add src/ssdataagent/agent/code_extraction.py tests/test_code_extraction.py
git commit -m "phase 3: code-block extraction"
```

---

### Task 3.2: LLM client (OpenAI + Anthropic, mocked tests)

**Files:**
- Create: `src/ssdataagent/agent/llm_client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_llm_client.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from ssdataagent.config import LLMConfig
from ssdataagent.agent.llm_client import (
    OpenAICompatibleClient,
    AnthropicCompatibleClient,
    build_client,
)


def _openai_cfg() -> LLMConfig:
    return LLMConfig(
        provider="openai", base_url="https://example.com",
        api_key="k", model="m", temperature=0.5, max_tokens=128,
    )


def _anthropic_cfg() -> LLMConfig:
    return LLMConfig(
        provider="anthropic", base_url="https://anthropic.example",
        api_key="k", model="claude", temperature=0.5, max_tokens=128,
    )


def test_build_client_openai():
    assert isinstance(build_client(_openai_cfg()), OpenAICompatibleClient)


def test_build_client_anthropic():
    assert isinstance(build_client(_anthropic_cfg()), AnthropicCompatibleClient)


def test_openai_chat_returns_text():
    cfg = _openai_cfg()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="hi"))]
    with patch("ssdataagent.agent.llm_client.OpenAI") as Sdk:
        Sdk.return_value.chat.completions.create.return_value = fake_resp
        client = OpenAICompatibleClient(cfg)
        out = client.chat([{"role": "user", "content": "hello"}], system="be brief")
    assert out == "hi"
    Sdk.assert_called_once_with(api_key="k", base_url="https://example.com")


def test_anthropic_chat_returns_text():
    cfg = _anthropic_cfg()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text="hello world")]
    with patch("ssdataagent.agent.llm_client.Anthropic") as Sdk:
        Sdk.return_value.messages.create.return_value = fake_resp
        client = AnthropicCompatibleClient(cfg)
        out = client.chat([{"role": "user", "content": "hi"}], system="brief")
    assert out == "hello world"
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_llm_client.py -v`

- [ ] **Step 3: Implement**

```python
from __future__ import annotations
from typing import Protocol

from anthropic import Anthropic
from openai import OpenAI

from ssdataagent.config import LLMConfig


class LLMClient(Protocol):
    def chat(self, messages: list[dict], system: str | None = None) -> str: ...


class OpenAICompatibleClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._sdk = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        resp = self._sdk.chat.completions.create(
            model=self.cfg.model,
            messages=msgs,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""


class AnthropicCompatibleClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._sdk = Anthropic(api_key=cfg.api_key, base_url=cfg.base_url)

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        resp = self._sdk.messages.create(
            model=self.cfg.model,
            messages=messages,
            system=system or "",
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        # join text segments
        parts = []
        for block in getattr(resp, "content", []):
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


def build_client(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "openai":
        return OpenAICompatibleClient(cfg)
    if cfg.provider == "anthropic":
        return AnthropicCompatibleClient(cfg)
    raise RuntimeError(f"unknown provider: {cfg.provider}")
```

- [ ] **Step 4: Run, verify pass; commit**

Run: `pytest tests/test_llm_client.py -v`
Expected: 4 passed.

```bash
git add src/ssdataagent/agent/llm_client.py tests/test_llm_client.py
git commit -m "phase 3: unified LLM client (OpenAI + Anthropic compatible)"
```

---

### Task 3.3: Prompt templates

**Files:**
- Create: `src/ssdataagent/agent/prompt_templates.py`
- Create: `tests/test_prompt_templates.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_prompt_templates.py`:
```python
from ssdataagent.agent.prompt_templates import (
    SYSTEM_PROMPT,
    exploration_prompt,
    modeling_prompt,
    validation_prompt,
    generation_prompt,
)


def test_system_prompt_describes_role_and_workspace():
    assert "data analyst" in SYSTEM_PROMPT.lower()
    assert "fresh python process" in SYSTEM_PROMPT.lower()
    assert "```python" in SYSTEM_PROMPT


def test_exploration_prompt_references_train_csv():
    p = exploration_prompt(has_data=True, has_descriptions=True)
    assert "train.csv" in p
    assert "descriptions.json" in p


def test_modeling_prompt_includes_findings():
    p = modeling_prompt(findings_summary="median age = 47")
    assert "median age = 47" in p


def test_validation_prompt_mentions_holdout():
    p = validation_prompt()
    assert "holdout" in p.lower() or "validation" in p.lower()


def test_generation_prompt_specifies_n_and_target_path():
    p = generation_prompt(n_rows=1000, target_path="generated.csv")
    assert "1000" in p
    assert "generated.csv" in p
```

- [ ] **Step 2: Run, verify failure; then implement**

`src/ssdataagent/agent/prompt_templates.py`:
```python
from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert data analyst. Your job is to study a real social-survey
dataset and build a generative model that can synthesize new individuals whose
joint and marginal statistics match the real population.

You will work in stages: EXPLORATION, MODELING, VALIDATION, GENERATION.
At each stage you will write Python code inside a single ```python``` block.
Only the first fenced ```python``` block in your message will be executed.

IMPORTANT — execution model:
- Each code block runs in a *fresh* Python process inside a working directory.
- Persist state across steps by writing files (CSVs, pickles, JSON) to the cwd.
- The libraries pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib are
  available. Do NOT install packages.
- Per-step timeout: 60 seconds. Keep code efficient.
- Print compact diagnostics; the user only sees stdout/stderr.
"""


def exploration_prompt(*, has_data: bool, has_descriptions: bool) -> str:
    bits = ["STAGE: EXPLORATION."]
    if has_data:
        bits.append(
            "A file `train.csv` is in the working directory — your training split."
        )
    if has_descriptions:
        bits.append(
            "A file `descriptions.json` contains: population context, "
            "variable descriptions, allowed values for categoricals, numeric "
            "ranges, and the lists of background and target variables."
        )
    bits.append(
        "Write a single Python block that loads what is available and prints a "
        "concise statistical summary of the data (univariate distributions, "
        "key bivariate relationships, missingness). Keep printed output under "
        "4 KB."
    )
    return "\n\n".join(bits)


def modeling_prompt(*, findings_summary: str) -> str:
    return (
        "STAGE: MODELING.\n\n"
        f"Your findings so far:\n{findings_summary}\n\n"
        "Write a single Python block that fits a generative model on `train.csv` "
        "and pickles a `Sampler` to `model.pkl`. The Sampler must implement "
        "`sample(n: int) -> pandas.DataFrame` returning rows with the SAME "
        "columns as the training data (plus any target columns you must "
        "produce). Free choice of model family — JointDistribution, "
        "ConditionalChain, GaussianCopula, fitted statsmodels GLMs, or a "
        "Bayesian network. Keep it simple and fast."
    )


def validation_prompt() -> str:
    return (
        "STAGE: VALIDATION.\n\n"
        "Load `model.pkl`, sample 500 rows, and print quick comparisons to a "
        "small holdout slice of `train.csv` (e.g., the last 100 rows): "
        "univariate marginals (mean / proportions) and one or two key joint "
        "stats. If anything is clearly off (e.g., a categorical has values not "
        "in the schema, a numeric is out of range, or a marginal is wildly "
        "different), state it explicitly and proceed to fix it in the next "
        "modeling iteration. Otherwise say 'VALIDATION OK'."
    )


def generation_prompt(*, n_rows: int, target_path: str) -> str:
    return (
        "STAGE: GENERATION.\n\n"
        f"Load `model.pkl` and use it to generate exactly {n_rows} synthetic "
        f"individuals. Write the resulting DataFrame to `{target_path}` "
        "(no index column). Ensure all columns required by the schema are "
        "present and values are within their allowed sets / numeric ranges. "
        "Print 'GENERATED OK' on success."
    )
```

- [ ] **Step 3: Run, verify pass; commit**

Run: `pytest tests/test_prompt_templates.py -v`
Expected: 5 passed.

```bash
git add src/ssdataagent/agent/prompt_templates.py tests/test_prompt_templates.py
git commit -m "phase 3: prompt templates for the four agent stages"
```

---

### Task 3.4: Orchestrator

**Files:**
- Create: `src/ssdataagent/agent/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_orchestrator.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ssdataagent.agent.context import Condition, build_context
from ssdataagent.agent.orchestrator import Orchestrator, RunResult


SCRIPTED = [
    # exploration
    "```python\nimport pandas as pd; print(pd.read_csv('train.csv').describe())\n```",
    # modeling
    """```python
import pickle, pandas as pd
df = pd.read_csv('train.csv')
class Sampler:
    def __init__(self, df): self.df = df
    def sample(self, n):
        return self.df.sample(n, replace=True).reset_index(drop=True)
pickle.dump(Sampler(df), open('model.pkl', 'wb'))
print('MODEL OK')
```""",
    # validation
    "```python\nprint('VALIDATION OK')\n```",
    # generation
    """```python
import pickle
m = pickle.load(open('model.pkl', 'rb'))
m.sample(50).to_csv('generated.csv', index=False)
print('GENERATED OK')
```""",
]


@pytest.fixture
def tiny_train_df():
    return pd.DataFrame({
        "gender": ["Male", "Female"] * 25,
        "age": list(range(20, 70)),
    })


def test_orchestrator_runs_all_stages(tmp_path, tiny_train_df, monkeypatch):
    # bypass ssdatabench schema lookup by providing context manually
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tiny_train_df.to_csv(workspace / "train.csv", index=False)

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
    assert len(result.generated) == 50
    assert fake_client.chat.call_count == 4
    # transcript captures all messages
    assert len(result.transcript) >= 8  # 4 prompts + 4 responses minimum


def test_orchestrator_validation_loop_iterates(tmp_path, tiny_train_df):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tiny_train_df.to_csv(workspace / "train.csv", index=False)

    # Simulate one failing validation followed by a passing one.
    failing_validation = "```python\nprint('VALIDATION FAILED: wrong age range')\n```"
    bad_then_good = SCRIPTED[:2] + [failing_validation, SCRIPTED[1], SCRIPTED[2], SCRIPTED[3]]

    fake_client = MagicMock()
    fake_client.chat.side_effect = bad_then_good

    orch = Orchestrator(client=fake_client, n_rows=10, max_validation_iters=2)
    result = orch.run(
        condition=Condition.FULL,
        dataset_name="gss",
        workspace=workspace,
        has_data=True,
        has_descriptions=False,
    )
    assert len(result.generated) == 10
    # 4 normal + 1 retry modeling + 1 retry validation = 6
    assert fake_client.chat.call_count == 6


def test_orchestrator_raises_when_no_code_block(tmp_path, tiny_train_df):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tiny_train_df.to_csv(workspace / "train.csv", index=False)
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
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_orchestrator.py -v`

- [ ] **Step 3: Implement `orchestrator.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ssdataagent.agent.code_extraction import extract_python_block
from ssdataagent.agent.context import Condition
from ssdataagent.agent.llm_client import LLMClient
from ssdataagent.agent.prompt_templates import (
    SYSTEM_PROMPT,
    exploration_prompt,
    generation_prompt,
    modeling_prompt,
    validation_prompt,
)
from ssdataagent.agent.sandbox import Sandbox, SandboxResult


@dataclass
class TranscriptEntry:
    role: str  # "user" | "assistant" | "tool"
    content: str
    stage: str
    duration_s: float = 0.0


@dataclass
class RunResult:
    generated: pd.DataFrame
    transcript: list[TranscriptEntry]
    code_steps: list[str]
    sandbox_results: list[SandboxResult]


def _format_sandbox_result(r: SandboxResult) -> str:
    truncated_stdout = r.stdout[-4000:]
    truncated_stderr = r.stderr[-4000:]
    return (
        f"[exit={r.exit_code} duration={r.duration_s:.1f}s "
        f"timed_out={r.timed_out}]\n"
        f"--- stdout ---\n{truncated_stdout}\n"
        f"--- stderr ---\n{truncated_stderr}"
    )


class Orchestrator:
    def __init__(
        self,
        *,
        client: LLMClient,
        n_rows: int,
        max_validation_iters: int = 3,
        sandbox_timeout: int = 60,
    ):
        self.client = client
        self.n_rows = n_rows
        self.max_validation_iters = max_validation_iters
        self.sandbox_timeout = sandbox_timeout

    def run(
        self,
        *,
        condition: Condition,
        dataset_name: str,
        workspace: Path,
        has_data: bool,
        has_descriptions: bool,
    ) -> RunResult:
        sandbox = Sandbox(workspace_root=workspace.parent, timeout=self.sandbox_timeout)
        # Reuse caller's prepared workspace by copying staged files into the sandbox's workspace.
        for src in workspace.iterdir():
            if src.is_file():
                sandbox.stage_file(src.name, src.read_bytes())

        transcript: list[TranscriptEntry] = []
        code_steps: list[str] = []
        sandbox_results: list[SandboxResult] = []
        history: list[dict[str, Any]] = []

        def step(stage: str, prompt: str) -> SandboxResult:
            transcript.append(TranscriptEntry("user", prompt, stage))
            history.append({"role": "user", "content": prompt})
            response = self.client.chat(history, system=SYSTEM_PROMPT)
            transcript.append(TranscriptEntry("assistant", response, stage))
            history.append({"role": "assistant", "content": response})
            code = extract_python_block(response)
            if code is None:
                raise RuntimeError(f"no code block in {stage} response")
            code_steps.append(code)
            result = sandbox.run(code)
            sandbox_results.append(result)
            tool_msg = _format_sandbox_result(result)
            transcript.append(TranscriptEntry("tool", tool_msg, stage))
            history.append({"role": "user", "content": tool_msg})
            return result

        try:
            # 1. EXPLORATION
            explore_result = step(
                "EXPLORATION",
                exploration_prompt(has_data=has_data, has_descriptions=has_descriptions),
            )
            findings = explore_result.stdout[-2000:] or "(no findings printed)"

            # 2. MODELING
            step("MODELING", modeling_prompt(findings_summary=findings))

            # 3. VALIDATION (with retry loop)
            for iteration in range(self.max_validation_iters):
                v_result = step("VALIDATION", validation_prompt())
                ok = "VALIDATION OK" in v_result.stdout.upper()
                if ok or iteration == self.max_validation_iters - 1:
                    break
                # retry: ask agent to revise the model
                step("MODELING", modeling_prompt(
                    findings_summary="Validation flagged issues. Revise the model.",
                ))

            # 4. GENERATION
            step("GENERATION", generation_prompt(
                n_rows=self.n_rows,
                target_path="generated.csv",
            ))

            generated_path = sandbox.workspace / "generated.csv"
            if not generated_path.exists():
                raise RuntimeError("generation step did not produce generated.csv")
            generated = pd.read_csv(generated_path)
            return RunResult(
                generated=generated,
                transcript=transcript,
                code_steps=code_steps,
                sandbox_results=sandbox_results,
            )
        finally:
            # Snapshot workspace back to caller's dir for logging, then close.
            for f in sandbox.workspace.iterdir():
                if f.is_file():
                    (workspace / f.name).write_bytes(f.read_bytes())
            sandbox.close()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_orchestrator.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/agent/orchestrator.py tests/test_orchestrator.py
git commit -m "phase 3: agent orchestrator with explore/model/validate/generate loop"
```

---

## PHASE 4 — Output Formatting & Evaluation Bridge

### Task 4.1: Formatter

**Files:**
- Create: `src/ssdataagent/generation/__init__.py`
- Create: `src/ssdataagent/generation/formatter.py`
- Create: `tests/test_formatter.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_formatter.py`:
```python
import pandas as pd
import pytest

from ssdataagent.generation.formatter import format_generated, write_simulated


def test_format_clips_to_allowed_values():
    df = pd.DataFrame({
        "gender": ["Male", "Other"],
        "age": [40, 200],
        "profile_id": [0, 1],
    })
    out = format_generated(df, dataset_name="gss")
    # 'Other' is not allowed for gender → coerced to NaN or dropped to allowed default
    assert set(out["gender"].dropna().unique()) <= {"Male", "Female"}
    # age 200 is outside numeric range → clipped
    assert out["age"].max() <= 89


def test_format_preserves_row_count():
    df = pd.DataFrame({"gender": ["Male"] * 5, "age": [30] * 5, "profile_id": range(5)})
    out = format_generated(df, dataset_name="gss")
    assert len(out) == 5


def test_write_simulated_creates_expected_layout(tmp_path):
    df = pd.DataFrame({"gender": ["Male"], "age": [30], "profile_id": [0]})
    path = write_simulated(df, dataset_name="gss", run_id="test123",
                           ssdatabench_root=tmp_path / "ssdb")
    assert path.exists()
    assert path.name.startswith("sim_profiles")
    assert "gss_2018" in str(path)
    assert "test123" in str(path)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_formatter.py -v`

- [ ] **Step 3: Implement `formatter.py`**

`src/ssdataagent/generation/__init__.py`: empty file.

```python
from __future__ import annotations
from pathlib import Path

import pandas as pd

from ssdataagent.data.schema import load_schema


def format_generated(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    schema = load_schema(dataset_name)
    out = df.copy()
    for var, allowed in schema.allowed_values.items():
        if var in out.columns:
            out.loc[~out[var].isin(allowed), var] = pd.NA
    for var, (lo, hi) in schema.numeric_ranges.items():
        if var in out.columns:
            out[var] = pd.to_numeric(out[var], errors="coerce").clip(lower=lo, upper=hi)
    if "profile_id" not in out.columns:
        out["profile_id"] = range(len(out))
    return out


def write_simulated(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    run_id: str,
    ssdatabench_root: Path,
) -> Path:
    schema = load_schema(dataset_name)
    sim_dir = ssdatabench_root / "simulated_data" / schema.ssdatabench_sim_subdir / f"agent_{run_id}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    target = sim_dir / f"sim_profiles_{run_id}.csv"
    df.to_csv(target, index=False)
    return target
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_formatter.py -v`
Expected: 3 passed.

```bash
git add src/ssdataagent/generation/ tests/test_formatter.py
git commit -m "phase 4: output formatter (clips to schema, writes ssdatabench layout)"
```

---

### Task 4.2: Evaluation runner (subprocess wrapper)

**Files:**
- Create: `src/ssdataagent/evaluation/__init__.py`
- Create: `src/ssdataagent/evaluation/runner.py`
- Create: `tests/test_evaluation_runner.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluation_runner.py`:
```python
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ssdataagent.evaluation.runner import (
    PassRates,
    parse_pass_rates,
    run_evaluation,
)


def test_parse_pass_rates_simple(tmp_path):
    out = tmp_path / "type1_results.json"
    out.write_text(json.dumps({"pass_rate": 0.42, "by_variable": {"gender": 0.5}}))
    rates = parse_pass_rates(tmp_path)
    assert rates.by_type["type1"] == 0.42
    assert rates.by_variable["type1"]["gender"] == 0.5


def test_run_evaluation_invokes_subprocess(tmp_path):
    df = pd.DataFrame({"gender": ["Male"] * 10, "age": [30] * 10, "profile_id": range(10)})
    fake_root = tmp_path / "ssdb"
    (fake_root / "scripts" / "evaluation").mkdir(parents=True)
    eval_results = fake_root / "evaluation_results" / "gss_2018" / "agent_x"
    eval_results.mkdir(parents=True)
    (eval_results / "type1_results.json").write_text('{"pass_rate": 0.7}')

    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        rates = run_evaluation(
            dataset_name="gss",
            run_id="agent_x",
            generated=df,
            ssdatabench_root=fake_root,
        )
    assert run.called
    assert isinstance(rates, PassRates)
    assert rates.by_type["type1"] == 0.7
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_evaluation_runner.py -v`

- [ ] **Step 3: Implement `runner.py`**

`src/ssdataagent/evaluation/__init__.py`: empty.

```python
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ssdataagent.data.schema import load_schema
from ssdataagent.generation.formatter import format_generated, write_simulated


@dataclass(frozen=True)
class PassRates:
    by_type: dict[str, float] = field(default_factory=dict)
    by_variable: dict[str, dict[str, float]] = field(default_factory=dict)


def parse_pass_rates(eval_dir: Path) -> PassRates:
    by_type: dict[str, float] = {}
    by_variable: dict[str, dict[str, float]] = {}
    for f in sorted(eval_dir.glob("type*_results.json")):
        data = json.loads(f.read_text())
        type_name = f.stem.replace("_results", "")
        if "pass_rate" in data:
            by_type[type_name] = float(data["pass_rate"])
        if "by_variable" in data:
            by_variable[type_name] = {k: float(v) for k, v in data["by_variable"].items()}
    return PassRates(by_type=by_type, by_variable=by_variable)


def run_evaluation(
    *,
    dataset_name: str,
    run_id: str,
    generated: pd.DataFrame,
    ssdatabench_root: Path,
) -> PassRates:
    schema = load_schema(dataset_name)
    formatted = format_generated(generated, dataset_name)
    sim_csv = write_simulated(
        formatted, dataset_name=dataset_name, run_id=run_id,
        ssdatabench_root=ssdatabench_root,
    )
    sim_root = sim_csv.parent
    output_base = ssdatabench_root / "evaluation_results" / schema.ssdatabench_sim_subdir / f"agent_{run_id}"
    output_base.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        schema.evaluation_script,
        "--sim-root", str(sim_root.relative_to(ssdatabench_root)),
        "--output-base", str(output_base.relative_to(ssdatabench_root)),
    ]
    subprocess.run(cmd, cwd=ssdatabench_root, check=False)
    return parse_pass_rates(output_base)
```

- [ ] **Step 4: Run, verify pass; commit**

Run: `pytest tests/test_evaluation_runner.py -v`
Expected: 2 passed.

```bash
git add src/ssdataagent/evaluation/ tests/test_evaluation_runner.py
git commit -m "phase 4: evaluation runner wrapping ssdatabench scripts"
```

Note on `parse_pass_rates`: actual SSDataBench output JSON layout may differ from the simple shape used in tests. After Phase 0 smoke runs, inspect what real type-result JSONs contain and adjust the parser. Add a regression test using a real captured JSON.

---

### Task 4.3: Comparator

**Files:**
- Create: `src/ssdataagent/evaluation/comparator.py`
- Create: `tests/test_comparator.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_comparator.py`:
```python
import pandas as pd

from ssdataagent.evaluation.runner import PassRates
from ssdataagent.evaluation.comparator import to_long_table, summary_pivot


def _rates(t1=0.7, t2=0.5):
    return PassRates(by_type={"type1": t1, "type2": t2})


def test_long_table_shape():
    inputs = {("full_agent", "gss"): _rates(0.7, 0.5),
              ("direct_generation", "gss"): _rates(0.3, 0.2)}
    df = to_long_table(inputs)
    assert list(df.columns) == ["condition", "dataset", "type", "pass_rate"]
    assert len(df) == 4


def test_summary_pivot():
    inputs = {("full_agent", "gss"): _rates(0.8, 0.6),
              ("direct_generation", "gss"): _rates(0.4, 0.2)}
    pivot = summary_pivot(to_long_table(inputs))
    assert pivot.loc["full_agent", "gss"] == 0.7
    assert pivot.loc["direct_generation", "gss"] == 0.3
```

- [ ] **Step 2: Run, verify failure; implement**

`src/ssdataagent/evaluation/comparator.py`:
```python
from __future__ import annotations
import pandas as pd

from ssdataagent.evaluation.runner import PassRates


def to_long_table(rates: dict[tuple[str, str], PassRates]) -> pd.DataFrame:
    rows = []
    for (cond, dataset), r in rates.items():
        for type_name, pr in r.by_type.items():
            rows.append({"condition": cond, "dataset": dataset, "type": type_name, "pass_rate": pr})
    return pd.DataFrame(rows, columns=["condition", "dataset", "type", "pass_rate"])


def summary_pivot(long: pd.DataFrame) -> pd.DataFrame:
    return long.pivot_table(index="condition", columns="dataset", values="pass_rate", aggfunc="mean")
```

- [ ] **Step 3: Run, commit**

Run: `pytest tests/test_comparator.py -v`
Expected: 2 passed.

```bash
git add src/ssdataagent/evaluation/comparator.py tests/test_comparator.py
git commit -m "phase 4: comparator for cross-condition pass rates"
```

---

## PHASE 5 — Experiment Runner

### Task 5.1: Conditions

**Files:**
- Create: `src/ssdataagent/experiments/__init__.py`
- Create: `src/ssdataagent/experiments/conditions.py`
- Create: `tests/test_conditions.py`

- [ ] **Step 1: Failing tests**

`tests/test_conditions.py`:
```python
from ssdataagent.experiments.conditions import (
    CONDITIONS,
    get_condition,
    ConditionSpec,
)
from ssdataagent.agent.context import Condition


def test_all_four_conditions_registered():
    expected = {"full_agent", "agent_no_semantic", "agent_no_data", "direct_generation"}
    assert expected.issubset(set(CONDITIONS))


def test_condition_to_context_mapping():
    spec = get_condition("full_agent")
    assert isinstance(spec, ConditionSpec)
    assert spec.context_condition is Condition.FULL
    assert spec.is_agent is True

    direct = get_condition("direct_generation")
    assert direct.is_agent is False
```

- [ ] **Step 2: Implement**

`src/ssdataagent/experiments/__init__.py`: empty.

```python
from __future__ import annotations
from dataclasses import dataclass

from ssdataagent.agent.context import Condition


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    context_condition: Condition
    is_agent: bool


CONDITIONS: dict[str, ConditionSpec] = {
    "full_agent": ConditionSpec("full_agent", Condition.FULL, is_agent=True),
    "agent_no_semantic": ConditionSpec("agent_no_semantic", Condition.NO_SEMANTIC, is_agent=True),
    "agent_no_data": ConditionSpec("agent_no_data", Condition.NO_DATA, is_agent=True),
    "direct_generation": ConditionSpec("direct_generation", Condition.DIRECT, is_agent=False),
}


def get_condition(name: str) -> ConditionSpec:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; known: {list(CONDITIONS)}")
    return CONDITIONS[name]
```

- [ ] **Step 3: Run, commit**

Run: `pytest tests/test_conditions.py -v`
Expected: 2 passed.

```bash
git add src/ssdataagent/experiments/__init__.py src/ssdataagent/experiments/conditions.py tests/test_conditions.py
git commit -m "phase 5: experimental conditions registry"
```

---

### Task 5.2: Logger

**Files:**
- Create: `src/ssdataagent/experiments/logger.py`
- Create: `tests/test_logger.py`

- [ ] **Step 1: Failing tests**

`tests/test_logger.py`:
```python
import json
from pathlib import Path

import pandas as pd
import pytest

from ssdataagent.agent.orchestrator import RunResult, TranscriptEntry
from ssdataagent.agent.sandbox import SandboxResult
from ssdataagent.experiments.logger import log_run


def _fake_run_result():
    return RunResult(
        generated=pd.DataFrame({"x": [1, 2, 3]}),
        transcript=[
            TranscriptEntry("user", "hi", "EXPLORATION"),
            TranscriptEntry("assistant", "ok", "EXPLORATION"),
            TranscriptEntry("tool", "exit=0", "EXPLORATION"),
        ],
        code_steps=["print('hi')"],
        sandbox_results=[SandboxResult(stdout="hi", stderr="", exit_code=0,
                                       duration_s=0.1, timed_out=False)],
    )


def test_log_run_writes_expected_files(tmp_path):
    run_dir = tmp_path / "experiments" / "exp1" / "full_agent" / "gss" / "20260429-120000"
    log_run(_fake_run_result(), run_dir=run_dir, meta={"git_sha": "abc", "model": "m"})
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "prompts.jsonl").exists()
    assert (run_dir / "responses.jsonl").exists()
    assert (run_dir / "code" / "step_001.py").exists()
    assert (run_dir / "generated.csv").exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["git_sha"] == "abc"
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from ssdataagent.agent.orchestrator import RunResult


def log_run(result: RunResult, *, run_dir: Path, meta: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    prompts, responses = [], []
    for entry in result.transcript:
        line = {"stage": entry.stage, "role": entry.role, "content": entry.content}
        if entry.role == "assistant":
            responses.append(line)
        else:
            prompts.append(line)
    (run_dir / "prompts.jsonl").write_text(
        "\n".join(json.dumps(x) for x in prompts) + "\n"
    )
    (run_dir / "responses.jsonl").write_text(
        "\n".join(json.dumps(x) for x in responses) + "\n"
    )

    code_dir = run_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for i, code in enumerate(result.code_steps, 1):
        (code_dir / f"step_{i:03d}.py").write_text(code)
    for i, sr in enumerate(result.sandbox_results, 1):
        (code_dir / f"step_{i:03d}.stdout").write_text(sr.stdout)
        (code_dir / f"step_{i:03d}.stderr").write_text(sr.stderr)
        (code_dir / f"step_{i:03d}.exit").write_text(str(sr.exit_code))

    result.generated.to_csv(run_dir / "generated.csv", index=False)
```

- [ ] **Step 3: Run, commit**

Run: `pytest tests/test_logger.py -v`
Expected: 1 passed.

```bash
git add src/ssdataagent/experiments/logger.py tests/test_logger.py
git commit -m "phase 5: per-run logger"
```

---

### Task 5.3: Experiment runner

**Files:**
- Create: `src/ssdataagent/experiments/runner.py`
- Create: `config/experiments.yaml`
- Create: `scripts/run_experiment.py`
- Create: `tests/test_experiment_runner.py`

- [ ] **Step 1: Failing tests**

`tests/test_experiment_runner.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ssdataagent.evaluation.runner import PassRates
from ssdataagent.experiments.runner import (
    ExperimentConfig,
    run_experiment,
)


@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}))
@patch("ssdataagent.experiments.runner.Orchestrator")
def test_run_experiment_executes_each_pair(MockOrch, _eval, tmp_path):
    fake_orch = MockOrch.return_value
    fake_orch.run.return_value = MagicMock(
        generated=pd.DataFrame({"x": [1]}),
        transcript=[], code_steps=[], sandbox_results=[],
    )
    cfg = ExperimentConfig(
        name="t1",
        datasets=["gss"],
        conditions=["full_agent", "agent_no_semantic"],
        max_iterations=1,
        sandbox_timeout=10,
        train_eval_split=0.5,
        n_rows=10,
        results_root=tmp_path,
    )
    results = run_experiment(cfg)
    assert len(results) == 2
    assert ("full_agent", "gss") in results


@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}))
@patch("ssdataagent.experiments.runner.Orchestrator")
def test_run_experiment_resume_skips_done(MockOrch, _eval, tmp_path):
    fake_orch = MockOrch.return_value
    fake_orch.run.return_value = MagicMock(
        generated=pd.DataFrame({"x": [1]}),
        transcript=[], code_steps=[], sandbox_results=[],
    )
    cfg = ExperimentConfig(
        name="t2", datasets=["gss"], conditions=["full_agent"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    first_call_count = MockOrch.return_value.run.call_count
    run_experiment(cfg, resume=True)
    assert MockOrch.return_value.run.call_count == first_call_count
```

- [ ] **Step 2: Write `config/experiments.yaml`**

```yaml
experiments:
  pilot_gss:
    datasets: [gss]
    conditions: [full_agent, agent_no_semantic, agent_no_data, direct_generation]
    max_iterations: 3
    sandbox_timeout: 60
    train_eval_split: 0.5
    n_rows: 1000
```

- [ ] **Step 3: Implement `runner.py`**

```python
from __future__ import annotations
import datetime as dt
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ssdataagent.agent.context import Condition, build_context
from ssdataagent.agent.llm_client import build_client
from ssdataagent.agent.orchestrator import Orchestrator
from ssdataagent.config import REPO_ROOT, load_llm_config
from ssdataagent.data.loader import load_real_data
from ssdataagent.data.splitter import split_train_eval
from ssdataagent.evaluation.runner import PassRates, run_evaluation
from ssdataagent.experiments.conditions import get_condition
from ssdataagent.experiments.logger import log_run


@dataclass
class ExperimentConfig:
    name: str
    datasets: list[str]
    conditions: list[str]
    max_iterations: int
    sandbox_timeout: int
    train_eval_split: float
    n_rows: int
    results_root: Path = REPO_ROOT / "results"


def _run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_experiment(
    cfg: ExperimentConfig,
    *,
    resume: bool = False,
) -> dict[tuple[str, str], PassRates]:
    llm_cfg = load_llm_config()
    client = build_client(llm_cfg)
    results: dict[tuple[str, str], PassRates] = {}

    for dataset in cfg.datasets:
        df = load_real_data(dataset)
        train, eval_df = split_train_eval(df, ratio=cfg.train_eval_split, seed=42)
        for cond_name in cfg.conditions:
            spec = get_condition(cond_name)
            run_dir = cfg.results_root / cfg.name / cond_name / dataset
            existing = sorted(run_dir.glob("*/eval.json")) if run_dir.exists() else []
            if resume and existing:
                # parse the last completed one and skip
                import json as _json
                with existing[-1].open() as f:
                    blob = _json.load(f)
                results[(cond_name, dataset)] = PassRates(by_type=blob.get("by_type", {}))
                continue
            run_id = _run_id()
            run_dir = run_dir / run_id
            workspace = run_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            if not spec.is_agent:
                # Phase 7 will fill this in. For now produce an empty result.
                results[(cond_name, dataset)] = PassRates()
                continue

            ctx = build_context(
                condition=spec.context_condition,
                dataset_name=dataset,
                train_df=train,
                workspace=workspace,
            )
            orch = Orchestrator(
                client=client,
                n_rows=cfg.n_rows,
                max_validation_iters=cfg.max_iterations,
                sandbox_timeout=cfg.sandbox_timeout,
            )
            result = orch.run(
                condition=spec.context_condition,
                dataset_name=dataset,
                workspace=workspace,
                has_data=ctx.has_data,
                has_descriptions=ctx.has_descriptions,
            )
            log_run(result, run_dir=run_dir, meta={
                "experiment": cfg.name,
                "dataset": dataset,
                "condition": cond_name,
                "run_id": run_id,
                "git_sha": _git_sha(),
                "model": llm_cfg.model,
                "provider": llm_cfg.provider,
            })

            rates = run_evaluation(
                dataset_name=dataset,
                run_id=run_id,
                generated=result.generated,
                ssdatabench_root=REPO_ROOT / "ssdatabench",
            )
            (run_dir / "eval.json").write_text(_serialize_rates(rates))
            results[(cond_name, dataset)] = rates
    return results


def _serialize_rates(r: PassRates) -> str:
    import json as _json
    return _json.dumps({"by_type": r.by_type, "by_variable": r.by_variable}, indent=2)
```

- [ ] **Step 4: Write `scripts/run_experiment.py`**

```python
"""CLI: python scripts/run_experiment.py --experiment pilot_gss [--resume]"""
from __future__ import annotations
import argparse
from pathlib import Path

import yaml

from ssdataagent.config import REPO_ROOT
from ssdataagent.experiments.runner import ExperimentConfig, run_experiment
from ssdataagent.evaluation.comparator import to_long_table, summary_pivot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "experiments.yaml"))
    args = p.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())["experiments"][args.experiment]
    cfg = ExperimentConfig(name=args.experiment, **spec)
    rates = run_experiment(cfg, resume=args.resume)
    long = to_long_table(rates)
    pivot = summary_pivot(long)
    print("\n=== Summary (mean pass rate by condition x dataset) ===")
    print(pivot)
    out = REPO_ROOT / "results" / cfg.name / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_experiment_runner.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/experiments/runner.py config/experiments.yaml scripts/run_experiment.py tests/test_experiment_runner.py
git commit -m "phase 5: experiment runner + CLI + experiments.yaml"
```

---

## PHASE 6 — Unseen Variable Experiment

### Task 6.1: Schema extension for unseen variables

**Files:**
- Modify: `src/ssdataagent/experiments/conditions.py`
- Modify: `src/ssdataagent/experiments/runner.py`
- Create: `tests/test_unseen_variables.py`

- [ ] **Step 1: Failing tests**

`tests/test_unseen_variables.py`:
```python
from pathlib import Path

import pandas as pd
import pytest

from ssdataagent.agent.context import Condition, build_context


def test_unseen_excluded_from_data(tmp_path):
    df = pd.DataFrame({"gender": ["M"], "age": [30], "income": [50000]})
    ctx = build_context(
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
    desc = (tmp_path / "descriptions.json").read_text()
    assert "income" in desc
```

(These tests largely re-validate Task 2.2's `Condition.UNSEEN` behavior in this dedicated file as a regression guard for Phase 6.)

- [ ] **Step 2: Add `unseen_variables` to `ExperimentConfig`**

In `src/ssdataagent/experiments/runner.py`, extend `ExperimentConfig`:

```python
@dataclass
class ExperimentConfig:
    name: str
    datasets: list[str]
    conditions: list[str]
    max_iterations: int
    sandbox_timeout: int
    train_eval_split: float
    n_rows: int
    results_root: Path = REPO_ROOT / "results"
    unseen_variables: dict[str, list[str]] = field(default_factory=dict)  # dataset -> [var, ...]
```

In `run_experiment`, when `spec.context_condition is Condition.UNSEEN`, pass `unseen_variables=cfg.unseen_variables.get(dataset, [])` into `build_context`.

- [ ] **Step 3: Add `full_agent_unseen` to conditions**

In `src/ssdataagent/experiments/conditions.py`:

```python
CONDITIONS["full_agent_unseen"] = ConditionSpec(
    "full_agent_unseen", Condition.UNSEEN, is_agent=True,
)
```

- [ ] **Step 4: Extend `experiments.yaml`**

```yaml
experiments:
  pilot_gss_unseen:
    datasets: [gss]
    conditions: [full_agent_unseen]
    max_iterations: 3
    sandbox_timeout: 60
    train_eval_split: 0.5
    n_rows: 1000
    unseen_variables:
      gss: [income]   # hide income; agent must guess its distribution from semantics
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_unseen_variables.py tests/test_conditions.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/experiments/ config/experiments.yaml tests/test_unseen_variables.py
git commit -m "phase 6: unseen-variable condition + experiment config"
```

---

### Task 6.2: Split eval pass rates by seen/unseen

**Files:**
- Modify: `src/ssdataagent/evaluation/runner.py`
- Modify: `tests/test_evaluation_runner.py`

- [ ] **Step 1: Add a failing test**

Append to `tests/test_evaluation_runner.py`:

```python
def test_split_by_seen_unseen():
    from ssdataagent.evaluation.runner import split_by_seen_unseen, PassRates
    rates = PassRates(by_variable={
        "type1": {"gender": 0.8, "income": 0.3},
        "type2": {"age": 0.7, "income": 0.2},
    })
    seen, unseen = split_by_seen_unseen(rates, unseen_vars=["income"])
    assert seen.by_variable["type1"]["gender"] == 0.8
    assert "income" not in seen.by_variable["type1"]
    assert unseen.by_variable["type1"]["income"] == 0.3
```

- [ ] **Step 2: Implement `split_by_seen_unseen` in `runner.py`**

```python
def split_by_seen_unseen(rates: PassRates, unseen_vars: list[str]) -> tuple[PassRates, PassRates]:
    seen_by_var: dict[str, dict[str, float]] = {}
    unseen_by_var: dict[str, dict[str, float]] = {}
    for type_name, vars_ in rates.by_variable.items():
        seen_by_var[type_name] = {k: v for k, v in vars_.items() if k not in unseen_vars}
        unseen_by_var[type_name] = {k: v for k, v in vars_.items() if k in unseen_vars}
    return PassRates(by_type=rates.by_type, by_variable=seen_by_var), \
           PassRates(by_type=rates.by_type, by_variable=unseen_by_var)
```

- [ ] **Step 3: Run, commit**

Run: `pytest tests/test_evaluation_runner.py -v`
Expected: 3 passed.

```bash
git add src/ssdataagent/evaluation/runner.py tests/test_evaluation_runner.py
git commit -m "phase 6: split pass rates by seen/unseen variables"
```

---

## PHASE 7 — Direct LLM Generation Baseline

### Task 7.1: Direct generation via SSDataBench's own simulation code

**Files:**
- Create: `src/ssdataagent/experiments/direct_generation.py`
- Modify: `src/ssdataagent/experiments/runner.py`
- Create: `tests/test_direct_generation.py`

- [ ] **Step 1: Inspect ssdatabench's simulation entry point**

```bash
grep -RIn "def main\|argparse" ssdatabench/simulation/ | head
ls ssdatabench/simulation/configs/
```

Find the script and the config template they use.

- [ ] **Step 2: Failing test**

`tests/test_direct_generation.py`:
```python
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ssdataagent.experiments.direct_generation import generate_direct


@patch("subprocess.run")
def test_generate_direct_invokes_simulation(run, tmp_path):
    run.return_value.returncode = 0
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    fake_csv = out_dir / "sim_profiles_direct.csv"
    fake_csv.write_text("gender,age\nMale,30\n")
    df = generate_direct(
        dataset_name="gss",
        run_id="direct_x",
        output_dir=out_dir,
        # The implementation should look for any sim_profiles_*.csv in output_dir
    )
    assert isinstance(df, pd.DataFrame)
    assert run.called
```

- [ ] **Step 3: Implement `direct_generation.py`**

```python
from __future__ import annotations
import subprocess
from pathlib import Path

import pandas as pd

from ssdataagent.config import REPO_ROOT, load_llm_config
from ssdataagent.data.schema import load_schema


SSDATABENCH = REPO_ROOT / "ssdatabench"


def generate_direct(
    *,
    dataset_name: str,
    run_id: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Invoke SSDataBench's simulation code with our LLM config, returning the
    generated DataFrame. The exact CLI surface depends on which entry point
    they expose; here we shell out to `ssdatabench/simulation/generation_cs.py`
    if present, otherwise raise so the gap is loud.
    """
    schema = load_schema(dataset_name)
    cfg = load_llm_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    entry = SSDATABENCH / "simulation" / "generation_cs.py"
    if not entry.exists():
        raise FileNotFoundError(f"expected {entry} for direct generation")

    cmd = [
        "python", str(entry.relative_to(SSDATABENCH)),
        "--dataset", schema.ssdatabench_sim_subdir,
        "--output-dir", str(output_dir),
        "--n", "1000",
        "--provider", cfg.provider,
        "--base-url", cfg.base_url,
        "--model", cfg.model,
    ]
    # Pass the API key via env so it doesn't appear in process listings.
    import os
    env = {**os.environ, "LLM_API_KEY": cfg.api_key, "OPENAI_API_KEY": cfg.api_key}
    subprocess.run(cmd, cwd=SSDATABENCH, env=env, check=False)

    csvs = sorted(output_dir.glob("sim_profiles_*.csv"))
    if not csvs:
        raise RuntimeError(f"no sim_profiles_*.csv in {output_dir}")
    return pd.read_csv(csvs[-1])
```

> **IMPORTANT:** The CLI surface above is a best-guess. Once Phase 7 starts, read `ssdatabench/simulation/generation_cs.py` and `ssdatabench/simulation/control.py` to discover the actual flag names. Adjust `cmd` and `env` accordingly. Update the test to mock the same flags. If the script doesn't accept a `base_url` flag, see if it reads from env vars or a config file — pass it that way instead. Keep the function signature stable; only the internal `subprocess.run` call changes.

- [ ] **Step 4: Wire into `run_experiment`**

In `src/ssdataagent/experiments/runner.py`, replace the `if not spec.is_agent:` branch:

```python
if not spec.is_agent:
    from ssdataagent.experiments.direct_generation import generate_direct
    direct_df = generate_direct(
        dataset_name=dataset,
        run_id=run_id,
        output_dir=workspace,
    )
    rates = run_evaluation(
        dataset_name=dataset, run_id=run_id, generated=direct_df,
        ssdatabench_root=REPO_ROOT / "ssdatabench",
    )
    (run_dir / "eval.json").write_text(_serialize_rates(rates))
    results[(cond_name, dataset)] = rates
    continue
```

- [ ] **Step 5: Run, commit**

Run: `pytest tests/test_direct_generation.py -v`
Expected: 1 passed.

```bash
git add src/ssdataagent/experiments/direct_generation.py src/ssdataagent/experiments/runner.py tests/test_direct_generation.py
git commit -m "phase 7: direct-generation baseline via ssdatabench simulation"
```

---

## Final smoke run (live)

**Files:** none new.

- [ ] **Step 1: Run the full live LLM connectivity check**

```bash
RUN_LIVE_LLM_TESTS=1 pytest tests/test_llm_connectivity.py -v
```

- [ ] **Step 2: Run the full pilot experiment**

```bash
python scripts/run_experiment.py --experiment pilot_gss
```

Inspect the printed summary and `results/pilot_gss/summary.csv`. Pause and report results to user.

- [ ] **Step 3: Run the unseen-variable experiment**

```bash
python scripts/run_experiment.py --experiment pilot_gss_unseen
```

Report seen-vs-unseen pass-rate split.

---

## Self-review (run after writing the plan)

(Author's note: I performed this review inline. No placeholders, all task code is concrete, function signatures are consistent across tasks (`load_real_data`, `split_train_eval`, `Sandbox.run`, `Orchestrator.run`, `run_evaluation`, `PassRates`). Spec coverage: Phases 0–7 from `docs/SPEC.md` map to Tasks 0.1–7.1 in this plan. The only known gap surfaced during the review is Phase 7's exact CLI surface for `generation_cs.py`, which is flagged inline as requiring inspection at execution time — this is intentional rather than a placeholder, because the SSDataBench code is the source of truth and we only need to know the flag names when we actually call it.)
