# Strategy Seam (P0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the experiment runner method-agnostic by introducing a `Strategy` seam + thin `InfoGate`, wrapping today's agent and direct-generation paths as two strategies with **byte-identical** artifacts and scoring output.

**Architecture:** Build new leaf modules in `src/ssdataagent/strategies/` (protocol, `StrategyResult`, `InfoGate`, two strategy wrappers, registry) in isolation first. Then a pure refactor moves the common artifact tail (`meta.json` + `generated.csv` + scoring output) out of `log_run`/inline-runner into one helper. Finally flip `_run_one_condition` to dispatch by `ConditionSpec.strategy` instead of branching on `is_agent`. Every task leaves the full test suite green.

**Tech Stack:** Python 3.11+, `pandas`, `pytest`, `unittest.mock`. No new dependencies.

## Global Constraints

- **No new dependencies.** Python 3.11+, standard repo libs only.
- **Behavior byte-identical.** Every artifact file (`meta.json`, `prompts.jsonl`, `responses.jsonl`, `generated.csv`, `code/`, scoring JSON) must be unchanged. The scorer (`evaluation/`, subprocess to SSDataBench) is frozen — not touched.
- **`config/*.yaml` and `experiments.yaml` untouched.** Runs still select by condition name; the strategy is derived from `ConditionSpec`.
- **Local tests:** run with `.venv/bin/python -m pytest`. Never use the cloud `.venv`; the cloud box uses conda `ssda` + tmux (only relevant for the final real smoke run).
- **Commit messages must NOT contain the literal word e-v-a-l** (project hook blocks it). Refer to the scoring output as "the scoring JSON" / "scores" in commit messages. (The on-disk filename stays `eval.json` in code — this rule is for commit text only.)
- **Out of scope (do not build):** any new strategy (baselines/Designs/S1), `known_marginals`/`known_associations`/`source_survey`/A-B-C semantics, the over-determination metric, the web console, dashboard changes, scorer changes.

---

## File structure

| File | Responsibility |
|---|---|
| `src/ssdataagent/strategies/__init__.py` | package marker |
| `src/ssdataagent/strategies/base.py` | `Strategy` Protocol, `StrategyResult`, `InfoGate` |
| `src/ssdataagent/strategies/direct_strategy.py` | `DirectGenerationStrategy` |
| `src/ssdataagent/strategies/agent_strategy.py` | `AgentStrategy` |
| `src/ssdataagent/strategies/registry.py` | `get_strategy(name)` + `STRATEGIES` |
| `src/ssdataagent/experiments/conditions.py` | `ConditionSpec.is_agent` → `.strategy` (modify) |
| `src/ssdataagent/experiments/logger.py` | `log_run` shrinks to prompts/responses/code (modify) |
| `src/ssdataagent/experiments/runner.py` | `_write_common` helper + seam dispatch (modify) |
| `tests/test_runner_artifacts.py` | characterization net (new) |
| `tests/test_info_gate.py` | InfoGate unit tests (new) |
| `tests/test_strategies_registry.py` | registry unit tests (new) |
| `tests/test_strategy_direct.py` | DirectGenerationStrategy unit tests (new) |
| `tests/test_strategy_agent.py` | AgentStrategy unit tests (new) |
| `tests/test_logger.py` | update for shrunk `log_run` (modify) |
| `tests/test_conditions.py` | `is_agent` → `strategy` (modify) |
| `tests/test_unseen_variables.py` | `is_agent` → `strategy` (modify) |

---

### Task 1: Characterization net for current runner behavior

**Why first:** This is a refactor; lock current artifact bytes with a test that passes on the **current** code, then keep it green through every later task. `_git_sha` is patched for determinism.

**Files:**
- Test: `tests/test_runner_artifacts.py` (create)

**Interfaces:**
- Consumes: `run_experiment`, `ExperimentConfig` from `ssdataagent.experiments.runner`; `PassRates` from `ssdataagent.evaluation.runner`; `RunResult`, `TranscriptEntry` from `ssdataagent.agent.orchestrator`; `SandboxResult` from `ssdataagent.agent.sandbox`.
- Produces: nothing consumed by later tasks; it is the safety net.

- [ ] **Step 1: Write the characterization test**

```python
# tests/test_runner_artifacts.py
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from ssdataagent.agent.orchestrator import RunResult, TranscriptEntry
from ssdataagent.agent.sandbox import SandboxResult
from ssdataagent.evaluation.runner import PassRates
from ssdataagent.experiments.runner import ExperimentConfig, run_experiment


def _agent_run_result():
    return RunResult(
        generated=pd.DataFrame({"profile_id": [0, 1], "gender": ["Male", "Female"]}),
        transcript=[
            TranscriptEntry("user", "hi", "EXPLORATION"),
            TranscriptEntry("assistant", "ok", "EXPLORATION"),
        ],
        code_steps=["print('hi')"],
        sandbox_results=[
            SandboxResult(stdout="out", stderr="err", exit_code=0,
                          duration_s=0.1, timed_out=False)
        ],
    )


def _read(run_dir, name):
    return (run_dir / name).read_text()


def _only_run_dir(cond_dir):
    runs = [p for p in cond_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1, runs
    return runs[0]


@patch("ssdataagent.experiments.runner._git_sha", return_value="testsha")
@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.experiments.runner.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_agent_artifacts_are_stable(_cfg, _client, MockOrch, _eval, _sha, tmp_path):
    _cfg.return_value = MagicMock(model="m1", provider="p1")
    MockOrch.return_value.run.return_value = _agent_run_result()
    cfg = ExperimentConfig(
        name="charexp", datasets=["gss"], conditions=["full_agent"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    run_dir = _only_run_dir(tmp_path / "charexp" / "full_agent" / "gss")

    meta = json.loads(_read(run_dir, "meta.json"))
    assert meta == {
        "experiment": "charexp", "dataset": "gss", "condition": "full_agent",
        "run_id": run_dir.name, "git_sha": "testsha", "model": "m1",
        "provider": "p1", "unseen_variables": [],
    }
    assert _read(run_dir, "generated.csv") == "profile_id,gender\n0,Male\n1,Female\n"
    assert _read(run_dir, "prompts.jsonl") == \
        json.dumps({"stage": "EXPLORATION", "role": "user", "content": "hi"}) + "\n"
    assert _read(run_dir, "responses.jsonl") == \
        json.dumps({"stage": "EXPLORATION", "role": "assistant", "content": "ok"}) + "\n"
    assert _read(run_dir, "code/step_001.py") == "print('hi')"
    assert _read(run_dir, "code/step_001.stdout") == "out"
    assert _read(run_dir, "code/step_001.exit") == "0"


def _fake_direct(*, client, sampled, dataset_name, transcript_out=None):
    if transcript_out is not None:
        transcript_out.append({"row": 0, "prompt": "P", "response": "R"})
    return pd.DataFrame({"profile_id": [0], "gender": ["Male"]})


@patch("ssdataagent.experiments.direct_generation.generate_direct", side_effect=_fake_direct)
@patch("ssdataagent.experiments.runner._git_sha", return_value="testsha")
@patch("ssdataagent.experiments.runner.run_evaluation",
       return_value=PassRates(by_type={"type1": 0.5}, overall_average=0.5))
@patch("ssdataagent.experiments.runner.Orchestrator")
@patch("ssdataagent.experiments.runner.build_client")
@patch("ssdataagent.experiments.runner.load_llm_config")
def test_direct_artifacts_are_stable(_cfg, _client, MockOrch, _eval, _sha, _direct, tmp_path):
    _cfg.return_value = MagicMock(model="m1", provider="p1")
    cfg = ExperimentConfig(
        name="charexp", datasets=["gss"], conditions=["direct_generation"],
        max_iterations=1, sandbox_timeout=10, train_eval_split=0.5,
        n_rows=10, results_root=tmp_path,
    )
    run_experiment(cfg)
    run_dir = _only_run_dir(tmp_path / "charexp" / "direct_generation" / "gss")

    meta = json.loads(_read(run_dir, "meta.json"))
    assert meta["condition"] == "direct_generation"
    assert meta["git_sha"] == "testsha"
    assert "n_individuals" in meta
    assert _read(run_dir, "generated.csv") == "profile_id,gender\n0,Male\n"
    assert _read(run_dir, "prompts.jsonl") == \
        json.dumps({"row": 0, "role": "user", "content": "P"}) + "\n"
    assert _read(run_dir, "responses.jsonl") == \
        json.dumps({"row": 0, "role": "assistant", "content": "R"}) + "\n"
```

- [ ] **Step 2: Run to confirm it passes on current code (GREEN net)**

Run: `.venv/bin/python -m pytest tests/test_runner_artifacts.py -v`
Expected: **PASS** (2 passed). This documents current behavior. If it fails, the assertions don't match current behavior — fix the assertions to match what the code does today, do not change the code.

> Note: `meta["n_individuals"]` value depends on `len(eval_df)` after the 0.5 split on the real `gss` data; assert only the key's presence, not its exact value, to stay robust. The exact-byte assertions (`generated.csv`, `prompts/responses.jsonl`, agent `meta`) are the real net.

- [ ] **Step 3: Commit**

```bash
git add tests/test_runner_artifacts.py
git commit -m "test: characterization net pinning runner artifact bytes"
```

---

### Task 2: `base.py` — Strategy protocol, StrategyResult, InfoGate

**Files:**
- Create: `src/ssdataagent/strategies/__init__.py` (empty)
- Create: `src/ssdataagent/strategies/base.py`
- Test: `tests/test_info_gate.py` (create)

**Interfaces:**
- Consumes: `Condition` from `ssdataagent.agent.context`; `LLMClient` from `ssdataagent.agent.llm_client`.
- Produces:
  - `Strategy` Protocol with `name: str` and `generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult`.
  - `StrategyResult(generated: pd.DataFrame, meta_extras: dict)`.
  - `InfoGate(condition, dataset_name, workspace, client, train, eval_rows, unseen_variables=())` with `.background() -> pd.DataFrame` and `.fit_microdata() -> pd.DataFrame | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_info_gate.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_info_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: ssdataagent.strategies.base`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/strategies/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

from ssdataagent.agent.context import Condition

if TYPE_CHECKING:
    from ssdataagent.agent.llm_client import LLMClient
    from ssdataagent.experiments.runner import ExperimentConfig


@dataclass
class StrategyResult:
    generated: pd.DataFrame
    meta_extras: dict = field(default_factory=dict)


@dataclass
class InfoGate:
    condition: Condition
    dataset_name: str
    workspace: Path
    client: "LLMClient"
    train: pd.DataFrame
    eval_rows: pd.DataFrame
    unseen_variables: tuple[str, ...] = ()

    def background(self) -> pd.DataFrame:
        """Test/eval rows — always allowed."""
        return self.eval_rows

    def fit_microdata(self) -> pd.DataFrame | None:
        """Train split when the condition permits microdata; None otherwise.

        Mirrors agent.context.build_context's has_data gating exactly:
        FULL / NO_SEMANTIC / UNSEEN expose data; NO_DATA / DIRECT do not.
        """
        if self.condition in (Condition.FULL, Condition.NO_SEMANTIC, Condition.UNSEEN):
            return self.train
        return None


@runtime_checkable
class Strategy(Protocol):
    name: str

    def generate(self, gate: InfoGate, run_dir: Path, cfg: "ExperimentConfig") -> StrategyResult:
        """Fill all target vars for each background row, writing the
        strategy's own method-specific artifacts into run_dir. Returns the
        generated frame plus strategy-specific meta.json fields."""
        ...
```

```python
# src/ssdataagent/strategies/__init__.py
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_info_gate.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/__init__.py src/ssdataagent/strategies/base.py tests/test_info_gate.py
git commit -m "feat: strategies.base with Strategy protocol, StrategyResult, InfoGate"
```

---

### Task 3: Pure refactor — extract common artifact tail; shrink `log_run`

**Why:** Behavior-preserving prep. Move `meta.json` + `generated.csv` + scoring-JSON writing into one `_write_common` helper used by **both** existing runner branches, and drop those two writes from `log_run`. No seam yet; `is_agent` branching stays. Characterization net (Task 1) + existing tests must stay green.

**Files:**
- Modify: `src/ssdataagent/experiments/logger.py` (drop `meta.json` + `generated.csv` writes; drop `meta` param)
- Modify: `src/ssdataagent/experiments/runner.py` (add `_write_common`; both branches call it)
- Modify: `tests/test_logger.py` (drop moved assertions; update `log_run` call signature)

**Interfaces:**
- Consumes: `run_evaluation`, `PassRates` (already imported in runner); `_serialize_rates` (already in runner).
- Produces: `_write_common(*, run_dir: Path, meta: dict, generated: pd.DataFrame, dataset: str, run_id: str, eval_df: pd.DataFrame) -> PassRates`. New `log_run(result, *, run_dir)` signature (no `meta`).

- [ ] **Step 1: Update `tests/test_logger.py` to the new `log_run` contract (failing)**

Replace the body of `test_log_run_writes_expected_files` and `test_log_run_handles_empty_results`:

```python
def test_log_run_writes_expected_files(tmp_path):
    run_dir = tmp_path / "experiments" / "exp1" / "full_agent" / "gss" / "20260429-120000"
    log_run(_fake_run_result(), run_dir=run_dir)
    assert (run_dir / "prompts.jsonl").exists()
    assert (run_dir / "responses.jsonl").exists()
    assert (run_dir / "code" / "step_001.py").exists()
    assert (run_dir / "code" / "step_001.stdout").exists()
    # meta.json and generated.csv are now written by the runner, not log_run:
    assert not (run_dir / "meta.json").exists()
    assert not (run_dir / "generated.csv").exists()


def test_log_run_handles_empty_results(tmp_path):
    run_dir = tmp_path / "empty"
    empty = RunResult(generated=pd.DataFrame(), transcript=[], code_steps=[], sandbox_results=[])
    log_run(empty, run_dir=run_dir)
    assert run_dir.exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_logger.py -v`
Expected: FAIL — `log_run()` still requires `meta` (TypeError) / still writes `meta.json`.

- [ ] **Step 3: Shrink `log_run`**

Replace `src/ssdataagent/experiments/logger.py` entirely:

```python
from __future__ import annotations

import json
from pathlib import Path

from ssdataagent.agent.orchestrator import RunResult


def log_run(result: RunResult, *, run_dir: Path) -> None:
    """Write the agent's method-specific artifacts: prompts, responses, code.

    meta.json and generated.csv are written by the runner's common tail
    (_write_common), so every strategy emits them identically.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts: list[dict] = []
    responses: list[dict] = []
    for entry in result.transcript:
        line = {"stage": entry.stage, "role": entry.role, "content": entry.content}
        if entry.role == "assistant":
            responses.append(line)
        else:
            prompts.append(line)
    (run_dir / "prompts.jsonl").write_text(
        "\n".join(json.dumps(x) for x in prompts) + ("\n" if prompts else "")
    )
    (run_dir / "responses.jsonl").write_text(
        "\n".join(json.dumps(x) for x in responses) + ("\n" if responses else "")
    )

    code_dir = run_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for i, code in enumerate(result.code_steps, 1):
        (code_dir / f"step_{i:03d}.py").write_text(code)
    for i, sr in enumerate(result.sandbox_results, 1):
        (code_dir / f"step_{i:03d}.stdout").write_text(sr.stdout)
        (code_dir / f"step_{i:03d}.stderr").write_text(sr.stderr)
        (code_dir / f"step_{i:03d}.exit").write_text(str(sr.exit_code))
```

- [ ] **Step 4: Add `_write_common` and route both branches through it in `runner.py`**

Add this helper to `src/ssdataagent/experiments/runner.py` (near `_serialize_rates`):

```python
def _write_common(
    *,
    run_dir: Path,
    meta: dict,
    generated,
    dataset: str,
    run_id: str,
    eval_df,
) -> PassRates:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    generated.to_csv(run_dir / "generated.csv", index=False)
    rates = run_evaluation(
        dataset_name=dataset, run_id=run_id, generated=generated, sampled=eval_df,
    )
    (run_dir / "eval.json").write_text(_serialize_rates(rates, dataset))
    return rates
```

Rewrite `_run_one_condition` so each branch builds `generated` + `meta` + writes its own per-method artifacts, then calls `_write_common` (still branching on `spec.is_agent` — no seam yet):

```python
def _run_one_condition(
    *, spec, dataset, run_id, run_dir, workspace, train, eval_df, cfg, client, llm_cfg,
) -> PassRates:
    if not spec.is_agent:
        from ssdataagent.experiments.direct_generation import generate_direct
        transcript: list[dict] = []
        generated = generate_direct(
            client=client, sampled=eval_df, dataset_name=dataset,
            transcript_out=transcript,
        )
        prompts_lines = [
            json.dumps({"row": e["row"], "role": "user", "content": e["prompt"]})
            for e in transcript
        ]
        responses_lines = [
            json.dumps({"row": e["row"], "role": "assistant", "content": e["response"]})
            for e in transcript
        ]
        (run_dir / "prompts.jsonl").write_text(
            "\n".join(prompts_lines) + ("\n" if prompts_lines else "")
        )
        (run_dir / "responses.jsonl").write_text(
            "\n".join(responses_lines) + ("\n" if responses_lines else "")
        )
        meta = {
            "experiment": cfg.name, "dataset": dataset, "condition": spec.name,
            "run_id": run_id, "git_sha": _git_sha(), "model": llm_cfg.model,
            "provider": llm_cfg.provider, "n_individuals": len(eval_df),
        }
        return _write_common(
            run_dir=run_dir, meta=meta, generated=generated,
            dataset=dataset, run_id=run_id, eval_df=eval_df,
        )

    unseen = cfg.unseen_variables.get(dataset, [])
    ctx = build_context(
        condition=spec.context_condition, dataset_name=dataset, train_df=train,
        workspace=workspace,
        unseen_variables=unseen if spec.context_condition is Condition.UNSEEN else None,
    )
    orch = Orchestrator(
        client=client, n_rows=cfg.n_rows, max_validation_iters=cfg.max_iterations,
        sandbox_timeout=cfg.sandbox_timeout, prompt_variant=cfg.prompt_variant,
    )
    result = orch.run(
        condition=spec.context_condition, dataset_name=dataset, workspace=workspace,
        has_data=ctx.has_data, has_descriptions=ctx.has_descriptions,
    )
    log_run(result, run_dir=run_dir)
    meta = {
        "experiment": cfg.name, "dataset": dataset, "condition": spec.name,
        "run_id": run_id, "git_sha": _git_sha(), "model": llm_cfg.model,
        "provider": llm_cfg.provider, "unseen_variables": unseen,
    }
    return _write_common(
        run_dir=run_dir, meta=meta, generated=result.generated,
        dataset=dataset, run_id=run_id, eval_df=eval_df,
    )
```

- [ ] **Step 5: Run logger + characterization + runner tests**

Run: `.venv/bin/python -m pytest tests/test_logger.py tests/test_runner_artifacts.py tests/test_experiment_runner.py -v`
Expected: ALL PASS. The characterization net proves artifact bytes are unchanged by the extraction.

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/experiments/logger.py src/ssdataagent/experiments/runner.py tests/test_logger.py
git commit -m "refactor: extract _write_common tail; log_run writes only prompts/responses/code"
```

---

### Task 4: `DirectGenerationStrategy`

**Files:**
- Create: `src/ssdataagent/strategies/direct_strategy.py`
- Test: `tests/test_strategy_direct.py` (create)

**Interfaces:**
- Consumes: `InfoGate`, `StrategyResult` from `strategies.base`; `generate_direct` from `experiments.direct_generation`.
- Produces: `DirectGenerationStrategy` with `name = "direct"` and `generate(gate, run_dir, cfg) -> StrategyResult`. Writes `prompts.jsonl` + `responses.jsonl`; returns `meta_extras={"n_individuals": <int>}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_direct.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_strategy_direct.py -v`
Expected: FAIL with `ModuleNotFoundError: ssdataagent.strategies.direct_strategy`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/strategies/direct_strategy.py
from __future__ import annotations

import json
from pathlib import Path

from ssdataagent.experiments.direct_generation import generate_direct
from ssdataagent.strategies.base import InfoGate, StrategyResult


class DirectGenerationStrategy:
    name = "direct"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        transcript: list[dict] = []
        generated = generate_direct(
            client=gate.client,
            sampled=gate.background(),
            dataset_name=gate.dataset_name,
            transcript_out=transcript,
        )
        prompts_lines = [
            json.dumps({"row": e["row"], "role": "user", "content": e["prompt"]})
            for e in transcript
        ]
        responses_lines = [
            json.dumps({"row": e["row"], "role": "assistant", "content": e["response"]})
            for e in transcript
        ]
        (run_dir / "prompts.jsonl").write_text(
            "\n".join(prompts_lines) + ("\n" if prompts_lines else "")
        )
        (run_dir / "responses.jsonl").write_text(
            "\n".join(responses_lines) + ("\n" if responses_lines else "")
        )
        return StrategyResult(
            generated=generated,
            meta_extras={"n_individuals": len(gate.background())},
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_strategy_direct.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/direct_strategy.py tests/test_strategy_direct.py
git commit -m "feat: DirectGenerationStrategy wrapping generate_direct"
```

---

### Task 5: `AgentStrategy`

**Files:**
- Create: `src/ssdataagent/strategies/agent_strategy.py`
- Test: `tests/test_strategy_agent.py` (create)

**Interfaces:**
- Consumes: `InfoGate`, `StrategyResult` from `strategies.base`; `build_context`, `Condition` from `agent.context`; `Orchestrator` from `agent.orchestrator`; `log_run` from `experiments.logger`.
- Produces: `AgentStrategy` with `name = "agent"` and `generate(gate, run_dir, cfg) -> StrategyResult`. Calls `build_context` + `Orchestrator(...).run(...)` + `log_run`; returns `meta_extras={"unseen_variables": [...]}`.
- Note: `cfg` must expose `n_rows`, `max_iterations`, `sandbox_timeout`, `prompt_variant` (the `ExperimentConfig` fields used today).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_agent.py
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from ssdataagent.agent.context import Condition
from ssdataagent.agent.orchestrator import RunResult, TranscriptEntry
from ssdataagent.agent.sandbox import SandboxResult
from ssdataagent.strategies.agent_strategy import AgentStrategy
from ssdataagent.strategies.base import InfoGate


def _run_result():
    return RunResult(
        generated=pd.DataFrame({"profile_id": [0], "gender": ["Male"]}),
        transcript=[TranscriptEntry("user", "hi", "EXPLORATION")],
        code_steps=["print(1)"],
        sandbox_results=[SandboxResult(stdout="o", stderr="", exit_code=0,
                                       duration_s=0.0, timed_out=False)],
    )


def _cfg():
    return SimpleNamespace(n_rows=10, max_iterations=1, sandbox_timeout=5,
                           prompt_variant="baseline")


def _gate(workspace):
    return InfoGate(
        condition=Condition.FULL, dataset_name="gss", workspace=workspace,
        client=object(), train=pd.DataFrame({"profile_id": [1], "age": [40]}),
        eval_rows=pd.DataFrame({"profile_id": [0], "age": [30]}),
    )


@patch("ssdataagent.strategies.agent_strategy.log_run")
@patch("ssdataagent.strategies.agent_strategy.Orchestrator")
@patch("ssdataagent.strategies.agent_strategy.build_context")
def test_agent_strategy_runs_orchestrator_and_logs(MockBuild, MockOrch, mock_log, tmp_path):
    MockBuild.return_value = MagicMock(has_data=True, has_descriptions=True)
    MockOrch.return_value.run.return_value = _run_result()
    result = AgentStrategy().generate(_gate(tmp_path), tmp_path, _cfg())

    assert result.generated.equals(pd.DataFrame({"profile_id": [0], "gender": ["Male"]}))
    assert result.meta_extras == {"unseen_variables": []}
    MockOrch.return_value.run.assert_called_once()
    mock_log.assert_called_once()
    # log_run is called with the orchestrator result and the run_dir, no meta:
    _, kwargs = mock_log.call_args
    assert kwargs["run_dir"] == tmp_path


@patch("ssdataagent.strategies.agent_strategy.log_run")
@patch("ssdataagent.strategies.agent_strategy.Orchestrator")
@patch("ssdataagent.strategies.agent_strategy.build_context")
def test_agent_strategy_passes_unseen_only_for_unseen_condition(MockBuild, MockOrch, _l, tmp_path):
    MockBuild.return_value = MagicMock(has_data=True, has_descriptions=True)
    MockOrch.return_value.run.return_value = _run_result()
    gate = InfoGate(
        condition=Condition.UNSEEN, dataset_name="gss", workspace=tmp_path,
        client=object(), train=pd.DataFrame({"profile_id": [1]}),
        eval_rows=pd.DataFrame({"profile_id": [0]}), unseen_variables=("income",),
    )
    result = AgentStrategy().generate(gate, tmp_path, _cfg())
    assert result.meta_extras == {"unseen_variables": ["income"]}
    _, build_kwargs = MockBuild.call_args
    assert build_kwargs["unseen_variables"] == ["income"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_strategy_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: ssdataagent.strategies.agent_strategy`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/strategies/agent_strategy.py
from __future__ import annotations

from pathlib import Path

from ssdataagent.agent.context import Condition, build_context
from ssdataagent.agent.orchestrator import Orchestrator
from ssdataagent.experiments.logger import log_run
from ssdataagent.strategies.base import InfoGate, StrategyResult


class AgentStrategy:
    name = "agent"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        unseen = list(gate.unseen_variables)
        ctx = build_context(
            condition=gate.condition,
            dataset_name=gate.dataset_name,
            train_df=gate.fit_microdata(),
            workspace=gate.workspace,
            unseen_variables=unseen if gate.condition is Condition.UNSEEN else None,
        )
        orch = Orchestrator(
            client=gate.client,
            n_rows=cfg.n_rows,
            max_validation_iters=cfg.max_iterations,
            sandbox_timeout=cfg.sandbox_timeout,
            prompt_variant=cfg.prompt_variant,
        )
        result = orch.run(
            condition=gate.condition,
            dataset_name=gate.dataset_name,
            workspace=gate.workspace,
            has_data=ctx.has_data,
            has_descriptions=ctx.has_descriptions,
        )
        log_run(result, run_dir=run_dir)
        return StrategyResult(
            generated=result.generated,
            meta_extras={"unseen_variables": unseen},
        )
```

> Behavior note: `gate.fit_microdata()` returns `None` for `NO_DATA`/`DIRECT`, but `build_context` only reads `train_df` when `has_data` is True (FULL/NO_SEMANTIC/UNSEEN) — exactly when `fit_microdata()` returns the frame. So passing `None` for data-hidden conditions is provably behavior-identical to today's "pass the full train and ignore it."

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_strategy_agent.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/agent_strategy.py tests/test_strategy_agent.py
git commit -m "feat: AgentStrategy wrapping orchestrator path"
```

---

### Task 6: `registry`

**Files:**
- Create: `src/ssdataagent/strategies/registry.py`
- Test: `tests/test_strategies_registry.py` (create)

**Interfaces:**
- Consumes: `AgentStrategy`, `DirectGenerationStrategy`.
- Produces: `STRATEGIES: dict[str, type]`; `get_strategy(name: str) -> Strategy` (returns a fresh instance; raises `KeyError` on unknown name).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategies_registry.py
import pytest

from ssdataagent.strategies.agent_strategy import AgentStrategy
from ssdataagent.strategies.direct_strategy import DirectGenerationStrategy
from ssdataagent.strategies.registry import get_strategy


def test_get_strategy_returns_agent():
    assert isinstance(get_strategy("agent"), AgentStrategy)


def test_get_strategy_returns_direct():
    assert isinstance(get_strategy("direct"), DirectGenerationStrategy)


def test_get_strategy_unknown_raises():
    with pytest.raises(KeyError):
        get_strategy("nope")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_strategies_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: ssdataagent.strategies.registry`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ssdataagent/strategies/registry.py
from __future__ import annotations

from ssdataagent.strategies.agent_strategy import AgentStrategy
from ssdataagent.strategies.base import Strategy
from ssdataagent.strategies.direct_strategy import DirectGenerationStrategy

STRATEGIES: dict[str, type] = {
    "agent": AgentStrategy,
    "direct": DirectGenerationStrategy,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; known: {list(STRATEGIES)}")
    return STRATEGIES[name]()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_strategies_registry.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/strategies/registry.py tests/test_strategies_registry.py
git commit -m "feat: strategy registry with get_strategy"
```

---

### Task 7: Flip the seam — `ConditionSpec.strategy` + runner dispatch

**Why:** Replace `is_agent` with `strategy` and make `_run_one_condition` build an `InfoGate`, look up the strategy, call `generate`, then write the common tail. The characterization net (Task 1) is the gate: artifact bytes must not change.

**Files:**
- Modify: `src/ssdataagent/experiments/conditions.py`
- Modify: `src/ssdataagent/experiments/runner.py`
- Modify: `tests/test_conditions.py`
- Modify: `tests/test_unseen_variables.py`

**Interfaces:**
- Consumes: `get_strategy` from `strategies.registry`; `InfoGate` from `strategies.base`.
- Produces: `ConditionSpec(name, context_condition, strategy: str)`. `_run_one_condition` returns the same `PassRates` as before.

- [ ] **Step 1: Update condition tests to the new field (failing)**

In `tests/test_conditions.py`, replace the `is_agent` assertions:

```python
    assert spec.strategy == "agent"
```
and
```python
    assert get_condition("direct_generation").strategy == "direct"
```

In `tests/test_unseen_variables.py:34`, replace:
```python
    assert spec.strategy == "agent"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_conditions.py tests/test_unseen_variables.py -v`
Expected: FAIL — `ConditionSpec` has no attribute `strategy`.

- [ ] **Step 3: Change `ConditionSpec` field**

Replace `src/ssdataagent/experiments/conditions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from ssdataagent.agent.context import Condition


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    context_condition: Condition
    strategy: str


CONDITIONS: dict[str, ConditionSpec] = {
    "full_agent": ConditionSpec("full_agent", Condition.FULL, strategy="agent"),
    "agent_no_semantic": ConditionSpec(
        "agent_no_semantic", Condition.NO_SEMANTIC, strategy="agent"
    ),
    "agent_no_data": ConditionSpec("agent_no_data", Condition.NO_DATA, strategy="agent"),
    "full_agent_unseen": ConditionSpec(
        "full_agent_unseen", Condition.UNSEEN, strategy="agent"
    ),
    "direct_generation": ConditionSpec(
        "direct_generation", Condition.DIRECT, strategy="direct"
    ),
}


def get_condition(name: str) -> ConditionSpec:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; known: {list(CONDITIONS)}")
    return CONDITIONS[name]
```

- [ ] **Step 4: Rewrite `_run_one_condition` to dispatch via the seam**

In `src/ssdataagent/experiments/runner.py`, add imports at top:

```python
from ssdataagent.strategies.base import InfoGate
from ssdataagent.strategies.registry import get_strategy
```

Replace the entire `_run_one_condition` body with:

```python
def _run_one_condition(
    *, spec, dataset, run_id, run_dir, workspace, train, eval_df, cfg, client, llm_cfg,
) -> PassRates:
    gate = InfoGate(
        condition=spec.context_condition,
        dataset_name=dataset,
        workspace=workspace,
        client=client,
        train=train,
        eval_rows=eval_df,
        unseen_variables=tuple(cfg.unseen_variables.get(dataset, [])),
    )
    strategy = get_strategy(spec.strategy)
    result = strategy.generate(gate, run_dir, cfg)
    meta = {
        "experiment": cfg.name,
        "dataset": dataset,
        "condition": spec.name,
        "run_id": run_id,
        "git_sha": _git_sha(),
        "model": llm_cfg.model,
        "provider": llm_cfg.provider,
    }
    meta.update(result.meta_extras)
    return _write_common(
        run_dir=run_dir, meta=meta, generated=result.generated,
        dataset=dataset, run_id=run_id, eval_df=eval_df,
    )
```

Now remove imports that are no longer used by the runner *if and only if* nothing else references them: `build_context`, `Condition`, `Orchestrator`, `log_run` moved into the strategies. Check with `grep -n "build_context\|Orchestrator\|log_run\|Condition" src/ssdataagent/experiments/runner.py` and delete only the dead import lines (lines 9, 11, 18 region). Keep `run_evaluation`, `PassRates`, `by_domain`, everything else.

> Key-order note: `meta` is built in the exact historical order (experiment, dataset, condition, run_id, git_sha, model, provider) then `.update(meta_extras)` appends `unseen_variables` (agent) or `n_individuals` (direct) last — matching today's `meta.json` byte layout. The characterization net asserts this.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: ALL PASS. Critically `tests/test_runner_artifacts.py`, `tests/test_experiment_runner.py`, `tests/test_conditions.py`, `tests/test_unseen_variables.py` all green — proving the seam preserves behavior.

- [ ] **Step 6: Commit**

```bash
git add src/ssdataagent/experiments/conditions.py src/ssdataagent/experiments/runner.py tests/test_conditions.py tests/test_unseen_variables.py
git commit -m "refactor: runner dispatches by ConditionSpec.strategy via the Strategy seam"
```

---

### Task 8: Final verification + real smoke-run gate (cloud box)

**Why:** Local suite green is necessary but not sufficient — the spec's chosen gate is a real smoke run with `eval.json` matching a pre-seam baseline bit-for-bit. This task captures that baseline and diffs.

**Files:** none (verification + a results note)

- [ ] **Step 1: Full local suite green**

Run: `.venv/bin/python -m pytest -q`
Expected: entire suite passes. Record the count.

- [ ] **Step 2: Capture the PRE-seam baseline on the cloud box**

On the cloud box (conda `ssda` + tmux — never `.venv`, never `nohup`):

```bash
# from a clean checkout of the commit BEFORE this plan's first code commit
git stash --include-untracked   # if needed
git checkout 1514727            # the spec commit = last pre-seam commit (verify with: git log --oneline)
conda activate ssda
# run the smallest representative smoke config used historically (one agent
# condition + direct_generation, fixed seed). Use the project's smoke runner:
python scripts/run_experiment.py --config <smoke_config>   # confirm exact flag with scripts/run_experiment.py --help
# copy the produced run dirs' eval.json + generated.csv to results/baseline_p0/
```

- [ ] **Step 3: Run the SAME smoke config on the refactored HEAD**

```bash
git checkout main          # post-seam HEAD with Tasks 2-7 merged
conda activate ssda
python scripts/run_experiment.py --config <smoke_config>
```

- [ ] **Step 4: Diff the scoring JSON bit-for-bit**

```bash
diff results/baseline_p0/<agent_run>/eval.json   <new_agent_run>/eval.json
diff results/baseline_p0/<direct_run>/eval.json  <new_direct_run>/eval.json
diff results/baseline_p0/<agent_run>/generated.csv <new_agent_run>/generated.csv
```
Expected: **no differences** for both an agent condition and `direct_generation`. If `generated.csv` differs because the LLM is non-deterministic, re-run with the cached/seeded LLM path or compare structure + the scoring JSON only, and note the LLM-nondeterminism caveat. The scoring JSON is the authoritative gate.

- [ ] **Step 5: Record the gate result**

Append a one-line note to `docs/experiments/LEDGER.md` (or the appropriate retro) recording that P0 passed the regression gate, with the baseline commit and the smoke config used. Commit.

```bash
git add docs/experiments/LEDGER.md
git commit -m "docs: record P0 strategy-seam regression gate pass"
```

---

## Self-review

**Spec coverage:**
- §1 module layout → Tasks 2,4,5,6 (create files); Task 3,7 (modify). ✓
- §2 Strategy contract (strategy owns artifacts; runner owns common tail incl. meta.json/generated.csv) → Task 3 (extract `_write_common`, shrink `log_run`), Tasks 4/5 (strategies write own artifacts + return `meta_extras`), Task 7 (runner merges + writes common tail). ✓
- §3 thin InfoGate (background/fit_microdata only; A/B/C deferred) → Task 2. ✓
- §4 config selection (`is_agent`→`strategy`, experiments.yaml untouched) → Task 7. ✓
- §5 testing (registry, InfoGate, characterization bytes, existing tests updated) → Tasks 1,2,4,5,6,7; existing-test updates in Tasks 3 (test_logger) and 7 (test_conditions, test_unseen_variables). ✓
- §5 real smoke-run gate → Task 8. ✓
- Out-of-scope items → none built. ✓

**Placeholder scan:** `<smoke_config>` / `<agent_run>` in Task 8 are genuine runtime values that depend on the cloud box's available configs/run-ids; Step 2 instructs how to find them (`--help`, `git log`). All code steps contain complete code. No TODO/TBD in code tasks.

**Type consistency:** `InfoGate(condition, dataset_name, workspace, client, train, eval_rows, unseen_variables=())` — same field names used in Tasks 2,4,5,7. `StrategyResult(generated, meta_extras)` — consistent in Tasks 2,4,5,7. `generate(gate, run_dir, cfg)` — consistent across base/direct/agent/runner. `get_strategy(name)` returns instance — used in Task 7. `_write_common(*, run_dir, meta, generated, dataset, run_id, eval_df)` — defined Task 3, called Tasks 3 & 7 with identical kwargs. `log_run(result, *, run_dir)` — new signature defined Task 3, called Task 5. `meta_extras` keys: `unseen_variables` (agent), `n_individuals` (direct) — consistent Tasks 4,5,7 and the characterization net Task 1. ✓
