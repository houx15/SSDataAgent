from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _heartbeat(msg: str) -> None:
    """Emit a timestamped progress line so background runs aren't opaque."""
    print(f"[orch {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

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
    return (
        f"[exit={r.exit_code} duration={r.duration_s:.1f}s "
        f"timed_out={r.timed_out}]\n"
        f"--- stdout ---\n{r.stdout[-4000:]}\n"
        f"--- stderr ---\n{r.stderr[-4000:]}"
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
        # Copy caller's prepared workspace files into the sandbox's workspace.
        for src in workspace.iterdir():
            if src.is_file():
                sandbox.stage_file(src.name, src.read_bytes())

        transcript: list[TranscriptEntry] = []
        code_steps: list[str] = []
        sandbox_results: list[SandboxResult] = []
        history: list[dict[str, Any]] = []

        def step(stage: str, prompt: str) -> SandboxResult:
            _heartbeat(f"{stage}: requesting LLM (history={len(history)} msgs)")
            t0 = time.time()
            transcript.append(TranscriptEntry("user", prompt, stage))
            history.append({"role": "user", "content": prompt})
            response = self.client.chat(history, system=SYSTEM_PROMPT)
            _heartbeat(f"{stage}: LLM responded in {time.time()-t0:.1f}s ({len(response)} chars)")
            transcript.append(TranscriptEntry("assistant", response, stage))
            history.append({"role": "assistant", "content": response})
            code = extract_python_block(response)
            if code is None:
                _heartbeat(f"{stage}: no code block, retrying with nudge")
                nudge = (
                    "Your previous response had no fenced Python code block. "
                    "Respond again with executable Python in a single ```python ... ``` block."
                )
                transcript.append(TranscriptEntry("user", nudge, stage))
                history.append({"role": "user", "content": nudge})
                t1 = time.time()
                response = self.client.chat(history, system=SYSTEM_PROMPT)
                _heartbeat(f"{stage}: nudge responded in {time.time()-t1:.1f}s")
                transcript.append(TranscriptEntry("assistant", response, stage))
                history.append({"role": "assistant", "content": response})
                code = extract_python_block(response)
                if code is None:
                    raise RuntimeError(f"no code block in {stage} response (after retry)")
            code_steps.append(code)
            _heartbeat(f"{stage}: executing sandbox ({len(code)} chars of code)")
            t2 = time.time()
            result = sandbox.run(code)
            _heartbeat(
                f"{stage}: sandbox exit={result.exit_code} timed_out={result.timed_out} "
                f"in {time.time()-t2:.1f}s"
            )
            sandbox_results.append(result)
            tool_msg = _format_sandbox_result(result)
            transcript.append(TranscriptEntry("tool", tool_msg, stage))
            history.append({"role": "user", "content": tool_msg})
            return result

        try:
            explore_result = step(
                "EXPLORATION",
                exploration_prompt(has_data=has_data, has_descriptions=has_descriptions),
            )
            findings = explore_result.stdout[-2000:] or "(no findings printed)"
            step("MODELING", modeling_prompt(findings_summary=findings))

            for iteration in range(self.max_validation_iters):
                v_result = step("VALIDATION", validation_prompt())
                ok = "VALIDATION OK" in v_result.stdout.upper()
                if ok or iteration == self.max_validation_iters - 1:
                    break
                step(
                    "MODELING",
                    modeling_prompt(
                        findings_summary="Validation flagged issues. Revise the model."
                    ),
                )

            step(
                "GENERATION",
                generation_prompt(n_rows=self.n_rows, target_path="generated.csv"),
            )

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
            # Snapshot workspace back into caller's dir for logging, then close.
            for f in sandbox.workspace.iterdir():
                if f.is_file():
                    (workspace / f.name).write_bytes(f.read_bytes())
            # Dump sandbox stdout/stderr alongside step_NNN.py so failures
            # stay debuggable even when run() raises before log_run is reached.
            for i, sr in enumerate(sandbox_results, 1):
                (workspace / f"step_{i:03d}.stdout").write_text(sr.stdout)
                (workspace / f"step_{i:03d}.stderr").write_text(sr.stderr)
                (workspace / f"step_{i:03d}.exit").write_text(str(sr.exit_code))
            sandbox.close()
