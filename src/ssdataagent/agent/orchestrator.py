from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def _heartbeat(msg: str) -> None:
    """Emit a timestamped progress line so background runs aren't opaque."""
    print(f"[orch {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

from ssdataagent.agent.code_extraction import extract_python_block
from ssdataagent.agent.context import Condition
from ssdataagent.agent.llm_client import LLMClient, ToolUsingResponse
from ssdataagent.agent.prompt_templates import PromptVariant, get_variant
from ssdataagent.agent.sandbox import Sandbox, SandboxResult
from ssdataagent.agent.tools import (
    all_tool_schemas,
    build_runtime_state,
    dispatch as dispatch_tool,
)


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
    code_steps: list[str] = field(default_factory=list)
    sandbox_results: list[SandboxResult] = field(default_factory=list)
    # Tool-using path additions — empty for legacy code-block runs.
    tool_call_log: list[dict] = field(default_factory=list)
    chain_meta: dict | None = None


def _format_sandbox_result(r: SandboxResult) -> str:
    return (
        f"[exit={r.exit_code} duration={r.duration_s:.1f}s "
        f"timed_out={r.timed_out}]\n"
        f"--- stdout ---\n{r.stdout[-4000:]}\n"
        f"--- stderr ---\n{r.stderr[-4000:]}"
    )


# Per-run cap on tool-using turns. Higher → more cost, fewer auto-commits.
# Lower → cheaper, but more agents run out of budget mid-fit. 40 chosen in
# the EXP-006 design doc as a conservative starting point; revisit after
# the spike.
DEFAULT_MAX_TURNS = 40


class Orchestrator:
    def __init__(
        self,
        *,
        client: LLMClient,
        n_rows: int,
        max_validation_iters: int = 3,
        sandbox_timeout: int = 60,
        prompt_variant: str | PromptVariant = "baseline",
        max_turns: int = DEFAULT_MAX_TURNS,
    ):
        self.client = client
        self.n_rows = n_rows
        self.max_validation_iters = max_validation_iters
        self.sandbox_timeout = sandbox_timeout
        self.max_turns = max_turns
        self.pv = (
            prompt_variant
            if isinstance(prompt_variant, PromptVariant)
            else get_variant(prompt_variant)
        )

    def run(
        self,
        *,
        condition: Condition,
        dataset_name: str,
        workspace: Path,
        has_data: bool,
        has_descriptions: bool,
    ) -> RunResult:
        """Dispatch to the legacy code-block path or the new tool-using path
        based on the prompt variant."""
        if self.pv.is_tool_using:
            return self._run_tool_using(
                dataset_name=dataset_name,
                workspace=workspace,
                has_data=has_data,
                has_descriptions=has_descriptions,
            )
        return self._run_code_block(
            condition=condition,
            dataset_name=dataset_name,
            workspace=workspace,
            has_data=has_data,
            has_descriptions=has_descriptions,
        )

    # ===================== Legacy code-block path (baseline / rubric) =====================

    def _run_code_block(
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
            response = self.client.chat(history, system=self.pv.system)
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
                response = self.client.chat(history, system=self.pv.system)
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
                self.pv.exploration(has_data=has_data, has_descriptions=has_descriptions),
            )
            findings = explore_result.stdout[-2000:] or "(no findings printed)"
            step("MODELING", self.pv.modeling(findings_summary=findings))

            for iteration in range(self.max_validation_iters):
                v_result = step("VALIDATION", self.pv.validation())
                ok = "VALIDATION OK" in v_result.stdout.upper()
                if ok or iteration == self.max_validation_iters - 1:
                    break
                step(
                    "MODELING",
                    self.pv.modeling(
                        findings_summary="Validation flagged issues. Revise the model."
                    ),
                )

            step(
                "GENERATION",
                self.pv.generation(n_rows=self.n_rows, target_path="generated.csv"),
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
            for i, sr in enumerate(sandbox_results, 1):
                (workspace / f"step_{i:03d}.stdout").write_text(sr.stdout)
                (workspace / f"step_{i:03d}.stderr").write_text(sr.stderr)
                (workspace / f"step_{i:03d}.exit").write_text(str(sr.exit_code))
            sandbox.close()

    # ===================== New tool-using path (rubric_tools, EXP-006) =====================

    def _run_tool_using(
        self,
        *,
        dataset_name: str,
        workspace: Path,
        has_data: bool,
        has_descriptions: bool,
    ) -> RunResult:
        # Load the prepared workspace files. `experiments/runner.py` has
        # already written train.csv (when has_data) and descriptions.json
        # (when has_descriptions) into `workspace`.
        train_path = workspace / "train.csv"
        if not train_path.exists():
            # NO_DATA condition still needs *some* DataFrame so the runtime
            # state has a `train_fit` to return errors against — give it an
            # empty frame; data tools will refuse via `data_withheld`.
            train = pd.DataFrame()
        else:
            train = pd.read_csv(train_path)

        descriptions: dict | None = None
        desc_path = workspace / "descriptions.json"
        if has_descriptions and desc_path.exists():
            descriptions = json.loads(desc_path.read_text())

        state = build_runtime_state(
            workspace=workspace,
            train=train,
            descriptions=descriptions,
            has_data=has_data,
            has_descriptions=has_descriptions,
        )

        tools = all_tool_schemas()
        system_prompt = _assemble_system_prompt(
            self.pv.system, dataset_name=dataset_name, descriptions=descriptions,
            has_data=has_data, has_descriptions=has_descriptions,
        )

        transcript: list[TranscriptEntry] = []
        tool_call_log: list[dict] = []
        history: list[dict[str, Any]] = [{
            "role": "user",
            "content": (
                f"Build a generative model for the {dataset_name!r} dataset. "
                f"Sample size will be {self.n_rows}. Use the tools available; "
                "call commit_generator when the chain is ready."
            ),
        }]
        transcript.append(TranscriptEntry("user", history[0]["content"], "turn_001"))

        turn = 0
        while turn < self.max_turns and not state.committed:
            turn += 1
            stage = f"turn_{turn:03d}"
            _heartbeat(f"{stage}: chat_with_tools (history={len(history)} msgs)")
            t0 = time.time()
            try:
                resp: ToolUsingResponse = self.client.chat_with_tools(
                    history, tools=tools, system=system_prompt,
                )
            except Exception as e:
                _heartbeat(f"{stage}: LLM call failed: {type(e).__name__}: {e}")
                raise
            _heartbeat(
                f"{stage}: LLM responded in {time.time()-t0:.1f}s "
                f"({len(resp.content)} chars, {len(resp.tool_calls)} tool_calls)"
            )

            transcript.append(TranscriptEntry(
                "assistant", _format_assistant_for_transcript(resp), stage,
            ))
            history.append(resp.assistant_message)

            if not resp.tool_calls:
                # The agent emitted text without any tool call. Two reasons:
                #  (a) it thinks it's done — but didn't call commit_generator
                #  (b) it's confused
                # Either way, nudge it to commit or keep going.
                nudge = (
                    "Your previous turn included no tool call. If the chain is "
                    "ready, call `commit_generator()`. Otherwise pick the next "
                    "tool that advances toward a complete chain."
                )
                history.append({"role": "user", "content": nudge})
                transcript.append(TranscriptEntry("user", nudge, stage))
                continue

            for tc in resp.tool_calls:
                tool_t0 = time.time()
                result = dispatch_tool(state, tc.name, tc.arguments)
                tool_dt = time.time() - tool_t0
                tool_call_log.append({
                    "turn": turn, "tool": tc.name,
                    "arguments": tc.arguments, "result": result,
                    "duration_s": round(tool_dt, 4),
                })
                tool_result_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                }
                history.append(tool_result_msg)
                transcript.append(TranscriptEntry(
                    "tool",
                    f"[{tc.name}({_brief_args(tc.arguments)})] -> {_brief_result(result)}",
                    stage, duration_s=tool_dt,
                ))
                _heartbeat(
                    f"{stage}: {tc.name}({_brief_args(tc.arguments)}) -> "
                    f"{_brief_result(result)} in {tool_dt:.2f}s"
                )

        # Force-commit if we ran out of turns without an explicit commit.
        if not state.committed:
            _heartbeat(f"max_turns ({self.max_turns}) reached without commit; auto-committing")
            from ssdataagent.agent.tools.commit import commit_generator
            forced = commit_generator(state)
            tool_call_log.append({
                "turn": turn + 1, "tool": "commit_generator (forced)",
                "arguments": {}, "result": forced, "duration_s": 0.0,
            })
            transcript.append(TranscriptEntry(
                "tool", f"[forced commit_generator] -> {forced!r}", "auto_commit",
            ))
            if forced.get("error"):
                raise RuntimeError(
                    f"max_turns reached and forced commit failed: {forced!r}"
                )

        try:
            # Sample N rows from the now-committed chain.
            _heartbeat(f"sampling {self.n_rows} rows from committed chain")
            generated = state.chain.sample(state.rng, self.n_rows)
            chain_meta = state.chain.to_meta()
            generated.to_csv(workspace / "generated.csv", index=False)
            return RunResult(
                generated=generated,
                transcript=transcript,
                code_steps=[],
                sandbox_results=[],
                tool_call_log=tool_call_log,
                chain_meta=chain_meta,
            )
        finally:
            # Persist artefacts even if sampling crashed — they're how we
            # debug the run after the fact.
            try:
                chain_meta = state.chain.to_meta()
                (workspace / "chain.json").write_text(json.dumps(chain_meta, indent=2, default=str))
            except Exception as e:
                (workspace / "chain.json.error").write_text(f"{type(e).__name__}: {e}")
            (workspace / "transcript.json").write_text(json.dumps(
                [{"role": e.role, "content": e.content, "stage": e.stage,
                  "duration_s": e.duration_s} for e in transcript],
                indent=2, default=str,
            ))
            (workspace / "tool_calls.json").write_text(json.dumps(tool_call_log, indent=2, default=str))
            if state.progress_log:
                (workspace / "progress.log").write_text("\n".join(state.progress_log) + "\n")


# ===================== helpers =====================


def _assemble_system_prompt(
    base: str, *, dataset_name: str, descriptions: dict | None,
    has_data: bool, has_descriptions: bool,
) -> str:
    """Append a dataset-specific footer to the variant's system prompt."""
    parts = [base.rstrip(), ""]
    parts.append(f"DATASET: {dataset_name}")
    parts.append(f"AVAILABLE INPUTS: train.csv={has_data}, descriptions={has_descriptions}")
    if descriptions:
        # Small subset of descriptions surfaced in the system prompt so the
        # agent doesn't need to round-trip a tool call for it.
        bg = descriptions.get("background_variables", [])
        tg = descriptions.get("target_variables", [])
        if bg or tg:
            parts.append(f"BACKGROUND VARIABLES: {bg}")
            parts.append(f"TARGET VARIABLES:     {tg}")
        ctx = descriptions.get("context")
        if ctx:
            parts.append("POPULATION CONTEXT:")
            parts.append(str(ctx).strip())
    return "\n".join(parts)


def _brief_args(args: dict) -> str:
    """One-line summary of tool args for log readability."""
    s = ", ".join(f"{k}={_brief_value(v)}" for k, v in list(args.items())[:4])
    if len(args) > 4:
        s += ", …"
    return s


def _brief_value(v: Any) -> str:
    if isinstance(v, list):
        return f"[{len(v)} items]" if len(v) > 3 else repr(v)
    if isinstance(v, str) and len(v) > 40:
        return repr(v[:37] + "…")
    return repr(v)


def _brief_result(result: dict) -> str:
    """One-line summary of a tool result for log readability."""
    if "error" in result:
        return f"error={result['error']}"
    keys = list(result.keys())[:5]
    return "{" + ", ".join(f"{k}={_brief_value(result[k])}" for k in keys) + ("…}" if len(result) > 5 else "}")


def _format_assistant_for_transcript(resp: ToolUsingResponse) -> str:
    """Compact assistant turn for the human-readable transcript."""
    if not resp.tool_calls:
        return resp.content or "(empty)"
    calls = ", ".join(f"{tc.name}({_brief_args(tc.arguments)})" for tc in resp.tool_calls)
    if resp.content:
        return f"{resp.content}\n[tool_calls: {calls}]"
    return f"[tool_calls: {calls}]"
