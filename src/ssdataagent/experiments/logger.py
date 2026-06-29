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
