from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssdataagent.agent.orchestrator import RunResult


def log_run(result: RunResult, *, run_dir: Path, meta: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

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

    result.generated.to_csv(run_dir / "generated.csv", index=False)
