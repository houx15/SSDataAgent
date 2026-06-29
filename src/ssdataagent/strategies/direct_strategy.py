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
