"""Phase-7 baseline: direct LLM generation, one individual at a time.

Mirrors SSDataBench's per-individual paradigm but uses our LLM client (so we
can hit any OpenAI-compatible provider, including DeepSeek). This is the
control condition we compare the agent against.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from ssdataagent.agent.llm_client import LLMClient
from ssdataagent.data.schema import load_schema


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

_DIRECT_SYSTEM = (
    "You are a survey-respondent simulator. Given a partial profile of an "
    "individual, fill in the requested target variables with plausible values "
    "drawn from the population. Return ONLY a JSON object whose keys are the "
    "target variable names; no prose, no markdown."
)


def _build_user_prompt(
    *, sampled_row: dict, targets: list[str], descriptions: dict[str, str],
    allowed: dict[str, list], numeric: dict[str, tuple[float, float]],
    population_context: str,
) -> str:
    lines = [
        f"Population context: {population_context}",
        "",
        "Background variables (already known for this individual):",
        json.dumps(sampled_row, default=str),
        "",
        "Target variables to fill in:",
    ]
    for var in targets:
        bits = [f"  {var}"]
        if descriptions.get(var):
            bits.append(f": {descriptions[var]}")
        if var in allowed:
            bits.append(f" (allowed: {allowed[var]})")
        elif var in numeric:
            lo, hi = numeric[var]
            bits.append(f" (numeric, range [{lo}, {hi}])")
        lines.append("".join(bits))
    lines.append("")
    lines.append('Respond with a JSON object like {"target1": value1, "target2": value2, ...}')
    return "\n".join(lines)


def _parse_targets(raw: str, targets: list[str]) -> dict[str, Any]:
    m = _JSON_OBJ.search(raw)
    if not m:
        return {t: None for t in targets}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {t: None for t in targets}
    return {t: obj.get(t) for t in targets}


def generate_direct(
    *,
    client: LLMClient,
    sampled: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Generate target variables for each row of *sampled* via direct LLM calls."""
    schema = load_schema(dataset_name)
    bg = [c for c in sampled.columns if c in schema.background_variables or c == "profile_id"]
    targets = list(schema.target_variables)

    rows: list[dict] = []
    for _, row in sampled.iterrows():
        sampled_row = {c: row[c] for c in bg}
        prompt = _build_user_prompt(
            sampled_row=sampled_row,
            targets=targets,
            descriptions=schema.descriptions,
            allowed=schema.allowed_values,
            numeric=schema.numeric_ranges,
            population_context=schema.population_context,
        )
        raw = client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=_DIRECT_SYSTEM,
        )
        parsed = _parse_targets(raw, targets)
        rows.append({**sampled_row, **parsed})
    return pd.DataFrame(rows)
