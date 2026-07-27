"""Level-correction channel: keep source A's marginal SHAPE + copula, correct only the
per-outcome LOCATION for the target. See
docs/superpowers/specs/2026-07-27-level-correction-design.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.transfer.blind import MODEL, _last_json_object
from ssdataagent.transfer.rescue import select_r2_source


_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO / "results" / "levelcorrect_cache"


def _is_numeric(s: pd.Series) -> bool:
    """>90% of non-missing values parse as numbers. Mirrors nodonor_bracket._is_numeric
    (the scorer's test) so the shifted set matches what the scorer grades as numeric."""
    s = s.dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def numeric_outcomes(a: pd.DataFrame, b: pd.DataFrame, outs: list[str]) -> list[str]:
    """Outcomes numeric in BOTH frames (order preserved)."""
    return [y for y in outs if y in a.columns and y in b.columns
            and _is_numeric(a[y]) and _is_numeric(b[y])]


def outcome_mean(frame: pd.DataFrame, y: str) -> float:
    """NaN-aware mean of a numeric column (NaN if empty)."""
    v = pd.to_numeric(frame[y], errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def oracle_shifts(a: pd.DataFrame, b: pd.DataFrame, ys: list[str]) -> dict[str, float]:
    """Δ_y = mean_B(y) - mean_A(y) (reads B -- the ceiling arm)."""
    return {y: outcome_mean(b, y) - outcome_mean(a, y) for y in ys}


def pooled_shifts(a: pd.DataFrame, sib_rew: pd.DataFrame, ys: list[str]) -> dict[str, float]:
    """Δ_y = mean(sib_rew[y]) - mean_A(y): siblings raked to B's public X-margins."""
    return {y: outcome_mean(sib_rew, y) - outcome_mean(a, y) for y in ys}


def hybrid_shifts(pooled: dict[str, float], llm: dict[str, float],
                  n_siblings: int, ess: float) -> dict[str, float]:
    """Per outcome, the ESS-gated fuse (B6 gate): the pooled (retrieval) shift when the
    sibling pool is plural AND effectively-sized, else the llm (description) shift. An
    outcome missing/non-finite in the chosen arm falls back to the other (0.0 if neither)."""
    use_pooled = select_r2_source(n_siblings, ess)
    primary, backup = (pooled, llm) if use_pooled else (llm, pooled)
    out: dict[str, float] = {}
    for y in set(pooled) | set(llm):
        val = primary.get(y)
        if val is None or not np.isfinite(val):
            val = backup.get(y, 0.0)
        out[y] = float(val if val is not None and np.isfinite(val) else 0.0)
    return out


def apply_level_shift(marg: pd.DataFrame, shifts: dict[str, float]) -> pd.DataFrame:
    """Copy of ``marg`` with each numeric column named in ``shifts`` shifted by its Δ (NaN
    preserved); all other columns byte-identical. A zero / non-finite Δ or a column absent
    from ``marg`` is a no-op."""
    out = marg.copy()
    for y, d in shifts.items():
        if y not in out.columns or not np.isfinite(d) or d == 0.0:
            continue
        out[y] = pd.to_numeric(out[y], errors="coerce") + d
    return out


def llm_level_prompt(ds: str, a: pd.DataFrame, ys: list[str]) -> str:
    """Ask ONLY for the population MEAN of each numeric outcome in the described target
    context. Reads B's audited description (blind_specs); never B's data."""
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    spec = BLIND_SPECS[ds]
    body = "\n".join(f'- "{y}": {spec.glosses.get(y, y)}' for y in ys)
    return (
        f"You are estimating population averages in {spec.population}.\n"
        f"Context:\n{spec.description}\n\n"
        f"Using ONLY your knowledge of this described population -- no external data -- give "
        f"the approximate population MEAN (average) value of EACH numeric variable below. "
        f"Reply with ONE JSON object mapping each variable name to a single number.\n\n"
        f"{body}\n\nOutput only the JSON."
    )


def parse_levels(text: str, ys: list[str]) -> dict[str, float]:
    """Parse the LLM JSON into {y: mean}; keep only finite numeric entries for ys, drop junk."""
    raw = _last_json_object(text)
    out: dict[str, float] = {}
    for y in ys:
        try:
            fv = float(raw.get(y))
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out[y] = fv
    return out


def llm_shifts(a: pd.DataFrame, ds: str, ys: list[str], *, client=None,
               cache_dir: Path | None = None, regenerate: bool = False) -> dict[str, float]:
    """Δ_y = llm_mean(y) - mean_A(y), with llm_mean elicited from B's DESCRIPTION only and
    cached to <cache_dir>/<ds>_levels.json. An outcome the LLM omits/garbles is dropped (no
    shift -> carryover for it). Reads no B data. Warm cache returns without calling the LLM."""
    cache_dir = cache_dir or _DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ds}_levels.json"
    if path.exists() and not regenerate:
        levels = json.loads(path.read_text())
    else:
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url="https://openrouter.ai/api/v1",
                            api_key=os.environ["OPENROUTER_API_KEY"])
        prompt = llm_level_prompt(ds, a, ys)
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}])
        levels = parse_levels(resp.choices[0].message.content, ys)
        path.write_text(json.dumps(levels, ensure_ascii=False, indent=2))
    return {y: float(levels[y]) - outcome_mean(a, y) for y in ys if y in levels}
