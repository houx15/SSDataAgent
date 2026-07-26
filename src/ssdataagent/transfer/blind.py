"""Blind face-swap (Approach A): transfer source A's copula, get the target's marginals
from an LLM that reads only the target's textual description. See
docs/superpowers/specs/2026-07-26-blind-faceswap-design.md.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.transfer.generate import _is_numeric

_logger = logging.getLogger(__name__)

MODEL = "anthropic/claude-sonnet-4.5"
_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO / "results" / "blind_cache"


def _synth_numeric(quantiles: list[float], L: int) -> np.ndarray:
    """Length-L numeric column whose empirical distribution matches ``quantiles`` (values at
    evenly spaced probabilities 0..1). Inverse-CDF interpolation on a regular p-grid."""
    q = np.sort(np.asarray(quantiles, dtype=float))
    ps = np.linspace(0.0, 1.0, len(q))
    grid = (np.arange(L) + 0.5) / L
    return np.interp(grid, ps, q)


def _synth_categorical(probs: dict, L: int) -> np.ndarray:
    """Length-L object column whose value_counts(normalize) match ``probs`` (largest-remainder
    rounding, so the length is exactly L and the result is deterministic)."""
    cats = np.array(list(probs.keys()), dtype=object)
    if len(cats) == 0:
        raise ValueError("empty probs")
    p = np.asarray([probs[c] for c in probs.keys()], dtype=float)
    p = p / p.sum()
    exact = p * L
    counts = np.floor(exact).astype(int)
    rem = int(L - counts.sum())
    if rem > 0:
        counts[np.argsort(-(exact - counts))[:rem]] += 1
    return np.repeat(cats, counts)


def build_marg_frame(elicited: dict, source_a: pd.DataFrame, cols: list[str], *,
                     L: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthesize the ``marg`` frame for transfer_build from LLM-elicited distributions.
    Numeric-ness, the category universe, and each column's missingness RATE come from
    ``source_a`` (transferred structure); the distribution SHAPE comes from ``elicited``.
    A column absent/malformed in ``elicited`` falls back to A's own marginal (carry-over)."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        num = _is_numeric(source_a[c])
        dist = elicited.get(c)
        try:
            if dist is None:
                raise ValueError("no elicited distribution")
            col = (_synth_numeric(dist["quantiles"], L) if num
                   else _synth_categorical(dist["probs"], L)).astype(object)
        except (KeyError, ValueError, TypeError) as e:
            _logger.warning("blind: column %r falls back to A's marginal (%s)", c, e)
            vals = source_a[c].dropna().to_numpy()
            col = (vals[rng.integers(0, len(vals), L)].astype(object) if len(vals)
                   else np.full(L, np.nan, dtype=object))
        miss = float(source_a[c].isna().mean())          # carry A's missingness rate
        if miss > 0:
            k = int(round(miss * L))
            if k > 0:
                col = col.copy()
                col[rng.choice(L, min(k, L), replace=False)] = np.nan
        out[c] = col
    return pd.DataFrame(out)


def elicit_prompt(ds: str, source_a: pd.DataFrame, cols: list[str], *,
                  max_cats: int = 20) -> str:
    """Prompt asking ONLY for per-variable marginal distributions of the described target
    context. Categorical variables list their category universe (from source A, the
    codebook level); numeric variables ask for 11 quantiles at probabilities 0.0..1.0."""
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    spec = BLIND_SPECS[ds]
    lines = []
    for c in cols:
        gloss = spec.glosses.get(c, c)
        if _is_numeric(source_a[c]):
            lines.append(f'- "{c}" (NUMERIC): {gloss}. Give "quantiles": 11 values at '
                         f'probabilities 0.0,0.1,...,1.0 (min..max).')
        else:
            cats = source_a[c].dropna().astype(str).value_counts().index.tolist()[:max_cats]
            lines.append(f'- "{c}" (CATEGORICAL, categories={cats}): {gloss}. '
                         f'Give "probs": a probability for each category (summing to ~1).')
    body = "\n".join(lines)
    return (
        f"You are estimating the population marginals of {spec.population}.\n"
        f"Context and coherence structure:\n{spec.description}\n\n"
        f"Using ONLY your knowledge of this described population — no external data — "
        f"estimate the marginal distribution of EACH variable below. Do not model any "
        f"joint relationship; marginals only.\n\n{body}\n\n"
        f'Reply with ONE JSON object keyed by variable name, each value either '
        f'{{"quantiles": [...]}} (numeric) or {{"probs": {{cat: p, ...}}}} (categorical). '
        f"Output only the JSON."
    )


def _last_json_object(text: str) -> dict:
    """The last complete top-level JSON object in ``text`` (brace-balanced, so nested
    objects/arrays parse -- the elicited marginals are nested, unlike B3's flat R^2 JSON,
    so this is self-contained here rather than reusing conditional_variance's flat-regex
    extractor). Returns {} if none parses."""
    decoder = json.JSONDecoder()
    last, idx = None, 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
            last, idx = obj, end
        except json.JSONDecodeError:
            idx = start + 1
    return last if isinstance(last, dict) else {}


def parse_marginals(text: str, source_a: pd.DataFrame, cols: list[str]) -> dict:
    """Parse the LLM's JSON into {var: dist}. Keeps only well-formed entries for ``cols``;
    a numeric var needs a non-empty ``quantiles`` list, a categorical var a non-empty
    ``probs`` dict. Malformed/absent entries are dropped (build_marg_frame then carries A)."""
    raw = _last_json_object(text)
    out: dict = {}
    for c in cols:
        d = raw.get(c) if isinstance(raw, dict) else None
        if not isinstance(d, dict):
            continue
        if _is_numeric(source_a[c]):
            q = d.get("quantiles")
            if isinstance(q, list) and len(q) >= 2:
                out[c] = {"quantiles": [float(x) for x in q]}
        else:
            pr = d.get("probs")
            if isinstance(pr, dict) and pr:
                out[c] = {"probs": {str(k): float(v) for k, v in pr.items()}}
    return out


def elicit_marginals(ds: str, source_a: pd.DataFrame, cols: list[str], *,
                     client=None, cache_dir: Path | None = None,
                     regenerate: bool = False) -> dict:
    """Elicit B's marginals from its description (LLM), cached to
    ``<cache_dir>/<ds>_marginals.json`` (durable, gitignored). Reads no B data. When the
    cache is warm and ``regenerate`` is False, returns it without calling the LLM."""
    cache_dir = cache_dir or _DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ds}_marginals.json"
    if path.exists() and not regenerate:
        return json.loads(path.read_text())
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
    prompt = elicit_prompt(ds, source_a, cols)
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}])
    parsed = parse_marginals(resp.choices[0].message.content, source_a, cols)
    path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    return parsed
