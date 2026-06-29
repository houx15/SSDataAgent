from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ssdataagent.data.schema import DatasetSchema

_PROMPT_VERSION = "designb-v1"
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You are a survey-distribution estimator. For a demographic subgroup, you "
    "estimate the DISTRIBUTION of each target variable across people in that "
    "subgroup — not a single typical value. Real subgroups have substantial "
    "internal variation; reflect that spread. Return ONLY a JSON object."
)


def target_support(schema: DatasetSchema, target: str, *, n_numeric_bins: int = 10) -> dict:
    if target in schema.numeric_ranges:
        lo, hi = schema.numeric_ranges[target]
        return {"kind": "num", "edges": np.linspace(float(lo), float(hi), n_numeric_bins + 1)}
    cats = schema.allowed_values.get(target) or []
    return {"kind": "cat", "support": list(cats)}


def _support_len(support: dict) -> int:
    return len(support["support"]) if support["kind"] == "cat" else len(support["edges"]) - 1


def known_vector(known_m_t: dict, support: dict) -> np.ndarray:
    if support["kind"] == "cat":
        probs = (known_m_t or {}).get("probs", {})
        v = np.array([float(probs.get(str(c), 0.0)) for c in support["support"]], float)
    else:
        quant = (known_m_t or {}).get("quantiles", {})
        edges = support["edges"]
        if quant:
            qs = sorted(float(q) for q in quant.keys())
            vals = [float(quant[str(q)]) if str(q) in quant else float(quant[q]) for q in qs]
            cdf = np.interp(edges, vals, qs, left=0.0, right=1.0)
            v = np.clip(np.diff(cdf), 0.0, None)
        else:
            v = np.zeros(_support_len(support))
    s = v.sum()
    return v / s if s > 0 else np.full(_support_len(support), 1.0 / _support_len(support))


def _normalize_to_support(raw, support: dict) -> np.ndarray | None:
    n = _support_len(support)
    if not isinstance(raw, (list, tuple)) or len(raw) != n:
        return None
    try:
        v = np.array([max(float(x), 0.0) for x in raw], float)
    except (TypeError, ValueError):
        return None
    s = v.sum()
    return v / s if s > 0 else None


def _describe_support(support: dict) -> str:
    if support["kind"] == "cat":
        return f"categories {support['support']} (give one probability per category, in order)"
    e = support["edges"]
    ranges = [f"[{e[i]:.4g},{e[i+1]:.4g})" for i in range(len(e) - 1)]
    return f"numeric bins {ranges} (give one probability per bin, in order)"


def _build_prompt(*, dataset, cell_desc, schema, targets, supports, known_vectors, transport) -> str:
    lines = [
        f"Population: {schema.population_context}",
        f"Demographic subgroup: {json.dumps(cell_desc, default=str)}",
        "",
        "For EACH target below, return a probability vector over its support "
        "(probabilities for that subgroup; reflect realistic within-subgroup spread):",
    ]
    for t in targets:
        desc = schema.descriptions.get(t, "")
        anchor = np.round(known_vectors[t], 4).tolist()
        lines.append(f"- {t}{(': ' + desc) if desc else ''} — {_describe_support(supports[t])}. "
                     f"Population-wide marginal (anchor, do not copy blindly): {anchor}")
    if transport:
        lines.append("")
        lines.append("NOTE: the anchors come from a DIFFERENT source population. Adapt each "
                     "subgroup distribution to THIS population's context, not the source's.")
    lines.append("")
    lines.append('Respond with ONLY JSON: {"<target>": [p1, p2, ...], ...}')
    return "\n".join(lines)


def _cache_key(dataset, condition, model, cell_key, targets) -> str:
    blob = json.dumps([dataset, condition, model, cell_key, sorted(targets), _PROMPT_VERSION],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def elicit_cell_distributions(
    client, *, dataset, condition, cell_descs, schema, targets, supports,
    known_vectors, run_dir, cache_dir, transport=False, max_retries=3,
) -> dict[str, dict[str, np.ndarray]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(run_dir) / "elicitation"
    log_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(getattr(client, "cfg", None), "model", "unknown")
    result: dict[str, dict[str, np.ndarray]] = {}

    for cell_key, cell_desc in cell_descs.items():
        key = _cache_key(dataset, condition, model, cell_key, targets)
        cache_file = cache_dir / f"{key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            result[cell_key] = {t: np.array(cached[t], float) for t in targets}
            continue

        prompt = _build_prompt(dataset=dataset, cell_desc=cell_desc, schema=schema,
                               targets=targets, supports=supports,
                               known_vectors=known_vectors, transport=transport)
        parsed: dict[str, np.ndarray] = {}
        raw = ""
        for attempt in range(max_retries + 1):
            raw = client.chat(messages=[{"role": "user", "content": prompt}], system=_SYSTEM)
            m = _JSON_OBJ.search(raw or "")
            obj = {}
            if m:
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    obj = {}
            ok = True
            parsed = {}
            for t in targets:
                vec = _normalize_to_support(obj.get(t), supports[t])
                if vec is None:
                    ok = False
                    break
                parsed[t] = vec
            if ok:
                break
        else:
            parsed = {}
        # fallback for any target not successfully parsed
        for t in targets:
            if t not in parsed:
                parsed[t] = np.array(known_vectors[t], float)

        (log_dir / f"{cell_key.replace('|', '_')}.prompt.txt").write_text(prompt)
        (log_dir / f"{cell_key.replace('|', '_')}.response.txt").write_text(raw or "")
        cache_file.write_text(json.dumps({t: parsed[t].tolist() for t in targets}))
        result[cell_key] = parsed
    return result
