"""Knowledge-based life-course event ages — the no-donor sibling of event_timing.

The donor-based ``conditional_joint_repair`` (event_timing.py) fixes T4 by copying
a joint event-age tuple from a covariate-matched *real donor*. That needs
microdata. This module answers the no-donor question: can the model's *knowledge*
of how lives unfold stand in for the donor?

The LLM emits one compact spec per demographic stratum — the distribution over
event *orderings*, the typical age *gaps* between consecutive events, and the
*occurrence* structure. A deterministic sampler then reconstructs each person's
event ages as ``anchor + positive gaps`` so the ordering holds *by construction*
(the mechanism T4 rewards), with occurrence and the anchor marginal calibrated to
the disjoint pool's aggregates. The ordering distribution — the actual T4 signal —
comes only from the LLM; nothing here reads the test reference.

Ordering labels are ``"<"``-joined event *column names*, e.g.
``"age_finished_education<age_at_first_marriage<age_at_first_child"``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.data.event_timing import event_timing_variables
from ssdataagent.evaluation.disclosure import _row_hashes

_DEFAULT_GAP = (2.0, 2.0)

# Canonical life-course order per dataset (earliest -> latest), used to build the
# fixture's ordering distribution. The real generator gets this from the LLM.
_CANONICAL_ORDER: dict[str, list[str]] = {
    "cfps": [
        "age_finished_education",
        "age_at_first_marriage",
        "age_at_first_child",
    ],
}

_LOW_EDU = {"primary school or below", "middle school"}


@dataclass(frozen=True)
class StratumEventSpec:
    """One demographic stratum's life-course structure, from knowledge.

    ordering:   "<"-joined event-column label -> probability (sums to ~1).
    gaps:       "earlier->later" event-column pair -> (mean_years, sd_years), >0.
    occurrence: event column -> P(event occurs). A hint; the sampler overrides it
                with the pool's aggregate rate when a pool is supplied.
    requires:   event column -> prerequisite event column (empty for cfps, where a
                child may precede marriage).
    """

    ordering: dict[str, float]
    gaps: dict[str, tuple[float, float]]
    occurrence: dict[str, float]
    requires: dict[str, str] = field(default_factory=dict)


def education_bucket(value: object) -> str:
    """Coarsen a cfps ``highest_education`` value to ``low`` / ``high``."""
    return "low" if str(value).strip().lower() in _LOW_EDU else "high"


def _ordering_distribution(canonical: list[str]) -> dict[str, float]:
    """A realistic spread over the 3! orderings: canonical dominant, a small
    child-before-marriage minority (the ~few-percent T4 tolerance lives here), and
    thin tails. Weights sum to 1.0 exactly."""
    e0, e1, e2 = canonical  # edu, marriage, child
    perms = {
        f"{e0}<{e1}<{e2}": 0.930,  # edu < marriage < child (canonical)
        f"{e0}<{e2}<{e1}": 0.040,  # child before marriage
        f"{e1}<{e0}<{e2}": 0.020,  # married before finishing education
        f"{e2}<{e1}<{e0}": 0.005,
        f"{e1}<{e2}<{e0}": 0.003,
        f"{e2}<{e0}<{e1}": 0.002,
    }
    return perms


def fixture_specs(dataset: str) -> dict[tuple, StratumEventSpec]:
    """Hand-authored per-stratum specs (gender x education bucket) from life-course
    common sense — no LLM. Lets the sampler, calibration, and integration be built
    and tested offline. The production path replaces this with LLM elicitation."""
    canonical = _CANONICAL_ORDER.get(dataset)
    if canonical is None:
        raise ValueError(f"no canonical event order known for dataset {dataset!r}")
    events = set(event_timing_variables(dataset))
    if not events.issubset(set(canonical)):
        raise ValueError(f"{dataset} T4 events {events} not covered by canonical order")
    e0, e1, e2 = canonical
    ordering = _ordering_distribution(canonical)

    specs: dict[tuple, StratumEventSpec] = {}
    for gender in ("Male", "Female"):
        for edu in ("low", "high"):
            edu_marriage_mean = 5.0 if edu == "high" else 3.0
            specs[(gender, edu)] = StratumEventSpec(
                ordering=dict(ordering),
                gaps={
                    f"{e0}->{e1}": (edu_marriage_mean, 3.0),  # finish edu -> marry
                    f"{e1}->{e2}": (2.0, 2.0),                # marry -> first child
                },
                occurrence={
                    e0: 0.98,
                    e1: 0.88 if gender == "Female" else 0.90,
                    e2: 0.82 if edu == "high" else 0.86,
                },
                requires={},
            )
    return specs


def _sentinel_map(pool: pd.DataFrame, event_vars: list[str]) -> dict[str, object]:
    """The non-occurrence code for each event column: the pool's most common
    non-numeric value (``never married``, …), or NaN if the column has none."""
    out: dict[str, object] = {}
    for c in event_vars:
        s = pool[c]
        nonnum = s[pd.to_numeric(s, errors="coerce").isna() & s.notna()].astype(str)
        out[c] = nonnum.value_counts().index[0] if len(nonnum) else np.nan
    return out


def _occurrence_rates(pool: pd.DataFrame, event_vars: list[str]) -> dict[str, float]:
    return {c: float(pd.to_numeric(pool[c], errors="coerce").notna().mean())
            for c in event_vars}


def _anchor_pools(pool: pd.DataFrame, event_vars: list[str]) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(pool[c], errors="coerce").dropna().to_numpy(dtype=float)
            for c in event_vars}


def sample_event_block(
    demographics: pd.DataFrame,
    specs: dict[tuple, StratumEventSpec],
    pool: pd.DataFrame,
    event_vars: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample the event-age columns for each person in ``demographics``.

    For each person: pick the stratum spec, draw an ordering from it, draw which
    events occur (rate from the pool aggregate, gated by ``requires``), then lay
    the occurring events out as ``anchor + positive integer gaps`` in the drawn
    order. Order therefore holds *by construction* — the mechanism T4 rewards —
    and the realized-order distribution reproduces the spec's. Non-occurring
    events emit the column's sentinel (or NaN). Returns one column per event_var.
    """
    event_vars = [c for c in event_vars if c in pool.columns]
    sentinel = _sentinel_map(pool, event_vars)
    occ_rate = _occurrence_rates(pool, event_vars)
    anchor = _anchor_pools(pool, event_vars)
    n = len(demographics)
    cols = {c: np.empty(n, dtype=object) for c in event_vars}

    genders = demographics["gender"].astype(str).to_numpy()
    if "highest_education" in demographics.columns:
        edus = demographics["highest_education"].map(education_bucket).to_numpy()
    else:
        edus = np.array(["high"] * n)
    any_spec = next(iter(specs.values()))

    for i in range(n):
        spec = specs.get((genders[i], edus[i]), any_spec)
        labels = list(spec.ordering.keys())
        probs = np.array(list(spec.ordering.values()), dtype=float)
        probs = probs / probs.sum()
        order = labels[rng.choice(len(labels), p=probs)].split("<")

        occurs = {c: rng.random() < occ_rate.get(c, spec.occurrence.get(c, 1.0))
                  for c in event_vars}
        for evt, prereq in spec.requires.items():
            if evt in occurs and not occurs.get(prereq, False):
                occurs[evt] = False

        for c in event_vars:
            if not occurs.get(c):
                cols[c][i] = sentinel.get(c, np.nan)

        occurring = [c for c in order if c in event_vars and occurs.get(c)]
        if not occurring:
            continue
        first = occurring[0]
        age = int(rng.choice(anchor[first])) if len(anchor[first]) else 18
        cols[first][i] = age
        prev, prev_age = first, age
        for c in occurring[1:]:
            mean, sd = spec.gaps.get(f"{prev}->{c}", _DEFAULT_GAP)
            prev_age += max(1, int(round(rng.normal(mean, sd))))
            cols[c][i] = prev_age
            prev = c

    return pd.DataFrame(cols, index=demographics.index)


def calibrate_event_block(
    block: pd.DataFrame,
    pool: pd.DataFrame,
    event_vars: list[str],
) -> pd.DataFrame:
    """Pull each event's *occurring* age marginal onto the pool's aggregate marginal
    without reordering anyone.

    The sampler anchors on the first event's pool marginal, so later events (built
    as anchor + gaps) drift off their own marginals. We rank-map each event column
    to its pool marginal (a pure aggregate stat), then, for the rare person whose
    within-event order would flip under that map, revert *that person's* ages to the
    constructed values. Order is therefore preserved exactly; marginals match the
    pool for everyone the map leaves ordered. Sentinels / NaN are untouched.
    """
    event_vars = [c for c in event_vars if c in block.columns and c in pool.columns]
    orig = {c: pd.to_numeric(block[c], errors="coerce").to_numpy(dtype=float)
            for c in event_vars}
    cal = {}
    for c in event_vars:
        v = orig[c].copy()
        mask = ~np.isnan(v)
        pv = np.sort(pd.to_numeric(pool[c], errors="coerce").dropna().to_numpy(float))
        if mask.sum() and len(pv):
            u = pd.Series(v[mask]).rank(pct=True, method="first").to_numpy()
            v[mask] = pv[np.clip((u * len(pv)).astype(int), 0, len(pv) - 1)]
        cal[c] = v

    O = np.column_stack([orig[c] for c in event_vars])
    C = np.column_stack([cal[c] for c in event_vars])
    out = {c: block[c].to_numpy(dtype=object, copy=True) for c in event_vars}
    for i in range(len(block)):
        occ = [j for j in range(len(event_vars)) if not np.isnan(O[i, j])]
        if not occ:
            continue
        src = C
        if len(occ) >= 2:
            ov = [O[i, j] for j in occ]
            cv = [C[i, j] for j in occ]
            if tuple(np.argsort(ov)) != tuple(np.argsort(cv)):
                src = O  # calibration would flip this person's order -> revert
        for j in occ:
            out[event_vars[j]][i] = float(src[i, j])
    return pd.DataFrame(out, index=block.index)


def _assert_disjoint(pool: pd.DataFrame, ref: pd.DataFrame, threshold: float = 0.5) -> None:
    """Honest-boundary guard: refuse to draw aggregates from a pool that is (mostly)
    the test reference. Compares row hashes on the shared columns."""
    common = [c for c in pool.columns if c in ref.columns]
    if not common:
        return
    ref_hashes = set(_row_hashes(ref, common))
    overlap = float(_row_hashes(pool, common).isin(ref_hashes).mean())
    if overlap > threshold:
        raise ValueError(
            f"pool overlaps the forbidden reference by {overlap:.0%} — the no-donor "
            "event-order module must not read the test data"
        )


def apply_event_order(
    frame: pd.DataFrame,
    dataset: str,
    specs: dict[tuple, StratumEventSpec],
    pool: pd.DataFrame,
    rng: np.random.Generator,
    *,
    forbid_ref: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with its T4 event-timing columns replaced by the
    knowledge-sampled, pool-calibrated event block. Only the columns
    ``event_timing_variables(dataset)`` change. When ``forbid_ref`` is given, raises
    if ``pool`` overlaps it (the honest-boundary guard)."""
    if forbid_ref is not None:
        _assert_disjoint(pool, forbid_ref)
    event_vars = [c for c in event_timing_variables(dataset) if c in frame.columns]
    if not event_vars:
        return frame.copy()
    block = sample_event_block(frame, specs, pool, event_vars, rng)
    block = calibrate_event_block(block, pool, event_vars)
    out = frame.copy()
    for c in event_vars:
        out[c] = block[c].to_numpy()
    return out


# --- LLM elicitation (the joint comes from here; nothing else reads the pool joint) ---

def _extract_json(text: str) -> dict:
    """Pull the JSON object out of an LLM reply (tolerates prose / code fences)."""
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return json.loads(text)


def parse_stratum_response(text: str) -> StratumEventSpec:
    """Parse one stratum's LLM reply into a `StratumEventSpec`, defensively:
    ordering renormalised to sum 1, gap means floored strictly positive, occurrence
    clipped to [0, 1]. Zero-weight orderings are dropped."""
    obj = _extract_json(text)
    ordering = {str(k): float(v) for k, v in (obj.get("ordering") or {}).items()
                if float(v) > 0}
    total = sum(ordering.values()) or 1.0
    ordering = {k: v / total for k, v in ordering.items()}
    gaps: dict[str, tuple[float, float]] = {}
    for k, v in (obj.get("gaps") or {}).items():
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            mean = max(0.5, float(v[0]))
            sd = float(v[1]) if len(v) > 1 else 1.0
            gaps[str(k)] = (mean, max(0.1, sd))
    occ = {str(k): min(1.0, max(0.0, float(v)))
           for k, v in (obj.get("occurrence") or {}).items()}
    req = {str(k): str(v) for k, v in (obj.get("requires") or {}).items()}
    return StratumEventSpec(ordering=ordering, gaps=gaps, occurrence=occ, requires=req)


def _elicit_prompt(stratum: tuple, events: list[str], canonical: list[str],
                   descriptions: dict[str, str] | None) -> str:
    desc = ""
    if descriptions:
        desc = "\n".join(f"- {e}: {descriptions[e]}" for e in events if e in descriptions)
    return f"""You are a demographer. Describe the life-course event timing for this
group of people, FROM YOUR KNOWLEDGE (no data is provided):

  stratum: {dict(zip(('gender', 'education'), stratum))}
  events (columns): {events}
  {desc}

Return ONE JSON object with keys:
- "ordering": map each plausible age ordering (a "<"-joined sequence of the event
  columns, e.g. "{'<'.join(canonical)}") to its probability for this group.
  Include the realistic minority orders (e.g. a first child before marriage).
- "gaps": map "earlier->later" adjacent event pairs to [mean_years, sd_years].
- "occurrence": map each event to the probability it ever occurs for this group.
- "requires": map an event to a prerequisite event, or {{}} if none.

Output only the JSON."""


def elicit_stratum_specs(
    dataset: str,
    strata: list[tuple],
    client,
    model: str,
    *,
    descriptions: dict[str, str] | None = None,
    cache_path: str | None = None,
) -> dict[tuple, StratumEventSpec]:
    """Elicit one `StratumEventSpec` per stratum via the LLM (one chat call each),
    caching raw replies to ``cache_path`` so re-scoring needs no LLM calls."""
    if cache_path and Path(cache_path).exists():
        raw = json.loads(Path(cache_path).read_text())
        return {tuple(k.split("|")): parse_stratum_response(v) for k, v in raw.items()}

    events = event_timing_variables(dataset)
    canonical = _CANONICAL_ORDER.get(dataset, events)
    raw, specs = {}, {}
    for stratum in strata:
        prompt = _elicit_prompt(stratum, events, canonical, descriptions)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=1500, temperature=0.3)
        text = resp.choices[0].message.content or ""
        raw["|".join(map(str, stratum))] = text
        specs[tuple(stratum)] = parse_stratum_response(text)
    if cache_path:
        Path(cache_path).write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    return specs
