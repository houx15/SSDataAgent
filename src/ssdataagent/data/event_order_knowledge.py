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

from dataclasses import dataclass, field

from ssdataagent.data.event_timing import event_timing_variables

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
