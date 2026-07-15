"""No-donor event-order module (A) — knowledge-based life-course event ages.

These pin the core promise: the LLM (here, a hand-authored fixture) supplies the
ordering *distribution* and gaps; the sampler turns that into per-person event
ages whose order holds by construction, with occurrence calibrated to the pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssdataagent.data.event_order_knowledge import (
    StratumEventSpec,
    fixture_specs,
)

CFPS_EVENTS = ["age_finished_education", "age_at_first_marriage", "age_at_first_child"]


def test_fixture_specs_wellformed():
    specs = fixture_specs("cfps")
    assert len(specs) >= 4, "expect >=4 strata (2 gender x 2 education buckets)"
    for key, spec in specs.items():
        assert isinstance(spec, StratumEventSpec)
        assert abs(sum(spec.ordering.values()) - 1.0) < 1e-6, f"{key} ordering !~ 1"
        assert all(m > 0 for (m, _sd) in spec.gaps.values()), f"{key} has non-positive gap"
        assert all(0.0 <= p <= 1.0 for p in spec.occurrence.values()), f"{key} bad occ"
        # every ordering label is a permutation of the cfps event columns
        for label in spec.ordering:
            assert sorted(label.split("<")) == sorted(CFPS_EVENTS), f"bad label {label}"
