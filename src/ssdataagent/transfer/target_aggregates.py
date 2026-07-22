from __future__ import annotations

import pandas as pd

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.copula_stability import pairwise_associations
from ssdataagent.transfer.generate import _is_numeric


def target_aggregates(pool: pd.DataFrame, cols: list[str], covariates: list[str],
                      outcomes: list[str]) -> dict:
    """Firewalled low-order aggregates of the TARGET's disjoint pool: per-pair
    associations and per-outcome covariate-R^2. Reads only ``pool`` — never a test or
    reference sample. Provenance-tagged so the firewall is auditable per cell."""
    assoc = pairwise_associations(pool, cols)
    pairwise_assoc = {k: v[0] for k, v in assoc.items()}
    pairwise_method = {k: v[1] for k, v in assoc.items()}
    num_pred = frozenset(c for c in covariates if _is_numeric(pool[c]))
    preds = [c for c in covariates if c in pool.columns]
    outcome_r2 = {
        y: (covariate_r2(pool, y, preds, numeric_predictors=num_pred)
            if y in pool.columns and _is_numeric(pool[y]) else None)
        for y in outcomes
    }
    return {
        "pairwise_assoc": pairwise_assoc,
        "pairwise_method": pairwise_method,
        "outcome_r2": outcome_r2,
        "provenance": {"source": "target_pool", "n_rows": int(len(pool)),
                       "reads": "marginals+pairwise_assoc+covariate_r2"},
    }
