from __future__ import annotations

import logging

import pandas as pd

from ssdataagent.data.conditional_variance import covariate_r2
from ssdataagent.transfer.copula_stability import pairwise_associations
from ssdataagent.transfer.generate import _is_numeric

_logger = logging.getLogger(__name__)


def target_aggregates(pool: pd.DataFrame, cols: list[str], covariates: list[str],
                      outcomes: list[str]) -> dict:
    """Firewalled low-order aggregates of the TARGET's disjoint pool: per-pair
    associations and per-outcome covariate-R^2. Reads only ``pool`` — never a test or
    reference sample. Provenance-tagged so the firewall is auditable per cell.

    ``outcome_r2`` no longer gates on ``_is_numeric(pool[y])``: a real survey column
    can be numeric except for a categorical sentinel (e.g. ``age_first_childbirth``
    carrying the string "No Child" for respondents who never had one), which fails
    that >90%-coercible check even though most of its rows are perfectly usable.
    ``covariate_r2`` already does the right thing on its own -- it coerces with
    ``pd.to_numeric(errors="coerce")`` internally and returns ``None`` once fewer than
    ``min_rows`` (20) rows survive -- so it independently distinguishes a
    numeric-with-sentinel column (computed on the coercible subpopulation) from a
    genuinely categorical one (``None``). Gating on ``_is_numeric`` here on top of that
    only ever throws away the sentinel case, which is exactly why only 2 of 8 outcomes
    on the real cps_1970_1980 pair were getting a target R^2 at all -- the consumer's
    ``bidirectional_r2_blend`` (recalibrate.py) already expects and repairs this case,
    but never saw a target for it because the producer here silently returned None."""
    assoc = pairwise_associations(pool, cols)
    pairwise_assoc = {k: v[0] for k, v in assoc.items()}
    pairwise_method = {k: v[1] for k, v in assoc.items()}
    preds = [c for c in covariates if c in pool.columns]
    num_pred = frozenset(c for c in preds if _is_numeric(pool[c]))
    outcome_r2 = {
        y: (covariate_r2(pool, y, preds, numeric_predictors=num_pred)
            if y in pool.columns else None)
        for y in outcomes
    }
    resolved = [y for y, r2 in outcome_r2.items() if r2 is not None]
    unresolved = [y for y, r2 in outcome_r2.items() if r2 is None]
    _logger.info(
        "target_aggregates: outcome_r2 resolved for %d/%d outcomes (%s); "
        "None for %s",
        len(resolved), len(outcomes), resolved, unresolved,
    )
    return {
        "pairwise_assoc": pairwise_assoc,
        "pairwise_method": pairwise_method,
        "outcome_r2": outcome_r2,
        "provenance": {"source": "target_pool", "n_rows": int(len(pool)),
                       "reads": "marginals+pairwise_assoc+covariate_r2"},
    }
