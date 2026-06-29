from __future__ import annotations

import logging

import pandas as pd

from ssdataagent.data.loader import load_real_data
from ssdataagent.data.schema import DatasetSchema

log = logging.getLogger(__name__)

# target dataset name -> source dataset name (source = earlier wave).
TRANSFER_PAIRS: dict[str, str] = {"gss": "gss1994", "cps": "cps1970"}


def load_source_wave(source_name: str) -> pd.DataFrame:
    """Load a source wave's cleaned CSV as fitting microdata (full wave)."""
    return load_real_data(source_name)


def compute_crosswalk(
    target_schema: DatasetSchema,
    source_schema: DatasetSchema,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> list[str]:
    """Variables usable for transfer: (background + target) vars present in BOTH
    schemas AND as columns in BOTH frames, ordered by the target schema."""
    candidate = list(target_schema.background_variables) + list(target_schema.target_variables)
    src_vars = set(source_schema.background_variables) | set(source_schema.target_variables)
    common = [v for v in candidate
              if v in src_vars and v in source_df.columns and v in target_df.columns]
    dropped = [v for v in candidate if v not in common]
    log.info("crosswalk: %d common variables (dropped %d: %s)",
             len(common), len(dropped), dropped)
    return common
