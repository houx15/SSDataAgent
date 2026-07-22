# src/ssdataagent/transfer/pairs.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ssdataagent.config import data_root
from ssdataagent.data.schema import load_schema

log = logging.getLogger(__name__)

# Wave time-identities: mechanically tied to the survey year (birth_year = year - age),
# so their support is disjoint across waves and they carry no transferable mechanism.
# Dropped from every transfer crosswalk (documented, like a data_audit trap).
NON_TRANSFERABLE = frozenset({"birth_year"})


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]


@dataclass(frozen=True)
class TransferPair:
    id: str
    source_csv: Path
    target_csv: Path
    schema_name: str        # "gss" | "cps": drives X/Y split + scoring config
    scored: bool             # True only for benchmark-backed targets
    target_dataset: str | None  # ds name passed to score() when scored


def _cps(name: str) -> Path:
    return data_root() / "cps" / name


def _gss(name: str) -> Path:
    return data_root() / "gss" / name


PAIRS: list[TransferPair] = [
    TransferPair("gss_1994_2018", _gss("gss1994.csv"), _gss("gss2018.csv"), "gss", True, "gss"),
    TransferPair("cps_1970_1980", _cps("cps-asec1970.csv"), _cps("cps-asec1980.csv"), "cps", True, "cps"),
    TransferPair("cps_1970_1990", _cps("cps-asec1970.csv"), _cps("cps-asec1990.csv"), "cps", False, None),
    TransferPair("cps_1980_1990", _cps("cps-asec1980.csv"), _cps("cps-asec1990.csv"), "cps", False, None),
    TransferPair("cps_1970_2000", _cps("cps-asec1970.csv"), _cps("cps-asec2000.csv"), "cps", False, None),
    TransferPair("cps_1980_2000", _cps("cps-asec1980.csv"), _cps("cps-asec2000.csv"), "cps", False, None),
    TransferPair("cps_1990_2000", _cps("cps-asec1990.csv"), _cps("cps-asec2000.csv"), "cps", False, None),
]


def crosswalk_columns(schema_name: str, source_df: pd.DataFrame,
                      target_df: pd.DataFrame) -> list[str]:
    """Background+target vars present as columns in BOTH frames, ordered by schema."""
    schema = load_schema(schema_name)
    candidate = list(schema.background_variables) + list(schema.target_variables)
    common = [v for v in candidate if v not in NON_TRANSFERABLE
              and v in source_df.columns and v in target_df.columns]
    dropped_identity = [v for v in candidate if v in NON_TRANSFERABLE]
    dropped_missing = [v for v in candidate
                       if v not in NON_TRANSFERABLE and v not in common]
    log.info("crosswalk[%s]: %d common; dropped %d not-in-both %s; "
             "dropped %d non-transferable identity %s",
             schema_name, len(common), len(dropped_missing), dropped_missing,
             len(dropped_identity), dropped_identity)
    return common


def covariates_outcomes(schema_name: str, cols: list[str]) -> tuple[list[str], list[str]]:
    schema = load_schema(schema_name)
    bg = set(schema.background_variables)
    x = [c for c in cols if c in bg]
    y = [c for c in cols if c not in bg]
    return x, y


def load_pair(pair: TransferPair) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    src = _drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    tgt = _drop_unnamed(pd.read_csv(pair.target_csv, low_memory=False))
    cols = crosswalk_columns(pair.schema_name, src, tgt)
    return src, tgt, cols
