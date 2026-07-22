# tests/test_transfer_pairs.py
from __future__ import annotations

import pandas as pd

from ssdataagent.transfer.pairs import (
    PAIRS, TransferPair, crosswalk_columns, covariates_outcomes,
)


def test_pairs_registry_shape():
    ids = [p.id for p in PAIRS]
    assert ids == [
        "gss_1994_2018", "cps_1970_1980", "cps_1970_1990", "cps_1980_1990",
        "cps_1970_2000", "cps_1980_2000", "cps_1990_2000",
    ]
    scored = {p.id for p in PAIRS if p.scored}
    assert scored == {"gss_1994_2018", "cps_1970_1980"}
    for p in PAIRS:
        assert isinstance(p, TransferPair)
        assert (p.target_dataset is not None) == p.scored


def test_crosswalk_keeps_common_logs_dropped():
    # cps background/target vars include age, gender, race, education, income, ...
    src = pd.DataFrame({"age": [1], "gender": [1], "race": [1], "education": [1],
                        "income": [1], "birth_year": [1], "source_only": [1]})
    tgt = pd.DataFrame({"age": [1], "gender": [1], "race": [1], "education": [1],
                        "birth_year": [1], "target_only": [1]})  # no income -> dropped
    cols = crosswalk_columns("cps", src, tgt)
    assert "age" in cols and "gender" in cols and "education" in cols
    assert "income" not in cols          # target lacks it
    assert "source_only" not in cols and "target_only" not in cols
    # birth_year is a wave time-identity (birth_year = year - age), disjoint support
    # across waves -> non-transferable, dropped even though present in both frames.
    assert "birth_year" not in cols


def test_covariates_outcomes_split():
    cols = crosswalk_columns("cps",
                             pd.DataFrame({c: [1] for c in
                                           ["age", "gender", "race", "education", "income"]}),
                             pd.DataFrame({c: [1] for c in
                                           ["age", "gender", "race", "education", "income"]}))
    x, y = covariates_outcomes("cps", cols)
    assert "age" in x and "gender" in x          # background/demographic
    assert "income" in y                          # an outcome
    assert set(x).isdisjoint(set(y))
