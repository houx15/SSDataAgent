from __future__ import annotations

import pandas as pd

from ssdataagent.evaluation.runner import PassRates


def to_long_table(rates: dict[tuple[str, str], PassRates]) -> pd.DataFrame:
    rows = []
    for (cond, dataset), r in rates.items():
        for type_name, pr in r.by_type.items():
            rows.append(
                {"condition": cond, "dataset": dataset, "type": type_name, "pass_rate": pr}
            )
    return pd.DataFrame(rows, columns=["condition", "dataset", "type", "pass_rate"])


def summary_pivot(long: pd.DataFrame) -> pd.DataFrame:
    return long.pivot_table(
        index="condition", columns="dataset", values="pass_rate", aggfunc="mean"
    )
