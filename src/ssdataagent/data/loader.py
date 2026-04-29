from __future__ import annotations

import pandas as pd

from ssdataagent.data.schema import load_schema


def load_real_data(name: str) -> pd.DataFrame:
    schema = load_schema(name)
    return pd.read_csv(schema.real_data_path)
