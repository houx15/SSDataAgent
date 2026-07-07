from __future__ import annotations

import pandas as pd

from ssdataagent.data.schema import load_schema


def load_real_data(name: str, *, n_sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load a dataset's real microdata.

    By default reads the fixed paper-aligned sample (`real_data_path`). When
    ``n_sample`` is set, draws that many rows from the dataset's larger
    ``full_source_path`` instead — used to give the sparse life-event-timing
    subset enough rows to stabilize the T4/T5 event-order tests.
    """
    schema = load_schema(name)
    if n_sample is not None:
        if schema.full_source_path is None:
            raise ValueError(
                f"dataset {name!r} has no full_source_path; cannot draw n_sample={n_sample}"
            )
        df = pd.read_csv(schema.full_source_path, low_memory=False)
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]
        n = min(n_sample, len(df))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        # The fixed paper sample ships a `profile_id` (added during cleaning) that
        # SSDataBench's matched-bootstrap eval requires; the raw source lacks it.
        if "profile_id" not in df.columns:
            df.insert(0, "profile_id", [f"p{i}" for i in range(len(df))])
        return df
    df = pd.read_csv(schema.real_data_path)
    return df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]
