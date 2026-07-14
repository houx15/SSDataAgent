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


class LeakageError(RuntimeError):
    """Raised when a training pool cannot be proven disjoint from the benchmark."""


def load_disjoint_train(name: str, *, n_sample: int, seed: int = 42) -> pd.DataFrame:
    """Draw `n_sample` training rows from the full source, provably disjoint from
    the benchmark sample.

    This is what lets us do two good things at once. The benchmark sample
    (`real_data_path`) is the paper's fixed reference — scoring against all of it
    keeps our numbers comparable to theirs. Training needs microdata that the
    reference never saw. Without a full source those two demands conflict, and the
    only way to satisfy both is to cut the reference in half and train on the other
    half — which is what we were doing, at the cost of both a starved model (cfps:
    400 fitting rows, several T3 responses with <50 observations) and a benchmark
    scored at half the paper's N.

    With a full source we can have both: the reference stays whole, and training
    draws from `full_source - reference`. The exclusion is enforced on the source's
    own unique row key, so disjointness is a proven property rather than an
    assumption. If we cannot prove it, we refuse rather than risk silently
    training on the rows we are about to be scored against.
    """
    schema = load_schema(name)
    if schema.full_source_path is None:
        raise LeakageError(
            f"dataset {name!r} has no full_source_path, so a disjoint training pool "
            "cannot be built; use the train/eval split instead"
        )
    key = schema.full_source_key
    if not key:
        raise LeakageError(
            f"dataset {name!r} declares no full_source_key; without a unique row key "
            "we cannot prove the training pool excludes the benchmark rows"
        )

    ref = pd.read_csv(schema.real_data_path, low_memory=False)
    full = pd.read_csv(schema.full_source_path, low_memory=False)
    full = full.loc[:, ~full.columns.str.match(r"^Unnamed: \d+$")]

    for frame, label in ((ref, "benchmark sample"), (full, "full source")):
        if key not in frame.columns:
            raise LeakageError(f"key {key!r} missing from {label} of {name!r}")
    if full[key].duplicated().any():
        raise LeakageError(
            f"key {key!r} is not unique in {name!r}'s full source; it cannot identify "
            "the benchmark rows for exclusion"
        )

    held_out = set(ref[key])
    pool = full[~full[key].isin(held_out)].reset_index(drop=True)
    if len(pool) + len(held_out & set(full[key])) != len(full):
        raise LeakageError(f"disjoint split of {name!r} did not partition the source")
    if pool.empty:
        raise LeakageError(f"disjoint pool for {name!r} is empty")

    n = min(n_sample, len(pool))
    out = pool.sample(n=n, random_state=seed).reset_index(drop=True)
    if set(out[key]) & held_out:  # belt and braces — the whole point of this function
        raise LeakageError(f"LEAKAGE: training draw for {name!r} contains benchmark rows")
    if "profile_id" not in out.columns:
        # SSDataBench's matched bootstrap bails out without profile_id; the raw
        # source lacks one (the paper's cleaning adds it). concat rather than
        # insert — these frames are ~770 columns wide and insert fragments them.
        ids = pd.DataFrame({"profile_id": [f"p{i}" for i in range(len(out))]})
        out = pd.concat([ids, out], axis=1)
    return out
