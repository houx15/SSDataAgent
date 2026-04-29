from __future__ import annotations

import numpy as np
import pandas as pd


def split_train_eval(
    df: pd.DataFrame, ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < ratio < 1:
        raise ValueError(f"ratio must be in (0, 1); got {ratio}")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(round(len(df) * ratio))
    return df.iloc[idx[:n_train]].copy(), df.iloc[idx[n_train:]].copy()
