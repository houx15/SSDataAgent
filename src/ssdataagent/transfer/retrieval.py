from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ssdataagent.transfer.decompose import effective_sample_size, raking_weights


def sibling_csvs(pair) -> list[Path]:
    """Same-instrument sibling CSVs under leave-one-context-out: every ``*.csv`` in the
    target wave's dataset directory except the target wave itself. The designated source
    wave IS a sibling (only the target is held out). Sorted for determinism."""
    tgt = pair.target_csv.resolve()
    return sorted(p for p in pair.target_csv.parent.glob("*.csv")
                  if p.resolve() != tgt)


def reweighted_pool(sib_frames: list[pd.DataFrame], target_pool: pd.DataFrame,
                    cols: list[str], rake_cols: list[str], n: int,
                    rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Concatenate siblings on ``cols``, rake to ``target_pool``'s ``rake_cols`` marginals
    (IPF, the KOB composition transport), and draw a weighted resample of ``n`` rows.

    Returns ``(sib_rew, ess_ratio)``. Raking simultaneously corrects each sibling's
    composition toward the target and pools siblings (a composition-nearer sibling gets
    more weight). ``ess_ratio`` is the Kish effective sample size over the stack size: a
    low value means the raking concentrated weight on a few rows -- a thin transport the
    caller must surface. Reads only the target's ``rake_cols`` margins (public X-margins);
    never the target's joint."""
    stack = pd.concat([f[cols] for f in sib_frames], ignore_index=True)
    w = raking_weights(stack, target_pool, rake_cols)
    ess_ratio = effective_sample_size(w) / len(w) if len(w) else 0.0
    idx = rng.choice(len(stack), size=n, replace=True, p=w / w.sum())
    return stack.iloc[idx].reset_index(drop=True), ess_ratio
