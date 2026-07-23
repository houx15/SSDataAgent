import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_b3_columns_restricts_and_splits():
    import transfer_b3
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    ds, cols, spec, predictors, downstream = transfer_b3.b3_columns(pair)
    assert ds == "cps"
    assert "birth_year" not in cols                       # non-transferable, dropped
    assert set(predictors) <= set(cols)
    assert set(downstream) == set(cols) - set(spec.seeds) - set(spec.derived)
    for c in spec.seeds:
        assert c not in downstream


def test_rung_alphas_raw_is_all_ones_and_pool_uses_pool_r2():
    import transfer_b3
    from ssdataagent.transfer.b3_specs import SPECS
    rng = np.random.default_rng(0)
    n = 300
    # synthetic raw + pool with a real age->income signal so R^2 is estimable
    age = rng.integers(18, 80, n)
    income = 500 * age + rng.normal(0, 3000, n)
    frame = pd.DataFrame({
        "age": age, "gender": rng.choice(["Male", "Female"], n),
        "race": rng.choice(["Black", "Hispanic", "Non-Black, Non-Hispanic"], n),
        "education": rng.choice(["High school", "College and above"], n),
        "income": income,
    })
    spec = SPECS["cps"]
    predictors = ["age", "gender", "race", "education"]
    elicited = {"income": 0.5}
    alphas = transfer_b3.rung_alphas(frame, spec, elicited, frame, predictors)
    assert set(alphas) == {"B3_raw", "B3_elicited", "B3_pool_R2"}
    assert alphas["B3_raw"] == {"income": 1.0}            # raw = no repair
    assert alphas["B3_pool_R2"]["income"] > 0             # a real alpha from pool R^2
    # pool R^2 alpha differs from the elicited-0.5 alpha (different targets)
    assert alphas["B3_pool_R2"]["income"] != alphas["B3_elicited"]["income"]


import pytest


def _cps_cache_warm():
    cache = REPO / "results" / "nodonor_cache"
    return (cache / "cps_cond_raw.csv").exists() and (cache / "cps_elicit.json").exists()


@pytest.mark.skipif(not _cps_cache_warm(), reason="cps LLM cache not present")
def test_run_b3_cps_off_warm_cache(tmp_path, monkeypatch):
    import transfer_b3
    from ssdataagent.transfer.pairs import PAIRS
    # redirect the output CSV into tmp so the test never clobbers a real result
    monkeypatch.setattr(transfer_b3, "OUT", tmp_path)
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    # small but real: 2 seeds, n=800, cheap bootstrap -- still deterministic off the cache
    df = transfer_b3.run_b3(pair, seeds=2, n=800, bootstrap_B=50)
    assert list(df["config"]) == ["B3_raw", "B3_elicited", "B3_pool_R2"]
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all()
        assert (df[col] >= 0).all() and (df[col] <= 1).all()
    out = tmp_path / "b3_cps_1970_1980.csv"
    assert out.exists()
    # B3_pool_R2 should not score below B3_raw on T3 by more than noise (repair helps or ties)
    raw_t3 = float(df.loc[df.config == "B3_raw", "T3"].iloc[0])
    pool_t3 = float(df.loc[df.config == "B3_pool_R2", "T3"].iloc[0])
    assert pool_t3 >= raw_t3 - 0.1
