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
