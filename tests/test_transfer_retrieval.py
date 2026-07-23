import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO)]


def test_sibling_csvs_loco_excludes_target():
    from ssdataagent.transfer.pairs import PAIRS
    from ssdataagent.transfer import retrieval
    cps = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    sibs = retrieval.sibling_csvs(cps)
    names = {p.name for p in sibs}
    assert cps.target_csv.name not in names          # LOCO: target held out
    assert "cps-asec1970.csv" in names                # designated source is a valid sibling
    assert len(sibs) == 3                             # 1970, 1990, 2000
    gss = [p for p in PAIRS if p.id == "gss_1994_2018"][0]
    assert len(retrieval.sibling_csvs(gss)) == 1       # degenerate: only 1994


def test_reweighted_pool_matches_target_margin_and_reports_ess():
    from ssdataagent.transfer import retrieval
    rng = np.random.default_rng(0)
    # sibling stack skews YOUNG; target skews OLD. After raking on age the resampled
    # pool's age composition must move toward the target's.
    sib = pd.DataFrame({"age": np.r_[np.full(800, 25), np.full(200, 65)],
                        "income": rng.normal(0, 1, 1000)})
    target = pd.DataFrame({"age": np.r_[np.full(200, 25), np.full(800, 65)],
                           "income": rng.normal(0, 1, 1000)})
    sib_rew, ess = retrieval.reweighted_pool([sib], target, ["age", "income"], ["age"],
                                             n=4000, rng=rng)
    assert len(sib_rew) == 4000
    frac_old = (sib_rew["age"] == 65).mean()
    assert frac_old > 0.6                              # raked toward target's 0.8, up from 0.2
    assert 0.0 < ess <= 1.0                             # Kish ratio is a reliability signal


def test_reweighted_pool_is_deterministic():
    from ssdataagent.transfer import retrieval
    sib = pd.DataFrame({"age": [25, 65, 25, 65] * 50, "income": list(range(200))})
    target = pd.DataFrame({"age": [65] * 100, "income": list(range(100))})
    a, _ = retrieval.reweighted_pool([sib], target, ["age", "income"], ["age"], 300,
                                     np.random.default_rng(7))
    b, _ = retrieval.reweighted_pool([sib], target, ["age", "income"], ["age"], 300,
                                     np.random.default_rng(7))
    assert a.equals(b)
