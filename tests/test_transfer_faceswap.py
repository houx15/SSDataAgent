import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def _stub_elicit(ds, source_a, cols, **kw):
    from ssdataagent.transfer.generate import _is_numeric
    out = {}
    for c in cols:
        if _is_numeric(source_a[c]):
            v = pd.to_numeric(source_a[c], errors="coerce").dropna()
            out[c] = {"quantiles": [float(v.quantile(p)) for p in np.linspace(0, 1, 11)]}
        else:
            pr = source_a[c].dropna().astype(str).value_counts(normalize=True)
            out[c] = {"probs": {k: float(v) for k, v in pr.items()}}
    return out


def test_run_faceswap_scores_all_configs(tmp_path, monkeypatch):
    import transfer_faceswap
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "gss_1994_2018"][0]
    monkeypatch.setattr(transfer_faceswap, "OUT", tmp_path)
    monkeypatch.setattr(transfer_faceswap, "elicit_marginals", _stub_elicit)
    df = transfer_faceswap.run_faceswap(pair, seeds=1, n=200, bootstrap_B=20)
    assert set(df["config"]) == {"FS_carryover", "FS_llm", "ref_oracle_comp",
                                 "ref_floor", "ref_ceiling"}
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all() and (df[col] >= 0).all() and (df[col] <= 1).all()
    assert (tmp_path / "faceswap_gss_1994_2018.csv").exists()
