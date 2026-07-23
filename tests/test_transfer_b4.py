import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_b4_columns_match_layer2_cols():
    # B4 must score on the SAME crosswalk variables as B0-B3, or the comparison is invalid.
    import transfer_b4
    import pandas as pd
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema
    from ssdataagent.transfer.pairs import PAIRS, load_pair
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    _, cols_b4, _, _ = transfer_b4.b4_columns(pair)
    # replicate run_layer2's derivation
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    ref = nb._drop_unnamed(pd.read_csv(load_schema(pair.target_dataset).real_data_path,
                                       low_memory=False))
    b_pool, _ = nb.carve_pool(pair.target_dataset)
    _, _, cols = load_pair(pair)
    expected = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    assert cols_b4 == expected


def test_run_b4_smoke_scores_both_configs(tmp_path, monkeypatch):
    # Small but real off the CPS microdata (no API key). Both configs score, bounded.
    import transfer_b4
    from ssdataagent.transfer.pairs import PAIRS
    monkeypatch.setattr(transfer_b4, "OUT", tmp_path)
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    df = transfer_b4.run_b4(pair, seeds=2, n=800, bootstrap_B=50)
    assert list(df["config"]) == ["B4_retrieval", "B4_retrieval_targetR2"]
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all()
        assert (df[col] >= 0).all() and (df[col] <= 1).all()
    assert (tmp_path / "b4_cps_1970_1980.csv").exists()
    assert "ess_ratio" in df.columns and (df["ess_ratio"] > 0).all()
