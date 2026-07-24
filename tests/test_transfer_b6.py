import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def _stub_predict(pair, *, n_siblings, ess):
    """Build a deterministic (learned, prior_only, ess, sib_rew, n_siblings) stub
    from the pair's real columns, with sib_rew = the carved target pool so the
    downstream draw/score run without the heavy LOCO fit."""
    import transfer_b6
    import nodonor_bracket as nb
    ds, cols, covs, outs = transfer_b6.b4_columns(pair)
    target_pool, _ = nb.carve_pool(ds)
    learned = {o: 0.60 for o in outs}          # retrieval-moved
    prior_only = {o: 0.25 for o in outs}       # distinct -> provenance is 'retrieval-blend'
    return learned, prior_only, float(ess), target_pool[cols].copy(), int(n_siblings)


def test_run_b6_gate_pass_uses_retrieval(tmp_path, monkeypatch):
    import transfer_b6
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    monkeypatch.setattr(transfer_b6, "OUT", tmp_path)
    monkeypatch.setattr(transfer_b6, "predict_target_r2",
                        lambda p: _stub_predict(p, n_siblings=3, ess=0.65))
    df = transfer_b6.run_b6(pair, seeds=1, n=200, bootstrap_B=20)
    assert list(df["config"]) == ["B6_hybrid"]
    assert df.iloc[0]["source"] == "retrieval"
    assert int(df.iloc[0]["n_siblings"]) == 3
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all()
    assert (tmp_path / "b6_cps_1970_1980.csv").exists()


def test_run_b6_gate_fail_uses_prior(tmp_path, monkeypatch):
    import transfer_b6
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "gss_1994_2018"][0]
    monkeypatch.setattr(transfer_b6, "OUT", tmp_path)
    monkeypatch.setattr(transfer_b6, "predict_target_r2",
                        lambda p: _stub_predict(p, n_siblings=1, ess=0.10))
    df = transfer_b6.run_b6(pair, seeds=1, n=200, bootstrap_B=20)
    assert df.iloc[0]["source"] == "prior"
    assert int(df.iloc[0]["n_siblings"]) == 1
