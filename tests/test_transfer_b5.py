import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_corpus_contexts_include_cps_gss_waves():
    import transfer_b5
    ctx = transfer_b5.corpus_contexts()
    ids = {c[0] for c in ctx}
    # all four cps waves and both gss waves are present as contexts
    assert {"cps_1970", "cps_1980", "cps_1990", "cps_2000"} <= ids
    assert {"gss1994", "gss2018"} <= ids
    for _, csv, _ in ctx:
        assert csv.exists()


def test_context_records_are_wellformed():
    import transfer_b5
    rows = transfer_b5.context_records("cps_1980",
                                       REPO / "real_data" / "cps" / "cps-asec1980.csv",
                                       "cps")
    assert rows, "cps_1980 must resolve at least one outcome R^2"
    for r in rows:
        assert {"entropy", "n_predictors", "is_numeric", "true_r2", "context", "outcome"} <= set(r)
        assert 0.0 <= r["true_r2"] <= 1.0 or r["true_r2"] != r["true_r2"]  # in [0,1] or NaN-guarded


def test_training_rows_exclude_target():
    import transfer_b5
    all_rows = transfer_b5.training_rows(exclude_context_ids=set())
    held = transfer_b5.training_rows(exclude_context_ids={"gss2018"})
    assert all(r["context"] != "gss2018" for r in held)
    assert len(held) < len(all_rows)


def test_noise_points_from_cps_are_positive():
    import transfer_b5
    pts = transfer_b5.noise_points(exclude_csv=REPO / "real_data" / "gss" / "gss2018.csv")
    assert len(pts) >= 2                    # >=2 cps pseudo-targets -> a real curve
    for ess, sq in pts:
        assert 0.0 < ess <= 1.0 and sq >= 0.0


def test_cps_pseudo_targets_exclude_the_real_target():
    import transfer_b5
    target = REPO / "real_data" / "cps" / "cps-asec1980.csv"
    pairs = transfer_b5._cps_pseudo_targets(target)
    tgt = target.resolve()
    assert pairs, "cps has pseudo-targets"
    # the real target is never a pseudo-target and never a sibling
    for w, sibs in pairs:
        assert w.resolve() != tgt
        assert all(s.resolve() != tgt for s in sibs)
        assert sibs, "each pseudo-target keeps >=1 sibling after LOCO"
    # excluding a real target strictly shrinks the sibling universe vs excluding nothing-in-cps
    gss = REPO / "real_data" / "gss" / "gss2018.csv"
    n_target = sum(len(s) for _, s in pairs)
    n_gss = sum(len(s) for _, s in transfer_b5._cps_pseudo_targets(gss))
    assert n_target < n_gss


def test_run_b5_smoke_scores_both_configs(tmp_path, monkeypatch):
    import transfer_b5
    from ssdataagent.transfer.pairs import PAIRS
    monkeypatch.setattr(transfer_b5, "OUT", tmp_path)
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    df = transfer_b5.run_b5(pair, seeds=2, n=800, bootstrap_B=50)
    assert list(df["config"]) == ["B5_learned", "B5_prior_only"]
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all()
        assert (df[col] >= 0).all() and (df[col] <= 1).all()
    assert "ess_ratio" in df.columns
    assert (tmp_path / "b5_cps_1970_1980.csv").exists()


def test_predict_target_r2_shapes(tmp_path):
    import transfer_b5
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "cps_1970_1980"][0]
    learned, prior_only, ess, sib_rew = transfer_b5.predict_target_r2(pair)
    assert set(learned) == set(prior_only)          # same outcome keys
    assert len(sib_rew) > 0                          # structure vehicle materialized
    assert learned                                   # non-empty
    assert 0.0 < ess <= 1.0
    for d in (learned, prior_only):
        for v in d.values():
            assert 0.0 <= v <= 1.0
