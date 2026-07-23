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
