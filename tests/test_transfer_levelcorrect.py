import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]

import numpy as np
import pandas as pd


def test_numeric_outcomes_picks_numeric_in_both():
    from ssdataagent.transfer.levelcorrect import numeric_outcomes
    a = pd.DataFrame({"inc": [1, 2, 3], "occ": ["x", "y", "z"], "kid": [0, 1, 2]})
    b = pd.DataFrame({"inc": [4, 5, 6], "occ": ["p", "q", "r"], "kid": [1, 2, 3]})
    assert numeric_outcomes(a, b, ["inc", "occ", "kid"]) == ["inc", "kid"]


def test_oracle_and_pooled_shifts_are_mean_diffs():
    from ssdataagent.transfer.levelcorrect import oracle_shifts, pooled_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0, 0.0, 0.0]})
    b = pd.DataFrame({"inc": [5.0, 5.0, 5.0, 5.0]})
    assert abs(oracle_shifts(a, b, ["inc"])["inc"] - 5.0) < 1e-9
    assert abs(pooled_shifts(a, b, ["inc"])["inc"] - 5.0) < 1e-9


def test_apply_level_shift_moves_mean_preserves_shape_and_others():
    from ssdataagent.transfer.levelcorrect import apply_level_shift, outcome_mean
    a = pd.DataFrame({"inc": [1.0, 2.0, 3.0, np.nan], "occ": ["x", "y", "z", "w"]})
    out = apply_level_shift(a, {"inc": 10.0})
    assert abs(outcome_mean(out, "inc") - (outcome_mean(a, "inc") + 10.0)) < 1e-9
    assert abs(pd.to_numeric(out["inc"]).var() - pd.to_numeric(a["inc"]).var()) < 1e-9
    assert out["inc"].isna().sum() == 1            # NaN preserved
    assert list(out["occ"]) == list(a["occ"])      # other columns untouched


def test_apply_level_shift_zero_nonfinite_and_missing_are_noops():
    from ssdataagent.transfer.levelcorrect import apply_level_shift
    a = pd.DataFrame({"inc": [1.0, 2.0]})
    assert list(apply_level_shift(a, {"inc": 0.0})["inc"]) == [1.0, 2.0]
    assert list(apply_level_shift(a, {"inc": float("nan")})["inc"]) == [1.0, 2.0]
    assert list(apply_level_shift(a, {"missing": 5.0}).columns) == ["inc"]


def test_hybrid_shifts_gate_routes_pooled_vs_llm():
    from ssdataagent.transfer.levelcorrect import hybrid_shifts
    pooled, llm = {"inc": 3.0}, {"inc": 7.0}
    assert hybrid_shifts(pooled, llm, n_siblings=3, ess=0.6)["inc"] == 3.0   # plural+effective
    assert hybrid_shifts(pooled, llm, n_siblings=1, ess=0.6)["inc"] == 7.0   # thin -> llm
    # an outcome absent from the chosen arm falls back to the other
    assert hybrid_shifts({}, {"inc": 7.0}, n_siblings=3, ess=0.6)["inc"] == 7.0


class _FakeClient:
    """Minimal stand-in for the OpenRouter client: .chat.completions.create(...) returns an
    object whose choices[0].message.content is the canned text."""
    def __init__(self, content):
        self._content = content
    @property
    def chat(self):
        return self
    @property
    def completions(self):
        return self
    def create(self, model, messages):
        msg = type("M", (), {"content": self._content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_llm_shifts_uses_elicited_mean(tmp_path):
    from ssdataagent.transfer.levelcorrect import llm_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0, 0.0, 0.0], "kid": [2.0, 2.0, 2.0, 2.0]})
    client = _FakeClient('{"inc": 5, "kid": 3}')
    sh = llm_shifts(a, "cps", ["inc", "kid"], client=client, cache_dir=tmp_path)
    assert abs(sh["inc"] - 5.0) < 1e-9        # 5 - mean 0
    assert abs(sh["kid"] - 1.0) < 1e-9        # 3 - mean 2
    assert (tmp_path / "cps_levels.json").exists()


def test_llm_shifts_drops_junk_entries(tmp_path):
    from ssdataagent.transfer.levelcorrect import llm_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0]})
    sh = llm_shifts(a, "cps", ["inc"], client=_FakeClient('{"inc": "NaN-ish"}'), cache_dir=tmp_path)
    assert "inc" not in sh                    # dropped -> carryover for that outcome


def test_llm_shifts_cache_hit_skips_client(tmp_path):
    import json
    from ssdataagent.transfer.levelcorrect import llm_shifts
    (tmp_path / "cps_levels.json").write_text(json.dumps({"inc": 9.0}))
    a = pd.DataFrame({"inc": [1.0, 1.0]})

    class Boom:
        @property
        def chat(self):
            raise AssertionError("client must not be called on a cache hit")
    sh = llm_shifts(a, "cps", ["inc"], client=Boom(), cache_dir=tmp_path)
    assert abs(sh["inc"] - 8.0) < 1e-9        # 9 - mean 1, from cache


def test_assemble_shifts_wires_all_four_arms(tmp_path):
    from ssdataagent.transfer.levelcorrect import assemble_shifts
    a = pd.DataFrame({"inc": [0.0, 0.0, 0.0, 0.0]})
    b = pd.DataFrame({"inc": [10.0, 10.0, 10.0, 10.0]})          # oracle Δ = 10
    sib_rew = pd.DataFrame({"inc": [4.0, 4.0, 4.0, 4.0]})        # pooled Δ = 4
    shifts = assemble_shifts(a, b, sib_rew, "cps", ["inc"], n_siblings=3, ess=0.6,
                             client=_FakeClient('{"inc": 7}'), cache_dir=tmp_path)  # llm Δ = 7
    assert abs(shifts["oracle"]["inc"] - 10.0) < 1e-9
    assert abs(shifts["pooled"]["inc"] - 4.0) < 1e-9
    assert abs(shifts["llm"]["inc"] - 7.0) < 1e-9
    assert abs(shifts["hybrid"]["inc"] - 4.0) < 1e-9             # 3 sib, ess .6 -> pooled


def test_apply_affine_matches_mean_and_std():
    from ssdataagent.transfer.levelcorrect import apply_affine_shift
    a = pd.DataFrame({"inc": np.arange(0.0, 100.0) + 0.5})   # non-integer -> no count rounding
    out = apply_affine_shift(a, {"inc": (200.0, 10.0)})
    x = pd.to_numeric(out["inc"])
    assert abs(x.mean() - 200.0) < 1e-6
    assert abs(x.std() - 10.0) < 1e-6


def test_apply_affine_rounds_and_floors_counts():
    from ssdataagent.transfer.levelcorrect import apply_affine_shift
    a = pd.DataFrame({"kid": np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7], dtype=float)})
    out = apply_affine_shift(a, {"kid": (1.0, 2.0)})
    x = pd.to_numeric(out["kid"]).dropna()
    assert (x >= 0).all()                                  # floored at A's min 0
    assert np.allclose(x.to_numpy(), np.round(x.to_numpy()))  # kept whole


def test_apply_affine_preserves_nan_and_other_cols():
    from ssdataagent.transfer.levelcorrect import apply_affine_shift
    a = pd.DataFrame({"inc": [1.0, 2.0, 3.0, np.nan], "occ": ["x", "y", "z", "w"]})
    out = apply_affine_shift(a, {"inc": (10.0, 2.0)})
    assert out["inc"].isna().sum() == 1
    assert list(out["occ"]) == list(a["occ"])


def test_oracle_affine_returns_target_mean_and_std():
    from ssdataagent.transfer.levelcorrect import oracle_affine
    a = pd.DataFrame({"inc": [0.0, 0.0, 0.0, 0.0]})
    b = pd.DataFrame({"inc": [1.0, 2.0, 3.0, 4.0]})
    t = oracle_affine(a, b, ["inc"])
    assert abs(t["inc"][0] - 2.5) < 1e-9                       # mean_B
    assert abs(t["inc"][1] - pd.Series([1, 2, 3, 4]).std()) < 1e-9  # std_B (ddof=1)
