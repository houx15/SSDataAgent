import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_synth_numeric_matches_quantiles():
    from ssdataagent.transfer.blind import _synth_numeric
    col = _synth_numeric([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], L=5000)
    assert len(col) == 5000
    assert abs(np.quantile(col, 0.5) - 5.0) < 0.2
    assert col.min() >= -0.1 and col.max() <= 10.1


def test_synth_categorical_matches_probs_and_length():
    from ssdataagent.transfer.blind import _synth_categorical
    col = _synth_categorical({"a": 0.5, "b": 0.3, "c": 0.2}, L=1000)
    assert len(col) == 1000
    vc = pd.Series(col).value_counts(normalize=True)
    assert abs(vc["a"] - 0.5) < 0.01 and abs(vc["b"] - 0.3) < 0.01


def test_build_marg_frame_uses_elicited_and_carries_A_missingness():
    from ssdataagent.transfer.blind import build_marg_frame
    a = pd.DataFrame({
        "age": [20, 30, 40, 50, np.nan, 60, 70, 80, 25, 35],   # numeric, 10% missing
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],  # categorical, 0% missing
    })
    elicited = {
        "age": {"quantiles": [18, 22, 30, 40, 50, 60, 65, 70, 75, 80, 90]},
        "sex": {"probs": {"M": 0.7, "F": 0.3}},
    }
    frame = build_marg_frame(elicited, a, ["age", "sex"], L=2000, seed=0)
    # elicited proportions win for sex
    vc = frame["sex"].dropna().astype(str).value_counts(normalize=True)
    assert abs(vc["M"] - 0.7) < 0.02
    # A's missingness RATE is carried (age ~10%, sex ~0%)
    assert abs(frame["age"].isna().mean() - 0.1) < 0.02
    assert frame["sex"].isna().mean() < 0.001
    # elicited numeric level wins (median ~60 from the quantiles, not A's ~40)
    assert abs(np.nanmedian(pd.to_numeric(frame["age"])) - 60) < 5


def test_build_marg_frame_falls_back_to_A_when_missing():
    from ssdataagent.transfer.blind import build_marg_frame
    a = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    frame = build_marg_frame({}, a, ["x"], L=1000, seed=0)   # nothing elicited -> carry A
    assert len(frame) == 1000
    assert 1 <= np.nanmedian(pd.to_numeric(frame["x"])) <= 10


def test_build_marg_frame_empty_probs_falls_back_not_crash():
    from ssdataagent.transfer.blind import build_marg_frame
    a = pd.DataFrame({"age": [20, 30, 40, 50, 60],
                      "sex": ["M", "F", "M", "F", "M"]})
    # sex has an empty probs dict -> must fall back to A's marginal, not crash
    frame = build_marg_frame({"sex": {"probs": {}}}, a, ["age", "sex"], L=500, seed=0)
    assert len(frame) == 500
    assert set(frame["sex"].dropna().astype(str).unique()) <= {"M", "F"}
    assert frame["sex"].notna().any()


class _StubMsg:
    def __init__(self, content): self.message = type("M", (), {"content": content})
class _StubResp:
    def __init__(self, content): self.choices = [_StubMsg(content)]
class _StubClient:
    def __init__(self, content): self._c = content; self.calls = 0
    @property
    def chat(self):
        outer = self
        class _Chat:
            class completions:
                @staticmethod
                def create(**kw):
                    outer.calls += 1
                    return _StubResp(outer._c)
        return _Chat()


def test_elicit_marginals_parses_and_caches(tmp_path):
    from ssdataagent.transfer.blind import elicit_marginals
    a = pd.DataFrame({"age": [20, 30, 40, 50, 60], "sex": ["M", "F", "M", "F", "M"]})
    payload = ('{"age": {"quantiles": [18,22,30,40,50,60,65,70,75,80,90]}, '
               '"sex": {"probs": {"M": 0.7, "F": 0.3}}}')
    client = _StubClient(payload)
    got = elicit_marginals("gss", a, ["age", "sex"], client=client,
                           cache_dir=tmp_path, regenerate=True)
    assert set(got) == {"age", "sex"}
    assert abs(got["sex"]["probs"]["M"] - 0.7) < 1e-9
    assert len(got["age"]["quantiles"]) == 11
    assert (tmp_path / "gss_marginals.json").exists()
    assert client.calls == 1
    # second call with the cache warm must NOT hit the client
    client2 = _StubClient(payload)
    again = elicit_marginals("gss", a, ["age", "sex"], client=client2, cache_dir=tmp_path)
    assert client2.calls == 0 and again["sex"]["probs"]["M"] == got["sex"]["probs"]["M"]


def test_elicit_prompt_lists_categories_from_A_not_B():
    from ssdataagent.transfer.blind import elicit_prompt
    a = pd.DataFrame({"age": [20, 30, 40], "sex": ["M", "F", "M"]})
    p = elicit_prompt("gss", a, ["age", "sex"])
    assert "quantiles" in p and "probs" in p
    assert "M" in p and "F" in p            # category universe surfaced from A


def test_parse_marginals_drops_malformed_values():
    from ssdataagent.transfer.blind import parse_marginals
    a = pd.DataFrame({"age": [20, 30, 40], "sex": ["M", "F", "M"],
                      "edu": ["hs", "col", "hs"]})
    text = ('{"age": {"quantiles": [1, "N/A", 3]}, '        # bad numeric value -> drop
            '"sex": {"probs": {"M": "lots", "F": 0.3}}, '    # bad prob value -> drop
            '"edu": {"probs": {"hs": 0.6, "col": 0.4}}}')    # well-formed -> kept
    got = parse_marginals(text, a, ["age", "sex", "edu"])
    assert set(got) == {"edu"}
    assert got["edu"]["probs"]["hs"] == 0.6


def test_parse_marginals_restricts_categorical_to_A_universe():
    from ssdataagent.transfer.blind import parse_marginals
    a = pd.DataFrame({"marital": ["Married", "Single", "Separated-Divorced-Widowed",
                                  "Married", "Single"]})
    # LLM drifted the label ("Divorced/widowed" not in A) -> that key is dropped,
    # the in-universe keys are kept (renormalized downstream by _synth_categorical).
    text = ('{"marital": {"probs": {"Married": 0.6, "Single": 0.25, '
            '"Divorced/widowed": 0.15}}}')
    got = parse_marginals(text, a, ["marital"])
    assert set(got["marital"]["probs"]) == {"Married", "Single"}
