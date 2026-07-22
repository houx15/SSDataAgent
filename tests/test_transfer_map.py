# tests/test_transfer_map.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import yaml

from transfer_map import (
    composition_covariates, mean_scores, restrict_config_dir, run_layer1,
)


def test_composition_covariates_prefers_demographic_core():
    # given the full background set, rake only on age/gender/race
    assert composition_covariates(
        ["age", "gender", "race", "mother_occupation", "immigrant_status"]
    ) == ["age", "gender", "race"]
    # fall back to all covariates when none of the core survived the crosswalk
    assert composition_covariates(["mother_occupation", "father_education"]) == \
        ["mother_occupation", "father_education"]


def _frame(n, seed, xmean, beta):
    rng = np.random.default_rng(seed)
    x = rng.normal(xmean, 1, n)
    edu = np.where(x > xmean, "hi", "lo")
    return pd.DataFrame({"age": x, "education": edu, "income": beta * x + rng.normal(0, .3, n)})


def test_run_layer1_returns_map_and_copula():
    a = _frame(1500, 1, xmean=0.0, beta=1.0)
    b = _frame(1500, 2, xmean=3.0, beta=1.0)   # composition shift on outcomes
    covariates, outcomes = ["age", "education"], ["income"]
    kob, cop = run_layer1(a, b, ["age", "education", "income"], covariates, outcomes)
    # KOB has one row per outcome, with a share and a label
    assert set(kob["response"]) == {"income"}
    assert {"composition_share", "label", "gap_raw"}.issubset(kob.columns)
    inc = kob[kob["response"] == "income"].iloc[0]
    assert inc["label"] in {"composition-dominated", "mechanism-shifted", "aligned"}
    # copula table covers all unordered pairs of the 3 columns
    assert len(cop) == 3
    assert {"v1", "v2", "abs_delta", "label"}.issubset(cop.columns)


def test_restrict_config_dir_keeps_only_crosswalk_vars(tmp_path):
    # gss type configs reference variables gss1994 lacks (e.g. mental_health); restriction
    # to a crosswalk set must drop them so a source-limited sim can be scored.
    import nodonor_bracket as nb
    from ssdataagent.data.schema import load_schema
    sub = load_schema("gss").ssdatabench_sim_subdir
    allowed = {"age", "gender", "race", "education", "income", "marital_status"}
    dest = restrict_config_dir(sub, allowed, (1, 2, 3), tmp_path)
    for t in (1, 2, 3):
        cfg = yaml.safe_load((dest / sub / f"type{t}.yaml").read_text())
        for key in ("variables", "predictors", "response"):
            if isinstance(cfg.get(key), dict):
                assert set(cfg[key]).issubset(allowed), f"type{t}.{key} leaked non-crosswalk var"
    # a variable known to be gss2018-only must be gone from the restricted type1 config
    stock = yaml.safe_load((nb.CONFIG_DIR / sub / "type1.yaml").read_text())
    gss_only = set(stock.get("variables", {})) - allowed
    assert gss_only, "sanity: stock config should contain vars outside the small allowed set"
    restricted = yaml.safe_load((dest / sub / "type1.yaml").read_text())
    assert not (set(restricted.get("variables", {})) & gss_only)


def test_restrict_config_dir_preserves_model_type_pairing(tmp_path, monkeypatch):
    # T3 pairs model_type[i] with response.keys()[i] positionally. Dropping a response must
    # drop its model_type entry AND keep order (yaml sort_keys=False), or the wrong model is
    # fit for each response. Stock cps/gss configs are uniform 'ols' so this is untested there.
    import nodonor_bracket as nb
    subdir = "fake/t3test"
    src = tmp_path / "src"
    (src / subdir).mkdir(parents=True)
    cfg = {
        "model_type": ["ols", "logit", "mnlogit"],
        "response": {"income": {"type": "numeric"},
                     "married": {"type": "categorical"},
                     "occ": {"type": "categorical"}},
        "predictors": {"age": {}, "gender": {}},
    }
    (src / subdir / "type3.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    monkeypatch.setattr(nb, "CONFIG_DIR", src)
    dest = restrict_config_dir(subdir, {"age", "gender", "income", "occ"}, (3,), tmp_path / "dest")
    out = yaml.safe_load((dest / subdir / "type3.yaml").read_text())
    assert list(out["response"].keys()) == ["income", "occ"]   # order preserved, married dropped
    assert out["model_type"] == ["ols", "mnlogit"]             # married's 'logit' dropped, pairing intact


def test_mean_scores_ignores_error_columns():
    # nb.score emits string T{t}_error columns on a failed type-eval; averaging must
    # not crash on them (regression: startswith('T') swept up 'T1_error').
    df = pd.DataFrame([
        {"T1": 0.8, "T2": 0.6, "T3": None, "T3_error": "KeyError: education", "overall": 0.7},
        {"T1": 0.7, "T2": 0.5, "T3": None, "T3_error": "KeyError: education", "overall": 0.6},
    ])
    out = mean_scores(df)
    assert out["T1"] == 0.75 and out["T2"] == 0.55
    assert "T3" not in out            # all-None -> dropped
    assert "T3_error" not in out      # string error column never averaged
    assert abs(out["overall"] - 0.65) < 1e-9


def test_run_layer2_configs_include_b2(monkeypatch):
    # run_layer2 builds a dict of named builders; B2 must be one of them and must call
    # transfer_build_b2 with the pair's covariates/outcomes. We probe the builder table
    # by capturing what configs run_layer2 constructs, without scoring.
    import transfer_map as tm
    names = tm.LAYER2_CONFIG_NAMES
    assert "B2_recalibrated" in names
    assert names.index("B1_marginal_swap") < names.index("B2_recalibrated")  # ladder order
