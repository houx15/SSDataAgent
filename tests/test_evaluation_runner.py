from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ssdataagent.evaluation.runner import (
    PassRates,
    parse_pass_rates,
    run_evaluation,
)


def test_parse_overall_summary(tmp_path):
    (tmp_path / "overall_insignificant_summary.csv").write_text(
        "type,avg_insignificant_rate,summary_path\n"
        "type1,0.96,foo.csv\n"
        "type2,0.85,bar.csv\n"
        "Overall Average,0.905,\n"
    )
    rates = parse_pass_rates(tmp_path)
    assert rates.by_type["type1"] == pytest.approx(0.96)
    assert rates.by_type["type2"] == pytest.approx(0.85)
    assert "Overall Average" not in rates.by_type
    assert rates.overall_average == pytest.approx(0.905)


def test_parse_per_variable(tmp_path):
    (tmp_path / "overall_insignificant_summary.csv").write_text(
        "type,avg_insignificant_rate,summary_path\n"
        "type1,0.96,summary_type1.csv\n"
        "Overall Average,0.96,\n"
    )
    (tmp_path / "summary_type1.csv").write_text(
        "variable,type,insignificant_rate,key\n"
        "gender,categorical,0.93,\n"
        "age,numeric,0.99,\n"
        ",,0.96,avg\n"
    )
    rates = parse_pass_rates(tmp_path)
    assert rates.by_variable["type1"]["gender"] == pytest.approx(0.93)
    assert rates.by_variable["type1"]["age"] == pytest.approx(0.99)
    assert "avg" not in rates.by_variable["type1"]


def test_parse_missing_summary(tmp_path):
    rates = parse_pass_rates(tmp_path)
    assert rates.by_type == {}
    assert rates.overall_average is None


def test_run_evaluation_invokes_subprocess(tmp_path):
    df = pd.DataFrame({"gender": ["Male"] * 10, "age": [30] * 10, "profile_id": range(10)})
    sampled = df.copy()
    fake_root = tmp_path / "ssdb"
    eval_results = fake_root / "evaluation_results" / "gss_2018" / "agent_x"
    eval_results.mkdir(parents=True)
    (eval_results / "overall_insignificant_summary.csv").write_text(
        "type,avg_insignificant_rate,summary_path\n"
        "type1,0.7,summary_type1.csv\n"
        "Overall Average,0.7,\n"
    )

    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        rates = run_evaluation(
            dataset_name="gss",
            run_id="x",
            generated=df,
            sampled=sampled,
            ssdatabench_root=fake_root,
        )
    assert run.called
    assert isinstance(rates, PassRates)
    assert rates.by_type["type1"] == pytest.approx(0.7)
