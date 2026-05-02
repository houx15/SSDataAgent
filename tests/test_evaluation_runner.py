from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ssdataagent.evaluation.runner import (
    PassRates,
    by_domain,
    parse_pass_rates,
    run_evaluation,
)
from ssdataagent.data.schema import DatasetSchema


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


def test_parse_type2_per_pair_and_per_variable(tmp_path):
    (tmp_path / "summary_type2.csv").write_text(
        "var1,var2,type1,type2,mode,insignificant_rate,key\n"
        "age,marital_status,numeric,categorical,strength,0.96,\n"
        "age,education,numeric,categorical,strength,0.80,\n"
        "gender,marital_status,categorical,categorical,strength,0.92,\n"
        ",,,,,0.89,avg\n"
    )
    rates = parse_pass_rates(tmp_path)
    assert len(rates.by_pair["type2"]) == 3
    # age appears in 2 pairs → mean(0.96, 0.80) = 0.88
    assert rates.by_variable["type2"]["age"] == pytest.approx(0.88)
    # marital_status in 2 pairs → mean(0.96, 0.92) = 0.94
    assert rates.by_variable["type2"]["marital_status"] == pytest.approx(0.94)
    # education in 1 pair
    assert rates.by_variable["type2"]["education"] == pytest.approx(0.80)


def test_parse_type3_per_response(tmp_path):
    (tmp_path / "summary_type3.csv").write_text(
        "response,model_type,mode,insignificant_rate,iterations,pass_count,fit_success,fit_fail,key\n"
        "age_first_childbirth,ols,strength,0.04,100,4,100,0,\n"
        "child_number,ols,strength,0.96,100,96,100,0,\n"
        ",,strength,0.5,,,,,avg_strength\n"
        ",,overall,0.5,,,,,avg_all\n"
    )
    rates = parse_pass_rates(tmp_path)
    assert rates.by_variable["type3"]["age_first_childbirth"] == pytest.approx(0.04)
    assert rates.by_variable["type3"]["child_number"] == pytest.approx(0.96)
    assert "avg_strength" not in rates.by_variable["type3"]


def test_by_domain_aggregates():
    schema = DatasetSchema(
        name="x",
        real_data_path=Path("/dev/null"),  # nosec - placeholder
        background_variables=[],
        target_variables=[],
        descriptions={},
        allowed_values={},
        numeric_ranges={},
        population_context="",
        ssdatabench_sim_subdir="x",
        evaluation_script="x.py",
        domains={"age": "Demography", "marital_status": "Marriage", "income": "SES"},
    )
    rates = PassRates(
        by_variable={
            "type1": {"age": 0.9, "marital_status": 0.8, "income": 0.5, "novel_var": 0.7},
        }
    )
    dom = by_domain(rates, schema)
    assert dom["type1"]["Demography"] == pytest.approx(0.9)
    assert dom["type1"]["Marriage"] == pytest.approx(0.8)
    assert dom["type1"]["SES"] == pytest.approx(0.5)
    assert dom["type1"]["Other"] == pytest.approx(0.7)


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
