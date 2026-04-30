"""Smoke tests proving the SSDataBench evaluation pipeline can be invoked
end-to-end. These tests do NOT use the LLM — they verify the data plumbing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
SSDATABENCH = REPO / "ssdatabench"
sys.path.insert(0, str(REPO / "scripts"))
from build_eval_subset import build_subset  # noqa: E402


@pytest.mark.live_eval
def test_real_data_loadable():
    """The user-provided cleaned CSVs in real_data/ load and have the documented columns."""
    meta = json.loads((REPO / "real_data" / "dataset_meta.json").read_text())
    for entry in meta:
        csv = REPO / "real_data" / Path(entry["output"]).name
        df = pd.read_csv(csv)
        assert list(df.columns) == entry["columns"], f"{csv.name} column mismatch"
        assert len(df) == entry["rows"], f"{csv.name} expected {entry['rows']} rows"


_DATASET_PARAMS = [
    ("gss", "gss_clean.csv", "gss_2018"),
    ("cps", "cps_clean.csv", "cps_1980"),
    ("acs", "acs_clean.csv", "acs_1980"),
]


@pytest.mark.live_eval
@pytest.mark.parametrize("short_name,csv_name,subdir", _DATASET_PARAMS)
def test_evaluation_single_folder_runs(short_name, csv_name, subdir):
    """Shell out to ssdatabench's <dataset> evaluation in --single mode against a folder
    containing identical sampled_/sim_ files. Proves the pipeline executes; near-100%
    pass rate is the expected outcome since real==sim."""
    real_csv = REPO / "real_data" / csv_name
    assert real_csv.exists()
    build_subset(short_name)

    sim_root = SSDATABENCH / "simulated_data" / subdir / "agent_smoke"
    sim_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(real_csv, sim_root / "sampled_smoke.csv")
    shutil.copy(real_csv, sim_root / "sim_profiles_smoke.csv")
    output_base = SSDATABENCH / "evaluation_results" / subdir / "agent_smoke"

    try:
        result = subprocess.run(
            [
                "python", f"scripts/evaluation/{subdir}.py",
                "--single",
                "--sim-root", str(sim_root.relative_to(SSDATABENCH)),
                "--output-base", str(output_base.relative_to(SSDATABENCH)),
                "--config", f"evaluation/config/{subdir}_subset/evaluation_master.yaml",
            ],
            cwd=SSDATABENCH,
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        shutil.rmtree(sim_root, ignore_errors=True)

    # The script must execute without a Python crash; it may produce warnings.
    assert "Traceback" not in result.stderr, (
        f"Evaluation crashed for {short_name}.\n--- stderr ---\n{result.stderr[-2000:]}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}"
    )
    # And must report something into output_base on success
    if output_base.exists():
        produced = list(output_base.rglob("*"))
        assert produced, f"No outputs produced in {output_base}"
