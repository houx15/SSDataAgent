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


@pytest.mark.live_eval
def test_evaluation_single_folder_runs(tmp_path):
    """Shell out to ssdatabench's GSS-2018 evaluation in --single mode against a folder
    containing identical sampled_/sim_ files. Proves the pipeline executes; near-100%
    pass rate is the expected outcome since real==sim."""
    real_csv = REPO / "real_data" / "gss_clean.csv"
    assert real_csv.exists()
    build_subset("gss")

    sim_root = SSDATABENCH / "simulated_data" / "gss_2018" / "agent_smoke"
    sim_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(real_csv, sim_root / "sampled_smoke.csv")
    shutil.copy(real_csv, sim_root / "sim_profiles_smoke.csv")
    output_base = SSDATABENCH / "evaluation_results" / "gss_2018" / "agent_smoke"

    try:
        result = subprocess.run(
            [
                "python", "scripts/evaluation/gss_2018.py",
                "--single",
                "--sim-root", str(sim_root.relative_to(SSDATABENCH)),
                "--output-base", str(output_base.relative_to(SSDATABENCH)),
                "--config", "evaluation/config/gss_2018_subset/evaluation_master.yaml",
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
        f"Evaluation crashed.\n--- stderr ---\n{result.stderr[-2000:]}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}"
    )
    # And must report something into output_base on success
    if output_base.exists():
        produced = list(output_base.rglob("*"))
        assert produced, f"No outputs produced in {output_base}"
