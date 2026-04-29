"""Manual smoke test: run SSDataBench's GSS-2018 evaluation on the real data
treated as both sampled and simulated. Useful first sanity check that the
evaluation pipeline works end-to-end. Expected outcome: high pass rates
(real == sim by construction).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SSDATABENCH = REPO / "ssdatabench"

import sys
sys.path.insert(0, str(REPO / "scripts"))
from build_eval_subset import build_subset  # noqa: E402


def main() -> None:
    build_subset("gss")
    real_csv = REPO / "real_data" / "gss_clean.csv"
    sim_root = SSDATABENCH / "simulated_data" / "gss_2018" / "agent_smoke"
    sim_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(real_csv, sim_root / "sampled_smoke.csv")
    shutil.copy(real_csv, sim_root / "sim_profiles_smoke.csv")
    output_base = SSDATABENCH / "evaluation_results" / "gss_2018" / "agent_smoke"
    try:
        subprocess.run(
            [
                "python", "scripts/evaluation/gss_2018.py",
                "--single",
                "--sim-root", str(sim_root.relative_to(SSDATABENCH)),
                "--output-base", str(output_base.relative_to(SSDATABENCH)),
                "--config", "evaluation/config/gss_2018_subset/evaluation_master.yaml",
            ],
            cwd=SSDATABENCH,
            check=False,
        )
        print("\n[smoke_eval] outputs under:", output_base)
    finally:
        shutil.rmtree(sim_root, ignore_errors=True)


if __name__ == "__main__":
    main()
