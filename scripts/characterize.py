#!/usr/bin/env python
"""Run the transfer characterization sweep (Q1-Q4) and write the tidy results table."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssdataagent.config import REPO_ROOT
from ssdataagent.transfer.characterize import run_characterization, write_outputs


def main() -> None:
    df = run_characterization()
    results_csv, committed = write_outputs(df, REPO_ROOT)
    print(f"characterization: {len(df)} rows across {df['pair'].nunique()} pairs")
    print(f"  results copy   -> {results_csv}")
    print(f"  committed copy -> {committed}")


if __name__ == "__main__":
    main()
