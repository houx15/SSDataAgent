import importlib
import sys
import pathlib

# Prepend repo root to sys.path so scripts/ can be imported
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_evaluation_exports_overdetermination():
    mod = importlib.import_module("ssdataagent.evaluation")
    assert hasattr(mod, "overdetermination")


def test_overdetermination_section_builder():
    from scripts.generate_exp_report import _overdetermination_section
    cells = {
        "gss": {"overdetermination": {
            "cell_based": {"headline_gap": 0.5, "coverage": 0.8, "n_cells": 12},
            "model_based": {"headline_gap": 0.3}}},
        "cps": None,
    }
    md = _overdetermination_section(cells, ["gss", "cps"])
    assert "Over-determination" in md
    assert "0.500" in md  # cell-based gap formatted
    assert "gss" in md and "cps" in md
