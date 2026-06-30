import json
from pathlib import Path

from ssdataagent import reports


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "results"
    run = root / "exp_a" / "full_agent" / "gss" / "20260630-100000"
    run.mkdir(parents=True)
    (run / "eval.json").write_text(json.dumps({
        "by_type": {"type1": 0.6}, "overall_average": 0.6,
        "overdetermination": {"cell_based": {"headline_gap": 0.3, "coverage": 0.9,
                                             "n_cells": 10},
                              "model_based": {"headline_gap": 0.25}}}))
    yaml_path = tmp_path / "experiments.yaml"
    yaml_path.write_text(
        "experiments:\n  exp_a:\n    datasets: [gss]\n    conditions: [full_agent]\n"
        "    max_iterations: 3\n    n_rows: 1000\n")
    paper = tmp_path / "paper.json"
    paper.write_text(json.dumps({"by_dataset_by_type": {"gss": {"T1": 0.5}},
                                 "by_dataset_overall": {"gss": 0.5}}))
    return root, yaml_path, paper


def test_render_markdown_contains_results_and_overdet(tmp_path: Path):
    root, yaml_path, paper = _setup(tmp_path)
    md = reports.render_markdown_report(
        "exp_a", results_root=root, experiments_yaml=yaml_path, paper_baselines=paper)
    assert "## Results" in md
    assert "Over-determination gap" in md
    assert "0.600" in md          # the type1 score, formatted
    assert "0.300" in md          # the over-determination headline gap


def test_render_html_wraps_markdown(tmp_path: Path):
    root, yaml_path, paper = _setup(tmp_path)
    md = reports.render_markdown_report(
        "exp_a", results_root=root, experiments_yaml=yaml_path, paper_baselines=paper)
    html = reports.render_html_report(md)
    assert "<html" in html.lower()
    assert "Results" in html
