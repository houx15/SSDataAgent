from pathlib import Path

from ssdataagent.dashboard.config import load_configs
from ssdataagent.dashboard.ledger import parse_ledger
from ssdataagent.dashboard.model import assemble
from ssdataagent.dashboard.render import render_html
from ssdataagent.dashboard.results import load_results
from ssdataagent.dashboard.retros import parse_retro

FIXTURES = Path(__file__).parent / "fixtures" / "dashboard"


def _build_dashboard():
    ledger = parse_ledger(FIXTURES / "LEDGER.md")
    configs = load_configs(FIXTURES / "experiments.yaml")
    return assemble(
        ledger,
        configs,
        lambda name: load_results(FIXTURES / "results" / name),
        lambda p: parse_retro(FIXTURES / p),
    )


def test_render_writes_html_file(tmp_path):
    dash = _build_dashboard()
    out = tmp_path / "dashboard.html"
    render_html(dash, out)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_render_html_contains_each_experiment_name(tmp_path):
    dash = _build_dashboard()
    out = tmp_path / "dashboard.html"
    render_html(dash, out)
    html = out.read_text(encoding="utf-8")
    for name in ["exp_demo_a", "exp_demo_b_cross", "exp_demo_b_long", "pilot_demo"]:
        assert name in html, f"{name} missing from rendered HTML"


def test_render_html_marks_the_champion(tmp_path):
    dash = _build_dashboard()
    out = tmp_path / "dashboard.html"
    render_html(dash, out)
    html = out.read_text(encoding="utf-8")
    assert "CHAMPION" in html.upper() or 'class="champion"' in html
    champ = next(e for e in dash.experiments if e.is_champion)
    # Champion appears in both the champion card and the table row.
    assert html.count(champ.exp_names[0]) >= 2


def test_render_html_embeds_inline_json_no_fetch(tmp_path):
    dash = _build_dashboard()
    out = tmp_path / "dashboard.html"
    render_html(dash, out)
    html = out.read_text(encoding="utf-8")
    # No external resources.
    assert "fetch(" not in html
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    # JSON payload is embedded.
    assert "window.DASHBOARD_DATA" in html or 'id="dashboard-data"' in html


def test_render_html_has_one_row_per_experiment(tmp_path):
    dash = _build_dashboard()
    out = tmp_path / "dashboard.html"
    render_html(dash, out)
    html = out.read_text(encoding="utf-8")
    assert html.count("experiment-row") >= len(dash.experiments)
