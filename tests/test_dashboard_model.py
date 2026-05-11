from pathlib import Path

import pytest

from ssdataagent.dashboard.config import load_configs
from ssdataagent.dashboard.ledger import parse_ledger
from ssdataagent.dashboard.model import assemble
from ssdataagent.dashboard.results import load_results
from ssdataagent.dashboard.retros import parse_retro

FIXTURES = Path(__file__).parent / "fixtures" / "dashboard"


def _build():
    ledger = parse_ledger(FIXTURES / "LEDGER.md")
    configs = load_configs(FIXTURES / "experiments.yaml")
    results_root = FIXTURES / "results"
    retros_root = FIXTURES / "retros"

    def results_loader(name: str):
        return load_results(results_root / name)

    def retro_loader(rel_path: str):
        return parse_retro(FIXTURES / rel_path)

    return assemble(ledger, configs, results_loader, retro_loader)


def test_assemble_returns_dashboard_with_one_entry_per_ledger_row():
    dash = _build()
    assert len(dash.experiments) == 3


def test_assemble_overall_mean_computed_from_full_agent_only():
    dash = _build()
    a = next(e for e in dash.experiments if "exp_demo_a" in e.exp_names)
    # full_agent scores 0.5, 0.6, 0.16 -> mean = 0.42
    assert a.overall_mean_full_agent == pytest.approx(0.42, abs=0.01)


def test_assemble_overall_mean_skips_missing_t4_t5():
    dash = _build()
    a = next(e for e in dash.experiments if "exp_demo_a" in e.exp_names)
    # T4 and T5 absent for cross-sectional - MUST NOT be zero-filled into the mean.
    assert a.overall_mean_full_agent > 0.4


def test_assemble_combines_multi_exp_row():
    dash = _build()
    b = next(e for e in dash.experiments if "exp_demo_b_cross" in e.exp_names)
    assert b.exp_names == ["exp_demo_b_cross", "exp_demo_b_long"]
    # Combined mean over BOTH experiments' full_agent cells.
    # cross: 0.6, 0.7, 0.4 ; long: 0.5, 0.65, 0.3, 0.2, 0.7
    # combined mean = (0.6+0.7+0.4+0.5+0.65+0.3+0.2+0.7) / 8 = 4.05/8 = 0.50625
    assert b.overall_mean_full_agent == pytest.approx(0.506, abs=0.01)


def test_assemble_champion_excludes_pilots():
    dash = _build()
    champ = next(e for e in dash.experiments if e.is_champion)
    assert not champ.is_pilot
    assert not any(n.startswith("pilot") for n in champ.exp_names)


def test_assemble_champion_is_highest_non_pilot():
    dash = _build()
    champ = next(e for e in dash.experiments if e.is_champion)
    # exp_demo_b combined mean (0.506) > exp_demo_a mean (0.42)
    assert "exp_demo_b_cross" in champ.exp_names


def test_assemble_resolves_hypothesis_from_frontmatter_when_present():
    dash = _build()
    a = next(e for e in dash.experiments if "exp_demo_a" in e.exp_names)
    assert "validates the parser" in a.hypothesis_text


def test_assemble_resolves_hypothesis_from_strategy_heading_for_drift_retro():
    dash = _build()
    b = next(e for e in dash.experiments if "exp_demo_b_cross" in e.exp_names)
    # No frontmatter; no `## Hypothesis`; falls back to `## Strategy`.
    assert "Combined cross+long demo" in b.hypothesis_text


def test_assemble_headline_comes_from_ledger():
    dash = _build()
    a = next(e for e in dash.experiments if "exp_demo_a" in e.exp_names)
    assert "0.42" in a.headline_text


def test_assemble_partial_data_when_one_exp_in_group_missing():
    """Multi-exp LEDGER row, one exp missing: still renders, marked partial."""
    ledger = parse_ledger(FIXTURES / "LEDGER.md")
    configs = load_configs(FIXTURES / "experiments.yaml")

    def loader_drop_long(name: str):
        if name == "exp_demo_b_long":
            return None
        return load_results(FIXTURES / "results" / name)

    dash = assemble(
        ledger,
        configs,
        loader_drop_long,
        lambda p: parse_retro(FIXTURES / p),
    )
    b = next(e for e in dash.experiments if "exp_demo_b_cross" in e.exp_names)
    assert b.is_partial is True
    assert b.overall_mean_full_agent is not None  # still computable from cross half


def test_assemble_hypothesis_falls_back_to_ledger_when_strategy_stub(tmp_path):
    """If a retro's Strategy section starts with a one-liner stub ending in
    ':' (e.g. 'From STRATEGY.md backlog:'), use the LEDGER's hypothesis
    field instead."""
    # Build a tiny in-memory LEDGER + retro pair.
    fixtures = tmp_path / "fix"
    (fixtures / "retros").mkdir(parents=True)
    (fixtures / "LEDGER.md").write_text(
        "| date | exp_name | model | git_sha | hypothesis | headline | retro |\n"
        "|------|----------|-------|---------|------------|----------|-------|\n"
        "| 2026-05-08 | `exp_x` | m | sha | meaningful hypothesis from the ledger | hl | [r](retros/r.md) |\n",
        encoding="utf-8",
    )
    (fixtures / "retros" / "r.md").write_text(
        "# exp_x\n\n## Strategy\n\nFrom STRATEGY.md backlog:\n\n"
        "> - [ ] **EXP-X** — described in the strategy doc.\n",
        encoding="utf-8",
    )

    from ssdataagent.dashboard.ledger import parse_ledger
    from ssdataagent.dashboard.config import load_configs
    from ssdataagent.dashboard.model import assemble
    from ssdataagent.dashboard.results import load_results
    from ssdataagent.dashboard.retros import parse_retro

    ledger = parse_ledger(fixtures / "LEDGER.md")
    # Empty config and results — we only care about hypothesis resolution.
    (fixtures / "experiments.yaml").write_text("experiments: {}\n", encoding="utf-8")
    configs = load_configs(fixtures / "experiments.yaml")
    dash = assemble(
        ledger, configs,
        lambda name: None,
        lambda link: parse_retro(fixtures / link) if link else None,
    )
    e = dash.experiments[0]
    assert "meaningful hypothesis from the ledger" in e.hypothesis_text
    assert "From STRATEGY.md backlog" not in e.hypothesis_text


def test_assemble_entry_with_no_results_renders_without_champion_flag():
    """Drop all results; entries render with overall_mean_full_agent=None, no champion."""
    ledger = parse_ledger(FIXTURES / "LEDGER.md")
    configs = load_configs(FIXTURES / "experiments.yaml")
    dash = assemble(
        ledger,
        configs,
        lambda name: None,
        lambda p: parse_retro(FIXTURES / p),
    )
    assert len(dash.experiments) == 3
    for e in dash.experiments:
        assert e.overall_mean_full_agent is None
        assert e.is_champion is False
