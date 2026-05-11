from pathlib import Path

from ssdataagent.dashboard.retros import RetroSections, parse_retro

FIXTURES = Path(__file__).parent / "fixtures" / "dashboard"


def test_parse_retro_reads_frontmatter():
    r = parse_retro(FIXTURES / "retros" / "2026-05-10-exp_demo_a-report.md")
    assert r.frontmatter["exp_name"] == "exp_demo_a"
    assert r.frontmatter["date"] == "2026-05-10"
    assert r.frontmatter["model"] == "gpt-5.4-2026-03-05"


def test_parse_retro_template_conformant_sections():
    r = parse_retro(FIXTURES / "retros" / "2026-05-10-exp_demo_a-report.md")
    assert "Hypothesis" in r.sections
    assert "Setup" in r.sections
    assert "Results" in r.sections
    assert "Retro" in r.sections
    assert "validates the parser end-to-end" in r.sections["Hypothesis"]


def test_parse_retro_drift_sections():
    r = parse_retro(FIXTURES / "retros" / "2026-05-11-exp_demo_b-report.md")
    assert "Strategy" in r.sections
    assert "Hypothesis" not in r.sections
    assert "Combined cross+long demo" in r.sections["Strategy"]


def test_parse_retro_extracts_bulleted_kv():
    """Bullets like `- **prompt_variant:** rubric_tools_v2` parse out the value."""
    r = parse_retro(FIXTURES / "retros" / "2026-05-11-exp_demo_b-report.md")
    bullets = r.bullets["Strategy"]
    kv = dict(bullets)
    assert kv.get("prompt_variant") == "rubric_tools_v2"
    assert kv.get("datasets") == "demo_x, demo_y"


def test_parse_retro_template_retro_bullets():
    r = parse_retro(FIXTURES / "retros" / "2026-05-10-exp_demo_a-report.md")
    retro_kv = dict(r.bullets["Retro"])
    assert "template parser hit" in retro_kv.get("What worked", "")


def test_parse_retro_handles_missing_frontmatter():
    r = parse_retro(FIXTURES / "retros" / "2026-05-11-exp_demo_b-report.md")
    assert r.frontmatter == {}


def test_parse_retro_handles_pilot_with_no_h2():
    r = parse_retro(FIXTURES / "retros" / "2026-05-09-pilot_demo-report.md")
    assert r.sections == {}
    assert r.bullets == {}
    assert "overall mean = 0.35" in r.raw_text


def test_parse_retro_missing_file_returns_none():
    assert parse_retro(FIXTURES / "retros" / "no-such-file.md") is None


def test_parse_retro_multiline_bullets_captured(tmp_path):
    """Bullet values that span multiple lines are joined; next bullet's
    marker does NOT leak into the previous bullet's value."""
    p = tmp_path / "retro.md"
    p.write_text(
        "## Retro\n\n"
        "- **What worked:** First line of value.\n"
        "  Second line continues the same bullet.\n"
        "- **What didn't:** Different bullet's value.\n"
        "- **Surprises:**\n"
        "  Value on the next line.\n",
        encoding="utf-8",
    )
    r = parse_retro(p)
    kv = dict(r.bullets["Retro"])
    assert kv["What worked"] == "First line of value. Second line continues the same bullet."
    assert kv["What didn't"] == "Different bullet's value."
    assert "**" not in kv["What worked"]
    assert "**" not in kv["What didn't"]
    assert kv["Surprises"] == "Value on the next line."
