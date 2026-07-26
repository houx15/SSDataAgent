import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]


def test_blind_specs_cover_both_datasets_with_fields():
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    for ds in ("cps", "gss"):
        s = BLIND_SPECS[ds]
        assert s.population and isinstance(s.population, str)
        assert s.description and isinstance(s.description, str)
        assert isinstance(s.glosses, dict) and s.glosses


def test_firewall_scrubbed_target_sample_numbers_are_gone():
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    # The gss child_number gloss must no longer quote the pool mean ("1.8").
    gss_text = BLIND_SPECS["gss"].description + " " + " ".join(BLIND_SPECS["gss"].glosses.values())
    assert "1.8" not in gss_text and "pool mean" not in gss_text.lower()
    # cps: no household-roster sample mean leaked into LLM-visible strings.
    cps_text = BLIND_SPECS["cps"].description + " " + " ".join(BLIND_SPECS["cps"].glosses.values())
    assert "0.66" not in cps_text


def test_audit_notes_document_removals():
    from ssdataagent.transfer.blind_specs import AUDIT_NOTES
    assert AUDIT_NOTES.get("gss") and AUDIT_NOTES.get("cps")
