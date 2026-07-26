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


def _llm_visible_text(spec) -> str:
    """Everything the LLM sees for a context: population + description + glosses. (Excludes
    AUDIT_NOTES, which legitimately quotes the removed phrases as internal documentation.)"""
    return " ".join([spec.population, spec.description, *spec.glosses.values()])


# Target-sample statistics that must NOT appear in any LLM-visible string. These are the
# specific leaks the reused b3 rules/glosses carried: pool means, modal-category claims,
# quantiles, and age-conditional statistics that could only be known from B's microdata.
_FORBIDDEN = {
    "gss": ["1.8", "pool mean", "0..8", "climbing to", "$10000 OR MORE", "typically 18-30"],
    "cps": ["0.66", "30-45", "late 50s", "~22", "~33", "a few thousand", "peaks"],
}


def test_firewall_no_target_sample_statistics_in_llm_visible_text():
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    for ds, forbidden in _FORBIDDEN.items():
        text = _llm_visible_text(BLIND_SPECS[ds]).lower()
        for phrase in forbidden:
            assert phrase.lower() not in text, f"{ds}: firewall leak {phrase!r} still present"


def test_audit_notes_document_removals():
    from ssdataagent.transfer.blind_specs import AUDIT_NOTES
    assert AUDIT_NOTES.get("gss") and AUDIT_NOTES.get("cps")
