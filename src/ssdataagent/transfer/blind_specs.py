"""Audited, description-only context specs for the blind face-swap. Every number derived
from the TARGET's sample has been removed from LLM-visible strings; only public/general
knowledge and qualitative structure remain. AUDIT_NOTES records each check/removal.

See docs/superpowers/specs/2026-07-26-blind-faceswap-design.md ("Firewall audit").
"""
from __future__ import annotations

from dataclasses import dataclass

from ssdataagent.transfer.b3_specs import SPECS as _B3


@dataclass(frozen=True)
class BlindSpec:
    population: str          # prose name of the context for the prompt
    description: str         # audited coherence rules / qualitative structure
    glosses: dict            # audited per-variable definitions


# --- gss: strip the pool mean from the child_number gloss --------------------
_gss_glosses = dict(_B3["gss"].glosses)
_gss_glosses["child_number"] = (
    "total number of children EVER BORN in the respondent's lifetime (GSS lifetime "
    "fertility, NOT resident children); a small non-negative count"
)

BLIND_SPECS: dict[str, BlindSpec] = {
    "cps": BlindSpec(_B3["cps"].population, _B3["cps"].rules, dict(_B3["cps"].glosses)),
    "gss": BlindSpec(_B3["gss"].population, _B3["gss"].rules, _gss_glosses),
}

AUDIT_NOTES: dict[str, list[str]] = {
    "cps": [
        "checked population/rules/glosses: no target-sample statistic appears in any "
        "LLM-visible string (the '0.66' resident-child mean lives only in a b3_specs "
        "code comment, not in rules/glosses); reused verbatim.",
    ],
    "gss": [
        "removed 'mean ~1.8' and the '0..8' range from the child_number gloss "
        "(both are target-sample statistics); replaced with a qualitative descriptor.",
    ],
}
