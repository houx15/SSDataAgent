"""Audited, description-only context specs for the blind face-swap. Every number derived
from the TARGET's sample has been removed from LLM-visible strings; only public/general
knowledge and definitional (codebook/instrument) structure remain. AUDIT_NOTES records
each removal.

Design note: these descriptions feed **marginal** elicitation only. The X->Y joint /
conditional structure (how an outcome varies WITH a covariate) is transferred from source
A and is NOT elicited here -- so conditional-pattern prose is deliberately excluded. That
prose was also the main firewall risk: age-conditional means (e.g. "child_number peaks at
30-45") could only be known from the target's microdata. The descriptions therefore carry
population identity + public/definitional coherence, never a conditional or modal statistic.

See docs/superpowers/specs/2026-07-26-blind-faceswap-design.md ("Firewall audit").
"""
from __future__ import annotations

from dataclasses import dataclass

from ssdataagent.transfer.b3_specs import SPECS as _B3


@dataclass(frozen=True)
class BlindSpec:
    population: str          # prose name of the context for the prompt (public/definitional)
    description: str         # population identity + public/definitional coherence only
    glosses: dict[str, str]  # audited per-variable definitions (no sample statistics)


# --- cps: population identity + public/definitional coherence (no sample statistics) ---
_CPS_DESCRIPTION = (
    "Every respondent belongs to the whole resident US population of 1980, children "
    "included, so age spans the full range from young children to the elderly. The "
    "variables must cohere as one real person: a young child has less than a high-school "
    "education, is not in the labor force, and has no occupation or income; educational "
    "attainment cannot exceed what a person's age allows. Fertility here is "
    "HOUSEHOLD-RESIDENT, not lifetime: child_number counts a person's own children under "
    "18 currently living with them, and age_first_childbirth is the age at which their "
    "first still-resident child was born ('No Child' when none live with them). A "
    "child_number of 0 corresponds to 'No Child'; any first-birth age is at least 12 and "
    "below the person's own age. income is in annual 1980 dollars and is null for someone "
    "with no earnings (such as a child); occupation is null when the person is not in the "
    "labor force."
)

# --- gss: population identity + public/definitional coherence (no sample statistics) ---
_GSS_DESCRIPTION = (
    "Every respondent is an adult (the GSS samples adults only), so everyone can have "
    "completed education, a labor-force status, and formed attitudes; there are no children "
    "in this sample. The variables must cohere as one real person. Fertility here is "
    "LIFETIME 'children ever born', not a household roster: child_number counts all "
    "children the person has ever had, and a child_number of 0 corresponds to "
    "age_first_childbirth 'No Child'; any first-birth age is at least 12 and below the "
    "person's own age. Educational attainment is bounded by age only at the young end. "
    "income is a categorical bracket, not a dollar amount, and 'Unemployed' is used when a "
    "person is not earning; occupation is a broad census category, and spouse_occupation is "
    "'No spouse' for the unmarried."
)

# gss glosses: strip the pool mean/range (child_number) and the quantile + conditional
# clauses (age_first_childbirth); keep definitional content.
_gss_glosses = dict(_B3["gss"].glosses)
_gss_glosses["child_number"] = (
    "total number of children EVER BORN in the respondent's lifetime (GSS lifetime "
    "fertility, NOT resident children); a small non-negative count"
)
_gss_glosses["age_first_childbirth"] = (
    "age at the respondent's FIRST live birth over their lifetime "
    "('No Child' if they never had a child)"
)

BLIND_SPECS: dict[str, BlindSpec] = {
    "cps": BlindSpec(_B3["cps"].population, _CPS_DESCRIPTION, dict(_B3["cps"].glosses)),
    "gss": BlindSpec(_B3["gss"].population, _GSS_DESCRIPTION, _gss_glosses),
}

AUDIT_NOTES: dict[str, list[str]] = {
    "cps": [
        "rewrote the description to population identity + public/definitional coherence "
        "only; removed the CPS-sample age-conditional statistics carried in the b3 rules -- "
        "child_number 'peaks around ages 30-45 and falls back to ~0 by the late 50s' and "
        "age_first_childbirth rising '~22 in their 30s to ~33 in their 60s' -- and the "
        "income quantile 'a few thousand is typical'.",
        "glosses reused from b3: definitional (categories/units) only; the '0.66' "
        "resident-child mean lives solely in a b3_specs code comment, never in an "
        "LLM-visible string.",
    ],
    "gss": [
        "rewrote the description to population identity + public/definitional coherence "
        "only; removed the pool-mean fertility pattern 'climbing to ~2 by the 40s and "
        "staying there' and the modal-income claim \"'$10000 OR MORE' dominates\".",
        "scrubbed the child_number gloss (pool 'mean ~1.8' and the '0..8' range) and the "
        "age_first_childbirth gloss (the 'typically 18-30' quantile plus the education/age "
        "conditional clauses); both are target-sample-derived.",
    ],
}
