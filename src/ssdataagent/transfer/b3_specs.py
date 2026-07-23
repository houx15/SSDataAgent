"""Per-dataset generation specs for the no-donor LLM full method (B3 + the durable
no-donor headline path). Shared by scripts/nodonor_fullmethod.py and scripts/transfer_b3.py
so the cps prompt has one source of truth.
"""
from __future__ import annotations

import pandas as pd


class Spec:
    """Per-dataset wiring. `predictors` and `numeric_predictors` MUST mirror the
    dataset's type3.yaml -- the repair calibrates R^2 against the statistic T3 scores,
    so a mismatch here silently mis-sizes every alpha."""

    def __init__(self, seeds, derived, predictors, numeric_predictors, log_vars,
                 types, population, rules, glosses):
        self.seeds = seeds                          # drawn independently from marginals
        self.derived = derived                      # exact identities, computed not drawn
        self.predictors = predictors                # held coherent through the repair
        self.numeric_predictors = numeric_predictors
        self.log_vars = log_vars
        self.types = types
        self.population = population                # prose name for the prompt
        self.rules = rules                          # coherence rules for the prompt
        self.glosses = glosses                      # outcome glosses for elicitation


SPECS = {
    # cps is a 1980 cross-section of the WHOLE population, children included (age 0-88).
    # birth_year is not drawn: age + birth_year == 1980 for every row, an identity of the
    # survey design, so drawing both independently would manufacture impossible people.
    "cps": Spec(
        seeds=["age", "gender", "race"],
        derived={"birth_year": lambda df: 1980.0 - pd.to_numeric(df["age"])},
        predictors=["age", "gender", "race", "education"],
        numeric_predictors=frozenset({"age"}),
        log_vars=frozenset({"income"}),
        types=(1, 2, 3),
        population="the 1980 US Current Population Survey (March ASEC)",
        # CRITICAL semantic correction (see docs/report/2026-07-20-cps-fertility-semantics.md):
        # `child_number` / `age_first_childbirth` are NOT lifetime fertility. This is the CPS
        # household roster -- OWN children UNDER 18 currently LIVING WITH the respondent. The
        # data proves it: mean child_number 0.66 (lifetime 1980 fertility was ~2.5-3), and 60+
        # respondents average 0.16 because grown children have moved out. Describing it as
        # "children ever born" made the LLM generate lifetime fertility -- monotonically rising
        # with age -- inverting the true age relationship (real Spearman(age, first-birth) =
        # +0.62; the LLM produced -0.26) and tanking T3. This is a DEFINITION correction from
        # public metadata, NOT the pool's conditional distribution, so T3 stays non-circular.
        rules="""- This is the WHOLE resident population, children included. A 7-year-old is
  'Less than high school', not in the labor force, no occupation, no income.
- Education is bounded by age: nobody under 18 is 'College and above'.
- FERTILITY IS HOUSEHOLD-RESIDENT, NOT LIFETIME. child_number counts this person's OWN
  children UNDER 18 who currently LIVE WITH THEM; age_first_childbirth is the age at which
  their FIRST still-resident child was born. Adult children have moved out and no longer
  count. Consequences you must honour:
    * Most people have child_number 0 and 'No Child' -- including the elderly, whose
      children grew up and left. A 65-year-old almost always shows 0 resident children even
      though they were a parent. Do NOT let child_number climb with age; it PEAKS around
      ages 30-45 and falls back to ~0 by the late 50s.
    * Among people who DO have a resident child, the older they are, the LATER that child was
      born (their early children have already left home) -- so age_first_childbirth RISES
      with the respondent's age, from ~22 in their 30s to ~33 in their 60s. It is always
      >= 12 and < the person's own age.
    * child_number 0 <=> 'No Child'; a nonzero child_number needs a real first-birth age.
- income is annual 1980 dollars (a few thousand is typical); null for someone with no
  earnings such as a child. occupation is null when not in the labor force.
- Marital status, education and income should cohere as one real person.""",
        glosses={
            "marital_status": "marital status (Married / Single / Separated-Divorced-Widowed)",
            "child_number": "number of own children UNDER 18 currently living in the "
                            "household (NOT lifetime births; adult children have moved out)",
            "age_first_childbirth": "age when the first still-resident child was born "
                                    "('No Child' if none live with the respondent)",
            "education": "educational attainment (3 levels)",
            "laborforce": "in the labor force or not",
            "occupation": "broad occupation category",
            "income": "total personal income last year, 1980 dollars",
            "poverty_status": "above or below the poverty line",
        },
    ),
    # gss is a 2018 cross-section of ADULTS (age 18-89). Unlike cps, child_number is
    # LIFETIME "children ever born" (pool mean ~1.8; stock type3.yaml confirms), so the age
    # relationship runs the OTHER way -- it rises then plateaus with age. income is a
    # categorical bracket, not a dollar amount, so it is not numeric and gets no R^2 repair.
    "gss": Spec(
        seeds=["age", "gender", "race"],
        derived={},
        predictors=["age", "gender", "race", "education"],
        numeric_predictors=frozenset({"age"}),
        log_vars=frozenset(),
        types=(1, 2, 3),
        population="the 2018 US General Social Survey (GSS), adults 18 and older",
        rules="""- Every respondent is an ADULT (18-89). There are no children in this
  sample: everyone can have completed education, a labor-force status, and attitudes.
- FERTILITY IS LIFETIME here, the opposite of a household roster. child_number counts ALL
  children the person has EVER had, so it RISES with age and then plateaus: near 0 in the
  late teens/early 20s, climbing to ~2 by the 40s and staying there into old age. Do NOT
  let it fall for older people.
    * child_number 0 <=> age_first_childbirth 'No Child'. A nonzero count needs a real
      first-birth age, always >= 12 and < the person's own age.
    * age_first_childbirth is the LIFETIME first birth, typically 18-30, only weakly
      related to current age; more-educated people tend to start later.
- Education (Less than high school / High school / College and above) is bounded by age
  only at the young end (a 19-year-old is rarely 'College and above' yet).
- vocabulary_test (0-10) rises with education and is roughly flat in age.
- income is a categorical BRACKET ('$10000 OR MORE' dominates; 'Unemployed' when not
  earning), not a dollar amount. occupation is a broad census category; 'Unemployed' or
  'Military Occupations' are valid. spouse_occupation is 'No spouse' for the unmarried.
- Attitudes (gender_role_attitude, political_view, trust, happy, satisfy_job, work_hard)
  and health should read as one coherent person, plausible against the marginals below.
- immigrant_status and parental background (mother/father education & occupation) are
  inferred coherently, kept plausible against the marginals.""",
        glosses={
            "child_number": "total number of children EVER BORN in the respondent's "
                            "lifetime (GSS lifetime fertility, NOT resident children); "
                            "0..8, mean ~1.8",
            "age_first_childbirth": "age at the respondent's FIRST live birth over their "
                                    "lifetime ('No Child' if they never had a child)",
            "vocabulary_test": "number of correct words (0-10) on the GSS WORDSUM "
                               "vocabulary test, an indicator of verbal/cognitive skill",
        },
    ),
}
