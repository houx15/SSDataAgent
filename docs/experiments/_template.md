---
exp_name: {{exp_name}}
date: {{date}}
model: {{model}}
git_sha: {{git_sha}}
baseline_exp: {{baseline_exp}}
status: planned
hypothesis: {{hypothesis}}
---

# {{exp_name}}

## Hypothesis
<!-- 2–3 sentences. What change is under test, and what score delta do we expect vs `baseline_exp`? -->

{{hypothesis}}

## Setup
- **Datasets:**
- **Conditions:**
- **Code change(s) under test:** (file + line refs, or commit hash)
- **Run command:**
  ```bash
  python scripts/run_experiment.py --experiment {{exp_name}}
  ```

## Results
<!-- Paste the full table from `python scripts/summarize_pilot.py {{exp_name}}` here. -->
<!-- Add a one-line headline above it: e.g. "overall=0.486 (+0.010 vs baseline pilot_paper_agents_gpt54)". -->

## Retro
- **What worked:**
- **What didn't:**
- **Surprises:**
- **Lesson worth preserving:** <!-- promote to STRATEGY.md "lessons" if non-obvious -->
- **Next experiment:** <!-- link to a STRATEGY.md backlog item, or name a new one -->
