---
exp_name: exp001_rubric_in_system_prompt
date: 2026-05-06
model: gpt-5.4-2026-03-05
git_sha: 6d9e112
baseline_exp: pilot_paper_agents_gpt54 + pilot_paper_longitudinal_gpt54
status: planned
hypothesis: Adding T1-T5 rubric block to SYSTEM_PROMPT lifts overall mean by 0.03-0.08 by letting the agent pick model architecture against the actual metrics instead of by vibes.
---

# exp001_rubric_in_system_prompt

## Hypothesis
The agent currently picks model architecture by vibes — none of the four stage prompts mention what is actually being measured (T1-T5). CPS DeepSeek happened to pick chained DT (great); AddHealth gpt-5.4 picked "regression on roots only" (T4 → 0). If we put the rubric in front of the agent, it should at least *attempt* the right architecture for the metric, even if execution is imperfect.

Expected lift vs the gpt-5.4 baselines: **+0.03 to +0.08 overall**, concentrated on T2/T3 (where the agent currently misses bivariate dependence and conditional missingness) and T4 (the cells where it currently scores 0.0 because no chronology pass is enforced).

## Setup
- **Yaml entries (matrix-of-variants):**
  - `exp001_rubric_cross` — gss/cps/acs × full_agent + agent_no_semantic + agent_no_data
  - `exp001_rubric_long`  — nlsy/addhealth/cfps/us × full_agent
- **Code change under test:** `src/ssdataagent/agent/prompt_templates.py` — new `rubric` variant in `PROMPT_VARIANTS`. Same exploration/modeling/validation/generation prompts as baseline; `SYSTEM_PROMPT` augmented with an EVALUATION RUBRIC block describing all five Ts and what architectural choices preserve them.
- **A/B baselines:** `pilot_paper_agents_gpt54` (cross-sectional) and `pilot_paper_longitudinal_gpt54` (longitudinal), same git_sha, same model.
- **Run command (cloud box):**
  ```bash
  # in tmux, after .env is set with LLM_API_KEY
  nohup python scripts/run_batch.py exp001_rubric_cross exp001_rubric_long \
      > batch.log 2>&1 &
  ```
- **Status check (from anywhere):**
  ```bash
  python scripts/status.py exp001_rubric_cross exp001_rubric_long
  tail -f results/exp001_rubric_cross/run.log
  ```

## Results
<!-- After the batch finishes, run:
       python scripts/generate_exp_report.py exp001_rubric_cross --baseline pilot_paper_agents_gpt54
       python scripts/generate_exp_report.py exp001_rubric_long  --baseline pilot_paper_longitudinal_gpt54
     and paste the headline + key cells here. -->

## Retro
- **What worked:**
- **What didn't:**
- **Surprises:**
- **Lesson worth preserving:** <!-- promote to STRATEGY.md "lessons" if non-obvious -->
- **Next experiment:** <!-- link to a STRATEGY.md backlog item, or name a new one -->
