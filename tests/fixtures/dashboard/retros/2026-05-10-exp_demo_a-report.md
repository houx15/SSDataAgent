---
exp_name: exp_demo_a
date: 2026-05-10
model: gpt-5.4-2026-03-05
git_sha: def5678
baseline_exp: pilot_demo
status: done
hypothesis: Demo run validates the parser end-to-end.
---

# exp_demo_a

## Hypothesis
Demo run validates the parser end-to-end on a template-conformant retro file.

## Setup
- **Datasets:** demo
- **Conditions:** full_agent
- **Code change(s) under test:** none — this is a parser fixture
- **Run command:**
  ```bash
  python scripts/run_experiment.py --experiment exp_demo_a
  ```

## Results
overall=0.42 on demo. T1=0.5, T2=0.6, T3=0.16.

## Retro
- **What worked:** template parser hit Hypothesis/Setup/Results/Retro cleanly.
- **What didn't:** nothing; this is a fixture.
- **Surprises:** none.
- **Lesson worth preserving:** template files are easy; the drift-y ones are the real test.
- **Next experiment:** exp_demo_b.
