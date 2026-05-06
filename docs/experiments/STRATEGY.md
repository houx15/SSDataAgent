# Strategy

The current state of our beliefs about why the agent scores what it scores, and what we want to try next. This file gets edited as we learn. The LEDGER is the receipt; this is the playbook.

## Current standing hypotheses

Diagnosis from reading the orchestrator + 4-stage prompts + 4 contrasting `step_002.py` files (CPS DeepSeek, GSS gpt-5.4, US gpt-5.4 T4-success, AddHealth gpt-5.4 T4-failure) on 2026-05-06. Ordered by expected lift × inverse-cost.

1. **The agent has never seen the rubric (T1–T5).** None of the four stage prompts mention what is being measured. The agent guesses the objective and picks model architecture by vibes. CPS-DeepSeek happened to pick chained DT (great); AddHealth-gpt5.4 picked "regression on roots only" (T4 → 0).
2. **VALIDATION self-passes trivially.** AddHealth gpt-5.4's validation stdout printed `ISSUES: none / VALIDATION OK` despite a sign-flipped correlation, 3× off conditional probability, and 16% TV-distance. The `max_validation_iters=3` budget collapses to 1 in practice.
3. **`preserve_missingness` is dead code.** Defined in `prompt_templates.py:48,61`, never invoked from `orchestrator.py:134,143`. All current models do independent Bernoulli per column for NaN, breaking the conditional missingness that T3 depends on.
4. **MODELING is wide-open.** "Free choice of model family … keep it simple and fast" pushes toward weak architectures, especially on longitudinal data where event-time chronology has to be enforced.
5. **No cross-run feedback.** The agent re-makes the same architecture mistake on every dataset because it never sees which architectures have ever worked.

## Backlog (in expected lift × inverse-cost order)

- [~] **EXP-001** — Add T1–T5 rubric block to `SYSTEM_PROMPT` as a new prompt variant. Yaml entries: `exp001_rubric_cross` + `exp001_rubric_long`. Run on cloud box via `python scripts/run_batch.py exp001_rubric_cross exp001_rubric_long`; report via `scripts/generate_exp_report.py exp001_rubric_cross --baseline pilot_paper_agents_gpt54`. *Expected lift:* +0.03–0.08 across the board. *Status:* in progress (matrix runner + variant landed 2026-05-06).
- [ ] **EXP-002** — Replace VALIDATION's "if anything is clearly off" with hard quantitative thresholds (TV ≤ 0.10, |Δr| ≤ 0.15, |ΔP| ≤ 0.05) and explicit refusal of `VALIDATION OK` if breached. *Expected lift:* +0.05–0.10 (revisions actually trigger). *Cost:* ~1h prompt + ~3h re-runs.
- [ ] **EXP-003** — Wire `preserve_missingness=True` from `orchestrator.py:134,143`. *Expected lift:* +0.05–0.15 on T3 specifically. *Cost:* 5-line code change + ~3h re-runs.
- [ ] **EXP-004** — MODELING decision rule: branch by dataset type (cross-sectional → chained autoregressive over inferred causal order; longitudinal → copula + event-time chronology pass). *Expected lift:* +0.05–0.15 on T4/T5 (the current 0.0 cells). *Cost:* ~3h prompt + new context plumbing + ~6h re-runs.
- [ ] **EXP-005** — `lessons.md` cross-run memory injected into `SYSTEM_PROMPT`, curated from the strongest cells we already have. *Expected lift:* +0.03–0.08, compounding. *Cost:* ~2h.

## Lessons (worth preserving across experiments)

Things that have stayed true across multiple runs and would be expensive to re-discover. Promote items here from individual experiment retros.

- **Architecture choice dominates LLM choice on cross-sectional T1.** CPS-DeepSeek 0.73 vs CPS-gpt5.4 0.59 isn't the LLM — it's that DeepSeek's run picked chained-DT-over-discretised-categoricals while gpt-5.4 picked Gaussian copula. Source: `2026-05-04-gpt54mini-vs-deepseek.md` + `2026-05-05-three-way-model-comparison.md`.
- **T4 is paradigm-bound, not model-bound.** Both DeepSeek and gpt-5.4-mini scored 0–0.05 on event-order chronology across all 4 longitudinal datasets. gpt-5.4 cracked it on US (0.27) only because the copula happened to capture marginal age ordering — not because anyone enforced chronology. Source: same.
- **ACS T3 is fragile to imputation choices.** Numbers swing 0.0 → 0.5+ depending on whether the agent imputes vs preserves NaN on conditionally-missing predictors. Confirmed across both models. Source: `2026-05-03-preserve-missingness-ablation.md`.
- **APIConnectionError on OpenAI is transient.** Always retry once with `--resume` before treating it as a real failure. Source: 2026-05-05 longitudinal run.

## Done
See LEDGER.md for the per-experiment table.
