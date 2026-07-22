# Delta plan — extending SSDataAgent

**Operative handoff doc.** This maps the design in `design-reference.md` onto the existing repo `github.com/houx15/SSDataAgent`. It is an **additive extension, not a rewrite**.

> Written from the repo's README, not the source. **Coding agent: verify every module path against the actual code before editing**; inferred paths are marked `[verify]`. Where the repo already implements something described here, extend it — do not add a parallel system.

---

## The one-line reason this work exists

You have already shown the **agent paradigm beats the SSDataBench paper** (+0.13 on gpt-5.4, first non-trivial T4). The gap is **attribution**: every result is agent-produced, so you cannot yet say *which mechanism* produces the gain. This plan factors the agent's implicit modeling into **explicit, directly-runnable strategies** (Designs A/B/C + statistical baselines), adds the **over-determination metric**, and—optionally, last—a **web console**. It turns a system result into a science result.

---

## Keep as-is (the bulk of the repo)

Do not touch these except to call them through the new seam:

- **`src/ssdataagent/data/`** — schema, loader, deterministic splitter.
- **`src/ssdataagent/evaluation/`** + **`config/paper_baselines.json`** — the subprocess wrapper around SSDataBench and Δ-vs-paper reporting. **This is the frozen scorer and the "reproduce-the-paper" gate — already solved.** Keep shelling out; do not import SSDataBench.
- **`src/ssdataagent/experiments/`** — runner, logger, conditions, `direct_generation`.
- **`scripts/`** — `run_experiment`, `run_batch`, `status`, `generate_exp_report`, `build_eval_subset`.
- **`src/ssdataagent/agent/`** — orchestrator, 16-tool registry, `Chain`/`Step`, `RuntimeState`. Becomes **one strategy among several**.
- **`docs/experiments/`** — `LEDGER.md`, `STRATEGY.md`, per-exp reports. This is the existing iteration loop; keep using it.
- **`tests/`** — extend, don't replace.

## Refactor — exactly one seam

**Introduce a `Strategy` interface so the runner is method-agnostic.** Today the runner branches on `PromptVariant.is_tool_using` / `direct_generation` to produce `generated.csv`. Generalize:

- `[verify/new]` `src/ssdataagent/strategies/base.py` — a protocol:
  ```python
  class Strategy(Protocol):
      name: str
      def generate(self, gate: InfoGate, targets, config) -> "DataFrame | path to generated.csv": ...
  ```
- Wrap the **two existing paths** as strategies with no behavior change: `AgentStrategy` (the orchestrator; `PromptVariant` routing stays *internal* to it) and `DirectGenerationStrategy` (existing `direct_generation`).
- The runner selects a strategy by a config key instead of branching.
- **`InfoGate`:** your `conditions` (`full_agent / no_semantic / no_data / full_agent_unseen`) already gate what the agent receives. Lift that gating into a shared object every strategy calls, so a fixed strategy is held to the same information budget as the agent. Prefer **extending the existing conditions** over adding a parallel object. (Condition ↔ A/B/C mapping below.)

This is the only structural change. Everything else is new files behind this seam.

**Regression gate:** after the refactor, `AgentStrategy` and `DirectGenerationStrategy` must reproduce today's `eval.json` numbers bit-for-bit on a smoke run. If they move, the seam changed behavior — fix before continuing.

## Add — new modules that touch nothing existing

All land under `[verify/new]` `src/ssdataagent/strategies/`:

1. **`baselines.py` — B1 statistical synthesizers.** hot-deck / k-NN (first), synthpop-style sequential CART, copula / Bayesian-net. Condition C → IPF / max-entropy to known marginals. Reuse `RuntimeState`'s `train_fit` / `held_out` split. This is the **honest bar**; expect it to beat the LLM in condition A — that is itself a finding.
2. **`design_b.py` — marginals + copula (build first).** LLM (reuse `agent/llm_client.py`) emits per-cell conditional distributions; **rake** to known/train marginals; fit a **copula** for cross-target dependence. Two independent ablatable knobs (marginal vs. dependence).
3. **`design_c.py` — retrieval + repair.** k-NN retrieve → hot-deck → (condition B) LLM **transport** to target context → **SIR / raking repair loop** to known moments.
4. **`design_a.py` — hierarchical Bayes with LLM-elicited priors (build last).** LLM proposes DAG / functional forms / wide priors → PyMC or NumPyro → partial pooling → posterior-predictive sampling. Heaviest; new dependency.
5. **`s1_distribution.py` — S1 diagnostic** (+ mixture-of-personas variant). Whole-view distribution emission; used to *isolate* sampling-collapse from prior-collapse, not to win.
6. **Over-determination metric** in `[verify]` `src/ssdataagent/evaluation/` — `H(target | demographics)` real vs. sim, computed locally on `generated.csv` vs. real (not via SSDataBench). Surface it in `eval.json` and `generate_exp_report.py` next to T1–T5. **This is the headline diagnostic** and is currently missing from the verify tools.

See `design-reference.md` §7–§8 for the full mechanism of each.

## Condition ↔ information-condition mapping

| Repo condition | design-reference condition | Note |
|---|---|---|
| `full_agent` | **A — in-distribution** | sees same-survey microdata |
| `full_agent_unseen` | **B — transfer** | confirm it implements source≠target, not just a held-out slice |
| `no_data` | **C — aggregate-only** | tighten to "marginals only," define what counts as known |
| `no_semantic` | ablation (orthogonal) | strips descriptions; keep as-is |

## Sequencing (minimize disruption)

- **P0 — Strategy seam.** Wrap agent + direct_generation; pass the regression gate.
- **P1 — B1 baselines (hot-deck first) + over-determination metric.** Fast signal, sets the bar.
- **P2 — Design B.**
- **P3 — Design C.**
- **P4 — Design A.**
- **P5 (optional, last) — web console** (`design-reference.md` §14). Keep using `status.py` + `LEDGER.md` until the CLI is the bottleneck. Do not let the console block or drive the refactor.

## Open research decisions (human, not the coding agent)

1. **Transfer pairs + variable crosswalk (condition B).** Start with **GSS year→year** (shares schema by construction). Confirm whether `full_agent_unseen` already provides a usable transfer setup or needs a real source→target crosswalk (e.g. NLSY↔CFPS may not align).
2. **Condition-C "known moments" rule** per dataset — marginals from withheld microdata vs. published toplines; which (if any) associations count as known.
3. **Sequence scope (T4/T5).** The repo already handles longitudinal panels + a chronology commit-gate **via the agent**. The fixed Designs A/B/C need an explicit sequence representation. **Recommended:** Designs A/B/C target tabular / attitudes first; sequences stay **agent-only** until a sequence-aware design lands.
4. **Base models** for `main` / `cheap` (repo defaults to gpt-5.4).

## Assumptions to verify in code

- `src/ssdataagent/strategies/` does not already exist / collide.
- The existing `conditions` are effectively the `InfoGate` — extend them rather than duplicate.
- `evaluation/` stays a subprocess to SSDataBench; the over-determination metric is computed locally and added alongside, not inside, SSDataBench.
- `AgentStrategy` must preserve the `commit_generator` chronology gate and full tool-call logging unchanged.
