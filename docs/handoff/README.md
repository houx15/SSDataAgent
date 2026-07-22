# Handoff — SSDataAgent extension

Drop this folder into the `SSDataAgent` repo (suggested: `docs/extension/`) to start coding.

**Reading order:**
1. **`delta-plan.md`** — start here. The operative plan: what to keep, the single refactor seam, what to add, sequencing, and the open decisions — all mapped onto the existing repo modules.
2. **`design-reference.md`** — the conceptual model behind the plan: thesis, hypotheses, the five strategy roles + Designs A/B/C, the A/B/C information conditions, the frozen scorer + over-determination metric, and the optional web console (§14).

**Core idea in one line:** the repo already shows the agent paradigm beats the paper; this work factors the agent's implicit modeling into explicit, attributable strategies (Designs A/B/C + statistical baselines), adds the over-determination metric, and optionally a local web console — turning a system result into a science result.

**Status:** design locked. Four research decisions remain open (see *Open research decisions* in `delta-plan.md`) — settle those before the coding agent builds condition B and the sequence handling.

**First coding step:** P0 in `delta-plan.md` — introduce the `Strategy` seam and wrap the existing agent + direct-generation paths behind it, then confirm the smoke-run `eval.json` is unchanged (regression gate).
