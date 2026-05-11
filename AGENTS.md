# Project Instructions

> Use data analyst agent to perform better in social data analysis and prediction tasks

## Guidelines

### Experiment dashboard

After landing a new experiment (new `results/<exp>/done.flag`, new
`docs/experiments/LEDGER.md` row, new retro), regenerate the dashboard:

```bash
.venv/bin/python scripts/build_dashboard.py
git add docs/dashboard/index.html
git commit -m "dashboard: rebuild after <exp_name>"
```

The dashboard is `docs/dashboard/index.html` — a self-contained HTML
file teammates can `git pull && open` with no install. See
`docs/dashboard/README.md` for what it reads, how the champion is
chosen, and `--verbose` / `--strict` flags.

Hypothesis text shown in the UI comes from the LEDGER `hypothesis`
column for most rows (real retros rarely have a `## Hypothesis`
section), so write a meaningful one-liner there when adding a LEDGER row.

## Shared Memory

**Always write new instructions, rules, and memory to `AGENTS.md` only.**

Never modify `CLAUDE.md` or `GEMINI.md` directly - they only import `AGENTS.md`.
This ensures Claude Code, Codex CLI, and Gemini CLI share the same context consistently.

## Project Structure

- `.claude/agents/` - Custom subagents for specialized tasks
- `.claude/skills/` - Claude Code skills (slash commands)
- `.claude/rules/` - Modular rules auto-loaded into context
- `.codex/skills/` - Codex CLI skills
- `.codex/prompts/` - Codex CLI custom slash commands
- `.gemini/skills/` - Gemini CLI skills
- `.gemini/commands/` - Gemini CLI custom slash commands (TOML)
- `.mcp.json` - MCP server configuration
