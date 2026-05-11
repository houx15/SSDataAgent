"""Merge parser outputs into the Dashboard payload."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ssdataagent.dashboard.config import ExperimentConfig
from ssdataagent.dashboard.ledger import LedgerEntry
from ssdataagent.dashboard.results import ExperimentResults
from ssdataagent.dashboard.retros import RetroSections


@dataclass
class DashboardExperiment:
    date: str
    exp_names: list[str]
    model: str
    git_sha: str
    is_pilot: bool

    headline_text: str
    hypothesis_text: str
    what_changed_text: str
    workflow_bullets: list[str]
    lessons_text: str

    configs: list[ExperimentConfig]
    results: list[ExperimentResults]
    retro: RetroSections | None
    retro_link: str

    overall_mean_full_agent: float | None
    by_type: dict[str, float]
    scores_grid: dict[str, dict[str, dict[str, float]]]
    # scores_grid[condition][dataset][type] = pass_rate

    is_partial: bool = False
    is_champion: bool = False


@dataclass
class Dashboard:
    experiments: list[DashboardExperiment] = field(default_factory=list)


ResultsLoader = Callable[[str], ExperimentResults | None]
RetroLoader = Callable[[str], RetroSections | None]


def assemble(
    ledger: list[LedgerEntry],
    configs: dict[str, ExperimentConfig],
    results_loader: ResultsLoader,
    retro_loader: RetroLoader,
) -> Dashboard:
    experiments: list[DashboardExperiment] = []
    for entry in ledger:
        all_results = [results_loader(name) for name in entry.exp_names]
        present_results = [r for r in all_results if r is not None]
        is_partial = (
            len(entry.exp_names) > 1
            and 0 < len(present_results) < len(entry.exp_names)
        )

        exp_configs = [
            configs[name] for name in entry.exp_names if name in configs
        ]
        retro = retro_loader(entry.retro_link) if entry.retro_link else None

        overall_mean, by_type, scores_grid = _aggregate_scores(present_results)

        experiments.append(
            DashboardExperiment(
                date=entry.date,
                exp_names=entry.exp_names,
                model=entry.model,
                git_sha=entry.git_sha,
                is_pilot=entry.is_pilot,
                headline_text=_resolve_headline(entry, retro),
                hypothesis_text=_resolve_hypothesis(entry, retro),
                what_changed_text=_resolve_what_changed(entry, retro),
                workflow_bullets=_resolve_workflow_bullets(retro, exp_configs),
                lessons_text=_resolve_lessons(retro),
                configs=exp_configs,
                results=present_results,
                retro=retro,
                retro_link=entry.retro_link,
                overall_mean_full_agent=overall_mean,
                by_type=by_type,
                scores_grid=scores_grid,
                is_partial=is_partial,
            )
        )

    _mark_champion(experiments)
    return Dashboard(experiments=experiments)


# --- aggregation ---

def _aggregate_scores(
    results: list[ExperimentResults],
) -> tuple[float | None, dict[str, float], dict[str, dict[str, dict[str, float]]]]:
    grid: dict[str, dict[str, dict[str, float]]] = {}
    full_agent_values: list[float] = []
    by_type_values: dict[str, list[float]] = {}

    for r in results:
        for (cond, dset, ttype), val in r.scores.items():
            grid.setdefault(cond, {}).setdefault(dset, {})[ttype] = val
            if cond == "full_agent":
                full_agent_values.append(val)
                by_type_values.setdefault(ttype, []).append(val)

    overall = (
        sum(full_agent_values) / len(full_agent_values)
        if full_agent_values
        else None
    )
    by_type = {
        t: sum(vs) / len(vs) for t, vs in by_type_values.items() if vs
    }
    return overall, by_type, grid


def _mark_champion(entries: list[DashboardExperiment]) -> None:
    candidates = [
        e for e in entries
        if not e.is_pilot and e.overall_mean_full_agent is not None
    ]
    if not candidates:
        return
    # Max mean wins; ties broken by most recent date.
    candidates.sort(
        key=lambda e: (e.overall_mean_full_agent, e.date),
        reverse=True,
    )
    candidates[0].is_champion = True


# --- slot resolution ---

def _resolve_headline(entry: LedgerEntry, retro: RetroSections | None) -> str:
    if entry.headline:
        return entry.headline
    if retro and "Results" in retro.sections:
        first = retro.sections["Results"].strip().splitlines()[:1]
        if first:
            return first[0].strip()
    return "—"


def _resolve_hypothesis(entry: LedgerEntry, retro: RetroSections | None) -> str:
    if retro and retro.frontmatter.get("hypothesis"):
        return retro.frontmatter["hypothesis"]
    if retro and "Hypothesis" in retro.sections:
        return retro.sections["Hypothesis"].strip().splitlines()[0]
    if retro and "Strategy" in retro.sections:
        for chunk in retro.sections["Strategy"].split("\n\n"):
            stripped = chunk.strip()
            if stripped and not stripped.startswith(">"):
                return stripped.splitlines()[0] if stripped else ""
        return retro.sections["Strategy"].strip().splitlines()[0]
    if entry.hypothesis:
        return entry.hypothesis
    return "—"


def _resolve_what_changed(entry: LedgerEntry, retro: RetroSections | None) -> str:
    if retro and "Setup" in retro.sections:
        return retro.sections["Setup"]
    if retro and "Strategy" in retro.sections:
        return retro.sections["Strategy"]
    return entry.hypothesis or "—"


def _resolve_workflow_bullets(
    retro: RetroSections | None,
    configs: list[ExperimentConfig],
) -> list[str]:
    bullets: list[str] = []
    if configs:
        cfg = configs[0]
        if cfg.prompt_variant:
            bullets.append(f"Prompt variant: {cfg.prompt_variant}")
        if cfg.conditions:
            bullets.append(f"Conditions: {', '.join(cfg.conditions)}")
        if cfg.datasets:
            bullets.append(f"Datasets: {', '.join(cfg.datasets)}")
    if retro and "Retro" in retro.bullets:
        for label, value in retro.bullets["Retro"][:3]:
            bullets.append(f"{label}: {value}")
    return bullets


def _resolve_lessons(retro: RetroSections | None) -> str:
    if not retro:
        return ""
    if "Retro" in retro.bullets:
        for label, value in retro.bullets["Retro"]:
            if "lesson" in label.lower():
                return value
    return ""
