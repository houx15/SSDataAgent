"""Render a Dashboard payload into a single self-contained HTML file."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ssdataagent.dashboard.model import Dashboard, DashboardExperiment

TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html(dashboard: Dashboard, output_path: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "jinja"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("index.html.jinja")

    payload = _to_jsonable(dashboard)
    champion = next(
        (e for e in dashboard.experiments if e.is_champion),
        None,
    )

    html = template.render(
        experiments=dashboard.experiments,
        champion=champion,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        payload_json=json.dumps(payload, ensure_ascii=False),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _to_jsonable(dashboard: Dashboard) -> dict:
    return {
        "experiments": [_exp_to_jsonable(e) for e in dashboard.experiments],
    }


def _exp_to_jsonable(e: DashboardExperiment) -> dict:
    return {
        "date": e.date,
        "exp_names": e.exp_names,
        "model": e.model,
        "git_sha": e.git_sha,
        "is_pilot": e.is_pilot,
        "is_champion": e.is_champion,
        "is_partial": e.is_partial,
        "headline_text": e.headline_text,
        "hypothesis_text": e.hypothesis_text,
        "what_changed_text": e.what_changed_text,
        "workflow_bullets": e.workflow_bullets,
        "lessons_text": e.lessons_text,
        "overall_mean_full_agent": e.overall_mean_full_agent,
        "by_type": e.by_type,
        "scores_grid": e.scores_grid,
        "retro_link": e.retro_link,
        "configs": [dataclasses.asdict(c) for c in e.configs],
    }
