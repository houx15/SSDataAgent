#!/usr/bin/env python
"""Build the experiments dashboard from LEDGER + results/ + retros + yaml.

Default: reads docs/experiments/LEDGER.md and writes docs/dashboard/index.html.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ssdataagent.dashboard.config import load_configs
from ssdataagent.dashboard.ledger import parse_ledger
from ssdataagent.dashboard.model import assemble
from ssdataagent.dashboard.render import render_html
from ssdataagent.dashboard.results import load_results
from ssdataagent.dashboard.retros import parse_retro


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("docs/experiments/LEDGER.md"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("config/experiments.yaml"))
    parser.add_argument("--output", type=Path, default=Path("docs/dashboard/index.html"))
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any LEDGER row lacks results or retro")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    log = logging.getLogger("build_dashboard")

    ledger = parse_ledger(args.ledger)
    log.info("parsed %d LEDGER rows", len(ledger))

    configs = load_configs(args.config)
    log.info("loaded %d yaml experiment configs", len(configs))

    warnings: list[str] = []

    def results_loader(name: str):
        r = load_results(args.results_root / name)
        if r is None:
            warnings.append(f"no results for {name}")
        return r

    def retro_loader(rel_path: str):
        if not rel_path:
            return None
        # LEDGER links may be relative to docs/experiments/ ("retros/X.md")
        # or use "../report/X.md" for older pilots. Resolve against the
        # LEDGER's parent directory.
        candidate = (args.ledger.parent / rel_path).resolve()
        if not candidate.exists():
            warnings.append(f"retro not found: {rel_path}")
            return None
        return parse_retro(candidate)

    dashboard = assemble(ledger, configs, results_loader, retro_loader)
    champ = next((e for e in dashboard.experiments if e.is_champion), None)
    log.info("champion: %s", champ.exp_names[0] if champ else "<none>")

    for w in warnings:
        log.warning(w)

    render_html(dashboard, args.output)
    log.info("wrote %s", args.output)

    if args.strict and warnings:
        log.error("strict mode: %d warning(s) -- failing build", len(warnings))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
