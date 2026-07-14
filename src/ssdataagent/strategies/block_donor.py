"""Block-donor strategy — generate by copying coherent blocks from real donors.

The diagnosis behind this (docs/report/2026-07-13-full-agent-failure-modes.md):
every step family the agent had available destroys something the benchmark
measures. Marginals annihilate missingness; regressions smear the marginal with
Gaussian noise; exact-tuple lookups fall back to the global marginal for 60-95%
of rows once a continuous column is in the key; and any per-column draw smears
the joint life-course ordering.

This strategy fixes all four by construction. Variables are grouped into blocks
using the ``domain`` field SSDataBench's own schema already carries (Demography,
SES, Marriage, Health, Attitude, Ability). Blocks are generated in a causal
order; for each simulated row one real donor is matched on a coarsened key over
already-generated columns, and the donor's whole block is copied verbatim.

  * values are real          -> the marginal is exact (T1)
  * NaN and 'never married' copy over -> occurrence rates, and therefore which
    rows survive the eval's dropna and enter the scored regressions, are right
  * a block moves as a unit  -> chronology and flag/value coherence hold exactly:
    no childless person acquires an age at first birth (T2/T3/T4)
  * later blocks condition on the event ages already generated -> the life-course
    ORDER survives across block boundaries too (T4/T5)

Donors come only from the training microdata the InfoGate hands us. Nothing here
reads the evaluation reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ssdataagent.agent.tools.donor import BlockDonorStep
from ssdataagent.data.event_timing import event_timing_variables
from ssdataagent.data.schema import load_schema
from ssdataagent.strategies.base import InfoGate, StrategyResult

if TYPE_CHECKING:
    from ssdataagent.experiments.runner import ExperimentConfig


# Life-course causal order over the schema's own `domain` labels: demography is
# fixed at birth, schooling and work follow, family formation follows those, and
# health / attitudes / abilities are measured downstream of all of it. Domains a
# dataset doesn't use are skipped; any domain we don't know about is appended.
DOMAIN_ORDER = ["Demography", "SES", "Marriage", "Health", "Attitude", "Attitudes", "Ability"]

# Columns every downstream block matches donors on. These are the covariates the
# T3/T5 configs actually condition on, and they are cheap to match (low
# cardinality once binned), so cells stay populated.
ANCHOR_CANDIDATES = ["gender", "birth_year", "highest_education", "education", "race", "minzu"]


def _blocks_from_schema(
    dataset: str, columns: list[str], granularity: str = "domain",
) -> list[tuple[list[str], list[str]]]:
    """(block_columns, condition_columns) in generation order.

    `granularity` is the central modeling dial, and it is a genuine trade-off
    rather than a tuning constant:

      "domain"  one block per schema domain. Maximal *conditioning* — each block
                is matched on the demography and the event ages already drawn — at
                the cost of splitting the joint across block boundaries.
      "mega"    demography, then every remaining column as ONE block. Maximal
                *joint fidelity*: every pair of non-root variables comes from the
                same real person, so all of T2's pairwise associations are exact.
                The price is that conditioning happens only through demography.

    On cfps "mega" wins T2 (+0.02) and "domain" wins T3 (+0.03) — the dial trades
    pairwise fidelity against conditional fidelity, which is exactly the choice a
    modeler (or the agent) should be making deliberately.
    """
    schema = load_schema(dataset)
    known = [c for c in columns if c in schema.domains or c in schema.background_variables]

    by_domain: dict[str, list[str]] = {}
    for c in known:
        by_domain.setdefault(schema.domains.get(c, "Other"), []).append(c)

    ordered = [d for d in DOMAIN_ORDER if d in by_domain]
    ordered += [d for d in by_domain if d not in ordered]
    if not ordered:
        return []

    roots = by_domain[ordered[0]]
    if granularity == "mega":
        rest = [c for d in ordered[1:] for c in by_domain[d]]
        if not rest:
            return [(roots, [])]
        # Condition on ALL of demography, not a subset. T2 scores every pair, and
        # pairs that straddle the root/rest boundary survive only through this key
        # — a 3-column key leaks them, and plain hotdeck (which copies whole rows,
        # so every pair is exact) beats us on T2 as a result. Known-strong
        # covariates go first so the progressive fallback, which drops from the
        # tail, gives them up last.
        anchors = [a for a in ANCHOR_CANDIDATES if a in roots]
        cond = anchors + [c for c in roots if c not in anchors]
        return [(roots, []), (rest, cond)]

    # Event-age columns are what T4/T5 score. Once generated they join the key for
    # every later block, so a block drawn later cannot contradict the chronology an
    # earlier block already established.
    event_vars = set(event_timing_variables(dataset))

    blocks: list[tuple[list[str], list[str]]] = [(roots, [])]
    generated: list[str] = list(roots)
    for dom in ordered[1:]:
        cols = by_domain[dom]
        anchors = [a for a in ANCHOR_CANDIDATES if a in generated]
        carried_events = [c for c in generated if c in event_vars]
        # Most-discriminating keys first: progressive fallback drops from the tail,
        # so whatever sits at the front survives the longest.
        blocks.append((cols, anchors[:3] + carried_events))
        generated.extend(cols)
    return blocks


class BlockDonorStrategy:
    name = "block_donor"

    def generate(self, gate: InfoGate, run_dir: Path, cfg: "ExperimentConfig") -> StrategyResult:
        donors = gate.fit_microdata()
        if donors is None or donors.empty:
            raise ValueError(
                f"{self.name} needs microdata to draw donors from; condition "
                f"{gate.condition.value!r} withholds it"
            )
        donors = donors.reset_index(drop=True)
        granularity = getattr(cfg, "donor_granularity", "domain")
        min_cell = getattr(cfg, "donor_min_cell", 25)
        blocks = _blocks_from_schema(gate.dataset_name, list(donors.columns), granularity)

        rng = np.random.default_rng(cfg.seed)
        out = pd.DataFrame(index=range(cfg.n_rows))
        meta_blocks = []
        for cols, cond in blocks:
            step = BlockDonorStep(
                col=cols[0], block_cols=list(cols), given=list(cond), min_cell=min_cell,
            ).fit(donors)
            drawn = step.sample_block(rng, cfg.n_rows, out)
            for c in cols:
                out[c] = drawn[c].to_numpy()
            meta_blocks.append({"block": cols, "given": cond})

        return StrategyResult(
            generated=out,
            meta_extras={
                "block_donor": {
                    "granularity": granularity,
                    "blocks": meta_blocks,
                    "n_donors": int(len(donors)),
                    "min_cell": min_cell,
                }
            },
        )
