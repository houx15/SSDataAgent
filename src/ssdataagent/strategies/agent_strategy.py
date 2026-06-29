from __future__ import annotations

from pathlib import Path

from ssdataagent.agent.context import Condition, build_context
from ssdataagent.agent.orchestrator import Orchestrator
from ssdataagent.experiments.logger import log_run
from ssdataagent.strategies.base import InfoGate, StrategyResult


class AgentStrategy:
    name = "agent"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        unseen = list(gate.unseen_variables)
        ctx = build_context(
            condition=gate.condition,
            dataset_name=gate.dataset_name,
            train_df=gate.fit_microdata(),
            workspace=gate.workspace,
            unseen_variables=unseen if gate.condition is Condition.UNSEEN else None,
        )
        orch = Orchestrator(
            client=gate.client,
            n_rows=cfg.n_rows,
            max_validation_iters=cfg.max_iterations,
            sandbox_timeout=cfg.sandbox_timeout,
            prompt_variant=cfg.prompt_variant,
        )
        result = orch.run(
            condition=gate.condition,
            dataset_name=gate.dataset_name,
            workspace=gate.workspace,
            has_data=ctx.has_data,
            has_descriptions=ctx.has_descriptions,
        )
        log_run(result, run_dir=run_dir)
        return StrategyResult(
            generated=result.generated,
            meta_extras={"unseen_variables": unseen},
        )
