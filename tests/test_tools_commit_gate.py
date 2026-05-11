"""EXP-006e: commit_generator must reject longitudinal commits that haven't
verified event chronology, and the agent must be told this directly.

The bug we're fixing: in EXP-006c (rubric_tools_v2) the agent fit
`age_at_first_marriage` and friends conditionally on prior events, then
checked `score_overall` (which is per-column-marginal-only), saw a low
score, called `replace_step` to refit them all as marginals, and committed
at a higher score_overall — destroying the chronology that T4 measures.

The recipe told the agent to "call score_event_order before committing"
but that was advisory. This change makes it a hard gate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssdataagent.agent.tools import dispatch
from ssdataagent.agent.tools.fit import (
    Chain,
    fit_conditional,
    fit_marginal,
    set_generation_order,
)
from ssdataagent.agent.tools.state import RuntimeState
from ssdataagent.agent.tools.verify import score_event_order


def _make_state(df: pd.DataFrame, tmp_path: Path) -> RuntimeState:
    """80/20 split, deterministic."""
    perm = np.random.default_rng(0).permutation(len(df))
    cut = int(len(df) * 0.8)
    train = df.iloc[perm[:cut]].reset_index(drop=True)
    held = df.iloc[perm[cut:]].reset_index(drop=True)
    return RuntimeState(
        workspace=tmp_path,
        train=train, train_fit=train, held_out=held,
        descriptions=None, has_data=True, has_descriptions=False,
        chain=Chain(), rng=np.random.default_rng(0),
    )


@pytest.fixture
def cross_state(tmp_path: Path) -> RuntimeState:
    """Cross-sectional: gender + age + income. No event-age columns."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "gender": rng.choice(["F", "M"], size=n),
        "age": rng.integers(20, 70, size=n).astype(float),
        "income": rng.normal(50_000, 15_000, size=n),
    })
    return _make_state(df, tmp_path)


@pytest.fixture
def long_state(tmp_path: Path) -> RuntimeState:
    """Longitudinal: ever_married + age_at_first_marriage + age_at_first_child.
    Two event-age columns means commit_generator should require an event-order
    check before accepting."""
    rng = np.random.default_rng(0)
    n = 300
    age_marriage = rng.integers(20, 35, size=n).astype(float)
    age_first_child = age_marriage + rng.integers(1, 8, size=n).astype(float)
    df = pd.DataFrame({
        "gender": rng.choice(["F", "M"], size=n),
        "ever_married": rng.choice([0, 1], size=n),
        "age_at_first_marriage": age_marriage,
        "age_at_first_child": age_first_child,
    })
    return _make_state(df, tmp_path)


def _fit_cross_chain(s: RuntimeState) -> None:
    set_generation_order(s, ["gender", "age", "income"])
    fit_marginal(s, "gender", family="categorical_empirical")
    fit_marginal(s, "age", family="empirical")
    fit_conditional(s, "income", given=["gender", "age"], family="linear_regression")


def _fit_long_chain(s: RuntimeState) -> None:
    set_generation_order(s, [
        "gender", "ever_married", "age_at_first_marriage", "age_at_first_child",
    ])
    fit_marginal(s, "gender", family="categorical_empirical")
    fit_marginal(s, "ever_married", family="categorical_empirical")
    fit_conditional(
        s, "age_at_first_marriage",
        given=["gender", "ever_married"], family="empirical_lookup",
    )
    fit_conditional(
        s, "age_at_first_child",
        given=["gender", "ever_married", "age_at_first_marriage"],
        family="empirical_lookup",
    )


# ---------- gate behavior ----------


def test_commit_allowed_on_cross_sectional(cross_state):
    """Cross-sectional chain has no event-age columns; commit should not
    require score_event_order."""
    _fit_cross_chain(cross_state)
    out = dispatch(cross_state, "commit_generator", {})
    assert out.get("committed") is True, f"unexpected error: {out}"
    assert cross_state.committed is True


def test_commit_rejected_when_longitudinal_lacks_event_order(long_state):
    """Two event-age columns + no score_event_order call = blocked commit.
    The error must name the gate so the agent self-corrects."""
    _fit_long_chain(long_state)
    out = dispatch(long_state, "commit_generator", {})
    assert out.get("committed") is not True
    assert out.get("error") == "missing_event_order_check"
    # The error must list the event-age columns the agent needs to score.
    detected = set(out.get("event_age_columns", []))
    assert {"age_at_first_marriage", "age_at_first_child"} <= detected
    assert long_state.committed is False


def test_commit_allowed_when_event_order_called(long_state):
    """Same chain, but score_event_order ran first → commit accepted."""
    _fit_long_chain(long_state)
    score_event_order(long_state, ["age_at_first_marriage", "age_at_first_child"])
    out = dispatch(long_state, "commit_generator", {})
    assert out.get("committed") is True, f"unexpected error: {out}"
    assert long_state.committed is True


def test_commit_allowed_when_only_one_event_age_column(tmp_path: Path):
    """One event-age col → no chronology to check → no gate."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "gender": rng.choice(["F", "M"], size=n),
        "age_at_first_marriage": rng.integers(20, 35, size=n).astype(float),
    })
    s = _make_state(df, tmp_path)
    set_generation_order(s, ["gender", "age_at_first_marriage"])
    fit_marginal(s, "gender", family="categorical_empirical")
    fit_marginal(s, "age_at_first_marriage", family="empirical")
    out = dispatch(s, "commit_generator", {})
    assert out.get("committed") is True, f"unexpected error: {out}"


def test_commit_gate_accepts_overlapping_event_order_call(long_state, tmp_path: Path):
    """If score_event_order covers a subset of the detected event-age cols,
    that's enough — the agent is engaging with chronology."""
    rng = np.random.default_rng(1)
    n = 300
    age_m = rng.integers(20, 35, size=n).astype(float)
    age_d = age_m + rng.integers(1, 15, size=n).astype(float)
    age_c = age_m + rng.integers(0, 10, size=n).astype(float)
    df = pd.DataFrame({
        "gender": rng.choice(["F", "M"], size=n),
        "ever_married": rng.choice([0, 1], size=n),
        "age_at_first_marriage": age_m,
        "age_at_first_divorce": age_d,
        "age_at_first_child": age_c,
    })
    s = _make_state(df, tmp_path)
    set_generation_order(s, [
        "gender", "ever_married",
        "age_at_first_marriage", "age_at_first_divorce", "age_at_first_child",
    ])
    fit_marginal(s, "gender", family="categorical_empirical")
    fit_marginal(s, "ever_married", family="categorical_empirical")
    fit_conditional(s, "age_at_first_marriage", given=["gender", "ever_married"], family="empirical_lookup")
    fit_conditional(s, "age_at_first_divorce", given=["age_at_first_marriage"], family="empirical_lookup")
    fit_conditional(s, "age_at_first_child", given=["age_at_first_marriage"], family="empirical_lookup")
    # Score only marriage→child, but there are 3 detected event cols.
    score_event_order(s, ["age_at_first_marriage", "age_at_first_child"])
    out = dispatch(s, "commit_generator", {})
    assert out.get("committed") is True, f"unexpected error: {out}"


# ---------- event-age column detection heuristic ----------


def test_event_age_heuristic_recognizes_common_patterns(tmp_path: Path):
    """The heuristic must recognize the column-name patterns SSDataBench
    uses for event-age fields. Anchor what counts as an event-age column."""
    from ssdataagent.agent.tools.commit import _event_age_columns

    cols = [
        "gender", "age", "income",                    # not event-age
        "age_at_first_marriage", "age_at_first_child",  # event-age
        "age_at_first_divorce",                       # event-age
        "age_finished_education", "age_started_work",  # event-age
        "birth_year",                                 # NOT event-age (timestamp, not an event timestamp on the person's life)
    ]
    detected = _event_age_columns(cols)
    assert "age_at_first_marriage" in detected
    assert "age_at_first_child" in detected
    assert "age_at_first_divorce" in detected
    assert "age_finished_education" in detected
    assert "age_started_work" in detected
    assert "age" not in detected
    assert "income" not in detected
    assert "birth_year" not in detected


# ---------- score_event_order audit trail ----------


def test_score_event_order_records_successful_call(long_state):
    """A successful score_event_order must be visible to commit_generator."""
    _fit_long_chain(long_state)
    out = score_event_order(long_state, ["age_at_first_marriage", "age_at_first_child"])
    assert "compliance_rate" in out, f"expected success, got {out}"
    # The state must remember this call so commit_generator can see it.
    assert hasattr(long_state, "event_order_calls")
    assert any(
        "age_at_first_marriage" in evs and "age_at_first_child" in evs
        for evs in long_state.event_order_calls
    )


def test_score_event_order_error_does_not_record(long_state):
    """A score_event_order that returns an error (bad args, missing col)
    must NOT count toward the gate."""
    _fit_long_chain(long_state)
    # Only one event passed → returns bad_arguments, must not count.
    out = score_event_order(long_state, ["age_at_first_marriage"])
    assert out.get("error") == "bad_arguments"
    if hasattr(long_state, "event_order_calls"):
        for evs in long_state.event_order_calls:
            assert "age_at_first_marriage" not in evs or len(evs) >= 2


# ---------- score_overall must hint when chronology is uncovered ----------


def test_score_overall_hint_present_when_event_age_cols(long_state):
    """score_overall covers per-column marginals only. When the chain has
    event-age cols, it must surface a hint so the agent doesn't treat it
    as the final criterion."""
    _fit_long_chain(long_state)
    out = dispatch(long_state, "score_overall", {})
    # The hint must mention score_event_order so the agent can act on it.
    assert "chronology_hint" in out, f"missing hint in {out}"
    assert "score_event_order" in out["chronology_hint"]


def test_score_overall_no_hint_when_no_event_age_cols(cross_state):
    """Cross-sectional chains should not get the chronology hint — it'd be
    misleading to say 'check chronology' when there's nothing to check."""
    _fit_cross_chain(cross_state)
    out = dispatch(cross_state, "score_overall", {})
    assert "chronology_hint" not in out


# ---------- EXP-006f: forced-commit bypass for max_turns escape hatch ----------


def test_commit_bypasses_gate_when_t4_unverified_flag_set(long_state):
    """When the orchestrator hits max_turns with the chronology gate as the
    only blocker, it sets `state.t4_unverified = True` and retries commit.
    The retry must succeed and surface the flag so downstream reporting
    can mark this run as having an unverified T4."""
    _fit_long_chain(long_state)

    # First commit blocked by gate (control).
    blocked = dispatch(long_state, "commit_generator", {})
    assert blocked.get("error") == "missing_event_order_check"

    # Orchestrator escape hatch: force-commit by flagging the state.
    long_state.t4_unverified = True
    out = dispatch(long_state, "commit_generator", {})
    assert out.get("committed") is True, f"expected committed, got {out}"
    assert out.get("t4_unverified") is True, (
        "result must surface t4_unverified so reporting can flag the run"
    )
    # The warning string must mention score_event_order so a human reader
    # of tool_calls.json can see what was skipped.
    assert "score_event_order" in (out.get("warning", "") or "")
    assert long_state.committed is True


def test_commit_with_t4_unverified_flag_does_not_warn_on_cross_sectional(cross_state):
    """The flag is harmless on chains that wouldn't have hit the gate anyway —
    no warning should be added, since chronology was never relevant."""
    _fit_cross_chain(cross_state)
    cross_state.t4_unverified = True  # simulate orchestrator setting it
    out = dispatch(cross_state, "commit_generator", {})
    assert out.get("committed") is True
    # Cross-sectional chain: no chronology to skip, no warning to surface.
    assert out.get("t4_unverified") is not True
    assert "warning" not in out


def test_commit_t4_unverified_does_not_bypass_other_errors(long_state):
    """The flag must ONLY bypass the chronology gate. An empty chain or
    unknown column should still error — those represent genuinely broken
    chains, not 'we ran out of time'."""
    long_state.t4_unverified = True
    # Empty chain: still an error, flag or not.
    out = dispatch(long_state, "commit_generator", {})
    assert out.get("error") == "empty_chain"
    assert long_state.committed is False


# ---------- EXP-006f follow-up: chain-consistency guard ----------


def test_commit_rejects_when_step_given_not_in_order(tmp_path: Path):
    """If a conditional Step's `given` column was dropped from
    generation_order (e.g. because set_generation_order was called again with
    a different list), commit must refuse rather than let sample() crash.

    Reproduces the addhealth failure in exp006f_tools_diag: the agent
    fit_copy_real'd several age cols (auto-extending the order), then issued
    set_generation_order with a list missing age_at_first_marriage. The chain
    committed anyway (n_steps=23, generation_order=20) and sample() raised
    KeyError on the conditional whose given still referenced the dropped col.
    """
    rng = np.random.default_rng(0)
    n = 300
    age_m = rng.integers(20, 35, size=n).astype(float)
    df = pd.DataFrame({
        "gender": rng.choice(["F", "M"], size=n),
        "age_at_first_marriage": age_m,
        "age_at_first_child": age_m + rng.integers(1, 8, size=n).astype(float),
    })
    s = _make_state(df, tmp_path)
    set_generation_order(s, ["gender", "age_at_first_marriage", "age_at_first_child"])
    fit_marginal(s, "gender", family="categorical_empirical")
    fit_marginal(s, "age_at_first_marriage", family="empirical")
    fit_conditional(
        s, "age_at_first_child",
        given=["age_at_first_marriage"], family="empirical_lookup",
    )
    score_event_order(s, ["age_at_first_marriage", "age_at_first_child"])
    # Now corrupt the chain by hand the way an over-eager set_generation_order
    # would (validation is added in the same change, so we bypass it here to
    # exercise the commit-side guard explicitly).
    s.chain.generation_order = ["gender", "age_at_first_child"]
    out = dispatch(s, "commit_generator", {})
    assert out.get("error") == "inconsistent_chain"
    assert "age_at_first_marriage" in out.get("details", "")
    assert s.committed is False
