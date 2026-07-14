"""Block-donor step family.

The properties under test are exactly the ones the older families broke on cfps:
missingness survives, censoring sentinels survive, the within-block joint (and so
chronology and flag/value coherence) survives, and conditioning does not collapse
into the global marginal when the key is continuous.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssdataagent.agent.tools import dispatch
from ssdataagent.agent.tools.state import build_runtime_state


def _train(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A miniature life-course table with the three properties that broke cfps:
    structural missingness, a string censoring sentinel, and a hard chronological
    tie between two age columns."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], n)
    birth_year = rng.integers(1940, 1995, n)
    # women marry earlier here, so a covariate-matched donor draw must reproduce it
    base = np.where(gender == "Female", 22, 26) + rng.integers(0, 6, n)
    married = rng.random(n) < 0.6
    observed = rng.random(n) < 0.7  # of the married, some ages weren't recorded

    age_marry, age_child = [], []
    for i in range(n):
        if not married[i]:
            age_marry.append("never married")
            age_child.append("never had child")
        elif not observed[i]:
            age_marry.append(np.nan)
            age_child.append(np.nan)
        else:
            m = int(base[i])
            age_marry.append(m)
            age_child.append(m + int(rng.integers(1, 5)))  # child ALWAYS after marriage
    return pd.DataFrame({
        "gender": gender, "birth_year": birth_year,
        "age_at_first_marriage": age_marry, "age_at_first_child": age_child,
    })


@pytest.fixture()
def state(tmp_path: Path):
    st = build_runtime_state(
        workspace=tmp_path, train=_train(), descriptions=None,
        has_data=True, has_descriptions=True, seed=7,
    )
    dispatch(st, "set_generation_order", {"cols": [
        "gender", "birth_year", "age_at_first_marriage", "age_at_first_child",
    ]})
    for c in ("gender", "birth_year"):
        dispatch(st, "fit_marginal", {"col": c})
    return st


def _block(state, **kw):
    args = {"cols": ["age_at_first_marriage", "age_at_first_child"],
            "given": ["gender", "birth_year"]}
    args.update(kw)
    return dispatch(state, "fit_block_donor", args)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def test_block_registers_and_samples_every_column(state):
    out = _block(state)
    assert out.get("registered") is True, out
    sim = state.chain.sample(np.random.default_rng(1), 400)
    assert list(sim.columns) == [
        "gender", "birth_year", "age_at_first_marriage", "age_at_first_child"]
    assert len(sim) == 400


def test_chronology_holds_because_the_block_comes_from_one_donor(state):
    """The whole point. Drawing the two ages independently smears the ordering;
    copying them from a single real person cannot."""
    _block(state)
    sim = state.chain.sample(np.random.default_rng(2), 800)
    m, c = _num(sim["age_at_first_marriage"]), _num(sim["age_at_first_child"])
    both = m.notna() & c.notna()
    assert both.sum() > 50, "need enough complete rows for this to mean anything"
    assert (c[both] > m[both]).all(), "child must never precede marriage"


def test_missingness_and_sentinel_are_reproduced(state):
    """fit_marginal drops NaN at fit time and never re-emits it, so a column that
    is ~28% missing in real came out 0% missing in sim — which silently changes
    which rows survive the eval's dropna. The donor copies the raw value."""
    _block(state)
    sim = state.chain.sample(np.random.default_rng(3), 2000)
    train = state.train_fit

    for col in ("age_at_first_marriage", "age_at_first_child"):
        real_na = train[col].isna().mean()
        sim_na = sim[col].isna().mean()
        assert real_na > 0.1, "fixture must actually have missingness"
        assert abs(sim_na - real_na) < 0.05, f"{col}: NaN rate {sim_na:.2f} vs real {real_na:.2f}"

    real_sent = (train["age_at_first_marriage"] == "never married").mean()
    sim_sent = (sim["age_at_first_marriage"] == "never married").mean()
    assert abs(sim_sent - real_sent) < 0.05


def test_flag_and_value_stay_coherent(state):
    """'never married' in one column implies 'never had child' in the other, in
    every real row. A per-column draw breaks that; a block draw cannot."""
    _block(state)
    sim = state.chain.sample(np.random.default_rng(4), 800)
    never_m = sim["age_at_first_marriage"].astype(str) == "never married"
    never_c = sim["age_at_first_child"].astype(str) == "never had child"
    assert (never_m == never_c).all()


def test_values_are_real_so_the_marginal_is_exact(state):
    """No synthesized values — every emitted age exists in the donor pool. This is
    why the donor family keeps T1 while a regression's Gaussian noise loses it."""
    _block(state)
    sim = state.chain.sample(np.random.default_rng(5), 500)
    real_vals = set(_num(state.train_fit["age_at_first_marriage"]).dropna())
    sim_vals = set(_num(sim["age_at_first_marriage"]).dropna())
    assert sim_vals <= real_vals


def test_conditioning_preserves_a_covariate_association(state):
    """Women marry ~4 years earlier in the fixture. A donor draw matched on gender
    must carry that through; an unconditional draw would wash it out."""
    _block(state)
    sim = state.chain.sample(np.random.default_rng(6), 3000)
    sim["m"] = _num(sim["age_at_first_marriage"])
    gap_sim = (sim.loc[sim.gender == "Male", "m"].mean()
               - sim.loc[sim.gender == "Female", "m"].mean())
    tr = state.train_fit.assign(m=lambda d: _num(d["age_at_first_marriage"]))
    gap_real = (tr.loc[tr.gender == "Male", "m"].mean()
                - tr.loc[tr.gender == "Female", "m"].mean())
    assert gap_real > 2
    assert abs(gap_sim - gap_real) < 1.0, f"sim gap {gap_sim:.2f} vs real {gap_real:.2f}"


def test_thin_cells_fall_back_instead_of_copying_one_person(state):
    """A huge min_cell makes every full-key cell too thin, so every row must fall
    back to a coarser key. It must still produce real, complete output rather than
    latching onto a single donor."""
    out = _block(state, min_cell=10_000)
    assert out.get("registered") is True
    sim = state.chain.sample(np.random.default_rng(7), 300)
    assert sim["age_at_first_marriage"].notna().sum() > 0
    assert _num(sim["age_at_first_marriage"]).nunique() > 3, "must not collapse to one donor"


def test_replace_step_drops_the_whole_block(state):
    """A block only means anything as a unit — dropping one column must not leave
    the others aliased to a step the agent thinks it deleted."""
    _block(state)
    assert state.chain.has("age_at_first_child")
    dispatch(state, "replace_step", {"col": "age_at_first_marriage"})
    assert not state.chain.has("age_at_first_marriage")
    assert not state.chain.has("age_at_first_child")


def test_given_must_precede_the_block(state):
    out = _block(state, cols=["gender"], given=["age_at_first_marriage"])
    assert out.get("error") == "given_after_col"


def test_unknown_column_is_reported(state):
    out = _block(state, cols=["nope"])
    assert out.get("error") == "unknown_column"


def test_meta_records_the_block_and_its_given(state):
    """chain.json must carry the dependency structure — without it a committed
    model cannot be audited or replayed from the artifact."""
    _block(state)
    meta = state.chain.to_meta()
    donors = [s for s in meta["steps"] if s["family"] == "block_donor"]
    assert len(donors) == 1, "a block must appear once, not once per column"
    assert donors[0]["block_cols"] == ["age_at_first_marriage", "age_at_first_child"]
    assert donors[0]["given"] == ["gender", "birth_year"]
