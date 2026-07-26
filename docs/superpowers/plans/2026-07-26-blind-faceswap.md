# Blind Face-swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A description-only cross-context generator — transfer source A's copula structure, and get the target's marginals ("features") from an LLM that reads only B's textual description — scored on SSDataBench T1–T5 against held-out B.

**Architecture:** Reuse the existing `transfer_build(struct=A, marg, …, "marginal-swap")` engine unchanged. The one new move: `marg` is synthesized from **LLM-elicited distributions** (from B's audited description) instead of B's data. Four units — a pure marginal-frame synthesizer, an audited description module, an LLM elicitation+cache function, and an orchestrator that scores the face-swap alongside deterministic reference configs.

**Tech Stack:** Python, numpy, pandas; OpenRouter (`anthropic/claude-sonnet-4.5`) via the OpenAI client for elicitation (cached, LLM-free at score time); existing `nodonor_bracket` scorer; pytest.

## Global Constraints

- **Firewall (stricter than the whole B0–B6 ladder):** the `FS_*` generation paths read ONLY source A's microdata + B's *audited textual description* + the variable schema. They never read B's marginals, X-margins, covariate-R², reference sample, or any numeric aggregate of B. Only `ref_oracle_comp` (a labeled upper-bound reference) and the scorer touch B's pool.
- **Structure vs features split:** numeric-ness, the category universe, and each column's **missingness rate** come from **source A** (transferred structure). Only the distribution SHAPE (category proportions / numeric quantiles) comes from the LLM (features).
- **Comparability:** `cols/covs/outs` are derived identically to `transfer_map.run_layer2` (source ∩ B-pool ∩ reference). `FS_carryover` uses the identical call as `B0_carryover`, and `ref_oracle_comp` the identical call as `B1_marginal_swap`.
- **Elicitation cached & pinned:** LLM outputs cached under `results/blind_cache/` (gitignored, durable); `MODEL = "anthropic/claude-sonnet-4.5"`; the prompt asks ONLY for marginal distributions (never a joint, never per-person rows). LLM-free at score time.
- **`results/` is gitignored** — scored CSVs and the cache are not committed.
- **Do not stage the untracked `ssdatabench` submodule.**
- **Scoring protocol (unchanged from B0–B6):** 3 seeds, `n=3000`, `bootstrap_B=200`, both benchmark pairs, same scorer/noise floor.

---

### Task 1: `build_marg_frame` — synthesize a marginal frame from elicited distributions (pure)

**Files:**
- Create: `src/ssdataagent/transfer/blind.py`
- Test: `tests/test_transfer_blind.py`

**Interfaces:**
- Consumes: `ssdataagent.transfer.generate._is_numeric`.
- Produces:
  - `_synth_numeric(quantiles: list[float], L: int) -> np.ndarray`
  - `_synth_categorical(probs: dict, L: int) -> np.ndarray`
  - `build_marg_frame(elicited: dict, source_a: pd.DataFrame, cols: list[str], *, L: int = 4000, seed: int = 0) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transfer_blind.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def test_synth_numeric_matches_quantiles():
    from ssdataagent.transfer.blind import _synth_numeric
    col = _synth_numeric([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], L=5000)
    assert len(col) == 5000
    assert abs(np.quantile(col, 0.5) - 5.0) < 0.2
    assert col.min() >= -0.1 and col.max() <= 10.1


def test_synth_categorical_matches_probs_and_length():
    from ssdataagent.transfer.blind import _synth_categorical
    col = _synth_categorical({"a": 0.5, "b": 0.3, "c": 0.2}, L=1000)
    assert len(col) == 1000
    vc = pd.Series(col).value_counts(normalize=True)
    assert abs(vc["a"] - 0.5) < 0.01 and abs(vc["b"] - 0.3) < 0.01


def test_build_marg_frame_uses_elicited_and_carries_A_missingness():
    from ssdataagent.transfer.blind import build_marg_frame
    a = pd.DataFrame({
        "age": [20, 30, 40, 50, np.nan, 60, 70, 80, 25, 35],   # numeric, 10% missing
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],  # categorical, 0% missing
    })
    elicited = {
        "age": {"quantiles": [18, 22, 30, 40, 50, 60, 65, 70, 75, 80, 90]},
        "sex": {"probs": {"M": 0.7, "F": 0.3}},
    }
    frame = build_marg_frame(elicited, a, ["age", "sex"], L=2000, seed=0)
    # elicited proportions win for sex
    vc = frame["sex"].dropna().astype(str).value_counts(normalize=True)
    assert abs(vc["M"] - 0.7) < 0.02
    # A's missingness RATE is carried (age ~10%, sex ~0%)
    assert abs(frame["age"].isna().mean() - 0.1) < 0.02
    assert frame["sex"].isna().mean() < 0.001
    # elicited numeric level wins (median ~60 from the quantiles, not A's ~40)
    assert abs(np.nanmedian(pd.to_numeric(frame["age"])) - 60) < 5


def test_build_marg_frame_falls_back_to_A_when_missing():
    from ssdataagent.transfer.blind import build_marg_frame
    a = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    frame = build_marg_frame({}, a, ["x"], L=1000, seed=0)   # nothing elicited -> carry A
    assert len(frame) == 1000
    assert 1 <= np.nanmedian(pd.to_numeric(frame["x"])) <= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_blind.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.transfer.blind'`.

- [ ] **Step 3: Write the implementation**

Create `src/ssdataagent/transfer/blind.py`:

```python
"""Blind face-swap (Approach A): transfer source A's copula, get the target's marginals
from an LLM that reads only the target's textual description. See
docs/superpowers/specs/2026-07-26-blind-faceswap-design.md.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ssdataagent.transfer.generate import _is_numeric

_logger = logging.getLogger(__name__)


def _synth_numeric(quantiles: list[float], L: int) -> np.ndarray:
    """Length-L numeric column whose empirical distribution matches ``quantiles`` (values at
    evenly spaced probabilities 0..1). Inverse-CDF interpolation on a regular p-grid."""
    q = np.sort(np.asarray(quantiles, dtype=float))
    ps = np.linspace(0.0, 1.0, len(q))
    grid = (np.arange(L) + 0.5) / L
    return np.interp(grid, ps, q)


def _synth_categorical(probs: dict, L: int) -> np.ndarray:
    """Length-L object column whose value_counts(normalize) match ``probs`` (largest-remainder
    rounding, so the length is exactly L and the result is deterministic)."""
    cats = np.array(list(probs.keys()), dtype=object)
    p = np.asarray([probs[c] for c in probs.keys()], dtype=float)
    p = p / p.sum()
    exact = p * L
    counts = np.floor(exact).astype(int)
    rem = int(L - counts.sum())
    if rem > 0:
        counts[np.argsort(-(exact - counts))[:rem]] += 1
    return np.repeat(cats, counts)


def build_marg_frame(elicited: dict, source_a: pd.DataFrame, cols: list[str], *,
                     L: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthesize the ``marg`` frame for transfer_build from LLM-elicited distributions.
    Numeric-ness, the category universe, and each column's missingness RATE come from
    ``source_a`` (transferred structure); the distribution SHAPE comes from ``elicited``.
    A column absent/malformed in ``elicited`` falls back to A's own marginal (carry-over)."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        num = _is_numeric(source_a[c])
        dist = elicited.get(c)
        try:
            if dist is None:
                raise ValueError("no elicited distribution")
            col = (_synth_numeric(dist["quantiles"], L) if num
                   else _synth_categorical(dist["probs"], L)).astype(object)
        except (KeyError, ValueError, TypeError) as e:
            _logger.warning("blind: column %r falls back to A's marginal (%s)", c, e)
            vals = source_a[c].dropna().to_numpy()
            col = (vals[rng.integers(0, len(vals), L)].astype(object) if len(vals)
                   else np.full(L, np.nan, dtype=object))
        miss = float(source_a[c].isna().mean())          # carry A's missingness rate
        if miss > 0:
            k = int(round(miss * L))
            if k > 0:
                col = col.copy()
                col[rng.choice(L, min(k, L), replace=False)] = np.nan
        out[c] = col
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_blind.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/blind.py tests/test_transfer_blind.py
git commit -m "feat(transfer): blind face-swap marginal-frame synthesizer"
```

---

### Task 2: `blind_specs.py` — audited, description-only context specs + firewall audit

**Files:**
- Create: `src/ssdataagent/transfer/blind_specs.py`
- Test: `tests/test_transfer_blind_specs.py`

**Interfaces:**
- Consumes: `ssdataagent.transfer.b3_specs.SPECS` (starting prose).
- Produces: `BLIND_SPECS: dict[str, BlindSpec]` with `.population`, `.description`, `.glosses`; `AUDIT_NOTES: dict[str, list[str]]`.

**Firewall rationale:** B3's prose is authored to be non-circular but carries a few target-sample statistics in LLM-visible strings (the gss `child_number` gloss ends `"...0..8, mean ~1.8"`). This module strips every such number, keeping only qualitative structure and public/general knowledge, and records each removal in `AUDIT_NOTES`. The test is a firewall regression: the scrubbed tokens must be absent.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transfer_blind_specs.py`:

```python
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src")]


def test_blind_specs_cover_both_datasets_with_fields():
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    for ds in ("cps", "gss"):
        s = BLIND_SPECS[ds]
        assert s.population and isinstance(s.population, str)
        assert s.description and isinstance(s.description, str)
        assert isinstance(s.glosses, dict) and s.glosses


def test_firewall_scrubbed_target_sample_numbers_are_gone():
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    # The gss child_number gloss must no longer quote the pool mean ("1.8").
    gss_text = BLIND_SPECS["gss"].description + " " + " ".join(BLIND_SPECS["gss"].glosses.values())
    assert "1.8" not in gss_text and "pool mean" not in gss_text.lower()
    # cps: no household-roster sample mean leaked into LLM-visible strings.
    cps_text = BLIND_SPECS["cps"].description + " " + " ".join(BLIND_SPECS["cps"].glosses.values())
    assert "0.66" not in cps_text


def test_audit_notes_document_removals():
    from ssdataagent.transfer.blind_specs import AUDIT_NOTES
    assert AUDIT_NOTES.get("gss") and AUDIT_NOTES.get("cps")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_blind_specs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ssdataagent.transfer.blind_specs'`.

- [ ] **Step 3: Write the implementation**

Create `src/ssdataagent/transfer/blind_specs.py`. Reuse B3's `population` and `rules` verbatim EXCEPT for the audited removals; audit the `glosses` to strip sample statistics. (The cps `rules`/`glosses` carry no LLM-visible sample number today — the `0.66` lives only in a code comment — so cps is reused as-is; the audit note records that this was checked. The gss `child_number` gloss is edited.)

```python
"""Audited, description-only context specs for the blind face-swap. Every number derived
from the TARGET's sample has been removed from LLM-visible strings; only public/general
knowledge and qualitative structure remain. AUDIT_NOTES records each check/removal.

See docs/superpowers/specs/2026-07-26-blind-faceswap-design.md ("Firewall audit").
"""
from __future__ import annotations

from dataclasses import dataclass

from ssdataagent.transfer.b3_specs import SPECS as _B3


@dataclass(frozen=True)
class BlindSpec:
    population: str          # prose name of the context for the prompt
    description: str         # audited coherence rules / qualitative structure
    glosses: dict            # audited per-variable definitions


# --- gss: strip the pool mean from the child_number gloss --------------------
_gss_glosses = dict(_B3["gss"].glosses)
_gss_glosses["child_number"] = (
    "total number of children EVER BORN in the respondent's lifetime (GSS lifetime "
    "fertility, NOT resident children); a small non-negative count"
)

BLIND_SPECS: dict[str, BlindSpec] = {
    "cps": BlindSpec(_B3["cps"].population, _B3["cps"].rules, dict(_B3["cps"].glosses)),
    "gss": BlindSpec(_B3["gss"].population, _B3["gss"].rules, _gss_glosses),
}

AUDIT_NOTES: dict[str, list[str]] = {
    "cps": [
        "checked population/rules/glosses: no target-sample statistic appears in any "
        "LLM-visible string (the '0.66' resident-child mean lives only in a b3_specs "
        "code comment, not in rules/glosses); reused verbatim.",
    ],
    "gss": [
        "removed 'mean ~1.8' and the '0..8' range from the child_number gloss "
        "(both are target-sample statistics); replaced with a qualitative descriptor.",
    ],
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_blind_specs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/blind_specs.py tests/test_transfer_blind_specs.py
git commit -m "feat(transfer): audited description-only context specs (firewall)"
```

---

### Task 3: `elicit_marginals` — LLM elicitation of B's marginals + durable cache

**Files:**
- Modify: `src/ssdataagent/transfer/blind.py` (append)
- Test: `tests/test_transfer_blind.py` (append)

**Interfaces:**
- Consumes: `blind_specs.BLIND_SPECS`, `generate._is_numeric`, `nodonor_bracket._drop_unnamed` (indirect), `ssdataagent.data.conditional_variance._extract_last_json`.
- Produces:
  - `MODEL = "anthropic/claude-sonnet-4.5"`
  - `elicit_prompt(ds: str, source_a: pd.DataFrame, cols: list[str], *, max_cats: int = 20) -> str`
  - `parse_marginals(text: str, source_a: pd.DataFrame, cols: list[str]) -> dict`
  - `elicit_marginals(ds: str, source_a: pd.DataFrame, cols: list[str], *, client=None, cache_dir: Path | None = None, regenerate: bool = False) -> dict`

Firewall note: the category UNIVERSE handed to the LLM comes from `source_a` (allowed — codebook-level), never from B. The LLM supplies only the probabilities/quantiles.

- [ ] **Step 1: Write the failing tests (append to `tests/test_transfer_blind.py`)**

```python
class _StubMsg:
    def __init__(self, content): self.message = type("M", (), {"content": content})
class _StubResp:
    def __init__(self, content): self.choices = [_StubMsg(content)]
class _StubClient:
    def __init__(self, content): self._c = content; self.calls = 0
    @property
    def chat(self):
        outer = self
        class _Chat:
            class completions:
                @staticmethod
                def create(**kw):
                    outer.calls += 1
                    return _StubResp(outer._c)
        return _Chat()


def test_elicit_marginals_parses_and_caches(tmp_path):
    from ssdataagent.transfer.blind import elicit_marginals
    a = pd.DataFrame({"age": [20, 30, 40, 50, 60], "sex": ["M", "F", "M", "F", "M"]})
    payload = ('{"age": {"quantiles": [18,22,30,40,50,60,65,70,75,80,90]}, '
               '"sex": {"probs": {"M": 0.7, "F": 0.3}}}')
    client = _StubClient(payload)
    got = elicit_marginals("gss", a, ["age", "sex"], client=client,
                           cache_dir=tmp_path, regenerate=True)
    assert set(got) == {"age", "sex"}
    assert abs(got["sex"]["probs"]["M"] - 0.7) < 1e-9
    assert len(got["age"]["quantiles"]) == 11
    assert (tmp_path / "gss_marginals.json").exists()
    assert client.calls == 1
    # second call with the cache warm must NOT hit the client
    client2 = _StubClient(payload)
    again = elicit_marginals("gss", a, ["age", "sex"], client=client2, cache_dir=tmp_path)
    assert client2.calls == 0 and again["sex"]["probs"]["M"] == got["sex"]["probs"]["M"]


def test_elicit_prompt_lists_categories_from_A_not_B():
    from ssdataagent.transfer.blind import elicit_prompt
    a = pd.DataFrame({"age": [20, 30, 40], "sex": ["M", "F", "M"]})
    p = elicit_prompt("gss", a, ["age", "sex"])
    assert "quantiles" in p and "probs" in p
    assert "M" in p and "F" in p            # category universe surfaced from A
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfer_blind.py -k "elicit" -v`
Expected: FAIL (`elicit_marginals` / `elicit_prompt` not defined).

- [ ] **Step 3: Write the implementation (append to `src/ssdataagent/transfer/blind.py`)**

```python
import json
import os
from pathlib import Path

MODEL = "anthropic/claude-sonnet-4.5"
_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO / "results" / "blind_cache"


def elicit_prompt(ds: str, source_a: pd.DataFrame, cols: list[str], *,
                  max_cats: int = 20) -> str:
    """Prompt asking ONLY for per-variable marginal distributions of the described target
    context. Categorical variables list their category universe (from source A, the
    codebook level); numeric variables ask for 11 quantiles at probabilities 0.0..1.0."""
    from ssdataagent.transfer.blind_specs import BLIND_SPECS
    spec = BLIND_SPECS[ds]
    lines = []
    for c in cols:
        gloss = spec.glosses.get(c, c)
        if _is_numeric(source_a[c]):
            lines.append(f'- "{c}" (NUMERIC): {gloss}. Give "quantiles": 11 values at '
                         f'probabilities 0.0,0.1,...,1.0 (min..max).')
        else:
            cats = source_a[c].dropna().astype(str).value_counts().index.tolist()[:max_cats]
            lines.append(f'- "{c}" (CATEGORICAL, categories={cats}): {gloss}. '
                         f'Give "probs": a probability for each category (summing to ~1).')
    body = "\n".join(lines)
    return (
        f"You are estimating the population marginals of {spec.population}.\n"
        f"Context and coherence structure:\n{spec.description}\n\n"
        f"Using ONLY your knowledge of this described population — no external data — "
        f"estimate the marginal distribution of EACH variable below. Do not model any "
        f"joint relationship; marginals only.\n\n{body}\n\n"
        f'Reply with ONE JSON object keyed by variable name, each value either '
        f'{{"quantiles": [...]}} (numeric) or {{"probs": {{cat: p, ...}}}} (categorical). '
        f"Output only the JSON."
    )


def parse_marginals(text: str, source_a: pd.DataFrame, cols: list[str]) -> dict:
    """Parse the LLM's JSON into {var: dist}. Keeps only well-formed entries for ``cols``;
    a numeric var needs a non-empty ``quantiles`` list, a categorical var a non-empty
    ``probs`` dict. Malformed/absent entries are dropped (build_marg_frame then carries A)."""
    import ssdataagent.data.conditional_variance as cv
    try:
        raw = cv._extract_last_json(text)
    except Exception:                                    # noqa: BLE001 -- robust to junk
        return {}
    out: dict = {}
    for c in cols:
        d = raw.get(c) if isinstance(raw, dict) else None
        if not isinstance(d, dict):
            continue
        if _is_numeric(source_a[c]):
            q = d.get("quantiles")
            if isinstance(q, list) and len(q) >= 2:
                out[c] = {"quantiles": [float(x) for x in q]}
        else:
            pr = d.get("probs")
            if isinstance(pr, dict) and pr:
                out[c] = {"probs": {str(k): float(v) for k, v in pr.items()}}
    return out


def elicit_marginals(ds: str, source_a: pd.DataFrame, cols: list[str], *,
                     client=None, cache_dir: Path | None = None,
                     regenerate: bool = False) -> dict:
    """Elicit B's marginals from its description (LLM), cached to
    ``<cache_dir>/<ds>_marginals.json`` (durable, gitignored). Reads no B data. When the
    cache is warm and ``regenerate`` is False, returns it without calling the LLM."""
    cache_dir = cache_dir or _DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ds}_marginals.json"
    if path.exists() and not regenerate:
        return json.loads(path.read_text())
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
    prompt = elicit_prompt(ds, source_a, cols)
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}])
    parsed = parse_marginals(resp.choices[0].message.content, source_a, cols)
    path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    return parsed
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfer_blind.py -v`
Expected: PASS (all Task 1 + Task 3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ssdataagent/transfer/blind.py tests/test_transfer_blind.py
git commit -m "feat(transfer): LLM marginal elicitation (blind, cached)"
```

---

### Task 4: `scripts/transfer_faceswap.py` — the orchestrator

**Files:**
- Create: `scripts/transfer_faceswap.py`
- Test: `tests/test_transfer_faceswap.py`

**Interfaces:**
- Consumes: `pairs.PAIRS`, `pairs.load_pair`, `pairs.covariates_outcomes`, `generate.transfer_build`, `blind.{elicit_marginals, build_marg_frame}`, `scoring.{restrict_config_dir, mean_scores}`, `nodonor_bracket.{_drop_unnamed, carve_pool, build, score, TYPES}`, `load_schema`.
- Produces: `run_faceswap(pair, *, seeds, n, bootstrap_B) -> pd.DataFrame`; writes `results/transfer_map/faceswap_<pair.id>.csv`. `elicit_marginals` is imported at module level so tests can monkeypatch it.

Configs scored (all on the crosswalk cols, restricted config): `FS_carryover` (== B0), `FS_llm` (the face-swap), `ref_oracle_comp` (== B1, reads B — labeled upper bound), `ref_floor`, `ref_ceiling`.

- [ ] **Step 1: Write the failing wiring test**

Create `tests/test_transfer_faceswap.py`. The test monkeypatches `elicit_marginals` so no LLM/network is touched; it returns a stub built from A's own quantiles/probs so `build_marg_frame` and the downstream scoring run for real at tiny n.

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]


def _stub_elicit(ds, source_a, cols, **kw):
    from ssdataagent.transfer.generate import _is_numeric
    out = {}
    for c in cols:
        if _is_numeric(source_a[c]):
            v = pd.to_numeric(source_a[c], errors="coerce").dropna()
            out[c] = {"quantiles": [float(v.quantile(p)) for p in np.linspace(0, 1, 11)]}
        else:
            pr = source_a[c].dropna().astype(str).value_counts(normalize=True)
            out[c] = {"probs": {k: float(v) for k, v in pr.items()}}
    return out


def test_run_faceswap_scores_all_configs(tmp_path, monkeypatch):
    import transfer_faceswap
    from ssdataagent.transfer.pairs import PAIRS
    pair = [p for p in PAIRS if p.id == "gss_1994_2018"][0]
    monkeypatch.setattr(transfer_faceswap, "OUT", tmp_path)
    monkeypatch.setattr(transfer_faceswap, "elicit_marginals", _stub_elicit)
    df = transfer_faceswap.run_faceswap(pair, seeds=1, n=200, bootstrap_B=20)
    assert set(df["config"]) == {"FS_carryover", "FS_llm", "ref_oracle_comp",
                                 "ref_floor", "ref_ceiling"}
    for col in ("T1", "T2", "T3", "overall"):
        assert df[col].notna().all() and (df[col] >= 0).all() and (df[col] <= 1).all()
    assert (tmp_path / "faceswap_gss_1994_2018.csv").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transfer_faceswap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'transfer_faceswap'`.

- [ ] **Step 3: Write `scripts/transfer_faceswap.py`**

```python
#!/usr/bin/env python
"""Blind face-swap (Approach A) -- description-only cross-context generation. Transfer
source A's copula; get the target's marginals from an LLM reading only the target's
audited description. Scores FS_llm alongside deterministic references. See
docs/superpowers/specs/2026-07-26-blind-faceswap-design.md.

    export OPENROUTER_API_KEY=...        # first run only; elicitation is cached
    .venv/bin/python scripts/transfer_faceswap.py gss_1994_2018 --seeds 3 --n 3000 --bootstrap-B 200
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ssdataagent.data.schema import load_schema  # noqa: E402
from ssdataagent.transfer.blind import build_marg_frame, elicit_marginals  # noqa: E402
from ssdataagent.transfer.generate import transfer_build  # noqa: E402
from ssdataagent.transfer.pairs import PAIRS, covariates_outcomes, load_pair  # noqa: E402
from ssdataagent.transfer.scoring import mean_scores, restrict_config_dir  # noqa: E402

OUT = REPO / "results" / "transfer_map"


def run_faceswap(pair, *, seeds, n, bootstrap_B):
    """Score FS_carryover (==B0), FS_llm (blind face-swap), ref_oracle_comp (==B1, reads B),
    ref_floor, ref_ceiling -- all on the crosswalk cols, identical protocol to B0-B6.
    The FS_* paths read only source A + B's description; only ref_oracle_comp and the
    scorer touch B's pool."""
    import nodonor_bracket as nb

    ds = pair.target_dataset
    a = nb._drop_unnamed(pd.read_csv(pair.source_csv, low_memory=False))
    schema = load_schema(ds)
    ref = nb._drop_unnamed(pd.read_csv(schema.real_data_path, low_memory=False))
    b_pool, guarantee = nb.carve_pool(ds)
    _, _, cols = load_pair(pair)
    cols = [c for c in cols if c in a.columns and c in b_pool.columns and c in ref.columns]
    covs, outs = covariates_outcomes(pair.schema_name, cols)
    types = nb.TYPES.get(ds, (1, 2, 3))

    # Blind elicitation (cached) -> synthetic marginal frame. Reads source A + description.
    elicited = elicit_marginals(ds, a, cols)
    n_elicited = sum(1 for c in cols if c in elicited)
    llm_marg = build_marg_frame(elicited, a, cols)
    print(f"{pair.id}: elicited {n_elicited}/{len(cols)} marginals; "
          f"{len(cols) - n_elicited} fell back to source A", flush=True)

    configs = {
        "FS_carryover":    lambda s: transfer_build(a, a, cols, n, s, "carryover"),
        "FS_llm":          lambda s: transfer_build(a, llm_marg, cols, n, s, "marginal-swap"),
        "ref_oracle_comp": lambda s: transfer_build(a, b_pool, cols, n, s, "marginal-swap"),
        "ref_floor":       lambda s: nb.build(b_pool, cols, n, s, "independence"),
        "ref_ceiling":     lambda s: nb.build(b_pool, cols, n, s, "rowresample"),
    }

    out_rows = []
    with tempfile.TemporaryDirectory() as cfg_td:
        cfg_dir = restrict_config_dir(schema.ssdatabench_sim_subdir, set(cols), types,
                                      Path(cfg_td))
        for name, builder in configs.items():
            recs = [nb.score(builder(s), ds, ref, types, seed=1000 + s,
                             bootstrap_B=bootstrap_B, config_dir=cfg_dir)
                    for s in range(1, seeds + 1)]
            row = {"pair": pair.id, "config": name, "guarantee": guarantee,
                   "n_elicited": n_elicited if name == "FS_llm" else ""}
            row.update(mean_scores(pd.DataFrame(recs)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"faceswap_{pair.id}.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", choices=[p.id for p in PAIRS if p.scored])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bootstrap-B", type=int, default=200)
    a = ap.parse_args()
    pair = [p for p in PAIRS if p.id == a.pair][0]
    df = run_faceswap(pair, seeds=a.seeds, n=a.n, bootstrap_B=a.bootstrap_B)
    print(df.to_string(index=False))
    print(f"\nwrote {OUT / f'faceswap_{pair.id}.csv'}")
    print("REGIME: blind. FS_* read only source A + the target's audited text description; "
          "ref_oracle_comp/scorer read B's pool (labeled upper bound / yardstick).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transfer_faceswap.py -v`
Expected: PASS. Runtime is seconds — the stub skips the LLM; scoring is tiny-n.

- [ ] **Step 5: Commit**

```bash
git add scripts/transfer_faceswap.py tests/test_transfer_faceswap.py
git commit -m "feat(transfer): blind face-swap orchestrator"
```

---

## Post-implementation (controller-run, not a subagent task)

Depends on a real LLM elicitation (needs `OPENROUTER_API_KEY`) and heavy scoring (reaped on the box):

1. **Elicit + score both pairs** end-to-end (elicitation cached durably under `results/blind_cache/`; heavy scoring via a resumable per-(config,seed) scorer mirroring `.superpowers/sdd/b6_incremental.py`). Protocol 3 seeds / n=3000 / B=200. Sanity checks: `FS_carryover` overall must match the ladder's `B0_carryover`, and `ref_oracle_comp` must match `B1_marginal_swap`, within noise (bit-identical if seeds align).
2. **Read the key comparisons:** `FS_llm` vs `FS_carryover` (does LLM composition add signal?), `FS_llm` vs `ref_oracle_comp` (price of blind composition), and per-type T1 (features) vs T2/T3 (transferred structure).
3. **Write the report** `docs/report/2026-07-26-blind-faceswap.md` — frame as the honest description-only regime; report the blind→given gap; note the elicitation model + that descriptions were firewall-audited.
4. **LEDGER row** `blind_faceswap` (newest on top) with a meaningful hypothesis; **rebuild dashboard**; **update memory**.

If `OPENROUTER_API_KEY` is unavailable, land the deterministic references (`FS_carryover`, `ref_*`) and report `FS_llm` as pending elicitation.

---

## Self-Review

**Spec coverage:** blind firewall — FS paths read only A + audited description (Task 4 config wiring; Task 2 audit) ✓; face-swap engine reuse (`transfer_build`, Task 4) ✓; LLM-elicited marginals synthesized into the marg frame (Tasks 1+3) ✓; composition ablation `FS_llm` vs `FS_carryover` vs `ref_oracle_comp` (Task 4 configs) ✓; comparability (`FS_carryover`==B0, `ref_oracle_comp`==B1 calls) ✓; elicitation cached/pinned (Task 3) ✓; firewall audit of descriptions (Task 2 + test) ✓; report/LEDGER/dashboard/memory (post-impl) ✓; mechanism deltas explicitly deferred (spec follow-on; not in any task) ✓.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; every test asserts.

**Type consistency:** `elicit_marginals(ds, source_a, cols, *, client, cache_dir, regenerate) -> dict` used identically in Task 3 tests and Task 4 orchestrator (monkeypatched by name). `build_marg_frame(elicited, source_a, cols, *, L, seed) -> DataFrame` consumed in Task 4. Distribution dict shape (`{"quantiles":[...]}` / `{"probs":{...}}`) is produced by `parse_marginals`, consumed by `build_marg_frame`, and emitted by the Task 4 stub — all three agree. `transfer_build(a, marg, cols, n, s, mode)` calls match `generate.py`. Config names match between `run_faceswap` and the Task 4 test.
```
