#!/usr/bin/env python
"""Automated data-vs-label audit: flag where a variable's DATA contradicts its LABEL.

This is the data-understanding layer, separated from the general strategy. It reads ONLY
the benchmark's documentation (typeN.yaml descriptions) and the DISJOINT POOL -- never the
test reference -- so anything it emits is train/test-clean and usable in the no-donor
regime (same discipline as the strategy).

Motivation: cps `child_number` is documented "Number of Child ever born" but the data is
IPUMS household-resident children. A human caught it on 2026-07-20; the LLM did not, and
generated lifetime fertility, inverting the strongest T3 relationship. The point of this
script is to catch that class of trap systematically instead of by hand.

Checks (each test-blind):
  cumulative-monotonicity  A label implying a CUMULATIVE lifetime quantity ("ever born",
                           "number of", "total ... ever") is monotone non-decreasing in
                           age BY DEFINITION. If mean(value | age) falls with age in the
                           pool, the label is definitionally impossible -> it is a
                           stock/resident measure. This is the fertility trap.
  numeric-sentinel         A field the config calls numeric that carries a recurring
                           non-numeric token ('No Child') -> a sentinel to model as
                           missingness/zero, not a number.
  linear-identity          Two numeric columns in an exact linear relation (age +
                           birth_year = 1980) -> DERIVE one, never draw both independently
                           (drawing both manufactures impossible people).
  log-scale                A positive, heavily right-skewed numeric (income) -> the T3
                           regression log-transforms it; flag so the strategy matches.

Usage:
    .venv/bin/python scripts/data_audit.py cps
    .venv/bin/python scripts/data_audit.py all
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nodonor_bracket import CONFIG_DIR, _drop_unnamed, carve_pool  # noqa: E402
from ssdataagent.data.schema import load_schema  # noqa: E402

# Words in a description that assert a value only ever ACCUMULATES over a life -- so it
# cannot fall with age. Deliberately conservative: better to under-flag than cry wolf.
_CUMULATIVE_HINTS = ("ever born", "ever had", "ever been", "number of child",
                     "children ever", "total number", "lifetime", "ever married",
                     "times married", "number of times")


def _descriptions(ds: str) -> dict[str, str]:
    """Variable descriptions from the benchmark's own typeN configs (the documentation a
    published system is handed). Merged across types; later types do not override."""
    import yaml
    sub = load_schema(ds).ssdatabench_sim_subdir
    out: dict[str, str] = {}
    for t in (1, 2, 3, 4, 5):
        cfg_path = CONFIG_DIR / sub / f"type{t}.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        for block in ("variables", "predictors", "response"):
            for name, spec in (cfg.get(block) or {}).items():
                if isinstance(spec, dict) and spec.get("description") and name not in out:
                    out[name] = str(spec["description"])
    return out


def _is_numeric(s: pd.Series) -> bool:
    s = s.dropna()
    return bool(len(s)) and pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def _age_profile(pool: pd.DataFrame, col: str, age: pd.Series) -> pd.Series | None:
    """mean(numeric value | age band). None if the pair is too thin to judge."""
    y = pd.to_numeric(pool[col], errors="coerce")
    band = pd.cut(age, [0, 15, 25, 35, 45, 60, 99])
    prof = y.groupby(band, observed=True).mean().dropna()
    return prof if len(prof) >= 4 else None


def _find_age(pool: pd.DataFrame) -> pd.Series | None:
    for c in ("age", "AGE", "Age"):
        if c in pool.columns and _is_numeric(pool[c]):
            return pd.to_numeric(pool[c], errors="coerce")
    return None


def audit(ds: str) -> list[dict]:
    ref = _drop_unnamed(pd.read_csv(load_schema(ds).real_data_path, low_memory=False))
    pool, guarantee = carve_pool(ds)
    descs = _descriptions(ds)
    schema = load_schema(ds)
    modeled = [c for c in ref.columns
               if (c in schema.domains or c in schema.background_variables)
               and c in pool.columns and c != "profile_id"]
    age = _find_age(pool)
    findings: list[dict] = []

    def flag(col, check, msg):
        findings.append({"col": col, "check": check, "msg": msg})

    # cumulative-monotonicity + numeric-sentinel + log-scale (per column)
    for c in modeled:
        desc = descs.get(c, "")
        low = desc.lower()
        s = pool[c]
        # numeric-sentinel: mostly-numeric field with a recurring string token
        parsed = pd.to_numeric(s, errors="coerce")
        nonnum = s.dropna()[parsed[s.notna()].isna()]
        if 0 < len(nonnum) and len(nonnum) < len(s.dropna()):
            frac = len(nonnum) / len(s.dropna())
            if frac > 0.02:
                top = nonnum.astype(str).value_counts().head(2).index.tolist()
                flag(c, "numeric-sentinel",
                     f"{frac:.0%} non-numeric in a numeric field; sentinel(s) {top} -- "
                     f"model as missing/zero, not a number")
        # cumulative-monotonicity
        if age is not None and any(h in low for h in _CUMULATIVE_HINTS) and _is_numeric(s):
            prof = _age_profile(pool, c, age)
            if prof is not None:
                peak_i = int(np.argmax(prof.values))
                drop = prof.values[peak_i] - prof.values[-1]
                if peak_i < len(prof) - 1 and drop > 0.15 * max(prof.values[peak_i], 1e-9):
                    flag(c, "cumulative-monotonicity",
                         f"label '{desc}' implies a lifetime cumulative count (monotone "
                         f"in age by definition), but pool mean PEAKS at band "
                         f"{prof.index[peak_i]} ({prof.values[peak_i]:.2f}) and FALLS to "
                         f"{prof.values[-1]:.2f} by the oldest -- so it is a stock/resident "
                         f"measure, NOT lifetime. Correct the definition before generating.")
        # log-scale
        if _is_numeric(s):
            v = parsed.dropna()
            if len(v) > 50 and (v >= 0).all() and v.max() > 0:
                sk = float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-9))
                if sk > 3 and v.max() / (v.median() + 1e-9) > 10:
                    flag(c, "log-scale",
                         f"positive, right-skewed (skew {sk:.1f}); T3 log-transforms it -- "
                         f"match with log1p in the strategy")

    # linear-identity (numeric column pairs in an exact linear relation)
    num = [c for c in modeled if _is_numeric(pool[c])]
    for i, a in enumerate(num):
        va = pd.to_numeric(pool[a], errors="coerce")
        for b in num[i + 1:]:
            vb = pd.to_numeric(pool[b], errors="coerce")
            m = va.notna() & vb.notna()
            if m.sum() < 50:
                continue
            s = (va[m] + vb[m])
            if s.nunique() == 1:
                flag(f"{a}+{b}", "linear-identity",
                     f"{a} + {b} = {s.iloc[0]:.0f} for every row -- DERIVE one from the "
                     f"other; drawing both independently manufactures impossible people")
            elif np.isclose(np.corrcoef(va[m], vb[m])[0, 1], -1.0, atol=1e-6) or \
                    np.isclose(np.corrcoef(va[m], vb[m])[0, 1], 1.0, atol=1e-6):
                flag(f"{a}~{b}", "linear-identity",
                     f"{a} and {b} are perfectly collinear -- treat one as derived")

    print(f"\n{'='*78}\n{ds}: pool={len(pool)} [{guarantee}], {len(modeled)} modeled vars, "
          f"{len(descs)} documented\n{'='*78}")
    if not findings:
        print("  no contradictions flagged")
    else:
        for f in findings:
            print(f"  [FLAG:{f['check']}] {f['col']}\n      {f['msg']}")
    return findings


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "cps"
    datasets = ["cps", "cfps", "gss", "acs", "addhealth"] if which == "all" else [which]
    total = 0
    for ds in datasets:
        try:
            total += len(audit(ds))
        except (Exception, SystemExit) as e:  # carve_pool raises SystemExit w/o a pool
            print(f"\n{ds}: SKIPPED ({type(e).__name__}: {e})")
    print(f"\n{total} finding(s) across {len(datasets)} dataset(s).")


if __name__ == "__main__":
    main()
