"""Scoring helpers shared by the transfer scripts (LLM-free).

Extracted from scripts/transfer_map.py so scripts/transfer_b3.py can reuse them
without a script importing a script.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def restrict_config_dir(subdir: str, cols: set[str], types, dest: Path) -> Path:
    """Write type configs restricted to the transferable (crosswalk) ``cols`` under
    ``dest/subdir/``, and return ``dest`` (a ``config_dir`` for nb.score).

    A transfer sim can only carry variables the SOURCE context has, so the target's stock
    config — which may test variables the source lacks (gss2018's depress/mental_health) —
    would KeyError. Restricting ``variables``/``predictors``/``response`` to ``cols`` scores
    exactly the transferable variables. For a pair whose crosswalk already covers the config
    (cps 1970->1980), this is a no-op and reproduces the stock score.
    """
    import yaml
    import nodonor_bracket as nb
    src = nb.CONFIG_DIR / subdir
    (dest / subdir).mkdir(parents=True, exist_ok=True)
    for t in types:
        p = src / f"type{t}.yaml"
        if not p.exists():
            continue
        cfg = yaml.safe_load(p.read_text())
        # T3 carries a model_type list aligned positionally to `response`; when we drop a
        # response we must drop its model_type entry too, or the runner raises
        # "N model types but M responses".
        resp = cfg.get("response")
        mt = cfg.get("model_type")
        if isinstance(resp, dict) and isinstance(mt, list) and len(mt) == len(resp):
            keep_mask = [k in cols for k in resp]
            cfg["model_type"] = [m for m, keep in zip(mt, keep_mask) if keep]
        for key in ("variables", "predictors", "response"):
            if isinstance(cfg.get(key), dict):
                cfg[key] = {k: v for k, v in cfg[key].items() if k in cols}
        # sort_keys=False: preserve `response` dict order so T3's positional `model_type`
        # list stays paired with the right response (see nodonor_bracket._cfg_with_B).
        (dest / subdir / f"type{t}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dest


def mean_scores(df: pd.DataFrame) -> dict:
    """Average the numeric per-type / overall score columns across seeds.

    nb.score() emits per-type rates as ``T1``..``T5`` and ``overall``, but also stores
    per-type FAILURES as string columns ``T{t}_error`` (e.g. a type-eval that KeyErrors on
    a crosswalk-dropped column). Those also start with 'T', so a naive ``startswith('T')``
    would call ``.mean()`` on a string column and crash. Select only ``overall`` and
    ``T<digit>`` columns explicitly.
    """
    keep = [c for c in df.columns
            if (c == "overall" or (c.startswith("T") and c[1:].isdigit()))
            and df[c].notna().any()]
    return {c: float(df[c].mean()) for c in keep}
