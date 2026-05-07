"""Generate a pruned SSDataBench evaluation config for the columns we actually have.

Reads each type*.yaml under ssdatabench/evaluation/config/<dataset>/ and
writes a pruned copy under <dataset>_subset/ keeping only the variables
present in the dataset CSV that the agent sees.

The CSV path comes from config/datasets.yaml via the loader, so this stays
in sync with whatever paper-faithful or legacy file is wired up.

Usage:
    python scripts/build_eval_subset.py gss
    python scripts/build_eval_subset.py cps
    python scripts/build_eval_subset.py acs
    python scripts/build_eval_subset.py gss_legacy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ssdataagent.config import ssdatabench_root  # noqa: E402
from ssdataagent.data.loader import load_real_data  # noqa: E402
from ssdataagent.data.schema import load_schema  # noqa: E402


def _prune_variables(spec: dict, available: set[str]) -> dict:
    out = dict(spec)
    # Type1, Type2, Type5 use a "variables:" block.
    if "variables" in out and isinstance(out["variables"], dict):
        out["variables"] = {k: v for k, v in out["variables"].items() if k in available}
    # Type3 uses "response:" (variables being modeled) and "predictors:".
    # Also has a parallel "model_type:" list that must match len(response).
    response_before = list((out.get("response") or {}).keys()) if isinstance(out.get("response"), dict) else None
    for key in ("response", "predictors", "covariates"):
        if isinstance(out.get(key), dict):
            out[key] = {k: v for k, v in out[key].items() if k in available}
        elif isinstance(out.get(key), list):
            out[key] = [k for k in out[key] if (isinstance(k, str) and k in available)
                        or (isinstance(k, dict) and k.get("name", k.get("var")) in available)]
    if response_before is not None and isinstance(out.get("model_type"), list):
        # Truncate or pad model_type to match the new response length.
        kept_indices = [i for i, k in enumerate(response_before) if k in (out.get("response") or {})]
        out["model_type"] = [out["model_type"][i] for i in kept_indices if i < len(out["model_type"])]
    return out


def _filter_pairs(items: list, available: set[str]) -> list:
    """Keep entries whose dependent variables are all present (used by type3/type5)."""
    out = []
    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        # Common keys used to declare dependent variables in their configs:
        keys_to_check = ("var", "x", "y", "vars", "variables")
        ok = True
        for key in keys_to_check:
            v = entry.get(key)
            if isinstance(v, str) and v not in available:
                ok = False
                break
            if isinstance(v, list) and not all(x in available for x in v):
                ok = False
                break
        if ok:
            out.append(entry)
    return out


def _prune_pair_blocks(spec: dict, available: set[str]) -> dict:
    out = dict(spec)
    for key in ("pairs", "edges", "groups", "joint_specs"):
        if isinstance(out.get(key), list):
            out[key] = _filter_pairs(out[key], available)
    return out


def build_subset(dataset: str) -> Path:
    schema = load_schema(dataset)
    subdir = schema.ssdatabench_sim_subdir
    df = load_real_data(dataset)
    # Variables fully NaN can't be evaluated (bootstrap samples zero-length series).
    available = {c for c in df.columns if df[c].notna().any()}
    dropped_empty = sorted(set(df.columns) - available)
    if dropped_empty:
        print(f"[build_eval_subset] dropping all-NaN columns: {dropped_empty}")
    bench = ssdatabench_root()
    src = bench / "evaluation" / "config" / subdir
    dst = bench / "evaluation" / "config" / f"{subdir}_subset"
    dst.mkdir(parents=True, exist_ok=True)

    for src_file in sorted(src.glob("type*.yaml")):
        spec = yaml.safe_load(src_file.read_text()) or {}
        spec = _prune_variables(spec, available)
        spec = _prune_pair_blocks(spec, available)
        (dst / src_file.name).write_text(yaml.safe_dump(spec, sort_keys=False))

    master_path = src / "evaluation_master.yaml"
    if master_path.exists():
        master = yaml.safe_load(master_path.read_text()) or {}
        master["type_config_dir"] = f"./evaluation/config/{subdir}_subset/"
        master.pop("real_csv", None)
        master.pop("sim_csv", None)
        master.pop("out_dir", None)
        (dst / "evaluation_master.yaml").write_text(yaml.safe_dump(master, sort_keys=False))

    print(f"[build_eval_subset] wrote pruned configs to {dst}")
    print(f"  variables kept: {sorted(available - {'profile_id'})}")
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    args = p.parse_args()
    build_subset(args.dataset)


if __name__ == "__main__":
    main()
