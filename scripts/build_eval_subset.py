"""Generate a pruned SSDataBench evaluation config for the columns we actually have.

The shipped GSS-2018 evaluation expects ~28 variables, but our cleaned
real_data/gss_clean.csv only contains 11. This script reads each type*.yaml
under ssdatabench/evaluation/config/<dataset>/ and writes a pruned copy under
<dataset>_subset/ that keeps only the variables present in our CSV.

Usage:
    python scripts/build_eval_subset.py gss      # GSS 2018
    python scripts/build_eval_subset.py cps      # CPS 1980
    python scripts/build_eval_subset.py acs      # ACS 1980
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]
SSDATABENCH = REPO / "ssdatabench"

DATASET_TO_SUBDIR_AND_CSV = {
    "gss": ("gss_2018", "gss_clean.csv"),
    "cps": ("cps_1980", "cps_clean.csv"),
    "acs": ("acs_1980", "acs_clean.csv"),
}


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
    subdir, csv_name = DATASET_TO_SUBDIR_AND_CSV[dataset]
    df = pd.read_csv(REPO / "real_data" / csv_name)
    # Variables fully NaN can't be evaluated (bootstrap samples zero-length series).
    available = {c for c in df.columns if df[c].notna().any()}
    dropped_empty = sorted(set(df.columns) - available)
    if dropped_empty:
        print(f"[build_eval_subset] dropping all-NaN columns: {dropped_empty}")
    src = SSDATABENCH / "evaluation" / "config" / subdir
    dst = SSDATABENCH / "evaluation" / "config" / f"{subdir}_subset"
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
    p.add_argument("dataset", choices=list(DATASET_TO_SUBDIR_AND_CSV))
    args = p.parse_args()
    build_subset(args.dataset)


if __name__ == "__main__":
    main()
