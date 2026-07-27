"""Transfer characterization study: measure how heterogeneous contexts are and why.

Analyst-side (reads A and B freely) -- see
docs/superpowers/specs/2026-07-27-transfer-characterization-study-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ssdataagent.config import data_root
from ssdataagent.transfer.decompose import _is_num
from ssdataagent.transfer.pairs import _drop_unnamed
from ssdataagent.transfer import copula_stability as _cop
from ssdataagent.transfer import decompose as _dec
from ssdataagent.transfer.pairs import covariates_outcomes, crosswalk_columns

_EPS = 1e-9


def marginal_distance(a_col: pd.Series, b_col: pd.Series) -> tuple[float, str]:
    """Distance between the marginal of ``a_col`` and ``b_col``. Both numeric ->
    standardized 1-Wasserstein (divide by pooled SD of the non-missing values);
    otherwise -> total variation (0.5 * sum|p-q|) with NaN bucketed as its own category."""
    if _is_num(a_col) and _is_num(b_col):
        av = pd.to_numeric(a_col, errors="coerce").dropna().to_numpy(dtype=float)
        bv = pd.to_numeric(b_col, errors="coerce").dropna().to_numpy(dtype=float)
        if len(av) == 0 or len(bv) == 0:
            return np.nan, "wasserstein"
        sd = float(np.std(np.concatenate([av, bv]))) or 1.0
        return float(wasserstein_distance(av / sd, bv / sd)), "wasserstein"
    pa = a_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    pb = b_col.astype("string").fillna("__nan__").value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    tv = 0.5 * float((pa.reindex(idx, fill_value=0.0)
                      - pb.reindex(idx, fill_value=0.0)).abs().sum())
    return tv, "tv"


def shape_level_split(a: pd.DataFrame, b: pd.DataFrame, response: str, focal: str,
                      *, bins: int = 10) -> dict:
    """Split the A->B conditional-mean gap of a NUMERIC ``response`` over bins of ``focal``
    into a level term (mean gap) and a shape term (rms residual). g(x) = E_B[Y|x] - E_A[Y|x]
    over shared quantile bins of ``focal``; level = mean_x g(x); shape = rms_x(g(x)-level);
    shape_ratio = shape/(|level|+shape+eps). ~0 => pure level shift; ~1 => shape change.
    Bins with no data in either context are skipped."""
    fa = pd.to_numeric(a[focal], errors="coerce")
    fb = pd.to_numeric(b[focal], errors="coerce")
    ya = pd.to_numeric(a[response], errors="coerce")
    yb = pd.to_numeric(b[response], errors="coerce")
    pooled = pd.concat([fa, fb]).dropna()
    empty = {"response": response, "focal": focal, "level": np.nan,
             "shape": np.nan, "shape_ratio": np.nan, "n_bins": 0}
    if len(pooled) == 0:
        return empty
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(pooled, qs))
    ca = np.digitize(fa.to_numpy(dtype=float), edges)
    cb = np.digitize(fb.to_numpy(dtype=float), edges)
    fa_ok, fb_ok = fa.notna().to_numpy(), fb.notna().to_numpy()
    ya_ok, yb_ok = ya.notna().to_numpy(), yb.notna().to_numpy()
    yav, ybv = ya.to_numpy(dtype=float), yb.to_numpy(dtype=float)
    gaps = []
    for k in np.unique(np.concatenate([ca, cb])):
        ma = (ca == k) & fa_ok & ya_ok
        mb = (cb == k) & fb_ok & yb_ok
        if ma.sum() == 0 or mb.sum() == 0:
            continue
        gaps.append(float(ybv[mb].mean() - yav[ma].mean()))
    if not gaps:
        return empty
    g = np.array(gaps, dtype=float)
    level = float(g.mean())
    shape = float(np.sqrt(np.mean((g - level) ** 2)))
    return {"response": response, "focal": focal, "level": level, "shape": shape,
            "shape_ratio": float(shape / (abs(level) + shape + _EPS)), "n_bins": len(g)}


CORE_DEMOGRAPHICS: dict[str, tuple[str, ...]] = {
    "cps": ("age", "gender", "race"),
    "gss": ("age", "gender", "race"),
    "cfps": ("gender", "sib_number"),
}
FOCAL: dict[str, str] = {"cps": "age", "gss": "age", "cfps": "birth_year"}
GROUP_COL: dict[str, str] = {"cps": "race", "gss": "race", "cfps": "minzu"}


@dataclass(frozen=True)
class Context:
    dataset: str
    csv: Path
    label: str
    group_col: str | None = None
    group_val: str | None = None
    negate: bool = False   # True -> keep rows where group_col != group_val (majority/rest)


@dataclass(frozen=True)
class Pair:
    id: str
    family: str            # "time" | "group"
    a: Context
    b: Context
    schema_name: str


def load_context(ctx: Context) -> pd.DataFrame:
    """Read a context's CSV, drop ``Unnamed:`` index columns, and apply the group filter.
    NaN in the grouping column is excluded from BOTH subgroups (== and != both drop it)."""
    df = _drop_unnamed(pd.read_csv(ctx.csv, low_memory=False))
    if ctx.group_col is not None:
        col = df[ctx.group_col].astype("string")
        mask = (col != ctx.group_val) if ctx.negate else (col == ctx.group_val)
        df = df[mask.fillna(False)].reset_index(drop=True)
    return df


def _cps(name: str) -> Path:
    return data_root() / "cps" / name


def _gss(name: str) -> Path:
    return data_root() / "gss" / name


def _cfps() -> Path:
    return data_root() / "cfps" / "cfps_2010_2022.csv"


CONTEXTS: dict[str, Context] = {
    "cps_1970": Context("cps", _cps("cps-asec1970.csv"), "cps 1970"),
    "cps_1980": Context("cps", _cps("cps-asec1980.csv"), "cps 1980"),
    "cps_1990": Context("cps", _cps("cps-asec1990.csv"), "cps 1990"),
    "cps_2000": Context("cps", _cps("cps-asec2000.csv"), "cps 2000"),
    "gss_1994": Context("gss", _gss("gss1994.csv"), "gss 1994"),
    "gss_2018": Context("gss", _gss("gss2018.csv"), "gss 2018"),
    "cps_1980_maj": Context("cps", _cps("cps-asec1980.csv"), "cps1980 non-Black",
                            "race", "Black", negate=True),
    "cps_1980_min": Context("cps", _cps("cps-asec1980.csv"), "cps1980 Black", "race", "Black"),
    "gss_2018_maj": Context("gss", _gss("gss2018.csv"), "gss2018 non-Black",
                            "race", "Black", negate=True),
    "gss_2018_min": Context("gss", _gss("gss2018.csv"), "gss2018 Black", "race", "Black"),
    "cfps_han": Context("cfps", _cfps(), "cfps han", "minzu", "han"),
    "cfps_min": Context("cfps", _cfps(), "cfps minority", "minzu", "minority"),
}

PAIRS: list[Pair] = [
    Pair("cps_1970_1980", "time", CONTEXTS["cps_1970"], CONTEXTS["cps_1980"], "cps"),
    Pair("cps_1980_1990", "time", CONTEXTS["cps_1980"], CONTEXTS["cps_1990"], "cps"),
    Pair("cps_1990_2000", "time", CONTEXTS["cps_1990"], CONTEXTS["cps_2000"], "cps"),
    Pair("cps_1970_2000", "time", CONTEXTS["cps_1970"], CONTEXTS["cps_2000"], "cps"),
    Pair("gss_1994_2018", "time", CONTEXTS["gss_1994"], CONTEXTS["gss_2018"], "gss"),
    Pair("cps_1980_race", "group", CONTEXTS["cps_1980_maj"], CONTEXTS["cps_1980_min"], "cps"),
    Pair("gss_2018_race", "group", CONTEXTS["gss_2018_maj"], CONTEXTS["gss_2018_min"], "gss"),
    Pair("cfps_minzu", "group", CONTEXTS["cfps_han"], CONTEXTS["cfps_min"], "cfps"),
]


def resolve_columns(pair: Pair, a: pd.DataFrame | None = None,
                    b: pd.DataFrame | None = None) -> dict:
    """Load both contexts and resolve the column sets for one pair. For a group pair the
    grouping variable (GROUP_COL[schema]) is dropped from both the Q1 core and Q2 sweep."""
    if a is None:
        a = load_context(pair.a)
    if b is None:
        b = load_context(pair.b)
    cols = crosswalk_columns(pair.schema_name, a, b)
    x, y = covariates_outcomes(pair.schema_name, cols)
    is_group = pair.family == "group"
    gcol = GROUP_COL.get(pair.schema_name)
    core = [c for c in CORE_DEMOGRAPHICS[pair.schema_name]
            if c in cols and not (is_group and c == gcol)]
    x_sweep = [c for c in x if not (is_group and c == gcol)]
    return {"a": a, "b": b, "cols": cols, "x": x, "y": y,
            "core": core, "x_sweep": x_sweep, "focal": FOCAL[pair.schema_name]}


def pair_records(pair: Pair) -> list[dict]:
    """Run Q1-Q4 for one pair and return tidy long-format rows."""
    r = resolve_columns(pair)
    a, b, cols = r["a"], r["b"], r["cols"]
    core, x_sweep, focal = r["core"], r["x_sweep"], r["focal"]
    base = {"pair": pair.id, "family": pair.family, "dataset": pair.schema_name}
    recs: list[dict] = []

    # Q1 -- composition vs mechanism (per outcome)
    for resp in r["y"]:
        d = _dec.kob_decompose(a, b, resp, core)
        recs.append({**base, "question": "Q1", "metric": "composition_share", "key": resp,
                     "value": d["composition_share"], "mechanism_share": d["mechanism_share"],
                     "ess_ratio": d["ess_ratio"], "label": d["label"]})
        if _dec._is_num(a[resp]) and _dec._is_num(b[resp]):
            ob = _dec.oaxaca_blinder(a, b, resp, core)
            recs.append({**base, "question": "Q1", "metric": "composition_share_ob",
                         "key": resp, "value": ob["composition_share_ob"],
                         "endowment": ob["endowment"], "coefficient": ob["coefficient"]})

    # Q2 -- X composition distance (per covariate)
    for xc in x_sweep:
        dist, kind = marginal_distance(a[xc], b[xc])
        recs.append({**base, "question": "Q2", "metric": "marginal_distance", "key": xc,
                     "value": dist, "kind": kind})

    # Q3 -- mechanism / association stability
    stab = _cop.copula_stability(a, b, cols)
    n = len(stab)
    counts = stab["label"].value_counts()
    for lab in ("stable", "shifted", "undefined"):
        recs.append({**base, "question": "Q3", "metric": f"pct_{lab}", "key": "all_pairs",
                     "value": (float(counts.get(lab, 0)) / n) if n else np.nan})
    recs.append({**base, "question": "Q3", "metric": "median_abs_delta", "key": "all_pairs",
                 "value": float(stab["abs_delta"].median(skipna=True)) if n else np.nan})

    # Q4 -- shape vs level (numeric outcomes; focal must be a real column in both frames)
    if focal in a.columns and focal in b.columns:
        for resp in r["y"]:
            if _dec._is_num(a[resp]) and _dec._is_num(b[resp]):
                s = shape_level_split(a, b, resp, focal)
                recs.append({**base, "question": "Q4", "metric": "shape_ratio", "key": resp,
                             "value": s["shape_ratio"], "level": s["level"],
                             "shape": s["shape"], "focal": focal, "n_bins": s["n_bins"]})
    return recs


def run_characterization(pairs: list[Pair] = PAIRS) -> pd.DataFrame:
    """Concatenate pair_records over all pairs into one tidy long-format table."""
    rows: list[dict] = []
    for p in pairs:
        rows.extend(pair_records(p))
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, repo_root: Path) -> tuple[Path, Path]:
    """Write the tidy table to results/ (gitignored) and a committed copy under docs/report."""
    out_dir = repo_root / "results" / "characterization"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "characterization.csv"
    df.to_csv(results_csv, index=False)
    committed = repo_root / "docs" / "report" / "2026-07-27-characterization-data.csv"
    committed.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(committed, index=False)
    return results_csv, committed
