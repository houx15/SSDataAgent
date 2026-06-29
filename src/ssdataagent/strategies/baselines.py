from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ssdataagent.data.schema import DatasetSchema, load_schema
from ssdataagent.strategies.base import InfoGate, StrategyResult


def classify_columns(schema: DatasetSchema, columns) -> tuple[list[str], list[str]]:
    """Split columns into (numerical, categorical). Numerical iff in
    schema.numeric_ranges; everything else is treated as categorical/ordinal."""
    numerical, categorical = [], []
    for c in columns:
        (numerical if c in schema.numeric_ranges else categorical).append(c)
    return numerical, categorical


def encode_numeric(df, columns, schema, *, stats=None):
    """Float feature matrix: numerical z-scored, categorical one-hot over
    schema.allowed_values. `stats` (means/sds fitted on train) is reused for
    eval rows so the scaling origin matches. Returns (X, stats)."""
    num, cat = classify_columns(schema, columns)
    if stats is None:
        means = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in num}
        sds = {c: (float(pd.to_numeric(df[c], errors="coerce").std(ddof=0)) or 1.0) for c in num}
        stats = {"means": means, "sds": sds}
    blocks = []
    for c in num:
        col = pd.to_numeric(df[c], errors="coerce").fillna(stats["means"][c]).to_numpy()
        blocks.append(((col - stats["means"][c]) / stats["sds"][c]).reshape(-1, 1))
    for c in cat:
        cats = schema.allowed_values.get(c) or sorted(df[c].dropna().unique().tolist())
        idx = {v: i for i, v in enumerate(cats)}
        oh = np.zeros((len(df), len(cats)), dtype=float)
        for r, v in enumerate(df[c].tolist()):
            if v in idx:
                oh[r, idx[v]] = 1.0
        blocks.append(oh)
    X = np.hstack(blocks) if blocks else np.zeros((len(df), 0))
    return X, stats


def ordinal_encode(df, columns, schema) -> np.ndarray:
    """Integer-code columns for tree models: numerical kept as float;
    categorical mapped to its index in allowed_values (unknown -> -1)."""
    cols = []
    for c in columns:
        if c in schema.numeric_ranges:
            cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1))
        else:
            cats = schema.allowed_values.get(c) or sorted(df[c].dropna().unique().tolist())
            idx = {v: i for i, v in enumerate(cats)}
            cols.append(np.array([idx.get(v, -1) for v in df[c].tolist()], dtype=float).reshape(-1, 1))
    return np.hstack(cols) if cols else np.zeros((len(df), 0))


def clip_decode(df, schema) -> pd.DataFrame:
    """Clip numerical columns to schema.numeric_ranges; leave categoricals."""
    out = df.copy()
    for c in out.columns:
        if c in schema.numeric_ranges:
            lo, hi = schema.numeric_ranges[c]
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lo, hi)
    return out


def background_frame(background, schema) -> pd.DataFrame:
    """Background columns (+ profile_id), target columns dropped (no leakage)."""
    cols = [c for c in background.columns
            if c in schema.background_variables or c == "profile_id"]
    out = background[cols].reset_index(drop=True).copy()
    if "profile_id" not in out.columns:
        out.insert(0, "profile_id", range(len(out)))
    return out


def hotdeck_generate(train, background, schema, *, k=10, seed=42) -> pd.DataFrame:
    """Generate targets for background rows via k-NN hot-deck imputation.

    For each background row, find its k nearest neighbors in the training set
    (by background variables), randomly pick one, and donate its targets.
    Returns a DataFrame with background variables + all targets, clipped to range.
    """
    bg_vars = list(schema.background_variables)
    targets = list(schema.target_variables)
    Xtr, stats = encode_numeric(train, bg_vars, schema)
    Xev, _ = encode_numeric(background, bg_vars, schema, stats=stats)
    k_eff = max(1, min(k, len(train)))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(Xtr)
    _, idx = nn.kneighbors(Xev)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k_eff, size=len(Xev))
    chosen = idx[np.arange(len(idx)), pick]
    donor = train.iloc[chosen][targets].reset_index(drop=True)
    out = background_frame(background, schema)
    for c in targets:
        out[c] = donor[c].to_numpy()
    return clip_decode(out, schema)


class HotDeckStrategy:
    """Hot-deck / k-NN baseline: donor imputation by nearest background neighbors."""

    name = "hotdeck"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        """Generate synthetic data using hot-deck imputation.

        Raises ValueError if no microdata is available (fit_microdata() is None).
        """
        train = gate.fit_microdata()
        if train is None:
            raise ValueError("hotdeck requires microdata; this condition exposes none")
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        generated = hotdeck_generate(train, bg, schema, k=10, seed=42)
        Path(run_dir, "fit_summary.json").write_text(
            json.dumps({"backend": "hotdeck", "k": 10, "n_train_fit": len(train)}, indent=2)
        )
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "hotdeck", "k": 10,
                         "n_train_fit": len(train), "n_individuals": len(bg)},
        )


def cart_generate(train, background, schema, *, min_samples_leaf=5, seed=42) -> pd.DataFrame:
    bg_vars = list(schema.background_variables)
    targets = list(schema.target_variables)
    rng = np.random.default_rng(seed)
    out = background_frame(background, schema)
    feat_cols = list(bg_vars)
    train_feat = train[bg_vars].copy().reset_index(drop=True)
    gen_feat = background[bg_vars].copy().reset_index(drop=True)
    for t in targets:
        Xtr = ordinal_encode(train_feat, feat_cols, schema)
        Xgen = ordinal_encode(gen_feat, feat_cols, schema)
        is_num = t in schema.numeric_ranges
        Tree = DecisionTreeRegressor if is_num else DecisionTreeClassifier
        y = pd.to_numeric(train[t], errors="coerce").to_numpy() if is_num \
            else train[t].astype(str).to_numpy()
        model = Tree(min_samples_leaf=min_samples_leaf, random_state=seed).fit(Xtr, y)
        leaf_tr = model.apply(Xtr)
        leaf_gen = model.apply(Xgen)
        raw = train[t].to_numpy()
        by_leaf: dict[int, list] = {}
        for lid, v in zip(leaf_tr, raw):
            by_leaf.setdefault(int(lid), []).append(v)
        drawn = []
        for lid in leaf_gen:
            pool = by_leaf.get(int(lid)) or list(raw)
            drawn.append(pool[int(rng.integers(0, len(pool)))])
        out[t] = drawn
        feat_cols = feat_cols + [t]
        train_feat[t] = train[t].to_numpy()
        gen_feat[t] = drawn
    return clip_decode(out, schema)


class CartStrategy:
    name = "cart"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        train = gate.fit_microdata()
        if train is None:
            raise ValueError("cart requires microdata; this condition exposes none")
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        generated = cart_generate(train, bg, schema, min_samples_leaf=5, seed=42)
        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "cart", "min_samples_leaf": 5,
             "target_order": list(schema.target_variables), "n_train_fit": len(train)},
            indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "cart", "min_samples_leaf": 5,
                         "n_train_fit": len(train), "n_individuals": len(bg)},
        )


_EPS = 1e-6


def _build_cuts(train, cols, schema) -> dict:
    """Per-column inversion data. numeric -> sorted train values;
    categorical -> (categories, cumulative upper edges)."""
    cuts: dict[str, dict] = {}
    n = len(train)
    for c in cols:
        if c in schema.numeric_ranges:
            vals = pd.to_numeric(train[c], errors="coerce").dropna().to_numpy()
            cuts[c] = {"kind": "num", "sorted": np.sort(vals)}
        else:
            cats = schema.allowed_values.get(c) or sorted(train[c].dropna().unique().tolist())
            counts = train[c].value_counts()
            probs = np.array([max(counts.get(v, 0), 0) for v in cats], dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(cats), 1.0 / len(cats))
            cuts[c] = {"kind": "cat", "cats": list(cats), "cum": np.cumsum(probs)}
    return cuts


def _latent_value(col_cut, value) -> float:
    if col_cut["kind"] == "num":
        s = col_cut["sorted"]
        if len(s) == 0 or pd.isna(value):
            return 0.0
        pos = int(np.searchsorted(s, float(value), side="right"))
        u = min(max((pos - 0.5) / len(s), _EPS), 1 - _EPS)
        return float(norm.ppf(u))
    cats, cum = col_cut["cats"], col_cut["cum"]
    if value not in cats:
        return 0.0
    i = cats.index(value)
    lo = cum[i - 1] if i > 0 else 0.0
    u = min(max((lo + cum[i]) / 2.0, _EPS), 1 - _EPS)
    return float(norm.ppf(u))


def _latent_matrix(df, cols, schema, cuts) -> np.ndarray:
    out = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        out[:, j] = [_latent_value(cuts[c], v) for v in df[c].tolist()]
    return out


def _invert(z_array, col, schema, col_cut) -> list:
    u = np.clip(norm.cdf(z_array), _EPS, 1 - _EPS)
    if col_cut["kind"] == "num":
        s = col_cut["sorted"]
        return list(np.quantile(s, u)) if len(s) else [0.0] * len(u)
    cats, cum = col_cut["cats"], col_cut["cum"]
    idx = np.searchsorted(cum, u, side="left")
    idx = np.clip(idx, 0, len(cats) - 1)
    return [cats[i] for i in idx]


def _make_pd(M, reg) -> np.ndarray:
    M = (M + M.T) / 2.0
    M = M + reg * np.eye(M.shape[0])
    w, V = np.linalg.eigh(M)
    w = np.clip(w, reg, None)
    return (V * w) @ V.T


def copula_generate(train, background, schema, *, regularization=1e-6, seed=42) -> pd.DataFrame:
    bg_vars = list(schema.background_variables)
    targets = list(schema.target_variables)
    cols = bg_vars + targets
    cuts = _build_cuts(train, cols, schema)
    Z = _latent_matrix(train, cols, schema, cuts)
    Sigma = _make_pd(np.corrcoef(Z, rowvar=False), regularization)
    bi = list(range(len(bg_vars)))
    ti = list(range(len(bg_vars), len(cols)))
    Sbb = Sigma[np.ix_(bi, bi)]
    Stt = Sigma[np.ix_(ti, ti)]
    Stb = Sigma[np.ix_(ti, bi)]
    Sbb_inv = np.linalg.pinv(Sbb)
    cond_cov = _make_pd(Stt - Stb @ Sbb_inv @ Stb.T, regularization)
    L = np.linalg.cholesky(cond_cov)
    Zb = _latent_matrix(background, bg_vars, schema, cuts)
    mu = (Stb @ Sbb_inv @ Zb.T).T
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((len(background), len(ti))) @ L.T
    Zt = mu + eps
    out = background_frame(background, schema)
    for j, t in enumerate(targets):
        out[t] = _invert(Zt[:, j], t, schema, cuts[t])
    return clip_decode(out, schema)


class CopulaStrategy:
    name = "copula"

    def generate(self, gate: InfoGate, run_dir: Path, cfg) -> StrategyResult:
        train = gate.fit_microdata()
        if train is None:
            raise ValueError("copula requires microdata; this condition exposes none")
        schema = load_schema(gate.dataset_name)
        bg = gate.background()
        generated = copula_generate(train, bg, schema, seed=42)
        Path(run_dir, "fit_summary.json").write_text(json.dumps(
            {"backend": "copula", "regularization": 1e-6, "n_train_fit": len(train)}, indent=2))
        return StrategyResult(
            generated=generated,
            meta_extras={"backend": "copula", "n_train_fit": len(train),
                         "n_individuals": len(bg)},
        )
