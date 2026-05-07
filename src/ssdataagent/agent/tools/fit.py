"""Fit-family tools — build the in-progress generative chain incrementally.

Design:
- `Chain` is a per-run ordered list of `Step` objects. Each Step knows how
  to draw values for its column given previously-drawn columns.
- `set_generation_order` fixes the order. Subsequent fits must respect it
  (a conditional's `given` columns must appear before the col in the order).
- Tools never raise; they return `{"error": ...}` on bad input.
- Fitted models live in memory on the Step. We don't persist the fitted
  model to disk for the spike — only the metadata (`col`, `family`,
  `given`, ...) goes into `chain.json`. Re-running requires a fresh fit.

Supported families (intentionally small for the spike — easy to extend later):
- Marginals: `empirical`, `kde`, `normal`, `categorical_empirical`
- Conditionals: `empirical_lookup`, `linear_regression`, `logistic_regression`
- `copy_real` (sentinel: at sample time, draw iid from the empirical column)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression, LogisticRegression

from ssdataagent.agent.tools.state import RuntimeState, data_withheld_error


SUPPORTED_MARGINAL_FAMILIES = ("empirical", "categorical_empirical", "kde", "normal")
SUPPORTED_CONDITIONAL_FAMILIES = ("empirical_lookup", "linear_regression", "logistic_regression")


# ===================== Step types =====================


@dataclass
class Step:
    """Base class. Subclasses override sample()."""
    col: str
    kind: str  # "marginal" | "conditional" | "copy_real"
    family: str

    def sample(self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def to_meta(self) -> dict:
        """JSON-serializable summary for chain.json — fitted model excluded."""
        return {"col": self.col, "kind": self.kind, "family": self.family}


@dataclass
class MarginalStep(Step):
    kind: str = field(default="marginal", init=False)
    fit_values: np.ndarray | None = None     # for empirical / categorical_empirical
    kde: Any = None                          # scipy gaussian_kde
    normal_params: tuple[float, float] | None = None  # (mu, sigma)

    def sample(self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame) -> np.ndarray:
        if self.family in ("empirical", "categorical_empirical"):
            assert self.fit_values is not None
            idx = rng.integers(0, len(self.fit_values), size=n_rows)
            return self.fit_values[idx]
        if self.family == "kde":
            assert self.kde is not None
            return self.kde.resample(n_rows, seed=rng).ravel()
        if self.family == "normal":
            assert self.normal_params is not None
            mu, sigma = self.normal_params
            return rng.normal(mu, sigma, size=n_rows)
        raise RuntimeError(f"unsupported marginal family {self.family!r}")


@dataclass
class ConditionalStep(Step):
    kind: str = field(default="conditional", init=False)
    given: list[str] = field(default_factory=list)
    allow_missing: bool = False

    # State for empirical_lookup
    lookup_table: dict | None = None         # tuple(given_vals) → np.ndarray of col values
    fallback_values: np.ndarray | None = None  # global empirical for missing-key fallback

    # State for linear_regression / logistic_regression
    sklearn_model: Any = None
    sklearn_feature_cols: list[str] | None = None
    sklearn_categorical_features: list[str] | None = None  # subset of feature_cols that need encoding
    sklearn_target_classes: np.ndarray | None = None       # for logistic_regression

    # NA model — present only when allow_missing=True
    na_model: Any = None                     # sklearn classifier predicting P(NA | given)

    def sample(self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame) -> np.ndarray:
        # Step 1: maybe sample NA mask first.
        na_mask = np.zeros(n_rows, dtype=bool)
        if self.allow_missing and self.na_model is not None:
            X = _encode_features(partial[self.given], self.sklearn_feature_cols, self.sklearn_categorical_features)
            p_na = self.na_model.predict_proba(X)[:, 1]
            na_mask = rng.random(n_rows) < p_na

        # Step 2: sample values for non-NA rows.
        if self.family == "empirical_lookup":
            values = self._sample_empirical_lookup(rng, n_rows, partial)
        elif self.family == "linear_regression":
            values = self._sample_linear_regression(rng, n_rows, partial)
        elif self.family == "logistic_regression":
            values = self._sample_logistic_regression(rng, n_rows, partial)
        else:
            raise RuntimeError(f"unsupported conditional family {self.family!r}")

        # Step 3: punch in NA where the NA mask said so.
        if self.allow_missing and na_mask.any():
            values = values.astype(object) if values.dtype.kind != "O" else values
            for i in np.where(na_mask)[0]:
                values[i] = None
        return values

    def _sample_empirical_lookup(
        self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame
    ) -> np.ndarray:
        out = np.empty(n_rows, dtype=object)
        keys = partial[self.given].apply(lambda r: tuple(r), axis=1).values
        for i, k in enumerate(keys):
            pool = self.lookup_table.get(k) if self.lookup_table else None
            if pool is None or len(pool) == 0:
                pool = self.fallback_values
            out[i] = pool[rng.integers(0, len(pool))]
        return out

    def _sample_linear_regression(
        self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame
    ) -> np.ndarray:
        X = _encode_features(partial[self.given], self.sklearn_feature_cols, self.sklearn_categorical_features)
        mu = self.sklearn_model.predict(X)
        # Add Gaussian noise sized by the residual std observed at fit time.
        sigma = getattr(self.sklearn_model, "_residual_std_", 0.0)
        return mu + rng.normal(0.0, sigma, size=n_rows)

    def _sample_logistic_regression(
        self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame
    ) -> np.ndarray:
        X = _encode_features(partial[self.given], self.sklearn_feature_cols, self.sklearn_categorical_features)
        probs = self.sklearn_model.predict_proba(X)
        out = np.empty(n_rows, dtype=object)
        for i in range(n_rows):
            out[i] = rng.choice(self.sklearn_target_classes, p=probs[i])
        return out


@dataclass
class CopyRealStep(Step):
    kind: str = field(default="copy_real", init=False)
    family: str = field(default="copy_real", init=False)
    fit_values: np.ndarray | None = None

    def sample(self, rng: np.random.Generator, n_rows: int, partial: pd.DataFrame) -> np.ndarray:
        assert self.fit_values is not None
        idx = rng.integers(0, len(self.fit_values), size=n_rows)
        return self.fit_values[idx]


# ===================== Encoding helper =====================


def _encode_features(
    df_in: pd.DataFrame,
    feature_cols: list[str] | None,
    categorical_features: list[str] | None,
) -> np.ndarray:
    """One-hot encode any categorical features; pass numerics through as float.
    Same treatment as fit time, controlled by the column lists captured then."""
    if feature_cols is None:
        return df_in.to_numpy(dtype=float)
    parts = []
    cats = set(categorical_features or [])
    for c in feature_cols:
        if c in cats:
            parts.append(pd.get_dummies(df_in[c], prefix=c, dummy_na=True).to_numpy(dtype=float))
        else:
            v = pd.to_numeric(df_in[c], errors="coerce").fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
            parts.append(v)
    return np.hstack(parts) if parts else np.zeros((len(df_in), 0))


def _build_feature_matrix(df_fit: pd.DataFrame, given: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X, feature_cols, categorical_feature_names). The dummy
    columns generated here are remembered so sample-time encoding produces
    the same columns even if some categories are unseen at sample time."""
    feature_cols: list[str] = []
    cat_features: list[str] = []
    parts = []
    for c in given:
        s = df_fit[c]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float).reshape(-1, 1))
            feature_cols.append(c)
        else:
            dummies = pd.get_dummies(s, prefix=c, dummy_na=True)
            parts.append(dummies.to_numpy(dtype=float))
            feature_cols.append(c)
            cat_features.append(c)
    X = np.hstack(parts) if parts else np.zeros((len(df_fit), 0))
    return X, feature_cols, cat_features


# ===================== Chain =====================


@dataclass
class Chain:
    """Owns the generation order + the registered Steps."""
    generation_order: list[str] = field(default_factory=list)
    steps: dict[str, Step] = field(default_factory=dict)  # col → Step

    def has(self, col: str) -> bool:
        return col in self.steps

    def position(self, col: str) -> int | None:
        return self.generation_order.index(col) if col in self.generation_order else None

    def add(self, step: Step) -> None:
        self.steps[step.col] = step

    def remove(self, col: str) -> bool:
        return self.steps.pop(col, None) is not None

    def sample(self, rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
        if not self.generation_order:
            raise RuntimeError("Chain.sample called before set_generation_order")
        partial = pd.DataFrame(index=range(n_rows))
        for col in self.generation_order:
            step = self.steps.get(col)
            if step is None:
                # Should not happen — orchestrator's auto-commit fills missing
                # cols with empirical marginals before reaching sample().
                raise RuntimeError(f"no Step registered for column {col!r}")
            partial[col] = step.sample(rng, n_rows, partial)
        return partial

    def to_meta(self) -> dict:
        return {
            "generation_order": list(self.generation_order),
            "steps": [self.steps[c].to_meta() for c in self.generation_order if c in self.steps],
        }


# ===================== Tool implementations =====================


def _refusal(state: RuntimeState) -> dict | None:
    if not state.has_data:
        return data_withheld_error()
    return None


def _column_missing(state: RuntimeState, col: str) -> dict | None:
    if col not in state.train_fit.columns:
        return {
            "error": "unknown_column",
            "details": f"column {col!r} not in train; available: {list(state.train_fit.columns)}",
        }
    return None


def set_generation_order(state: RuntimeState, cols: list[str]) -> dict:
    """Declare the order columns will be sampled in. Required before any
    fit_conditional. Calling again replaces the order."""
    if (refusal := _refusal(state)) is not None:
        return refusal
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        return {"error": "bad_arguments", "details": "cols must be a list of column-name strings"}
    unknown = [c for c in cols if c not in state.train_fit.columns]
    if unknown:
        return {"error": "unknown_column", "details": f"unknown columns: {unknown}"}
    if len(set(cols)) != len(cols):
        return {"error": "duplicate_columns", "details": "cols contains duplicates"}
    state.chain.generation_order = list(cols)
    return {"set": True, "order": list(cols)}


def fit_marginal(state: RuntimeState, col: str, family: str = "empirical") -> dict:
    """Fit a marginal distribution for `col` and register it in the chain."""
    if (refusal := _refusal(state)) is not None:
        return refusal
    if (err := _column_missing(state, col)) is not None:
        return err
    if family not in SUPPORTED_MARGINAL_FAMILIES:
        return {
            "error": "unknown_family",
            "details": f"family must be one of {list(SUPPORTED_MARGINAL_FAMILIES)}; got {family!r}",
        }
    s = state.train_fit[col]
    nn = s.dropna()
    if len(nn) == 0:
        return {"error": "all_missing", "details": f"column {col!r} is entirely NaN in train_fit"}

    step = MarginalStep(col=col, family=family)
    if family in ("empirical", "categorical_empirical"):
        step.fit_values = nn.to_numpy()
    elif family == "kde":
        if not pd.api.types.is_numeric_dtype(s):
            return {"error": "non_numeric_for_kde", "details": f"kde requires numeric column; {col!r} is {s.dtype}"}
        if nn.nunique() < 2:
            return {"error": "kde_needs_variance", "details": f"column {col!r} has fewer than 2 unique non-NaN values"}
        step.kde = gaussian_kde(nn.to_numpy(dtype=float))
    elif family == "normal":
        if not pd.api.types.is_numeric_dtype(s):
            return {"error": "non_numeric_for_normal", "details": f"normal requires numeric column; {col!r} is {s.dtype}"}
        step.normal_params = (float(nn.mean()), float(nn.std(ddof=0) or 1e-9))

    state.chain.add(step)
    if col not in state.chain.generation_order:
        # Auto-extend the order so the agent can fit marginals before
        # calling set_generation_order. They can re-order later.
        state.chain.generation_order.append(col)
    return {
        "col": col,
        "family": family,
        "n_used": int(len(nn)),
        "n_missing": int(s.isna().sum()),
        "registered": True,
    }


def fit_conditional(
    state: RuntimeState,
    col: str,
    given: list[str],
    family: str = "empirical_lookup",
    allow_missing: bool = False,
) -> dict:
    """Fit a conditional distribution P(col | given) and register it."""
    if (refusal := _refusal(state)) is not None:
        return refusal
    if (err := _column_missing(state, col)) is not None:
        return err
    if not isinstance(given, list) or not given:
        return {"error": "bad_arguments", "details": "given must be a non-empty list of column names"}
    for g in given:
        if g not in state.train_fit.columns:
            return {"error": "unknown_column", "details": f"given column {g!r} not in train"}
    if family not in SUPPORTED_CONDITIONAL_FAMILIES:
        return {
            "error": "unknown_family",
            "details": f"family must be one of {list(SUPPORTED_CONDITIONAL_FAMILIES)}; got {family!r}",
        }

    order = state.chain.generation_order
    if col not in order:
        return {
            "error": "col_not_in_order",
            "details": f"call set_generation_order with {col!r} included before fit_conditional",
        }
    col_pos = order.index(col)
    misordered = [g for g in given if g not in order or order.index(g) >= col_pos]
    if misordered:
        return {
            "error": "given_after_col",
            "details": f"every column in given must appear before {col!r} in generation_order; offenders: {misordered}",
        }

    df_fit = state.train_fit
    s_col = df_fit[col]
    is_numeric = pd.api.types.is_numeric_dtype(s_col)

    # Family / dtype compatibility checks.
    if family == "linear_regression" and not is_numeric:
        return {"error": "family_dtype_mismatch", "details": "linear_regression requires numeric col"}
    if family == "logistic_regression" and is_numeric:
        return {"error": "family_dtype_mismatch", "details": "logistic_regression requires categorical col"}

    step = ConditionalStep(col=col, family=family, given=list(given), allow_missing=bool(allow_missing))

    # Build the rows where col is observed (used by all families).
    obs_mask = s_col.notna()
    n_obs = int(obs_mask.sum())
    if n_obs < 5:
        return {"error": "too_few_obs", "details": f"need ≥5 rows where {col!r} is non-NaN; have {n_obs}"}

    if family == "empirical_lookup":
        lut: dict[tuple, np.ndarray] = {}
        keys = df_fit.loc[obs_mask, given].apply(tuple, axis=1)
        values = s_col[obs_mask].to_numpy()
        for k, v in zip(keys, values):
            lut.setdefault(k, []).append(v)
        step.lookup_table = {k: np.asarray(vs) for k, vs in lut.items()}
        step.fallback_values = values
        score = None  # no in-sample R²/accuracy for lookups; the agent verifies via score_*
    else:
        X_fit, feature_cols, cat_features = _build_feature_matrix(df_fit.loc[obs_mask], given)
        y_fit = s_col[obs_mask].to_numpy()
        step.sklearn_feature_cols = feature_cols
        step.sklearn_categorical_features = cat_features

        if family == "linear_regression":
            model = LinearRegression()
            model.fit(X_fit, y_fit)
            preds = model.predict(X_fit)
            residuals = y_fit - preds
            model._residual_std_ = float(np.std(residuals)) if len(residuals) > 1 else 0.0
            ss_tot = float(np.var(y_fit) * len(y_fit)) or 1e-12
            score = 1.0 - float(np.sum(residuals ** 2)) / ss_tot
            step.sklearn_model = model
        elif family == "logistic_regression":
            classes = np.unique(y_fit.astype(object))
            if len(classes) < 2:
                return {"error": "logistic_needs_2_classes", "details": f"col {col!r} has only one class in train"}
            model = LogisticRegression(max_iter=1000)
            model.fit(X_fit, y_fit)
            score = float(model.score(X_fit, y_fit))
            step.sklearn_model = model
            step.sklearn_target_classes = classes
        else:
            return {"error": "internal", "details": f"unhandled family {family!r}"}

    # NA model — only when allow_missing.
    if allow_missing:
        # Predict P(col is NA | given). Only useful when there's any NA at all.
        n_na = int((~obs_mask).sum())
        if n_na == 0:
            step.na_model = None  # nothing to predict
        else:
            X_na, feat_na, cat_na = _build_feature_matrix(df_fit, given)
            y_na = (~obs_mask).astype(int).to_numpy()
            try:
                na_clf = LogisticRegression(max_iter=1000)
                na_clf.fit(X_na, y_na)
                step.na_model = na_clf
                # When the NA-model is trained, sample-time encoding for it
                # uses the same feature_cols/categoricals as the value model.
                # (Rare: na features differ from value features; we keep them
                # identical for simplicity.)
                step.sklearn_feature_cols = feat_na
                step.sklearn_categorical_features = cat_na
            except ValueError:
                # Edge case: only one class of NA observed. Skip NA modeling.
                step.na_model = None

    state.chain.add(step)
    return {
        "col": col,
        "given": list(given),
        "family": family,
        "allow_missing": bool(allow_missing),
        "n_obs": n_obs,
        "score": float(score) if score is not None else None,
        "registered": True,
    }


def fit_copy_real(state: RuntimeState, col: str) -> dict:
    """Register a column to be drawn iid from its empirical distribution.
    Useful for unseen-variable runs where we can't model from the data."""
    if (refusal := _refusal(state)) is not None:
        return refusal
    if (err := _column_missing(state, col)) is not None:
        return err
    s = state.train_fit[col]
    nn = s.dropna()
    if len(nn) == 0:
        return {"error": "all_missing", "details": f"column {col!r} is entirely NaN in train_fit"}
    step = CopyRealStep(col=col, fit_values=nn.to_numpy())
    state.chain.add(step)
    if col not in state.chain.generation_order:
        state.chain.generation_order.append(col)
    return {"col": col, "kind": "copy_real", "n_used": int(len(nn)), "registered": True}


def replace_step(state: RuntimeState, col: str) -> dict:
    """Drop the existing fit for `col` so the agent can refit it differently."""
    removed = state.chain.remove(col)
    return {"removed": col, "was_present": bool(removed)}
