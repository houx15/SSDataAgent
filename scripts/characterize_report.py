#!/usr/bin/env python
"""Render the transfer characterization report: one self-contained HTML with Q1-Q4 figures."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssdataagent.config import REPO_ROOT   # noqa: E402

DATA_CSV = REPO_ROOT / "docs" / "report" / "2026-07-27-characterization-data.csv"
OUT_HTML = REPO_ROOT / "docs" / "report" / "2026-07-27-transfer-characterization.html"
_FAMILY_COLOR = {"time": "#3b6ea5", "group": "#a5533b"}


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig) -> str:
    return f'<img src="data:image/png;base64,{_b64(fig)}" style="max-width:100%;height:auto"/>'


def _fig_q1(df):
    d = df[(df["question"] == "Q1") & (df["metric"] == "composition_share")].dropna(subset=["value"])
    fig, ax = plt.subplots(figsize=(7, 4))
    fams = ["time", "group"]
    data = [d[d["family"] == f]["value"].to_numpy() for f in fams]
    ax.boxplot(data, labels=[f"{f}\n(n={len(v)})" for f, v in zip(fams, data)], showmeans=True)
    for i, f in enumerate(fams, start=1):
        sub = d[d["family"] == f]
        ax.scatter([i] * len(sub), sub["value"], alpha=0.5,
                   color=[_FAMILY_COLOR[f]] * len(sub), zorder=3, s=18)
    ax.set_ylabel("composition_share (per outcome)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Q1 — how much of each Y-gap is composition vs mechanism")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    return fig


def _fig_q2(df):
    d = df[(df["question"] == "Q2") & (df["metric"] == "marginal_distance")].dropna(subset=["value"])
    means = d.groupby(["pair", "family"])["value"].mean().reset_index().sort_values("value")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(means))))
    ax.barh(means["pair"], means["value"],
            color=[_FAMILY_COLOR.get(f, "#777") for f in means["family"]])
    ax.set_xlabel("mean X-composition distance (TV / std-Wasserstein)")
    ax.set_title("Q2 — how different is demographic composition, per pair")
    return fig


def _fig_q3(df):
    d = df[df["question"] == "Q3"]
    piv = d[d["metric"].isin(["pct_stable", "pct_shifted", "pct_undefined"])]
    piv = piv.pivot_table(index="pair", columns="metric", values="value", aggfunc="first").fillna(0)
    for c in ("pct_stable", "pct_shifted", "pct_undefined"):
        if c not in piv:
            piv[c] = 0.0
    piv = piv[["pct_stable", "pct_shifted", "pct_undefined"]]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(piv))))
    left = [0.0] * len(piv)
    for c, color in [("pct_stable", "#4a9"), ("pct_shifted", "#c74"), ("pct_undefined", "#bbb")]:
        ax.barh(piv.index, piv[c], left=left, label=c.replace("pct_", ""), color=color)
        left = [l + v for l, v in zip(left, piv[c])]
    ax.set_xlabel("fraction of variable-pairs")
    ax.set_title("Q3 — mechanism (association) stability, per pair")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def _fig_q4(df):
    d = df[(df["question"] == "Q4") & (df["metric"] == "shape_ratio")].dropna(subset=["value"])
    fig, ax = plt.subplots(figsize=(7, 4))
    fams = ["time", "group"]
    data = [d[d["family"] == f]["value"].to_numpy() for f in fams]
    data = [v if len(v) else [float("nan")] for v in data]
    ax.boxplot(data, labels=fams, showmeans=True)
    ax.set_ylabel("shape_ratio  (0 = pure level shift, 1 = shape change)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Q4 — when mechanism moves, is it level or shape? (numeric Y)")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    return fig


_SECTIONS = [
    ("Q1 — composition vs mechanism", _fig_q1,
     "Per outcome, the share of the A→B gap explained by reweighting A's demographics to B's "
     "(composition); the remainder is mechanism. Above the dashed line = composition-dominated."),
    ("Q2 — X-composition distance", _fig_q2,
     "Mean distance between the two contexts' demographic marginals; larger = the populations "
     "differ more in who they contain."),
    ("Q3 — mechanism stability", _fig_q3,
     "Fraction of variable-pairs whose association is stable vs shifted between contexts "
     "(|Δ| threshold 0.10). More 'stable' = the dependence structure transfers."),
    ("Q4 — shape vs level", _fig_q4,
     "For numeric outcomes, whether the conditional curve moved by a constant offset (level, "
     "cheaply correctable) or changed slope (shape, needs real adaptation)."),
]


def build_report_html(df: pd.DataFrame) -> str:
    blocks = []
    for title, fn, caption in _SECTIONS:
        blocks.append(f"<section><h2>{title}</h2><p class='cap'>{caption}</p>{_img(fn(df))}</section>")
    body = "\n".join(blocks)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transfer characterization — cps / gss / cfps</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto;
          padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
  .cap {{ color: #555; font-size: 0.9rem; }}
  section {{ border-top: 1px solid #eee; padding-top: 0.5rem; }}
  footer {{ color: #888; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #eee; padding-top: 1rem; }}
</style></head><body>
<h1>Transfer characterization: how heterogeneous are contexts, and why?</h1>
<p class="cap">cps / gss / cfps · time and group (ethnicity) families · analyst-side, reads A and B.
Q5 (a learned composition model) is not built: with ~{df['pair'].nunique()} pairs we stay below the
corpus threshold to learn transport, so it remains a corpus-gated follow-on.</p>
{body}
<footer>Generated by scripts/characterize_report.py from
docs/report/2026-07-27-characterization-data.csv. See
docs/superpowers/specs/2026-07-27-transfer-characterization-study-design.md.</footer>
</body></html>"""


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    OUT_HTML.write_text(build_report_html(df), encoding="utf-8")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
