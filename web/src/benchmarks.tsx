import React from "react";
import { PNAS_BENCHMARKS, TypeCmp } from "./api";

export const fmt = (v: number | null | undefined): string =>
  v == null ? "—" : Number.isNaN(v) ? "NaN" : v.toFixed(3);

const GOOD = "#1b7a3d";
const BAD = "#b23b3b";

export function deltaColor(d: number | null | undefined): string | undefined {
  if (d == null || Number.isNaN(d)) return undefined;
  return d >= 0 ? GOOD : BAD;
}

/** "+0.220" / "−0.056" with color, or "—". */
export function DeltaBadge({ d, big }: { d: number | null | undefined; big?: boolean }) {
  if (d == null || Number.isNaN(d)) return <span>—</span>;
  const s = `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(3)}`;
  return (
    <span style={{ color: deltaColor(d), fontWeight: big ? 700 : 500 }}>
      {s} {d >= 0 ? "▲" : "▼"}
    </span>
  );
}

/** A T-score cell: our value colored by whether it beats the paper, with the
 *  paper value + delta in the hover title. */
export function ScoreCell({ c }: { c: TypeCmp }) {
  const title =
    c.paper == null
      ? "no paper baseline"
      : `ours ${fmt(c.ours)} vs PNAS ${fmt(c.paper)} (Δ ${
          c.delta == null ? "—" : (c.delta >= 0 ? "+" : "−") + Math.abs(c.delta).toFixed(3)
        })`;
  return (
    <span title={title} style={{ color: deltaColor(c.delta) }}>
      {fmt(c.ours)}
    </span>
  );
}

/** Legend explaining T1–T5 as the five PNAS benchmarks. */
export function BenchmarkLegend() {
  return (
    <p style={{ fontSize: "0.85em", color: "#555", margin: "0.4em 0 1em" }}>
      <strong>PNAS benchmarks:</strong>{" "}
      {Object.values(PNAS_BENCHMARKS).map((b, i) => (
        <span key={b.short}>
          {i > 0 ? " · " : ""}
          <strong>{b.short}</strong> {b.label}
        </span>
      ))}
      . Scores are pass-rates (higher = closer to real data);{" "}
      <span style={{ color: "#1b7a3d" }}>green</span> beats the PNAS paper best,{" "}
      <span style={{ color: "#b23b3b" }}>red</span> is worse.
    </p>
  );
}
