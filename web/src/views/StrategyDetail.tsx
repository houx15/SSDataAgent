import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, StrategyDetail as SD, PNAS_BENCHMARKS } from "../api";
import { BenchmarkLegend, DeltaBadge, ScoreCell, fmt } from "../benchmarks";

const TH: React.CSSProperties = { padding: "6px 10px", borderBottom: "2px solid #ccc", textAlign: "left", whiteSpace: "nowrap" };
const TD: React.CSSProperties = { padding: "6px 10px", borderBottom: "1px solid #eee" };

export function StrategyDetailView({ detail }: { detail: SD }) {
  const shorts = Object.values(PNAS_BENCHMARKS)
    .map((b) => b.short)
    .filter((s) => detail.rows.some((r) => r.types.some((t) => t.t === s && t.ours != null)));
  const cellFor = (r: SD["rows"][number], short: string) => r.types.find((t) => t.t === short);

  return (
    <div>
      <p style={{ marginBottom: "0.5rem" }}><Link to="/">← All strategies</Link></p>
      <h2 style={{ marginBottom: "0.25rem" }}>{detail.family}</h2>

      <section style={{ margin: "0.5rem 0 1rem" }}>
        <h3 style={{ marginBottom: "0.25rem" }}>How it works</h3>
        <p style={{ marginTop: 0, maxWidth: "52rem", lineHeight: 1.5 }}>{detail.blurb}</p>
      </section>

      <section style={{ display: "flex", gap: "2rem", flexWrap: "wrap", margin: "0 0 1rem" }}>
        <div><strong>Model(s):</strong> {detail.models.length ? detail.models.join(", ") : "—"}</div>
        <div><strong>Datasets:</strong> {detail.datasets.join(", ")}</div>
        <div><strong>Runs:</strong> {detail.n_runs}</div>
      </section>

      <section style={{ margin: "0 0 1rem" }}>
        <strong>Ways the data was used:</strong>
        <ul style={{ marginTop: "0.25rem" }}>
          {detail.data_modes.map((m) => <li key={m}>{m}</li>)}
        </ul>
      </section>

      <h3 style={{ marginBottom: "0.25rem" }}>Benchmark results vs PNAS</h3>
      <BenchmarkLegend />
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr>
            <th style={TH}>Dataset</th>
            <th style={TH}>Data used</th>
            <th style={TH}>Model</th>
            <th style={TH}>Timepoint</th>
            {shorts.map((s) => <th key={s} style={{ ...TH, textAlign: "right" }}>{s}</th>)}
            <th style={{ ...TH, textAlign: "right" }}>Overall</th>
            <th style={{ ...TH, textAlign: "right" }}>PNAS</th>
            <th style={{ ...TH, textAlign: "right" }}>vs PNAS</th>
          </tr>
        </thead>
        <tbody>
          {detail.rows.map((r, i) => (
            <tr key={i}>
              <td style={TD}>{r.dataset}</td>
              <td style={{ ...TD, fontSize: "0.85em", color: "#555" }}>{r.data_mode}</td>
              <td style={{ ...TD, fontSize: "0.85em" }}>{r.model ?? "—"}</td>
              <td style={TD}><code>{r.run_id ?? "—"}</code></td>
              {shorts.map((s) => {
                const c = cellFor(r, s);
                return <td key={s} style={{ ...TD, textAlign: "right" }}>{c ? <ScoreCell c={c} /> : "—"}</td>;
              })}
              <td style={{ ...TD, textAlign: "right", fontWeight: 600 }}>{fmt(r.overall_average)}</td>
              <td style={{ ...TD, textAlign: "right", color: "#888" }}>{fmt(r.paper_overall)}</td>
              <td style={{ ...TD, textAlign: "right" }}><DeltaBadge d={r.delta_overall} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function StrategyDetailPage() {
  const { family = "" } = useParams();
  const [detail, setDetail] = useState<SD | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null); setError(null);
    api.strategyDetail(family).then(setDetail).catch((e: unknown) => setError(String(e)));
  }, [family]);

  if (error) return <p style={{ color: "red" }}>Error: {error}</p>;
  if (!detail) return <p>Loading…</p>;
  return <StrategyDetailView detail={detail} />;
}
