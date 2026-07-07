import React, { useEffect, useState } from "react";
import { api, LeaderboardRow, PNAS_BENCHMARKS } from "../api";

const TH: React.CSSProperties = { padding: "4px 8px", borderBottom: "2px solid #ccc", textAlign: "left" };
const TD: React.CSSProperties = { padding: "4px 8px" };

function typeHeader(t: string): { text: string; title: string } {
  const b = PNAS_BENCHMARKS[t];
  return b ? { text: b.short, title: `${b.short} — ${b.label} (SSDataBench/PNAS benchmark)` }
           : { text: t, title: t };
}

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
      . Scores are pass-rates (higher = closer to real data). ★ = champion for its
      (condition × dataset) cell. Paper comparison is in the <em>Reports</em> view.
    </p>
  );
}

export function LeaderboardTable({ rows }: { rows: LeaderboardRow[] }) {
  const allTypes = Array.from(new Set(rows.flatMap((r) => Object.keys(r.by_type))));
  return (
    <table style={{ borderCollapse: "collapse", width: "100%" }}>
      <thead>
        <tr>
          <th style={TH}>Champion</th>
          <th style={TH}>Experiment</th>
          <th style={TH}>Strategy</th>
          <th style={TH}>Dataset</th>
          <th style={TH}>Model</th>
          <th style={TH}>Timepoint</th>
          {allTypes.map((t) => {
            const h = typeHeader(t);
            return <th key={t} style={TH} title={h.title}>{h.text}</th>;
          })}
          <th style={TH}>Overall Avg</th>
          <th style={TH}>OD Gap</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr
            key={i}
            style={{ background: row.is_champion ? "#fffde7" : undefined }}
          >
            <td style={{ ...TD, textAlign: "center" }}>
              {row.is_champion ? "★" : ""}
            </td>
            <td style={TD}>{row.experiment}</td>
            <td style={TD}>{row.condition}</td>
            <td style={TD}>{row.dataset}</td>
            <td style={TD}>{row.model ?? "—"}</td>
            <td style={TD}><code>{row.run_id ?? "—"}</code></td>
            {allTypes.map((t) => (
              <td key={t} style={TD}>
                {row.by_type[t] != null ? (row.by_type[t] as number).toFixed(3) : "—"}
              </td>
            ))}
            <td style={TD}>
              {row.overall_average != null ? row.overall_average.toFixed(3) : "—"}
            </td>
            <td style={TD}>
              {row.overdetermination_gap != null ? row.overdetermination_gap.toFixed(3) : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Leaderboard() {
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .leaderboard()
      .then((data) => setRows(data.rows))
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading…</p>;
  if (error) return <p style={{ color: "red" }}>Error: {error}</p>;
  return (
    <div>
      <h2>Leaderboard</h2>
      <BenchmarkLegend />
      <LeaderboardTable rows={rows} />
    </div>
  );
}
