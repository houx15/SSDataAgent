import React, { useEffect, useState } from "react";
import { api, LeaderboardRow } from "../api";

export function LeaderboardTable({ rows }: { rows: LeaderboardRow[] }) {
  const allTypes = Array.from(new Set(rows.flatMap((r) => Object.keys(r.by_type))));
  return (
    <table style={{ borderCollapse: "collapse", width: "100%" }}>
      <thead>
        <tr>
          <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Champion</th>
          <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Experiment</th>
          <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Condition</th>
          <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Dataset</th>
          {allTypes.map((t) => (
            <th key={t} style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>{t}</th>
          ))}
          <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Overall Avg</th>
          <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>OD Gap</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr
            key={i}
            style={{ background: row.is_champion ? "#fffde7" : undefined }}
          >
            <td style={{ padding: "4px 8px", textAlign: "center" }}>
              {row.is_champion ? "★" : ""}
            </td>
            <td style={{ padding: "4px 8px" }}>{row.experiment}</td>
            <td style={{ padding: "4px 8px" }}>{row.condition}</td>
            <td style={{ padding: "4px 8px" }}>{row.dataset}</td>
            {allTypes.map((t) => (
              <td key={t} style={{ padding: "4px 8px" }}>
                {row.by_type[t] != null ? (row.by_type[t] as number).toFixed(3) : "—"}
              </td>
            ))}
            <td style={{ padding: "4px 8px" }}>
              {row.overall_average != null ? row.overall_average.toFixed(3) : "—"}
            </td>
            <td style={{ padding: "4px 8px" }}>
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
      <LeaderboardTable rows={rows} />
    </div>
  );
}
