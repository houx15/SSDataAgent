import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, StrategyBoardRow, PNAS_BENCHMARKS } from "../api";
import { BenchmarkLegend, DeltaBadge, ScoreCell, fmt } from "../benchmarks";

const TH: React.CSSProperties = { padding: "6px 10px", borderBottom: "2px solid #ccc", textAlign: "left", whiteSpace: "nowrap" };
const TD: React.CSSProperties = { padding: "6px 10px", borderBottom: "1px solid #eee", verticalAlign: "top" };

export function StrategyBoard({ rows }: { rows: StrategyBoardRow[] }) {
  // union of T-types present across all strategies' best runs
  const tkeys = Object.keys(PNAS_BENCHMARKS).filter((k) =>
    rows.some((r) => r.types.some((t) => t.t === PNAS_BENCHMARKS[k].short && t.ours != null))
  );
  const shorts = tkeys.map((k) => PNAS_BENCHMARKS[k].short);
  const cellFor = (r: StrategyBoardRow, short: string) =>
    r.types.find((t) => t.t === short);

  return (
    <table style={{ borderCollapse: "collapse", width: "100%" }}>
      <thead>
        <tr>
          <th style={TH}>Strategy</th>
          <th style={TH}>Best on</th>
          <th style={TH}>Data used</th>
          {shorts.map((s) => (
            <th key={s} style={{ ...TH, textAlign: "right" }}
                title={Object.values(PNAS_BENCHMARKS).find((b) => b.short === s)?.label}>{s}</th>
          ))}
          <th style={{ ...TH, textAlign: "right" }}>Overall</th>
          <th style={{ ...TH, textAlign: "right" }}>PNAS best</th>
          <th style={{ ...TH, textAlign: "right" }}>vs PNAS</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.family}>
            <td style={TD}>
              <Link to={`/strategy/${encodeURIComponent(r.family)}`}><strong>{r.family}</strong></Link>
              <div style={{ fontSize: "0.75em", color: "#888" }}>{r.n_runs} run{r.n_runs === 1 ? "" : "s"}</div>
            </td>
            <td style={TD}>{r.best_dataset}</td>
            <td style={{ ...TD, fontSize: "0.85em", color: "#555", maxWidth: "16rem" }}>{r.data_mode}</td>
            {shorts.map((s) => {
              const c = cellFor(r, s);
              return <td key={s} style={{ ...TD, textAlign: "right" }}>{c ? <ScoreCell c={c} /> : "—"}</td>;
            })}
            <td style={{ ...TD, textAlign: "right", fontWeight: 700 }}>{fmt(r.overall_average)}</td>
            <td style={{ ...TD, textAlign: "right", color: "#888" }}>{fmt(r.paper_overall)}</td>
            <td style={{ ...TD, textAlign: "right" }}><DeltaBadge d={r.delta_overall} big /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function StrategyBoardPage() {
  const [rows, setRows] = useState<StrategyBoardRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.strategies().then((d) => setRows(d.strategies))
      .catch((e: unknown) => setError(String(e))).finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading…</p>;
  if (error) return <p style={{ color: "red" }}>Error: {error}</p>;
  return (
    <div>
      <h2>Strategies vs the PNAS paper</h2>
      <p style={{ color: "#555", marginTop: 0 }}>
        Each strategy's best result across all runs, benchmarked against the SSDataBench
        (PNAS) paper's best-of-15-LLMs. Click a strategy for how it works, what data it
        used, and its full results.
      </p>
      <BenchmarkLegend />
      <StrategyBoard rows={rows} />
    </div>
  );
}
