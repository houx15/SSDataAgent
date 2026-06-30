import React, { useState } from "react";
import Plot from "react-plotly.js";
import { api } from "../api";

type Selector = { experiment: string; condition: string; dataset: string };

type Cell = {
  selector: Selector;
  by_type: Record<string, number | null>;
  overall_average: number | null;
  overdetermination_gap: number | null;
};

export function CompareHeatmap({
  types,
  matrix,
  cells,
}: {
  types: string[];
  matrix: (number | null)[][];
  cells: Cell[];
}) {
  const yLabels = cells.map(
    (c) => `${c.selector.experiment}/${c.selector.condition}/${c.selector.dataset}`
  );
  return (
    <div>
      <Plot
        data={[
          {
            type: "heatmap",
            z: matrix,
            x: types,
            y: yLabels,
            colorscale: "Viridis",
            zmin: 0,
            zmax: 1,
          },
        ]}
        layout={{
          title: "Score Heatmap",
          height: Math.max(300, 60 + cells.length * 40),
          margin: { l: 200, r: 40, t: 50, b: 80 },
        }}
        style={{ width: "100%" }}
        config={{ responsive: true }}
      />
    </div>
  );
}

export function Compare() {
  const [selectors, setSelectors] = useState<Selector[]>([
    { experiment: "", condition: "", dataset: "" },
  ]);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setError(null);
    setLoading(true);
    try {
      const active = selectors.filter((s) => s.experiment.trim());
      const data = await api.compare(active);
      setResult(data);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const updateSel = (i: number, field: keyof Selector, value: string) =>
    setSelectors((prev) =>
      prev.map((s, j) => (j === i ? { ...s, [field]: value } : s))
    );

  const addRow = () =>
    setSelectors((p) => [...p, { experiment: "", condition: "", dataset: "" }]);

  const removeRow = (i: number) =>
    setSelectors((p) => p.filter((_, j) => j !== i));

  return (
    <div>
      <h2>Compare</h2>
      <p style={{ color: "#666", fontSize: "0.9em" }}>
        Add one or more (experiment, condition, dataset) selectors and click Compare.
      </p>
      {selectors.map((s, i) => (
        <div key={i} style={{ display: "flex", gap: "4px", marginBottom: "4px" }}>
          <input
            placeholder="experiment"
            value={s.experiment}
            onChange={(e) => updateSel(i, "experiment", e.target.value)}
          />
          <input
            placeholder="condition"
            value={s.condition}
            onChange={(e) => updateSel(i, "condition", e.target.value)}
          />
          <input
            placeholder="dataset"
            value={s.dataset}
            onChange={(e) => updateSel(i, "dataset", e.target.value)}
          />
          <button onClick={() => removeRow(i)} style={{ color: "red" }}>✕</button>
        </div>
      ))}
      <div style={{ display: "flex", gap: "8px", marginBottom: "1rem" }}>
        <button onClick={addRow}>+ Add selector</button>
        <button onClick={run} disabled={loading}>
          {loading ? "Loading…" : "Compare"}
        </button>
      </div>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {result && (
        <>
          <CompareHeatmap types={result.types} matrix={result.matrix} cells={result.cells} />
          <table style={{ borderCollapse: "collapse", width: "100%", marginTop: "1rem" }}>
            <thead>
              <tr>
                <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Experiment</th>
                <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Condition</th>
                <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Dataset</th>
                {result.types.map((t: string) => (
                  <th key={t} style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>{t}</th>
                ))}
                <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>Overall</th>
                <th style={{ padding: "4px 8px", borderBottom: "2px solid #ccc" }}>OD Gap</th>
              </tr>
            </thead>
            <tbody>
              {result.cells.map((cell: Cell, i: number) => (
                <tr key={i}>
                  <td style={{ padding: "4px 8px" }}>{cell.selector.experiment}</td>
                  <td style={{ padding: "4px 8px" }}>{cell.selector.condition}</td>
                  <td style={{ padding: "4px 8px" }}>{cell.selector.dataset}</td>
                  {result.types.map((t: string) => (
                    <td key={t} style={{ padding: "4px 8px" }}>
                      {cell.by_type[t] != null ? (cell.by_type[t] as number).toFixed(3) : "—"}
                    </td>
                  ))}
                  <td style={{ padding: "4px 8px" }}>
                    {cell.overall_average != null ? cell.overall_average.toFixed(3) : "—"}
                  </td>
                  <td style={{ padding: "4px 8px" }}>
                    {cell.overdetermination_gap != null
                      ? cell.overdetermination_gap.toFixed(3)
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
