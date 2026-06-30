import React, { useState } from "react";
import { api } from "../api";

type Format = "html" | "markdown";

export function Reports() {
  const [experiment, setExperiment] = useState("");
  const [condition, setCondition] = useState("");
  const [baseline, setBaseline] = useState("");
  const [format, setFormat] = useState<Format>("html");
  const [result, setResult] = useState<{ format: string; content: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const generate = async () => {
    setError(null);
    setLoading(true);
    try {
      const body: Record<string, string> = { experiment, format };
      if (condition) body.condition = condition;
      if (baseline) body.baseline = baseline;
      const data = await api.report(body);
      setResult(data);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const download = () => {
    if (!result) return;
    const ext = result.format === "html" ? "html" : "md";
    const blob = new Blob([result.content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${experiment}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <h2>Reports</h2>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxWidth: "400px" }}>
        <input
          placeholder="experiment (required)"
          value={experiment}
          onChange={(e) => setExperiment(e.target.value)}
        />
        <input
          placeholder="condition (optional)"
          value={condition}
          onChange={(e) => setCondition(e.target.value)}
        />
        <input
          placeholder="baseline experiment (optional)"
          value={baseline}
          onChange={(e) => setBaseline(e.target.value)}
        />
        <select value={format} onChange={(e) => setFormat(e.target.value as Format)}>
          <option value="html">HTML</option>
          <option value="markdown">Markdown</option>
        </select>
        <button onClick={generate} disabled={loading || !experiment}>
          {loading ? "Generating…" : "Generate Report"}
        </button>
      </div>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {result && (
        <div style={{ marginTop: "1rem" }}>
          <button onClick={download} style={{ marginBottom: "8px" }}>
            Download .{result.format === "html" ? "html" : "md"}
          </button>
          {result.format === "html" ? (
            <iframe
              srcDoc={result.content}
              style={{ width: "100%", height: "600px", border: "1px solid #ccc" }}
              title="report"
            />
          ) : (
            <pre
              style={{
                overflow: "auto",
                maxHeight: "600px",
                background: "#f8f8f8",
                padding: "12px",
                border: "1px solid #ddd",
              }}
            >
              {result.content}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
