import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";

export function RunDetail() {
  const { name } = useParams<{ name: string }>();
  const [detail, setDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!name) return;
    api
      .runDetail(name)
      .then(setDetail)
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [name]);

  if (loading) return <p>Loading…</p>;
  if (error) return <p style={{ color: "red" }}>Error: {error}</p>;
  if (!detail) return <p>Not found.</p>;

  return (
    <div>
      <h2>Experiment: {detail.experiment}</h2>
      {Array.isArray(detail.runs) && detail.runs.map((run: any, i: number) => (
        <div
          key={i}
          style={{ border: "1px solid #ccc", margin: "8px 0", padding: "12px", borderRadius: "4px" }}
        >
          <h3 style={{ margin: "0 0 8px" }}>
            {run.condition} / {run.dataset}
            {run.run_id ? <span style={{ fontSize: "0.8em", color: "#666" }}> #{run.run_id}</span> : null}
          </h3>
          {run.eval && (
            <details>
              <summary>Evaluation scores</summary>
              <pre style={{ overflow: "auto", maxHeight: "200px" }}>
                {JSON.stringify(run.eval, null, 2)}
              </pre>
            </details>
          )}
          {run.meta && (
            <details>
              <summary>Meta</summary>
              <pre style={{ overflow: "auto", maxHeight: "200px" }}>
                {JSON.stringify(run.meta, null, 2)}
              </pre>
            </details>
          )}
          {run.run_dir && (
            <div style={{ marginTop: "8px" }}>
              <strong>Artifacts: </strong>
              <code>{run.run_dir}</code>
              {Array.isArray(run.artifacts) && run.artifacts.map((a: string, j: number) => (
                <span key={j} style={{ marginLeft: "8px" }}>
                  <a href={`file://${run.run_dir}/${a}`} target="_blank" rel="noreferrer">{a}</a>
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
