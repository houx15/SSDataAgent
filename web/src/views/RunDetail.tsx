import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";

interface RunArtifacts {
  [key: string]: string;
}

interface RunEval {
  by_type?: Record<string, number>;
  overall_average?: number;
  overdetermination?: Record<string, unknown>;
}

interface RunEntry {
  condition: string;
  dataset: string;
  run_id?: string;
  eval?: RunEval;
  meta?: Record<string, unknown>;
  artifacts: RunArtifacts;
}

interface ExperimentInfo {
  name: string;
  status?: string;
  model?: string;
}

export interface RunDetail {
  experiment: ExperimentInfo;
  runs: RunEntry[];
}

export function RunDetailView({ detail }: { detail: RunDetail }) {
  return (
    <div>
      <h2>Experiment: <span>{detail.experiment.name}</span></h2>
      {detail.experiment.status && (
        <p>
          Status: {detail.experiment.status}
          {detail.experiment.model ? ` · Model: ${detail.experiment.model}` : ""}
        </p>
      )}
      {Array.isArray(detail.runs) && detail.runs.map((run, i) => (
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
          {run.artifacts && Object.keys(run.artifacts).length > 0 && (
            <div style={{ marginTop: "8px" }}>
              <strong>Artifacts:</strong>
              <ul style={{ margin: "4px 0", paddingLeft: "20px" }}>
                {Object.entries(run.artifacts).map(([key, path]) => (
                  <li key={key}>
                    {key}: <code>{path}</code>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function RunDetail() {
  const { name } = useParams<{ name: string }>();
  const [detail, setDetail] = useState<RunDetail | null>(null);
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

  return <RunDetailView detail={detail} />;
}
