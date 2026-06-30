import React, { useEffect, useState } from "react";
import { api } from "../api";

type NoteForm = {
  hypothesis: string;
  change: string;
  result: string;
  interpretation: string;
  next: string;
  experiments: string;
};

const EMPTY_FORM: NoteForm = {
  hypothesis: "",
  change: "",
  result: "",
  interpretation: "",
  next: "",
  experiments: "",
};

function Field({
  label,
  value,
  onChange,
  multiline,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  multiline?: boolean;
}) {
  return (
    <div style={{ marginBottom: "6px" }}>
      <label style={{ display: "block", fontWeight: "bold", fontSize: "0.85em", marginBottom: "2px" }}>
        {label}
      </label>
      {multiline ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={3}
          style={{ width: "100%", boxSizing: "border-box" }}
        />
      ) : (
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          style={{ width: "100%", boxSizing: "border-box" }}
        />
      )}
    </div>
  );
}

export function Notebook() {
  const [entries, setEntries] = useState<any[]>([]);
  const [form, setForm] = useState<NoteForm>(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = () =>
    api.notebook().then((d) => setEntries(d.entries)).catch(() => {});

  useEffect(() => {
    refresh();
  }, []);

  const set = (k: keyof NoteForm) => (v: string) => setForm((p) => ({ ...p, [k]: v }));

  const submit = async () => {
    setError(null);
    setLoading(true);
    try {
      await api.addNote({
        ...form,
        experiments: form.experiments
          ? form.experiments.split(",").map((s) => s.trim()).filter(Boolean)
          : [],
      });
      setForm(EMPTY_FORM);
      refresh();
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Lab Notebook</h2>
      <div
        style={{
          border: "1px solid #ccc",
          padding: "12px",
          borderRadius: "4px",
          marginBottom: "1rem",
          maxWidth: "600px",
        }}
      >
        <h3 style={{ margin: "0 0 8px" }}>New Entry</h3>
        <Field label="Hypothesis" value={form.hypothesis} onChange={set("hypothesis")} multiline />
        <Field label="Change" value={form.change} onChange={set("change")} multiline />
        <Field label="Result" value={form.result} onChange={set("result")} multiline />
        <Field label="Interpretation" value={form.interpretation} onChange={set("interpretation")} multiline />
        <Field label="Next steps" value={form.next} onChange={set("next")} />
        <Field
          label="Linked experiments (comma-separated)"
          value={form.experiments}
          onChange={set("experiments")}
        />
        {error && <p style={{ color: "red" }}>{error}</p>}
        <button onClick={submit} disabled={loading}>
          {loading ? "Saving…" : "Add Note"}
        </button>
      </div>

      {entries.length === 0 && <p style={{ color: "#888" }}>No entries yet.</p>}
      {entries.map((e, i) => (
        <div
          key={i}
          style={{
            border: "1px solid #ddd",
            margin: "8px 0",
            padding: "12px",
            borderRadius: "4px",
          }}
        >
          {e.timestamp && (
            <div style={{ fontSize: "0.8em", color: "#888", marginBottom: "6px" }}>
              {e.timestamp}
            </div>
          )}
          {e.hypothesis && <p><strong>Hypothesis:</strong> {e.hypothesis}</p>}
          {e.change && <p><strong>Change:</strong> {e.change}</p>}
          {e.result && <p><strong>Result:</strong> {e.result}</p>}
          {e.interpretation && <p><strong>Interpretation:</strong> {e.interpretation}</p>}
          {e.next && <p><strong>Next:</strong> {e.next}</p>}
          {Array.isArray(e.experiments) && e.experiments.length > 0 && (
            <p>
              <strong>Experiments:</strong>{" "}
              {e.experiments.map((ex: string, j: number) => (
                <span key={j} style={{ background: "#e8f4fd", padding: "2px 6px", borderRadius: "3px", marginRight: "4px" }}>
                  {ex}
                </span>
              ))}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
