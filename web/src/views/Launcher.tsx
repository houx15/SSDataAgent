import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

const STATUS_COLOR: Record<string, string> = {
  running: "green",
  failed: "red",
  done: "blue",
  queued: "orange",
};

export function Launcher() {
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [log, setLog] = useState<string>("");
  const [name, setName] = useState("");
  const [forkFrom, setForkFrom] = useState("");
  const [newName, setNewName] = useState("");
  const [overrides, setOverrides] = useState("{}");
  const [isFork, setIsFork] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = () =>
    api.runs().then((d) => setExperiments(d.experiments)).catch(() => {});

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (!selected) return;
    const poll = () => api.log(selected).then((d) => setLog(d.log)).catch(() => {});
    poll();
    intervalRef.current = setInterval(poll, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [selected]);

  const launch = async () => {
    setErr(null);
    try {
      if (isFork) {
        let parsed: any = {};
        try { parsed = JSON.parse(overrides); } catch { parsed = {}; }
        await api.launch({ fork_from: forkFrom, new_name: newName, overrides: parsed });
      } else {
        await api.launch({ name });
      }
      refresh();
    } catch (e: unknown) {
      setErr(String(e));
    }
  };

  const cancel = async () => {
    if (!selected) return;
    setErr(null);
    try {
      await api.cancel(selected);
      refresh();
    } catch (e: unknown) {
      setErr(String(e));
    }
  };

  return (
    <div>
      <h2>Runs</h2>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {experiments.map((e, i) => (
          <li
            key={i}
            style={{ padding: "4px 0", cursor: "pointer", background: selected === e.name ? "#f0f0f0" : undefined }}
            onClick={() => setSelected(e.name)}
          >
            <span style={{ color: STATUS_COLOR[e.status] ?? "gray", marginRight: "8px" }}>
              [{e.status}]
            </span>
            <Link to={`/runs/${e.name}`} onClick={(ev) => ev.stopPropagation()}>
              {e.name}
            </Link>
            {e.model && <span style={{ color: "#888", fontSize: "0.85em", marginLeft: "8px" }}>{e.model}</span>}
          </li>
        ))}
      </ul>

      <h3>Launch</h3>
      <label style={{ marginBottom: "8px", display: "block" }}>
        <input
          type="checkbox"
          checked={isFork}
          onChange={(e) => setIsFork(e.target.checked)}
          style={{ marginRight: "6px" }}
        />
        Fork from existing experiment
      </label>
      {isFork ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", maxWidth: "400px" }}>
          <input
            placeholder="fork_from (experiment name)"
            value={forkFrom}
            onChange={(e) => setForkFrom(e.target.value)}
          />
          <input
            placeholder="new_name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <textarea
            placeholder='overrides JSON (e.g. {"model": "gpt-4o"})'
            value={overrides}
            onChange={(e) => setOverrides(e.target.value)}
            rows={4}
          />
        </div>
      ) : (
        <input
          placeholder="experiment name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ minWidth: "300px" }}
        />
      )}
      <div style={{ marginTop: "8px", display: "flex", gap: "8px" }}>
        <button onClick={launch}>Launch</button>
        {selected && <button onClick={cancel}>Cancel "{selected}"</button>}
        <button onClick={refresh}>Refresh</button>
      </div>
      {err && <p style={{ color: "red" }}>{err}</p>}

      {selected && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Log: {selected}</h3>
          <pre
            style={{
              maxHeight: "300px",
              overflow: "auto",
              background: "#1e1e1e",
              color: "#d4d4d4",
              padding: "8px",
              fontSize: "0.8em",
            }}
          >
            {log || "(no log yet)"}
          </pre>
        </div>
      )}
    </div>
  );
}
