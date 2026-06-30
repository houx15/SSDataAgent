import { Link, Route, Routes } from "react-router-dom";
import { Leaderboard } from "./views/Leaderboard";
import { RunDetail } from "./views/RunDetail";
import { Launcher } from "./views/Launcher";
import { Compare } from "./views/Compare";
import { Reports } from "./views/Reports";
import { Notebook } from "./views/Notebook";

export function App() {
  return (
    <div style={{ fontFamily: "system-ui", maxWidth: "72rem", margin: "1rem auto" }}>
      <nav style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
        <Link to="/">Leaderboard</Link><Link to="/runs">Runs</Link>
        <Link to="/compare">Compare</Link><Link to="/reports">Reports</Link>
        <Link to="/notebook">Notebook</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Leaderboard />} />
        <Route path="/runs" element={<Launcher />} />
        <Route path="/runs/:name" element={<RunDetail />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/notebook" element={<Notebook />} />
      </Routes>
    </div>
  );
}
