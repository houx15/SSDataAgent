export type LeaderboardRow = {
  experiment: string; condition: string; dataset: string;
  run_id: string | null; model: string | null;
  by_type: Record<string, number | null>;
  overall_average: number | null; overdetermination_gap: number | null;
  is_pilot: boolean; is_champion: boolean;
};

// The five SSDataBench (PNAS) benchmarks, keyed by eval.json's by_type keys.
export const PNAS_BENCHMARKS: Record<string, { short: string; label: string }> = {
  type1: { short: "T1", label: "Marginal distributions" },
  type2: { short: "T2", label: "Pairwise associations" },
  type3: { short: "T3", label: "Regression (conditional)" },
  type4: { short: "T4", label: "Life-course event order" },
  type5: { short: "T5", label: "Event-order × covariate" },
};

export type TypeCmp = { t: string; ours: number | null; paper: number | null; delta: number | null };

export type StrategyBoardRow = {
  family: string; blurb: string;
  best_dataset: string; data_mode: string; condition: string;
  experiment: string; run_id: string | null; model: string | null;
  overall_average: number; paper_overall: number | null; delta_overall: number | null;
  types: TypeCmp[]; n_runs: number;
};

export type StrategyDetailRow = {
  dataset: string; condition: string; data_mode: string;
  model: string | null; run_id: string | null; experiment: string;
  overall_average: number; paper_overall: number | null; delta_overall: number | null;
  types: TypeCmp[];
};

export type StrategyDetail = {
  family: string; blurb: string; models: string[]; datasets: string[];
  data_modes: string[]; n_runs: number; rows: StrategyDetailRow[];
};

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export const api = {
  strategies: () => getJSON<{ strategies: StrategyBoardRow[] }>("/api/strategies"),
  strategyDetail: (family: string) => getJSON<StrategyDetail>(`/api/strategies/${encodeURIComponent(family)}`),
  leaderboard: () => getJSON<{ rows: LeaderboardRow[] }>("/api/leaderboard"),
  runs: () => getJSON<{ experiments: any[] }>("/api/runs"),
  runDetail: (name: string) => getJSON<any>(`/api/runs/${name}/detail`),
  launch: (body: any) => postJSON<{ enqueued: string }>("/api/runs", body),
  cancel: (name: string) => postJSON<{ cancelled: boolean }>(`/api/runs/${name}/cancel`, {}),
  log: (name: string) => getJSON<{ log: string }>(`/api/runs/${name}/log`),
  compare: (selectors: any[]) => postJSON<any>("/api/compare", { selectors }),
  report: (body: any) => postJSON<{ format: string; content: string }>("/api/reports", body),
  notebook: () => getJSON<{ entries: any[] }>("/api/notebook"),
  addNote: (body: any) => postJSON<any>("/api/notebook", body),
};
