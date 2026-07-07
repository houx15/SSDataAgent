import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { StrategyBoard } from "./StrategyBoard";
import { StrategyBoardRow } from "../api";

const row: StrategyBoardRow = {
  family: "Design A", blurb: "…", best_dataset: "gss",
  data_mode: "full training microdata", condition: "design_a_full",
  experiment: "ship_designs_ac_cross", run_id: "20260707-083351", model: "gpt-5.4",
  overall_average: 0.61, paper_overall: 0.39, delta_overall: 0.22,
  types: [
    { t: "T1", ours: 0.4, paper: 0.13, delta: 0.27 },
    { t: "T2", ours: 0.77, paper: 0.71, delta: 0.06 },
  ],
  n_runs: 7,
};

test("board shows strategy, best dataset, overall and vs-PNAS delta", () => {
  render(<MemoryRouter><StrategyBoard rows={[row]} /></MemoryRouter>);
  expect(screen.getByText("Design A")).toBeTruthy();
  expect(screen.getByText("gss")).toBeTruthy();
  expect(screen.getByText("0.610")).toBeTruthy();      // overall
  expect(screen.getByText(/\+0\.220/)).toBeTruthy();   // delta vs PNAS
  // T1 cell is paired ours / paper: 0.400 (ours) and its PNAS baseline 0.130
  expect(screen.getByText("0.400")).toBeTruthy();
  expect(screen.getByText(/\/ 0\.130/)).toBeTruthy();
  // family name links to its detail page
  const link = screen.getByText("Design A").closest("a");
  expect(link?.getAttribute("href")).toBe("/strategy/Design%20A");
});
