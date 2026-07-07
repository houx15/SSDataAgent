import { render, screen } from "@testing-library/react";
import { LeaderboardTable } from "./Leaderboard";

test("renders champion marker, scores, and timepoint", () => {
  render(<LeaderboardTable rows={[{
    experiment: "exp_b", condition: "full_agent", dataset: "gss",
    run_id: "20260706-231310", model: "gpt-5.4-2026-03-05",
    by_type: { type1: 0.7 }, overall_average: 0.7, overdetermination_gap: 0.2,
    is_pilot: false, is_champion: true,
  }]} />);
  expect(screen.getByText("exp_b")).toBeTruthy();
  expect(screen.getByText("★")).toBeTruthy();
  // timepoint (run_id) is surfaced in the leaderboard, not just run-detail
  expect(screen.getByText("20260706-231310")).toBeTruthy();
});
