import { render, screen } from "@testing-library/react";
import { LeaderboardTable } from "./Leaderboard";

test("renders champion marker and scores", () => {
  render(<LeaderboardTable rows={[{
    experiment: "exp_b", condition: "full_agent", dataset: "gss",
    by_type: { type1: 0.7 }, overall_average: 0.7, overdetermination_gap: 0.2,
    is_pilot: false, is_champion: true,
  }]} />);
  expect(screen.getByText("exp_b")).toBeTruthy();
  expect(screen.getByText("★")).toBeTruthy();
});
