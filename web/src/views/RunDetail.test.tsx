import { render, screen } from "@testing-library/react";
import { RunDetailView } from "./RunDetail";

test("renders experiment name and artifact path without object-as-child crash", () => {
  const detail = {
    experiment: { name: "exp_a", status: "done", model: "m" },
    runs: [{ condition: "full_agent", dataset: "gss", run_id: "20260630-100000",
             eval: { by_type: { type1: 0.6 }, overall_average: 0.6 },
             meta: { git_sha: "abc" },
             artifacts: { generated_csv: "exp_a/full_agent/gss/20260630-100000/generated.csv" } }],
  };
  render(<RunDetailView detail={detail} />);
  expect(screen.getByText("exp_a")).toBeTruthy();
  expect(screen.getByText(/generated\.csv/)).toBeTruthy();
});
