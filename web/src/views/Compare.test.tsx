// vi.mock is hoisted by vitest before imports, so this mock intercepts
// the react-plotly.js import inside Compare.tsx — preventing jsdom canvas errors.
vi.mock("react-plotly.js", () => ({ default: () => null }));

import { render } from "@testing-library/react";
import { CompareHeatmap } from "./Compare";

test("renders heatmap container with types", () => {
  const { container } = render(
    <CompareHeatmap types={["type1"]} matrix={[[0.5]]}
      cells={[{ selector: { experiment: "a", condition: "c", dataset: "d" },
                by_type: { type1: 0.5 }, overall_average: 0.5, overdetermination_gap: null }]} />);
  expect(container).toBeTruthy();
});
