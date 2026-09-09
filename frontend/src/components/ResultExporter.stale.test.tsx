import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ResultExporter from "./ResultExporter";

const HEADERS = ["Variable", "Estimate"];
const ROWS = [["AGE", 1.04]];

describe("ResultExporter — stale results cannot be exported", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("exports normally when the result is current", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    render(<ResultExporter title="Cox" headers={HEADERS} rows={ROWS} />);
    const csv = screen.getByRole("button", { name: "CSV" });
    expect(csv).toBeEnabled();
    await userEvent.click(csv);
    expect(click).toHaveBeenCalled();
  });

  it("disables every export button while stale", () => {
    render(<ResultExporter title="Cox" headers={HEADERS} rows={ROWS} stale staleReason="the data changed" />);
    for (const name of ["CSV", "XLSX", "⧉ Copy"]) {
      expect(screen.getByRole("button", { name })).toBeDisabled();
    }
  });

  it("says why, so a blocked button is not a mystery", () => {
    render(<ResultExporter title="Cox" headers={HEADERS} rows={ROWS} stale staleReason="the data changed" />);
    expect(screen.getByText("Export blocked")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "CSV" }))
      .toHaveAttribute("title", "Recompute first — this result predates the data changed");
  });

  it("appends the engine and library versions to a CSV", async () => {
    // A CSV leaves the app with none of the interface around it, so a figure
    // whose provenance was only on screen becomes a figure with none.
    let written = "";
    const originalCreate = URL.createObjectURL;
    URL.createObjectURL = ((blob: Blob) => {
      void blob.text().then((t) => { written = t; });
      return "blob:stub";
    }) as typeof URL.createObjectURL;
    URL.revokeObjectURL = (() => {}) as typeof URL.revokeObjectURL;
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(
      <ResultExporter
        title="Cox"
        headers={HEADERS}
        rows={ROWS}
        provenance={{
          runtime: "server", engine: "python", languageVersion: "3.11.9",
          packages: { scipy: "1.14.1" }, engineVersion: "0.1.0", at: 0,
        }}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: "CSV" }));
    await new Promise((r) => setTimeout(r, 0));

    expect(written).toContain("Provenance");
    expect(written).toContain("Engine");
    expect(written).toContain("scipy 1.14.1");
    expect(written).toContain("Computed");
    URL.createObjectURL = originalCreate;
  });

  it("exports the plain table when nothing reported a run", async () => {
    let written = "";
    const originalCreate = URL.createObjectURL;
    URL.createObjectURL = ((blob: Blob) => {
      void blob.text().then((t) => { written = t; });
      return "blob:stub";
    }) as typeof URL.createObjectURL;
    URL.revokeObjectURL = (() => {}) as typeof URL.revokeObjectURL;
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(<ResultExporter title="Cox" headers={HEADERS} rows={ROWS} />);
    await userEvent.click(screen.getByRole("button", { name: "CSV" }));
    await new Promise((r) => setTimeout(r, 0));

    expect(written).not.toContain("Provenance");
    expect(written.trim().split("\r\n")).toHaveLength(2);
    URL.createObjectURL = originalCreate;
  });

  it("writes no file even if the click gets through", async () => {
    // Belt and braces: `handle` returns early on stale, so a programmatic
    // click on a disabled-looking button still cannot produce a download.
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    render(<ResultExporter title="Cox" headers={HEADERS} rows={ROWS} stale />);
    screen.getByRole("button", { name: "CSV" }).click();
    expect(click).not.toHaveBeenCalled();
  });
});
