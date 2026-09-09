import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import IngestReportNotice from "./IngestReportNotice";
import { useStore } from "../store";
import type { IngestColumnReport, IngestReport } from "../api";
import { installSession } from "../test/testUtils";

function column(over: Partial<IngestColumnReport> = {}): IngestColumnReport {
  return {
    column: "CRP",
    decision: "kept_text",
    reason: "kept as text: 1 value(s) at a measurement limit (e.g. <, >).",
    n_values: 100,
    n_changed: 0,
    n_missing_coded: 0,
    n_discarded: 0,
    decimal_separator: null,
    missing_codes: {},
    discarded_examples: [],
    ...over,
  };
}

function report(over: Partial<IngestReport> = {}): IngestReport {
  return {
    n_columns_examined: 3,
    n_columns_converted: 1,
    n_cells_changed: 2,
    n_cells_discarded: 0,
    needs_review: false,
    columns: [],
    ...over,
  };
}

function install(r: IngestReport | null) {
  installSession();
  useStore.getState().setIngestReport(r);
}

describe("IngestReportNotice", () => {
  beforeEach(() => install(null));

  it("says nothing when the import changed nothing", () => {
    const { container } = render(<IngestReportNotice />);
    expect(container).toBeEmptyDOMElement();
  });

  it("says nothing when there is no report at all", () => {
    install(null);
    const { container } = render(<IngestReportNotice />);
    expect(container).toBeEmptyDOMElement();
  });

  it("names the measurement-limit values and calls them a result, not a blank", () => {
    install(report({
      needs_review: true,
      columns: [column({
        censored: {
          n: 1,
          operators: { "<": 1 },
          limits: [0.1],
          examples: [{ row: 40, value: "<0.1" }],
        },
      })],
    }));
    render(<IngestReportNotice />);

    expect(screen.getByText(/need a decision/i)).toBeInTheDocument();
    // Row numbers are 1-based on screen; the payload is 0-based.
    expect(screen.getByText(/“<0.1” \(row 41\)/)).toBeInTheDocument();
    expect(screen.getByText(/is a result, not a blank/i)).toBeInTheDocument();
    expect(screen.getByText("kept as text")).toBeInTheDocument();
  });

  it("warns about mixed units rather than stripping them", () => {
    install(report({
      needs_review: true,
      columns: [column({
        column: "GLU",
        units: {
          n: 2,
          suffixes: { "mg/dL": 1, "mmol/L": 1 },
          examples: [{ row: 0, value: "12 mg/dL" }],
        },
      })],
    }));
    render(<IngestReportNotice />);
    expect(screen.getByText(/mg\/dL, mmol\/L/)).toBeInTheDocument();
    expect(screen.getByText(/order-of-magnitude error/i)).toBeInTheDocument();
  });

  it("says where the original text of a blanked cell can be read", () => {
    install(report({
      needs_review: true,
      n_cells_discarded: 1,
      columns: [column({
        column: "X",
        decision: "numeric",
        n_discarded: 1,
        discarded_examples: [{ row: 99, value: "see notes" }],
      })],
    }));
    render(<IngestReportNotice />);
    expect(screen.getByText(/“see notes” \(row 100\)/)).toBeInTheDocument();
    expect(screen.getByText(/original text is kept with the session/i)).toBeInTheDocument();
  });

  it("stays quiet in tone when the only changes lose nothing", () => {
    install(report({
      needs_review: false,
      n_cells_changed: 2,
      columns: [column({
        column: "BMI",
        decision: "numeric",
        decimal_separator: ",",
        n_changed: 2,
        reason: "converted to numeric from text",
      })],
    }));
    render(<IngestReportNotice />);
    expect(screen.queryByText(/need a decision/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Import tidied 2 cells in 1 column/)).toBeInTheDocument();
    expect(screen.getByText(/Decimal separator read as a comma/)).toBeInTheDocument();
  });

  it("lists declared missing codes with their counts", () => {
    install(report({
      columns: [column({
        column: "LDL",
        decision: "numeric",
        n_missing_coded: 3,
        missing_codes: { NA: 2, "?": 1 },
      })],
    }));
    render(<IngestReportNotice />);
    expect(screen.getByText(/“NA” ×2, “\?” ×1/)).toBeInTheDocument();
  });

  it("collapses and dismisses", async () => {
    install(report({ columns: [column({ column: "BMI", decision: "numeric" })] }));
    render(<IngestReportNotice />);
    expect(screen.getByText("BMI")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { expanded: true }));
    expect(screen.queryByText("BMI")).not.toBeInTheDocument();

    await userEvent.click(screen.getByTitle(/Dismiss/));
    expect(useStore.getState().ingestReport).toBeNull();
  });
});
