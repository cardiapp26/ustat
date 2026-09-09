import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import ResultProvenanceLine from "./ResultProvenanceLine";
import { useStore } from "../store";
import { installSession } from "../test/testUtils";
import type { Provenance } from "../lib/engine/provenance";

const serverPython: Provenance = {
  runtime: "server",
  engine: "python",
  languageVersion: "3.11.9",
  packages: { scipy: "1.14.1", numpy: "2.0.2" },
  engineVersion: "0.1.0",
  engineFingerprint: "b8a7237331bddcf1",
  at: 0,
};

const localR: Provenance = {
  runtime: "local",
  engine: "r",
  languageVersion: "4.5.2",
  packages: { webR: "0.6.0" },
  at: 0,
};

describe("ResultProvenanceLine", () => {
  beforeEach(() => installSession());

  it("says nothing when nothing reported a run", () => {
    const { container } = render(<ResultProvenanceLine provenance={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("names the engine, the version that decides the number, and where it ran", () => {
    render(<ResultProvenanceLine provenance={serverPython} />);
    expect(screen.getByText(/Python 3.11.9 · scipy 1.14.1 · on the server/)).toBeInTheDocument();
  });

  it("calls out an R session whose number Python computed", () => {
    // The reported bug, at the only place it can be checked: beside the number.
    useStore.getState().setEngine("r");
    render(<ResultProvenanceLine provenance={serverPython} />);
    expect(screen.getByText(/Computed with Python, not R/)).toBeInTheDocument();
  });

  it("stays quiet when the engine is the one the session chose", () => {
    useStore.getState().setEngine("r");
    render(<ResultProvenanceLine provenance={localR} />);
    expect(screen.queryByText(/not R/)).not.toBeInTheDocument();
    expect(screen.getByText(/R 4.5.2 · in your browser/)).toBeInTheDocument();
  });

  it("carries every version in the tooltip, not just the headline one", () => {
    render(<ResultProvenanceLine provenance={serverPython} />);
    const title = screen.getByText(/on the server/).closest("p")?.getAttribute("title") ?? "";
    expect(title).toContain("numpy 2.0.2");
    expect(title).toContain("scipy 1.14.1");
    expect(title).toContain("uSTAT engine: 0.1.0 (b8a7237331bddcf1)");
  });

  it("says why the server answered when it was a fallback", () => {
    useStore.getState().setEngine("r");
    render(
      <ResultProvenanceLine
        provenance={{ ...serverPython, fellBackBecause: "not-implemented-in-r" }}
      />,
    );
    expect(screen.getByText(/not-implemented-in-r/)).toBeInTheDocument();
  });
});
