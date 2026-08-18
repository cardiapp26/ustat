/**
 * The one place provenance is shown, so the one place it can be wrong.
 *
 * The case that matters is a reader who chose R, is looking at a number Python
 * produced, and has no way to tell. That has to be stated on the same screen as
 * the number -- and only then, because an amber line that appears when nothing
 * is wrong stops being read.
 *
 * The second case is quieter and was reported from use: the header named R on a
 * session whose welcome screen had Python selected. Nothing was broken -- the
 * session had been resumed in the engine it was worked in -- but nothing said
 * so either, so the chip has to.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { EngineChip, EngineFallbackNotice } from "./EngineProvenance";
import { useStore } from "../store";

beforeEach(() => {
  useStore.setState({
    engine: "python",
    engineSource: "gate",
    activeTab: "tests",
    engineNotices: {},
  });
});

describe("the chip", () => {
  it("names the Python engine in a Python session", () => {
    render(<EngineChip />);
    expect(screen.getByText(/Python-based statistics/)).toBeInTheDocument();
  });

  it("names the R engine in an R session, before anything has run", () => {
    useStore.setState({ engine: "r" });
    render(<EngineChip />);
    expect(screen.getByText(/R-based statistics/)).toBeInTheDocument();
    expect(screen.queryByText(/Not yet available in R/)).not.toBeInTheDocument();
  });

  it("shows what actually answered once a local run has reported it", () => {
    useStore.setState({
      engine: "r",
      engineNotices: { "stats.ttest": { engine: "r", engineDetail: "R 4.6.0 · webR 0.6.0", at: 1 } },
    });
    render(<EngineChip />);
    expect(screen.getByText(/R 4\.6\.0 · webR 0\.6\.0/)).toBeInTheDocument();
  });

  it("says the engine was restored rather than chosen, when it was", () => {
    useStore.setState({ engine: "r", engineSource: "resume" });
    render(<EngineChip />);
    expect(screen.getByText(/restored/)).toBeInTheDocument();
    expect(screen.getByTitle(/did not pass the welcome screen/)).toBeInTheDocument();
  });

  it("says nothing about restoring when the gate is where the choice was made", () => {
    useStore.setState({ engine: "r", engineSource: "gate" });
    render(<EngineChip />);
    expect(screen.queryByText(/restored/)).not.toBeInTheDocument();
    expect(screen.getByTitle(/Chosen on the welcome screen/)).toBeInTheDocument();
  });
});

describe("the amber notice", () => {
  it("appears when an R session's tab was last answered by Python", () => {
    useStore.setState({
      engine: "r",
      activeTab: "power",
      engineNotices: {
        "stats.power": {
          engine: "python",
          engineDetail: "Python 3.12 · scipy 1.14.1",
          fellBackBecause: "no-r-implementation",
          at: 1,
        },
      },
    });
    render(<EngineFallbackNotice />);
    expect(
      screen.getByText(
        "Not yet available in R — computed with Python (scipy 1.14.1). Numbers are unchanged from the Python engine.",
      ),
    ).toBeInTheDocument();
  });

  it("stays away in a Python session, where Python answering is the whole point", () => {
    useStore.setState({
      engine: "python",
      activeTab: "power",
      engineNotices: {
        "stats.power": { engine: "python", fellBackBecause: "disabled-by-user", at: 1 },
      },
    });
    const { container } = render(<EngineFallbackNotice />);
    expect(container).toBeEmptyDOMElement();
  });

  it("reads the tab on screen, not whatever ran last anywhere", () => {
    useStore.setState({
      engine: "r",
      activeTab: "tests",
      engineNotices: {
        "stats.ttest": { engine: "r", engineDetail: "R 4.6.0 · webR 0.6.0", at: 1 },
        "stats.power": { engine: "python", fellBackBecause: "no-r-implementation", at: 2 },
      },
    });
    const { container } = render(<EngineFallbackNotice />);
    // Power fell back more recently, but the reader is looking at the t-test.
    expect(container).toBeEmptyDOMElement();
  });

  it("clears once the same tab is answered by R again", () => {
    useStore.setState({
      engine: "r",
      activeTab: "tests",
      engineNotices: {
        "stats.ttest": { engine: "python", fellBackBecause: "r-runtime-load-failed", at: 1 },
      },
    });
    const { container, rerender } = render(<EngineFallbackNotice />);
    expect(screen.getByText(/Not yet available in R/)).toBeInTheDocument();

    useStore.setState({
      engineNotices: {
        "stats.ttest": { engine: "r", engineDetail: "R 4.6.0 · webR 0.6.0", at: 2 },
      },
    });
    rerender(<EngineFallbackNotice />);
    expect(container).toBeEmptyDOMElement();
  });

  it("does not repeat the fallback engine in the chip, which the amber line already names", () => {
    useStore.setState({
      engine: "r",
      activeTab: "power",
      engineNotices: {
        "stats.power": {
          engine: "python",
          engineDetail: "Python 3.12 · scipy 1.14.1",
          fellBackBecause: "no-r-implementation",
          at: 1,
        },
      },
    });
    render(<EngineChip />);
    expect(screen.queryByText(/Python 3\.12 · scipy 1\.14\.1/)).not.toBeInTheDocument();
  });
});
