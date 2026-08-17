/**
 * The one place provenance is shown, so the one place it can be wrong.
 *
 * The case that matters is a reader who chose R, is looking at a number Python
 * produced, and has no way to tell. That has to be stated on the same screen as
 * the number -- and only then, because an amber line that appears when nothing
 * is wrong stops being read.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import EngineBadgeBar from "./EngineBadgeBar";
import { useStore } from "../store";

beforeEach(() => {
  useStore.setState({ engine: "python", activeTab: "tests", engineNotices: {} });
});

describe("the persistent half", () => {
  it("names the Python engine in a Python session", () => {
    render(<EngineBadgeBar />);
    expect(screen.getByText("Python-based statistics")).toBeInTheDocument();
  });

  it("names the R engine in an R session, before anything has run", () => {
    useStore.setState({ engine: "r" });
    render(<EngineBadgeBar />);
    expect(screen.getByText("R-based statistics")).toBeInTheDocument();
    expect(screen.queryByText(/Not yet available in R/)).not.toBeInTheDocument();
  });

  it("shows what actually answered once a local run has reported it", () => {
    useStore.setState({
      engine: "r",
      engineNotices: { "stats.ttest": { engine: "r", engineDetail: "R 4.6.0 · webR 0.6.0", at: 1 } },
    });
    render(<EngineBadgeBar />);
    expect(screen.getByText("R 4.6.0 · webR 0.6.0")).toBeInTheDocument();
  });
});

describe("the amber half", () => {
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
    render(<EngineBadgeBar />);
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
    render(<EngineBadgeBar />);
    expect(screen.queryByText(/Not yet available in R/)).not.toBeInTheDocument();
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
    render(<EngineBadgeBar />);
    // Power fell back more recently, but the reader is looking at the t-test.
    expect(screen.queryByText(/Not yet available in R/)).not.toBeInTheDocument();
    expect(screen.getByText("R 4.6.0 · webR 0.6.0")).toBeInTheDocument();
  });

  it("clears once the same tab is answered by R again", () => {
    useStore.setState({
      engine: "r",
      activeTab: "tests",
      engineNotices: {
        "stats.ttest": { engine: "python", fellBackBecause: "r-runtime-load-failed", at: 1 },
      },
    });
    const { rerender } = render(<EngineBadgeBar />);
    expect(screen.getByText(/Not yet available in R/)).toBeInTheDocument();

    useStore.setState({
      engineNotices: {
        "stats.ttest": { engine: "r", engineDetail: "R 4.6.0 · webR 0.6.0", at: 2 },
      },
    });
    rerender(<EngineBadgeBar />);
    expect(screen.queryByText(/Not yet available in R/)).not.toBeInTheDocument();
  });
});
