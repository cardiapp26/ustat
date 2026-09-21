import { describe, expect, it } from "vitest";
import { compareCells, type SortDir } from "./cellSort";
import { dateSortKeys } from "./dateSortKeys";

function sortDates(values: (string | null)[], dir: SortDir = "asc"): (string | null)[] {
  const keys = dateSortKeys(values);
  return [...values].sort((a, b) => compareCells(a, b, dir, keys));
}

describe("date sort keys", () => {
  it("orders day.month.year text by date, not by its first digits", () => {
    // The column from the report: sorted as text, 14.09.2022 came before 23.09.2013.
    const column = ["14.09.2022", "23.09.2013", "01.03.2017", "18.07.2014", "20.01.2014", "24.02.2020", "04.08.2017", "04.09.2013"];
    expect(sortDates(column)).toEqual([
      "04.09.2013", "23.09.2013", "20.01.2014", "18.07.2014",
      "01.03.2017", "04.08.2017", "24.02.2020", "14.09.2022",
    ]);
    expect(sortDates(column, "desc")[0]).toBe("14.09.2022");
  });

  it("decides day-first or month-first once for the whole column", () => {
    // 03/04/2024 is valid both ways; its neighbours say which way the column runs.
    expect(sortDates(["15/06/2024", "03/04/2024", "01/05/2024"])).toEqual(["03/04/2024", "01/05/2024", "15/06/2024"]);
    expect(sortDates(["03/15/2024", "04/02/2024", "03/20/2024"])).toEqual(["03/15/2024", "03/20/2024", "04/02/2024"]);
  });

  it("reads ISO, month names in Turkish and English, and 2-digit years", () => {
    expect(sortDates(["5 Ocak 2024", "2023-12-31", "Jan 2, 2024", "1.1.99", "3 Şubat 2024"])).toEqual([
      "1.1.99", "2023-12-31", "Jan 2, 2024", "5 Ocak 2024", "3 Şubat 2024",
    ]);
  });

  it("breaks same-day ties by time of day, and sorts bare times by time", () => {
    expect(sortDates(["01.02.2024 13:45", "01.02.2024 08:05"])).toEqual(["01.02.2024 08:05", "01.02.2024 13:45"]);
    expect(sortDates(["13:45", "9:05", "00:30:15"])).toEqual(["00:30:15", "9:05", "13:45"]);
  });

  it("keeps missing values last and non-dates after dates, in both directions", () => {
    const column = [null, "unknown", "02.01.2020", "", "01.01.2020"];
    expect(sortDates(column, "asc")).toEqual(["01.01.2020", "02.01.2020", "unknown", null, ""]);
    expect(sortDates(column, "desc")).toEqual(["02.01.2020", "01.01.2020", "unknown", null, ""]);
  });
});

describe("compareCells without date keys", () => {
  it("sinks missing values to the end in a descending sort too", () => {
    const sorted = [3, null, 1, 2].sort((a, b) => compareCells(a, b, "desc"));
    expect(sorted).toEqual([3, 2, 1, null]);
  });

  it("compares numbers numerically and text naturally", () => {
    expect([10, 9, 100].sort((a, b) => compareCells(a, b, "asc"))).toEqual([9, 10, 100]);
    expect(["a10", "a9", "A1"].sort((a, b) => compareCells(a, b, "asc"))).toEqual(["A1", "a9", "a10"]);
  });
});
