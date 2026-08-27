import { describe, expect, it } from "vitest";
import {
  normalizeDictionaryName,
  parseWideDictionaryRows,
  suggestDictionaryColumnMatches,
} from "./dictionaryValueLabels";

describe("dictionary value-label import", () => {
  it("parses wide label:code and code=label dictionaries", () => {
    const parsed = parseWideDictionaryRows([
      ["Cinsiyet", "Kateter Çeşidi", "Exitus nedeni"],
      ["E:1", "1=Normal;20F", "PEG ned:0"],
      ["K:0", "0=Takılmayan", "PEG dışı.1"],
      ["Unknown:U", null, null],
    ]);

    expect(parsed.columns).toHaveLength(3);
    expect(parsed.columns[0].labels).toEqual({ "0": "K", "1": "E", U: "Unknown" });
    expect(parsed.columns[1].labels).toEqual({ "0": "Takılmayan", "1": "Normal;20F" });
    expect(parsed.columns[2].labels).toEqual({ "0": "PEG ned", "1": "PEG dışı" });
    expect(parsed.columns[2].warnings[0]).toMatch(/interpreted/);
  });

  it("keeps first duplicate code and reports malformed cells", () => {
    const parsed = parseWideDictionaryRows([
      ["Group"],
      ["Control:0"],
      ["Patient:0"],
      ["not a definition"],
    ]);

    expect(parsed.columns[0].labels).toEqual({ "0": "Control" });
    expect(parsed.columns[0].warnings).toHaveLength(2);
  });

  it("stores special string codes without changing object prototypes", () => {
    const parsed = parseWideDictionaryRows([
      ["Group"],
      ["Prototype:__proto__"],
      ["Constructor:constructor"],
    ]);

    const labels = parsed.columns[0].labels;
    expect(Object.getPrototypeOf(labels)).toBeNull();
    expect(Object.prototype.hasOwnProperty.call(labels, "__proto__")).toBe(true);
    expect(labels.__proto__).toBe("Prototype");
    expect(labels.constructor).toBe("Constructor");
  });

  it("normalizes Turkish and SPSS-safe names", () => {
    expect(normalizeDictionaryName("Başvuru Yeri")).toBe("basvuruyeri");
    expect(normalizeDictionaryName("Ba_vuru_Yeri")).toBe("bavuruyeri");
  });

  it("matches headers to variable names or labels and leaves weak matches empty", () => {
    const definitions = parseWideDictionaryRows([
      ["Cinsiyet", "PEG Endikasyonu", "Başvuru Yeri", "Unrelated"],
      ["K:0", "SVO:3", "Servis:1", "No:0"],
    ]).columns;
    const matches = suggestDictionaryColumnMatches(definitions, [
      { name: "Cinsiyet" },
      { name: "PEG_Endikasyonu", label: "PEG Endikasyonu" },
      { name: "Ba_vuru_Yeri", label: "Başvuru Yeri" },
      { name: "CRP" },
    ]);

    expect(matches.map((match) => match.targetName)).toEqual([
      "Cinsiyet",
      "PEG_Endikasyonu",
      "Ba_vuru_Yeri",
      "",
    ]);
    expect(matches[2].reason).toBe("label");
  });
});
