/** Picker for the treated level of a two-level text treatment (PSM / IPTW). */

interface TreatedLevelPickerProps {
  levels: string[];
  value: string;
  onChange: (level: string) => void;
  /** Defaults to the PSM / IPTW wording. */
  label?: string;
  placeholder?: string;
}

/**
 * Which level of a text treatment is the treated group. Nothing is preselected:
 * the choice sets the sign of the effect, so the user has to make it.
 */
export function TreatedLevelPicker({
  levels, value, onChange,
  label = "Treated level (the other one is control)",
  placeholder = "Select the treated level…",
}: TreatedLevelPickerProps) {
  const chosen = levels.includes(value) ? value : "";
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] text-gray-500">{label}</span>
      <select className="select w-full" value={chosen} onChange={(e) => onChange(e.target.value)}>
        <option value="" disabled>{placeholder}</option>
        {levels.map((l) => <option key={l} value={l}>{l}</option>)}
      </select>
      {!chosen && (
        <span className="text-[10px] text-amber-600">Pick a level before running.</span>
      )}
    </label>
  );
}
