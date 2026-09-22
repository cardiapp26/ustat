import { useState } from "react";
import { staleExportTitle, useStaleGuard } from "../lib/staleGuard";

interface CopyTextButtonProps {
  /** The text, or a function building it when the button is pressed. */
  text: string | (() => string);
  label?: string;
  title?: string;
  className?: string;
}

const DEFAULT_CLASS =
  "text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-gray-500";

/** A Copy button that refuses to copy an out-of-date result: a pasted
 *  paragraph carries no trace of the data it described. */
export default function CopyTextButton({ text, label = "Copy", title, className = DEFAULT_CLASS }: CopyTextButtonProps) {
  const guard = useStaleGuard();
  const [copied, setCopied] = useState(false);

  const handle = async () => {
    if (guard.stale) return;
    try {
      await navigator.clipboard.writeText(typeof text === "function" ? text() : text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard refused (permissions / insecure context): nothing to undo */
    }
  };

  return (
    <button
      type="button"
      onClick={handle}
      disabled={guard.stale}
      title={guard.stale ? staleExportTitle(guard.reason) : title}
      className={className}
    >
      {copied ? "Copied ✓" : label}
    </button>
  );
}
