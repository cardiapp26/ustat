import { type ReactNode } from "react";

/**
 * Shared hover-tooltip component.
 * Renders a small circled "?" that reveals a tooltip on hover.
 */
export function Tip({ text, wide }: { text: string; wide?: boolean }) {
  return (
    <span className="relative group inline-block ml-1 align-middle">
      <span className="text-[10px] text-gray-300 hover:text-indigo-400 cursor-help border border-gray-200 hover:border-indigo-300 rounded-full w-3.5 h-3.5 inline-flex items-center justify-center transition-colors leading-none select-none">?</span>
      {/* Spans, not divs: Tip is used inside <p> in a dozen panels, and a <div>
          there is invalid DOM that React reports as a hydration error. The
          hidden/group-hover:block classes drive display, so this renders the
          same as before. */}
      <span className={`absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 ${wide ? "w-72" : "w-56"} bg-gray-900 text-white text-[11px] leading-relaxed rounded-lg px-3 py-2.5 shadow-xl pointer-events-none`}>
        {text}
        <span className="block absolute top-full left-4 border-4 border-transparent border-t-gray-900" />
      </span>
    </span>
  );
}

/**
 * Label + inline Tip in one component.
 * Usage: <LabelTip tip="Explanation">Label text</LabelTip>
 */
export function LabelTip({ children, tip, wide }: { children: ReactNode; tip: string; wide?: boolean }) {
  return (
    <span className="inline-flex items-center gap-0">
      {children}
      <Tip text={tip} wide={wide} />
    </span>
  );
}

/**
 * Thin info banner — a light-blue callout for interpreting a specific result.
 */
export function InfoBanner({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 bg-indigo-50 border border-indigo-100 rounded-lg px-3 py-2 text-xs text-indigo-700 leading-relaxed">
      <span className="mt-0.5 flex-shrink-0">💡</span>
      <span>{children}</span>
    </div>
  );
}
