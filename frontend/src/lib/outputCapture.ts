/**
 * Copy a result area into the output document.
 *
 * `<StaleGuard>` brackets its children with two hidden marker elements (see
 * StaleGuard.tsx); the area is everything between them. That avoids wrapping
 * the result in a new element, which would have changed the layout of every
 * panel that uses the guard (Tailwind's space-y and grid rules apply to
 * direct children).
 *
 * Plots are live Plotly elements whose SVG does not survive being copied
 * reliably, so each one is rendered to a PNG first and the copy carries the
 * image in its place.
 */
import { sanitizeOutputHtml } from "./outputSanitize";

/** Attribute names on the two markers <StaleGuard> renders. */
export const OUTPUT_START_ATTR = "data-output-start";
export const OUTPUT_END_ATTR = "data-output-end";
/** Mark any element that must not be copied (toolbars, notices). */
export const OUTPUT_SKIP_ATTR = "data-output-skip";
/** On the start marker: whether the guarded result is out of date now. */
export const OUTPUT_STALE_ATTR = "data-output-stale";

/** Whether the guarded result is current right now (read off the DOM, so an
 *  async loop can wait for a re-render it cannot observe through React). */
export function guardIsCurrent(captureId: string): boolean {
  const start = document.querySelector(`[${OUTPUT_START_ATTR}="${CSS.escape(captureId)}"]`);
  return start?.getAttribute(OUTPUT_STALE_ATTR) === "false";
}

interface PlotlyLike {
  toImage: (gd: HTMLElement, opts: Record<string, unknown>) => Promise<string>;
}

async function plotToPng(gd: HTMLElement): Promise<string | null> {
  try {
    let Plotly = (gd as HTMLElement & { _Plotly?: PlotlyLike })._Plotly;
    if (!Plotly?.toImage) {
      Plotly = (await import("plotly.js/dist/plotly")).default as unknown as PlotlyLike;
    }
    const width = Math.max(gd.clientWidth, 320);
    const height = Math.max(gd.clientHeight, 240);
    return await Plotly.toImage(gd, { format: "png", width, height, scale: 2, setBackground: "opaque" });
  } catch {
    return null;
  }
}

function markers(captureId: string): [Element, Element] | null {
  const start = document.querySelector(`[${OUTPUT_START_ATTR}="${CSS.escape(captureId)}"]`);
  const end = document.querySelector(`[${OUTPUT_END_ATTR}="${CSS.escape(captureId)}"]`);
  return start && end ? [start, end] : null;
}

const CONTENT_SELECTOR = "table, img, h1, h2, h3, h4, h5";

/**
 * Drop the settings that sit inside some result areas.
 *
 * Several panels guard their inputs together with the result (the Fine-Gray
 * column pickers, a chart's width and height boxes). From each form control,
 * climb to the largest ancestor that holds no result content -- no table,
 * figure or heading -- and remove it, which takes a control's label and its
 * group caption with it while leaving the result itself alone.
 */
function removeControlGroups(root: HTMLElement): void {
  const controls = Array.from(root.querySelectorAll("input, select, textarea, button"));
  for (const control of controls) {
    if (!root.contains(control)) continue;
    let target: Element = control;
    while (target.parentElement && target.parentElement !== root
      && !target.parentElement.querySelector(CONTENT_SELECTOR)) {
      target = target.parentElement;
    }
    target.remove();
  }
}

export interface CapturedOutput {
  title: string;
  html: string;
}

/** Capture the area between a guard's markers, or null if it is not on screen. */
export async function captureGuardedArea(captureId: string): Promise<CapturedOutput | null> {
  const found = markers(captureId);
  if (!found) return null;
  const range = document.createRange();
  range.setStartAfter(found[0]);
  range.setEndBefore(found[1]);

  const livePlots = Array.from(document.querySelectorAll<HTMLElement>(".js-plotly-plot"))
    .filter((el) => range.intersectsNode(el));
  const images = await Promise.all(livePlots.map(plotToPng));

  const fragment = range.cloneContents();
  const clonedPlots = Array.from(fragment.querySelectorAll<HTMLElement>(".js-plotly-plot"));
  clonedPlots.forEach((clone, i) => {
    const src = images[i];
    if (src) {
      const img = document.createElement("img");
      img.setAttribute("src", src);
      img.setAttribute("alt", "Figure");
      img.setAttribute("class", "max-w-full");
      clone.replaceWith(img);
    } else {
      clone.remove();
    }
  });
  fragment.querySelectorAll(`[${OUTPUT_SKIP_ATTR}], details, .modebar-container`).forEach((el) => el.remove());

  const holder = document.createElement("div");
  holder.appendChild(fragment);
  removeControlGroups(holder);
  const heading = holder.querySelector("h1, h2, h3, h4, h5");
  const title = heading?.textContent?.trim().replace(/\s+/g, " ").slice(0, 200) ?? "";
  return { title, html: sanitizeOutputHtml(holder.innerHTML) };
}
