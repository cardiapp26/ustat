/**
 * The output document: one ordered record of the results a user chose to
 * keep from this session, in the spirit of SPSS's Output Viewer.
 *
 * Each item is a still copy of a result area at the moment it was added:
 * sanitised HTML (tables and text as the panel drew them, plots as PNG), a
 * title and a free-text note. A still copy rather than a live reference
 * because a document is a record of what was reported: re-running the panel
 * later must not silently change a table already written up.
 *
 * Kept in its own small store so the main store does not grow, and persisted
 * through `collectUiState` / `applyUiState`, which is how autosave and the
 * .ustat project file already carry the rest of the panel state.
 */
import { create } from "zustand";
import { sanitizeOutputHtml } from "./outputSanitize";
import { useStore } from "../store";

export interface OutputItem {
  id: string;
  title: string;
  /** Header tab the result came from, for the item's subtitle. */
  tab: string;
  createdAt: number;
  /** Sanitised HTML of the captured result area. */
  html: string;
  note: string;
}

interface OutputDocState {
  items: OutputItem[];
  add: (item: Omit<OutputItem, "id" | "createdAt" | "note"> & Partial<Pick<OutputItem, "note">>) => OutputItem;
  remove: (id: string) => void;
  move: (id: string, delta: -1 | 1) => void;
  setNote: (id: string, note: string) => void;
  setTitle: (id: string, title: string) => void;
  clear: () => void;
  replaceAll: (items: OutputItem[]) => void;
}

const newId = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

export const useOutputDoc = create<OutputDocState>((set) => ({
  items: [],
  add: (input) => {
    const item: OutputItem = {
      id: newId(),
      createdAt: Date.now(),
      note: input.note ?? "",
      title: input.title,
      tab: input.tab,
      html: input.html,
    };
    set((s) => ({ items: [...s.items, item] }));
    return item;
  },
  remove: (id) => set((s) => ({ items: s.items.filter((i) => i.id !== id) })),
  move: (id, delta) =>
    set((s) => {
      const from = s.items.findIndex((i) => i.id === id);
      const to = from + delta;
      if (from < 0 || to < 0 || to >= s.items.length) return s;
      const items = [...s.items];
      const [moved] = items.splice(from, 1);
      items.splice(to, 0, moved);
      return { items };
    }),
  setNote: (id, note) => set((s) => ({ items: s.items.map((i) => (i.id === id ? { ...i, note } : i)) })),
  setTitle: (id, title) => set((s) => ({ items: s.items.map((i) => (i.id === id ? { ...i, title } : i)) })),
  clear: () => set({ items: [] }),
  replaceAll: (items) => set({ items }),
}));

/**
 * Items read back from a saved session or a project file. The file may come
 * from someone else, so every field is checked and the HTML is sanitised
 * again: it is about to be rendered with dangerouslySetInnerHTML.
 */
export function restoreOutputItems(raw: unknown): OutputItem[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter(
      (i): i is OutputItem =>
        typeof i === "object" && i !== null &&
        typeof (i as OutputItem).id === "string" &&
        typeof (i as OutputItem).html === "string" &&
        typeof (i as OutputItem).title === "string",
    )
    .map((i) => ({
      id: i.id,
      title: i.title.slice(0, 300),
      tab: typeof i.tab === "string" ? i.tab : "",
      createdAt: typeof i.createdAt === "number" ? i.createdAt : Date.now(),
      note: typeof i.note === "string" ? i.note : "",
      html: sanitizeOutputHtml(i.html),
    }));
}

// A document belongs to one dataset session, like saved analyses do: a new
// dataset starts an empty document. applyUiState runs after setSession, so a
// restored session refills it right after this empties it.
useStore.subscribe((state, prev) => {
  if (state.session?.session_id !== prev.session?.session_id) {
    useOutputDoc.getState().clear();
  }
});
