import { useState, useMemo, useRef, useEffect, useLayoutEffect } from "react";
import type { CSSProperties, Dispatch, ReactNode, RefObject, SetStateAction } from "react";
import { createPortal } from "react-dom";
import { BookOpen, X } from "lucide-react";
import { useStore } from "../store";
import type { ColMeta, Session } from "../store";
import api from "../api";
import { renameColumn } from "../api";
import DataDictionaryPanel from "./DataDictionaryPanel";

// ── Kind cycling ───────────────────────────────────────────────────────────────

const KIND_CYCLE: ColMeta["kind"][] = ["numeric", "categorical", "ordinal", "text", "date"];

const KIND_STYLE: Record<string, string> = {
  numeric:     "bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200",
  categorical: "bg-orange-100 text-orange-700 border-orange-300 hover:bg-orange-200",
  ordinal:     "bg-teal-100 text-teal-700 border-teal-300 hover:bg-teal-200",
  text:        "bg-gray-100 text-gray-500 border-gray-300 hover:bg-gray-200",
  date:        "bg-purple-100 text-purple-700 border-purple-300 hover:bg-purple-200",
};

const KIND_LABEL: Record<string, string> = {
  numeric: "num", categorical: "cat", ordinal: "ord", text: "txt", date: "date",
};

import { SelectCasesModal } from "./datatable/SelectCasesModal";
import { ValueLabelsModal } from "./datatable/ValueLabelsModal";
import { FindReplaceModal } from "./datatable/FindReplaceModal";
import { ParseDatesModal } from "./datatable/ParseDatesModal";
type SortDir = "asc" | "desc";

type ContextMenuAnchor = { x: number; y: number };

const CONTEXT_MENU_MARGIN = 8;

function useViewportContextMenuStyle(
  anchor: ContextMenuAnchor | null,
  menuRef: RefObject<HTMLDivElement | null>,
  fallbackWidth: number,
): CSSProperties {
  const [position, setPosition] = useState({
    left: CONTEXT_MENU_MARGIN,
    top: CONTEXT_MENU_MARGIN,
    // Start tall, not 0 — a 0 here (or a transient innerHeight=0 in PWA/iframe
    // contexts) collapses the menu to ~nothing, so it looks like it closes the
    // instant you try to click an item.
    maxHeight: 9999,
  });

  useLayoutEffect(() => {
    if (!anchor) return;

    const updatePosition = () => {
      const menu = menuRef.current;
      const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1024;
      // innerHeight can briefly report 0 in embedded/PWA/iframe contexts; fall
      // back to the document height and never clamp below a usable minimum.
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 768;
      const maxHeight = Math.max(200, viewportHeight - CONTEXT_MENU_MARGIN * 2);
      const menuWidth = menu?.offsetWidth || fallbackWidth;
      const menuHeight = Math.min(
        menu?.scrollHeight || menu?.offsetHeight || maxHeight,
        maxHeight,
      );

      const preferredLeft =
        anchor.x + menuWidth <= viewportWidth - CONTEXT_MENU_MARGIN
          ? anchor.x
          : anchor.x - menuWidth;
      const left = Math.max(
        CONTEXT_MENU_MARGIN,
        Math.min(preferredLeft, viewportWidth - menuWidth - CONTEXT_MENU_MARGIN),
      );

      const spaceBelow = viewportHeight - CONTEXT_MENU_MARGIN - anchor.y;
      const spaceAbove = anchor.y - CONTEXT_MENU_MARGIN;
      let top: number;
      if (menuHeight <= spaceBelow) {
        top = anchor.y;
      } else if (menuHeight <= spaceAbove) {
        top = anchor.y - menuHeight;
      } else {
        top = CONTEXT_MENU_MARGIN;
      }

      setPosition((current) =>
        current.left === left &&
        current.top === top &&
        current.maxHeight === maxHeight
          ? current
          : { left, top, maxHeight },
      );
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);

    const resizeObserver = new ResizeObserver(updatePosition);
    const mutationObserver = new MutationObserver(updatePosition);
    if (menuRef.current) {
      resizeObserver.observe(menuRef.current);
      mutationObserver.observe(menuRef.current, { childList: true, subtree: true });
    }

    return () => {
      window.removeEventListener("resize", updatePosition);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
    };
  }, [anchor, fallbackWidth, menuRef]);

  return {
    ...position,
    maxWidth: `calc(100vw - ${CONTEXT_MENU_MARGIN * 2}px)`,
    overflowY: "auto",
    overscrollBehavior: "contain",
    scrollbarGutter: "stable",
  };
}

const MENU_ITEM_CLS =
  "w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2";

const SUBMENU_W = 192;

/** One collapsible group in the column context menu. The menu had grown to ~20
 *  flat items; each group opens its items in a flyout on hover so the top level
 *  stays short.
 *
 *  The flyout is PORTALLED to <body>: the menu sets `overflow-y: auto`, and CSS
 *  makes that clip the x-axis too, so a nested absolute flyout was invisible.
 *  Portalling also means the flyout isn't a DOM descendant of the button, hence
 *  the small close delay (so the pointer can travel between the two) and the
 *  `data-colmenu-flyout` marker the outside-click handler looks for.
 *  `flip` opens leftwards when the menu sits near the right viewport edge. */
function ColMenuGroup({
  label, groupKey, activeKey, setActiveKey, flip, tone = "default", children,
}: {
  label: string;
  groupKey: string;
  activeKey: string | null;
  setActiveKey: Dispatch<SetStateAction<string | null>>;
  flip: boolean;
  tone?: "default" | "amber";
  children: ReactNode;
}) {
  const open = activeKey === groupKey;
  const btnRef = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const closeTimer = useRef<number | null>(null);

  const cancelClose = () => {
    if (closeTimer.current !== null) { window.clearTimeout(closeTimer.current); closeTimer.current = null; }
  };
  const openNow = () => { cancelClose(); setActiveKey(groupKey); };
  const scheduleClose = () => {
    cancelClose();
    closeTimer.current = window.setTimeout(() => {
      // Only close if THIS group is still the open one — otherwise a pending
      // timer from the group we just left would slam shut the one the pointer
      // has already moved on to.
      setActiveKey((prev) => (prev === groupKey ? null : prev));
    }, 140);
  };
  useEffect(() => cancelClose, []);

  /* eslint-disable react-hooks/set-state-in-effect -- positioning a portal
     requires a synchronous DOM measurement before paint; this is the documented
     "synchronize with an external system" use of an effect. */
  useLayoutEffect(() => {
    if (!open || !btnRef.current) { setPos(null); return; }
    const btn = btnRef.current.getBoundingClientRect();
    // Anchor to the MENU's edge, not the button's: the menu reserves a
    // scrollbar gutter, so the button stops short and the flyout would overlap
    // the menu's border.
    const menuEl = btnRef.current.closest('[role="menu"]');
    const menuBox = menuEl ? menuEl.getBoundingClientRect() : btn;
    const left = Math.max(
      4,
      Math.min(
        flip ? menuBox.left - SUBMENU_W - 4 : menuBox.right + 4,
        window.innerWidth - SUBMENU_W - 4
      )
    );
    setPos({ top: Math.max(4, btn.top), left });
  }, [open, flip]);
  /* eslint-enable react-hooks/set-state-in-effect */

  return (
    <div className="relative" onMouseEnter={openNow} onMouseLeave={scheduleClose}>
      <button
        ref={btnRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        className={`w-full text-left px-3 py-1.5 text-xs flex items-center justify-between gap-2 ${
          tone === "amber"
            ? "text-amber-700 hover:bg-amber-50"
            : "text-gray-700 hover:bg-gray-50"
        } ${open ? (tone === "amber" ? "bg-amber-50" : "bg-gray-50") : ""}`}
      >
        <span className="flex items-center gap-2">{label}</span>
        <span className="text-gray-400 text-[10px]">▸</span>
      </button>
      {open && pos && createPortal(
        <div
          role="menu"
          data-colmenu-flyout=""
          onMouseEnter={openNow}
          onMouseLeave={scheduleClose}
          className="fixed z-[60] overflow-y-auto bg-white border border-gray-200 rounded-xl shadow-xl py-1"
          style={{
            top: pos.top,
            left: pos.left,
            width: SUBMENU_W,
            maxHeight: `calc(100vh - ${pos.top + 8}px)`,
          }}
        >
          {children}
        </div>,
        document.body
      )}
    </div>
  );
}

export default function DataTable() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <DataTableBody session={session} />;
}

function DataTableBody({ session }: { session: Session }) {
  const updateColumnKind = useStore((s) => s.updateColumnKind);
  const updatePreviewCell = useStore((s) => s.updatePreviewCell);
  const reorderColumns   = useStore((s) => s.reorderColumns);
  const caseFilter       = useStore((s) => s.caseFilter);
  const setCaseFilter    = useStore((s) => s.setCaseFilter);
  const undo             = useStore((s) => s.undo);
  const redo             = useStore((s) => s.redo);
  const undoLen          = useStore((s) => s.undoDepth);
  const redoLen          = useStore((s) => s.redoDepth);
  const columnDecimals   = useStore((s) => s.columnDecimals);
  const setColumnDecimals = useStore((s) => s.setColumnDecimals);
  const clearColumnDecimals = useStore((s) => s.clearColumnDecimals);

  const [sortCol,     setSortCol]     = useState<string | null>(null);
  const [sortDir,     setSortDir]     = useState<SortDir>("asc");
  const [filters,     setFilters]     = useState<Record<string, string>>({});
  const [showFilters, setShowFilters] = useState(false);
  const [editCell,       setEditCell]      = useState<{ rowIdx: number; col: string } | null>(null);
  const [editValue,      setEditValue]     = useState("");
  const [saving,         setSaving]        = useState(false);
  const [showMissingOnly, setShowMissingOnly] = useState(false);
  const [showSelectCases, setShowSelectCases] = useState(false);
  const [showDictionary,  setShowDictionary]  = useState(false);

  // Drag & drop column reordering
  const [dragIdx,  setDragIdx]  = useState<number | null>(null);
  const [dropIdx,  setDropIdx]  = useState<number | null>(null);

  // Frozen (pinned-left) columns. `#` row-number column is always pinned.
  // `frozenCount` = number of leading data columns to freeze.
  const [frozenCount, setFrozenCount] = useState(0);
  const HASH_COL_W = 30;       // width of `#` (row-number) column — kept narrow
  const FROZEN_COL_W = 150;    // forced width per frozen data column
  const frozenLeft = (colIdx: number) => HASH_COL_W + colIdx * FROZEN_COL_W;
  const isFrozenCol = (colIdx: number) => colIdx < frozenCount;
  // Clamp frozenCount when columns are deleted
  useEffect(() => {
    const n = session?.columns.length ?? 0;
    setFrozenCount((c) => Math.min(c, n));
  }, [session?.columns.length]);

  // Column rename
  const [renameCol, setRenameCol] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState("");
  const renameRef = useRef<HTMLInputElement>(null);

  // Right-click context menu (columns)
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; col: string } | null>(null);
  const [openSub, setOpenSub] = useState<string | null>(null);  // expanded submenu group
  const [fillMode, setFillMode] = useState<string | null>(null);
  const [fillVal, setFillVal] = useState("");
  const ctxRef = useRef<HTMLDivElement>(null);
  const fillRef = useRef<HTMLInputElement>(null);

  // Value labels editor
  const [valueLabelCol, setValueLabelCol] = useState<string | null>(null);
  const [findReplaceCol, setFindReplaceCol] = useState<string | null>(null);
  const [parseDateCol, setParseDateCol] = useState<string | null>(null);
  const [valueLabelDraft, setValueLabelDraft] = useState<Record<string, string>>({});

  // Analysis-exclude flag + move-to-position + name-suggestion modals
  const setColumnAnalysisExcluded = useStore((s) => s.setColumnAnalysisExcluded);
  const [moveCol, setMoveCol] = useState<string | null>(null);
  const [suggestOpen, setSuggestOpen] = useState(false);
  const [suggestDraft, setSuggestDraft] = useState<Record<string, string>>({});  // col → target name (editable)
  const [suggestAccept, setSuggestAccept] = useState<Record<string, boolean>>({});
  const [suggestBusy, setSuggestBusy] = useState(false);

  // Multi-cell selection
  const [selectedCells, setSelectedCells] = useState<Set<string>>(new Set());
  const [selAnchor, setSelAnchor] = useState<{ row: number; col: string } | null>(null);
  const [selFocus, setSelFocus] = useState<{ row: number; col: string } | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const dragSelectingRef = useRef(false);
  const dragAnchorRef = useRef<{ row: number; col: string } | null>(null);
  const dragAdditiveRef = useRef(false);
  const dragBaseSelectionRef = useRef<Set<string>>(new Set());

  // Right-click context menu (cells)
  const [cellCtx, setCellCtx] = useState<{ x: number; y: number; row: number; col: string } | null>(null);
  const cellCtxRef = useRef<HTMLDivElement>(null);

  // Right-click context menu (rows)
  const [rowCtx, setRowCtx] = useState<{ x: number; y: number; idx: number } | null>(null);
  const rowCtxRef = useRef<HTMLDivElement>(null);
  // Range spec typed into the row context menu ("100-500, 750")
  const [rowRangeInput, setRowRangeInput] = useState("");
  // Range spec typed into the column context menu ("3-10, 15")
  const [colRangeInput, setColRangeInput] = useState("");
  const columnMenuStyle = useViewportContextMenuStyle(ctxMenu, ctxRef, 192);
  // Flyouts open rightwards by default; near the right edge there isn't room
  // for menu (192) + submenu (192), so flip them to the left instead.
  const subFlip = ctxMenu !== null
    && typeof window !== "undefined"
    && ctxMenu.x + 192 + 192 > window.innerWidth;
  const cellMenuStyle = useViewportContextMenuStyle(cellCtx, cellCtxRef, 192);
  const rowMenuStyle = useViewportContextMenuStyle(rowCtx, rowCtxRef, 176);

  const inputRef   = useRef<HTMLInputElement>(null);
  const committingCellsRef = useRef<Set<string>>(new Set());

  // Paste notification
  const [pasteMsg, setPasteMsg] = useState<string | null>(null);

  // Bulk tick-and-delete: checkboxes in the row gutter and column headers.
  // checkedRows holds ORIGINAL (backend-positional) row indices; checkedCols
  // holds column names. Kept separate from the cell-selection Set above.
  const [checkedRows, setCheckedRows] = useState<Set<number>>(new Set());
  const [checkedCols, setCheckedCols] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (editCell) setTimeout(() => inputRef.current?.focus(), 0);
  }, [editCell]);

  useEffect(() => {
    setSortCol(null); setFilters({}); setShowMissingOnly(false); setSelectedCells(new Set());
    setSelAnchor(null); setSelFocus(null);
  }, [session?.session_id]);

  const { preview, columns } = session;

  type IndexedRow = Record<string, unknown> & { _idx: number };
  const indexedRows = useMemo(
    () => preview.map((row, idx): IndexedRow => ({ ...row, _idx: idx })),
    [preview]
  );

  // Per-column missing counts (computed once over full preview)
  const missingCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const col of columns) {
      counts[col.name] = preview.filter(
        (row) => row[col.name] === null || row[col.name] === undefined || row[col.name] === ""
      ).length;
    }
    return counts;
  }, [preview, columns]);

  const totalMissingRows = useMemo(
    () => indexedRows.filter((row) =>
      columns.some((col) => row[col.name] === null || row[col.name] === undefined || row[col.name] === "")
    ).length,
    [indexedRows, columns]
  );

  const filtered = useMemo(() => {
    const hasFilters = Object.values(filters).some(Boolean);
    let rows = indexedRows;

    if (showMissingOnly) {
      rows = rows.filter((row) =>
        columns.some((col) => row[col.name] === null || row[col.name] === undefined || row[col.name] === "")
      );
    }

    if (!hasFilters) return rows;
    return rows.filter((row) =>
      columns.every((col) => {
        const f = filters[col.name];
        if (!f) return true;
        const cell = row[col.name];
        if (cell === null || cell === undefined) return f === "";
        return String(cell).toLowerCase().includes(f.toLowerCase());
      })
    );
  }, [indexedRows, filters, columns, showMissingOnly]);

  const displayRows = useMemo(() => {
    if (!sortCol) return filtered;
    return [...filtered].sort((a, b) => {
      const av = a[sortCol], bv = b[sortCol];
      if (av == null) return 1;
      if (bv == null) return -1;
      const cmp =
        typeof av === "number" && typeof bv === "number"
          ? av - bv
          : String(av).localeCompare(String(bv), undefined, { numeric: true, sensitivity: "base" });
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [filtered, sortCol, sortDir]);

  // ── Row virtualisation ──────────────────────────────────────────────────
  // A 1000 x 125 sheet is 126,000 <td>s and ~256k DOM nodes; rendering them
  // all took ~5.8s before the grid became usable (the upload itself is 0.4s).
  // Rows are a uniform 33px, so a fixed-height window is exact. Small sheets
  // render in full so short datasets — and the existing tests — are untouched.
  const ROW_H = 33;
  const VIRTUALIZE_ABOVE = 150;
  const OVERSCAN = 12;

  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportH, setViewportH] = useState(0);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => setViewportH(el.clientHeight);
    measure();
    // ResizeObserver is absent in jsdom and older embedded webviews; a single
    // measurement is a fine fallback there.
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Coalesce scroll events into one state update per frame.
  const rafRef = useRef<number | null>(null);
  const onGridScroll = () => {
    const el = scrollRef.current;
    if (!el || rafRef.current != null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      setScrollTop(el.scrollTop);
    });
  };
  useEffect(() => () => { if (rafRef.current != null) cancelAnimationFrame(rafRef.current); }, []);

  const virtualized = displayRows.length > VIRTUALIZE_ABOVE;
  const { startIdx, endIdx } = useMemo(() => {
    if (!virtualized) return { startIdx: 0, endIdx: displayRows.length };
    const visible = Math.ceil((viewportH || 800) / ROW_H);
    const first = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
    return { startIdx: first, endIdx: Math.min(displayRows.length, first + visible + OVERSCAN * 2) };
  }, [virtualized, displayRows.length, viewportH, scrollTop]);

  const visibleRows = useMemo(
    () => (virtualized ? displayRows.slice(startIdx, endIdx) : displayRows),
    [virtualized, displayRows, startIdx, endIdx],
  );
  const padTop = virtualized ? startIdx * ROW_H : 0;
  const padBottom = virtualized ? (displayRows.length - endIdx) * ROW_H : 0;

  // A filter/sort change can leave the window scrolled past the new end.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && el.scrollTop > displayRows.length * ROW_H) el.scrollTop = 0;
  }, [displayRows.length]);

  useEffect(() => {
    if (renameCol) setTimeout(() => renameRef.current?.focus(), 0);
  }, [renameCol]);

  // Keep the bulk-delete selections from going stale. Row indices are
  // positional, so any change to the row COUNT (add/delete/undo) invalidates
  // them — drop the row ticks. Column ticks survive as long as the column
  // still exists, so prune to current columns rather than clearing outright.
  useEffect(() => { setCheckedRows(new Set()); }, [session?.session_id, preview.length]);
  useEffect(() => {
    setCheckedCols((prev) => {
      const valid = new Set(columns.map((c) => c.name));
      const next = new Set([...prev].filter((name) => valid.has(name)));
      return next.size === prev.size ? prev : next;
    });
  }, [session?.session_id, columns]);

  // Never reopen the column menu with a stale submenu expanded.
  useEffect(() => { setOpenSub(null); }, [ctxMenu?.col, ctxMenu?.x, ctxMenu?.y]);

  // Close context menus on outside click
  useEffect(() => {
    if (!ctxMenu && !rowCtx && !cellCtx) return;
    const handler = (e: MouseEvent) => {
      // Submenu flyouts are portalled to <body>, so they're outside ctxRef —
      // without this the menu would close on mousedown and the click would
      // never reach the flyout's button.
      if ((e.target as Element | null)?.closest?.("[data-colmenu-flyout]")) return;
      if (ctxMenu && ctxRef.current && !ctxRef.current.contains(e.target as Node)) setCtxMenu(null);
      if (rowCtx && rowCtxRef.current && !rowCtxRef.current.contains(e.target as Node)) setRowCtx(null);
      if (cellCtx && cellCtxRef.current && !cellCtxRef.current.contains(e.target as Node)) setCellCtx(null);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [ctxMenu, rowCtx, cellCtx]);

  // Bump undo depth after each backend mutation
  const bumpUndo = () => useStore.setState((s) => ({ undoDepth: s.undoDepth + 1, redoDepth: 0, dataVersion: s.dataVersion + 1 }));

  const deleteColumn = async (colName: string) => {
    if (!session) return;

    setCtxMenu(null);
    try {
      await api.delete(`/api/compute/${session.session_id}/column/${encodeURIComponent(colName)}`);
      const updatedCols = session.columns.filter((c) => c.name !== colName);
      const updatedPreview = session.preview.map((row) => {
        const r = { ...row }; delete r[colName]; return r;
      });
      useStore.getState().setSession({ ...session, columns: updatedCols, preview: updatedPreview }); bumpUndo();
    } catch { /* ignore */ }
  };

  const copyRow = (rowIdx: number) => {
    if (!session) return;
    setRowCtx(null);
    const row = preview[rowIdx];
    if (!row) return;
    const headers = columns.map((c) => c.name);
    const vals = headers.map((h) => String(row[h] ?? ""));
    const tsv = headers.join("\t") + "\n" + vals.join("\t");
    navigator.clipboard.writeText(tsv).catch(() => {});
  };

  const copyColumn = async (colName: string) => {
    if (!session) return;
    setCtxMenu(null);
    try {
      // Pull the WHOLE column from the backend: `preview` is capped at 2000
      // rows, so copying from it silently truncated bigger datasets.
      const res = await api.get(
        `/api/compute/${session.session_id}/column_values/${encodeURIComponent(colName)}`
      );
      const values = (res.data.values as unknown[]).map((v) => (v === null || v === undefined ? "" : String(v)));
      await navigator.clipboard.writeText(colName + "\n" + values.join("\n"));
      setPasteMsg(`Column "${colName}" copied (${values.length} values)`);
      setTimeout(() => setPasteMsg(null), 2500);
    } catch {
      // Previously this failed silently, so a denied clipboard looked like a
      // broken menu item.
      setPasteMsg("Copy failed — clipboard access was denied");
      setTimeout(() => setPasteMsg(null), 3500);
    }
  };

  /** Paste the clipboard as a NEW column (the paste side of "Copy column").
   *  Works across windows/sessions since the clipboard is the only channel. */
  const pasteColumn = async (afterCol?: string) => {
    if (!session) return;
    setCtxMenu(null);
    try {
      const text = await navigator.clipboard.readText();
      const body = text.replace(/\r\n?/g, "\n").replace(/\n+$/, "");
      if (!body.trim()) {
        setPasteMsg("Clipboard is empty");
        setTimeout(() => setPasteMsg(null), 3000);
        return;
      }
      const lines = body.split("\n");
      if (lines.some((l) => l.includes("\t"))) {
        setPasteMsg("Clipboard holds multiple columns — select a cell and press Ctrl/Cmd+V instead");
        setTimeout(() => setPasteMsg(null), 4000);
        return;
      }
      if (lines.length < 2) {
        setPasteMsg("Clipboard needs a header line plus values to paste as a column");
        setTimeout(() => setPasteMsg(null), 4000);
        return;
      }
      // "Copy column" always writes the name on the first line.
      const [rawName, ...values] = lines;
      let name = rawName.trim() || "PASTED";
      const existing = new Set(columns.map((c) => c.name));
      if (existing.has(name)) {
        let i = 2;
        while (existing.has(`${name}_${i}`)) i++;
        name = `${name}_${i}`;
      }
      const position = afterCol ? columns.findIndex((c) => c.name === afterCol) + 1 : -1;
      const res = await api.post(`/api/compute/${session.session_id}/paste_column`, { name, values, position });
      const refresh = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...refresh.data }); bumpUndo();
      const { n_truncated: truncated, n_padded: padded } = res.data;
      const note = truncated ? ` · ${truncated} extra value${truncated > 1 ? "s" : ""} dropped`
                 : padded    ? ` · ${padded} row${padded > 1 ? "s" : ""} left blank`
                 : "";
      setPasteMsg(`Column "${name}" pasted${note}`);
      setTimeout(() => setPasteMsg(null), 4000);
    } catch (err: unknown) {
      setPasteMsg((err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Paste column failed");
      setTimeout(() => setPasteMsg(null), 3500);
    }
  };

  const addRow = async (position: number) => {
    if (!session) return;
    setRowCtx(null);
    try {
      await api.post(`/api/compute/${session.session_id}/add_row`, { position });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch { /* ignore */ }
  };

  const addColumn = async (position?: number) => {
    if (!session) return;
    const name = prompt("New column name:");
    const trimmed = name?.trim();
    if (!trimmed) return;
    // Client-side duplicate guard so the user sees the conflict before the
    // round-trip; backend also validates as a safety net.
    if (session.columns.some((c) => c.name === trimmed)) {
      alert(`Column "${trimmed}" already exists. Pick a different name.`);
      return;
    }
    try {
      await api.post(`/api/compute/${session.session_id}/add_column`, { name: trimmed, position: position ?? -1 });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch (e: unknown) {
      alert((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Failed to add column");
    }
  };

  const deleteRow = async (rowIdx: number) => {
    if (!session) return;

    setRowCtx(null);
    try {
      await api.post(`/api/compute/${session.session_id}/delete_rows`, { row_indices: [rowIdx] });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch { /* ignore */ }
  };

  // ── Bulk tick-and-delete (row gutter + column header checkboxes) ────────────
  const toggleCheckedRow = (idx: number) => setCheckedRows((prev) => {
    const next = new Set(prev);
    if (next.has(idx)) next.delete(idx); else next.add(idx);
    return next;
  });
  const toggleCheckedCol = (name: string) => setCheckedCols((prev) => {
    const next = new Set(prev);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });
  const allRowsChecked = displayRows.length > 0 && displayRows.every((r) => checkedRows.has(r._idx as number));
  const toggleAllRows = () => setCheckedRows(() =>
    allRowsChecked ? new Set() : new Set(displayRows.map((r) => r._idx as number)));
  const allColsChecked = columns.length > 0 && columns.every((c) => checkedCols.has(c.name));
  const toggleAllCols = () => setCheckedCols(() =>
    allColsChecked ? new Set() : new Set(columns.map((c) => c.name)));
  const clearChecks = () => { setCheckedRows(new Set()); setCheckedCols(new Set()); };

  // Shift+click extends the tick from the last-clicked row/column (Excel-style
  // range selection). Anchors live in refs — they never need a re-render.
  const lastRowTickRef = useRef<number | null>(null);
  const lastColTickRef = useRef<string | null>(null);

  const tickRow = (origIdx: number, shiftKey: boolean) => {
    const anchor = lastRowTickRef.current;
    if (shiftKey && anchor !== null && anchor !== origIdx) {
      const ids = displayRows.map((r) => r._idx as number);
      const a = ids.indexOf(anchor);
      const b = ids.indexOf(origIdx);
      if (a >= 0 && b >= 0) {
        // The whole range takes the anchor's state, so shift-extending after an
        // untick clears the range rather than re-ticking it.
        const on = checkedRows.has(anchor);
        setCheckedRows((prev) => {
          const next = new Set(prev);
          for (let i = Math.min(a, b); i <= Math.max(a, b); i++) {
            if (on) next.add(ids[i]); else next.delete(ids[i]);
          }
          return next;
        });
        lastRowTickRef.current = origIdx;
        return;
      }
    }
    toggleCheckedRow(origIdx);
    lastRowTickRef.current = origIdx;
  };

  const tickCol = (name: string, shiftKey: boolean) => {
    const anchor = lastColTickRef.current;
    if (shiftKey && anchor !== null && anchor !== name) {
      const names = columns.map((c) => c.name);
      const a = names.indexOf(anchor);
      const b = names.indexOf(name);
      if (a >= 0 && b >= 0) {
        const on = checkedCols.has(anchor);
        setCheckedCols((prev) => {
          const next = new Set(prev);
          for (let i = Math.min(a, b); i <= Math.max(a, b); i++) {
            if (on) next.add(names[i]); else next.delete(names[i]);
          }
          return next;
        });
        lastColTickRef.current = name;
        return;
      }
    }
    toggleCheckedCol(name);
    lastColTickRef.current = name;
  };

  // Range spec from the row context menu: "100-500, 750, 900-950". Numbers are
  // the DISPLAYED gutter numbers (1-based), so an active filter/sort is
  // respected. Returns how many rows the spec matched (0 = nothing valid).
  const applyRowRangeSpec = (spec: string): number => {
    const n = displayRows.length;
    const picked: number[] = [];
    for (const raw of spec.split(/[,;]+/)) {
      const tok = raw.trim();
      if (!tok) continue;
      const m = /^(\d+)(?:\s*[-–:]\s*(\d+))?$/.exec(tok);
      if (!m) continue;
      let a = parseInt(m[1], 10);
      let b = m[2] ? parseInt(m[2], 10) : a;
      if (a > b) [a, b] = [b, a];
      for (let i = Math.max(1, a); i <= Math.min(n, b); i++) {
        picked.push(displayRows[i - 1]._idx as number);
      }
    }
    if (picked.length === 0) return 0;
    setCheckedRows((prev) => new Set([...prev, ...picked]));
    return picked.length;
  };

  // Tick every displayed row from the given display position downwards.
  const tickRowsFromHere = (fromVisualIdx: number) => {
    if (fromVisualIdx < 0) return;
    setCheckedRows((prev) => {
      const next = new Set(prev);
      for (let i = fromVisualIdx; i < displayRows.length; i++) {
        next.add(displayRows[i]._idx as number);
      }
      return next;
    });
  };

  // Invert the tick over the displayed rows (hidden rows drop out).
  const invertCheckedRows = () =>
    setCheckedRows((prev) => new Set(
      displayRows.map((r) => r._idx as number).filter((i) => !prev.has(i)),
    ));

  // Column counterparts — numbers are the 1-based column numbers shown in the
  // column-number row.
  const applyColRangeSpec = (spec: string): number => {
    const n = columns.length;
    const picked: string[] = [];
    for (const raw of spec.split(/[,;]+/)) {
      const tok = raw.trim();
      if (!tok) continue;
      const m = /^(\d+)(?:\s*[-–:]\s*(\d+))?$/.exec(tok);
      if (!m) continue;
      let a = parseInt(m[1], 10);
      let b = m[2] ? parseInt(m[2], 10) : a;
      if (a > b) [a, b] = [b, a];
      for (let i = Math.max(1, a); i <= Math.min(n, b); i++) {
        picked.push(columns[i - 1].name);
      }
    }
    if (picked.length === 0) return 0;
    setCheckedCols((prev) => new Set([...prev, ...picked]));
    return picked.length;
  };

  const tickColsFromHere = (fromIdx: number) => {
    if (fromIdx < 0) return;
    setCheckedCols((prev) => {
      const next = new Set(prev);
      for (let i = fromIdx; i < columns.length; i++) next.add(columns[i].name);
      return next;
    });
  };

  const invertCheckedCols = () =>
    setCheckedCols((prev) => new Set(
      columns.map((c) => c.name).filter((name) => !prev.has(name)),
    ));

  // Escape clears the tick selection (unless the user is typing in a field —
  // the cell editor's own Escape must win).
  useEffect(() => {
    if (checkedRows.size === 0 && checkedCols.size === 0) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      clearChecks();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [checkedRows.size, checkedCols.size]);

  const deleteChecked = async () => {
    if (!session) return;
    const rows = [...checkedRows];
    const cols = [...checkedCols];
    if (rows.length === 0 && cols.length === 0) return;
    if (cols.length >= columns.length) {
      alert("You can't delete every column.");
      return;
    }
    const parts: string[] = [];
    if (rows.length) parts.push(`${rows.length} row${rows.length > 1 ? "s" : ""}`);
    if (cols.length) parts.push(`${cols.length} column${cols.length > 1 ? "s" : ""}`);
    if (!window.confirm(`Delete ${parts.join(" and ")}? You can undo this with Ctrl/Cmd+Z.`)) return;
    try {
      // Rows first: delete_rows drops by position and resets the index; column
      // deletion is unaffected by that, so the order is safe.
      if (rows.length) {
        await api.post(`/api/compute/${session.session_id}/delete_rows`, { row_indices: rows });
      }
      if (cols.length) {
        await api.post(`/api/compute/${session.session_id}/delete_columns`, { columns: cols });
      }
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data });
      bumpUndo();
      clearChecks();
    } catch (e: unknown) {
      alert((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Bulk delete failed");
    }
  };

  // ── Cell selection helpers ──────────────────────────────────────────────────
  const cellKey = (row: number, col: string) => `${row}:${col}`;

  type CellPosition = { row: number; col: string };

  const visibleRowIds = () => displayRows.map((row) => row._idx as number);

  const rangeKeys = (anchor: CellPosition, focus: CellPosition): Set<string> => {
    const rowIds = visibleRowIds();
    const colNames = columns.map((c) => c.name);
    const r1 = rowIds.indexOf(anchor.row);
    const r2 = rowIds.indexOf(focus.row);
    const c1 = colNames.indexOf(anchor.col);
    const c2 = colNames.indexOf(focus.col);
    const next = new Set<string>();
    if (r1 < 0 || r2 < 0 || c1 < 0 || c2 < 0) return next;
    for (let r = Math.min(r1, r2); r <= Math.max(r1, r2); r++) {
      for (let c = Math.min(c1, c2); c <= Math.max(c1, c2); c++) {
        next.add(cellKey(rowIds[r], colNames[c]));
      }
    }
    return next;
  };

  const selectRange = (anchor: CellPosition, focus: CellPosition, additive = false) => {
    const range = rangeKeys(anchor, focus);
    setSelectedCells((prev) => additive ? new Set([...prev, ...range]) : range);
    setSelAnchor(anchor);
    setSelFocus(focus);
  };

  const selectSingleCell = (row: number, col: string) => {
    const pos = { row, col };
    setSelectedCells(new Set([cellKey(row, col)]));
    setSelAnchor(pos);
    setSelFocus(pos);
  };

  const beginCellSelection = (row: number, col: string, e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    gridRef.current?.focus({ preventScroll: true });
    const mod = e.ctrlKey || e.metaKey;
    const pos = { row, col };
    dragBaseSelectionRef.current = new Set(selectedCells);

    if (e.shiftKey) {
      const anchor = selAnchor ?? pos;
      selectRange(anchor, pos, mod && selAnchor !== null);
      dragAnchorRef.current = anchor;
      dragAdditiveRef.current = mod && selAnchor !== null;
    } else if (mod) {
      setSelectedCells((prev) => {
        const next = new Set(prev);
        const key = cellKey(row, col);
        if (next.has(key)) next.delete(key); else next.add(key);
        return next;
      });
      setSelAnchor(pos);
      setSelFocus(pos);
      dragAnchorRef.current = null;
      dragSelectingRef.current = false;
      return;
    } else {
      selectSingleCell(row, col);
      dragAnchorRef.current = pos;
      dragAdditiveRef.current = false;
    }
    dragSelectingRef.current = true;
  };

  const extendMouseSelection = (row: number, col: string, e: React.MouseEvent) => {
    if (!dragSelectingRef.current || e.buttons !== 1 || !dragAnchorRef.current) {
      if (e.buttons !== 1) dragSelectingRef.current = false;
      return;
    }
    const focus = { row, col };
    const range = rangeKeys(dragAnchorRef.current, focus);
    setSelectedCells(
      dragAdditiveRef.current
        ? new Set([...dragBaseSelectionRef.current, ...range])
        : range
    );
    setSelFocus(focus);
  };

  const selectVisibleRow = (row: number, additive: boolean) => {
    const keys = columns.map((c) => cellKey(row, c.name));
    setSelectedCells((prev) => additive ? new Set([...prev, ...keys]) : new Set(keys));
    const anchor = { row, col: columns[0]?.name ?? "" };
    const focus = { row, col: columns[columns.length - 1]?.name ?? "" };
    setSelAnchor(anchor);
    setSelFocus(focus);
    gridRef.current?.focus({ preventScroll: true });
  };

  const selectVisibleColumn = (col: string, additive: boolean) => {
    const rows = visibleRowIds();
    const keys = rows.map((row) => cellKey(row, col));
    setSelectedCells((prev) => additive ? new Set([...prev, ...keys]) : new Set(keys));
    if (rows.length > 0) {
      setSelAnchor({ row: rows[0], col });
      setSelFocus({ row: rows[rows.length - 1], col });
    }
    gridRef.current?.focus({ preventScroll: true });
  };

  const moveSelectionFocus = (rowDelta: number, colDelta: number, extend: boolean, toEdge: boolean) => {
    if (!selFocus || columns.length === 0 || displayRows.length === 0) return;
    const rows = visibleRowIds();
    const currentRow = Math.max(0, rows.indexOf(selFocus.row));
    const currentCol = Math.max(0, columns.findIndex((c) => c.name === selFocus.col));
    const nextRow = toEdge
      ? (rowDelta < 0 ? 0 : rowDelta > 0 ? rows.length - 1 : currentRow)
      : Math.max(0, Math.min(rows.length - 1, currentRow + rowDelta));
    const nextCol = toEdge
      ? (colDelta < 0 ? 0 : colDelta > 0 ? columns.length - 1 : currentCol)
      : Math.max(0, Math.min(columns.length - 1, currentCol + colDelta));
    const next = { row: rows[nextRow], col: columns[nextCol].name };
    if (extend) selectRange(selAnchor ?? selFocus, next);
    else selectSingleCell(next.row, next.col);
    requestAnimationFrame(() => {
      gridRef.current
        ?.querySelector<HTMLElement>(`[data-grid-row="${next.row}"][data-grid-col="${nextCol}"]`)
        ?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
  };

  const clearSelectedCells = async () => {
    if (!session || selectedCells.size === 0) return;
    const cells = Array.from(selectedCells).map((k) => {
      const [r, ...cParts] = k.split(":");
      return { row_index: Number(r), column: cParts.join(":") };
    });
    try {
      await api.post(`/api/sessions/${session.session_id}/clear_cells`, { cells });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
      setSelectedCells(new Set());
      setSelAnchor(null);
      setSelFocus(null);
    } catch { /* ignore */ }
  };

  // ── Clipboard for cell copy/paste ──────────────────────────────────────────
  const [copiedCells, setCopiedCells] = useState<{ tsv: string; rows: number; cols: number } | null>(null);

  const selectedCellsTsv = (): { tsv: string; rows: number; cols: number } | null => {
    if (!session || selectedCells.size === 0) return null;
    const cells = Array.from(selectedCells).map((k) => {
      const [r, ...cParts] = k.split(":");
      return { row: Number(r), col: cParts.join(":") };
    });
    const visibleOrder = visibleRowIds();
    const rows = [...new Set(cells.map((c) => c.row))].sort(
      (a, b) => visibleOrder.indexOf(a) - visibleOrder.indexOf(b)
    );
    const cols = [...new Set(cells.map((c) => c.col))];
    const colOrder = columns.map((c) => c.name);
    cols.sort((a, b) => colOrder.indexOf(a) - colOrder.indexOf(b));
    const tsv = rows.map((r) =>
      cols.map((c) => {
        if (!selectedCells.has(cellKey(r, c))) return "";
        const val = preview[r]?.[c];
        return val === null || val === undefined ? "" : String(val);
      }).join("\t")
    ).join("\n");
    return { tsv, rows: rows.length, cols: cols.length };
  };

  const copyCells = async (): Promise<boolean> => {
    const copied = selectedCellsTsv();
    if (!copied) return false;
    try {
      await navigator.clipboard.writeText(copied.tsv);
      setCopiedCells(copied);
      setPasteMsg(`${copied.rows}×${copied.cols} cells copied`);
      setTimeout(() => setPasteMsg(null), 2500);
      return true;
    } catch {
      setPasteMsg("Clipboard access was denied");
      setTimeout(() => setPasteMsg(null), 3000);
      return false;
    }
  };

  const cutCells = async () => {
    if (await copyCells()) {
      await clearSelectedCells();
      setPasteMsg("Cells cut to clipboard");
      setTimeout(() => setPasteMsg(null), 2500);
    }
  };

  const pasteCellsAt = async (startRow: number, startCol: string, tsv: string) => {
    if (!session) return;
    try {
      const rowOrder = visibleRowIds();
      const rowPos = rowOrder.indexOf(startRow);
      const colPos = columns.findIndex((c) => c.name === startCol);
      const res = await api.post(`/api/compute/${session.session_id}/paste_cells`, {
        start_row: startRow,
        start_col: startCol,
        row_indices: rowPos >= 0 ? rowOrder.slice(rowPos) : undefined,
        target_columns: colPos >= 0 ? columns.slice(colPos).map((c) => c.name) : undefined,
        tsv,
      });
      const refresh = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...refresh.data }); bumpUndo();
      setPasteMsg(`${res.data.pasted} cells pasted`);
      setTimeout(() => setPasteMsg(null), 2500);
    } catch (err: unknown) {
      setPasteMsg((err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Paste failed");
      setTimeout(() => setPasteMsg(null), 3500);
    }
  };

  const duplicateColumn = async (colName: string) => {
    if (!session) return;
    setCtxMenu(null);
    try {
      await api.post(`/api/compute/${session.session_id}/duplicate_column`, { column: colName });
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch { /* ignore */ }
  };

  const sendToEnd = (colName: string) => {
    if (!session) return;

    setCtxMenu(null);
    const idx = session.columns.findIndex((c) => c.name === colName);
    if (idx < 0 || idx === session.columns.length - 1) return;
    reorderColumns(idx, session.columns.length - 1);
  };

  // Move a column to an explicit 1-based position (shifts the rest along).
  const moveToPosition = (colName: string, oneBased: number) => {
    if (!session) return;
    const idx = session.columns.findIndex((c) => c.name === colName);
    if (idx < 0) return;
    const target = Math.max(0, Math.min(oneBased - 1, session.columns.length - 1));
    if (target !== idx) reorderColumns(idx, target);
    setMoveCol(null);
  };

  // Open the bulk rename modal seeded with Sentence-case suggestions for EVERY
  // column (editable, so the user can also rename acronyms the auto-suggester
  // intentionally leaves untouched). Columns with a suggestion are pre-ticked.
  const openSuggestNames = async () => {
    if (!session) return;
    setCtxMenu(null);
    setSuggestBusy(true);
    try {
      const { getNameSuggestions } = await import("../api");
      const res = await getNameSuggestions(session.session_id);
      const s: Record<string, string> = res.data?.suggestions ?? {};
      const draft: Record<string, string> = {};
      const acc: Record<string, boolean> = {};
      for (const c of session.columns) {
        draft[c.name] = s[c.name] ?? c.name;
        acc[c.name] = c.name in s;  // pre-tick only the ones we actually changed
      }
      setSuggestDraft(draft);
      setSuggestAccept(acc);
      setSuggestOpen(true);
    } catch { /* ignore */ } finally { setSuggestBusy(false); }
  };

  // Apply ticked rows whose target differs from the current name, then refresh.
  const applySuggestions = async () => {
    if (!session) return;
    const pairs = session.columns
      .map((c) => c.name)
      .filter((n) => suggestAccept[n] && suggestDraft[n]?.trim() && suggestDraft[n].trim() !== n)
      .map((n) => [n, suggestDraft[n].trim()] as [string, string]);
    if (pairs.length === 0) { setSuggestOpen(false); return; }
    setSuggestBusy(true);
    try {
      const { renameColumn } = await import("../api");
      for (const [oldName, newName] of pairs) {
        try { await renameColumn(session.session_id, oldName, newName); } catch { /* skip dup/invalid */ }
      }
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      const cur = useStore.getState().session;
      if (cur) { useStore.getState().setSession({ ...cur, ...res.data }); bumpUndo(); }
    } catch { /* ignore */ } finally { setSuggestBusy(false); setSuggestOpen(false); }
  };

  const fillBlanks = async (colName: string, fillValue: string) => {
    if (!session || !fillValue.trim()) return;

    setCtxMenu(null);
    try {
      await api.post(`/api/compute/${session.session_id}/fill_blanks`, {
        column: colName, value: fillValue.trim(),
      });
      // Refresh preview
      const res = await api.get(`/api/stats/${session.session_id}/refresh`);
      useStore.getState().setSession({ ...session, ...res.data }); bumpUndo();
    } catch (e: unknown) {
      // Surface the failure — previously swallowed, so MICE/fill errors looked
      // like nothing happened.
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      alert(detail ?? "Could not fill blanks for this column.");
    }
  };

  const startRename = (colName: string) => {
    setRenameCol(colName);
    setRenameVal(colName);
  };

  const commitRename = async () => {
    if (!renameCol || !session) return;
    const oldName = renameCol;  // capture before clearing state
    const newName = renameVal.trim();
    if (!newName || newName === oldName) {
      setRenameCol(null);
      return;
    }

    // Client-side duplicate-name guard. Keeps editor open so the user can
    // adjust the name instead of losing the input on a silent revert.
    const existingNames = new Set(session.columns.map((c) => c.name));
    if (existingNames.has(newName)) {
      alert(`Column "${newName}" already exists. Pick a different name.`);
      // Keep the input open with the rejected name selected for quick re-edit.
      setTimeout(() => renameRef.current?.select(), 0);
      return;
    }

    setRenameCol(null);
    try {
      await renameColumn(session.session_id, oldName, newName);
      // Update local state
      const updatedCols = session.columns.map((c) =>
        c.name === oldName ? { ...c, name: newName } : c
      );
      const updatedPreview = session.preview.map((row) => {
        const r = { ...row };
        if (oldName in r) { r[newName] = r[oldName]; delete r[oldName]; }
        return r;
      });
      // Remap per-column decimal formatting so the rename carries the user's
      // formatting choice over to the new column name.
      if (oldName in columnDecimals) {
        const next: Record<string, number> = { ...columnDecimals };
        next[newName] = next[oldName];
        delete next[oldName];
        useStore.setState({ columnDecimals: next });
      }
      useStore.getState().setSession({ ...session, columns: updatedCols, preview: updatedPreview });
      // Every other panel's persisted variable/covariate selection still
      // points at the old name (they're plain cached strings, not live
      // references) — remap them so a rename doesn't silently break the
      // next analysis run in a panel the user isn't currently looking at.
      useStore.getState().renameInPanelCache(oldName, newName);
      bumpUndo();
    } catch (e: unknown) {
      // Surface backend errors (422 duplicate, network, etc.) instead of
      // silently dropping the rename. Falls back to a generic message.
      const detail =
        (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? (e instanceof Error ? e.message : String(e));
      alert(`Rename failed: ${detail}`);
    }
  };

  const toggleSort = (colName: string) => {
    if (sortCol === colName) {
      if (sortDir === "asc") setSortDir("desc");
      else setSortCol(null);
    } else {
      setSortCol(colName);
      setSortDir("asc");
    }
  };

  const cycleKind = (colName: string) => {
    const cur = columns.find((c) => c.name === colName)?.kind ?? "numeric";
    const next = KIND_CYCLE[(KIND_CYCLE.indexOf(cur) + 1) % KIND_CYCLE.length];
    updateColumnKind(colName, next);
  };

  const startEdit = (rowIdx: number, col: string, initialValue?: string) => {
    const val = preview[rowIdx]?.[col];
    selectSingleCell(rowIdx, col);
    setEditCell({ rowIdx, col });
    setEditValue(initialValue ?? (val === null || val === undefined ? "" : String(val)));
  };

  const commitEdit = async (restoreGridFocus = false) => {
    if (!editCell) return;

    const { rowIdx, col } = editCell;
    const commitKey = cellKey(rowIdx, col);
    if (committingCellsRef.current.has(commitKey)) return;
    committingCellsRef.current.add(commitKey);
    setEditCell(null);
    if (restoreGridFocus) {
      requestAnimationFrame(() => gridRef.current?.focus({ preventScroll: true }));
    }

    const original = preview[rowIdx]?.[col];
    const rawVal   = editValue.trim();
    const newVal   = rawVal === "" ? null : rawVal;

    if (String(original ?? "") === String(newVal ?? "")) {
      committingCellsRef.current.delete(commitKey);
      return;
    }

    const colKind = columns.find((c) => c.name === col)?.kind;
    const parsedNumber = rawVal === "" ? null : Number(rawVal);
    const optimisticValue =
      colKind === "numeric" && parsedNumber !== null && Number.isFinite(parsedNumber)
        ? parsedNumber
        : newVal;

    // Show the edit immediately; the backend response below normalizes the
    // value to the column dtype. Revert only when persistence fails.
    updatePreviewCell(rowIdx, col, optimisticValue);
    setSaving(true);
    try {
      const res = await api.patch(`/api/sessions/${session.session_id}/cell`, {
        row_index: rowIdx,
        column: col,
        value: newVal,
      });
      updatePreviewCell(rowIdx, col, res.data.value);
      bumpUndo();
    } catch {
      updatePreviewCell(rowIdx, col, original);
      setPasteMsg("Cell update failed; the previous value was restored");
      setTimeout(() => setPasteMsg(null), 3500);
    } finally {
      committingCellsRef.current.delete(commitKey);
      setSaving(false);
    }
  };

  const handleGridKeyDown = async (e: React.KeyboardEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    if (
      editCell || renameCol ||
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target instanceof HTMLButtonElement ||
      target instanceof HTMLAnchorElement ||
      target.isContentEditable
    ) return;

    const mod = e.metaKey || e.ctrlKey;

    if (mod && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (e.shiftKey) redo(); else undo();
      return;
    }
    if (mod && e.key.toLowerCase() === "y") {
      e.preventDefault();
      redo();
      return;
    }
    if (mod && e.key.toLowerCase() === "a") {
      e.preventDefault();
      const all = new Set<string>();
      for (const row of visibleRowIds()) {
        for (const col of columns) all.add(cellKey(row, col.name));
      }
      setSelectedCells(all);
      if (displayRows.length && columns.length) {
        setSelAnchor({ row: displayRows[0]._idx as number, col: columns[0].name });
        setSelFocus({
          row: displayRows[displayRows.length - 1]._idx as number,
          col: columns[columns.length - 1].name,
        });
      }
      return;
    }
    if (mod && e.key.toLowerCase() === "c" && selectedCells.size > 0) {
      e.preventDefault();
      await copyCells();
      return;
    }
    if (mod && e.key.toLowerCase() === "x" && selectedCells.size > 0) {
      e.preventDefault();
      await cutCells();
      return;
    }
    if (mod && e.key.toLowerCase() === "v" && session) {
      e.preventDefault();
      try {
        const text = await navigator.clipboard.readText();
        if (!text.trim()) return;
        const destination = selAnchor ?? selFocus;
        if (destination) {
          await pasteCellsAt(destination.row, destination.col, text);
          return;
        }
        // A single-column clipboard (no tabs) with no cell selected is almost
        // always a "Copy column" payload. Appending it as ROWS — which is what
        // this used to fall through to — tacked a pile of blank rows onto the
        // bottom of the dataset. Treat it as a column paste instead; only a
        // genuine multi-column grid still appends rows.
        const singleColumn = !text.includes("\t")
          && text.replace(/\r\n?/g, "\n").replace(/\n+$/, "").split("\n").length > 1;
        if (singleColumn) {
          await pasteColumn();
          return;
        }
        const res = await api.post(`/api/compute/${session.session_id}/paste`, {
          tsv: text, has_header: true, mode: "append",
        });
        const refresh = await api.get(`/api/stats/${session.session_id}/refresh`);
        useStore.getState().setSession({ ...session, ...refresh.data }); bumpUndo();
        setPasteMsg(`${res.data.n_pasted} rows pasted`);
        setTimeout(() => setPasteMsg(null), 3000);
      } catch (err: unknown) {
        setPasteMsg((err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Paste failed");
        setTimeout(() => setPasteMsg(null), 4000);
      }
      return;
    }

    if ((e.key === "Delete" || e.key === "Backspace") && selectedCells.size > 0) {
      e.preventDefault();
      await clearSelectedCells();
      return;
    }
    if (e.key === "Escape" && selectedCells.size > 0) {
      e.preventDefault();
      setSelectedCells(new Set());
      setSelAnchor(null);
      setSelFocus(null);
      return;
    }

    const directions: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    // Seed a focus cell the first time an arrow / Tab / Enter is pressed while
    // the grid has keyboard focus but nothing is selected yet — otherwise the
    // whole navigation scheme silently does nothing until the user clicks a cell.
    const ensureSeedFocus = (): { row: number; col: string } | null => {
      if (selFocus) return selFocus;
      const rows = visibleRowIds();
      if (rows.length === 0 || columns.length === 0) return null;
      const seed = { row: rows[0], col: columns[0].name };
      selectSingleCell(seed.row, seed.col);
      return seed;
    };

    if (e.key in directions) {
      e.preventDefault();
      if (!ensureSeedFocus()) return;
      const [rowDelta, colDelta] = directions[e.key];
      moveSelectionFocus(rowDelta, colDelta, e.shiftKey, mod);
      return;
    }
    if (e.key === "Tab") {
      e.preventDefault();
      if (!ensureSeedFocus()) return;
      moveSelectionFocus(0, e.shiftKey ? -1 : 1, false, false);
      return;
    }
    if (e.key === "Enter" || e.key === "F2") {
      e.preventDefault();
      const focus = ensureSeedFocus();
      if (focus) startEdit(focus.row, focus.col);
      return;
    }
    if (!mod && !e.altKey && e.key.length === 1) {
      e.preventDefault();
      const focus = ensureSeedFocus();
      if (focus) startEdit(focus.row, focus.col, e.key);
    }
  };

  const activeFilters = Object.values(filters).filter(Boolean).length;

  return (
    <div
      ref={gridRef}
      tabIndex={0}
      onKeyDown={handleGridKeyDown}
      onMouseUp={() => { dragSelectingRef.current = false; }}
      className="relative flex flex-col gap-2 h-full focus:outline-none"
      style={{ minHeight: 0 }}
    >
      {showSelectCases && session && (
        <SelectCasesModal
          columns={columns}
          sessionId={session.session_id}
          existing={caseFilter?.conditions ?? []}
          onApply={(conditions, selected, total) => {
            setCaseFilter({ conditions, selected, total });
            setShowSelectCases(false);
          }}
          onClear={() => {
            setCaseFilter(null);
            setShowSelectCases(false);
          }}
          onClose={() => setShowSelectCases(false)}
        />
      )}

      {/* ── Toolbar ── */}
      <div className="flex items-center justify-between flex-shrink-0">
        <p className="text-sm text-gray-500">
          Showing{" "}
          <span className="text-gray-900 font-medium">{displayRows.length}</span>
          {displayRows.length !== preview.length && (
            <span className="text-gray-400"> of {preview.length} previewed</span>
          )}{" "}rows ·{" "}
          <span className="text-gray-900 font-medium">{session.rows.toLocaleString()}</span> total
          {" "}· {columns.length} columns
          {saving && <span className="ml-3 text-indigo-500 text-xs animate-pulse">saving…</span>}
          {pasteMsg && <span className="ml-3 text-emerald-600 text-xs">{pasteMsg}</span>}
          {selectedCells.size > 1 && (
            <span className="ml-3 text-blue-600 text-xs font-medium">
              {selectedCells.size} cells selected
              <button onClick={() => setSelectedCells(new Set())} className="ml-1 text-blue-400 hover:text-blue-600">✕</button>
            </span>
          )}
          {copiedCells && (
            <span className="ml-2 text-green-600 text-xs">
              {copiedCells.rows}x{copiedCells.cols} copied
            </span>
          )}
        </p>

        <div className="flex items-center gap-2">
          {/* Dictionary modal opener — moved from the Compute combo so the
              variable-metadata view sits next to the data grid it describes. */}
          <button
            onClick={() => setShowDictionary(true)}
            title="Edit variable labels, value labels, and column metadata"
            className="text-xs px-2 py-1 rounded-lg border border-indigo-300 text-indigo-600 hover:bg-indigo-50 transition-colors flex items-center gap-1"
          >
            <BookOpen size={12} /> Dictionary
          </button>

          <div className="w-px h-5 bg-gray-200" />

          {/* Add Row / Add Column */}
          <button onClick={() => addRow(-1)}
            className="text-xs px-2 py-1 rounded-lg border border-emerald-300 text-emerald-600 hover:bg-emerald-50 transition-colors">
            + Row
          </button>
          <button onClick={() => addColumn()}
            className="text-xs px-2 py-1 rounded-lg border border-emerald-300 text-emerald-600 hover:bg-emerald-50 transition-colors">
            + Column
          </button>

          <div className="w-px h-5 bg-gray-200" />

          {/* Undo / Redo */}
          <button onClick={undo} disabled={undoLen === 0}
            title="Undo (Ctrl+Z)"
            className={`text-xs px-2 py-1 rounded-lg border transition-colors ${undoLen > 0 ? "text-gray-600 border-gray-300 hover:bg-gray-100" : "text-gray-300 border-gray-200 cursor-default"}`}>
            ↩ Undo
          </button>
          <button onClick={redo} disabled={redoLen === 0}
            title="Redo (Ctrl+Y)"
            className={`text-xs px-2 py-1 rounded-lg border transition-colors ${redoLen > 0 ? "text-gray-600 border-gray-300 hover:bg-gray-100" : "text-gray-300 border-gray-200 cursor-default"}`}>
            ↪ Redo
          </button>

          <div className="w-px h-5 bg-gray-200" />

          {/* Freeze (pin-left) columns */}
          <div className="flex items-center gap-0.5 text-xs">
            <span className="text-gray-500 mr-1" title="Freeze leading columns so they stay visible while scrolling right">❄ Freeze</span>
            <button
              onClick={() => setFrozenCount((n) => Math.max(0, n - 1))}
              disabled={frozenCount === 0}
              title="Freeze one fewer column"
              className={`w-6 h-6 rounded border transition-colors flex items-center justify-center ${frozenCount > 0 ? "text-gray-600 border-gray-300 hover:bg-gray-100" : "text-gray-300 border-gray-200 cursor-default"}`}
            >−</button>
            <span className="w-6 text-center font-medium text-gray-700">{frozenCount}</span>
            <button
              onClick={() => setFrozenCount((n) => Math.min(columns.length, n + 1))}
              disabled={frozenCount >= columns.length}
              title="Freeze one more column"
              className={`w-6 h-6 rounded border transition-colors flex items-center justify-center ${frozenCount < columns.length ? "text-gray-600 border-gray-300 hover:bg-gray-100" : "text-gray-300 border-gray-200 cursor-default"}`}
            >+</button>
            {frozenCount > 0 && (
              <button
                onClick={() => setFrozenCount(0)}
                title="Unfreeze all"
                className="ml-1 text-[10px] text-orange-600 hover:text-orange-700 border border-orange-300 rounded px-1.5 py-0.5"
              >✕</button>
            )}
          </div>

          {sortCol && (
            <button
              onClick={() => setSortCol(null)}
              className="text-xs text-orange-600 hover:text-orange-700 border border-orange-300 rounded-lg px-2.5 py-1 transition-colors bg-orange-50"
            >
              ✕ Sort: {sortCol} {sortDir === "asc" ? "▲" : "▼"}
            </button>
          )}
          {activeFilters > 0 && (
            <button
              onClick={() => setFilters({})}
              className="text-xs text-gray-600 hover:text-gray-800 border border-gray-300 rounded-lg px-2.5 py-1 transition-colors"
            >
              ✕ Clear {activeFilters} filter{activeFilters > 1 ? "s" : ""}
            </button>
          )}

          {/* ── Missing value button — always visible, fixed position before Filter ── */}
          <button
            onClick={() => totalMissingRows > 0 && setShowMissingOnly((v) => !v)}
            title={totalMissingRows > 0
              ? `${totalMissingRows} rows have missing values — click to show only those rows`
              : "No missing values in this dataset"}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors
              ${totalMissingRows === 0
                ? "text-gray-300 border-gray-200 cursor-default"
                : showMissingOnly
                  ? "bg-amber-100 text-amber-700 border-amber-400"
                  : "text-amber-600 border-amber-300 bg-amber-50 hover:bg-amber-100"}`}
          >
            ⚠ Missing
            {totalMissingRows > 0 && (
              <span className={`text-[9px] font-bold rounded-full px-1.5 py-0.5
                ${showMissingOnly ? "bg-amber-600 text-white" : "bg-amber-200 text-amber-800"}`}>
                {totalMissingRows}
              </span>
            )}
          </button>

          {/* Select Cases */}
          <button
            onClick={() => setShowSelectCases(true)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors
              ${caseFilter
                ? "bg-violet-100 text-violet-700 border-violet-400"
                : "text-gray-500 border-gray-300 hover:text-gray-700 hover:border-gray-400"}`}
          >
            ⊂ Select Cases
            {caseFilter && (
              <span className="bg-violet-600 text-white text-[9px] font-bold rounded-full px-1.5 py-0.5">
                {caseFilter.selected.toLocaleString()}
              </span>
            )}
          </button>

          <button
            onClick={() => setShowFilters((v) => !v)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors
              ${showFilters || activeFilters > 0
                ? "bg-indigo-50 text-indigo-600 border-indigo-300"
                : "text-gray-500 border-gray-300 hover:text-gray-700 hover:border-gray-400"}`}
          >
            ⟁ Filter
            {activeFilters > 0 && (
              <span className="bg-indigo-600 text-white text-[9px] font-bold rounded-full w-4 h-4 flex items-center justify-center">
                {activeFilters}
              </span>
            )}
          </button>

        </div>
      </div>

      {/* ── Table ── */}
      <div ref={scrollRef} onScroll={onGridScroll}
        className="overflow-auto rounded-xl border border-gray-200 flex-1" style={{ minHeight: 0 }}>
        <table className="w-full text-sm border-collapse">
          <thead className="sticky top-0 z-10">

            {/* Column-number row (1, 2, 3 … above each column) — also carries the
                per-column bulk-select checkboxes + the "select all columns" master. */}
            <tr className="bg-gray-50 border-b border-gray-100">
              <th
                className="py-0 text-center border-r border-gray-200 select-none sticky left-0 bg-gray-50 z-20"
                style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
              >
                <span className="inline-flex h-4 items-center justify-center">
                  <input
                    type="checkbox"
                    className={`h-3 w-3 accent-indigo-500 cursor-pointer transition-opacity ${checkedCols.size > 0 ? "opacity-100" : "opacity-30 hover:opacity-100"}`}
                    checked={allColsChecked}
                    ref={(el) => { if (el) el.indeterminate = checkedCols.size > 0 && !allColsChecked; }}
                    onChange={toggleAllCols}
                    title="Select / clear all columns"
                  />
                </span>
              </th>
              {columns.map((col, colIdx) => {
                const frozen = isFrozenCol(colIdx);
                const isChecked = checkedCols.has(col.name);
                return (
                  <th
                    key={col.name}
                    className={`group py-0 text-center text-gray-300 text-[9px] font-normal border-r border-gray-200 select-none cursor-pointer ${isChecked ? "bg-indigo-100/80" : ""} ${frozen ? "sticky z-20" : ""} ${frozen && !isChecked ? "bg-gray-50" : ""}`}
                    style={frozen ? { left: frozenLeft(colIdx), width: FROZEN_COL_W, minWidth: FROZEN_COL_W, maxWidth: FROZEN_COL_W } : undefined}
                    onClick={(e) => tickCol(col.name, e.shiftKey)}
                    onMouseDown={(e) => { if (e.shiftKey) e.preventDefault(); }}
                    onContextMenu={(e) => { e.preventDefault(); setColRangeInput(""); setCtxMenu({ x: e.clientX, y: e.clientY, col: col.name }); }}
                    title={`Click to select column "${col.name}" · Shift+click extends the range · Right-click for range select`}
                  >
                    {/* Fixed-height swap: the number shows at rest, the checkbox on
                        hover or when ticked — the row never changes height. */}
                    <span className="inline-flex h-4 items-center justify-center">
                      <input
                        type="checkbox"
                        tabIndex={-1}
                        className={`h-3 w-3 accent-indigo-500 cursor-pointer pointer-events-none ${isChecked ? "inline-block" : "hidden group-hover:inline-block"}`}
                        checked={isChecked}
                        readOnly
                      />
                      <span className={isChecked ? "hidden" : "group-hover:hidden"}>{colIdx + 1}</span>
                    </span>
                  </th>
                );
              })}
            </tr>

            {/* Column headers */}
            <tr className="bg-gray-50 border-b border-gray-200">
              <th
                className="px-1 py-2 text-center text-gray-400 text-xs font-normal border-r border-gray-200 select-none sticky left-0 bg-gray-50 z-20"
                style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
                title="Select / clear all visible rows"
              >
                <input
                  type="checkbox"
                  className={`h-3 w-3 align-middle accent-indigo-500 cursor-pointer transition-opacity ${checkedRows.size > 0 ? "opacity-100" : "opacity-30 hover:opacity-100"}`}
                  checked={allRowsChecked}
                  ref={(el) => { if (el) el.indeterminate = checkedRows.size > 0 && !allRowsChecked; }}
                  onChange={toggleAllRows}
                />
              </th>
              {columns.map((col, colIdx) => {
                const isSorted = sortCol === col.name;
                const nMissing = missingCounts[col.name] ?? 0;
                const isDragOver = dropIdx === colIdx && dragIdx !== colIdx;
                const frozen = isFrozenCol(colIdx);
                const draggable = !frozen && renameCol !== col.name;
                return (
                  <th
                    key={col.name}
                    draggable={draggable}
                    onMouseDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                        e.preventDefault();
                        e.stopPropagation();
                        selectVisibleColumn(col.name, true);
                      }
                    }}
                    onClickCapture={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                        e.preventDefault();
                        e.stopPropagation();
                      }
                    }}
                    onDragStart={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                        e.preventDefault();
                        return;
                      }
                      if (!draggable) { e.preventDefault(); return; }
                      setDragIdx(colIdx);
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", String(colIdx));
                    }}
                    onDragOver={(e) => {
                      // Block dropping unfrozen → frozen region and vice versa
                      if (dragIdx === null) return;
                      const srcFrozen = isFrozenCol(dragIdx);
                      if (srcFrozen !== frozen) return;
                      e.preventDefault();
                      e.dataTransfer.dropEffect = "move";
                      setDropIdx(colIdx);
                    }}
                    onDragLeave={() => { if (dropIdx === colIdx) setDropIdx(null); }}
                    onDrop={(e) => {
                      e.preventDefault();
                      if (dragIdx !== null && dragIdx !== colIdx && isFrozenCol(dragIdx) === frozen) {
                        reorderColumns(dragIdx, colIdx);
                      }
                      setDragIdx(null);
                      setDropIdx(null);
                    }}
                    onDragEnd={() => { setDragIdx(null); setDropIdx(null); }}
                    onContextMenu={(e) => { e.preventDefault(); setColRangeInput(""); setCtxMenu({ x: e.clientX, y: e.clientY, col: col.name }); }}
                    className={`px-2 py-2 border-r border-gray-200
                      ${frozen ? "sticky bg-gray-50 z-20" : "min-w-[130px] max-w-[200px]"}
                      ${!frozen && checkedCols.has(col.name) ? "bg-indigo-100/60" : ""}
                      ${renameCol === col.name || frozen ? "" : "cursor-grab active:cursor-grabbing select-none"}
                      ${dragIdx === colIdx ? "opacity-40" : ""}
                      ${isDragOver ? "border-l-2 border-l-indigo-500" : ""}`}
                    style={frozen ? { left: frozenLeft(colIdx), width: FROZEN_COL_W, minWidth: FROZEN_COL_W, maxWidth: FROZEN_COL_W } : undefined}
                    title="Ctrl/Cmd+Shift+click selects the visible column"
                  >
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center gap-1 justify-between">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className="text-gray-300 text-[8px] flex-shrink-0 cursor-grab" title="Drag to reorder">⠿</span>
                          <button
                            onClick={() => cycleKind(col.name)}
                            title={`Type: ${col.kind} — click to change`}
                            className={`text-[9px] font-bold px-1.5 py-0.5 rounded border flex-shrink-0 transition-colors ${KIND_STYLE[col.kind] ?? KIND_STYLE.text}`}
                          >
                            {KIND_LABEL[col.kind] ?? col.kind}
                          </button>
                          {renameCol === col.name ? (
                            <input ref={renameRef}
                              className="text-xs font-medium text-gray-900 bg-white border border-indigo-400 rounded px-1 py-0 w-24 focus:outline-none select-text"
                              value={renameVal}
                              onClick={(e) => e.stopPropagation()}
                              onMouseDown={(e) => e.stopPropagation()}
                              onChange={(e) => setRenameVal(e.target.value)}
                              onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenameCol(null); }}
                              onBlur={commitRename}
                            />
                          ) : (
                            <span
                              className={`text-left text-xs font-medium truncate cursor-text ${
                                col.analysis_excluded
                                  ? "text-gray-400 line-through"
                                  : /^Column_\d+$/.test(col.name)
                                  ? "text-gray-400 italic"
                                  : "text-gray-700"
                              }`}
                              onDoubleClick={() => startRename(col.name)}
                              title={col.analysis_excluded ? "Excluded from analysis · double-click to rename" : "Double-click to rename"}>
                              {col.name}
                            </span>
                          )}
                          {col.analysis_excluded && (
                            <span className="flex-shrink-0 text-[8px] font-bold px-1 py-0.5 rounded bg-violet-100 text-violet-600 border border-violet-300"
                              title="Excluded from analysis (kept in the dataset)">excl</span>
                          )}
                        </div>
                        <button
                          onClick={() => toggleSort(col.name)}
                          title="Sort"
                          className={`flex-shrink-0 text-xs w-5 h-5 rounded flex items-center justify-center transition-colors
                            ${isSorted
                              ? "text-indigo-600 bg-indigo-100"
                              : "text-gray-300 hover:text-gray-500 hover:bg-gray-100"}`}
                        >
                          {isSorted ? (sortDir === "asc" ? "▲" : "▼") : "⇅"}
                        </button>
                      </div>
                      {nMissing > 0 && (() => {
                        const pct = preview.length ? (nMissing / preview.length) * 100 : 0;
                        const pctLabel = pct >= 10 ? pct.toFixed(0) : pct.toFixed(1);
                        return (
                          <div className="flex justify-start">
                            <button
                              onClick={() => {
                                setShowMissingOnly(true);
                                setFilters((prev) => ({ ...prev, [col.name]: "" }));
                              }}
                              title={`${nMissing} missing values (${pctLabel}% of ${preview.length} rows) — click to filter`}
                              className="flex-shrink-0 text-[8px] font-semibold px-1 py-0.5 rounded bg-amber-100 text-amber-700 border border-amber-300 hover:bg-amber-200 transition-colors"
                            >
                              {nMissing}✕ · {pctLabel}%
                            </button>
                          </div>
                        );
                      })()}
                    </div>
                  </th>
                );
              })}
            </tr>

            {/* Filter row */}
            {showFilters && (
              <tr className="bg-gray-50 border-b border-gray-200">
                <td
                  className="border-r border-gray-200 sticky left-0 bg-gray-50 z-20"
                  style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
                />
                {columns.map((col, colIdx) => {
                  const frozen = isFrozenCol(colIdx);
                  return (
                    <td
                      key={col.name}
                      className={`px-1.5 py-1 border-r border-gray-200 ${frozen ? "sticky bg-gray-50 z-20" : ""}`}
                      style={frozen ? { left: frozenLeft(colIdx), width: FROZEN_COL_W, minWidth: FROZEN_COL_W, maxWidth: FROZEN_COL_W } : undefined}
                    >
                      <input
                        className="w-full bg-white border border-gray-300 rounded px-2 py-0.5 text-xs text-gray-700
                          placeholder-gray-300 focus:outline-none focus:border-indigo-400"
                        placeholder="filter…"
                        value={filters[col.name] ?? ""}
                        onChange={(e) =>
                          setFilters((prev) => ({ ...prev, [col.name]: e.target.value }))
                        }
                      />
                    </td>
                  );
                })}
              </tr>
            )}
          </thead>

          <tbody>
            {padTop > 0 && (
              <tr aria-hidden="true"><td colSpan={columns.length + 1} style={{ height: padTop, padding: 0, border: 0 }} /></tr>
            )}
            {visibleRows.map((row, windowIdx) => {
              const visualIdx = startIdx + windowIdx;
              const origIdx = row._idx as number;
              const rowChecked = checkedRows.has(origIdx);
              return (
                <tr
                  key={origIdx}
                  className="group border-t border-gray-100 hover:bg-gray-50 transition-colors"
                >
                  <td
                    className={`px-1 py-1.5 text-gray-300 text-[11px] border-r border-gray-200 select-none text-center cursor-pointer sticky left-0 z-10 ${rowChecked ? "bg-indigo-100/80" : "bg-white group-hover:bg-gray-50"}`}
                    style={{ width: HASH_COL_W, minWidth: HASH_COL_W, maxWidth: HASH_COL_W }}
                    onMouseDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                        e.preventDefault();
                        selectVisibleRow(origIdx, true);
                        return;
                      }
                      // Prevent shift+click text selection before the range-tick
                      if (e.shiftKey) e.preventDefault();
                    }}
                    onClick={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.shiftKey) return;
                      tickRow(origIdx, e.shiftKey);
                    }}
                    onContextMenu={(e) => { e.preventDefault(); setRowRangeInput(""); setRowCtx({ x: e.clientX, y: e.clientY, idx: origIdx }); }}
                    title={`Original row #${origIdx + 1} · Click to select the row · Shift+click extends the range · Right-click for range select`}
                  >
                    {/* Checkbox reveals on row hover or when checked; otherwise the
                        row number shows. Keeps the 30px gutter uncluttered. The
                        whole gutter cell is the click target — the checkbox itself
                        is display-only. */}
                    <input
                      type="checkbox"
                      tabIndex={-1}
                      className={`h-3 w-3 accent-indigo-500 cursor-pointer pointer-events-none ${rowChecked ? "inline-block" : "hidden group-hover:inline-block"}`}
                      checked={rowChecked}
                      readOnly
                    />
                    <span className={rowChecked ? "hidden" : "group-hover:hidden"}>{visualIdx + 1}</span>
                  </td>

                  {columns.map((col, colIdx) => {
                    const isEditing = editCell?.rowIdx === origIdx && editCell?.col === col.name;
                    const cellVal   = row[col.name];
                    const isNull    = cellVal === null || cellVal === undefined;
                    const isSel     = selectedCells.has(cellKey(origIdx, col.name));
                    const frozen    = isFrozenCol(colIdx);
                    // Whole-line tint when the row or the column is ticked for
                    // bulk delete — the selection must be visible along its full
                    // length, not only at the gutter checkbox.
                    const ticked    = rowChecked || checkedCols.has(col.name);

                    return (
                      <td
                        key={col.name}
                        data-grid-row={origIdx}
                        data-grid-col={colIdx}
                        onMouseDown={(e) => {
                          if (!isEditing) beginCellSelection(origIdx, col.name, e);
                        }}
                        onMouseEnter={(e) => {
                          if (!isEditing) extendMouseSelection(origIdx, col.name, e);
                        }}
                        onDoubleClick={() => {
                          if (!isEditing) startEdit(origIdx, col.name);
                        }}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          // If right-clicking an unselected cell, select just that cell
                          if (!selectedCells.has(cellKey(origIdx, col.name))) {
                            setSelectedCells(new Set([cellKey(origIdx, col.name)]));
                            setSelAnchor({ row: origIdx, col: col.name });
                            setSelFocus({ row: origIdx, col: col.name });
                          }
                          gridRef.current?.focus({ preventScroll: true });
                          setCellCtx({ x: e.clientX, y: e.clientY, row: origIdx, col: col.name });
                        }}
                        className={`border-r border-gray-200 font-mono text-xs transition-colors
                          ${frozen ? "sticky z-10 group-hover:bg-gray-50" : ""}
                          ${isEditing
                            ? "p-0 bg-indigo-50"
                            : isSel
                              ? "px-3 py-1.5 cursor-pointer bg-blue-100 outline outline-1 outline-blue-400"
                              : isNull
                                ? `px-3 py-1.5 cursor-pointer ${ticked ? "bg-indigo-50" : "bg-amber-50/60 hover:bg-amber-100/60"}`
                                : `px-3 py-1.5 cursor-pointer hover:bg-indigo-50/50 ${ticked ? "bg-indigo-50" : frozen ? "bg-white" : ""}`}`}
                        style={frozen ? { left: frozenLeft(colIdx), width: FROZEN_COL_W, minWidth: FROZEN_COL_W, maxWidth: FROZEN_COL_W } : undefined}
                      >
                        {isEditing ? (
                          <input
                            ref={inputRef}
                            className="w-full bg-white border border-indigo-400 rounded-sm px-3 py-1.5 text-xs text-gray-900 focus:outline-none"
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                e.preventDefault();
                                void commitEdit(true);
                                moveSelectionFocus(1, 0, false, false);
                              }
                              if (e.key === "Tab") {
                                e.preventDefault();
                                void commitEdit(true);
                                moveSelectionFocus(0, e.shiftKey ? -1 : 1, false, false);
                              }
                              // Arrow keys commit the edit and move the selection
                              // so navigation keeps flowing after a value change
                              // (Excel-style). Left/Right only navigate when the
                              // text cursor is at the boundary, so in-value cursor
                              // movement still works.
                              if (e.key === "ArrowUp") {
                                e.preventDefault();
                                void commitEdit(true);
                                moveSelectionFocus(-1, 0, false, false);
                              }
                              if (e.key === "ArrowDown") {
                                e.preventDefault();
                                void commitEdit(true);
                                moveSelectionFocus(1, 0, false, false);
                              }
                              if (e.key === "ArrowLeft" && e.currentTarget.selectionStart === 0) {
                                e.preventDefault();
                                void commitEdit(true);
                                moveSelectionFocus(0, -1, false, false);
                              }
                              if (e.key === "ArrowRight" && e.currentTarget.selectionStart === e.currentTarget.value.length) {
                                e.preventDefault();
                                void commitEdit(true);
                                moveSelectionFocus(0, 1, false, false);
                              }
                              if (e.key === "Escape") {
                                setEditCell(null);
                                requestAnimationFrame(() => gridRef.current?.focus({ preventScroll: true }));
                              }
                            }}
                            onBlur={() => { void commitEdit(false); }}
                          />
                        ) : isNull ? (
                          <span className="text-amber-400 italic text-[10px] font-medium">null</span>
                        ) : (
                          <span className={col.kind === "numeric" ? "text-gray-700" : "text-gray-600"}>
                            {(() => {
                              // Always derive display from the raw stored value + the
                              // CURRENT decimals setting — never trust a stale display
                              // string. A hand-typed value can be stored as a string
                              // (e.g. right after edit, before the next full reload),
                              // so coerce with Number() rather than gating on typeof.
                              if (col.name in columnDecimals) {
                                const n = typeof cellVal === "number" ? cellVal : Number(cellVal);
                                if (Number.isFinite(n)) return n.toFixed(columnDecimals[col.name]);
                              }
                              return String(cellVal);
                            })()}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}

            {padBottom > 0 && (
              <tr aria-hidden="true"><td colSpan={columns.length + 1} style={{ height: padBottom, padding: 0, border: 0 }} /></tr>
            )}
            {displayRows.length === 0 && (
              <tr>
                <td
                  colSpan={columns.length + 1}
                  className="px-6 py-16 text-center text-gray-400 text-sm"
                >
                  No rows match the current filters
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* ── Legend ── */}
      <div className="flex-shrink-0 flex items-center gap-4 text-[10px] text-gray-400 px-1">
        <span>Click a <span className="text-blue-600">type badge</span> to toggle num / cat / txt / date</span>
        <span>·</span>
        <span>Double-click <span className="text-gray-500">header</span> to rename · Right-click to delete</span>
        <span>·</span>
        <span>Click to select · Double-click / Enter to edit · Drag or Shift+arrows for range · Ctrl/Cmd+C/X/V</span>
      </div>

      {/* ── Right-click context menu ── */}
      {ctxMenu && (
        <div ref={ctxRef}
          className="fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 w-48"
          style={columnMenuStyle}
          role="menu">
          <div className="sticky top-0 z-10 bg-white px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100 truncate">
            {ctxMenu.col}
            {(missingCounts[ctxMenu.col] ?? 0) > 0 && (
              <span className="ml-1 text-amber-500">({missingCounts[ctxMenu.col]} missing)</span>
            )}
          </div>
          {/* Everyday actions stay at the top level; the rest live in the
              hover-out groups below so the menu isn't a 20-item scroll. */}
          <button onClick={() => { startRename(ctxMenu.col); setCtxMenu(null); }}
            className={MENU_ITEM_CLS}>
            ✏️ Rename
          </button>
          <button onClick={() => { void copyColumn(ctxMenu.col); }}
            className={MENU_ITEM_CLS}>
            📋 Copy column
          </button>
          <button onClick={() => { void pasteColumn(ctxMenu.col); }}
            title="Paste a copied column from the clipboard as a new column after this one — works across windows"
            className={MENU_ITEM_CLS}>
            📥 Paste column
          </button>

          <div className="border-t border-gray-100 mt-0.5" />

          {/* Everyday formatting lives at the top level; Change type and Parse
              as date moved into the More group. */}
          <button onClick={() => {
            const col = columns.find((c) => c.name === ctxMenu.col);
            setValueLabelDraft(col?.value_labels ? { ...col.value_labels } : {});
            setValueLabelCol(ctxMenu.col);
            setCtxMenu(null);
          }}
            className={MENU_ITEM_CLS}>
            🔤 Value Labels
          </button>
          <button onClick={() => { setFindReplaceCol(ctxMenu.col); setCtxMenu(null); }}
            title="Convert value: swap codes (e.g. 1 ↔ 2) or recode missing → 0"
            className={MENU_ITEM_CLS}>
            🔁 Convert value…
          </button>
          {/* Decimal places selector — explanation lives in the tooltip */}
          {columns.find((c) => c.name === ctxMenu.col)?.kind === "numeric" && (
            <div className="px-3 py-1"
              title={'Also applied to Summary, Histogram and Table 1 metrics. "A" = automatic (0 decimals for integer columns, 2 otherwise).'}>
              <span className="text-xs text-gray-500">🔢 Decimals</span>
              <div className="flex items-center gap-1 mt-1">
                {[0, 1, 2, 3, 4, "auto"].map((d) => (
                  <button key={String(d)}
                    onClick={() => {
                      if (d === "auto") {
                        clearColumnDecimals(ctxMenu.col);
                      } else {
                        setColumnDecimals(ctxMenu.col, d as number);
                      }
                      setCtxMenu(null);
                    }}
                    className={`text-[10px] w-6 h-5 rounded flex items-center justify-center transition-colors ${
                      (d === "auto" && !(ctxMenu.col in columnDecimals)) || columnDecimals[ctxMenu.col] === d
                        ? "bg-indigo-600 text-white"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                    }`}>
                    {d === "auto" ? "A" : d}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="border-t border-gray-100 mt-0.5" />

          <ColMenuGroup label="↔️ Position" groupKey="position" activeKey={openSub} setActiveKey={setOpenSub} flip={subFlip}>
            <button onClick={() => { const idx = columns.findIndex((c) => c.name === ctxMenu.col); setCtxMenu(null); addColumn(idx); }}
              className={MENU_ITEM_CLS}>
              ⬅️ Insert column left
            </button>
            <button onClick={() => { const idx = columns.findIndex((c) => c.name === ctxMenu.col); setCtxMenu(null); addColumn(idx + 1); }}
              className={MENU_ITEM_CLS}>
              ➡️ Insert column right
            </button>
            <button onClick={() => sendToEnd(ctxMenu.col)}
              className={MENU_ITEM_CLS}>
              ➡️ Send to end
            </button>
            <button onClick={() => { setMoveCol(ctxMenu.col); setCtxMenu(null); }}
              className={MENU_ITEM_CLS}>
              📍 Send to position…
            </button>
            <button
              onClick={() => {
                const idx = columns.findIndex((c) => c.name === ctxMenu.col);
                if (idx >= 0) setFrozenCount(idx + 1);
                setCtxMenu(null);
              }}
              className={MENU_ITEM_CLS}>
              ❄ Freeze up to here
            </button>
            {frozenCount > 0 && (
              <button
                onClick={() => { setFrozenCount(0); setCtxMenu(null); }}
                className={MENU_ITEM_CLS}>
                ❄ Unfreeze all
              </button>
            )}
          </ColMenuGroup>

          <ColMenuGroup label="☑️ Select columns" groupKey="select" activeKey={openSub} setActiveKey={setOpenSub} flip={subFlip}>
            {/* Range spec uses the 1-based numbers of the column-number row */}
            <div className="px-3 py-1.5">
              <input
                type="text"
                value={colRangeInput}
                onChange={(e) => setColRangeInput(e.target.value)}
                onKeyDown={(e) => {
                  e.stopPropagation();
                  if (e.key === "Enter" && applyColRangeSpec(colRangeInput) > 0) setCtxMenu(null);
                  if (e.key === "Escape") setCtxMenu(null);
                }}
                placeholder="Select e.g. 3-10, 15"
                className="w-full border border-gray-200 rounded px-1.5 py-1 text-[11px]
                  placeholder-gray-300 focus:outline-none focus:border-indigo-400"
              />
            </div>
            <button onClick={() => {
              if (applyColRangeSpec(colRangeInput) > 0) setCtxMenu(null);
            }}
              className={MENU_ITEM_CLS}>
              ✅ Select column range
            </button>
            <button onClick={() => {
              tickColsFromHere(columns.findIndex((c) => c.name === ctxMenu.col));
              setCtxMenu(null);
            }}
              className={MENU_ITEM_CLS}>
              ⤵️ Select here → end
            </button>
            <button onClick={() => { invertCheckedCols(); setCtxMenu(null); }}
              className={MENU_ITEM_CLS}>
              🔁 Invert selection
            </button>
            {(checkedCols.size > 0 || checkedRows.size > 0) && (
              <button onClick={() => { clearChecks(); setCtxMenu(null); }}
                className={MENU_ITEM_CLS}>
                ✖️ Clear selection
              </button>
            )}
          </ColMenuGroup>

          <ColMenuGroup label="⋯ More" groupKey="more" activeKey={openSub} setActiveKey={setOpenSub} flip={subFlip}>
            <button onClick={() => { cycleKind(ctxMenu.col); setCtxMenu(null); }}
              className={MENU_ITEM_CLS}>
              🏷️ Change type
            </button>
            <button onClick={() => { setParseDateCol(ctxMenu.col); setCtxMenu(null); }}
              className={MENU_ITEM_CLS}>
              📅 Parse as date…
            </button>
            <button onClick={() => duplicateColumn(ctxMenu.col)}
              className={MENU_ITEM_CLS}>
              📑 Duplicate column
            </button>
            <button onClick={openSuggestNames}
              className={MENU_ITEM_CLS}>
              💡 Suggest names…
            </button>
            <button onClick={() => {
              const col = columns.find((c) => c.name === ctxMenu.col);
              setColumnAnalysisExcluded(ctxMenu.col, !(col?.analysis_excluded ?? false));
              setCtxMenu(null);
            }}
              className={MENU_ITEM_CLS}>
              {columns.find((c) => c.name === ctxMenu.col)?.analysis_excluded
                ? "✅ Include in analysis" : "🚫 Exclude from analysis"}
            </button>
          </ColMenuGroup>

          {(missingCounts[ctxMenu.col] ?? 0) > 0 && (
            <ColMenuGroup
              label={`📊 Fill ${missingCounts[ctxMenu.col]} blanks`}
              groupKey="fill" activeKey={openSub} setActiveKey={setOpenSub} flip={subFlip} tone="amber">
              <button onClick={() => { fillBlanks(ctxMenu.col, "__mean__"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                📊 Mean
              </button>
              <button onClick={() => { fillBlanks(ctxMenu.col, "__median__"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                📊 Median
              </button>
              <button onClick={() => { fillBlanks(ctxMenu.col, "0"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                0️⃣ Zero
              </button>
              <button onClick={() => { fillBlanks(ctxMenu.col, "__mice__"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                🧬 MICE (multiple imputation)
              </button>
              <button onClick={() => { fillBlanks(ctxMenu.col, "__rownum__"); }}
                className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                🔢 Case number (1…n)
              </button>
              {fillMode === ctxMenu.col ? (
                <div className="px-3 py-1 flex items-center gap-1">
                  <input ref={fillRef} autoFocus
                    className="text-xs border border-gray-300 rounded px-1.5 py-0.5 w-20 focus:outline-none focus:border-indigo-400"
                    placeholder="value"
                    value={fillVal}
                    onChange={(e) => setFillVal(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") { fillBlanks(ctxMenu.col, fillVal); setFillMode(null); setFillVal(""); }
                      if (e.key === "Escape") { setFillMode(null); setFillVal(""); }
                    }}
                  />
                  <button onClick={() => { fillBlanks(ctxMenu.col, fillVal); setFillMode(null); setFillVal(""); }}
                    className="text-[10px] px-1.5 py-0.5 bg-indigo-600 text-white rounded hover:bg-indigo-700">Fill</button>
                </div>
              ) : (
                <button onClick={() => { setFillMode(ctxMenu.col); setFillVal(""); }}
                  className="w-full text-left px-3 py-1 text-xs text-gray-700 hover:bg-amber-50 flex items-center gap-2">
                  ✏️ Custom value...
                </button>
              )}
            </ColMenuGroup>
          )}

          <div className="border-t border-gray-100 mt-0.5" />
          <button onClick={() => deleteColumn(ctxMenu.col)}
            className="w-full text-left px-3 py-1.5 text-xs text-red-500 hover:bg-red-50 flex items-center gap-2">
            🗑️ Delete column
          </button>
        </div>
      )}

      {/* ── Cell right-click context menu ── */}
      {cellCtx && (
        <div ref={cellCtxRef}
          className="fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 w-48"
          style={cellMenuStyle}
          role="menu">
          <div className="sticky top-0 z-10 bg-white px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100 truncate">
            {selectedCells.size > 1
              ? `${selectedCells.size} cells selected`
              : `Row ${cellCtx.row + 1}, ${cellCtx.col}`}
          </div>
          <button onClick={() => { clearSelectedCells(); setCellCtx(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            🧹 Clear {selectedCells.size > 1 ? `${selectedCells.size} cells` : "cell"}
          </button>
          <button onClick={() => { copyCells(); setCellCtx(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📋 Copy {selectedCells.size > 1 ? `${selectedCells.size} cells` : "cell"}
          </button>
          <button onClick={() => { void cutCells(); setCellCtx(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ✂ Cut {selectedCells.size > 1 ? `${selectedCells.size} cells` : "cell"}
          </button>
          <button onClick={async () => {
            setCellCtx(null);
            try {
              const text = await navigator.clipboard.readText();
              if (text.trim()) await pasteCellsAt(cellCtx.row, cellCtx.col, text);
            } catch { /* clipboard denied */ }
          }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📌 Paste here
          </button>
        </div>
      )}

      {/* ── Row right-click context menu ── */}
      {rowCtx && (
        <div ref={rowCtxRef}
          className="fixed z-50 bg-white border border-gray-200 rounded-xl shadow-xl py-1 w-44"
          style={rowMenuStyle}
          role="menu">
          <div className="sticky top-0 z-10 bg-white px-3 py-1.5 text-xs text-gray-400 font-medium border-b border-gray-100">Row {rowCtx.idx + 1}</div>
          <button onClick={() => copyRow(rowCtx.idx)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            📋 Copy row
          </button>
          <div className="border-t border-gray-100 mt-0.5" />
          <button onClick={() => addRow(rowCtx.idx)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ⬆️ Insert row above
          </button>
          <button onClick={() => addRow(rowCtx.idx + 1)}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ⬇️ Insert row below
          </button>
          <div className="border-t border-gray-100 mt-0.5" />
          {/* Bulk tick helpers — range spec uses the displayed gutter numbers */}
          <div className="px-3 pt-1.5 pb-0.5 text-[10px] uppercase tracking-wide text-gray-400 select-none">
            Select rows
          </div>
          <div className="px-3 py-1">
            <input
              type="text"
              value={rowRangeInput}
              onChange={(e) => setRowRangeInput(e.target.value)}
              onKeyDown={(e) => {
                e.stopPropagation();
                if (e.key === "Enter" && applyRowRangeSpec(rowRangeInput) > 0) setRowCtx(null);
                if (e.key === "Escape") setRowCtx(null);
              }}
              placeholder="Select e.g. 100-500, 750"
              className="w-full border border-gray-200 rounded px-1.5 py-1 text-[11px]
                placeholder-gray-300 focus:outline-none focus:border-indigo-400"
              autoFocus
            />
          </div>
          <button onClick={() => {
            if (applyRowRangeSpec(rowRangeInput) > 0) setRowCtx(null);
          }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ✅ Select row range
          </button>
          <button onClick={() => {
            tickRowsFromHere(displayRows.findIndex((r) => (r._idx as number) === rowCtx.idx));
            setRowCtx(null);
          }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            ⤵️ Select here → end
          </button>
          <button onClick={() => { invertCheckedRows(); setRowCtx(null); }}
            className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
            🔁 Invert selection
          </button>
          {checkedRows.size > 0 && (
            <button onClick={() => { clearChecks(); setRowCtx(null); }}
              className="w-full text-left px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2">
              ✖️ Clear selection
            </button>
          )}
          <div className="border-t border-gray-100 mt-0.5" />
          <button onClick={() => deleteRow(rowCtx.idx)}
            className="w-full text-left px-3 py-1.5 text-xs text-red-500 hover:bg-red-50 flex items-center gap-2">
            🗑️ Delete row
          </button>
        </div>
      )}

      {/* ── Value Labels Modal ── */}
      {valueLabelCol && (
        <ValueLabelsModal
          colName={valueLabelCol}
          columns={columns}
          preview={preview}
          draft={valueLabelDraft}
          setDraft={setValueLabelDraft}
          session={session}
          onClose={() => setValueLabelCol(null)}
        />
      )}

      {/* ── Find & Replace Modal ── */}
      {findReplaceCol && (
        <FindReplaceModal
          colName={findReplaceCol}
          columns={columns}
          preview={preview}
          session={session}
          onClose={() => setFindReplaceCol(null)}
          onApplied={bumpUndo}
        />
      )}

      {/* ── Parse as Date Modal ── */}
      {parseDateCol && (
        <ParseDatesModal
          colName={parseDateCol}
          columns={columns}
          session={session}
          onClose={() => setParseDateCol(null)}
          onApplied={bumpUndo}
        />
      )}

      {/* ── Send column to position ── */}
      {moveCol && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30"
          onMouseDown={(e) => { if (e.target === e.currentTarget) setMoveCol(null); }}>
          <div className="bg-white rounded-xl shadow-2xl w-72" onMouseDown={(e) => e.stopPropagation()}>
            <div className="px-4 py-3 border-b border-gray-200">
              <h3 className="text-sm font-semibold text-gray-800">Send column to position</h3>
              <p className="text-[11px] text-gray-400 mt-0.5 truncate">{moveCol}</p>
            </div>
            <div className="px-4 py-3 space-y-2">
              <label className="text-xs text-gray-500">New position (1–{columns.length})</label>
              <input type="number" min={1} max={columns.length} autoFocus
                defaultValue={columns.findIndex((c) => c.name === moveCol) + 1}
                onKeyDown={(e) => { if (e.key === "Enter") moveToPosition(moveCol, parseInt((e.target as HTMLInputElement).value, 10)); if (e.key === "Escape") setMoveCol(null); }}
                id="move-pos-input"
                className="w-full text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" />
              <p className="text-[10px] text-gray-400">Other columns shift to make room. The “#” and frozen columns stay pinned.</p>
            </div>
            <div className="px-4 py-3 border-t border-gray-200 flex justify-end gap-2">
              <button onClick={() => setMoveCol(null)} className="px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50">Cancel</button>
              <button onClick={() => { const el = document.getElementById("move-pos-input") as HTMLInputElement | null; moveToPosition(moveCol, el ? parseInt(el.value, 10) : 1); }}
                className="px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700">Send</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Suggest names (bulk review) ── */}
      {suggestOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setSuggestOpen(false)}>
          <div className="bg-white rounded-xl shadow-2xl w-[28rem] max-h-[80vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
            <div className="px-5 py-3.5 border-b border-gray-200">
              <h3 className="text-sm font-semibold text-gray-800">Rename columns</h3>
              <p className="text-[11px] text-gray-400 mt-0.5">Edit any target name. Sentence-case suggestions are pre-filled and ticked; medical acronyms (LDL, DM…) keep their case — tick &amp; edit to change them too. Applying renames the ticked rows.</p>
            </div>
            <div className="flex-1 overflow-y-auto px-5 py-3 space-y-1">
              {(session?.columns ?? []).map((c) => {
                const changed = (suggestDraft[c.name] ?? c.name).trim() !== c.name;
                return (
                  <div key={c.name} className="flex items-center gap-2 text-xs">
                    <input type="checkbox" checked={suggestAccept[c.name] ?? false}
                      onChange={(e) => setSuggestAccept((p) => ({ ...p, [c.name]: e.target.checked }))}
                      className="accent-indigo-500 flex-shrink-0" />
                    <span className="font-mono text-gray-400 truncate w-32 flex-shrink-0" title={c.name}>{c.name}</span>
                    <span className="text-gray-300 flex-shrink-0">→</span>
                    <input
                      value={suggestDraft[c.name] ?? c.name}
                      onChange={(e) => {
                        setSuggestDraft((p) => ({ ...p, [c.name]: e.target.value }));
                        setSuggestAccept((p) => ({ ...p, [c.name]: e.target.value.trim() !== c.name }));
                      }}
                      className={`flex-1 text-xs border rounded px-2 py-0.5 focus:outline-none focus:border-indigo-400 ${changed ? "border-indigo-300 text-gray-900" : "border-gray-200 text-gray-500"}`} />
                  </div>
                );
              })}
            </div>
            <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-between">
              <button onClick={() => setSuggestAccept((session?.columns ?? []).reduce((a, c) => ({ ...a, [c.name]: false }), {}))}
                className="text-xs text-gray-400 hover:text-gray-700">Untick all</button>
              <div className="flex gap-2">
                <button onClick={() => setSuggestOpen(false)} className="px-3 py-1.5 text-xs text-gray-500 border border-gray-200 rounded-lg hover:bg-gray-50">Cancel</button>
                <button onClick={applySuggestions} disabled={suggestBusy}
                  className="px-3 py-1.5 text-xs bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50">
                  {suggestBusy ? "Applying…" : "Apply ticked"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Dictionary modal ────────────────────────────────────────────── */}
      {showDictionary && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={() => setShowDictionary(false)}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-gray-100">
              <div className="flex items-center gap-2">
                <BookOpen size={18} className="text-indigo-500" />
                <h2 className="font-semibold text-gray-800">Variable Dictionary</h2>
                <span className="text-xs text-gray-400">
                  Edit labels, value codings, and metadata for every column.
                </span>
              </div>
              <button
                onClick={() => setShowDictionary(false)}
                className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100"
                aria-label="Close dictionary"
              >
                <X size={18} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <DataDictionaryPanel />
            </div>
          </div>
        </div>
      )}

      {/* Bulk-delete action bar — appears when any rows/columns are ticked. */}
      {(checkedRows.size > 0 || checkedCols.size > 0) && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2 rounded-full bg-gray-900/95 backdrop-blur text-white pl-4 pr-1.5 py-1.5 shadow-xl text-xs">
          <span className="font-medium whitespace-nowrap">
            {[
              checkedRows.size > 0 ? `${checkedRows.size} row${checkedRows.size > 1 ? "s" : ""}` : null,
              checkedCols.size > 0 ? `${checkedCols.size} column${checkedCols.size > 1 ? "s" : ""}` : null,
            ].filter(Boolean).join(" · ")}
          </span>
          <button
            onClick={deleteChecked}
            className="rounded-full bg-red-500 hover:bg-red-600 px-3 py-1 font-medium transition-colors"
          >
            Delete
          </button>
          <button
            onClick={clearChecks}
            title="Clear selection (Esc)"
            className="w-6 h-6 rounded-full flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
