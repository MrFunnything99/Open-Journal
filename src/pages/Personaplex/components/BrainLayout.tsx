import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage, JournalEntry } from "../hooks/useJournalHistory";
import { formatCalendarDayHeading, localCalendarDayKey } from "../knowledgeBaseMarkdownZip";

export type BrainLibraryCategory = "book" | "podcast" | "article" | "research";

export type LibraryItemRow = {
  id: string;
  title: string;
  author?: string;
  date_completed?: string;
  note?: string;
};

type Selection =
  | { kind: "journal"; id: string }
  | { kind: "journal_day"; dayKey: string }
  | { kind: "conversation"; id: string }
  | { kind: "library"; category: BrainLibraryCategory; id: string };

type YearMonthTree = {
  year: number;
  months: { month: number; monthLabel: string; entries: JournalEntry[] }[];
}[];

function buildYearMonthTree(sorted: JournalEntry[]): YearMonthTree {
  const byYear = new Map<number, Map<number, JournalEntry[]>>();
  for (const e of sorted) {
    const d = new Date(e.date);
    const y = d.getFullYear();
    const m = d.getMonth();
    if (!byYear.has(y)) byYear.set(y, new Map());
    const ym = byYear.get(y)!;
    if (!ym.has(m)) ym.set(m, []);
    ym.get(m)!.push(e);
  }
  const years = [...byYear.keys()].sort((a, b) => b - a);
  return years.map((year) => {
    const monthsMap = byYear.get(year)!;
    const months = [...monthsMap.keys()].sort((a, b) => b - a);
    return {
      year,
      months: months.map((month) => ({
        month,
        monthLabel: new Date(year, month, 1).toLocaleString("en-US", { month: "long" }),
        entries: monthsMap.get(month)!,
      })),
    };
  });
}

function isConversationEntry(e: JournalEntry): boolean {
  return e.entrySource === "conversation";
}

type BrainLayoutProps = {
  entries: JournalEntry[];
  getFormattedDate: (entry: JournalEntry) => string;
  onDeleteEntry: (id: string) => void;
  onUpdateJournalEntry: (id: string, fullTranscript: ChatMessage[]) => void;
  onToast?: (message: string) => void;
  libraryItems: {
    books: LibraryItemRow[];
    podcasts: LibraryItemRow[];
    articles: LibraryItemRow[];
    research: LibraryItemRow[];
  };
  libraryLoading: boolean;
  libraryAddCategory: BrainLibraryCategory | null;
  libraryDraftText: string;
  setLibraryDraftText: (s: string) => void;
  librarySubmitting: boolean;
  onSubmitLibraryAdd: () => void;
  onCancelLibraryAdd: () => void;
  onClickAddLibrary: (cat: BrainLibraryCategory) => void;
  onEditLibraryItem: (cat: BrainLibraryCategory, id: string) => void;
  onDeleteLibraryItem: (cat: BrainLibraryCategory, id: string) => void;
  onDownloadKnowledgeBase?: () => void;
  onImportKnowledgeBaseFile?: (file: File) => void;
  onImportJournalDumpFolder?: (files: FileList) => void;
};

function groupJournalMonthByCalendarDay(monthEntries: JournalEntry[]): { dayKey: string; entries: JournalEntry[] }[] {
  const m = new Map<string, JournalEntry[]>();
  for (const e of monthEntries) {
    const k = localCalendarDayKey(e.date);
    if (!m.has(k)) m.set(k, []);
    m.get(k)!.push(e);
  }
  const keys = [...m.keys()].sort();
  return keys.map((dayKey) => ({
    dayKey,
    entries: (m.get(dayKey) ?? []).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
  }));
}

/** Local wall-clock time for an entry (Knowledge base header + sidebar). */
function formatEntryTimeLabel(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

function formatRelativeSaved(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const diff = Date.now() - t;
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} minute${min === 1 ? "" : "s"} ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} hour${hr === 1 ? "" : "s"} ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day} day${day === 1 ? "" : "s"} ago`;
  return new Date(iso).toLocaleDateString();
}

function buildThinkingLogsText(entry: JournalEntry, getFormattedDate: (e: JournalEntry) => string): string {
  const lines: string[] = [`AI Thinking Logs — ${getFormattedDate(entry)}`, "=".repeat(50), ""];
  let aiIndex = 0;
  entry.fullTranscript.forEach((msg) => {
    if (msg.role === "ai") {
      aiIndex += 1;
      lines.push(`--- AI Response ${aiIndex} ---`, "", "Reply:", msg.text, "");
      if (msg.retrievalLog) {
        lines.push("Memory context from vector DB:", msg.retrievalLog);
      } else {
        lines.push("(No memory context retrieved for this response.)");
      }
      lines.push("");
    }
  });
  if (aiIndex === 0) lines.push("No AI responses in this entry.");
  return lines.join("\n");
}

const CAT_KEY: Record<BrainLibraryCategory, keyof BrainLayoutProps["libraryItems"]> = {
  book: "books",
  podcast: "podcasts",
  article: "articles",
  research: "research",
};

const CAT_LABEL: Record<BrainLibraryCategory, string> = {
  book: "Books",
  podcast: "Podcasts",
  article: "News & articles",
  research: "Research papers",
};

const knowledgeBaseToolbarBtnClass =
  "rounded-full border border-gray-300 bg-transparent px-3 py-1.5 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-100 dark:border-white/20 dark:text-white/85 dark:hover:bg-white/10";
const journalDumpBtnClass =
  "w-full rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-left text-xs font-medium text-white/80 transition-colors hover:bg-white/10 hover:text-white";

export const BrainLayout: FC<BrainLayoutProps> = ({
  entries,
  getFormattedDate,
  onDeleteEntry,
  onUpdateJournalEntry,
  onToast,
  libraryItems,
  libraryLoading,
  libraryAddCategory,
  libraryDraftText,
  setLibraryDraftText,
  librarySubmitting,
  onSubmitLibraryAdd,
  onCancelLibraryAdd,
  onClickAddLibrary,
  onEditLibraryItem,
  onDeleteLibraryItem,
  onDownloadKnowledgeBase,
  onImportKnowledgeBaseFile,
  onImportJournalDumpFolder,
}) => {
  type ExplorerTab = "journals" | "conversations" | "library";
  const [explorerTab, setExplorerTab] = useState<ExplorerTab>("journals");
  const [selection, setSelection] = useState<Selection | null>(null);
  const [journalExpandedYears, setJournalExpandedYears] = useState<Set<number>>(() => new Set());
  const [journalExpandedMonths, setJournalExpandedMonths] = useState<Set<string>>(() => new Set());
  const [convExpandedYears, setConvExpandedYears] = useState<Set<number>>(() => new Set());
  const [convExpandedMonths, setConvExpandedMonths] = useState<Set<string>>(() => new Set());
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    () => new Set(["books", "podcasts", "articles", "research"])
  );
  /** `${entryId}:${messageIndex}` so vector-log toggles are unique in day scroll. */
  const [expandedLogKey, setExpandedLogKey] = useState<string | null>(null);
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);
  const knowledgeBaseFileRef = useRef<HTMLInputElement>(null);
  const journalDumpFolderRef = useRef<HTMLInputElement>(null);
  const [journalEditing, setJournalEditing] = useState(false);
  const [journalDraft, setJournalDraft] = useState<ChatMessage[]>([]);

  const journalSorted = useMemo(() => {
    const list = entries.filter((e) => !isConversationEntry(e));
    // Chronological (earlier in the day first — e.g. 1 PM above 11 PM on the same date).
    return list.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }, [entries]);

  const conversationSorted = useMemo(() => {
    const list = entries.filter(isConversationEntry);
    return list.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  }, [entries]);

  const journalTree = useMemo(() => buildYearMonthTree(journalSorted), [journalSorted]);
  const conversationTree = useMemo(() => buildYearMonthTree(conversationSorted), [conversationSorted]);

  useEffect(() => {
    if (journalTree.length === 0) return;
    const y = journalTree[0].year;
    const m = journalTree[0].months[0];
    setJournalExpandedYears((prev) => new Set(prev).add(y));
    setJournalExpandedMonths((prev) => new Set(prev).add(`${y}-${m.month}`));
  }, [journalTree]);

  useEffect(() => {
    if (conversationTree.length === 0) return;
    const y = conversationTree[0].year;
    const m = conversationTree[0].months[0];
    setConvExpandedYears((prev) => new Set(prev).add(y));
    setConvExpandedMonths((prev) => new Set(prev).add(`c-${y}-${m.month}`));
  }, [conversationTree]);

  const selectFirstForTab = useCallback(
    (tab: ExplorerTab): Selection | null => {
      if (tab === "journals") {
        if (journalSorted.length === 0) return null;
        const newest = journalSorted[journalSorted.length - 1];
        return { kind: "journal_day", dayKey: localCalendarDayKey(newest.date) };
      }
      if (tab === "conversations") {
        return conversationSorted[0] ? { kind: "conversation", id: conversationSorted[0].id } : null;
      }
      for (const cat of ["book", "podcast", "article", "research"] as BrainLibraryCategory[]) {
        const list = libraryItems[CAT_KEY[cat]];
        if (list.length > 0) return { kind: "library", category: cat, id: list[0].id };
      }
      return null;
    },
    [journalSorted, conversationSorted, libraryItems]
  );

  const switchExplorerTab = useCallback(
    (next: ExplorerTab) => {
      if (next === explorerTab) return;
      if (journalEditing) {
        if (!confirm("Discard unsaved edits to switch tabs?")) return;
        setJournalEditing(false);
        setJournalDraft([]);
      }
      setExplorerTab(next);
      setSelection(selectFirstForTab(next));
      setExpandedLogKey(null);
    },
    [explorerTab, journalEditing, selectFirstForTab]
  );

  useEffect(() => {
    setSelection((prev) => {
      if (prev?.kind === "journal" && journalSorted.some((e) => e.id === prev.id)) return prev;
      if (
        prev?.kind === "journal_day" &&
        journalSorted.some((e) => localCalendarDayKey(e.date) === prev.dayKey)
      ) {
        return prev;
      }
      if (prev?.kind === "conversation" && conversationSorted.some((e) => e.id === prev.id)) return prev;
      if (prev?.kind === "library") {
        const list = libraryItems[CAT_KEY[prev.category]];
        if (list.some((x) => x.id === prev.id)) return prev;
      }
      return selectFirstForTab(explorerTab);
    });
  }, [journalSorted, conversationSorted, libraryItems, explorerTab, selectFirstForTab]);

  useEffect(() => {
    setMoreOpen(false);
  }, [selection]);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) setMoreOpen(false);
    };
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);

  const selectedTranscript = useMemo(() => {
    if (selection?.kind === "journal") return journalSorted.find((e) => e.id === selection.id) ?? null;
    if (selection?.kind === "conversation") return conversationSorted.find((e) => e.id === selection.id) ?? null;
    return null;
  }, [selection, journalSorted, conversationSorted]);

  const selectedJournalDayEntries = useMemo(() => {
    if (selection?.kind !== "journal_day") return null;
    return journalSorted.filter((e) => localCalendarDayKey(e.date) === selection.dayKey);
  }, [selection, journalSorted]);

  /** User-driven selection changes; prompts if leaving a transcript with unsaved edits. */
  const trySelect = useCallback(
    (next: Selection) => {
      if (journalEditing && (selection?.kind === "journal" || selection?.kind === "conversation")) {
        const leaving = next.kind !== selection.kind || next.id !== selection.id;
        if (leaving && !confirm("Discard unsaved edits to this entry?")) return;
        if (leaving) {
          setJournalEditing(false);
          setJournalDraft([]);
        }
      }
      setSelection(next);
      setExpandedLogKey(null);
    },
    [journalEditing, selection]
  );

  useEffect(() => {
    setJournalEditing(false);
    setJournalDraft([]);
  }, [selectedTranscript?.id]);

  const selectedLibrary = useMemo(() => {
    if (selection?.kind !== "library") return null;
    return libraryItems[CAT_KEY[selection.category]].find((x) => x.id === selection.id) ?? null;
  }, [selection, libraryItems]);

  const toggleJournalYear = (y: number) => {
    setJournalExpandedYears((prev) => {
      const next = new Set(prev);
      if (next.has(y)) next.delete(y);
      else next.add(y);
      return next;
    });
  };

  const toggleJournalMonth = (key: string) => {
    setJournalExpandedMonths((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const toggleConvYear = (y: number) => {
    setConvExpandedYears((prev) => {
      const next = new Set(prev);
      if (next.has(y)) next.delete(y);
      else next.add(y);
      return next;
    });
  };

  const toggleConvMonth = (key: string) => {
    setConvExpandedMonths((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const toggleSection = (key: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const exportEntryToFile = useCallback(
    (entry: JournalEntry) => {
      const dateStr = new Date(entry.date).toISOString().slice(0, 10);
      const filename = `Journal_Entry_${dateStr}.txt`;
      const lines = [
        getFormattedDate(entry),
        "",
        ...entry.fullTranscript.map((msg) =>
          msg.role === "user" ? `You: ${msg.text}` : `AI: ${msg.text}`
        ),
      ];
      const blob = new Blob([lines.join("\n")], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      onToast?.("Downloaded.");
    },
    [getFormattedDate, onToast]
  );

  const downloadThinkingLogs = useCallback(
    (entry: JournalEntry) => {
      const content = buildThinkingLogsText(entry, getFormattedDate);
      const dateStr = new Date(entry.date).toISOString().slice(0, 10);
      const blob = new Blob([content], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `AI_Thinking_Logs_${dateStr}.txt`;
      a.click();
      URL.revokeObjectURL(url);
      onToast?.("Thinking logs downloaded.");
    },
    [getFormattedDate, onToast]
  );

  const exportLibraryItem = useCallback(
    (item: LibraryItemRow) => {
      const lines = [item.title, item.author ? `Author: ${item.author}` : "", item.date_completed ? `Completed: ${item.date_completed}` : "", "", item.note ?? ""].filter(
        Boolean
      );
      const blob = new Blob([lines.join("\n")], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${item.title.replace(/[^\w\s-]/g, "").slice(0, 40) || "library-item"}.txt`;
      a.click();
      URL.revokeObjectURL(url);
      onToast?.("Downloaded.");
    },
    [onToast]
  );

  const journalDayActive = (dayKey: string) => selection?.kind === "journal_day" && selection.dayKey === dayKey;
  const conversationActive = (id: string) => selection?.kind === "conversation" && selection.id === id;
  const libraryActive = (cat: BrainLibraryCategory, id: string) =>
    selection?.kind === "library" && selection.category === cat && selection.id === id;

  const hasAnyContent =
    journalSorted.length + conversationSorted.length > 0 ||
    libraryItems.books.length +
      libraryItems.podcasts.length +
      libraryItems.articles.length +
      libraryItems.research.length >
      0;

  if (!hasAnyContent && !libraryLoading) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 min-h-[240px]">
        <p className="text-gray-500 dark:text-gray-400 text-center text-sm max-w-sm">
          Nothing in your brain yet. Save a journal session or add books and media from the explorer.
        </p>
      </div>
    );
  }

  const renderLibraryRows = (cat: BrainLibraryCategory) => {
    const items = libraryItems[CAT_KEY[cat]];
    return (
      <div className="ml-1 border-l border-gray-100 dark:border-gray-600 pl-2 space-y-0.5 pb-1">
        {libraryAddCategory === cat && (
          <div className="mb-2 rounded-lg border border-gray-200 bg-gray-50/80 p-2 dark:border-white/10 dark:bg-black/20">
            <textarea
              value={libraryDraftText}
              onChange={(e) => setLibraryDraftText(e.target.value)}
              rows={3}
              placeholder="Paste titles (one per line)…"
              className="w-full px-2 py-1.5 rounded-md bg-white border border-gray-200 text-gray-900 text-xs dark:bg-[#2f2f2f] dark:border-gray-600 dark:text-gray-100"
            />
            <div className="flex gap-2 mt-2">
              <button
                type="button"
                onClick={onSubmitLibraryAdd}
                disabled={librarySubmitting || !libraryDraftText.trim()}
                className="px-2 py-1 rounded-md bg-gray-900 text-xs font-medium text-white hover:bg-gray-800 disabled:opacity-50 dark:bg-white dark:text-gray-900 dark:hover:bg-white/90"
              >
                {librarySubmitting ? "Adding…" : "Add"}
              </button>
              <button type="button" onClick={onCancelLibraryAdd} className="px-2 py-1 text-xs text-gray-600 dark:text-gray-400">
                Cancel
              </button>
            </div>
          </div>
        )}
        {items.length === 0 && libraryAddCategory !== cat ? (
          <p className="text-xs text-gray-400 px-2 py-1">No items yet.</p>
        ) : (
          items.map((entry) => (
            <button
              key={entry.id}
              type="button"
              onClick={() => {
                trySelect({ kind: "library", category: cat, id: entry.id });
              }}
              className={`flex w-full flex-col items-start gap-0.5 rounded-lg px-2 py-2 text-left text-sm transition-colors ${
                libraryActive(cat, entry.id)
                  ? "bg-white text-gray-900 shadow-sm dark:bg-white dark:text-gray-900"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#404040]"
              }`}
            >
              <span className="font-medium truncate w-full">{entry.title}</span>
              {(entry.author || entry.date_completed) && (
                <span className={`text-xs truncate w-full ${libraryActive(cat, entry.id) ? "text-white/90" : "text-gray-500 dark:text-gray-400"}`}>
                  {[entry.author, entry.date_completed ? "Completed" : null].filter(Boolean).join(" · ")}
                </span>
              )}
            </button>
          ))
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-1 min-h-0 flex-col lg:flex-row gap-0 overflow-hidden">
      <aside className="w-full lg:w-[min(100%,340px)] lg:flex-shrink-0 flex flex-col min-h-0 border-b lg:border-b-0 lg:border-r border-gray-100 dark:border-gray-700 bg-white/90 dark:bg-[#2f2f2f]">
        <div className="border-b border-gray-100 bg-gray-50/90 px-3 py-2 dark:border-white/10 dark:bg-black/20 dark:backdrop-blur-md">
          <p className="mb-2 text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-gray-500 dark:text-white/45">
            Explorer
          </p>
          <div className="inline-flex max-w-full flex-wrap gap-0.5 rounded-full border border-gray-200/90 bg-gray-100/90 p-1 dark:border-white/[0.08] dark:bg-black/20 dark:backdrop-blur-md">
            <button
              type="button"
              onClick={() => switchExplorerTab("journals")}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                explorerTab === "journals"
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:bg-gray-200/80 hover:text-gray-900 dark:text-white/60 dark:hover:bg-white/10 dark:hover:text-white"
              }`}
            >
              Journals
            </button>
            <button
              type="button"
              onClick={() => switchExplorerTab("conversations")}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                explorerTab === "conversations"
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:bg-gray-200/80 hover:text-gray-900 dark:text-white/60 dark:hover:bg-white/10 dark:hover:text-white"
              }`}
            >
              Conversations
            </button>
            <button
              type="button"
              onClick={() => switchExplorerTab("library")}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                explorerTab === "library"
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-600 hover:bg-gray-200/80 hover:text-gray-900 dark:text-white/60 dark:hover:bg-white/10 dark:hover:text-white"
              }`}
            >
              Library
            </button>
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto scrollbar p-2" aria-label="Brain explorer">
          {libraryLoading && explorerTab === "library" && (
            <p className="text-xs text-gray-400 px-2 py-2">Loading library…</p>
          )}
          {explorerTab === "journals" && (
            <div className="mb-2">
              <div className="ml-1 border-l border-gray-100 dark:border-gray-600 pl-2">
                {journalTree.length === 0 ? (
                  <p className="text-xs text-gray-400 py-1 px-1">No journal entries yet.</p>
                ) : (
                  journalTree.map(({ year, months }) => (
                    <div key={year} className="mb-1">
                      <button
                        type="button"
                        onClick={() => toggleJournalYear(year)}
                        className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm font-semibold text-gray-800 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-[#404040]"
                      >
                        <svg
                          className={`h-3.5 w-3.5 shrink-0 text-gray-500 transition-transform ${journalExpandedYears.has(year) ? "rotate-90" : ""}`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        {year}
                      </button>
                      {journalExpandedYears.has(year) && (
                        <div className="ml-3 border-l border-gray-100 dark:border-gray-600 pl-2 space-y-0.5">
                          {months.map(({ month, monthLabel, entries: monthEntries }) => {
                            const mKey = `${year}-${month}`;
                            return (
                              <div key={mKey}>
                                <button
                                  type="button"
                                  onClick={() => toggleJournalMonth(mKey)}
                                  className="flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#404040]"
                                >
                                  <svg
                                    className={`h-3 w-3 shrink-0 text-gray-400 transition-transform ${journalExpandedMonths.has(mKey) ? "rotate-90" : ""}`}
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                  {monthLabel}
                                </button>
                                {journalExpandedMonths.has(mKey) && (
                                  <div className="ml-3 border-l border-gray-100 dark:border-gray-600 pl-2 space-y-2 pb-1">
                                    {groupJournalMonthByCalendarDay(monthEntries).map((dayGroup) => (
                                      <div key={dayGroup.dayKey}>
                                        <button
                                          type="button"
                                          onClick={() => trySelect({ kind: "journal_day", dayKey: dayGroup.dayKey })}
                                          className={`flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-xs transition-colors ${
                                            journalDayActive(dayGroup.dayKey)
                                              ? "bg-white text-gray-900 shadow-sm dark:bg-white dark:text-gray-900"
                                              : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#404040]"
                                          }`}
                                        >
                                          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                                          </svg>
                                          <span className="min-w-0 flex-1 truncate font-medium">{formatCalendarDayHeading(dayGroup.dayKey)}</span>
                                        </button>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {explorerTab === "conversations" && (
            <div className="mb-2">
              <div className="ml-1 border-l border-gray-100 dark:border-gray-600 pl-2">
                {conversationTree.length === 0 ? (
                  <p className="text-xs text-gray-400 py-1 px-1">No AI conversations yet. Connect and open the Session panel (right edge) to build this list.</p>
                ) : (
                  conversationTree.map(({ year, months }) => (
                    <div key={`conv-${year}`} className="mb-1">
                      <button
                        type="button"
                        onClick={() => toggleConvYear(year)}
                        className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm font-semibold text-gray-800 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-[#404040]"
                      >
                        <svg
                          className={`h-3.5 w-3.5 shrink-0 text-gray-500 transition-transform ${convExpandedYears.has(year) ? "rotate-90" : ""}`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        {year}
                      </button>
                      {convExpandedYears.has(year) && (
                        <div className="ml-3 border-l border-gray-100 dark:border-gray-600 pl-2 space-y-0.5">
                          {months.map(({ month, monthLabel, entries: monthEntries }) => {
                            const mKey = `c-${year}-${month}`;
                            return (
                              <div key={mKey}>
                                <button
                                  type="button"
                                  onClick={() => toggleConvMonth(mKey)}
                                  className="flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#404040]"
                                >
                                  <svg
                                    className={`h-3 w-3 shrink-0 text-gray-400 transition-transform ${convExpandedMonths.has(mKey) ? "rotate-90" : ""}`}
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                  {monthLabel}
                                </button>
                                {convExpandedMonths.has(mKey) && (
                                  <ul className="ml-3 border-l border-gray-100 dark:border-gray-600 pl-2 space-y-0.5 pb-1">
                                    {monthEntries.map((entry) => {
                                      const timeLbl = formatEntryTimeLabel(entry.date);
                                      return (
                                        <li key={entry.id}>
                                          <button
                                            type="button"
                                            onClick={() => {
                                              trySelect({ kind: "conversation", id: entry.id });
                                            }}
                                            className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs transition-colors ${
                                              conversationActive(entry.id)
                                                ? "bg-white text-gray-900 shadow-sm dark:bg-white dark:text-gray-900"
                                                : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#404040]"
                                            }`}
                                          >
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                                            </svg>
                                            <span className="truncate">
                                              {getFormattedDate(entry)}
                                              {timeLbl ? ` · ${timeLbl}` : ""}
                                            </span>
                                          </button>
                                        </li>
                                      );
                                    })}
                                  </ul>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {explorerTab === "library" && (
            <div className="mb-2">
              <div className="ml-1 border-l border-gray-100 dark:border-gray-600 pl-2">
                {(["book", "podcast", "article", "research"] as BrainLibraryCategory[]).map((cat) => {
                  const secKey = cat === "book" ? "books" : cat === "podcast" ? "podcasts" : cat === "article" ? "articles" : "research";
                  return (
                    <div key={cat} className="mb-1">
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => toggleSection(secKey)}
                          className="flex flex-1 min-w-0 items-center gap-2 rounded-lg px-2 py-2 text-left text-sm font-semibold text-gray-800 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-[#404040]"
                        >
                          <svg className={`h-4 w-4 shrink-0 text-gray-500 transition-transform ${expandedSections.has(secKey) ? "rotate-90" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                          <span className="truncate">{CAT_LABEL[cat]}</span>
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onClickAddLibrary(cat);
                          }}
                          className="shrink-0 rounded-lg border border-gray-200 bg-white p-1.5 text-gray-900 shadow-sm hover:bg-gray-50 dark:border-white/10 dark:bg-white/10 dark:text-white dark:hover:bg-white/20"
                          aria-label={`Add ${CAT_LABEL[cat]}`}
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                          </svg>
                        </button>
                      </div>
                      {expandedSections.has(secKey) && renderLibraryRows(cat)}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </nav>
        {onImportJournalDumpFolder && (
          <div className="border-t border-white/10 p-2">
            <input
              ref={journalDumpFolderRef}
              type="file"
              multiple
              accept=".md,text/markdown,.txt,text/plain"
              className="hidden"
              aria-hidden
              {...({ webkitdirectory: "" } as any)}
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) onImportJournalDumpFolder(e.target.files);
                e.target.value = "";
              }}
            />
            <button
              type="button"
              className={journalDumpBtnClass}
              onClick={() => journalDumpFolderRef.current?.click()}
              title="Import a folder like Journal/YYYY-MM/YYYY-MM-DD.md"
            >
              Journal dump upload
            </button>
          </div>
        )}
      </aside>

      <section className="flex-1 flex flex-col min-h-0 min-w-0 bg-[#F9FAFB] dark:bg-[#212121]">
        {selectedTranscript ? (
          <div className="flex-1 flex flex-col min-h-0 m-3 md:m-4 rounded-2xl bg-white border border-gray-100 shadow-sm dark:bg-[#2f2f2f] dark:border-gray-700 overflow-hidden">
            {onDownloadKnowledgeBase && onImportKnowledgeBaseFile && (
              <>
                <input
                  ref={knowledgeBaseFileRef}
                  type="file"
                  accept=".zip,application/zip,application/x-zip-compressed,application/json,.json"
                  className="hidden"
                  aria-hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onImportKnowledgeBaseFile(f);
                    e.target.value = "";
                  }}
                />
                <div className="flex shrink-0 justify-end gap-2 border-b border-gray-100 px-4 pb-2.5 pt-3 dark:border-white/10 dark:bg-[#252525]">
                  <button
                    type="button"
                    onClick={onDownloadKnowledgeBase}
                    className={knowledgeBaseToolbarBtnClass}
                    title="Download a .zip of Markdown files: journals/, conversations/, library/"
                  >
                    Download Markdown folder
                  </button>
                  <button
                    type="button"
                    onClick={() => knowledgeBaseFileRef.current?.click()}
                    className={knowledgeBaseToolbarBtnClass}
                    title="Upload a .zip export (or legacy .json). Layout matches The Brain explorer."
                  >
                    Upload Markdown folder
                  </button>
                </div>
              </>
            )}
            <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-[#343541]/40">
              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">The Brain</span>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-gray-400 dark:text-gray-500">Last saved: {formatRelativeSaved(selectedTranscript.date)}</span>
                <button
                  type="button"
                  onClick={() => {
                    if (!selectedTranscript) return;
                    if (journalEditing) {
                      const hasText = journalDraft.some((m) => m.text.trim().length > 0);
                      if (!hasText) {
                        onToast?.("Add some text before saving.");
                        return;
                      }
                      onUpdateJournalEntry(selectedTranscript.id, journalDraft);
                      setJournalEditing(false);
                      setJournalDraft([]);
                      onToast?.("Saved on this device.");
                      return;
                    }
                    onToast?.("Already saved in your browser. Use the pencil to edit this entry.");
                  }}
                  className="rounded-full bg-white px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition-colors hover:bg-white/90 dark:bg-white dark:text-gray-900 dark:hover:bg-white/90"
                >
                  {journalEditing ? "Save changes" : "Quick Save"}
                </button>
                <button
                  type="button"
                  className={`rounded-lg p-2 ${journalEditing ? "bg-gray-200 text-gray-900 dark:bg-white/15 dark:text-white" : "text-gray-500 hover:bg-gray-100 dark:hover:bg-[#404040]"}`}
                  aria-label={journalEditing ? "Stop editing" : "Edit transcript"}
                  title={journalEditing ? "Stop editing (discard unsaved changes)" : "Edit transcript text"}
                  onClick={() => {
                    if (!selectedTranscript) return;
                    if (journalEditing) {
                      setJournalEditing(false);
                      setJournalDraft([]);
                      return;
                    }
                    setJournalDraft(selectedTranscript.fullTranscript.map((m) => ({ ...m })));
                    setJournalEditing(true);
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                </button>
                <div className="relative" ref={moreRef}>
                  <button
                    type="button"
                    onClick={() => setMoreOpen((o) => !o)}
                    className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-[#404040]"
                    aria-label="More actions"
                    aria-expanded={moreOpen}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
                    </svg>
                  </button>
                  {moreOpen && (
                    <div className="absolute right-0 top-full mt-1 z-20 min-w-[200px] rounded-xl border border-gray-100 bg-white py-1 shadow-lg dark:bg-[#343541] dark:border-gray-600">
                      <button
                        type="button"
                        className="block w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-[#404040]"
                        onClick={() => {
                          exportEntryToFile(selectedTranscript);
                          setMoreOpen(false);
                        }}
                      >
                        Download as .txt
                      </button>
                      <button
                        type="button"
                        className="block w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-[#404040]"
                        onClick={() => {
                          downloadThinkingLogs(selectedTranscript);
                          setMoreOpen(false);
                        }}
                      >
                        AI thinking logs
                      </button>
                      <button
                        type="button"
                        className="block w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-950/30"
                        onClick={() => {
                          if (confirm("Delete this entry? This cannot be undone.")) {
                            setJournalEditing(false);
                            setJournalDraft([]);
                            onDeleteEntry(selectedTranscript.id);
                            setMoreOpen(false);
                          }
                        }}
                      >
                        Delete entry
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto scrollbar px-6 py-8 md:px-10 md:py-10">
              <h3
                className={`text-2xl md:text-3xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight ${journalEditing ? "mb-2" : "mb-2"}`}
              >
                {getFormattedDate(selectedTranscript)}
              </h3>
              {formatEntryTimeLabel(selectedTranscript.date) ? (
                <p
                  className={`text-lg font-medium text-gray-600 dark:text-white/80 tabular-nums ${journalEditing ? "mb-4" : "mb-6"}`}
                >
                  {formatEntryTimeLabel(selectedTranscript.date)}
                </p>
              ) : null}
              {journalEditing ? (
                <p className="mb-6 text-sm text-gray-600 dark:text-white/70">
                  Editing — use <span className="font-medium">Save changes</span> to keep edits, or the pencil to discard.
                </p>
              ) : null}
              <div className="max-w-3xl space-y-8 text-[15px] md:text-base leading-[1.75] text-gray-800 dark:text-gray-200 font-sans">
                {journalEditing
                  ? journalDraft.map((msg, i) => (
                      <div key={i} className="space-y-2">
                        <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500">{msg.role === "user" ? "You" : "AI"}</p>
                        <textarea
                          value={msg.text}
                          onChange={(e) => {
                            const t = e.target.value;
                            setJournalDraft((d) => {
                              const next = [...d];
                              next[i] = { ...next[i], text: t };
                              return next;
                            });
                          }}
                          rows={Math.max(4, Math.min(24, msg.text.split("\n").length + 2))}
                          className="min-h-[100px] w-full resize-y rounded-xl border border-gray-200 bg-white px-3 py-3 text-gray-900 shadow-sm placeholder-gray-400 focus:border-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-400/25 dark:border-gray-600 dark:bg-[#343541] dark:text-gray-100 dark:focus:border-white/30 dark:focus:ring-white/15"
                          aria-label={msg.role === "user" ? "Your message" : "AI message"}
                        />
                        {msg.role === "ai" && msg.retrievalLog && (
                          <div className="pt-2">
                            <button
                              type="button"
                              onClick={() =>
                                setExpandedLogKey((prev) =>
                                  prev === `${selectedTranscript.id}:${i}` ? null : `${selectedTranscript.id}:${i}`
                                )
                              }
                              className="text-xs font-medium text-gray-700 hover:underline dark:text-white/80"
                            >
                              {expandedLogKey === `${selectedTranscript.id}:${i}` ? "Hide" : "Show"} memory context (vector DB)
                            </button>
                            {expandedLogKey === `${selectedTranscript.id}:${i}` && (
                              <pre className="mt-2 p-3 rounded-xl bg-gray-50 text-gray-600 text-xs whitespace-pre-wrap break-words border border-gray-100 max-h-48 overflow-y-auto dark:bg-[#343541] dark:text-gray-400 dark:border-gray-600">
                                {msg.retrievalLog}
                              </pre>
                            )}
                          </div>
                        )}
                      </div>
                    ))
                  : selectedTranscript.fullTranscript.map((msg, i) => (
                      <div key={i} className="space-y-2">
                        <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500">{msg.role === "user" ? "You" : "AI"}</p>
                        <p className="whitespace-pre-wrap">{msg.text}</p>
                        {msg.role === "ai" && msg.retrievalLog && (
                          <div className="pt-2">
                            <button
                              type="button"
                              onClick={() =>
                                setExpandedLogKey((prev) => (prev === `${selectedTranscript.id}:${i}` ? null : `${selectedTranscript.id}:${i}`))
                              }
                              className="text-xs font-medium text-gray-700 hover:underline dark:text-white/80"
                            >
                              {expandedLogKey === `${selectedTranscript.id}:${i}` ? "Hide" : "Show"} memory context (vector DB)
                            </button>
                            {expandedLogKey === `${selectedTranscript.id}:${i}` && (
                              <pre className="mt-2 p-3 rounded-xl bg-gray-50 text-gray-600 text-xs whitespace-pre-wrap break-words border border-gray-100 max-h-48 overflow-y-auto dark:bg-[#343541] dark:text-gray-400 dark:border-gray-600">
                                {msg.retrievalLog}
                              </pre>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
              </div>
            </div>
          </div>
        ) : selectedJournalDayEntries && selectedJournalDayEntries.length > 0 ? (
          <div className="flex-1 flex flex-col min-h-0 m-3 md:m-4 rounded-2xl bg-white border border-gray-100 shadow-sm dark:bg-[#2f2f2f] dark:border-gray-700 overflow-hidden">
            {onDownloadKnowledgeBase && onImportKnowledgeBaseFile && (
              <>
                <input
                  ref={knowledgeBaseFileRef}
                  type="file"
                  accept=".zip,application/zip,application/x-zip-compressed,application/json,.json"
                  className="hidden"
                  aria-hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onImportKnowledgeBaseFile(f);
                    e.target.value = "";
                  }}
                />
                <div className="flex shrink-0 justify-end gap-2 border-b border-gray-100 px-4 pb-2.5 pt-3 dark:border-white/10 dark:bg-[#252525]">
                  <button
                    type="button"
                    onClick={onDownloadKnowledgeBase}
                    className={knowledgeBaseToolbarBtnClass}
                    title="Download a .zip of Markdown files: journals/, conversations/, library/"
                  >
                    Download Markdown folder
                  </button>
                  <button
                    type="button"
                    onClick={() => knowledgeBaseFileRef.current?.click()}
                    className={knowledgeBaseToolbarBtnClass}
                    title="Upload a .zip export (or legacy .json). Layout matches The Brain explorer."
                  >
                    Upload Markdown folder
                  </button>
                </div>
              </>
            )}
            <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-[#343541]/40">
              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">The Brain</span>
              <span className="text-xs text-gray-400 dark:text-gray-500">
                {selectedJournalDayEntries.length} entr{selectedJournalDayEntries.length === 1 ? "y" : "ies"} · scroll to read oldest → newest
              </span>
            </div>
            <div className="flex-1 overflow-y-auto scrollbar px-6 py-8 md:px-10 md:py-10">
              <h3 className="text-2xl md:text-3xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight mb-8">
                {formatCalendarDayHeading(localCalendarDayKey(selectedJournalDayEntries[0].date))}
              </h3>
              <div className="max-w-3xl space-y-12 text-[15px] md:text-base leading-[1.75] text-gray-800 dark:text-gray-200 font-sans">
                {selectedJournalDayEntries.map((entry, entryIdx) => {
                  const timeLbl = formatEntryTimeLabel(entry.date);
                  return (
                    <article key={entry.id} className={entryIdx > 0 ? "pt-12 border-t border-gray-100 dark:border-white/10" : ""}>
                      <div className="flex flex-wrap items-start justify-between gap-2 mb-6">
                        <div>
                          {timeLbl ? (
                            <p className="text-lg font-medium text-gray-600 dark:text-white/80 tabular-nums">{timeLbl}</p>
                          ) : (
                            <p className="text-lg font-medium text-gray-600 dark:text-white/80">{getFormattedDate(entry)}</p>
                          )}
                          <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">Saved {formatRelativeSaved(entry.date)}</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => trySelect({ kind: "journal", id: entry.id })}
                          className="shrink-0 rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-800 shadow-sm hover:bg-gray-50 dark:border-white/15 dark:bg-white/10 dark:text-white dark:hover:bg-white/15"
                        >
                          Edit this entry
                        </button>
                      </div>
                      <div className="space-y-8">
                        {entry.fullTranscript.map((msg, i) => {
                          const logKey = `${entry.id}:${i}`;
                          return (
                            <div key={i} className="space-y-2">
                              <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500">
                                {msg.role === "user" ? "You" : "AI"}
                              </p>
                              <p className="whitespace-pre-wrap">{msg.text}</p>
                              {msg.role === "ai" && msg.retrievalLog && (
                                <div className="pt-2">
                                  <button
                                    type="button"
                                    onClick={() => setExpandedLogKey((prev) => (prev === logKey ? null : logKey))}
                                    className="text-xs font-medium text-gray-700 hover:underline dark:text-white/80"
                                  >
                                    {expandedLogKey === logKey ? "Hide" : "Show"} memory context (vector DB)
                                  </button>
                                  {expandedLogKey === logKey && (
                                    <pre className="mt-2 p-3 rounded-xl bg-gray-50 text-gray-600 text-xs whitespace-pre-wrap break-words border border-gray-100 max-h-48 overflow-y-auto dark:bg-[#343541] dark:text-gray-400 dark:border-gray-600">
                                      {msg.retrievalLog}
                                    </pre>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </article>
                  );
                })}
              </div>
            </div>
          </div>
        ) : selectedLibrary ? (
          <div className="flex-1 flex flex-col min-h-0 m-3 md:m-4 rounded-2xl bg-white border border-gray-100 shadow-sm dark:bg-[#2f2f2f] dark:border-gray-700 overflow-hidden">
            {onDownloadKnowledgeBase && onImportKnowledgeBaseFile && (
              <>
                <input
                  ref={knowledgeBaseFileRef}
                  type="file"
                  accept=".zip,application/zip,application/x-zip-compressed,application/json,.json"
                  className="hidden"
                  aria-hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onImportKnowledgeBaseFile(f);
                    e.target.value = "";
                  }}
                />
                <div className="flex shrink-0 justify-end gap-2 border-b border-gray-100 px-4 pb-2.5 pt-3 dark:border-white/10 dark:bg-[#252525]">
                  <button
                    type="button"
                    onClick={onDownloadKnowledgeBase}
                    className={knowledgeBaseToolbarBtnClass}
                    title="Download a .zip of Markdown files: journals/, conversations/, library/"
                  >
                    Download Markdown folder
                  </button>
                  <button
                    type="button"
                    onClick={() => knowledgeBaseFileRef.current?.click()}
                    className={knowledgeBaseToolbarBtnClass}
                    title="Upload a .zip export (or legacy .json). Layout matches The Brain explorer."
                  >
                    Upload Markdown folder
                  </button>
                </div>
              </>
            )}
            <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-[#343541]/40">
              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">The Brain</span>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-gray-400 dark:text-gray-500">
                  {selectedLibrary.date_completed ? `Completed: ${selectedLibrary.date_completed}` : "In your library"}
                </span>
                <button
                  type="button"
                  onClick={() => onToast?.("Notes save when you edit from the menu.")}
                  className="rounded-full bg-white px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition-colors hover:bg-white/90 dark:bg-white dark:text-gray-900 dark:hover:bg-white/90"
                >
                  Quick Save
                </button>
                <button
                  type="button"
                  className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-[#404040]"
                  aria-label="Edit"
                  onClick={() => {
                    if (selection?.kind === "library") onEditLibraryItem(selection.category, selection.id);
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                </button>
                <div className="relative" ref={moreRef}>
                  <button
                    type="button"
                    onClick={() => setMoreOpen((o) => !o)}
                    className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-[#404040]"
                    aria-label="More actions"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
                    </svg>
                  </button>
                  {moreOpen && selection?.kind === "library" && (
                    <div className="absolute right-0 top-full mt-1 z-20 min-w-[200px] rounded-xl border border-gray-100 bg-white py-1 shadow-lg dark:bg-[#343541] dark:border-gray-600">
                      <button
                        type="button"
                        className="block w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-[#404040]"
                        onClick={() => {
                          exportLibraryItem(selectedLibrary);
                          setMoreOpen(false);
                        }}
                      >
                        Download as .txt
                      </button>
                      <button
                        type="button"
                        className="block w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-[#404040]"
                        onClick={() => {
                          onEditLibraryItem(selection.category, selection.id);
                          setMoreOpen(false);
                        }}
                      >
                        Edit notes
                      </button>
                      <button
                        type="button"
                        className="block w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-950/30"
                        onClick={() => {
                          if (confirm("Remove this item from your library?")) {
                            onDeleteLibraryItem(selection.category, selection.id);
                            setMoreOpen(false);
                          }
                        }}
                      >
                        Remove from library
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto scrollbar px-6 py-8 md:px-10 md:py-10">
              <h3 className="text-2xl md:text-3xl font-semibold text-gray-900 dark:text-gray-100 mb-2 tracking-tight">{selectedLibrary.title}</h3>
              {selectedLibrary.author && <p className="text-lg text-gray-600 dark:text-gray-400 mb-8">{selectedLibrary.author}</p>}
              <div className="max-w-3xl space-y-6 text-[15px] md:text-base leading-[1.75] text-gray-800 dark:text-gray-200 font-sans">
                {selectedLibrary.note ? (
                  <p className="whitespace-pre-wrap">{selectedLibrary.note}</p>
                ) : (
                  <p className="text-gray-400 italic">No notes yet. Use Edit to add how you felt or what stood out.</p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500 dark:text-gray-400 text-sm">Select an item from the explorer.</div>
        )}
      </section>
    </div>
  );
};
