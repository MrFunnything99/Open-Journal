import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage, JournalEntry } from "../hooks/useJournalHistory";
import {
  buildYearMonthTree,
  formatCalendarDayHeading,
  groupJournalMonthByCalendarDay,
  localCalendarDayKey,
} from "../knowledgeBaseMarkdownZip";

/* Knowledge Base: teal/violet shell aligned with Personaplex gradient.
 * Elsewhere, neutral grays may still be appropriate: e.g. MemoryDiagram.tsx, VoiceMemoTab (non-Brain), index.css .scrollbar thumb — not updated here. */
const kbAsideClass =
  "w-full lg:w-[min(100%,340px)] lg:flex-shrink-0 flex flex-col min-h-0 border-b lg:border-b-0 lg:border-r border-[rgba(120,180,200,0.12)] bg-[rgba(15,28,40,0.6)] backdrop-blur-md";
const kbExplorerHeaderClass =
  "border-b border-[rgba(120,180,200,0.1)] bg-[rgba(10,18,28,0.35)] px-4 py-3 backdrop-blur-sm";
const kbTabRailClass =
  "flex w-full gap-0.5 rounded-xl border border-[rgba(120,180,200,0.12)] bg-[rgba(20,37,52,0.8)] p-1 backdrop-blur-sm";
const kbTabSelectedClass = "bg-white text-gray-900 shadow-sm dark:bg-white/90";
const kbTabIdleClass = "text-[#9BB1BE] hover:bg-white/[0.06] hover:text-[#E8F1F5]";
const kbDetailShellClass =
  "flex-1 flex flex-col min-h-0 m-3 md:m-4 rounded-2xl border border-[rgba(120,180,200,0.14)] bg-[rgba(15,28,40,0.55)] backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.2)] overflow-hidden";
const kbToolbarStripWrapClass =
  "flex shrink-0 flex-wrap justify-end gap-2 border-b border-[rgba(120,180,200,0.1)] bg-[rgba(10,18,28,0.4)] px-4 pb-2.5 pt-3 backdrop-blur-sm";
const kbMetaRowClass =
  "flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-[rgba(120,180,200,0.12)] bg-[rgba(12,22,32,0.35)] backdrop-blur-sm";
const kbTreeBorder = "border-[rgba(120,180,200,0.12)]";
const kbHoverRow = "hover:bg-white/[0.06]";
const kbEmptyListClass =
  "text-xs text-[#5F7585] py-1 pl-2 pr-1 border-l border-[rgba(45,212,191,0.25)]";

/** Serialized after AI turn body; JSON-stringified retrieval log inside tilde fence (avoids breaking on ``` in logs). */
const KB_RETRIEVAL_FENCE_OPEN = "\n~~~selfmeridian-retrieval\n";
const KB_RETRIEVAL_FENCE_CLOSE = "\n~~~";

function parseAiMarkdownSegment(raw: string): ChatMessage {
  const open = raw.lastIndexOf(KB_RETRIEVAL_FENCE_OPEN);
  if (open < 0) return { role: "ai", text: raw.trimEnd() };
  const text = raw.slice(0, open).trimEnd();
  const after = raw.slice(open + KB_RETRIEVAL_FENCE_OPEN.length);
  const closeIdx = after.lastIndexOf(KB_RETRIEVAL_FENCE_CLOSE);
  if (closeIdx < 0) return { role: "ai", text: raw.trimEnd() };
  const jsonRaw = after.slice(0, closeIdx).trim();
  try {
    return { role: "ai", text, retrievalLog: JSON.parse(jsonRaw) as string };
  } catch {
    return { role: "ai", text: raw.trimEnd() };
  }
}

/** ChatMessage[] ↔ markdown for Brain single-textarea edit. Mirrors read labels (## You / ## AI). */
function transcriptToMarkdownForEdit(messages: ChatMessage[]): string {
  const chunks: string[] = [];
  for (const m of messages) {
    const head = m.role === "user" ? "## You" : "## AI";
    let body = m.text.replace(/\r\n/g, "\n").trimEnd();
    if (m.role === "ai" && m.retrievalLog != null && String(m.retrievalLog).trim() !== "") {
      body += KB_RETRIEVAL_FENCE_OPEN + JSON.stringify(m.retrievalLog) + KB_RETRIEVAL_FENCE_CLOSE;
    }
    chunks.push(`${head}\n\n${body}`);
  }
  return `${chunks.join("\n\n")}\n`;
}

function markdownToTranscript(source: string): ChatMessage[] {
  const normalized = source.replace(/\r\n/g, "\n").trimEnd();
  if (!normalized) return [];
  const parts = normalized.split(/^## (You|AI)\s*$/m);
  if (parts.length === 1) {
    const t = parts[0].trim();
    return t ? [{ role: "user", text: t }] : [];
  }
  const out: ChatMessage[] = [];
  if (parts[0].trim()) {
    out.push({ role: "user", text: parts[0].trim() });
  }
  for (let i = 1; i < parts.length; i += 2) {
    const label = parts[i];
    const segment = (parts[i + 1] ?? "").trim();
    if (label === "You") out.push({ role: "user", text: segment });
    else if (label === "AI") out.push(parseAiMarkdownSegment(segment));
  }
  return out;
}

type Selection =
  | { kind: "journal"; id: string }
  | { kind: "journal_day"; dayKey: string }
  | { kind: "conversation"; id: string };

function isConversationEntry(e: JournalEntry): boolean {
  return e.entrySource === "conversation";
}

type BrainLayoutProps = {
  entries: JournalEntry[];
  getFormattedDate: (entry: JournalEntry) => string;
  onDeleteEntry: (id: string) => void;
  onUpdateJournalEntry: (id: string, fullTranscript: ChatMessage[]) => void;
  onToast?: (message: string) => void;
  /** Sidebar: download all journals as `Journals/` zip (explorer-aligned tree). */
  onDownloadJournals?: () => void | Promise<void>;
  onImportKnowledgeBaseFile?: (file: File) => void;
  /** If provided, runs before opening the file picker; return false to cancel (e.g. user dismissed confirm). */
  onPrepareKnowledgeBaseUpload?: () => boolean;
  onImportJournalDumpFolder?: (files: FileList) => void;
  onPrepareJournalDumpUpload?: () => boolean;
  /** Wipe server vectors for this instance and clear local journals, caches, and in-progress chat (Start fresh). */
  onStartFresh?: () => void | Promise<void>;
  /** After local journal edits, re-post to `/ingest-history` (server replaces vectors by stable session_id). */
  syncUnsyncedEntries?: () => Promise<number>;
};

function countJournalText(entries: JournalEntry[], mode: "tokens" | "words"): number {
  let total = 0;
  for (const entry of entries) {
    for (const msg of entry.fullTranscript) {
      if (mode === "words") {
        total += msg.text.trim() ? msg.text.trim().split(/\s+/).length : 0;
      } else {
        total += msg.text.length;
      }
    }
  }
  return mode === "tokens" ? Math.ceil(total / 4) : total;
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

function PencilEditIcon({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
    </svg>
  );
}

const knowledgeBaseToolbarBtnClass =
  "rounded-full border border-[rgba(120,180,200,0.14)] bg-[rgba(20,37,52,0.8)] px-3 py-1.5 text-xs font-medium text-[#E8F1F5] backdrop-blur-sm transition-colors hover:bg-[rgba(27,46,64,0.9)]";
const journalDumpBtnClass =
  "w-full rounded-xl border border-[rgba(120,180,200,0.14)] bg-[rgba(20,37,52,0.8)] px-2 py-2 text-center text-xs font-medium text-[#E8F1F5] backdrop-blur-sm transition-colors hover:bg-[rgba(27,46,64,0.9)] leading-snug";

export const BrainLayout: FC<BrainLayoutProps> = ({
  entries,
  getFormattedDate,
  onDeleteEntry,
  onUpdateJournalEntry,
  onToast,
  onDownloadJournals,
  onImportKnowledgeBaseFile,
  onPrepareKnowledgeBaseUpload,
  onImportJournalDumpFolder,
  onPrepareJournalDumpUpload,
  onStartFresh,
  syncUnsyncedEntries,
}) => {
  type ExplorerTab = "journals" | "conversations";
  const [explorerTab, setExplorerTab] = useState<ExplorerTab>("journals");
  const [selection, setSelection] = useState<Selection | null>(null);
  const [journalExpandedYears, setJournalExpandedYears] = useState<Set<number>>(() => new Set());
  const [journalExpandedMonths, setJournalExpandedMonths] = useState<Set<string>>(() => new Set());
  const [journalCountMode, setJournalCountMode] = useState<"tokens" | "words">("tokens");
  const [convExpandedYears, setConvExpandedYears] = useState<Set<number>>(() => new Set());
  const [convExpandedMonths, setConvExpandedMonths] = useState<Set<string>>(() => new Set());
  /** `${entryId}:${messageIndex}` so vector-log toggles are unique in day scroll. */
  const [expandedLogKey, setExpandedLogKey] = useState<string | null>(null);
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);
  const [journalImportOpen, setJournalImportOpen] = useState(false);
  const journalImportMenuRef = useRef<HTMLDivElement>(null);
  const knowledgeBaseFileRef = useRef<HTMLInputElement>(null);
  const journalDumpFolderRef = useRef<HTMLInputElement>(null);
  const journalDumpFilesRef = useRef<HTMLInputElement>(null);

  const openKnowledgeBaseFilePicker = useCallback(() => {
    if (onPrepareKnowledgeBaseUpload && !onPrepareKnowledgeBaseUpload()) return;
    knowledgeBaseFileRef.current?.click();
  }, [onPrepareKnowledgeBaseUpload]);

  const openJournalDumpFolderPicker = useCallback(() => {
    if (onPrepareJournalDumpUpload && !onPrepareJournalDumpUpload()) return;
    journalDumpFolderRef.current?.click();
  }, [onPrepareJournalDumpUpload]);

  const openJournalDumpFilesPicker = useCallback(() => {
    if (onPrepareJournalDumpUpload && !onPrepareJournalDumpUpload()) return;
    journalDumpFilesRef.current?.click();
  }, [onPrepareJournalDumpUpload]);

  const [journalEditing, setJournalEditing] = useState(false);
  const [journalMarkdownDraft, setJournalMarkdownDraft] = useState("");
  const editBaselineMarkdownRef = useRef("");
  /** Inline Markdown edit for one manual journal card while staying on the day list (scroll other entries). */
  const [dayListEditEntryId, setDayListEditEntryId] = useState<string | null>(null);
  const [dayListEditMarkdown, setDayListEditMarkdown] = useState("");
  const dayListEditBaselineRef = useRef("");

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
      return conversationSorted[0] ? { kind: "conversation", id: conversationSorted[0].id } : null;
    },
    [journalSorted, conversationSorted]
  );

  const switchExplorerTab = useCallback(
    (next: ExplorerTab) => {
      if (next === explorerTab) return;
      if (journalEditing) {
        const dirty = journalMarkdownDraft !== editBaselineMarkdownRef.current;
        if (dirty && !confirm("Discard unsaved edits to switch tabs?")) return;
        setJournalEditing(false);
        setJournalMarkdownDraft("");
        editBaselineMarkdownRef.current = "";
      }
      if (dayListEditEntryId) {
        const dirty = dayListEditMarkdown !== dayListEditBaselineRef.current;
        if (dirty && !confirm("Discard unsaved edits to switch tabs?")) return;
        setDayListEditEntryId(null);
        setDayListEditMarkdown("");
        dayListEditBaselineRef.current = "";
      }
      setExplorerTab(next);
      setSelection(selectFirstForTab(next));
      setExpandedLogKey(null);
    },
    [explorerTab, journalEditing, journalMarkdownDraft, dayListEditEntryId, dayListEditMarkdown, selectFirstForTab]
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
      return selectFirstForTab(explorerTab);
    });
  }, [journalSorted, conversationSorted, explorerTab, selectFirstForTab]);

  useEffect(() => {
    setMoreOpen(false);
  }, [selection]);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      const t = e.target as Node;
      if (moreRef.current && !moreRef.current.contains(t)) setMoreOpen(false);
      if (journalImportMenuRef.current && !journalImportMenuRef.current.contains(t)) setJournalImportOpen(false);
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
      if (dayListEditEntryId) {
        const sameJournalDay =
          next.kind === "journal_day" &&
          selection?.kind === "journal_day" &&
          next.dayKey === selection.dayKey;
        if (!sameJournalDay) {
          const dirty = dayListEditMarkdown !== dayListEditBaselineRef.current;
          if (dirty && !confirm("Discard unsaved edits to this entry?")) return;
          setDayListEditEntryId(null);
          setDayListEditMarkdown("");
          dayListEditBaselineRef.current = "";
        }
      }
      if (journalEditing && (selection?.kind === "journal" || selection?.kind === "conversation")) {
        const leaving =
          next.kind !== selection.kind
            ? true
            : selection.kind === "journal" && next.kind === "journal"
              ? next.id !== selection.id
              : selection.kind === "conversation" && next.kind === "conversation"
                ? next.id !== selection.id
                : true;
        const dirty = journalMarkdownDraft !== editBaselineMarkdownRef.current;
        if (leaving && dirty && !confirm("Discard unsaved edits to this entry?")) return;
        if (leaving) {
          setJournalEditing(false);
          setJournalMarkdownDraft("");
          editBaselineMarkdownRef.current = "";
        }
      }
      setSelection(next);
      setExpandedLogKey(null);
    },
    [dayListEditEntryId, dayListEditMarkdown, journalEditing, journalMarkdownDraft, selection]
  );

  useEffect(() => {
    setJournalEditing(false);
    setJournalMarkdownDraft("");
    editBaselineMarkdownRef.current = "";
  }, [selectedTranscript?.id]);

  useEffect(() => {
    if (dayListEditEntryId && !journalSorted.some((e) => e.id === dayListEditEntryId)) {
      setDayListEditEntryId(null);
      setDayListEditMarkdown("");
      dayListEditBaselineRef.current = "";
    }
  }, [journalSorted, dayListEditEntryId]);

  const beginDayListInlineEdit = useCallback(
    (entry: JournalEntry) => {
      if (dayListEditEntryId === entry.id) return;
      if (dayListEditEntryId && dayListEditMarkdown !== dayListEditBaselineRef.current) {
        if (!confirm("Discard unsaved edits to this entry?")) return;
      }
      const md = transcriptToMarkdownForEdit(entry.fullTranscript);
      dayListEditBaselineRef.current = md;
      setDayListEditMarkdown(md);
      setDayListEditEntryId(entry.id);
    },
    [dayListEditEntryId, dayListEditMarkdown]
  );

  const cancelDayListInlineEdit = useCallback(() => {
    if (dayListEditMarkdown !== dayListEditBaselineRef.current) {
      if (!confirm("Discard unsaved changes?")) return;
    }
    setDayListEditEntryId(null);
    setDayListEditMarkdown("");
    dayListEditBaselineRef.current = "";
  }, [dayListEditMarkdown]);

  const persistJournalEditAndReingest = useCallback(
    async (entryId: string, parsed: ChatMessage[]) => {
      onUpdateJournalEntry(entryId, parsed);
      onToast?.("Saved on this device.");
      if (!syncUnsyncedEntries) return;
      await new Promise<void>((r) => setTimeout(r, 0));
      try {
        const n = await syncUnsyncedEntries();
        if (n === 0) {
          onToast?.(
            "Your edit is saved on this device, but server memory did not update. Check your connection and try again."
          );
        }
      } catch {
        onToast?.("Your edit is saved on this device, but syncing to server failed.");
      }
    },
    [onUpdateJournalEntry, onToast, syncUnsyncedEntries]
  );

  const saveDayListInlineEdit = useCallback(
    (entryId: string) => {
      const parsed = markdownToTranscript(dayListEditMarkdown);
      const hasText = parsed.some((m) => m.text.trim().length > 0);
      if (!hasText) {
        onToast?.("Add some text before saving.");
        return;
      }
      void persistJournalEditAndReingest(entryId, parsed);
      setDayListEditEntryId(null);
      setDayListEditMarkdown("");
      dayListEditBaselineRef.current = "";
    },
    [dayListEditMarkdown, onToast, persistJournalEditAndReingest]
  );

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

  const journalDayActive = (dayKey: string) => selection?.kind === "journal_day" && selection.dayKey === dayKey;
  const conversationActive = (id: string) => selection?.kind === "conversation" && selection.id === id;

  const hasAnyContent = journalSorted.length + conversationSorted.length > 0;

  return (
    <div className="flex flex-1 min-h-0 flex-col lg:flex-row gap-0 overflow-hidden">
      <aside className={kbAsideClass}>
        <div className={kbExplorerHeaderClass}>
          <p className="mb-2.5 text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-[#5F7585]">
            Explorer
          </p>
          <div className={kbTabRailClass}>
            {(
              [
                { key: "journals", label: "Manual Journals" },
                { key: "conversations", label: "AI-Assisted Journals" },
              ] as { key: ExplorerTab; label: string }[]
            ).map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => switchExplorerTab(tab.key)}
                className={`flex-1 rounded-lg px-2 py-1.5 text-[0.7rem] font-medium leading-tight transition-colors ${
                  explorerTab === tab.key ? kbTabSelectedClass : kbTabIdleClass
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto scrollbar p-2" aria-label="Brain explorer">
          {explorerTab === "journals" && (
            <div className="mb-2">
              {journalTree.length > 0 && (
                <div className="mb-1.5 flex items-center justify-end px-2">
                  <div
                    className={`flex items-center rounded-full border ${kbTreeBorder} bg-[rgba(20,37,52,0.75)] p-0.5 text-[0.6rem] font-medium backdrop-blur-sm`}
                  >
                    <button
                      type="button"
                      onClick={() => setJournalCountMode("tokens")}
                      className={`rounded-full px-2 py-0.5 transition-colors ${journalCountMode === "tokens" ? "bg-white/95 text-gray-900 shadow-sm" : "text-[#9BB1BE] hover:text-[#E8F1F5]"}`}
                    >
                      tokens
                    </button>
                    <button
                      type="button"
                      onClick={() => setJournalCountMode("words")}
                      className={`rounded-full px-2 py-0.5 transition-colors ${journalCountMode === "words" ? "bg-white/95 text-gray-900 shadow-sm" : "text-[#9BB1BE] hover:text-[#E8F1F5]"}`}
                    >
                      words
                    </button>
                  </div>
                </div>
              )}
              <div className={`ml-1 border-l ${kbTreeBorder} pl-2`}>
                {journalTree.length === 0 ? (
                  <p className={kbEmptyListClass}>No manual journal entries yet.</p>
                ) : (
                  journalTree.map(({ year, months }) => {
                    const yearEntries = months.flatMap((m) => m.entries);
                    return (
                    <div key={year} className="mb-1">
                      <button
                        type="button"
                        onClick={() => toggleJournalYear(year)}
                        className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm font-semibold text-[#E8F1F5] ${kbHoverRow}`}
                      >
                        <svg
                          className={`h-3.5 w-3.5 shrink-0 text-[#5F7585] transition-transform ${journalExpandedYears.has(year) ? "rotate-90" : ""}`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <span className="flex-1">{year}</span>
                        <span className="ml-auto shrink-0 text-[0.6rem] font-normal text-[#5F7585]">
                          {countJournalText(yearEntries, journalCountMode).toLocaleString()} {journalCountMode === "tokens" ? "tok" : "w"}
                        </span>
                      </button>
                      {journalExpandedYears.has(year) && (
                        <div className={`ml-3 border-l ${kbTreeBorder} pl-2 space-y-0.5`}>
                          {months.map(({ month, monthLabel, entries: monthEntries }) => {
                            const mKey = `${year}-${month}`;
                            return (
                              <div key={mKey}>
                                <button
                                  type="button"
                                  onClick={() => toggleJournalMonth(mKey)}
                                  className={`flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left text-xs font-medium text-[#9BB1BE] ${kbHoverRow}`}
                                >
                                  <svg
                                    className={`h-3 w-3 shrink-0 text-[#5F7585] transition-transform ${journalExpandedMonths.has(mKey) ? "rotate-90" : ""}`}
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                  <span className="flex-1">{monthLabel}</span>
                                  <span className="ml-auto shrink-0 text-[0.6rem] text-[#5F7585]">
                                    {countJournalText(monthEntries, journalCountMode).toLocaleString()} {journalCountMode === "tokens" ? "tok" : "w"}
                                  </span>
                                </button>
                                {journalExpandedMonths.has(mKey) && (
                                  <div className={`ml-3 border-l ${kbTreeBorder} pl-2 space-y-2 pb-1`}>
                                    {groupJournalMonthByCalendarDay(monthEntries).map((dayGroup) => (
                                      <div key={dayGroup.dayKey}>
                                        <button
                                          type="button"
                                          onClick={() => trySelect({ kind: "journal_day", dayKey: dayGroup.dayKey })}
                                          className={`flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-xs transition-colors ${
                                            journalDayActive(dayGroup.dayKey)
                                              ? "bg-white text-gray-900 shadow-sm dark:bg-white dark:text-gray-900"
                                              : "text-[#C5D4DE] hover:bg-white/[0.06] hover:text-[#E8F1F5]"
                                          }`}
                                        >
                                          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                                          </svg>
                                          <span className="min-w-0 flex-1 truncate font-medium">{formatCalendarDayHeading(dayGroup.dayKey)}</span>
                                          <span className="ml-1 shrink-0 text-[0.6rem] text-[#5F7585]">
                                            {countJournalText(dayGroup.entries, journalCountMode).toLocaleString()} {journalCountMode === "tokens" ? "tok" : "w"}
                                          </span>
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
                  );
                  })
                )}
              </div>
            </div>
          )}

          {explorerTab === "conversations" && (
            <div className="mb-2">
              <div className={`ml-1 border-l ${kbTreeBorder} pl-2`}>
                {conversationTree.length === 0 ? (
                  <p className={kbEmptyListClass}>No AI-assisted journal entries yet.</p>
                ) : (
                  conversationTree.map(({ year, months }) => (
                    <div key={`conv-${year}`} className="mb-1">
                      <button
                        type="button"
                        onClick={() => toggleConvYear(year)}
                        className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm font-semibold text-[#E8F1F5] ${kbHoverRow}`}
                      >
                        <svg
                          className={`h-3.5 w-3.5 shrink-0 text-[#5F7585] transition-transform ${convExpandedYears.has(year) ? "rotate-90" : ""}`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        {year}
                      </button>
                      {convExpandedYears.has(year) && (
                        <div className={`ml-3 border-l ${kbTreeBorder} pl-2 space-y-0.5`}>
                          {months.map(({ month, monthLabel, entries: monthEntries }) => {
                            const mKey = `c-${year}-${month}`;
                            return (
                              <div key={mKey}>
                                <button
                                  type="button"
                                  onClick={() => toggleConvMonth(mKey)}
                                  className={`flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left text-xs font-medium text-[#9BB1BE] ${kbHoverRow}`}
                                >
                                  <svg
                                    className={`h-3 w-3 shrink-0 text-[#5F7585] transition-transform ${convExpandedMonths.has(mKey) ? "rotate-90" : ""}`}
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                  {monthLabel}
                                </button>
                                {convExpandedMonths.has(mKey) && (
                                  <ul className={`ml-3 border-l ${kbTreeBorder} pl-2 space-y-0.5 pb-1`}>
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
                                                : "text-[#C5D4DE] hover:bg-white/[0.06] hover:text-[#E8F1F5]"
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

        </nav>
        {(onDownloadJournals || onImportJournalDumpFolder) && (
          <div className={`border-t ${kbTreeBorder} p-2`}>
            {onImportJournalDumpFolder ? (
              <>
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
                <input
                  ref={journalDumpFilesRef}
                  type="file"
                  multiple
                  accept=".md,text/markdown,.txt,text/plain"
                  className="hidden"
                  aria-hidden
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) onImportJournalDumpFolder(e.target.files);
                    e.target.value = "";
                  }}
                />
                <div className="relative w-full" ref={journalImportMenuRef}>
                  <button
                    type="button"
                    className={`${journalDumpBtnClass} inline-flex items-center justify-center gap-1.5`}
                    onClick={() => setJournalImportOpen((o) => !o)}
                    aria-expanded={journalImportOpen}
                    aria-haspopup="menu"
                    title="Import .md/.txt journals from a folder or pick one or more files; filing dates from path/name (API)"
                  >
                    Import journals
                    <span className="text-[#9BB1BE]" aria-hidden>
                      {journalImportOpen ? "▴" : "▾"}
                    </span>
                  </button>
                  {journalImportOpen && (
                    <div
                      className={`absolute bottom-full left-0 right-0 z-30 mb-1 overflow-hidden rounded-xl border border-[rgba(120,180,200,0.18)] bg-[rgba(12,22,36,0.95)] py-1 shadow-lg backdrop-blur-md`}
                      role="menu"
                    >
                      <button
                        type="button"
                        role="menuitem"
                        className="block w-full px-3 py-2.5 text-left text-xs font-medium text-[#E8F1F5] transition-colors hover:bg-white/[0.08]"
                        onClick={() => {
                          setJournalImportOpen(false);
                          openJournalDumpFolderPicker();
                        }}
                      >
                        Choose folder…
                      </button>
                      <button
                        type="button"
                        role="menuitem"
                        className="block w-full px-3 py-2.5 text-left text-xs font-medium text-[#E8F1F5] transition-colors hover:bg-white/[0.08]"
                        onClick={() => {
                          setJournalImportOpen(false);
                          openJournalDumpFilesPicker();
                        }}
                      >
                        Choose files…
                      </button>
                    </div>
                  )}
                </div>
              </>
            ) : null}
            {onDownloadJournals ? (
              <div className={onImportJournalDumpFolder ? "mt-3" : ""}>
                <button
                  type="button"
                  onClick={() => void onDownloadJournals()}
                  className={journalDumpBtnClass}
                  title="Download all journals as a .zip (Manual and AI-Assisted, explorer folder layout)"
                >
                  Download Journals
                </button>
              </div>
            ) : null}
            {onStartFresh && (
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() => void onStartFresh()}
                  className="w-full rounded-lg border border-[rgba(244,63,94,0.4)] bg-[rgba(244,63,94,0.12)] px-3 py-2 text-left text-xs font-semibold text-[#F43F5E] transition-colors hover:bg-[rgba(244,63,94,0.18)]"
                >
                  Start fresh
                </button>
              </div>
            )}
          </div>
        )}
      </aside>

      <section className="flex-1 flex flex-col min-h-0 min-w-0 bg-transparent">
        {selectedTranscript ? (
          <div className={kbDetailShellClass}>
            <div className={kbMetaRowClass}>
              <span className="text-xs font-semibold uppercase tracking-widest text-[#5F7585]">The Brain</span>
              <div className="flex flex-wrap items-center gap-2">
                {journalEditing ? (
                  <button
                    type="button"
                    onClick={() => {
                      if (!selectedTranscript) return;
                      const parsed = markdownToTranscript(journalMarkdownDraft);
                      const hasText = parsed.some((m) => m.text.trim().length > 0);
                      if (!hasText) {
                        onToast?.("Add some text before saving.");
                        return;
                      }
                      void persistJournalEditAndReingest(selectedTranscript.id, parsed);
                      setJournalEditing(false);
                      setJournalMarkdownDraft("");
                      editBaselineMarkdownRef.current = "";
                    }}
                    className="rounded-full bg-white/95 px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition-colors hover:bg-white"
                  >
                    Save changes
                  </button>
                ) : null}
                <button
                  type="button"
                  className={`rounded-lg p-2 ${journalEditing ? "bg-white/15 text-white" : "text-[#9BB1BE] hover:bg-white/[0.08] hover:text-[#E8F1F5]"}`}
                  aria-label={journalEditing ? "Discard edit" : "Edit entry"}
                  title={journalEditing ? "Discard edit (cancel)" : "Edit as Markdown"}
                  onClick={() => {
                    if (!selectedTranscript) return;
                    if (journalEditing) {
                      const dirty = journalMarkdownDraft !== editBaselineMarkdownRef.current;
                      if (dirty && !confirm("Discard unsaved changes?")) return;
                      setJournalEditing(false);
                      setJournalMarkdownDraft("");
                      editBaselineMarkdownRef.current = "";
                      return;
                    }
                    const md = transcriptToMarkdownForEdit(selectedTranscript.fullTranscript);
                    editBaselineMarkdownRef.current = md;
                    setJournalMarkdownDraft(md);
                    setJournalEditing(true);
                  }}
                >
                  <PencilEditIcon />
                </button>
                <div className="relative" ref={moreRef}>
                  <button
                    type="button"
                    onClick={() => setMoreOpen((o) => !o)}
                    className="p-2 rounded-lg text-[#9BB1BE] hover:bg-white/[0.08] hover:text-[#E8F1F5]"
                    aria-label="More actions"
                    aria-expanded={moreOpen}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
                    </svg>
                  </button>
                  {moreOpen && (
                    <div
                      className={`absolute right-0 top-full mt-1 z-20 min-w-[200px] rounded-xl border ${kbTreeBorder} bg-[rgba(12,22,36,0.96)] py-1 shadow-lg backdrop-blur-md`}
                    >
                      <button
                        type="button"
                        className="block w-full px-4 py-2 text-left text-sm text-[#E8F1F5] hover:bg-white/[0.08]"
                        onClick={() => {
                          exportEntryToFile(selectedTranscript);
                          setMoreOpen(false);
                        }}
                      >
                        Download as .txt
                      </button>
                      <button
                        type="button"
                        className="block w-full px-4 py-2 text-left text-sm text-[#F43F5E] hover:bg-[rgba(244,63,94,0.12)]"
                        onClick={() => {
                          if (confirm("Delete this entry? This cannot be undone.")) {
                            setJournalEditing(false);
                            setJournalMarkdownDraft("");
                            editBaselineMarkdownRef.current = "";
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
            <div
              className={`flex min-h-0 flex-1 flex-col px-6 py-8 md:px-10 md:py-10 ${journalEditing ? "overflow-hidden" : "overflow-y-auto scrollbar"}`}
            >
              <h3 className="text-2xl font-semibold tracking-tight text-[#E8F1F5] md:text-3xl shrink-0">
                {getFormattedDate(selectedTranscript)}
              </h3>
              {formatEntryTimeLabel(selectedTranscript.date) ? (
                <p className={`shrink-0 text-lg font-medium tabular-nums text-[#9BB1BE] ${journalEditing ? "mt-1 mb-3" : "mt-1 mb-6"}`}>
                  {formatEntryTimeLabel(selectedTranscript.date)}
                </p>
              ) : null}
              {journalEditing ? (
                <textarea
                  value={journalMarkdownDraft}
                  onChange={(e) => setJournalMarkdownDraft(e.target.value)}
                  spellCheck
                  aria-label="Journal entry (Markdown)"
                  className="mt-3 min-h-[min(50vh,20rem)] w-full max-w-3xl flex-1 resize-y rounded-lg border border-[rgba(120,180,200,0.14)] bg-transparent px-3 py-3 font-sans text-[15px] leading-[1.75] text-[#E8F1F5] placeholder:text-[#5F7585] focus:border-[rgba(45,212,191,0.45)] focus:outline-none focus:ring-2 focus:ring-[rgba(45,212,191,0.2)] md:text-base"
                />
              ) : (
                <div className="max-w-3xl space-y-8 text-[15px] md:text-base leading-[1.75] font-sans text-[#E8F1F5]">
                  {selectedTranscript.fullTranscript.map((msg, i) => (
                    <div key={i} className="space-y-2">
                      <p className="text-xs font-semibold uppercase tracking-widest text-[#5F7585]">{msg.role === "user" ? "You" : "AI"}</p>
                      <p className="whitespace-pre-wrap">{msg.text}</p>
                      {msg.role === "ai" && msg.retrievalLog && (
                        <div className="pt-2">
                          <button
                            type="button"
                            onClick={() =>
                              setExpandedLogKey((prev) => (prev === `${selectedTranscript.id}:${i}` ? null : `${selectedTranscript.id}:${i}`))
                            }
                            className="text-xs font-medium text-[#9BB1BE] hover:underline hover:text-[#E8F1F5]"
                          >
                            {expandedLogKey === `${selectedTranscript.id}:${i}` ? "Hide" : "Show"} memory context (vector DB)
                          </button>
                          {expandedLogKey === `${selectedTranscript.id}:${i}` && (
                            <pre
                              className={`mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded-xl border ${kbTreeBorder} bg-[rgba(10,18,28,0.7)] p-3 text-xs text-[#C5D4DE] backdrop-blur-sm`}
                            >
                              {msg.retrievalLog}
                            </pre>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : selectedJournalDayEntries && selectedJournalDayEntries.length > 0 ? (
          <div className={kbDetailShellClass}>
            <div className={kbMetaRowClass}>
              <span className="text-xs font-semibold uppercase tracking-widest text-[#5F7585]">The Brain</span>
              <span className="text-xs text-[#9BB1BE]">
                {selectedJournalDayEntries.length} entr{selectedJournalDayEntries.length === 1 ? "y" : "ies"}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto scrollbar px-6 py-8 md:px-10 md:py-10">
              <h3 className="mb-8 text-2xl font-semibold tracking-tight text-[#E8F1F5] md:text-3xl">
                {formatCalendarDayHeading(localCalendarDayKey(selectedJournalDayEntries[0].date))}
              </h3>
              <div className="max-w-3xl space-y-12 text-[15px] font-sans leading-[1.75] text-[#E8F1F5] md:text-base">
                {selectedJournalDayEntries.map((entry, entryIdx) => {
                  const timeLbl = formatEntryTimeLabel(entry.date);
                  return (
                    <article key={entry.id} className={entryIdx > 0 ? `pt-12 border-t ${kbTreeBorder}` : ""}>
                      <div className="mb-6 flex flex-wrap items-start justify-between gap-2">
                        <div>
                          {timeLbl ? (
                            <p className="text-lg font-medium tabular-nums text-[#9BB1BE]">{timeLbl}</p>
                          ) : (
                            <p className="text-lg font-medium text-[#9BB1BE]">{getFormattedDate(entry)}</p>
                          )}
                          <p className="mt-0.5 text-xs text-[#5F7585]">Saved {formatRelativeSaved(entry.date)}</p>
                        </div>
                        {dayListEditEntryId === entry.id ? (
                          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
                            <button
                              type="button"
                              onClick={() => saveDayListInlineEdit(entry.id)}
                              className="rounded-full bg-white/95 px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition-colors hover:bg-white"
                            >
                              Save
                            </button>
                            <button
                              type="button"
                              onClick={cancelDayListInlineEdit}
                              className="rounded-lg px-2 py-1.5 text-xs font-medium text-[#9BB1BE] transition-colors hover:bg-white/[0.08] hover:text-[#E8F1F5]"
                            >
                              Cancel
                            </button>
                          </div>
                        ) : (
                          <button
                            type="button"
                            onClick={() => beginDayListInlineEdit(entry)}
                            className="shrink-0 rounded-lg p-2 text-[#9BB1BE] transition-colors hover:bg-white/[0.08] hover:text-[#E8F1F5]"
                            aria-label="Edit this entry"
                            title="Edit this entry"
                          >
                            <PencilEditIcon />
                          </button>
                        )}
                      </div>
                      {dayListEditEntryId === entry.id ? (
                        <textarea
                          value={dayListEditMarkdown}
                          onChange={(e) => setDayListEditMarkdown(e.target.value)}
                          spellCheck
                          aria-label="Journal entry (Markdown)"
                          className="mt-2 min-h-[min(40vh,16rem)] w-full resize-y rounded-lg border border-[rgba(120,180,200,0.14)] bg-transparent px-3 py-3 font-sans text-[15px] leading-[1.75] text-[#E8F1F5] placeholder:text-[#5F7585] focus:border-[rgba(45,212,191,0.45)] focus:outline-none focus:ring-2 focus:ring-[rgba(45,212,191,0.2)] md:text-base"
                        />
                      ) : (
                        <div className="space-y-8">
                          {entry.fullTranscript.map((msg, i) => {
                            const logKey = `${entry.id}:${i}`;
                            return (
                              <div key={i} className="space-y-2">
                                <p className="text-xs font-semibold uppercase tracking-widest text-[#5F7585]">
                                  {msg.role === "user" ? "You" : "AI"}
                                </p>
                                <p className="whitespace-pre-wrap">{msg.text}</p>
                                {msg.role === "ai" && msg.retrievalLog && (
                                  <div className="pt-2">
                                    <button
                                      type="button"
                                      onClick={() => setExpandedLogKey((prev) => (prev === logKey ? null : logKey))}
                                      className="text-xs font-medium text-[#9BB1BE] hover:underline hover:text-[#E8F1F5]"
                                    >
                                      {expandedLogKey === logKey ? "Hide" : "Show"} memory context (vector DB)
                                    </button>
                                    {expandedLogKey === logKey && (
                                      <pre
                                        className={`mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded-xl border ${kbTreeBorder} bg-[rgba(10,18,28,0.7)] p-3 text-xs text-[#C5D4DE] backdrop-blur-sm`}
                                      >
                                        {msg.retrievalLog}
                                      </pre>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            </div>
          </div>
        ) : !hasAnyContent ? (
          <div className="m-3 flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-[rgba(120,180,200,0.14)] bg-[rgba(15,28,40,0.45)] shadow-sm backdrop-blur-md md:m-4">
            {onImportKnowledgeBaseFile && (
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
                <div className={kbToolbarStripWrapClass}>
                  <button
                    type="button"
                    onClick={openKnowledgeBaseFilePicker}
                    className={knowledgeBaseToolbarBtnClass}
                    title="Upload a .zip export (or legacy .json). Layout matches The Brain explorer."
                  >
                    Upload Markdown folder
                  </button>
                </div>
              </>
            )}
            <div className="flex min-h-[200px] flex-1 items-center justify-center p-6">
              <div className="w-full max-w-lg rounded-2xl border-2 border-dashed border-[rgba(120,180,200,0.28)] bg-[rgba(20,37,52,0.35)] px-6 py-10 text-center backdrop-blur-sm">
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-[#E8F1F5]">
                  Knowledge base is empty
                </h3>
                <p className="mb-6 text-sm leading-relaxed text-[#9BB1BE]">
                  Import a Markdown <strong className="font-medium text-[#C5D4DE]">.zip</strong> to load manual journals and
                  AI-assisted journals. Or use <strong className="font-medium text-[#C5D4DE]">Import journals</strong>{" "}
                  in the sidebar — a folder or one or more{" "}
                  <code className="rounded bg-[rgba(10,18,28,0.6)] px-1 text-xs text-[#C5D4DE]">.md</code> files (filing dates from path/name via the API).
                  New entries from AI-Assisted Journal Mode appear here after you save.
                </p>
                {onImportKnowledgeBaseFile ? (
                  <button
                    type="button"
                    onClick={openKnowledgeBaseFilePicker}
                    className="rounded-full border border-[rgba(120,180,200,0.14)] bg-[rgba(20,37,52,0.9)] px-6 py-2.5 text-sm font-medium text-[#E8F1F5] shadow-sm backdrop-blur-sm transition-colors hover:bg-[rgba(27,46,64,0.95)]"
                  >
                    Upload Markdown folder
                  </button>
                ) : null}
                {!onImportKnowledgeBaseFile && (
                  <p className="text-xs text-[#5F7585]">Upload is not available in this context.</p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center px-4 text-center text-sm text-[#9BB1BE]">
            Select an item from the explorer.
          </div>
        )}
      </section>
    </div>
  );
};
