import { useCallback, useEffect, useRef, useState } from "react";
import { backendFetch } from "../../../backendApi";

export type ChatMessage = { role: "user" | "ai"; text: string; retrievalLog?: string };

export type JournalEntry = {
  id: string;
  date: string;
  preview: string;
  fullTranscript: ChatMessage[];
  /** True after this entry has been sent to /ingest-history so recommendations use it */
  syncedToMemory?: boolean;
  /**
   * `conversation` = live Personaplex AI session (saveOrUpdateEntry).
   * Omitted or `journal` = imported / pasted / one-off saves (saveEntry).
   */
  entrySource?: "journal" | "conversation";
};

const STORAGE_KEY = "openjournal-history";

function loadFromStorage(): JournalEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveToStorage(entries: JournalEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch (e) {
    console.warn("[Personaplex] Failed to save journal history:", e);
  }
}

function generateId(): string {
  return `entry-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function formatDate(isoDate: string): string {
  try {
    const d = new Date(isoDate);
    const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
    const day = d.getDate();
    const year = d.getFullYear();
    return `${month} ${day}, ${year}`;
  } catch {
    return isoDate;
  }
}

function transcriptToPreview(messages: ChatMessage[], maxLen = 100): string {
  const text = messages.map((m) => m.text).join(" ");
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trim() + "…";
}

export const EXPORT_VERSION = 1;

export type ExportPayload = {
  version: number;
  exportedAt: string;
  entries: JournalEntry[];
};

function isExportPayload(obj: unknown): obj is ExportPayload {
  if (obj == null || typeof obj !== "object") return false;
  const o = obj as Record<string, unknown>;
  return (
    typeof o.version === "number" &&
    typeof o.exportedAt === "string" &&
    Array.isArray(o.entries)
  );
}

export const KNOWLEDGE_BASE_EXPORT_VERSION = 2 as const;

export type KnowledgeBaseLibraryItem = {
  id: string;
  title: string;
  author?: string;
  date_completed?: string;
  note?: string;
};

export type KnowledgeBaseLibrarySnapshot = {
  books: KnowledgeBaseLibraryItem[];
  podcasts: KnowledgeBaseLibraryItem[];
  articles: KnowledgeBaseLibraryItem[];
  research: KnowledgeBaseLibraryItem[];
};

export type KnowledgeBaseExport = {
  version: typeof KNOWLEDGE_BASE_EXPORT_VERSION;
  exportedAt: string;
  entries: JournalEntry[];
  library: KnowledgeBaseLibrarySnapshot;
};

function isKnowledgeBaseExport(obj: unknown): obj is KnowledgeBaseExport {
  if (!isExportPayload(obj)) return false;
  const o = obj as Record<string, unknown>;
  if (o.version !== KNOWLEDGE_BASE_EXPORT_VERSION) return false;
  const lib = o.library;
  if (lib == null || typeof lib !== "object") return false;
  const L = lib as Record<string, unknown>;
  return (
    Array.isArray(L.books) &&
    Array.isArray(L.podcasts) &&
    Array.isArray(L.articles) &&
    Array.isArray(L.research)
  );
}

export function buildKnowledgeBaseJson(
  entries: JournalEntry[],
  library: KnowledgeBaseLibrarySnapshot
): string {
  const payload: KnowledgeBaseExport = {
    version: KNOWLEDGE_BASE_EXPORT_VERSION,
    exportedAt: new Date().toISOString(),
    entries,
    library: {
      books: [...library.books],
      podcasts: [...library.podcasts],
      articles: [...library.articles],
      research: [...library.research],
    },
  };
  return JSON.stringify(payload, null, 2);
}

export type ParsedKnowledgeBaseFile =
  | { kind: "full"; data: KnowledgeBaseExport }
  | { kind: "journalsOnly"; data: ExportPayload };

export function parseKnowledgeBaseFile(text: string): ParsedKnowledgeBaseFile | null {
  try {
    const j = JSON.parse(text) as unknown;
    if (isKnowledgeBaseExport(j)) return { kind: "full", data: j };
    if (isExportPayload(j)) return { kind: "journalsOnly", data: j };
    return null;
  } catch {
    return null;
  }
}

function normalizeMessage(m: unknown): ChatMessage | null {
  if (m == null || typeof m !== "object") return null;
  const o = m as Record<string, unknown>;
  if (o.role !== "user" && o.role !== "ai") return null;
  if (typeof o.text !== "string") return null;
  const retrievalLog = typeof o.retrievalLog === "string" ? o.retrievalLog : undefined;
  return { role: o.role as "user" | "ai", text: o.text, retrievalLog };
}

function normalizeEntry(raw: unknown): JournalEntry | null {
  if (raw == null || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const fullTranscript = Array.isArray(o.fullTranscript)
    ? o.fullTranscript.map(normalizeMessage).filter((m): m is ChatMessage => m != null)
    : [];
  if (fullTranscript.length === 0) return null;
  const id = typeof o.id === "string" ? o.id : generateId();
  const date = typeof o.date === "string" ? o.date : new Date().toISOString();
  const preview = typeof o.preview === "string" ? o.preview : transcriptToPreview(fullTranscript);
  const syncedToMemory = o.syncedToMemory === true;
  let entrySource: "journal" | "conversation" | undefined;
  if (o.entrySource === "conversation") entrySource = "conversation";
  else if (o.entrySource === "journal") entrySource = "journal";
  return { id, date, preview, fullTranscript, syncedToMemory, entrySource };
}

export const useJournalHistory = () => {
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const entriesRef = useRef<JournalEntry[]>(entries);
  entriesRef.current = entries;

  useEffect(() => {
    setEntries(loadFromStorage());
  }, []);

  useEffect(() => {
    saveToStorage(entries);
  }, [entries]);

  const saveEntry = useCallback((transcript: ChatMessage[], dateOverride?: string) => {
    if (transcript.length === 0) return "";
    const id = generateId();
    const date =
      dateOverride && !Number.isNaN(Date.parse(dateOverride))
        ? dateOverride
        : new Date().toISOString();
    const preview = transcriptToPreview(transcript);
    const entry: JournalEntry = {
      id,
      date,
      preview,
      fullTranscript: transcript,
      syncedToMemory: false,
      entrySource: "journal",
    };
    setEntries((prev) => [entry, ...prev]);
    return id;
  }, []);

  /**
   * Create or update one active history entry while a live session is running.
   * Returns the stable entry id so callers can keep updating the same card.
   */
  const saveOrUpdateEntry = useCallback(
    (entryId: string | null, transcript: ChatMessage[], dateOverride?: string) => {
      if (transcript.length === 0) return entryId ?? "";
      const date =
        dateOverride && !Number.isNaN(Date.parse(dateOverride))
          ? dateOverride
          : new Date().toISOString();
      const preview = transcriptToPreview(transcript);
      const id = entryId || generateId();
      setEntries((prev) => {
        const idx = prev.findIndex((e) => e.id === id);
        const nextEntry: JournalEntry = {
          id,
          date,
          preview,
          fullTranscript: transcript,
          // Any update means this entry should be re-synced.
          syncedToMemory: false,
          entrySource: "conversation",
        };
        if (idx === -1) return [nextEntry, ...prev];
        const copy = [...prev];
        copy[idx] = nextEntry;
        return copy;
      });
      return id;
    },
    []
  );

  const markEntrySynced = useCallback((id: string) => {
    setEntries((prev) =>
      prev.map((e) => (e.id === id ? { ...e, syncedToMemory: true } : e))
    );
  }, []);

  /** Send every history entry that isn't yet synced to the backend. Returns count of newly synced. */
  const syncUnsyncedEntries = useCallback(async (): Promise<number> => {
    const list = entriesRef.current;
    const unsynced = list.filter((e) => !e.syncedToMemory);
    if (unsynced.length === 0) return 0;
    const syncedIds: string[] = [];
    for (const entry of unsynced) {
      const text = entry.fullTranscript
        .map((m) => (m.role === "user" ? "User: " + m.text : "Assistant: " + m.text))
        .join("\n\n");
      if (!text.trim()) continue;
      try {
        const r = await backendFetch("/ingest-history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text.trim(), entry_date: entry.date }),
        });
        const data = r.ok ? await r.json().catch(() => ({})) : { ok: false };
        if (data && data.ok !== false) syncedIds.push(entry.id);
      } catch {
        /* skip failed (e.g. 524 timeout); entry stays unsynced and will retry later */
      }
    }
    if (syncedIds.length > 0) {
      setEntries((prev) =>
        prev.map((e) => (syncedIds.includes(e.id) ? { ...e, syncedToMemory: true } : e))
      );
    }
    return syncedIds.length;
  }, []);

  const clearHistory = useCallback(() => {
    setEntries([]);
  }, []);

  const deleteEntry = useCallback((id: string) => {
    setEntries((prev) => prev.filter((e) => e.id !== id));
  }, []);

  /** Replace transcript for an existing entry (e.g. edits on The Brain). Marks unsynced for re-ingest. */
  const updateJournalEntry = useCallback((id: string, fullTranscript: ChatMessage[]) => {
    if (fullTranscript.length === 0) return;
    const hasText = fullTranscript.some((m) => m.text.trim().length > 0);
    if (!hasText) return;
    setEntries((prev) => {
      const idx = prev.findIndex((e) => e.id === id);
      if (idx === -1) return prev;
      const preview = transcriptToPreview(fullTranscript);
      const copy = [...prev];
      copy[idx] = {
        ...copy[idx],
        fullTranscript,
        preview,
        syncedToMemory: false,
      };
      return copy;
    });
  }, []);

  const getFormattedDate = useCallback((entry: JournalEntry) => {
    return formatDate(entry.date);
  }, []);

  const exportAllJournals = useCallback((): string => {
    const payload: ExportPayload = {
      version: EXPORT_VERSION,
      exportedAt: new Date().toISOString(),
      entries,
    };
    return JSON.stringify(payload, null, 2);
  }, [entries]);

  const importEntriesFromExport = useCallback((payload: ExportPayload): number => {
    const normalized = payload.entries
      .map((raw) => normalizeEntry(raw))
      .filter((e): e is JournalEntry => e != null);
    if (normalized.length === 0) return 0;
    const withNewIds = normalized.map((e) => ({
      ...e,
      id: generateId(),
    }));
    setEntries((prev) => [...withNewIds, ...prev]);
    return withNewIds.length;
  }, []);

  /** Replace local journal history entirely (used after server vector wipe + knowledge base re-import). */
  const importEntriesReplaceAll = useCallback((payload: ExportPayload): number => {
    const normalized = payload.entries
      .map((raw) => normalizeEntry(raw))
      .filter((e): e is JournalEntry => e != null);
    if (normalized.length === 0) {
      setEntries([]);
      saveToStorage([]);
      return 0;
    }
    const withNewIds = normalized.map((e) => ({
      ...e,
      id: generateId(),
      syncedToMemory: false as const,
    }));
    setEntries(withNewIds);
    saveToStorage(withNewIds);
    return withNewIds.length;
  }, []);

  return {
    entries,
    saveEntry,
    saveOrUpdateEntry,
    markEntrySynced,
    syncUnsyncedEntries,
    clearHistory,
    deleteEntry,
    updateJournalEntry,
    getFormattedDate,
    exportAllJournals,
    importEntriesFromExport,
    importEntriesReplaceAll,
    isExportPayload,
  };
};
