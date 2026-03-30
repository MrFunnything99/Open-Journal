import { useCallback, useEffect, useRef, useState } from "react";
import { backendFetch } from "../../backendApi";
import type { ChatMessage } from "./hooks/useJournalHistory";
import { parseKnowledgeBaseFile, useJournalHistory } from "./hooks/useJournalHistory";
import {
  buildKnowledgeBaseMarkdownZip,
  extractJournalEntriesFromMarkdownDump,
  parseKnowledgeBaseMarkdownZip,
} from "./knowledgeBaseMarkdownZip";
import { BrainLayout, type BrainLibraryCategory } from "./components/BrainLayout";
import { VoiceMemoTab } from "./components/VoiceMemoTab";
import { LearningTab } from "./components/LearningTab";
import { BrainCalendarPanel } from "./components/BrainCalendarPanel";
import { PersonaplexChatProvider, type PersonaplexNavigateAction } from "./PersonaplexChatContext";
import { MobileAskComposerDockGate } from "./components/GlobalAskAnythingBar";
import { PersonaplexLeftRail } from "./components/PersonaplexLeftRail";
import { HomeChatSidebar } from "./components/HomeChatSidebar";

const RECOMMENDATIONS_CACHE_KEY = "openjournal-recommendations-cache";
const LIBRARY_CACHE_KEY = "openjournal-library-cache";

type RecItem = { title: string; author?: string; reason?: string; url?: string };
type RecommendationsBundle = {
  books: RecItem[];
  podcasts: RecItem[];
  articles: RecItem[];
  research: RecItem[];
  news: RecItem[];
};
const EMPTY_RECOMMENDATIONS: RecommendationsBundle = {
  books: [],
  podcasts: [],
  articles: [],
  research: [],
  news: [],
};

function RecFeedbackLinks({
  category,
  item,
}: {
  category: "book" | "podcast" | "article" | "research" | "news";
  item: RecItem;
}) {
  const send = (action: "like" | "dislike") => {
    void backendFetch("/recommendations/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        content_type: category,
        item_title: item.title,
        topic_tags: [item.title, item.author].filter(Boolean).join(" · ").slice(0, 200),
      }),
    });
  };
  return (
    <div className="mt-1.5 flex flex-wrap gap-3 text-[11px]">
      <button
        type="button"
        className="text-gray-500 hover:text-[#10a37f] dark:text-gray-400 dark:hover:text-emerald-400"
        onClick={() => send("like")}
      >
        More like this
      </button>
            <button
              type="button"
        className="text-gray-500 hover:text-amber-800 dark:text-gray-400 dark:hover:text-amber-400/90"
        onClick={() => send("dislike")}
            >
        Not for me
            </button>
    </div>
  );
}

function readRecommendationsCache(): RecommendationsBundle {
  try {
    const raw = localStorage.getItem(RECOMMENDATIONS_CACHE_KEY);
    if (!raw) return { ...EMPTY_RECOMMENDATIONS };
    const parsed = JSON.parse(raw) as Partial<RecommendationsBundle>;
    return {
      books: Array.isArray(parsed.books) ? parsed.books : [],
      podcasts: Array.isArray(parsed.podcasts) ? parsed.podcasts : [],
      articles: Array.isArray(parsed.articles) ? parsed.articles : [],
      research: Array.isArray(parsed.research) ? parsed.research : [],
      news: Array.isArray(parsed.news) ? parsed.news : [],
    };
  } catch {
    return { ...EMPTY_RECOMMENDATIONS };
  }
}

const KB_UPLOAD_CONFIRM_MESSAGE =
  "Uploading a knowledge base replaces everything for this session.\n\n" +
  "• Server: all journal embeddings and your library vector index are deleted, then rebuilt from this file.\n" +
  "• This device: journals and library are replaced by the import (not merged).\n" +
  "• Cached recommendations are cleared.\n\n" +
  "Stay online until syncing finishes. This cannot be undone.\n\n" +
  "Continue?";

const JOURNAL_FOLDER_UPLOAD_CONFIRM_MESSAGE =
  "Import all journal files from the selected folder? New entries will be added to your knowledge base.";

type LibraryBulkImportPayloadItem = {
  id: string;
  type: "book" | "podcast" | "article" | "research";
  title: string;
  author?: string;
  note?: string;
  date_completed?: string;
  url?: string;
  liked: boolean;
};

function librarySnapshotToBulkPayload(next: {
  books: Array<{ id: string; title: string; author?: string; note?: string; date_completed?: string }>;
  podcasts: Array<{ id: string; title: string; author?: string; note?: string; date_completed?: string }>;
  articles: Array<{ id: string; title: string; author?: string; note?: string; date_completed?: string }>;
  research: Array<{ id: string; title: string; author?: string; note?: string; date_completed?: string }>;
}): LibraryBulkImportPayloadItem[] {
  const items: LibraryBulkImportPayloadItem[] = [];
  const push = (row: (typeof next.books)[number], type: LibraryBulkImportPayloadItem["type"]) => {
    items.push({
      id: row.id,
      type,
      title: row.title,
      author: row.author,
      note: row.note,
      date_completed: row.date_completed,
      liked: true,
    });
  };
  next.books.forEach((r) => push(r, "book"));
  next.podcasts.forEach((r) => push(r, "podcast"));
  next.articles.forEach((r) => push(r, "article"));
  next.research.forEach((r) => push(r, "research"));
  return items;
}

type PersonaplexView = "voice_memo" | "brain" | "recommendations" | "learning";

export const Personaplex = () => {
  const [view, setView] = useState<PersonaplexView>("voice_memo");
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [railExpanded, setRailExpanded] = useState(true);
  const [mobileRailOpen, setMobileRailOpen] = useState(false);
  const chatToast = useCallback((msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  }, []);
  const handleChatAgentAction = useCallback(
    (actions: PersonaplexNavigateAction[]) => {
      let navigated = false;
      for (const a of actions) {
        if (a.type !== "navigate") continue;
        const target: PersonaplexView = a.view === "journal" ? "voice_memo" : a.view;
        setView(target);
        if (target === "brain" && a.brainSection) setBrainSection(a.brainSection);
        navigated = true;
      }
      if (navigated) chatToast("Opened the screen you asked for.");
    },
    [chatToast]
  );
  const [recommendations, setRecommendations] = useState<RecommendationsBundle>(() => readRecommendationsCache());
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const recommendationsInFlightRef = useRef(false);
  const [recColumnLoading, setRecColumnLoading] = useState<Partial<Record<keyof RecommendationsBundle, boolean>>>({});
  const recColumnInFlightRef = useRef<Set<string>>(new Set());
  const [consumedIds, setConsumedIds] = useState<Set<string>>(new Set());
  const [removingKeys, setRemovingKeys] = useState<Set<string>>(new Set());
  const [libraryAddCategory, setLibraryAddCategory] = useState<"book" | "podcast" | "article" | "research" | null>(null);
  const [libraryDraftText, setLibraryDraftText] = useState("");
  const [librarySubmitting, setLibrarySubmitting] = useState(false);
  type LibraryItem = { id: string; title: string; author?: string; date_completed?: string; note?: string };
  const [libraryItems, setLibraryItems] = useState<{ books: LibraryItem[]; podcasts: LibraryItem[]; articles: LibraryItem[]; research: LibraryItem[] }>({
    books: [], podcasts: [], articles: [], research: [],
  });
  const [libraryEditingId, setLibraryEditingId] = useState<string | null>(null);
  const [libraryEditDate, setLibraryEditDate] = useState("");
  const [libraryEditNote, setLibraryEditNote] = useState("");
  const [libraryUpdateSaving, setLibraryUpdateSaving] = useState(false);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [showLibraryInterview, setShowLibraryInterview] = useState(false);
  const [libraryInterviewMessages, setLibraryInterviewMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([]);
  const [libraryInterviewSessionId, setLibraryInterviewSessionId] = useState<string | null>(null);
  const [libraryInterviewLoading, setLibraryInterviewLoading] = useState(false);
  const [libraryInterviewInput, setLibraryInterviewInput] = useState("");
  const [calendarMonth, setCalendarMonth] = useState(() => {
    const d = new Date();
    return { year: d.getFullYear(), month: d.getMonth() };
  });
  const [calendarSelectedDate, setCalendarSelectedDate] = useState<string | null>(null);
  const [calendarDaySummary, setCalendarDaySummary] = useState<string | null>(null);
  const [calendarDaySummaryLoading, setCalendarDaySummaryLoading] = useState(false);
  /** Left sidebar: Knowledge base (journals, transcripts, library) vs Calendar */
  const [brainSection, setBrainSection] = useState<"knowledgeBase" | "calendar">("knowledgeBase");
  const fetchLibrary = useCallback((showCachedFirst = false): Promise<void> => {
    if (showCachedFirst) {
      try {
        const cached = localStorage.getItem(LIBRARY_CACHE_KEY);
        if (cached) {
          const parsed = JSON.parse(cached) as { books?: unknown; podcasts?: unknown; articles?: unknown; research?: unknown };
          if (parsed && typeof parsed === "object") {
            setLibraryItems({
              books: Array.isArray(parsed.books) ? parsed.books as LibraryItem[] : [],
              podcasts: Array.isArray(parsed.podcasts) ? parsed.podcasts as LibraryItem[] : [],
              articles: Array.isArray(parsed.articles) ? parsed.articles as LibraryItem[] : [],
              research: Array.isArray(parsed.research) ? parsed.research as LibraryItem[] : [],
            });
          }
        }
      } catch {
        /* ignore cache parse errors */
      }
    }
    return backendFetch("/library")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed to load library"))))
      .then((data: { books?: LibraryItem[]; podcasts?: LibraryItem[]; articles?: LibraryItem[]; research?: LibraryItem[] }) => {
        const next = {
          books: Array.isArray(data.books) ? data.books : [],
          podcasts: Array.isArray(data.podcasts) ? data.podcasts : [],
          articles: Array.isArray(data.articles) ? data.articles : [],
          research: Array.isArray(data.research) ? data.research : [],
        };
        setLibraryItems(next);
        try {
          localStorage.setItem(LIBRARY_CACHE_KEY, JSON.stringify(next));
        } catch {
          /* ignore */
        }
      })
      .catch(() => {
        if (!showCachedFirst) setLibraryItems({ books: [], podcasts: [], articles: [], research: [] });
      });
  }, []);
  useEffect(() => {
    if (view !== "brain") return;
    setLibraryLoading(true);
    fetchLibrary(true).finally(() => setLibraryLoading(false));
  }, [view, fetchLibrary]);

  const sendLibraryInterviewMessage = useCallback(
    (msg: string) => {
      if (!msg.trim() || libraryInterviewLoading) return;
      setLibraryInterviewMessages((prev) => [...prev, { role: "user", content: msg.trim() }]);
      setLibraryInterviewInput("");
      setLibraryInterviewLoading(true);
      const snapshot = libraryItems.books.map((b) => ({ id: b.id, title: b.title, author: b.author ?? undefined, note: b.note ?? undefined }));
      backendFetch("/library-interview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg.trim(),
          session_id: libraryInterviewSessionId ?? undefined,
          library_snapshot: snapshot,
        }),
      })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Interview failed"))))
        .then((data: { response?: string; session_id?: string; notes_saved?: { item_id: string; note: string }[] }) => {
          setLibraryInterviewSessionId(data.session_id ?? null);
          setLibraryInterviewMessages((prev) => [...prev, { role: "assistant", content: data.response ?? "" }]);
          if (data.notes_saved?.length) {
            fetchLibrary();
            const first = data.notes_saved[0];
            const title = libraryItems.books.find((b) => b.id === first.item_id)?.title ?? "a book";
            setToastMessage(`Note saved for ${title}.`);
            setTimeout(() => setToastMessage(null), 3000);
          }
        })
        .catch(() => {
          setLibraryInterviewMessages((prev) => [...prev, { role: "assistant", content: "Something went wrong. Please try again." }]);
          setToastMessage("Interview request failed.");
          setTimeout(() => setToastMessage(null), 3000);
        })
        .finally(() => setLibraryInterviewLoading(false));
    },
    [libraryInterviewLoading, libraryInterviewSessionId, libraryItems.books, fetchLibrary]
  );

  const {
    entries,
    saveEntry,
    syncUnsyncedEntries,
    deleteEntry,
    updateJournalEntry,
    getFormattedDate,
    importEntriesReplaceAll,
  } = useJournalHistory();

  const prepareKnowledgeBaseUpload = useCallback(() => window.confirm(KB_UPLOAD_CONFIRM_MESSAGE), []);

  const prepareJournalDumpUpload = useCallback(() => window.confirm(JOURNAL_FOLDER_UPLOAD_CONFIRM_MESSAGE), []);

  const resetServerKnowledgeBaseMemory = useCallback(async (): Promise<boolean> => {
    try {
      const r = await backendFetch("/memory-reset-knowledge-base-import", { method: "POST" });
      const data = (await r.json().catch(() => ({}))) as { ok?: boolean };
      return r.ok && data.ok !== false;
    } catch {
      return false;
    }
  }, []);

  const pushLibraryBulkToServer = useCallback(async (items: LibraryBulkImportPayloadItem[]) => {
    if (items.length === 0) return true;
    try {
      const r = await backendFetch("/library/bulk-import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items }),
      });
      const data = (await r.json().catch(() => ({}))) as { ok?: boolean };
      return r.ok && data.ok !== false;
    } catch {
      return false;
    }
  }, []);

  const handleDownloadKnowledgeBase = useCallback(async () => {
    try {
      const blob = await buildKnowledgeBaseMarkdownZip(entries, libraryItems);
      const a = document.createElement("a");
      const url = URL.createObjectURL(blob);
      a.href = url;
      a.download = `selfmeridian-knowledge-base-${new Date().toISOString().slice(0, 10)}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      setToastMessage("Downloaded Markdown folder (.zip). Unzip to browse journals, conversations, and library.");
      setTimeout(() => setToastMessage(null), 4000);
    } catch {
      setToastMessage("Download failed.");
      setTimeout(() => setToastMessage(null), 3000);
    }
  }, [entries, libraryItems]);

  const replaceImportedLibrary = useCallback(
    (lib: {
      books: (typeof libraryItems.books)[number][];
      podcasts: (typeof libraryItems.podcasts)[number][];
      articles: (typeof libraryItems.articles)[number][];
      research: (typeof libraryItems.research)[number][];
    }) => {
      const base = Date.now();
      const remap = (items: (typeof libraryItems.books)[number][], tag: string) =>
        items.map((item, i) => ({
          ...item,
          id: `lib-${base}-${tag}-${i}-${Math.random().toString(36).slice(2, 8)}`,
        }));
      const next = {
        books: remap(lib.books, "book"),
        podcasts: remap(lib.podcasts, "podcast"),
        articles: remap(lib.articles, "article"),
        research: remap(lib.research, "research"),
      };
      setLibraryItems(next);
      try {
        localStorage.setItem(LIBRARY_CACHE_KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    },
    []
  );

  const handleImportKnowledgeBaseFile = useCallback(
    async (file: File) => {
      const applyKnowledgeBaseImport = async (
        exportPayload: { version: number; exportedAt: string; entries: (typeof entries)[number][] },
        lib: {
          books: (typeof libraryItems.books)[number][];
          podcasts: (typeof libraryItems.podcasts)[number][];
          articles: (typeof libraryItems.articles)[number][];
          research: (typeof libraryItems.research)[number][];
        }
      ) => {
        if (!(await resetServerKnowledgeBaseMemory())) {
          setToastMessage("Could not reset server memory. Check your connection and try again.");
          setTimeout(() => setToastMessage(null), 6000);
          return;
        }
        try {
          localStorage.removeItem(RECOMMENDATIONS_CACHE_KEY);
        } catch {
          /* ignore */
        }
        setRecommendations({ ...EMPTY_RECOMMENDATIONS });
        const nEntries = importEntriesReplaceAll({
          version: exportPayload.version,
          exportedAt: exportPayload.exportedAt,
          entries: exportPayload.entries,
        });
        const nextLib = replaceImportedLibrary(lib);
        const bulk = librarySnapshotToBulkPayload(nextLib);
        const libOk = await pushLibraryBulkToServer(bulk);
        if (!libOk) {
          setToastMessage("Journals updated locally; library sync failed. Check the API and try uploading again.");
          setTimeout(() => setToastMessage(null), 6000);
        }
        const synced = await syncUnsyncedEntries();
        void fetchLibrary();
        const nLib = bulk.length;
        setToastMessage(
          `Knowledge base replaced: ${nEntries} journal entr${nEntries === 1 ? "y" : "ies"}, ${nLib} library item${nLib === 1 ? "" : "s"}. ` +
            `${synced} synced to server memory.`
        );
        setTimeout(() => setToastMessage(null), 6500);
      };

      try {
        const looksZip =
          file.name.toLowerCase().endsWith(".zip") ||
          file.type === "application/zip" ||
          file.type === "application/x-zip-compressed";

        if (looksZip) {
          const parsed = await parseKnowledgeBaseMarkdownZip(file);
          if (!parsed) {
            setToastMessage("That ZIP isn’t a valid Markdown knowledge base (expected journals/, conversations/, library/).");
            setTimeout(() => setToastMessage(null), 5000);
            return;
          }
          await applyKnowledgeBaseImport(
            { version: 1, exportedAt: new Date().toISOString(), entries: parsed.entries },
            parsed.library
          );
          return;
        }

        const text = await file.text();
        const parsed = parseKnowledgeBaseFile(text);
        if (!parsed) {
          setToastMessage("Use a .zip of Markdown folders, or a legacy .json backup.");
          setTimeout(() => setToastMessage(null), 4000);
          return;
        }
        if (parsed.kind === "journalsOnly") {
          await applyKnowledgeBaseImport(parsed.data, {
            books: [],
            podcasts: [],
            articles: [],
            research: [],
          });
          return;
        }
        await applyKnowledgeBaseImport(
          {
            version: parsed.data.version,
            exportedAt: parsed.data.exportedAt,
            entries: parsed.data.entries,
          },
          parsed.data.library
        );
      } catch {
        setToastMessage("Could not read that file.");
        setTimeout(() => setToastMessage(null), 4000);
      }
    },
    [fetchLibrary, importEntriesReplaceAll, pushLibraryBulkToServer, replaceImportedLibrary, resetServerKnowledgeBaseMemory, syncUnsyncedEntries]
  );

  const handleImportJournalDumpFolder = useCallback(
    async (files: FileList) => {
      const list = Array.from(files).filter((f) => /\.(md|txt)$/i.test(f.name));
      if (list.length === 0) {
        setToastMessage("No .md or .txt files found in that folder.");
        setTimeout(() => setToastMessage(null), 3500);
        return;
      }

      type Row = { date: string | null; transcript: ChatMessage[] };
      const rows: Row[] = [];
      for (const f of list) {
        const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath ?? f.name;
        const text = (await f.text()).trim();
        if (!text) continue;
        for (const item of extractJournalEntriesFromMarkdownDump(rel, text)) {
          rows.push(item);
        }
      }

      rows.sort((a, b) => {
        const ta = a.date ? Date.parse(a.date) : 0;
        const tb = b.date ? Date.parse(b.date) : 0;
        return ta - tb;
      });

      let imported = 0;
      for (const row of rows) {
        const id = saveEntry(row.transcript, row.date ?? undefined);
        if (id) imported += 1;
      }

      if (imported === 0) {
        setToastMessage("No valid journal content found.");
      } else {
        setToastMessage(`Imported ${imported} journal entr${imported === 1 ? "y" : "ies"} from folder.`);
      }
      setTimeout(() => setToastMessage(null), 4500);
    },
    [saveEntry]
  );

  const hasRunInitialSyncRef = useRef(false);
  useEffect(() => {
    if (entries.length === 0 || hasRunInitialSyncRef.current) return;
    hasRunInitialSyncRef.current = true;
    syncUnsyncedEntries();
  }, [entries.length, syncUnsyncedEntries]);

  /** Network fetch only — user clicks "Refresh recommendations". Cached lists stay until then. */
  const refreshRecommendationsFromApi = useCallback((retryCount = 0) => {
    if (recommendationsInFlightRef.current) return;
    const doFetch = (isRetry: boolean) => {
      const ac = new AbortController();
      const timeoutId = setTimeout(() => ac.abort(), 125000);
      recommendationsInFlightRef.current = true;
      backendFetch("/recommendations", { signal: ac.signal })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed to load"))))
        .then((data: Partial<RecommendationsBundle>) => {
          const next: RecommendationsBundle = {
            books: data.books ?? [],
            podcasts: data.podcasts ?? [],
            articles: data.articles ?? [],
            research: data.research ?? [],
            news: data.news ?? [],
          };
          setRecommendations(next);
          try {
            localStorage.setItem(RECOMMENDATIONS_CACHE_KEY, JSON.stringify(next));
          } catch {
            /* ignore */
          }
        })
        .catch(() => {
          if (!isRetry && retryCount < 1) {
            setTimeout(() => refreshRecommendationsFromApi(1), 2500);
            return;
          }
          setRecommendations(readRecommendationsCache());
        })
        .finally(() => {
          clearTimeout(timeoutId);
          recommendationsInFlightRef.current = false;
          setRecommendationsLoading(false);
        });
    };
    setRecommendationsLoading(true);
    doFetch(retryCount >= 1);
  }, []);

  const recColumnBtnClass =
    "shrink-0 px-2 py-1 rounded-md text-[11px] font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-[#404040] dark:text-gray-400 dark:hover:bg-[#505050] disabled:opacity-50 disabled:cursor-not-allowed transition-colors";

  const refreshRecommendationsColumn = useCallback((cat: keyof RecommendationsBundle) => {
    if (recommendationsInFlightRef.current || recColumnInFlightRef.current.has(cat)) return;
    recColumnInFlightRef.current.add(cat);
    setRecColumnLoading((s) => ({ ...s, [cat]: true }));
    const ac = new AbortController();
    const timeoutId = setTimeout(() => ac.abort(), 125000);
    backendFetch(`/recommendations?category=${encodeURIComponent(cat)}`, { signal: ac.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed to load"))))
      .then((data: Partial<RecommendationsBundle>) => {
        const slice = data[cat];
        if (!Array.isArray(slice)) return;
        setRecommendations((prev) => {
          const next = { ...prev, [cat]: slice };
          try {
            localStorage.setItem(RECOMMENDATIONS_CACHE_KEY, JSON.stringify(next));
          } catch {
            /* ignore */
          }
          return next;
        });
      })
      .catch(() => {
        setToastMessage("Could not refresh this column. Try again.");
        setTimeout(() => setToastMessage(null), 4000);
      })
      .finally(() => {
        clearTimeout(timeoutId);
        recColumnInFlightRef.current.delete(cat);
        setRecColumnLoading((s) => ({ ...s, [cat]: false }));
      });
  }, []);

  useEffect(() => {
    if (view !== "recommendations") return;
    void syncUnsyncedEntries();
    setRecommendations((prev) => {
      const hasAny =
        prev.books.length > 0 ||
        prev.podcasts.length > 0 ||
        prev.articles.length > 0 ||
        prev.research.length > 0 ||
        prev.news.length > 0;
      if (hasAny) return prev;
      return readRecommendationsCache();
    });
  }, [view, syncUnsyncedEntries]);

  useEffect(() => {
    if (
      view === "recommendations" &&
      (recommendations.books.length > 0 ||
        recommendations.podcasts.length > 0 ||
        recommendations.articles.length > 0 ||
        recommendations.research.length > 0 ||
        recommendations.news.length > 0)
    ) {
      try {
        localStorage.setItem(RECOMMENDATIONS_CACHE_KEY, JSON.stringify(recommendations));
      } catch {
        /* ignore */
      }
    }
  }, [view, recommendations]);

  const markConsumed = useCallback(
    (type: "book" | "podcast" | "article" | "research" | "news", item: RecItem) => {
      const key = `${type}:${item.title}`;
      if (consumedIds.has(key)) return;
      backendFetch("/recommendations/consumed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type,
          title: item.title ?? "",
          author: item.author ?? undefined,
          url: item.url ?? undefined,
          liked: true,
        }),
      })
        .then(async (r) => {
          const data = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(typeof data?.detail === "string" ? data.detail : "Failed to save.");
          if (data && data.ok === false) throw new Error("Failed to save.");
          return data;
        })
        .then(() => {
          setConsumedIds((prev) => new Set(prev).add(key));
          setToastMessage(
            type === "podcast"
              ? "Marked as listened."
              : "Marked as read.",
          );
          setTimeout(() => setToastMessage(null), 2500);
          setRemovingKeys((prev) => new Set(prev).add(key));
          const removeAfter = 320;
          setTimeout(() => {
            setRecommendations((prev) => ({
              ...prev,
              books: type === "book" ? prev.books.filter((x) => x.title !== item.title) : prev.books,
              podcasts: type === "podcast" ? prev.podcasts.filter((x) => x.title !== item.title) : prev.podcasts,
              articles: type === "article" ? prev.articles.filter((x) => x.title !== item.title) : prev.articles,
              research: type === "research" ? prev.research.filter((x) => x.title !== item.title) : prev.research,
              news: type === "news" ? prev.news.filter((x) => x.title !== item.title) : prev.news,
            }));
            setRemovingKeys((prev) => {
              const next = new Set(prev);
              next.delete(key);
              return next;
            });
          }, removeAfter);
        })
        .catch((err) => {
          setToastMessage(err?.message === "Failed to save." ? "Failed to save." : "Could not mark as read. Try again.");
          setTimeout(() => setToastMessage(null), 4000);
        });
    },
    [consumedIds]
  );

  return (
    <PersonaplexChatProvider onToast={chatToast} onAgentAction={handleChatAgentAction}>
    <div className="relative flex h-screen w-full flex-row overflow-hidden bg-[#0a0a12] font-sans text-white antialiased">
      {/* Ethereal mesh + animated ambient orbs (pointer-events none on layer) */}
      <div className="pointer-events-none fixed inset-0 z-0" aria-hidden>
        <div className="absolute inset-0 bg-gradient-to-br from-[#0c1228] via-[#0d1f2d] to-[#1a0f2e]" />
        <div
          className="absolute inset-0 opacity-90"
          style={{
            background:
              "radial-gradient(ellipse 85% 55% at 50% -15%, rgba(45, 212, 191, 0.22), transparent 55%), radial-gradient(ellipse 55% 45% at 100% 45%, rgba(168, 85, 247, 0.2), transparent 50%), radial-gradient(ellipse 50% 40% at 0% 85%, rgba(59, 130, 246, 0.14), transparent 50%)",
          }}
        />
        <div
          className="ambient-orb ambient-orb-1 absolute left-[6%] top-[10%] h-96 w-96 rounded-full blur-[100px]"
          style={{
            background:
              "radial-gradient(circle at 35% 35%, rgba(94, 234, 212, 0.45), rgba(45, 212, 191, 0.12) 45%, transparent 70%)",
          }}
        />
        <div
          className="ambient-orb ambient-orb-2 absolute bottom-[6%] right-[2%] h-96 w-96 rounded-full blur-[110px]"
          style={{
            background:
              "radial-gradient(circle at 40% 40%, rgba(216, 180, 254, 0.4), rgba(168, 85, 247, 0.14) 50%, transparent 72%)",
          }}
        />
        <div
          className="ambient-orb ambient-orb-3 absolute right-[18%] top-[32%] h-96 w-96 rounded-full blur-[95px]"
          style={{
            background:
              "radial-gradient(circle at 30% 50%, rgba(129, 140, 248, 0.38), rgba(99, 102, 241, 0.1) 48%, transparent 70%)",
          }}
        />
        <div
          className="ambient-orb ambient-orb-4 absolute left-[28%] bottom-[22%] h-80 w-80 rounded-full blur-[100px] md:h-96 md:w-96"
          style={{
            background:
              "radial-gradient(circle at 50% 50%, rgba(167, 139, 250, 0.32), rgba(192, 132, 252, 0.1) 55%, transparent 72%)",
          }}
        />
      </div>

      <PersonaplexLeftRail
        expanded={railExpanded}
        setExpanded={setRailExpanded}
        mobileOpen={mobileRailOpen}
        setMobileOpen={setMobileRailOpen}
        view={view}
        setView={setView}
      />

      <div className="relative z-10 flex min-h-0 min-w-0 flex-1 flex-row overflow-hidden">
      <div className="relative z-10 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      {/* Header — centered brand; nav lives in left rail (Open WebUI–style shell) */}
      <header className="relative z-20 flex-none px-4 py-3 sm:px-6 sm:py-4">
        <div className="flex w-full items-center gap-2">
          <div className="flex min-w-0 flex-1 justify-start">
            <button
              type="button"
              className="rounded-full border border-white/15 bg-white/10 p-2.5 text-white shadow-sm backdrop-blur-md md:hidden"
              onClick={() => setMobileRailOpen(true)}
              aria-label="Open menu"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
          <div className="glass-panel flex min-w-0 max-w-full shrink-0 flex-wrap items-center justify-center gap-x-3 gap-y-1 rounded-full px-4 py-2.5 text-center">
            <h1 className="shrink-0 text-xs font-medium uppercase tracking-[0.2em] text-white sm:text-sm md:text-base">
              Selfmeridian
            </h1>
          </div>
          <div className="min-w-0 flex-1" />
        </div>
      </header>

      {toastMessage && (
        <div
          className="fixed bottom-24 left-1/2 z-50 -translate-x-1/2 rounded-xl border border-white/10 bg-white/10 px-4 py-2 text-sm font-medium text-white shadow-lg backdrop-blur-xl"
          role="status"
        >
          {toastMessage}
        </div>
      )}

      {/* Main content */}
      <main className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        <div
          className={`flex-1 flex flex-col min-h-0 transition-opacity duration-300 ${
            view === "brain" || view === "voice_memo" || view === "recommendations" || view === "learning"
              ? "overflow-hidden"
              : "overflow-y-auto"
          } opacity-100`}
        >
          {view === "voice_memo" && (
            <VoiceMemoTab onToast={chatToast} saveEntry={saveEntry} syncUnsyncedEntries={syncUnsyncedEntries} />
          )}
          {view === "brain" && (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <div className="flex-none border-b border-white/10 px-4 py-3 md:px-5">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between lg:gap-4">
                  <div className="min-w-0 flex-1">
                    <h2 className="text-sm font-medium text-white/90 md:text-base">
                      {brainSection === "knowledgeBase" ? "Knowledge base" : "Calendar"}
                </h2>
                    <p className="mt-1 text-xs text-white/50 md:text-sm">
                      {brainSection === "knowledgeBase"
                        ? "Journal entries, conversation transcripts, and your books & media library."
                        : "Click a date for an AI summary of that day (journal entries + memory)."}
                    </p>
                        </div>
                  <div
                    className="flex shrink-0 gap-1 rounded-xl border border-white/10 bg-white/[0.06] p-1"
                    role="tablist"
                    aria-label="Brain section"
                  >
                  <button
                    type="button"
                      role="tab"
                      aria-selected={brainSection === "knowledgeBase"}
                      onClick={() => setBrainSection("knowledgeBase")}
                      className={`flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-colors md:text-sm ${
                        brainSection === "knowledgeBase"
                          ? "bg-white/[0.14] text-white shadow-sm ring-1 ring-white/10"
                          : "text-white/60 hover:bg-white/10 hover:text-white"
                      }`}
                    >
                      <svg className="h-4 w-4 shrink-0 opacity-80" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                        </svg>
                      <span className="whitespace-nowrap">Knowledge base</span>
                      </button>
                      <button
                        type="button"
                      role="tab"
                      aria-selected={brainSection === "calendar"}
                      onClick={() => setBrainSection("calendar")}
                      className={`flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-colors md:text-sm ${
                        brainSection === "calendar"
                          ? "bg-white/[0.14] text-white shadow-sm ring-1 ring-white/10"
                          : "text-white/60 hover:bg-white/10 hover:text-white"
                      }`}
                    >
                      <svg className="h-4 w-4 shrink-0 opacity-80" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                      <span className="whitespace-nowrap">Calendar</span>
                      </button>
                    </div>
                  </div>
                </div>
              {brainSection === "knowledgeBase" ? (
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                <BrainLayout
                  entries={entries}
                  onDeleteEntry={deleteEntry}
                  onUpdateJournalEntry={updateJournalEntry}
                  getFormattedDate={getFormattedDate}
                  onToast={(msg) => {
                    setToastMessage(msg);
                    setTimeout(() => setToastMessage(null), 3000);
                  }}
                  libraryItems={libraryItems}
                  libraryLoading={libraryLoading}
                  libraryAddCategory={libraryAddCategory}
                  libraryDraftText={libraryDraftText}
                  setLibraryDraftText={setLibraryDraftText}
                  librarySubmitting={librarySubmitting}
                  onSubmitLibraryAdd={() => {
                    if (!libraryDraftText.trim() || librarySubmitting || !libraryAddCategory) return;
                                    setLibrarySubmitting(true);
                                    const payload = libraryDraftText.trim();
                                    backendFetch("/library-notes", {
                                      method: "POST",
                                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ text: payload, type: libraryAddCategory }),
                                    })
                                      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Library update failed"))))
                                      .then((data: { ok?: boolean; items_added?: number }) => {
                                        const added = data?.items_added ?? 0;
                                        setLibraryDraftText("");
                        setLibraryAddCategory(null);
                                        if (added > 0) fetchLibrary();
                        setToastMessage(added > 0 ? `Added ${added} item(s).` : "No items recognized; try clearer titles.");
                                        setTimeout(() => setToastMessage(null), 4000);
                                      })
                                      .catch(() => {
                                        setToastMessage("Library update failed.");
                                        setTimeout(() => setToastMessage(null), 4000);
                                      })
                                      .finally(() => setLibrarySubmitting(false));
                                  }}
                  onCancelLibraryAdd={() => {
                    setLibraryAddCategory(null);
                                    setLibraryDraftText("");
                                  }}
                  onClickAddLibrary={(cat: BrainLibraryCategory) => {
                    setLibraryAddCategory((prev) => (prev === cat ? null : cat));
                    setLibraryDraftText("");
                  }}
                  onEditLibraryItem={(_cat, id) => {
                    setLibraryEditingId(id);
                    const e = [...libraryItems.books, ...libraryItems.podcasts, ...libraryItems.articles, ...libraryItems.research].find((x) => x.id === id);
                    setLibraryEditDate(e?.date_completed ?? "");
                    setLibraryEditNote(e?.note ?? "");
                  }}
                  onDeleteLibraryItem={(_cat, id) => {
                    backendFetch(`/library/${encodeURIComponent(id)}`, { method: "DELETE" })
                                  .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Delete failed"))))
                                  .then((data: { ok?: boolean }) => {
                                    if (data?.ok !== false) {
                                      fetchLibrary();
                          setToastMessage("Removed from library.");
                                      setTimeout(() => setToastMessage(null), 2000);
                                    }
                                  })
                                  .catch(() => {
                                    setToastMessage("Failed to remove.");
                                    setTimeout(() => setToastMessage(null), 3000);
                      });
                  }}
                  onDownloadKnowledgeBase={handleDownloadKnowledgeBase}
                  onImportKnowledgeBaseFile={handleImportKnowledgeBaseFile}
                  onPrepareKnowledgeBaseUpload={prepareKnowledgeBaseUpload}
                  onImportJournalDumpFolder={handleImportJournalDumpFolder}
                  onPrepareJournalDumpUpload={prepareJournalDumpUpload}
                />
              </div>
              ) : (
                <BrainCalendarPanel
                  entries={entries}
                  calendarMonth={calendarMonth}
                  setCalendarMonth={setCalendarMonth}
                  calendarSelectedDate={calendarSelectedDate}
                  setCalendarSelectedDate={setCalendarSelectedDate}
                  calendarDaySummary={calendarDaySummary}
                  setCalendarDaySummary={setCalendarDaySummary}
                  calendarDaySummaryLoading={calendarDaySummaryLoading}
                  setCalendarDaySummaryLoading={setCalendarDaySummaryLoading}
                />
              )}
            </div>
          )}
          {view === "recommendations" && (
            <div className="flex h-full min-h-0 flex-1 flex-col overflow-hidden px-3 pb-3 pt-2 sm:px-4 sm:pb-4 sm:pt-3 md:px-6">
              <div className="mb-3 flex flex-shrink-0 flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-medium text-gray-800 dark:text-gray-200 uppercase tracking-wider">
                    Recommendations
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                    Based on your journal memory and what you’ve already read or listened to. Suggestions are kept on this
                    device until you refresh. Use <strong className="font-medium text-gray-600 dark:text-gray-300">Refresh</strong> in a
                    column to update only that category (faster). Mark items as read/listened so future runs get better.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                              <button
                                type="button"
                    onClick={() => refreshRecommendationsFromApi()}
                    disabled={recommendationsLoading}
                    className="px-3 py-2 rounded-lg bg-gray-100 text-gray-600 text-sm font-medium hover:bg-gray-200 dark:bg-[#404040] dark:text-gray-400 dark:hover:bg-[#505050] disabled:opacity-50 transition-colors"
                  >
                    {recommendationsLoading ? "Updating…" : "Refresh recommendations"}
                              </button>
                            </div>
                          </div>
              <div className="min-h-0 flex-1 overflow-hidden">
              {recommendationsLoading &&
              !recommendations.books.length &&
              !recommendations.podcasts.length &&
              !recommendations.articles.length &&
              !recommendations.research.length &&
              !recommendations.news.length ? (
                <div className="h-full min-h-0 overflow-y-auto pr-1">
                <p className="text-gray-500 dark:text-gray-400 text-sm">
                  Loading recommendations… This can take up to ~90 seconds.
                </p>
                        </div>
              ) : !recommendationsLoading &&
                !recommendations.books.length &&
                !recommendations.podcasts.length &&
                !recommendations.articles.length &&
                !recommendations.research.length &&
                !recommendations.news.length ? (
                <div className="h-full min-h-0 overflow-y-auto pr-1">
                <p className="text-gray-500 dark:text-gray-400 text-sm">
                  No cached suggestions yet. Click <strong className="font-medium text-gray-700 dark:text-gray-300">Refresh recommendations</strong> to
                  generate a new set (heavy; runs only when you ask).
                </p>
                      </div>
              ) : (
                <div className="grid h-full min-h-0 auto-rows-[minmax(0,1fr)] grid-cols-1 gap-4 md:grid-cols-2 md:gap-5 xl:grid-cols-5 xl:gap-4">
                  {/* Books */}
                  <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-gray-100 bg-white p-4 shadow-sm dark:rounded-xl dark:border-gray-700 dark:bg-[#2f2f2f]">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                      <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">Books</h3>
                      <button
                        type="button"
                        className={recColumnBtnClass}
                        disabled={recommendationsLoading || recColumnLoading.books}
                        onClick={() => refreshRecommendationsColumn("books")}
                      >
                        {recColumnLoading.books ? "…" : "Refresh"}
                      </button>
                    </div>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.books.length === 0 ? (
                        <p className="text-gray-500 dark:text-gray-400 text-xs">No book suggestions right now.</p>
                      ) : (
                        recommendations.books.map((item, i) => {
                          const cardKey = `book:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`book-${i}-${item.title}`}
                              className={`rounded-lg bg-white dark:bg-[#343541] p-3 border border-gray-200 dark:border-gray-600 shadow-sm transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              <a
                                href={
                                  item.url && item.url.startsWith("http")
                                    ? item.url
                                    : `https://www.amazon.com/s?k=${encodeURIComponent([item.title, item.author].filter(Boolean).join(" "))}`
                                }
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-900 dark:text-gray-100 text-sm font-medium hover:text-[#10a37f] dark:hover:text-emerald-400 hover:underline cursor-pointer"
                              >
                                {item.title}
                              </a>
                              {item.author && <p className="text-gray-500 dark:text-gray-400 text-xs mt-0.5">{item.author}</p>}
                              {item.reason && <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">{item.reason}</p>}
                              <RecFeedbackLinks category="book" item={item} />
                              <button
                                type="button"
                                onClick={() => markConsumed("book", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-[#10a37f] hover:bg-[#0d8c6e] disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
                              >
                                {consumedIds.has(cardKey) ? "Marked as read" : "I've read this"}
                              </button>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </section>
                  {/* Podcasts */}
                  <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-gray-100 bg-white p-4 shadow-sm dark:rounded-xl dark:border-gray-700 dark:bg-[#2f2f2f]">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                      <div className="flex flex-wrap items-baseline gap-2 min-w-0">
                        <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">Podcasts</h3>
                      <a
                        href="https://www.listennotes.com/"
                        target="_blank"
                        rel="noopener noreferrer"
                          className="text-[10px] text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors flex items-baseline gap-0.5"
                        title="Podcast data by Listen Notes"
                      >
                        <span className="lowercase font-normal">powered by</span>
                          <span className="uppercase font-semibold text-gray-500 dark:text-gray-400">LISTEN NOTES</span>
                        </a>
                      </div>
                      <button
                        type="button"
                        className={recColumnBtnClass}
                        disabled={recommendationsLoading || recColumnLoading.podcasts}
                        onClick={() => refreshRecommendationsColumn("podcasts")}
                      >
                        {recColumnLoading.podcasts ? "…" : "Refresh"}
                      </button>
                    </div>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.podcasts.length === 0 ? (
                        <p className="text-gray-500 dark:text-gray-400 text-xs">No podcast suggestions right now.</p>
                      ) : (
                        recommendations.podcasts.map((item, i) => {
                          const cardKey = `podcast:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`podcast-${i}-${item.title}`}
                              className={`rounded-lg bg-white dark:bg-[#343541] p-3 border border-gray-200 dark:border-gray-600 shadow-sm transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              {item.author && (
                                <p className="text-gray-500 dark:text-gray-400 text-xs font-medium">{item.author}</p>
                              )}
                              <a
                                href={
                                  item.url && (item.url.includes("spotify.com") || item.url.includes("podcasts.apple.com") || item.url.includes("listennotes.com"))
                                    ? item.url
                                    : `https://open.spotify.com/search/${encodeURIComponent([item.author, item.title].filter(Boolean).join(" "))}`
                                }
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-900 dark:text-gray-100 text-sm font-medium hover:text-[#10a37f] dark:hover:text-emerald-400 hover:underline cursor-pointer block"
                              >
                                {item.title}
                              </a>
                              {(!item.url || (!item.url.includes("spotify.com") && !item.url.includes("podcasts.apple.com") && !item.url.includes("listennotes.com"))) && (
                                <p className="text-xs mt-0.5 flex gap-2 flex-wrap">
                                  <a
                                    href={`https://open.spotify.com/search/${encodeURIComponent([item.author, item.title].filter(Boolean).join(" "))}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-[#10a37f] hover:underline dark:text-emerald-400"
                                  >
                                    Spotify
                                  </a>
                                  <a
                                    href={`https://podcasts.apple.com/us/search?term=${encodeURIComponent([item.author, item.title].filter(Boolean).join(" "))}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-[#10a37f] hover:underline dark:text-emerald-400"
                                  >
                                    Apple Podcasts
                                  </a>
                                </p>
                              )}
                              {item.reason && <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">{item.reason}</p>}
                              <RecFeedbackLinks category="podcast" item={item} />
                              <button
                                type="button"
                                onClick={() => markConsumed("podcast", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-[#10a37f] hover:bg-[#0d8c6e] disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
                              >
                                {consumedIds.has(cardKey) ? "Marked as listened" : "I've listened to this"}
                              </button>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </section>
                  {/* Articles */}
                  <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-gray-100 bg-white p-4 shadow-sm dark:rounded-xl dark:border-gray-700 dark:bg-[#2f2f2f]">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                      <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">Articles</h3>
                      <button
                        type="button"
                        className={recColumnBtnClass}
                        disabled={recommendationsLoading || recColumnLoading.articles}
                        onClick={() => refreshRecommendationsColumn("articles")}
                      >
                        {recColumnLoading.articles ? "…" : "Refresh"}
                      </button>
                    </div>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.articles.length === 0 ? (
                        <p className="text-gray-500 dark:text-gray-400 text-xs">No article suggestions right now.</p>
                      ) : (
                        recommendations.articles.map((item, i) => {
                          const cardKey = `article:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`article-${i}-${item.title}`}
                              className={`rounded-lg bg-white dark:bg-[#343541] p-3 border border-gray-200 dark:border-gray-600 shadow-sm transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              <a
                                href={item.url && item.url.startsWith("http") ? item.url : `https://www.google.com/search?q=${encodeURIComponent([item.title, item.author].filter(Boolean).join(" "))}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-900 dark:text-gray-100 text-sm font-medium hover:text-[#10a37f] dark:hover:text-emerald-400 hover:underline cursor-pointer"
                              >
                                {item.title}
                              </a>
                              {item.author && <p className="text-gray-500 dark:text-gray-400 text-xs mt-0.5">{item.author}</p>}
                              {item.reason && <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">{item.reason}</p>}
                              <RecFeedbackLinks category="article" item={item} />
                              <button
                                type="button"
                                onClick={() => markConsumed("article", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-[#10a37f] hover:bg-[#0d8c6e] disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
                              >
                                {consumedIds.has(cardKey) ? "Marked as read" : "I've read this"}
                              </button>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </section>
                  {/* Research papers */}
                  <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-gray-100 bg-white p-4 shadow-sm dark:rounded-xl dark:border-gray-700 dark:bg-[#2f2f2f]">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                      <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">Research papers</h3>
                      <button
                        type="button"
                        className={recColumnBtnClass}
                        disabled={recommendationsLoading || recColumnLoading.research}
                        onClick={() => refreshRecommendationsColumn("research")}
                      >
                        {recColumnLoading.research ? "…" : "Refresh"}
                      </button>
                    </div>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.research.length === 0 ? (
                        <p className="text-gray-500 dark:text-gray-400 text-xs">No research suggestions right now.</p>
                      ) : (
                        recommendations.research.map((item, i) => {
                          const cardKey = `research:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`research-${i}-${item.title}`}
                              className={`rounded-lg bg-white dark:bg-[#343541] p-3 border border-gray-200 dark:border-gray-600 shadow-sm transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              <a
                                href={item.url && item.url.startsWith("http") ? item.url : `https://www.google.com/search?q=${encodeURIComponent([item.title, item.author].filter(Boolean).join(" "))}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-900 dark:text-gray-100 text-sm font-medium hover:text-[#10a37f] dark:hover:text-emerald-400 hover:underline cursor-pointer"
                              >
                                {item.title}
                              </a>
                              {item.author && <p className="text-gray-500 dark:text-gray-400 text-xs mt-0.5">{item.author}</p>}
                              {item.reason && <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">{item.reason}</p>}
                              <RecFeedbackLinks category="research" item={item} />
                              <button
                                type="button"
                                onClick={() => markConsumed("research", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-[#10a37f] hover:bg-[#0d8c6e] disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
                              >
                                {consumedIds.has(cardKey) ? "Marked as read" : "I've read this"}
                              </button>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </section>
                  {/* News */}
                  <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-gray-100 bg-white p-4 shadow-sm dark:rounded-xl dark:border-gray-700 dark:bg-[#2f2f2f]">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                      <div className="flex flex-wrap items-baseline gap-2 min-w-0">
                        <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest">News</h3>
                        <span className="text-[10px] text-gray-400 dark:text-gray-500">Perplexity</span>
                      </div>
                      <button
                        type="button"
                        className={recColumnBtnClass}
                        disabled={recommendationsLoading || recColumnLoading.news}
                        onClick={() => refreshRecommendationsColumn("news")}
                      >
                        {recColumnLoading.news ? "…" : "Refresh"}
                      </button>
                    </div>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.news.length === 0 ? (
                        <p className="text-gray-500 dark:text-gray-400 text-xs">No news suggestions right now.</p>
                      ) : (
                        recommendations.news.map((item, i) => {
                          const cardKey = `news:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`news-${i}-${item.title}`}
                              className={`rounded-lg bg-white dark:bg-[#343541] p-3 border border-gray-200 dark:border-gray-600 shadow-sm transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              <a
                                href={
                                  item.url && item.url.startsWith("http")
                                    ? item.url
                                    : `https://www.google.com/search?q=${encodeURIComponent([item.title, item.author].filter(Boolean).join(" "))}`
                                }
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-900 dark:text-gray-100 text-sm font-medium hover:text-[#10a37f] dark:hover:text-emerald-400 hover:underline cursor-pointer"
                              >
                                {item.title}
                              </a>
                              {item.author && <p className="text-gray-500 dark:text-gray-400 text-xs mt-0.5">{item.author}</p>}
                              {item.reason && <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">{item.reason}</p>}
                              <RecFeedbackLinks category="news" item={item} />
                              <button
                                type="button"
                                onClick={() => markConsumed("news", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-[#10a37f] hover:bg-[#0d8c6e] disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
                              >
                                {consumedIds.has(cardKey) ? "Marked as read" : "I've read this"}
                              </button>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </section>
                </div>
              )}
              </div>
              {/* Library interview modal */}
              {showLibraryInterview && (
                <div
                  className="fixed inset-0 z-50 flex items-center justify-center p-4"
                  role="dialog"
                  aria-modal="true"
                  aria-labelledby="library-interview-title"
                >
                  <div
                    className="absolute inset-0 bg-black/60"
                    onClick={() => setShowLibraryInterview(false)}
                  />
                  <div className="relative bg-white border border-gray-200 rounded-xl shadow-xl max-w-lg w-full max-h-[85vh] flex flex-col dark:bg-[#2f2f2f] dark:border-gray-700">
                    <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between gap-2 flex-shrink-0">
                      <h2 id="library-interview-title" className="text-gray-900 dark:text-gray-100 font-medium text-lg">
                        Interview about your books
                      </h2>
                      <button
                        type="button"
                        onClick={() => setShowLibraryInterview(false)}
                        className="p-1.5 rounded-lg text-gray-500 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-gray-100 dark:hover:bg-[#404040] transition-colors"
                        aria-label="Close"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    <p className="px-4 pb-2 text-xs text-gray-500 dark:text-gray-400 flex-shrink-0">
                      Chat about what you liked (or didn’t) — we’ll save short notes to improve recommendations. No hallucination; notes are brief and factual.
                    </p>
                    <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
                      {libraryInterviewMessages.length === 0 && !libraryInterviewLoading && (
                        <p className="text-gray-500 dark:text-gray-400 text-sm">Say &quot;Start&quot; or &quot;Hi&quot; to begin. The agent will ask about your books one by one.</p>
                      )}
                      {libraryInterviewMessages.map((m, i) => (
                        <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                          <div
                            className={`max-w-[90%] rounded-lg px-3 py-2 text-sm ${
                              m.role === "user"
                                ? "bg-[#10a37f]/15 text-gray-900 dark:text-gray-100"
                                : "bg-gray-100 text-gray-800 border border-gray-200 dark:bg-[#343541] dark:text-gray-200 dark:border-gray-600"
                            }`}
                          >
                            <span className="font-medium text-gray-500 dark:text-gray-400 text-xs block mb-0.5">{m.role === "user" ? "You" : "Agent"}</span>
                            {m.content}
                          </div>
                        </div>
                      ))}
                      {libraryInterviewLoading && (
                        <div className="flex justify-start">
                          <div className="rounded-lg px-3 py-2 text-sm bg-gray-100 text-gray-500 border border-gray-200 dark:bg-[#343541] dark:text-gray-400 dark:border-gray-600">
                            Thinking…
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="p-4 border-t border-gray-200 dark:border-gray-700 flex gap-2 flex-shrink-0">
                      <input
                        type="text"
                        value={libraryInterviewInput}
                        onChange={(e) => setLibraryInterviewInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            const msg = libraryInterviewInput.trim();
                            if (msg && !libraryInterviewLoading) sendLibraryInterviewMessage(msg);
                          }
                        }}
                        placeholder="Type your reply…"
                        className="flex-1 min-w-0 px-3 py-2 rounded-lg bg-white border border-gray-200 text-gray-900 text-sm placeholder-gray-400 dark:bg-[#343541] dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#10a37f]/40"
                        disabled={libraryInterviewLoading}
                      />
                      <button
                        type="button"
                        onClick={() => {
                          const msg = libraryInterviewInput.trim();
                          if (msg && !libraryInterviewLoading) sendLibraryInterviewMessage(msg);
                        }}
                        disabled={libraryInterviewLoading || !libraryInterviewInput.trim()}
                        className="px-4 py-2 rounded-lg bg-[#10a37f] hover:bg-[#0d8c6e] text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-[#10a37f]/40"
                      >
                        Send
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          {view === "learning" && <LearningTab onToast={chatToast} />}
        </div>
      </main>

      {/* Footer — mobile Home uses a fixed dock when the conversation is active; Chat uses the inline composer */}
      <footer className="pointer-events-none relative z-10 flex-shrink-0 px-4 pb-[calc(4.5rem+env(safe-area-inset-bottom))] pt-1 text-center md:pb-5">
        <p className="pointer-events-auto text-xs text-white/60">
          On Home (desktop), open or resize the chat column on the right for history—the same width rules as Open WebUI. Record or attach audio for journal entries.
        </p>
        <p className="pointer-events-auto mx-auto mt-1 max-w-xl text-[10px] leading-relaxed text-white/40">
          This is a prototype. Please avoid sharing highly sensitive personal information until our data pipeline is more secure. For private or stress testing, run the app locally and use local LLMs.
        </p>
        <div className="pointer-events-auto pt-1">
          <p className="flex flex-wrap items-center justify-center gap-2 text-[10px] text-white/40">
            <a
              href="https://github.com/MrFunnything99/Open-Journal"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-white/50 transition-colors hover:text-white/80"
              aria-label="View on GitHub"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 16 16" fill="currentColor" className="inline-block">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
              GitHub
            </a>
          </p>
        </div>
      </footer>

      <MobileAskComposerDockGate railOpen={mobileRailOpen} activeView={view} />
      </div>

      <HomeChatSidebar active={view === "voice_memo"} />
      </div>

      {libraryEditingId ? (() => {
        const editingEntry = [...libraryItems.books, ...libraryItems.podcasts, ...libraryItems.articles, ...libraryItems.research].find((e) => e.id === libraryEditingId);
        return (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            role="dialog"
            aria-modal="true"
            aria-labelledby="library-modal-title"
          >
            <div
              className="absolute inset-0 bg-black/60"
              onClick={() => {
                setLibraryEditingId(null);
                setLibraryEditDate("");
                setLibraryEditNote("");
              }}
            />
            <div className="relative bg-white border border-gray-200 rounded-xl shadow-xl max-w-lg w-full max-h-[90vh] flex flex-col dark:bg-[#2f2f2f] dark:border-gray-700">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <h2 id="library-modal-title" className="text-gray-900 dark:text-gray-100 font-medium text-lg truncate">
                  {editingEntry?.title ?? "Library item"}
                </h2>
              </div>
              <div className="p-4 flex-1 min-h-0 overflow-y-auto flex flex-col gap-4">
                <div>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">Notes</label>
                  <textarea
                    value={libraryEditNote}
                    onChange={(e) => setLibraryEditNote(e.target.value)}
                    rows={10}
                    placeholder="How you felt, what stood out — used for better recommendations."
                    className="w-full px-3 py-2.5 rounded-lg bg-white border border-gray-200 text-gray-900 text-sm placeholder-gray-400 dark:bg-[#343541] dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#10a37f]/40 focus:border-[#10a37f] resize-y min-h-[200px]"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">Date completed</label>
                  <input
                    type="text"
                    value={libraryEditDate}
                    onChange={(e) => setLibraryEditDate(e.target.value)}
                    placeholder="Year e.g. 2024 or 2024-06"
                    className="w-full px-3 py-2 rounded-lg bg-white border border-gray-200 text-gray-900 text-sm placeholder-gray-400 dark:bg-[#343541] dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#10a37f]/40 focus:border-[#10a37f]"
                  />
                </div>
              </div>
              <div className="p-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between gap-4">
                <button
                  type="button"
                  onClick={() => {
                    if (!libraryEditingId || libraryUpdateSaving) return;
                    if (!confirm("Remove from library and mark as unread? It may be recommended again.")) return;
                    setLibraryUpdateSaving(true);
                    backendFetch(`/library/${encodeURIComponent(libraryEditingId)}`, { method: "DELETE" })
                      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Delete failed"))))
                      .then((data: { ok?: boolean }) => {
                        if (data?.ok !== false) {
                          fetchLibrary();
                          setLibraryEditingId(null);
                          setLibraryEditDate("");
                          setLibraryEditNote("");
                          setToastMessage("Removed.");
                          setTimeout(() => setToastMessage(null), 2000);
                        }
                      })
                      .catch(() => {
                        setToastMessage("Failed to remove.");
                        setTimeout(() => setToastMessage(null), 3000);
                      })
                      .finally(() => setLibraryUpdateSaving(false));
                  }}
                  disabled={libraryUpdateSaving}
                  className="text-red-400 hover:text-red-300 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-red-500/50 rounded px-2 py-1 disabled:opacity-50"
                >
                  Delete / Mark unread
                </button>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setLibraryEditingId(null);
                      setLibraryEditDate("");
                      setLibraryEditNote("");
                    }}
                    className="px-4 py-2 rounded-lg bg-gray-100 text-gray-800 text-sm font-medium hover:bg-gray-200 dark:bg-[#404040] dark:text-gray-200 dark:hover:bg-[#505050] border border-gray-200 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-400/50"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (!libraryEditingId || libraryUpdateSaving) return;
                      setLibraryUpdateSaving(true);
                      backendFetch(`/library/${encodeURIComponent(libraryEditingId)}`, {
                        method: "PATCH",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          date_completed: libraryEditDate.trim(),
                          note: libraryEditNote.trim(),
                        }),
                      })
                        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Update failed"))))
                        .then((data: { ok?: boolean }) => {
                          if (data?.ok !== false) {
                            fetchLibrary();
                            setLibraryEditingId(null);
                            setToastMessage("Saved.");
                            setTimeout(() => setToastMessage(null), 2000);
                          }
                        })
                        .catch(() => {
                          setToastMessage("Failed to save.");
                          setTimeout(() => setToastMessage(null), 3000);
                        })
                        .finally(() => setLibraryUpdateSaving(false));
                    }}
                    disabled={libraryUpdateSaving}
                    className="px-4 py-2 rounded-lg bg-teal-600 text-white text-sm font-medium hover:bg-teal-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 disabled:opacity-50"
                  >
                    {libraryUpdateSaving ? "Saving…" : "Save"}
                  </button>
                </div>
              </div>
            </div>
    </div>
        );
      })() : null}

    </div>
    </PersonaplexChatProvider>
  );
};
