import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { backendFetch } from "../../backendApi";
import { usePersonaplexSession, type TranscriptEntry } from "./hooks/usePersonaplexSession";
import { SessionSidePanel } from "./components/SessionSidePanel";
import { useTheme } from "../../hooks/useTheme";
import { ThemeToggle } from "../../components/ThemeToggle";
import { parseKnowledgeBaseFile, useJournalHistory } from "./hooks/useJournalHistory";
import { buildKnowledgeBaseMarkdownZip, parseKnowledgeBaseMarkdownZip } from "./knowledgeBaseMarkdownZip";
import { type OrbState } from "./components/Orb";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { BrainLayout, type BrainLibraryCategory } from "./components/BrainLayout";
import { VoiceMemoTab } from "./components/VoiceMemoTab";
import { BrainCalendarPanel } from "./components/BrainCalendarPanel";

type VoiceOption = { voice_id: string; name: string };

/** Default journaling assistant prompt (base; personalization is always \"high\" / memory-connected) */
const DEFAULT_PERSONAPLEX_PROMPT = `You are an empathetic and insightful conversational journaling assistant. Your goal is to provide a supportive space for the user to reflect on their thoughts, experiences, and emotions. Read the user's entries and respond naturally. Ask open-ended questions to encourage further exploration, but always let the user guide the direction and depth of the conversation. Avoid being overly prescriptive, giving unsolicited advice, or summarizing their thoughts unnecessarily. Just be a curious, active listener. Always facilitate conversation that gets the user exploring their thoughts and emotions. Try to keep responses brief and concise when possible to conserve tokens.`;

const INTRUSIVENESS_LABELS: Record<number, string> = {
  0: "Context building only; follow the user's lead; gather and reflect back; avoid probing or emotional questions.",
  0.25: "Mostly context building; ask sparingly and only to clarify or expand.",
  0.5: "Balanced; mix context-building with occasional reflective questions.",
  0.75: "More dynamic questions; ask how things made them feel, what they noticed, etc., when it fits.",
  1: "Dynamic questions; actively ask \"how did this make you feel?\", \"what was that like?\", and other reflective, feeling-focused questions to deepen exploration.",
};

const DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // Rachel
const RECOMMENDATIONS_CACHE_KEY = "openjournal-recommendations-cache";
const LIBRARY_CACHE_KEY = "openjournal-library-cache";

/** Fallback when /api/voices is unavailable (e.g. API server not running) */
const FALLBACK_VOICES: VoiceOption[] = [
  { voice_id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel" },
  { voice_id: "pNInz6obpgDQGcFmaJgB", name: "Adam" },
  { voice_id: "EXAVITQu4vr4xnSDxMaL", name: "Bella" },
  { voice_id: "ErXwobaYiN019PkySvjV", name: "Antoni" },
  { voice_id: "MF3mGyEYCl7XYWbV9V6O", name: "Elli" },
  { voice_id: "TxGEqnHWrfWFTfGW9XjX", name: "Josh" },
  { voice_id: "VR6AewLTigWG4xSOukaG", name: "Arnold" },
  { voice_id: "onwK4e9ZLuTAKqWW03F9", name: "Domi" },
  { voice_id: "N2lVS1w4EtoT3dr4eOWO", name: "Sam" },
];

type PersonaplexView = "voice_memo" | "brain" | "recommendations" | "journal";

/** Glass nav — active item solid white pill; inactive muted on glass */
function PersonaplexNavButtons({
  view,
  setView,
}: {
  view: PersonaplexView;
  setView: Dispatch<SetStateAction<PersonaplexView>>;
}) {
  const base =
    "inline-flex items-center justify-center rounded-full text-sm font-medium transition-colors gap-1.5 " +
    "min-h-[44px] min-w-[44px] px-2 sm:px-2.5 md:min-h-0 md:min-w-0 md:px-3.5 md:py-2";
  const active = "bg-white text-gray-900 shadow-sm";
  const inactive = "text-white/60 hover:bg-white/10 hover:text-white";
  return (
    <>
      <button
        type="button"
        onClick={() => setView("voice_memo")}
        className={`${base} ${view === "voice_memo" ? active : inactive}`}
        title="Home — chat, dictate, annotate or hear replies"
        aria-label="Home"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 md:h-4 md:w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
        <span className="hidden sm:inline">Home</span>
      </button>
      <button
        type="button"
        onClick={() => setView("brain")}
        className={`${base} ${view === "brain" ? active : inactive}`}
        title="The Brain — journals and library"
        aria-label="The Brain"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 md:h-4 md:w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
        <span className="hidden sm:inline">The Brain</span>
      </button>
      <button
        type="button"
        onClick={() => setView("recommendations")}
        className={`${base} ${view === "recommendations" ? active : inactive}`}
        title="Personalized recommendations"
        aria-label="Personalized recommendations"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 md:h-4 md:w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
        </svg>
        <span className="hidden sm:inline">Recommendations</span>
      </button>
      <button
        type="button"
        onClick={() => setView("journal")}
        className={`${base} ${view === "journal" ? active : inactive}`}
        title="Journal — reflect and chat"
        aria-label="Journal"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 md:h-4 md:w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <span className="hidden sm:inline">Journal</span>
      </button>
    </>
  );
}

export const Personaplex = () => {
  const { mode, toggle } = useTheme();
  // Personalization is always \"high\": the agent should actively connect past memories and journals to the current conversation when relevant.
  const personalization = 1;
  const intrusiveness = 0.5;
  const [sessionMode, setSessionMode] = useState<"journal" | "recommendations">("journal");
  const [voices, setVoices] = useState<VoiceOption[]>(FALLBACK_VOICES);
  const [selectedVoiceId, setSelectedVoiceId] = useState(DEFAULT_VOICE_ID);
  // Fixed voice settings: speed 1.0, moderate stability to reduce ElevenLabs variability/errors.
  const voiceSettings = useMemo(
    () => ({
      stability: 0.4,
      similarity_boost: 0.75,
      style: 0.4,
      speed: 1.0,
    }),
    []
  );
  const textPrompt = useMemo(
    () =>
      DEFAULT_PERSONAPLEX_PROMPT +
      "\n\nPersonalization: Always look for relevant connections between today's journaling and the user's past entries and memories. When appropriate, bring prior sessions or stories back into the conversation to deepen reflection (for example, if the user asks what to journal about today, suggest follow-ups based on past sessions)." +
      "\n\nQuestioning style (intrusiveness): " +
      Math.round(intrusiveness * 100) +
      "%. " +
      (INTRUSIVENESS_LABELS[intrusiveness as keyof typeof INTRUSIVENESS_LABELS] ?? INTRUSIVENESS_LABELS[0.5]),
    [intrusiveness]
  );
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [expandedLogIndex, setExpandedLogIndex] = useState<number | null>(null);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [view, setView] = useState<PersonaplexView>("voice_memo");
  const [sessionPanelOpen, setSessionPanelOpen] = useState(false);
  const [sessionSettingsExpanded, setSessionSettingsExpanded] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [showLiveTranscription, setShowLiveTranscription] = useState(true);
  const [inputMode, setInputMode] = useState<"voice" | "text">("voice");
  const [typedInput, setTypedInput] = useState("");
  type RecItem = { title: string; author?: string; reason?: string; url?: string };
  const [recommendations, setRecommendations] = useState<{ books: RecItem[]; podcasts: RecItem[]; articles: RecItem[]; research: RecItem[] }>({ books: [], podcasts: [], articles: [], research: [] });
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const recommendationsInFlightRef = useRef(false);
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
    saveOrUpdateEntry,
    markEntrySynced,
    syncUnsyncedEntries,
    deleteEntry,
    updateJournalEntry,
    getFormattedDate,
    importEntriesFromExport,
  } = useJournalHistory();

  const handleDownloadKnowledgeBase = useCallback(async () => {
    try {
      const blob = await buildKnowledgeBaseMarkdownZip(entries, libraryItems, getFormattedDate);
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
  }, [entries, libraryItems, getFormattedDate]);

  const mergeImportedLibrary = useCallback(
    (lib: {
      books: (typeof libraryItems.books)[number][];
      podcasts: (typeof libraryItems.podcasts)[number][];
      articles: (typeof libraryItems.articles)[number][];
      research: (typeof libraryItems.research)[number][];
    }) => {
      const remap = (items: (typeof libraryItems.books)[number][]) =>
        items.map((item, i) => ({
          ...item,
          id: `lib-${Date.now()}-${i}-${Math.random().toString(36).slice(2, 9)}`,
        }));
      setLibraryItems((prev) => {
        const next = {
          books: [...prev.books, ...remap(lib.books)],
          podcasts: [...prev.podcasts, ...remap(lib.podcasts)],
          articles: [...prev.articles, ...remap(lib.articles)],
          research: [...prev.research, ...remap(lib.research)],
        };
        try {
          localStorage.setItem(LIBRARY_CACHE_KEY, JSON.stringify(next));
        } catch {
          /* ignore */
        }
        return next;
      });
    },
    []
  );

  const handleImportKnowledgeBaseFile = useCallback(
    async (file: File) => {
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
          const nEntries = importEntriesFromExport({
            version: 1,
            exportedAt: new Date().toISOString(),
            entries: parsed.entries,
          });
          mergeImportedLibrary(parsed.library);
          const nLib =
            parsed.library.books.length +
            parsed.library.podcasts.length +
            parsed.library.articles.length +
            parsed.library.research.length;
          setToastMessage(
            `Imported ${nEntries} entr${nEntries === 1 ? "y" : "ies"} and ${nLib} library item${nLib === 1 ? "" : "s"} from Markdown.`
          );
          setTimeout(() => setToastMessage(null), 5000);
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
          const n = importEntriesFromExport(parsed.data);
          setToastMessage(
            n > 0 ? `Imported ${n} journal entr${n === 1 ? "y" : "ies"} from JSON.` : "No entries found in that file."
          );
          setTimeout(() => setToastMessage(null), 4000);
          return;
        }
        const nEntries = importEntriesFromExport({
          version: parsed.data.version,
          exportedAt: parsed.data.exportedAt,
          entries: parsed.data.entries,
        });
        mergeImportedLibrary(parsed.data.library);
        const nLib =
          parsed.data.library.books.length +
          parsed.data.library.podcasts.length +
          parsed.data.library.articles.length +
          parsed.data.library.research.length;
        setToastMessage(
          `Imported ${nEntries} journal entr${nEntries === 1 ? "y" : "ies"} and ${nLib} library item${nLib === 1 ? "" : "s"} from JSON.`
        );
        setTimeout(() => setToastMessage(null), 5000);
      } catch {
        setToastMessage("Could not read that file.");
        setTimeout(() => setToastMessage(null), 4000);
      }
    },
    [importEntriesFromExport, mergeImportedLibrary]
  );

  const handleImportJournalDumpFolder = useCallback(
    async (files: FileList) => {
      const list = Array.from(files).filter((f) => /\.(md|txt)$/i.test(f.name));
      if (list.length === 0) {
        setToastMessage("No .md or .txt files found in that folder.");
        setTimeout(() => setToastMessage(null), 3500);
        return;
      }

      const parseDateFromFile = (f: File): string | null => {
        const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath ?? f.name;
        const m = rel.match(/(20\d{2})-(\d{2})-(\d{2})/);
        if (m) {
          const iso = `${m[1]}-${m[2]}-${m[3]}T12:00:00.000Z`;
          if (!Number.isNaN(Date.parse(iso))) return iso;
        }
        const m2 = rel.match(/(20\d{2})-(\d{2})/);
        if (m2) {
          const iso = `${m2[1]}-${m2[2]}-01T12:00:00.000Z`;
          if (!Number.isNaN(Date.parse(iso))) return iso;
        }
        return null;
      };

      const dated = await Promise.all(
        list.map(async (f) => ({
          file: f,
          date: parseDateFromFile(f),
          text: (await f.text()).trim(),
        }))
      );

      const valid = dated
        .filter((x) => x.text.length > 0)
        .sort((a, b) => {
          const ta = a.date ? Date.parse(a.date) : 0;
          const tb = b.date ? Date.parse(b.date) : 0;
          return ta - tb;
        });

      let imported = 0;
      for (const row of valid) {
        const transcript = [{ role: "user" as const, text: row.text }];
        const id = saveEntry(transcript, row.date ?? undefined);
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

  useEffect(() => {
    fetch("/api/voices")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Not found"))))
      .then((data: { voices?: VoiceOption[] }) => {
        const list = data.voices ?? [];
        if (list.length > 0) {
          const hasRachel = list.some((v) => v.voice_id === DEFAULT_VOICE_ID);
          const listWithDefault = hasRachel
            ? list
            : [{ voice_id: DEFAULT_VOICE_ID, name: "Rachel" }, ...list];
          const sorted = [...listWithDefault].sort((a, b) => {
            if (a.voice_id === DEFAULT_VOICE_ID) return -1;
            if (b.voice_id === DEFAULT_VOICE_ID) return 1;
            return 0;
          });
          setVoices(sorted);
        }
      })
      .catch(() => {
        /* Keep FALLBACK_VOICES from initial state */
      });
  }, []);

  const hasRunInitialSyncRef = useRef(false);
  useEffect(() => {
    if (entries.length === 0 || hasRunInitialSyncRef.current) return;
    hasRunInitialSyncRef.current = true;
    syncUnsyncedEntries();
  }, [entries.length, syncUnsyncedEntries]);

  const fetchRecommendations = useCallback((showLoadingUnlessCached = false, retryCount = 0) => {
    if (recommendationsInFlightRef.current) return;
    const doFetch = (isRetry: boolean) => {
      const ac = new AbortController();
      const timeoutId = setTimeout(() => ac.abort(), 125000);
      recommendationsInFlightRef.current = true;
      backendFetch("/recommendations", { signal: ac.signal })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed to load"))))
        .then((data: { books?: RecItem[]; podcasts?: RecItem[]; articles?: RecItem[]; research?: RecItem[] }) => {
          const next = {
            books: data.books ?? [],
            podcasts: data.podcasts ?? [],
            articles: data.articles ?? [],
            research: data.research ?? [],
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
            setTimeout(() => fetchRecommendations(showLoadingUnlessCached, 1), 2500);
            return;
          }
          setRecommendations({ books: [], podcasts: [], articles: [], research: [] });
        })
        .finally(() => {
          clearTimeout(timeoutId);
          recommendationsInFlightRef.current = false;
          setRecommendationsLoading(false);
        });
    };

    try {
      const cached = localStorage.getItem(RECOMMENDATIONS_CACHE_KEY);
      if (cached && showLoadingUnlessCached) {
        const parsed = JSON.parse(cached) as { books?: RecItem[]; podcasts?: RecItem[]; articles?: RecItem[]; research?: RecItem[] };
        if (parsed && (parsed.books?.length || parsed.podcasts?.length || parsed.articles?.length || parsed.research?.length)) {
          setRecommendations({
            books: parsed.books ?? [],
            podcasts: parsed.podcasts ?? [],
            articles: parsed.articles ?? [],
            research: parsed.research ?? [],
          });
          setRecommendationsLoading(true);
          doFetch(retryCount >= 1);
          return;
        }
      }
    } catch {
      /* ignore cache parse errors */
    }
    setRecommendationsLoading(true);
    doFetch(retryCount >= 1);
  }, []);

  useEffect(() => {
    if (view !== "recommendations") return;
    syncUnsyncedEntries()
      .then((n) => {
        if (n > 0) {
          try {
            localStorage.removeItem(RECOMMENDATIONS_CACHE_KEY);
          } catch {
            /* ignore */
          }
        }
      })
      .then(() => {
        try {
          const cached = localStorage.getItem(RECOMMENDATIONS_CACHE_KEY);
          if (cached) {
            const parsed = JSON.parse(cached) as { books?: RecItem[]; podcasts?: RecItem[]; articles?: RecItem[]; research?: RecItem[] };
            if (parsed && Array.isArray(parsed.books) && Array.isArray(parsed.podcasts) && Array.isArray(parsed.articles)) {
              setRecommendations({
                books: parsed.books ?? [],
                podcasts: parsed.podcasts ?? [],
                articles: parsed.articles ?? [],
                research: Array.isArray(parsed.research) ? parsed.research : [],
              });
            }
          }
        } catch {
          /* ignore */
        }
        fetchRecommendations(true);
      })
      .catch(() => fetchRecommendations(true));
  }, [view, syncUnsyncedEntries, fetchRecommendations]);

  useEffect(() => {
    if (view === "recommendations" && (recommendations.books.length > 0 || recommendations.podcasts.length > 0 || recommendations.articles.length > 0 || recommendations.research.length > 0)) {
      try {
        localStorage.setItem(RECOMMENDATIONS_CACHE_KEY, JSON.stringify(recommendations));
      } catch {
        /* ignore */
      }
    }
  }, [view, recommendations]);

  const markConsumed = useCallback(
    (type: "book" | "podcast" | "article" | "research", item: RecItem) => {
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
          setToastMessage(type === "book" || type === "article" || type === "research" ? "Marked as read." : type === "podcast" ? "Marked as listened." : "Marked as read.");
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

  const {
    status,
    errorMessage,
    isProcessing,
    connect,
    disconnect,
    commitManual,
    submitTextTurn,
    cancelUserCapture,
    resumeVoiceCapture,
    isConnected,
    isUserSpeaking,
    isAiSpeaking,
    isVoiceMemoMode,
    isVoiceMemoRecording,
    startVoiceMemoRecording,
    stopVoiceMemoRecording,
    lastPlaybackFailed,
    playLastFailedPlayback,
  } = usePersonaplexSession({
    systemPrompt: textPrompt,
    selectedVoiceId,
    manualMode: true,
    personalization,
    intrusiveness,
    sessionMode,
    voiceSettings,
    onTranscriptUpdate: useCallback((updater) => {
      setTranscript((prev) => {
        const next = typeof updater === "function" ? updater(prev) : updater;
        return next;
      });
    }, []),
    onInterimTranscript: setInterimTranscript,
    onNotesSaved: useCallback(
      (notes: { item_id: string; note: string }[]) => {
        fetchLibrary();
        if (notes.length) {
          setToastMessage(notes.length === 1 ? "Note saved." : `Saved ${notes.length} notes.`);
          setTimeout(() => setToastMessage(null), 3000);
        }
      },
      [fetchLibrary]
    ),
    showLiveTranscription,
    allowVoiceCapture: inputMode === "voice",
  });

  const transcriptScrollRef = useRef<HTMLDivElement>(null);
  const autoScrollEnabledRef = useRef(true);

  const orbState: OrbState = useMemo(() => {
    if (isUserSpeaking) return "userSpeaking";
    if (isAiSpeaking) return "aiSpeaking";
    if (isProcessing) return "aiThinking";
    return "idle";
  }, [isUserSpeaking, isAiSpeaking, isProcessing]);

  const [thinkingProgress, setThinkingProgress] = useState(0);
  const thinkingStartRef = useRef<number | null>(null);
  const thinkingRafRef = useRef<number | null>(null);
  const activeSessionEntryIdRef = useRef<string | null>(null);
  const lastAutoSavedSignatureRef = useRef("");

  useEffect(() => {
    if (!isProcessing) {
      setThinkingProgress(0);
      thinkingStartRef.current = null;
      if (thinkingRafRef.current != null) {
        cancelAnimationFrame(thinkingRafRef.current);
        thinkingRafRef.current = null;
      }
      return;
    }
    thinkingStartRef.current = Date.now();
    const durationMs = 12000;

    const tick = () => {
      const start = thinkingStartRef.current;
      if (start == null) return;
      const elapsed = Date.now() - start;
      const progress = Math.min(1, elapsed / durationMs);
      setThinkingProgress(progress);
      if (progress < 1) thinkingRafRef.current = requestAnimationFrame(tick);
    };
    thinkingRafRef.current = requestAnimationFrame(tick);
    return () => {
      if (thinkingRafRef.current != null) cancelAnimationFrame(thinkingRafRef.current);
    };
  }, [isProcessing]);

  const handleConnect = useCallback(() => {
    activeSessionEntryIdRef.current = null;
    lastAutoSavedSignatureRef.current = "";
    connect();
  }, [connect]);

  // Autosave after every completed AI turn so abrupt disconnects still preserve history.
  useEffect(() => {
    if (transcript.length === 0) return;
    const last = transcript[transcript.length - 1];
    if (!last || last.role !== "ai") return;
    const signature = `${transcript.length}|${last.text}`;
    if (signature === lastAutoSavedSignatureRef.current) return;
    lastAutoSavedSignatureRef.current = signature;
    const id = saveOrUpdateEntry(activeSessionEntryIdRef.current, transcript);
    if (id) activeSessionEntryIdRef.current = id;
  }, [transcript, saveOrUpdateEntry]);

  const handleDisconnect = useCallback(() => {
    if (transcript.length > 0) {
      const id = saveOrUpdateEntry(activeSessionEntryIdRef.current, transcript);
      if (id) activeSessionEntryIdRef.current = id;
      const transcriptText = transcript
        .map((e) => (e.role === "user" ? "User: " + e.text : "Assistant: " + e.text))
        .join("\n\n");
      backendFetch("/ingest-history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: transcriptText }),
      })
        .then(async (r) => {
          const data = r.ok ? await r.json().catch(() => ({})) : { ok: false };
          if (data?.ok !== false && id) markEntrySynced(id);
        })
        .catch(() => {});
      setToastMessage("Journal entry saved and synced to memory.");
      setTimeout(() => setToastMessage(null), 3000);
    }
    activeSessionEntryIdRef.current = null;
    lastAutoSavedSignatureRef.current = "";
    setTranscript([]);
    setInterimTranscript("");
    disconnect();
  }, [disconnect, transcript, saveOrUpdateEntry, markEntrySynced]);

  const handleSendTypedInput = useCallback(() => {
    const sent = submitTextTurn(typedInput);
    if (sent) setTypedInput("");
  }, [submitTextTurn, typedInput]);

  const handleInputModeChange = useCallback((mode: "voice" | "text") => {
    setInputMode(mode);
  }, []);
  const isModeToggleLocked = isAiSpeaking || isProcessing;

  const previousInputModeRef = useRef<"voice" | "text">(inputMode);
  useEffect(() => {
    if (!isConnected) {
      previousInputModeRef.current = inputMode;
      return;
    }
    if (previousInputModeRef.current === inputMode) return;
    if (inputMode === "text") cancelUserCapture();
    else resumeVoiceCapture();
    previousInputModeRef.current = inputMode;
  }, [inputMode, isConnected, cancelUserCapture, resumeVoiceCapture]);

  useEffect(() => {
    if (!isConnected) return;
    if (inputMode !== "voice") return;
    if (isAiSpeaking || isProcessing) return;
    // If user switched to voice while AI was still talking, start listening as soon as it's safe.
    resumeVoiceCapture();
  }, [inputMode, isConnected, isAiSpeaking, isProcessing, resumeVoiceCapture]);

  useEffect(() => {
    if (!isConnected) {
      setTranscript([]);
      setExpandedLogIndex(null);
      setInterimTranscript("");
    }
  }, [isConnected]);

  const handleTranscriptScroll = useCallback(() => {
    const el = transcriptScrollRef.current;
    if (!el) return;
    const isNearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 50;
    autoScrollEnabledRef.current = isNearBottom;
  }, []);

  useEffect(() => {
    const scrollEl = transcriptScrollRef.current;
    if (!scrollEl || !autoScrollEnabledRef.current) return;
    scrollEl.scrollTop = scrollEl.scrollHeight - scrollEl.clientHeight;
  }, [transcript, interimTranscript]);

  return (
    <div className="relative flex h-screen w-full flex-col overflow-hidden bg-[#0a0a12] font-sans text-white antialiased">
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

      {/* Header — glass pills */}
      <header className="relative z-20 flex-none px-4 py-3 sm:px-6 sm:py-4">
        {/* Mobile */}
        <div className="flex flex-col gap-3 md:hidden">
          <div className="flex items-center justify-between gap-3 min-w-0">
            <div className="glass-panel flex min-w-0 max-w-[55%] items-center gap-2 rounded-full px-3 py-2 text-white">
              <h1 className="truncate text-xs font-medium uppercase tracking-[0.2em] text-white sm:text-sm">Selfmeridian</h1>
            </div>
            <div className="flex items-center justify-end gap-2 shrink-0">
              <ThemeToggle mode={mode} onToggle={toggle} className="border-white/15 bg-white/10 text-white backdrop-blur-md hover:bg-white/15" />
              <div className="glass-panel inline-flex max-w-[min(100%,12rem)] flex-wrap justify-end gap-0.5 rounded-full p-1">
              <PersonaplexNavButtons view={view} setView={setView} />
            </div>
          </div>
          </div>
          <div className="glass-panel-subtle flex flex-wrap items-center gap-x-3 gap-y-2 rounded-full px-3 py-2">
            <ConnectionStatus status={status} className="shrink-0 text-sm text-white/70" />
            {errorMessage && (
              <span className="w-full basis-full text-sm text-red-300">{errorMessage}</span>
            )}
          </div>
        </div>

        {/* Desktop */}
        <div className="hidden md:flex md:items-center md:justify-between md:gap-4 md:w-full">
          <div className="glass-panel flex min-w-0 max-w-[55%] items-center gap-3 rounded-full px-4 py-2.5 sm:max-w-none">
            <h1 className="shrink-0 text-sm font-medium uppercase tracking-[0.25em] text-white sm:text-base">Selfmeridian</h1>
            <ConnectionStatus status={status} className="text-sm text-white/70" />
            {errorMessage && <span className="truncate text-sm text-red-300">{errorMessage}</span>}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <ThemeToggle mode={mode} onToggle={toggle} className="border-white/15 bg-white/10 text-white backdrop-blur-md hover:bg-white/15" />
            <div className="glass-panel inline-flex rounded-full p-1">
            <PersonaplexNavButtons view={view} setView={setView} />
            </div>
          </div>
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
      <main className="relative z-10 flex min-h-0 flex-1 flex-col">
        <div
          className={`flex-1 flex flex-col min-h-0 transition-opacity duration-300 ${
            view === "brain" || view === "voice_memo" || view === "journal" ? "overflow-hidden" : "overflow-y-auto"
          } opacity-100`}
        >
          {(view === "voice_memo" || view === "journal") && (
            <VoiceMemoTab
              onOpenSessionPanel={() => setSessionPanelOpen(true)}
              onToast={(msg) => {
                setToastMessage(msg);
                setTimeout(() => setToastMessage(null), 4000);
              }}
            />
          )}
          {view === "brain" && (
            <div className="flex min-h-0 flex-1 flex-row overflow-hidden">
              <aside
                className="flex w-[11.5rem] shrink-0 flex-col border-r border-white/10 bg-black/25 px-2 py-2 sm:w-52 sm:py-3"
                aria-label="Brain sections"
              >
                <p className="px-2 pb-2 text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-white/45">
                  The Brain
                </p>
                <nav className="flex flex-col gap-0.5">
                  <button
                    type="button"
                    onClick={() => setBrainSection("knowledgeBase")}
                    className={`rounded-2xl px-3 py-2.5 text-left text-sm font-medium transition-colors ${
                      brainSection === "knowledgeBase"
                        ? "bg-white/15 text-white shadow-sm"
                        : "text-white/65 hover:bg-white/10 hover:text-white"
                    }`}
                    title="Journal entries, conversation transcripts, and your media library"
                  >
                    Knowledge base
                  </button>
                  <button
                    type="button"
                    onClick={() => setBrainSection("calendar")}
                    className={`rounded-2xl px-3 py-2.5 text-left text-sm font-medium transition-colors ${
                      brainSection === "calendar"
                        ? "bg-white/15 text-white shadow-sm"
                        : "text-white/65 hover:bg-white/10 hover:text-white"
                    }`}
                    title="Month view and day summaries"
                  >
                    Calendar
                  </button>
                </nav>
              </aside>
              <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
              {brainSection === "knowledgeBase" ? (
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                <div className="flex-none border-b border-white/10 px-4 py-2.5">
                  <h2 className="text-sm font-medium text-white/90">Knowledge base</h2>
                  <p className="mt-0.5 text-xs text-white/50">
                    Journal entries, conversation transcripts, and your books &amp; media library.
                  </p>
                </div>
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
                  onImportJournalDumpFolder={handleImportJournalDumpFolder}
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
            </div>
          )}
          {view === "recommendations" && (
            <div className="flex-1 flex flex-col min-h-0 px-4 pt-4 pb-0 md:px-6 md:pt-6 md:pb-1 overflow-hidden">
              <div className="flex-shrink-0 flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-lg font-medium text-gray-800 dark:text-gray-200 uppercase tracking-wider">
                    Recommendations
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                Based on your journal memory and what you’ve already read or listened to. Mark items as read/listened so future suggestions get better.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => fetchRecommendations(false)}
                    disabled={recommendationsLoading}
                    className="px-3 py-2 rounded-lg bg-gray-100 text-gray-600 text-sm font-medium hover:bg-gray-200 dark:bg-[#404040] dark:text-gray-400 dark:hover:bg-[#505050] disabled:opacity-50 transition-colors"
                  >
                    {recommendationsLoading ? "Updating…" : "Refresh recommendations"}
                  </button>
                </div>
              </div>
              <div className="flex-1 min-h-0 overflow-auto">
              {recommendationsLoading && !recommendations.books.length && !recommendations.podcasts.length && !recommendations.articles.length && !recommendations.research.length ? (
                <p className="text-gray-500 dark:text-gray-400 text-sm">Loading recommendations… This can take up to a minute.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 pb-0">
                  {/* Books */}
                  <section className="rounded-2xl bg-white border border-gray-100 shadow-sm dark:rounded-xl dark:bg-[#2f2f2f] dark:border-gray-700 p-4 flex flex-col">
                    <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest mb-3">Books</h3>
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
                                href={`https://www.amazon.com/s?k=${encodeURIComponent([item.title, item.author].filter(Boolean).join(" "))}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-900 dark:text-gray-100 text-sm font-medium hover:text-[#10a37f] dark:hover:text-emerald-400 hover:underline cursor-pointer"
                              >
                                {item.title}
                              </a>
                              {item.author && <p className="text-gray-500 dark:text-gray-400 text-xs mt-0.5">{item.author}</p>}
                              {item.reason && <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">{item.reason}</p>}
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
                  <section className="rounded-2xl bg-white border border-gray-100 shadow-sm dark:rounded-xl dark:bg-[#2f2f2f] dark:border-gray-700 p-4 flex flex-col">
                    <div className="flex flex-wrap items-baseline gap-2 mb-3">
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
                  <section className="rounded-2xl bg-white border border-gray-100 shadow-sm dark:rounded-xl dark:bg-[#2f2f2f] dark:border-gray-700 p-4 flex flex-col">
                    <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest mb-3">News & articles</h3>
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
                  <section className="rounded-2xl bg-white border border-gray-100 shadow-sm dark:rounded-xl dark:bg-[#2f2f2f] dark:border-gray-700 p-4 flex flex-col">
                    <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-widest mb-3">Research papers</h3>
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
        </div>
      </main>

      <SessionSidePanel
        open={sessionPanelOpen}
        onOpen={() => setSessionPanelOpen(true)}
        onClose={() => setSessionPanelOpen(false)}
        settingsExpanded={sessionSettingsExpanded}
        onToggleSettings={() => setSessionSettingsExpanded((v) => !v)}
        status={status}
        onConnect={handleConnect}
        onDisconnect={handleDisconnect}
        errorMessage={errorMessage}
        sessionMode={sessionMode}
        setSessionMode={setSessionMode}
        voices={voices}
        selectedVoiceId={selectedVoiceId}
        setSelectedVoiceId={setSelectedVoiceId}
        showLiveTranscription={showLiveTranscription}
        setShowLiveTranscription={setShowLiveTranscription}
        isVoiceMemoMode={isVoiceMemoMode}
        inputMode={inputMode}
        handleInputModeChange={handleInputModeChange}
        isModeToggleLocked={isModeToggleLocked}
        isConnected={isConnected}
        orbState={orbState}
        thinkingProgress={thinkingProgress}
        isAiSpeaking={isAiSpeaking}
        isUserSpeaking={isUserSpeaking}
        isProcessing={isProcessing}
        isVoiceMemoRecording={isVoiceMemoRecording}
        startVoiceMemoRecording={startVoiceMemoRecording}
        stopVoiceMemoRecording={stopVoiceMemoRecording}
        lastPlaybackFailed={lastPlaybackFailed}
        playLastFailedPlayback={playLastFailedPlayback}
        commitManual={commitManual}
        typedInput={typedInput}
        setTypedInput={setTypedInput}
        handleSendTypedInput={handleSendTypedInput}
        transcript={transcript}
        interimTranscript={interimTranscript}
        expandedLogIndex={expandedLogIndex}
        setExpandedLogIndex={setExpandedLogIndex}
        transcriptScrollRef={transcriptScrollRef}
        handleTranscriptScroll={handleTranscriptScroll}
      />

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

      {/* Footer — muted, ethereal */}
      <footer className="pointer-events-none relative z-10 mt-auto flex-shrink-0 px-4 pb-3 pt-1 text-center">
        <p className="pointer-events-auto text-xs text-white/60">
          {!isConnected
            ? "Connect to begin your journaling session."
            : isProcessing
              ? "Thinking..."
              : inputMode === "text"
                ? "Type your message to the AI."
              : "Speak naturally. The AI is listening."}
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
    </div>
  );
};
