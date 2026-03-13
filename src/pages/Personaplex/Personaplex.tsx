import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type VoiceOption = { voice_id: string; name: string };
import { usePersonaplexSession, type TranscriptEntry } from "./hooks/usePersonaplexSession";

function TranscriptBubble({
  entry,
  isLogExpanded,
  onToggleLog,
}: {
  entry: TranscriptEntry;
  isLogExpanded: boolean;
  onToggleLog: () => void;
}) {
  const isUser = entry.role === "user";
  const hasLog = !isUser && entry.retrievalLog;
  return (
    <div className={`text-sm flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] break-words ${
          isUser ? "text-violet-200 text-right" : "text-slate-300 text-left"
        }`}
      >
        <span className="font-medium opacity-80 block mb-0.5">
          {isUser ? "You" : "AI"}
        </span>
        {entry.text}
        {hasLog && (
          <div className="mt-2 text-left">
            <button
              type="button"
              onClick={onToggleLog}
              className="text-xs text-violet-400/90 hover:text-violet-300 font-medium"
            >
              {isLogExpanded ? "Hide" : "Show"} memory context (vector DB)
            </button>
            {isLogExpanded && (
              <pre className="mt-1.5 p-2 rounded bg-slate-800/80 text-slate-400 text-xs whitespace-pre-wrap break-words border border-slate-700/50 max-h-48 overflow-y-auto">
                {entry.retrievalLog}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
import { useJournalHistory } from "./hooks/useJournalHistory";
import { Orb, OrbState } from "./components/Orb";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { ConnectButton } from "./components/ConnectButton";
import { JournalGallery } from "./components/JournalGallery";
import { MemoryEditor } from "./components/MemoryEditor";

/** Default journaling assistant prompt (base; personalization is always \"high\" / memory-connected) */
const DEFAULT_PERSONAPLEX_PROMPT = `You are an empathetic and insightful conversational journaling assistant. Your goal is to provide a supportive space for the user to reflect on their thoughts, experiences, and emotions. Read the user's entries and respond naturally. Ask open-ended questions to encourage further exploration, but always let the user guide the direction and depth of the conversation. Avoid being overly prescriptive, giving unsolicited advice, or summarizing their thoughts unnecessarily. Just be a curious, active listener. Always facilitate conversation that gets the user exploring their thoughts and emotions. Try to keep responses brief and concise when possible to conserve tokens.`;

const INTRUSIVENESS_LEVELS = [0, 0.25, 0.5, 0.75, 1] as const;
const INTRUSIVENESS_LABELS: Record<number, string> = {
  0: "Context building only; follow the user's lead; gather and reflect back; avoid probing or emotional questions.",
  0.25: "Mostly context building; ask sparingly and only to clarify or expand.",
  0.5: "Balanced; mix context-building with occasional reflective questions.",
  0.75: "More dynamic questions; ask how things made them feel, what they noticed, etc., when it fits.",
  1: "Dynamic questions; actively ask \"how did this make you feel?\", \"what was that like?\", and other reflective, feeling-focused questions to deepen exploration.",
};

const DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // Rachel
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";
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

export const Personaplex = () => {
  // Personalization is always \"high\": the agent should actively connect past memories and journals to the current conversation when relevant.
  const personalization = 1;
  const [intrusiveness, setIntrusiveness] = useState(0.5);
  const [sessionMode, setSessionMode] = useState<"journal" | "recommendations" | "extreme" | "therapy">("journal");
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
  const [view, setView] = useState<"session" | "history" | "memory" | "recommendations" | "calendar">("session");
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [priorJournalText, setPriorJournalText] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [memoryStats, setMemoryStats] = useState<{ gist_facts_count: number; episodic_log_count: number } | null>(null);
  const [isWipingMemory, setIsWipingMemory] = useState(false);
  const [showLiveTranscription, setShowLiveTranscription] = useState(true);
  type RecItem = { title: string; author?: string; reason?: string; url?: string };
  const [recommendations, setRecommendations] = useState<{ books: RecItem[]; podcasts: RecItem[]; articles: RecItem[]; research: RecItem[] }>({ books: [], podcasts: [], articles: [], research: [] });
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const [consumedIds, setConsumedIds] = useState<Set<string>>(new Set());
  const [removingKeys, setRemovingKeys] = useState<Set<string>>(new Set());
  const [showLibrary, setShowLibrary] = useState(false);
  const [libraryExpandedCategory, setLibraryExpandedCategory] = useState<"book" | "podcast" | "article" | "research" | null>(null);
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
  const [memoryFacts, setMemoryFacts] = useState<{ id: number; document: string; session_id?: string; timestamp?: string }[]>([]);
  const [memorySummaries, setMemorySummaries] = useState<{ id: number; document: string; session_id?: string; timestamp?: string; metadata_json?: string | null }[]>([]);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const fetchMemoryList = useCallback(() => {
    setMemoryLoading(true);
    Promise.all([
      fetch(`${BACKEND_URL}/memory/facts`).then((r) => (r.ok ? r.json() : { facts: [] })),
      fetch(`${BACKEND_URL}/memory/summaries`).then((r) => (r.ok ? r.json() : { summaries: [] })),
    ])
      .then(([factsRes, summariesRes]) => {
        setMemoryFacts(Array.isArray(factsRes.facts) ? factsRes.facts : []);
        setMemorySummaries(Array.isArray(summariesRes.summaries) ? summariesRes.summaries : []);
      })
      .catch(() => {
        setMemoryFacts([]);
        setMemorySummaries([]);
      })
      .finally(() => setMemoryLoading(false));
  }, []);
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
    return fetch(`${BACKEND_URL}/library`)
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
    if (!showLibrary) return;
    setLibraryLoading(true);
    fetchLibrary(true).finally(() => setLibraryLoading(false));
  }, [showLibrary, fetchLibrary]);

  const sendLibraryInterviewMessage = useCallback(
    (msg: string) => {
      if (!msg.trim() || libraryInterviewLoading) return;
      setLibraryInterviewMessages((prev) => [...prev, { role: "user", content: msg.trim() }]);
      setLibraryInterviewInput("");
      setLibraryInterviewLoading(true);
      const snapshot = libraryItems.books.map((b) => ({ id: b.id, title: b.title, author: b.author ?? undefined, note: b.note ?? undefined }));
      fetch(`${BACKEND_URL}/library-interview`, {
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
    deleteEntry,
    getFormattedDate,
    exportAllJournals,
    importEntriesFromExport,
    isExportPayload,
  } = useJournalHistory();
  const importFileInputRef = useRef<HTMLInputElement>(null);
  const [isImporting, setIsImporting] = useState(false);

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

  const fetchMemoryStats = useCallback(() => {
    fetch(`${BACKEND_URL}/memory-stats`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed to load"))))
      .then((data: { gist_facts_count?: number; episodic_log_count?: number }) => {
        setMemoryStats({
          gist_facts_count: data.gist_facts_count ?? 0,
          episodic_log_count: data.episodic_log_count ?? 0,
        });
      })
      .catch(() => setMemoryStats(null));
  }, []);

  useEffect(() => {
    if (view === "memory") {
      fetchMemoryStats();
      fetchMemoryList();
    }
  }, [view, fetchMemoryStats, fetchMemoryList]);

  const fetchRecommendations = useCallback((showLoadingUnlessCached = false) => {
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
          const ac = new AbortController();
          const timeoutId = setTimeout(() => ac.abort(), 125000);
          fetch(`${BACKEND_URL}/recommendations`, { signal: ac.signal })
            .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed to load"))))
            .then((data: { books?: RecItem[]; podcasts?: RecItem[]; articles?: RecItem[]; research?: RecItem[] }) => {
              const next = {
                books: data.books ?? [],
                podcasts: data.podcasts ?? [],
                articles: data.articles ?? [],
                research: data.research ?? [],
              };
              setRecommendations(next);
              localStorage.setItem(RECOMMENDATIONS_CACHE_KEY, JSON.stringify(next));
            })
            .catch(() => {})
            .finally(() => {
              clearTimeout(timeoutId);
              setRecommendationsLoading(false);
            });
          return;
        }
      }
    } catch {
      /* ignore cache parse errors */
    }
    setRecommendationsLoading(true);
    const ac = new AbortController();
    const timeoutId = setTimeout(() => ac.abort(), 125000);
    fetch(`${BACKEND_URL}/recommendations`, { signal: ac.signal })
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
      .catch(() => setRecommendations({ books: [], podcasts: [], articles: [], research: [] }))
      .finally(() => {
        clearTimeout(timeoutId);
        setRecommendationsLoading(false);
      });
  }, []);

  useEffect(() => {
    if (view === "recommendations") {
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
    }
  }, [view, fetchRecommendations]);

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
      fetch(`${BACKEND_URL}/recommendations/consumed`, {
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

  const handleIngestPriorJournal = useCallback(async () => {
    if (!priorJournalText.trim()) return;
    setIsIngesting(true);
    const textToIngest = priorJournalText.trim();
    let inferredDate: string | undefined;
    try {
      const ir = await fetch(`${BACKEND_URL}/infer-entry-date`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToIngest }),
      });
      if (ir.ok) {
        const data = await ir.json();
        if (data?.date) inferredDate = data.date;
      }
    } catch {
      /* use no date / today as fallback */
    }
    try {
      const r = await fetch(`${BACKEND_URL}/ingest-history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToIngest, entry_date: inferredDate }),
      });
      const data = r.ok ? await r.json().catch(() => ({})) : { ok: false };
      if (data && data.ok === false) {
        setToastMessage("Failed to add to memory. Check backend logs.");
        setTimeout(() => setToastMessage(null), 4000);
        return;
      }
      setPriorJournalText("");
      saveEntry([{ role: "user", text: textToIngest }], inferredDate);
      setToastMessage("Journal added to memory and saved to History.");
      setTimeout(() => setToastMessage(null), 5000);
      fetchMemoryStats();
    } catch (err) {
      setToastMessage(err instanceof Error ? err.message : "Failed to add to memory");
      setTimeout(() => setToastMessage(null), 4000);
    } finally {
      setIsIngesting(false);
    }
  }, [priorJournalText, fetchMemoryStats, saveEntry]);

  const handleWipeMemory = useCallback(() => {
    if (!window.confirm("Wipe all data from the vector DB? This cannot be undone. The AI will have no prior journal memory until you add entries again.")) return;
    setIsWipingMemory(true);
    fetch(`${BACKEND_URL}/memory-wipe`, { method: "POST" })
      .then((r) => {
        if (!r.ok) return r.json().then((d) => Promise.reject(new Error(d.detail ?? "Wipe failed")));
      })
      .then(() => {
        fetchMemoryStats();
        setToastMessage("Memory wiped.");
        setTimeout(() => setToastMessage(null), 3000);
      })
      .catch((err) => {
        setToastMessage(err instanceof Error ? err.message : "Failed to wipe memory");
        setTimeout(() => setToastMessage(null), 4000);
      })
      .finally(() => setIsWipingMemory(false));
  }, [fetchMemoryStats]);

  const {
    status,
    errorMessage,
    isProcessing,
    connect,
    disconnect,
    commitManual,
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
    connect();
  }, [connect]);

  const handleDisconnect = useCallback(() => {
    if (transcript.length > 0) {
      saveEntry(transcript);
      setToastMessage("Journal entry saved.");
      setTimeout(() => setToastMessage(null), 3000);
    }
    setTranscript([]);
    setInterimTranscript("");
    disconnect();
  }, [disconnect, transcript, saveEntry]);

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

  const handleDownloadAllJournals = useCallback(() => {
    const json = exportAllJournals();
    const dateStr = new Date().toISOString().slice(0, 10);
    const filename = `openjournal-journals-${dateStr}.json`;
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    setToastMessage("Journals downloaded.");
    setTimeout(() => setToastMessage(null), 3000);
  }, [exportAllJournals]);

  const handleImportFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;
      // Capture file(s) before clearing input — some browsers clear FileList when value is reset.
      const fileList = Array.from(files);
      e.target.value = "";

      // Multiple files: treat each as a journal (no JSON export handling).
      if (fileList.length > 1) {
        setIsImporting(true);
        (async () => {
          let imported = 0;
          let synced = 0;
          try {
            for (const file of fileList) {
              const lowerName = file.name.toLowerCase();
              if (!/\.(txt|md|markdown|journal|log|json)$/.test(lowerName)) continue;
              let text: string;
              try {
                text = (await file.text()).trim();
              } catch {
                continue;
              }
              if (!text) continue;
              imported += 1;
              let inferredDate: string | undefined;
              try {
                const ir = await fetch(`${BACKEND_URL}/infer-entry-date`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ text, filename: file.name }),
                });
                if (ir.ok) {
                  const data = await ir.json();
                  if (data?.date) inferredDate = data.date;
                }
              } catch {
                /* use import time as fallback */
              }
              saveEntry([{ role: "user", text }], inferredDate);
              try {
                const r = await fetch(`${BACKEND_URL}/ingest-history`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ text, entry_date: inferredDate }),
                });
                if (r.ok) synced += 1;
              } catch {
                /* continue */
              }
            }
            if (imported > 0) fetchMemoryStats();
            if (imported === 0) {
              setToastMessage("No readable journal files in selection.");
            } else if (synced === imported) {
              setToastMessage(`Imported ${imported} journal${imported === 1 ? "" : "s"} and synced to memory.`);
            } else {
              setToastMessage(`Imported ${imported} journal${imported === 1 ? "" : "s"}. ${synced} synced to memory.`);
            }
          } finally {
            setTimeout(() => setToastMessage(null), 5000);
            setIsImporting(false);
          }
        })();
        return;
      }

      // Single file: JSON export or one journal.
      const file = fileList[0]!;
      setIsImporting(true);
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const text = reader.result;
          if (typeof text !== "string") throw new Error("Invalid file");
          const lowerName = file.name.toLowerCase();

          if (lowerName.endsWith(".json")) {
            const parsed = JSON.parse(text) as unknown;
            if (!isExportPayload(parsed)) throw new Error("Not a valid OpenJournal export file");
            const count = importEntriesFromExport(parsed);
            if (count === 0) {
              setToastMessage("No valid entries in file.");
              setTimeout(() => setToastMessage(null), 4000);
              return;
            }
            setToastMessage(`Imported ${count} journal${count === 1 ? "" : "s"}. Syncing to memory…`);
            const entries = parsed.entries as { fullTranscript?: { role: string; text: string }[]; date?: string }[];
            let synced = 0;
            for (const entry of entries) {
              const msgs = entry?.fullTranscript;
              if (!Array.isArray(msgs) || msgs.length === 0) continue;
              const transcriptText = msgs
                .map((m) => (m?.role === "user" ? "You: " + (m?.text ?? "") : "AI: " + (m?.text ?? "")))
                .join("\n");
              if (!transcriptText.trim()) continue;
              try {
                const r = await fetch(`${BACKEND_URL}/ingest-history`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ text: transcriptText, entry_date: entry?.date }),
                });
                if (r.ok) synced += 1;
              } catch {
                /* continue with next entry */
              }
            }
            fetchMemoryStats();
            setToastMessage(
              synced === entries.length
                ? `Imported ${count} journal${count === 1 ? "" : "s"} and synced to memory.`
                : `Imported ${count} journal${count === 1 ? "" : "s"}. ${synced} synced to memory.`
            );
          } else {
            const content = text.trim();
            if (!content) throw new Error("File is empty.");
            let inferredDate: string | undefined;
            try {
              const ir = await fetch(`${BACKEND_URL}/infer-entry-date`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: content, filename: file.name }),
              });
              if (ir.ok) {
                const data = await ir.json();
                if (data?.date) inferredDate = data.date;
              }
            } catch {
              /* use import time as fallback */
            }
            saveEntry([{ role: "user", text: content }], inferredDate);
            try {
              const ingestRes = await fetch(`${BACKEND_URL}/ingest-history`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: content, entry_date: inferredDate }),
              });
              const data = ingestRes.ok ? await ingestRes.json().catch(() => ({})) : { ok: false };
              fetchMemoryStats();
              setToastMessage(data?.ok === false ? "Imported journal, but syncing to memory failed." : "Imported journal and synced to memory.");
            } catch {
              setToastMessage("Imported journal, but syncing to memory failed.");
            }
          }
        } catch (err) {
          setToastMessage(err instanceof Error ? err.message : "Import failed.");
        }
        setTimeout(() => setToastMessage(null), 5000);
        setIsImporting(false);
      };
      reader.onerror = () => {
        setToastMessage("Failed to read file.");
        setTimeout(() => setToastMessage(null), 3000);
        setIsImporting(false);
      };
      reader.readAsText(file);
    },
    [importEntriesFromExport, isExportPayload, fetchMemoryStats, saveEntry]
  );

  useEffect(() => {
    const scrollEl = transcriptScrollRef.current;
    if (!scrollEl || !autoScrollEnabledRef.current) return;
    scrollEl.scrollTop = scrollEl.scrollHeight - scrollEl.clientHeight;
  }, [transcript, interimTranscript]);

  return (
    <div className="h-screen w-full flex flex-col overflow-hidden bg-slate-950 text-slate-100">
      {/* Background gradient */}
      <div
        className="fixed inset-0 pointer-events-none"
        aria-hidden
      >
        <div className="absolute inset-0 bg-gradient-to-b from-slate-950 via-slate-900/50 to-slate-950" />
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-violet-500/5 blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] rounded-full bg-cyan-500/5 blur-3xl" />
      </div>

      {/* Header */}
      <header className="flex-none relative z-10 grid grid-cols-[1fr_auto_1fr] items-center gap-2 px-4 sm:px-6 py-3 sm:py-4">
        <div className="flex items-center gap-2 sm:gap-4 min-w-0">
          <h1 className="text-base sm:text-xl font-light tracking-widest text-slate-300 uppercase truncate">
            OpenJournal
          </h1>
          <ConnectionStatus status={status} />
          {errorMessage && (
            <span className="text-sm text-red-400">{errorMessage}</span>
          )}
        </div>
        <div className="flex justify-center mt-1.5">
          <ConnectButton
            status={status}
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
          />
        </div>
        <div className="flex items-center justify-end gap-1 sm:gap-2">
          <button
            type="button"
            onClick={() => setView("session")}
            className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
              view === "session" ? "bg-violet-600/80 text-white" : "bg-slate-700/50 text-slate-300 hover:bg-slate-600/50"
            }`}
            title="Journaling session"
          >
            <span className="hidden sm:inline">Session</span>
          </button>
          <button
            type="button"
            onClick={() => setView("history")}
            className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
              view === "history" ? "bg-violet-600/80 text-white" : "bg-slate-700/50 text-slate-300 hover:bg-slate-600/50"
            }`}
            title="Journal history"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            <span className="hidden sm:inline">History</span>
          </button>
          <button
            type="button"
            onClick={() => setView("memory")}
            className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
              view === "memory" ? "bg-violet-600/80 text-white" : "bg-slate-700/50 text-slate-300 hover:bg-slate-600/50"
            }`}
            title="Brain (AI memory editor)"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span className="hidden sm:inline">Brain</span>
          </button>
          <button
            type="button"
            onClick={() => setView("recommendations")}
            className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
              view === "recommendations" ? "bg-violet-600/80 text-white" : "bg-slate-700/50 text-slate-300 hover:bg-slate-600/50"
            }`}
            title="Personalized recommendations"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
            </svg>
            <span className="hidden sm:inline">Recommendations</span>
          </button>
          <button
            type="button"
            onClick={() => setView("calendar")}
            className={`px-2 py-1.5 sm:px-3 sm:py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
              view === "calendar" ? "bg-violet-600/80 text-white" : "bg-slate-700/50 text-slate-300 hover:bg-slate-600/50"
            }`}
            title="Calendar — day highlights"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span className="hidden sm:inline">Calendar</span>
          </button>
        </div>
      </header>

      {toastMessage && (
        <div
          className="fixed bottom-24 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-lg bg-emerald-500/90 text-slate-900 text-sm font-medium shadow-lg"
          role="status"
        >
          {toastMessage}
        </div>
      )}

      {/* Main content - 3-column grid or gallery */}
      <main className="flex-1 flex flex-col min-h-0 relative z-10">
          <div
            className={`flex-1 min-h-0 p-4 md:p-6 transition-opacity duration-300 overflow-y-auto overflow-x-hidden lg:overflow-visible ${
              view !== "session" ? "opacity-0 pointer-events-none absolute inset-0" : "opacity-100"
            }`}
          >
          {/* 3-column grid: desktop | scrollable single column: mobile/tablet */}
          <div className="min-h-full lg:h-full lg:min-h-0 grid grid-cols-1 lg:grid-cols-[1fr_2fr_1fr] gap-4 lg:gap-6 grid-rows-[auto auto auto] lg:grid-rows-[minmax(0,1fr)]">
            {/* Left column - Settings (order 1 on mobile) */}
            <div className="order-1 lg:order-none flex flex-col min-h-0 min-w-0 rounded-xl bg-slate-900/50 border border-slate-700/50 p-4">
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">
                Settings
              </h2>
              <div>
                <label htmlFor="personaplex-session-mode" className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1.5">
                  Session mode
                </label>
                <select
                  id="personaplex-session-mode"
                  value={sessionMode}
                  onChange={(e) => setSessionMode(e.target.value as "journal" | "recommendations" | "extreme" | "therapy")}
                  disabled={isConnected}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700/50 text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50 disabled:opacity-60 disabled:cursor-not-allowed"
                  aria-label="Interview mode: journal, recommendations, extreme, or therapy"
                >
                  <option value="journal">Regular interview (journal entries)</option>
                  <option value="recommendations">Recommendations (talk about your books)</option>
                  <option value="extreme">Extreme (intrusive reflection)</option>
                  <option value="therapy">Therapy mode</option>
                </select>
                {sessionMode === "recommendations" && (
                  <p className="text-xs text-slate-500 mt-1">
                    Same speech-to-text; agent asks about your library and saves short notes for better recommendations.
                  </p>
                )}
                {sessionMode === "extreme" && (
                  <p className="text-xs text-slate-500 mt-1">
                    Asks direct, private questions to help you reflect and feel better. No advice—just reflection.
                  </p>
                )}
                {sessionMode === "therapy" && (
                  <p className="text-xs text-slate-500 mt-1">
                    Supportive, reflective listening. No advice—just space to explore.
                  </p>
                )}
              </div>
              <div className="mt-3">
                <label htmlFor="personaplex-voice" className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1.5">
                  Voice
                </label>
                <select
                  id="personaplex-voice"
                  value={selectedVoiceId}
                  onChange={(e) => setSelectedVoiceId(e.target.value)}
                  disabled={isConnected}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700/50 text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50 disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {voices.map((v) => (
                    <option key={v.voice_id} value={v.voice_id}>
                      {v.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="mt-3 space-y-1.5">
                <label htmlFor="personaplex-intrusiveness" className="block text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Questioning style
                </label>
                <div className="hidden sm:block">
                  <input
                    id="personaplex-intrusiveness"
                    type="range"
                    min={0}
                    max={INTRUSIVENESS_LEVELS.length - 1}
                    step={1}
                    value={Math.max(0, INTRUSIVENESS_LEVELS.findIndex((p) => p === intrusiveness))}
                    onChange={(e) => setIntrusiveness(INTRUSIVENESS_LEVELS[Number(e.target.value)])}
                    disabled={isConnected}
                    className="w-full h-2 rounded-full bg-slate-600 accent-violet-500 disabled:opacity-60"
                    aria-valuenow={Math.round(intrusiveness * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuetext={`${Math.round(intrusiveness * 100)}%`}
                  />
                </div>
                <select
                  aria-label="Questioning style"
                  value={intrusiveness}
                  onChange={(e) => setIntrusiveness(Number(e.target.value))}
                  disabled={isConnected}
                  className="sm:hidden w-full px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700/50 text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                >
                  {INTRUSIVENESS_LEVELS.map((p) => (
                    <option key={p} value={p}>
                      {Math.round(p * 100)}%
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-500">
                  {Math.round(intrusiveness * 100)}% — context building ↔ dynamic questions
                </p>
              </div>
              <div className="mt-3 space-y-1.5">
                <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Live transcription
                </label>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setShowLiveTranscription((v) => !v)}
                    className={`relative w-10 h-5 rounded-full transition-colors duration-200 ease-out ${
                      showLiveTranscription ? "bg-violet-500" : "bg-slate-600"
                    }`}
                    aria-pressed={showLiveTranscription}
                    aria-label="Toggle live transcription"
                    disabled={isVoiceMemoMode}
                  >
                    <span
                      className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform duration-200 ease-out ${
                        showLiveTranscription ? "translate-x-5" : ""
                      }`}
                    />
                  </button>
                  <span className="text-xs text-slate-500">
                    {isVoiceMemoMode
                      ? "On mobile, transcription appears after you tap Done."
                      : showLiveTranscription
                        ? "Show words as you speak."
                        : "Only show what you said after you tap Done."}
                  </span>
                </div>
              </div>
              <div className="mt-3 flex-1 min-h-0 flex flex-col">
                <label htmlFor="personaplex-text-prompt" className="block text-xs font-medium text-slate-400 uppercase tracking-wider mb-1.5">
                  System Prompt
                </label>
                <textarea
                  id="personaplex-text-prompt"
                  value={textPrompt}
                  readOnly
                  rows={4}
                  className="w-full min-w-0 px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700/50 text-slate-200 text-sm resize-none min-h-[72px] max-h-[120px] overflow-auto"
                  aria-label="System prompt (updates with personalization and questioning style)"
                />
              </div>
            </div>

            {/* Center column - Orb (order 2 on mobile) */}
            <div className="order-2 lg:order-none flex flex-col items-center justify-center gap-2 sm:gap-4 min-h-0 py-2 sm:py-4 lg:py-0 min-w-0 w-full">
              <div className="flex-none flex flex-col items-center gap-3">
                <Orb state={orbState} thinkingProgress={thinkingProgress} />
                {isVoiceMemoMode && isConnected && (
                  isVoiceMemoRecording ? (
                    <button
                      type="button"
                      onClick={stopVoiceMemoRecording}
                      className="px-6 py-3 rounded-full bg-red-500/80 hover:bg-red-500 text-white text-sm font-medium transition-colors"
                    >
                      Done
                    </button>
                  ) : lastPlaybackFailed ? (
                    <button
                      type="button"
                      onClick={playLastFailedPlayback}
                      className="px-6 py-3 rounded-full bg-violet-500/80 hover:bg-violet-500 text-white text-sm font-medium transition-colors"
                    >
                      Play response
                    </button>
                  ) : !isAiSpeaking ? (
                    <button
                      type="button"
                      onClick={startVoiceMemoRecording}
                      className="px-6 py-3 rounded-full bg-emerald-500/80 hover:bg-emerald-500 text-white text-sm font-medium transition-colors"
                    >
                      Record
                    </button>
                  ) : null
                )}
                {!isVoiceMemoMode && isConnected && isUserSpeaking && !isAiSpeaking && (
                  <button
                    type="button"
                    onClick={commitManual}
                    className="px-6 py-3 rounded-full bg-violet-500/80 hover:bg-violet-500 text-white text-sm font-medium transition-colors"
                  >
                    Done recording
                  </button>
                )}
              </div>
            </div>

            {/* Right column - Transcript (order 3 on mobile) */}
            <div
              className="order-3 lg:order-none flex min-h-0 flex-col rounded-xl bg-slate-900/50 border border-slate-700/50 overflow-hidden min-h-[120px] lg:min-h-0"
              aria-label="Conversation transcript"
            >
              <div className="flex-none shrink-0 px-4 py-2 border-b border-slate-700/50">
                <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                  Transcript
                </h2>
              </div>
              <div
                ref={transcriptScrollRef}
                onScroll={handleTranscriptScroll}
                className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden scrollbar p-4 space-y-3"
              >
                {transcript.length === 0 && !interimTranscript ? (
                  <p className="text-sm text-slate-500 italic">
                    {isVoiceMemoMode
                      ? "Tap Record, speak, then tap Done. Your words will appear here."
                      : "Conversation will appear here as you speak. Tap Done recording to finish your turn."}
                  </p>
                ) : (
                  <>
                    {transcript.map((entry, i) => (
                      <TranscriptBubble
                        key={i}
                        entry={entry}
                        isLogExpanded={expandedLogIndex === i}
                        onToggleLog={() => setExpandedLogIndex((prev) => (prev === i ? null : i))}
                      />
                    ))}
                    {interimTranscript && (
                      <div className="flex justify-end">
                        <div className="max-w-[85%] break-words text-violet-200/80 text-right italic">
                          <span className="font-medium opacity-80 block mb-0.5">
                            You (speaking...)
                          </span>
                          {interimTranscript}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        <div
          className={`flex-1 flex flex-col min-h-0 overflow-y-auto transition-opacity duration-300 ${
            view === "history" || view === "memory" || view === "recommendations" || view === "calendar" ? "opacity-100" : "opacity-0 pointer-events-none absolute inset-0"
          }`}
        >
          {view === "history" && (
            <>
              <div className="p-4 md:p-6 flex-shrink-0 space-y-4">
                <div className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 max-w-2xl">
                  <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">
                    Export & import
                  </h3>
                  <p className="text-xs text-slate-500 mb-3">
                    Download all journal entries as one JSON file, or upload a previously exported file to restore them here.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={handleDownloadAllJournals}
                      disabled={entries.length === 0}
                      className="px-4 py-2 rounded-lg bg-slate-700/80 hover:bg-slate-600/80 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 text-sm font-medium transition-colors flex items-center gap-2"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Download all journals
                    </button>
                    <input
                      ref={importFileInputRef}
                      type="file"
                      accept=".json,.txt,.md,.markdown,.journal,.log,application/json,text/plain"
                      multiple
                      onChange={handleImportFileChange}
                      className="hidden"
                      aria-label="Import journal(s) or export file"
                    />
                    <button
                      type="button"
                      onClick={() => importFileInputRef.current?.click()}
                      disabled={isImporting}
                      className="px-4 py-2 rounded-lg bg-slate-700/80 hover:bg-slate-600/80 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 text-sm font-medium transition-colors flex items-center gap-2"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      {isImporting ? "Importing…" : "Import from file"}
                    </button>
                  </div>
                </div>
                <div className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 max-w-2xl">
                  <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">
                    Add prior journal to memory
                  </h3>
                  <p className="text-xs text-slate-500 mb-3">
                    Paste journal text to ingest into memory. It will be summarized and stored so the AI can personalize at 100%. View stats on Memory.
                  </p>
                  <textarea
                    value={priorJournalText}
                    onChange={(e) => setPriorJournalText(e.target.value)}
                    placeholder="Paste journal text here..."
                    rows={4}
                    className="w-full px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700/50 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 resize-y mb-3"
                  />
                  <button
                    type="button"
                    onClick={handleIngestPriorJournal}
                    disabled={!priorJournalText.trim() || isIngesting}
                    className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors"
                  >
                    {isIngesting ? "Adding… (may take a minute)" : "Add to memory"}
                  </button>
                </div>
              </div>
              <div className="flex-1 min-h-0">
                <JournalGallery
                  entries={entries}
                  onDeleteEntry={deleteEntry}
                  getFormattedDate={getFormattedDate}
                  onToast={(msg) => {
                    setToastMessage(msg);
                    setTimeout(() => setToastMessage(null), 3000);
                  }}
                />
              </div>
            </>
          )}
          {view === "memory" && (
            <MemoryEditor
              facts={memoryFacts}
              summaries={memorySummaries}
              stats={memoryStats}
              loading={memoryLoading}
              onRefresh={fetchMemoryList}
              onRefreshStats={fetchMemoryStats}
              onToast={(msg) => {
                setToastMessage(msg);
                setTimeout(() => setToastMessage(null), 2000);
              }}
              onWipeMemory={handleWipeMemory}
              isWipingMemory={isWipingMemory}
            />
          )}
          {view === "calendar" && (
            <div className="flex-1 flex flex-col min-h-0 p-4 md:p-6 overflow-hidden">
              <h2 className="text-lg font-medium text-slate-300 uppercase tracking-wider mb-2">
                Calendar
              </h2>
              <p className="text-sm text-slate-500 mb-4">
                Click a date to see an AI summary of that day (raw journal + memory DB).
              </p>
              <div className="flex flex-col sm:flex-row gap-6 flex-1 min-h-0">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-between gap-4 mb-3">
                    <button
                      type="button"
                      onClick={() => setCalendarMonth((prev) => {
                        const d = new Date(prev.year, prev.month - 1, 1);
                        return { year: d.getFullYear(), month: d.getMonth() };
                      })}
                      className="p-2 rounded-lg bg-slate-700/50 text-slate-300 hover:bg-slate-600/50"
                      aria-label="Previous month"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
                    </button>
                    <span className="text-slate-200 font-medium">
                      {new Date(calendarMonth.year, calendarMonth.month, 1).toLocaleString("default", { month: "long", year: "numeric" })}
                    </span>
                    <button
                      type="button"
                      onClick={() => setCalendarMonth((prev) => {
                        const d = new Date(prev.year, prev.month + 1, 1);
                        return { year: d.getFullYear(), month: d.getMonth() };
                      })}
                      className="p-2 rounded-lg bg-slate-700/50 text-slate-300 hover:bg-slate-600/50"
                      aria-label="Next month"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
                    </button>
                  </div>
                  <div className="grid grid-cols-7 gap-1 text-center">
                    {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((day) => (
                      <div key={day} className="text-xs text-slate-500 font-medium py-1">{day}</div>
                    ))}
                    {(() => {
                      const first = new Date(calendarMonth.year, calendarMonth.month, 1);
                      const last = new Date(calendarMonth.year, calendarMonth.month + 1, 0);
                      const startPad = first.getDay();
                      const daysInMonth = last.getDate();
                      const cells: (number | null)[] = [];
                      for (let i = 0; i < startPad; i++) cells.push(null);
                      for (let d = 1; d <= daysInMonth; d++) cells.push(d);
                      const dateStr = (d: number) => `${calendarMonth.year}-${String(calendarMonth.month + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
                      const hasEntry = (d: number) => entries.some((e) => (e.date || "").startsWith(dateStr(d)));
                      return cells.map((d, i) => (
                        <div key={i}>
                          {d === null ? (
                            <div className="w-9 h-9" />
                          ) : (
                            <button
                              type="button"
                              onClick={() => {
                                const date = dateStr(d);
                                setCalendarSelectedDate(date);
                                setCalendarDaySummary(null);
                                setCalendarDaySummaryLoading(true);
                                const dayEntries = entries.filter((e) => (e.date || "").startsWith(date));
                                const rawTranscript = dayEntries.length
                                  ? dayEntries.map((e) => e.fullTranscript.map((m) => `${m.role}: ${m.text}`).join("\n")).join("\n\n")
                                  : "";
                                fetch(`${BACKEND_URL}/calendar-day-summary`, {
                                  method: "POST",
                                  headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify({ date, raw_transcript: rawTranscript }),
                                })
                                  .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
                                  .then((data: { summary?: string }) => {
                                    setCalendarDaySummary(data.summary ?? "");
                                  })
                                  .catch(() => setCalendarDaySummary("Could not load summary."))
                                  .finally(() => setCalendarDaySummaryLoading(false));
                              }}
                              className={`w-9 h-9 rounded-lg text-sm font-medium transition-colors ${
                                calendarSelectedDate === dateStr(d)
                                  ? "bg-violet-600 text-white"
                                  : hasEntry(d)
                                    ? "bg-slate-700/80 text-slate-200 hover:bg-slate-600/80"
                                    : "text-slate-400 hover:bg-slate-700/50"
                              }`}
                            >
                              {d}
                            </button>
                          )}
                        </div>
                      ));
                    })()}
                  </div>
                </div>
                <div className="flex-1 min-h-0 rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 flex flex-col">
                  {calendarSelectedDate ? (
                    <>
                      <h3 className="text-slate-300 font-medium mb-2">
                        {new Date(calendarSelectedDate + "T12:00:00").toLocaleDateString("default", { weekday: "long", month: "long", day: "numeric", year: "numeric" })}
                      </h3>
                      {calendarDaySummaryLoading ? (
                        <p className="text-slate-500 text-sm">Analyzing journal and memory…</p>
                      ) : calendarDaySummary ? (
                        <p className="text-slate-200 text-sm whitespace-pre-wrap flex-1 overflow-y-auto">{calendarDaySummary}</p>
                      ) : null}
                    </>
                  ) : (
                    <p className="text-slate-500 text-sm">Click a date to see the day summary.</p>
                  )}
                </div>
              </div>
            </div>
          )}
          {view === "recommendations" && (
            <div className="flex-1 flex flex-col min-h-0 p-4 md:p-6 overflow-hidden">
              <div className="flex-shrink-0 flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-lg font-medium text-slate-300 uppercase tracking-wider">
                    Recommendations
                  </h2>
                  <p className="text-sm text-slate-500 mt-0.5">
                Based on your journal memory and what you’ve already read or listened to. Mark items as read/listened so future suggestions get better.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => fetchRecommendations(false)}
                    disabled={recommendationsLoading}
                    className="px-3 py-2 rounded-lg bg-slate-700/50 text-slate-400 text-sm font-medium hover:bg-slate-600/50 disabled:opacity-50 transition-colors"
                  >
                    {recommendationsLoading ? "Updating…" : "Refresh recommendations"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowLibrary((prev) => !prev)}
                    className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors border ${
                      showLibrary
                        ? "bg-violet-600/80 text-white border-violet-500"
                        : "bg-slate-900/40 text-slate-300 border-slate-600 hover:bg-slate-800/60"
                    }`}
                  >
                    Library
                  </button>
                </div>
              </div>
              <div className="flex-1 min-h-0 overflow-auto">
              {showLibrary ? (
                <div className="space-y-2 pb-4">
                  <h3 className="text-base font-medium text-slate-300 uppercase tracking-wider">
                    My Library
                  </h3>
                  <p className="text-xs text-slate-500 mb-4">
                    Add titles per category. Use the + next to each to paste a list; an agent will organize them for better recommendations.
                  </p>
                  {libraryLoading && libraryItems.books.length === 0 && libraryItems.podcasts.length === 0 && libraryItems.articles.length === 0 && libraryItems.research.length === 0 ? (
                    <p className="text-slate-500 text-sm py-4">Loading library…</p>
                  ) : null}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {(["book", "podcast", "article", "research"] as const).map((cat) => {
                      const key = cat === "book" ? "books" : cat === "podcast" ? "podcasts" : cat === "article" ? "articles" : "research";
                      const items = libraryItems[key] ?? [];
                      const sectionTitle = cat === "book" ? "Books" : cat === "podcast" ? "Podcasts" : cat === "article" ? "News & articles" : "Research papers";
                      return (
                        <section
                          key={cat}
                          className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 flex flex-col"
                        >
                          <div className="relative z-10 flex items-center justify-between gap-2 mb-3 flex-shrink-0">
                            <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                              {sectionTitle}
                            </h4>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                setLibraryExpandedCategory((prev) => (prev === cat ? null : cat));
                                setLibraryDraftText("");
                                setLibrarySubmitting(false);
                              }}
                              className="p-1.5 rounded-lg bg-emerald-600/90 hover:bg-emerald-500 text-white transition-colors shrink-0"
                              aria-label={`Add ${cat}`}
                            >
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                              </svg>
                            </button>
                          </div>
                          {libraryExpandedCategory === cat ? (
                            <div className="mb-3 flex-shrink-0 flex flex-col">
                              <textarea
                                value={libraryDraftText}
                                onChange={(e) => setLibraryDraftText(e.target.value)}
                                rows={3}
                                placeholder={
                                  cat === "book"
                                    ? "Paste book titles, one per line\n e.g. dune\n body keeps the score"
                                    : cat === "podcast"
                                      ? "Paste podcast or episode names\n e.g. Huberman Lab – Sleep"
                                      : cat === "article"
                                        ? "Paste article titles or URLs"
                                        : "Paste paper titles or citations"
                                }
                                className="w-full px-3 py-2 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-xs placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 resize-y mb-2"
                              />
                              <div className="flex gap-2">
                                <button
                                  type="button"
                                  onClick={() => {
                                    if (!libraryDraftText.trim() || librarySubmitting) return;
                                    setLibrarySubmitting(true);
                                    const payload = libraryDraftText.trim();
                                    fetch(`${BACKEND_URL}/library-notes`, {
                                      method: "POST",
                                      headers: { "Content-Type": "application/json" },
                                      body: JSON.stringify({ text: payload, type: cat }),
                                    })
                                      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Library update failed"))))
                                      .then((data: { ok?: boolean; items_added?: number }) => {
                                        const added = data?.items_added ?? 0;
                                        setLibraryDraftText("");
                                        if (added > 0) fetchLibrary();
                                        setToastMessage(added > 0 ? `Added ${added} to ${sectionTitle}.` : "No items recognized; try clearer titles.");
                                        setTimeout(() => setToastMessage(null), 4000);
                                      })
                                      .catch(() => {
                                        setToastMessage("Library update failed.");
                                        setTimeout(() => setToastMessage(null), 4000);
                                      })
                                      .finally(() => setLibrarySubmitting(false));
                                  }}
                                  disabled={librarySubmitting || !libraryDraftText.trim()}
                                  className="px-3 py-1.5 rounded-lg bg-violet-600 text-white text-xs font-medium hover:bg-violet-500 disabled:opacity-50"
                                >
                                  {librarySubmitting ? "Adding…" : "Add"}
                                </button>
                                <button
                                  type="button"
                                  onClick={() => {
                                    setLibraryExpandedCategory(null);
                                    setLibraryDraftText("");
                                  }}
                                  className="px-3 py-1.5 rounded-lg bg-slate-700/60 text-slate-400 text-xs font-medium hover:bg-slate-600/60"
                                >
                                  Cancel
                                </button>
                              </div>
                            </div>
                          ) : null}
                          <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                            {items.length === 0 && libraryExpandedCategory !== cat ? (
                              <p className="text-slate-500 text-xs">No items yet. Use + to add.</p>
                            ) : (
                              items.map((entry, i) => (
                                <div
                                  key={entry.id || `${key}-${i}-${entry.title}`}
                                  className="rounded-lg bg-slate-800/50 border border-slate-700/50 cursor-pointer hover:border-slate-600 transition-colors"
                                >
                                  <button
                                    type="button"
                                    className="w-full text-left p-3"
                                    onClick={() => {
                                      setLibraryEditingId(entry.id);
                                      setLibraryEditDate(entry.date_completed ?? "");
                                      setLibraryEditNote(entry.note ?? "");
                                    }}
                                  >
                                    <p className="text-slate-200 text-sm font-medium">{entry.title}</p>
                                    {entry.author && <p className="text-slate-500 text-xs mt-0.5">{entry.author}</p>}
                                    {entry.date_completed && (
                                      <p className="text-[10px] text-slate-400 mt-1">Completed: {entry.date_completed}</p>
                                    )}
                                    <div className="mt-1.5 min-h-[1.25rem]">
                                      {entry.note ? (
                                        <p className="text-xs text-slate-400 line-clamp-2">{entry.note}</p>
                                      ) : (
                                        <p className="text-[10px] text-slate-500 italic">Add note…</p>
                                      )}
                                    </div>
                                    <p className="text-[10px] text-slate-500 mt-1">Completed</p>
                                  </button>
                                </div>
                              ))
                            )}
                          </div>
                        </section>
                      );
                    })}
                  </div>
                  {/* Library item edit modal */}
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
                        <div className="relative bg-slate-900 border border-slate-700 rounded-xl shadow-xl max-w-lg w-full max-h-[90vh] flex flex-col">
                          <div className="p-4 border-b border-slate-700">
                            <h2 id="library-modal-title" className="text-slate-200 font-medium text-lg truncate">
                              {editingEntry?.title ?? "Library item"}
                            </h2>
                          </div>
                          <div className="p-4 flex-1 min-h-0 overflow-y-auto flex flex-col gap-4">
                            <div>
                              <label className="block text-xs text-slate-500 uppercase tracking-wider mb-2">Notes</label>
                              <textarea
                                value={libraryEditNote}
                                onChange={(e) => setLibraryEditNote(e.target.value)}
                                rows={10}
                                placeholder="How you felt, what stood out — used for better recommendations."
                                className="w-full px-3 py-2.5 rounded-lg bg-slate-950 border border-slate-700 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50 resize-y min-h-[200px]"
                              />
                            </div>
                            <div>
                              <label className="block text-xs text-slate-500 uppercase tracking-wider mb-2">Date completed</label>
                              <input
                                type="text"
                                value={libraryEditDate}
                                onChange={(e) => setLibraryEditDate(e.target.value)}
                                placeholder="Year e.g. 2024 or 2024-06"
                                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                              />
                            </div>
                          </div>
                          <div className="p-4 border-t border-slate-700 flex items-center justify-between gap-4">
                            <button
                              type="button"
                              onClick={() => {
                                if (!libraryEditingId || libraryUpdateSaving) return;
                                if (!confirm("Remove from library and mark as unread? It may be recommended again.")) return;
                                setLibraryUpdateSaving(true);
                                fetch(`${BACKEND_URL}/library/${encodeURIComponent(libraryEditingId)}`, { method: "DELETE" })
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
                                className="px-4 py-2 rounded-lg bg-slate-800/80 text-slate-300 text-sm font-medium hover:bg-slate-700/80 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-slate-500/50"
                              >
                                Cancel
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  if (!libraryEditingId || libraryUpdateSaving) return;
                                  setLibraryUpdateSaving(true);
                                  fetch(`${BACKEND_URL}/library/${encodeURIComponent(libraryEditingId)}`, {
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
              ) : recommendationsLoading && !recommendations.books.length && !recommendations.podcasts.length && !recommendations.articles.length && !recommendations.research.length ? (
                <p className="text-slate-500 text-sm">Loading recommendations… This can take up to a minute.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 pb-4">
                  {/* Books */}
                  <section className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 flex flex-col">
                    <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">Books</h3>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.books.length === 0 ? (
                        <p className="text-slate-500 text-xs">No book suggestions right now.</p>
                      ) : (
                        recommendations.books.map((item, i) => {
                          const cardKey = `book:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`book-${i}-${item.title}`}
                              className={`rounded-lg bg-slate-800/50 p-3 border border-slate-700/50 transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              <a
                                href={`https://www.amazon.com/s?k=${encodeURIComponent([item.title, item.author].filter(Boolean).join(" "))}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-slate-200 text-sm font-medium hover:text-violet-300 hover:underline cursor-pointer"
                              >
                                {item.title}
                              </a>
                              {item.author && <p className="text-slate-500 text-xs mt-0.5">{item.author}</p>}
                              {item.reason && <p className="text-slate-400 text-xs mt-1">{item.reason}</p>}
                              <button
                                type="button"
                                onClick={() => markConsumed("book", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-violet-600/80 hover:bg-violet-500/80 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
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
                  <section className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 flex flex-col">
                    <div className="flex flex-wrap items-baseline gap-2 mb-3">
                      <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Podcasts</h3>
                      <a
                        href="https://www.listennotes.com/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] text-slate-500 hover:text-slate-400 transition-colors flex items-baseline gap-0.5"
                        title="Podcast data by Listen Notes"
                      >
                        <span className="lowercase font-normal">powered by</span>
                        <span className="uppercase font-semibold text-slate-400">LISTEN NOTES</span>
                      </a>
                    </div>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.podcasts.length === 0 ? (
                        <p className="text-slate-500 text-xs">No podcast suggestions right now.</p>
                      ) : (
                        recommendations.podcasts.map((item, i) => {
                          const cardKey = `podcast:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`podcast-${i}-${item.title}`}
                              className={`rounded-lg bg-slate-800/50 p-3 border border-slate-700/50 transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              {item.author && (
                                <p className="text-slate-500 text-xs font-medium">{item.author}</p>
                              )}
                              <a
                                href={
                                  item.url && (item.url.includes("spotify.com") || item.url.includes("podcasts.apple.com") || item.url.includes("listennotes.com"))
                                    ? item.url
                                    : `https://open.spotify.com/search/${encodeURIComponent([item.author, item.title].filter(Boolean).join(" "))}`
                                }
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-slate-200 text-sm font-medium hover:text-violet-300 hover:underline cursor-pointer block"
                              >
                                {item.title}
                              </a>
                              {(!item.url || (!item.url.includes("spotify.com") && !item.url.includes("podcasts.apple.com") && !item.url.includes("listennotes.com"))) && (
                                <p className="text-xs mt-0.5 flex gap-2 flex-wrap">
                                  <a
                                    href={`https://open.spotify.com/search/${encodeURIComponent([item.author, item.title].filter(Boolean).join(" "))}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-violet-400 hover:underline"
                                  >
                                    Spotify
                                  </a>
                                  <a
                                    href={`https://podcasts.apple.com/us/search?term=${encodeURIComponent([item.author, item.title].filter(Boolean).join(" "))}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-violet-400 hover:underline"
                                  >
                                    Apple Podcasts
                                  </a>
                                </p>
                              )}
                              {item.reason && <p className="text-slate-400 text-xs mt-1">{item.reason}</p>}
                              <button
                                type="button"
                                onClick={() => markConsumed("podcast", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-violet-600/80 hover:bg-violet-500/80 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
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
                  <section className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 flex flex-col">
                    <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">News & articles</h3>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.articles.length === 0 ? (
                        <p className="text-slate-500 text-xs">No article suggestions right now.</p>
                      ) : (
                        recommendations.articles.map((item, i) => {
                          const cardKey = `article:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`article-${i}-${item.title}`}
                              className={`rounded-lg bg-slate-800/50 p-3 border border-slate-700/50 transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              <a
                                href={item.url && item.url.startsWith("http") ? item.url : `https://www.google.com/search?q=${encodeURIComponent([item.title, item.author].filter(Boolean).join(" "))}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-slate-200 text-sm font-medium hover:text-violet-300 hover:underline cursor-pointer"
                              >
                                {item.title}
                              </a>
                              {item.author && <p className="text-slate-500 text-xs mt-0.5">{item.author}</p>}
                              {item.reason && <p className="text-slate-400 text-xs mt-1">{item.reason}</p>}
                              <button
                                type="button"
                                onClick={() => markConsumed("article", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-violet-600/80 hover:bg-violet-500/80 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
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
                  <section className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 flex flex-col">
                    <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">Research papers</h3>
                    <div className="space-y-3 overflow-y-auto flex-1 min-h-0">
                      {recommendations.research.length === 0 ? (
                        <p className="text-slate-500 text-xs">No research suggestions right now.</p>
                      ) : (
                        recommendations.research.map((item, i) => {
                          const cardKey = `research:${item.title}`;
                          const isRemoving = removingKeys.has(cardKey);
                          return (
                            <div
                              key={`research-${i}-${item.title}`}
                              className={`rounded-lg bg-slate-800/50 p-3 border border-slate-700/50 transition-all duration-300 ease-out ${
                                isRemoving ? "opacity-0 -translate-x-4 scale-95 pointer-events-none" : ""
                              }`}
                            >
                              <a
                                href={item.url && item.url.startsWith("http") ? item.url : `https://www.google.com/search?q=${encodeURIComponent([item.title, item.author].filter(Boolean).join(" "))}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-slate-200 text-sm font-medium hover:text-violet-300 hover:underline cursor-pointer"
                              >
                                {item.title}
                              </a>
                              {item.author && <p className="text-slate-500 text-xs mt-0.5">{item.author}</p>}
                              {item.reason && <p className="text-slate-400 text-xs mt-1">{item.reason}</p>}
                              <button
                                type="button"
                                onClick={() => markConsumed("research", item)}
                                disabled={consumedIds.has(cardKey)}
                                className="mt-2 px-2 py-1 rounded text-xs font-medium bg-violet-600/80 hover:bg-violet-500/80 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
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
                  <div className="relative bg-slate-900 border border-slate-700 rounded-xl shadow-xl max-w-lg w-full max-h-[85vh] flex flex-col">
                    <div className="p-4 border-b border-slate-700 flex items-center justify-between gap-2 flex-shrink-0">
                      <h2 id="library-interview-title" className="text-slate-200 font-medium text-lg">
                        Interview about your books
                      </h2>
                      <button
                        type="button"
                        onClick={() => setShowLibraryInterview(false)}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-700/60 transition-colors"
                        aria-label="Close"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    <p className="px-4 pb-2 text-xs text-slate-500 flex-shrink-0">
                      Chat about what you liked (or didn’t) — we’ll save short notes to improve recommendations. No hallucination; notes are brief and factual.
                    </p>
                    <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
                      {libraryInterviewMessages.length === 0 && !libraryInterviewLoading && (
                        <p className="text-slate-500 text-sm">Say &quot;Start&quot; or &quot;Hi&quot; to begin. The agent will ask about your books one by one.</p>
                      )}
                      {libraryInterviewMessages.map((m, i) => (
                        <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                          <div
                            className={`max-w-[90%] rounded-lg px-3 py-2 text-sm ${
                              m.role === "user"
                                ? "bg-violet-600/30 text-violet-100"
                                : "bg-slate-800/80 text-slate-200 border border-slate-700/50"
                            }`}
                          >
                            <span className="font-medium text-slate-400 text-xs block mb-0.5">{m.role === "user" ? "You" : "Agent"}</span>
                            {m.content}
                          </div>
                        </div>
                      ))}
                      {libraryInterviewLoading && (
                        <div className="flex justify-start">
                          <div className="rounded-lg px-3 py-2 text-sm bg-slate-800/80 text-slate-400 border border-slate-700/50">
                            Thinking…
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="p-4 border-t border-slate-700 flex gap-2 flex-shrink-0">
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
                        className="flex-1 min-w-0 px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500/50"
                        disabled={libraryInterviewLoading}
                      />
                      <button
                        type="button"
                        onClick={() => {
                          const msg = libraryInterviewInput.trim();
                          if (msg && !libraryInterviewLoading) sendLibraryInterviewMessage(msg);
                        }}
                        disabled={libraryInterviewLoading || !libraryInterviewInput.trim()}
                        className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-500/50"
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

      {/* Footer */}
      <footer className="flex-none z-0 bg-slate-950/80 backdrop-blur-sm py-2 px-4 text-center space-y-2 border-t border-slate-800/60">
        <p className="text-xs text-slate-500">
          {!isConnected
            ? "Connect to begin your journaling session."
            : isProcessing
              ? "Thinking..."
              : "Speak naturally. The AI is listening."}
        </p>
        <div className="pt-2 space-y-1">
          <p className="text-[10px] text-slate-600">
            By John Stewart, Sherelle McDaniel, Aniyah Tucker, Dominique Sanchez, Andy Coto, Jackeline Garcia Ulloa
          </p>
          <p className="text-[10px] text-slate-600 flex items-center justify-center gap-2 flex-wrap">
            <a
              href="https://github.com/MrFunnything99/Open-Journal"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-slate-500 hover:text-violet-400 transition-colors"
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
