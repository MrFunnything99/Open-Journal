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
import { MemoryDiagram } from "./components/MemoryDiagram";

/** Default journaling assistant prompt (base; personalization suffix is appended from slider) */
const DEFAULT_PERSONAPLEX_PROMPT = `You are an empathetic and insightful conversational journaling assistant. Your goal is to provide a supportive space for the user to reflect on their thoughts, experiences, and emotions. Read the user's entries and respond naturally. Ask open-ended questions to encourage further exploration, but always let the user guide the direction and depth of the conversation. Avoid being overly prescriptive, giving unsolicited advice, or summarizing their thoughts unnecessarily. Just be a curious, active listener. Always facilitate conversation that gets the user exploring their thoughts and emotions. Try to keep responses brief and concise when possible to conserve tokens.`;

const PERSONALIZATION_LEVELS = [0, 0.25, 0.5, 0.75, 1] as const;
const PERSONALIZATION_LABELS: Record<number, string> = {
  0: "Do not use memory or prior context; keep questions general and present-focused only.",
  0.25: "Keep questions general and present-focused.",
  0.5: "Occasionally reference what you know about the user.",
  0.75: "Often tailor questions to the user's life and past entries.",
  1: "Fully personalize: connect deeply to the user's life and past journals.",
};

const INTRUSIVENESS_LEVELS = [0, 0.25, 0.5, 0.75, 1] as const;
const INTRUSIVENESS_LABELS: Record<number, string> = {
  0: "Very gentle; soft, open-ended questions only; let the user lead entirely.",
  0.25: "Ask sparingly; avoid probing.",
  0.5: "Balanced; follow the user's lead but ask when it helps.",
  0.75: "More direct; you may ask probing questions when supportive.",
  1: "More direct or probing when it feels supportive; still respect boundaries.",
};

const DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // Rachel
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

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
  const [personalization, setPersonalization] = useState(0.5);
  const [intrusiveness, setIntrusiveness] = useState(0.5);
  const [voices, setVoices] = useState<VoiceOption[]>(FALLBACK_VOICES);
  const [selectedVoiceId, setSelectedVoiceId] = useState(DEFAULT_VOICE_ID);
  const [voiceDynamics, setVoiceDynamics] = useState(0.5);
  // Manual mode (commented out – single flow: VAD + "I'm done open journal" to stop)
  // const [manualMode, setManualMode] = useState(false);
  const voiceSettings = useMemo(
    () => ({
      stability: 1 - voiceDynamics,
      similarity_boost: 0.75,
      style: voiceDynamics,
    }),
    [voiceDynamics]
  );
  const textPrompt = useMemo(
    () =>
      DEFAULT_PERSONAPLEX_PROMPT +
      "\n\nLevel of personalization: " +
      Math.round(personalization * 100) +
      "%. " +
      (PERSONALIZATION_LABELS[personalization as keyof typeof PERSONALIZATION_LABELS] ?? PERSONALIZATION_LABELS[0.5]) +
      "\n\nQuestioning style (intrusiveness): " +
      Math.round(intrusiveness * 100) +
      "%. " +
      (INTRUSIVENESS_LABELS[intrusiveness as keyof typeof INTRUSIVENESS_LABELS] ?? INTRUSIVENESS_LABELS[0.5]),
    [personalization, intrusiveness]
  );
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [expandedLogIndex, setExpandedLogIndex] = useState<number | null>(null);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [view, setView] = useState<"session" | "history" | "memory">("session");
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [priorJournalText, setPriorJournalText] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [memoryStats, setMemoryStats] = useState<{ gist_facts_count: number; episodic_log_count: number } | null>(null);
  const [isWipingMemory, setIsWipingMemory] = useState(false);

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
    if (view === "memory") fetchMemoryStats();
  }, [view, fetchMemoryStats]);

  const handleIngestPriorJournal = useCallback(() => {
    if (!priorJournalText.trim()) return;
    setIsIngesting(true);
    const textToIngest = priorJournalText.trim();
    fetch(`${BACKEND_URL}/ingest-history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: textToIngest }),
    })
      .then((r) => {
        if (!r.ok) return r.json().then((d) => Promise.reject(new Error(d.detail ?? "Ingest failed")));
      })
      .then(() => {
        setPriorJournalText("");
        saveEntry([{ role: "user", text: textToIngest }]);
        setToastMessage("Journal added to memory and saved to History.");
        setTimeout(() => setToastMessage(null), 5000);
        fetchMemoryStats();
      })
      .catch((err) => {
        setToastMessage(err instanceof Error ? err.message : "Failed to add to memory");
        setTimeout(() => setToastMessage(null), 4000);
      })
      .finally(() => setIsIngesting(false));
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
    manualMode: false,
    personalization,
    intrusiveness,
    voiceSettings,
    onTranscriptUpdate: useCallback((updater) => {
      setTranscript((prev) => {
        const next = typeof updater === "function" ? updater(prev) : updater;
        return next;
      });
    }, []),
    onInterimTranscript: setInterimTranscript,
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
      const file = e.target.files?.[0];
      e.target.value = "";
      if (!file) return;
      setIsImporting(true);
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const text = reader.result;
          if (typeof text !== "string") throw new Error("Invalid file");
          const parsed = JSON.parse(text) as unknown;
          if (!isExportPayload(parsed)) throw new Error("Not a valid OpenJournal export file");
          const count = importEntriesFromExport(parsed);
          if (count === 0) {
            setToastMessage("No valid entries in file.");
            setTimeout(() => setToastMessage(null), 4000);
            return;
          }
          setToastMessage(`Imported ${count} journal${count === 1 ? "" : "s"}. Syncing to memory…`);
          const entries = parsed.entries as { fullTranscript?: { role: string; text: string }[] }[];
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
                body: JSON.stringify({ text: transcriptText }),
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
    [importEntriesFromExport, isExportPayload, fetchMemoryStats]
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
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v3m0 0V10m0 3V7a3 3 0 016 0v3m-4 0h.01M12 19h.01M12 16h.01M12 13h.01M12 10h.01M12 7h.01M8 19h.01M8 16h.01" />
            </svg>
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
            title="Memory visualization"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span className="hidden sm:inline">Memory</span>
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
                <label htmlFor="personaplex-voice-dynamics" className="block text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Voice dynamics
                </label>
                <input
                  id="personaplex-voice-dynamics"
                  type="range"
                  min={0}
                  max={100}
                  value={Math.round(voiceDynamics * 100)}
                  onChange={(e) => setVoiceDynamics(Number(e.target.value) / 100)}
                  disabled={isConnected}
                  className="w-full h-2 rounded-full bg-slate-600 accent-violet-500 disabled:opacity-60"
                />
                <p className="text-xs text-slate-500">
                  {Math.round(voiceDynamics * 100)}% — calmer ↔ more expressive
                </p>
              </div>
              {/* Manual mode (commented out – single flow: say "I'm done open journal" to stop)
              {!isVoiceMemoMode && (
                <div className="mt-3 space-y-1.5">
                  <label ...>
                    <span>Manual mode</span>
                    <input type="checkbox" checked={manualMode} onChange={...} />
                  </label>
                  <p>Tap "Done speaking" when finished (no auto-detect).</p>
                </div>
              )}
              */}
              <div className="mt-3 space-y-1.5">
                <label htmlFor="personaplex-personalization" className="block text-xs font-medium text-slate-400 uppercase tracking-wider">
                  Personalization
                </label>
                <div className="hidden sm:block">
                  <input
                    id="personaplex-personalization"
                    type="range"
                    min={0}
                    max={PERSONALIZATION_LEVELS.length - 1}
                    step={1}
                    value={Math.max(0, PERSONALIZATION_LEVELS.findIndex((p) => p === personalization))}
                    onChange={(e) => setPersonalization(PERSONALIZATION_LEVELS[Number(e.target.value)])}
                    disabled={isConnected}
                    className="w-full h-2 rounded-full bg-slate-600 accent-violet-500 disabled:opacity-60"
                    aria-valuenow={Math.round(personalization * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuetext={`${Math.round(personalization * 100)}%`}
                  />
                </div>
                <select
                  aria-label="Personalization level"
                  value={personalization}
                  onChange={(e) => setPersonalization(Number(e.target.value))}
                  disabled={isConnected}
                  className="sm:hidden w-full px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700/50 text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                >
                  {PERSONALIZATION_LEVELS.map((p) => (
                    <option key={p} value={p}>
                      {Math.round(p * 100)}%
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-500">
                  {Math.round(personalization * 100)}%
                </p>
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
                  {Math.round(intrusiveness * 100)}% — gentle ↔ more probing
                </p>
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
                {/* Manual mode: "Done speaking" button (commented out)
                {!isVoiceMemoMode && isConnected && manualMode && isUserSpeaking && (
                  <button type="button" onClick={commitManual} ...>Done speaking</button>
                )}
                */}
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
                      : "Conversation will appear here. Say \"I'm done open journal\" when you're finished. Say \"open journal\" to interrupt the AI."}
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
            view === "history" || view === "memory" ? "opacity-100" : "opacity-0 pointer-events-none absolute inset-0"
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
                      accept=".json,application/json"
                      onChange={handleImportFileChange}
                      className="hidden"
                      aria-label="Import journals from file"
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
                    {isIngesting ? "Adding…" : "Add to memory"}
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
            <div className="flex-1 flex flex-col min-h-0 p-4 md:p-6 overflow-auto">
              <h2 className="text-lg font-medium text-slate-300 uppercase tracking-wider mb-4 flex-shrink-0">
                Memory visualization
              </h2>
              <div className="flex flex-col lg:flex-row gap-4 flex-1 min-h-0">
                <div className="flex-shrink-0">
                  <div className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-6 max-w-md">
                    <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">
                      Memory stats
                    </h3>
                    <p className="text-xs text-slate-500 mb-4">
                      Vector DB counts. The AI uses this memory when personalization is above 0%.
                    </p>
                    {memoryStats !== null ? (
                      <div className="space-y-3 text-sm text-slate-300">
                        <p>
                          Gist facts: <span className="font-medium text-violet-300">{memoryStats.gist_facts_count}</span>
                        </p>
                        <p>
                          Episodic summaries: <span className="font-medium text-violet-300">{memoryStats.episodic_log_count}</span>
                        </p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <button
                            type="button"
                            onClick={fetchMemoryStats}
                            className="px-3 py-1.5 rounded-lg bg-slate-700/50 text-slate-400 text-xs font-medium hover:bg-slate-600/50 transition-colors"
                          >
                            Refresh
                          </button>
                          <button
                            type="button"
                            onClick={handleWipeMemory}
                            disabled={isWipingMemory}
                            className="px-3 py-1.5 rounded-lg bg-red-900/40 text-red-300 text-xs font-medium hover:bg-red-800/50 disabled:opacity-50 transition-colors"
                          >
                            {isWipingMemory ? "Wiping…" : "Wipe memory"}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <p className="text-xs text-slate-500">Backend not reached. Start the Python backend and try again.</p>
                    )}
                  </div>
                </div>
                <div className="flex-1 min-h-0 min-w-0 flex flex-col">
                  <MemoryDiagram onRefreshStats={fetchMemoryStats} />
                </div>
              </div>
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
