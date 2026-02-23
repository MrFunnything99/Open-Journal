import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type VoiceOption = { voice_id: string; name: string };
import { usePersonaplexSession } from "./hooks/usePersonaplexSession";
import { useJournalHistory } from "./hooks/useJournalHistory";
import { Orb, OrbState } from "./components/Orb";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { ConnectButton } from "./components/ConnectButton";
import { JournalGallery } from "./components/JournalGallery";

/** Default journaling assistant prompt */
const DEFAULT_PERSONAPLEX_PROMPT = `You are an empathetic and insightful conversational journaling assistant. Your goal is to provide a supportive space for the user to reflect on their thoughts, experiences, and emotions. Read the user's entries and respond naturally. Ask open-ended questions to encourage further exploration, but always let the user guide the direction and depth of the conversation. Avoid being overly prescriptive, giving unsolicited advice, or summarizing their thoughts unnecessarily. Just be a curious, active listener. Always facilitate conversation that gets the user exploring their thoughts and emotions. Try to keep responses brief and concise when possible to conserve tokens.`;

const DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // Rachel

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
  const [textPrompt, setTextPrompt] = useState(DEFAULT_PERSONAPLEX_PROMPT);
  const [voices, setVoices] = useState<VoiceOption[]>(FALLBACK_VOICES);
  const [selectedVoiceId, setSelectedVoiceId] = useState(DEFAULT_VOICE_ID);
  const [manualMode, setManualMode] = useState(false);
  const [transcript, setTranscript] = useState<Array<{ role: "user" | "ai"; text: string }>>([]);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [showGallery, setShowGallery] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const { entries, saveEntry, deleteEntry, getFormattedDate } = useJournalHistory();

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

  const {
    status,
    errorMessage,
    connect,
    disconnect,
    commitManual,
    isConnected,
    isUserSpeaking,
    isAiSpeaking,
  } = usePersonaplexSession({
    systemPrompt: textPrompt,
    selectedVoiceId,
    manualMode,
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
    return "idle";
  }, [isUserSpeaking, isAiSpeaking]);

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
      <header className="flex-none relative z-10 flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 gap-2 flex-wrap sm:flex-nowrap">
        <div className="flex items-center gap-2 sm:gap-4 min-w-0">
          <h1 className="text-base sm:text-xl font-light tracking-widest text-slate-300 uppercase truncate">
            OpenJournal
          </h1>
          <ConnectionStatus status={status} />
          {errorMessage && (
            <span className="text-sm text-red-400">{errorMessage}</span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
            <button
            type="button"
            onClick={() => setShowGallery((v) => !v)}
            className="px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg bg-slate-700/50 text-slate-300 text-sm font-medium hover:bg-slate-600/50 transition-colors flex items-center gap-2"
            title={showGallery ? "Back to session" : "Journal history"}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              aria-hidden
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
              />
            </svg>
            {showGallery ? "Session" : "History"}
          </button>
          <ConnectButton
            status={status}
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
          />
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
          className={`flex-1 min-h-0 p-4 md:p-6 transition-opacity duration-300 ${
            showGallery ? "opacity-0 pointer-events-none absolute inset-0" : "opacity-100"
          }`}
        >
          {/* 3-column grid: desktop | single column: mobile/tablet */}
          <div className="h-full min-h-0 grid grid-cols-1 lg:grid-cols-[1fr_2fr_1fr] gap-4 lg:gap-6 grid-rows-[auto auto minmax(0,1fr)] lg:grid-rows-[minmax(0,1fr)]">
            {/* Left column - Settings (order 1 on mobile) */}
            <div className="order-1 lg:order-none flex flex-col min-h-0 rounded-xl bg-slate-900/50 border border-slate-700/50 p-4">
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">
                Settings
              </h2>
              <div>
                <label htmlFor="personaplex-voice" className="block text-sm font-medium text-slate-400 uppercase tracking-wider mb-1.5">
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
              <div className="mt-4">
                <label
                  className={`flex flex-col sm:flex-row sm:items-center gap-2 cursor-pointer select-none ${
                    isConnected ? "cursor-not-allowed opacity-70" : ""
                  }`}
                >
                  <span className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                    Manual mode
                  </span>
                  <div className="relative w-11 h-6 shrink-0">
                    <input
                      type="checkbox"
                      checked={manualMode}
                      onChange={(e) => setManualMode(e.target.checked)}
                      disabled={isConnected}
                      className="peer sr-only"
                    />
                    <div
                      className="absolute inset-0 rounded-full bg-slate-600 transition-colors duration-200 ease-out
                        peer-checked:bg-violet-500 peer-focus-visible:ring-2 peer-focus-visible:ring-violet-500/50 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-slate-900"
                    />
                    <div
                      className="absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm
                        transition-transform duration-200 ease-out peer-checked:translate-x-5"
                    />
                  </div>
                </label>
                <p className="mt-1.5 text-xs text-slate-500">
                  When on, tap &quot;Done speaking&quot; when you finish instead of waiting for auto-detection.
                </p>
              </div>
            </div>

            {/* Center column - Prompt, Orb (order 2 on mobile) */}
            <div className="order-2 lg:order-none flex flex-col items-center justify-center gap-2 sm:gap-4 min-h-0 py-2 sm:py-4 lg:py-0">
              <div className="w-full max-w-md">
                <label htmlFor="personaplex-text-prompt" className="block text-sm font-medium text-slate-400 uppercase tracking-wider mb-1.5">
                  System Prompt
                </label>
                <textarea
                  id="personaplex-text-prompt"
                  value={textPrompt}
                  onChange={(e) => setTextPrompt(e.target.value)}
                  disabled={isConnected}
                  rows={6}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900/80 border border-slate-700/50 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50 disabled:opacity-60 disabled:cursor-not-allowed resize-y min-h-[120px] max-h-[200px]"
                  placeholder="Enter system prompt for the AI..."
                />
              </div>
              <div className="flex-none flex flex-col items-center gap-3">
                <Orb state={orbState} />
                {isConnected && manualMode && isUserSpeaking && (
                  <button
                    type="button"
                    onClick={commitManual}
                    className="px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium transition-colors"
                  >
                    Done speaking
                  </button>
                )}
              </div>
            </div>

            {/* Right column - Transcript (order 3 on mobile) */}
            <div
              className="order-3 lg:order-none flex min-h-0 flex-col rounded-xl bg-slate-900/50 border border-slate-700/50 overflow-hidden"
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
                    Conversation will appear here as you speak...
                  </p>
                ) : (
                  <>
                    {transcript.map((entry, i) => (
                      <div
                        key={i}
                        className={`text-sm flex ${
                          entry.role === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        <div
                          className={`max-w-[85%] break-words ${
                            entry.role === "user"
                              ? "text-violet-200 text-right"
                              : "text-slate-300 text-left"
                          }`}
                        >
                          <span className="font-medium opacity-80 block mb-0.5">
                            {entry.role === "user" ? "You" : "AI"}
                          </span>
                          {entry.text}
                        </div>
                      </div>
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
          className={`flex-1 flex flex-col min-h-0 transition-opacity duration-300 ${
            showGallery ? "opacity-100" : "opacity-0 pointer-events-none absolute inset-0"
          }`}
        >
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
      </main>

      {/* Footer */}
      <footer className="flex-none z-0 bg-slate-950/80 backdrop-blur-sm py-2 px-4 text-center space-y-2 border-t border-slate-800/60">
        <p className="text-xs text-slate-500">
          {isConnected
            ? "Speak naturally. The AI is listening."
            : "Connect to begin your journaling session."}
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
