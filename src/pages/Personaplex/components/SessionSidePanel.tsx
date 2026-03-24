import { useEffect, type Ref } from "react";
import type { PersonaplexConnectionStatus, TranscriptEntry } from "../hooks/usePersonaplexSession";
import { Orb, type OrbState } from "./Orb";
import { TranscriptBubble } from "./TranscriptBubble";
import { ConnectButton } from "./ConnectButton";
import { ConnectionStatus } from "./ConnectionStatus";

export type VoiceOption = { voice_id: string; name: string };

export type SessionSidePanelProps = {
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
  settingsExpanded: boolean;
  onToggleSettings: () => void;
  status: PersonaplexConnectionStatus;
  onConnect: () => void;
  onDisconnect: () => void;
  errorMessage: string | null;
  sessionMode: "journal" | "recommendations";
  setSessionMode: (v: "journal" | "recommendations") => void;
  voices: VoiceOption[];
  selectedVoiceId: string;
  setSelectedVoiceId: (id: string) => void;
  showLiveTranscription: boolean;
  setShowLiveTranscription: (v: boolean | ((p: boolean) => boolean)) => void;
  isVoiceMemoMode: boolean;
  inputMode: "voice" | "text";
  handleInputModeChange: (mode: "voice" | "text") => void;
  isModeToggleLocked: boolean;
  isConnected: boolean;
  orbState: OrbState;
  thinkingProgress: number;
  isAiSpeaking: boolean;
  isUserSpeaking: boolean;
  isProcessing: boolean;
  isVoiceMemoRecording: boolean;
  startVoiceMemoRecording: () => void;
  stopVoiceMemoRecording: () => void;
  lastPlaybackFailed: boolean;
  playLastFailedPlayback: () => void;
  commitManual: () => void;
  typedInput: string;
  setTypedInput: (s: string) => void;
  handleSendTypedInput: () => void;
  transcript: TranscriptEntry[];
  interimTranscript: string;
  expandedLogIndex: number | null;
  setExpandedLogIndex: (v: number | null | ((p: number | null) => number | null)) => void;
  transcriptScrollRef: Ref<HTMLDivElement>;
  handleTranscriptScroll: () => void;
};

export function SessionSidePanel({
  open,
  onOpen,
  onClose,
  settingsExpanded,
  onToggleSettings,
  status,
  onConnect,
  onDisconnect,
  errorMessage,
  sessionMode,
  setSessionMode,
  voices,
  selectedVoiceId,
  setSelectedVoiceId,
  showLiveTranscription,
  setShowLiveTranscription,
  isVoiceMemoMode,
  inputMode,
  handleInputModeChange,
  isModeToggleLocked,
  isConnected,
  orbState,
  thinkingProgress,
  isAiSpeaking,
  isUserSpeaking,
  isProcessing,
  isVoiceMemoRecording,
  startVoiceMemoRecording,
  stopVoiceMemoRecording,
  lastPlaybackFailed,
  playLastFailedPlayback,
  commitManual,
  typedInput,
  setTypedInput,
  handleSendTypedInput,
  transcript,
  interimTranscript,
  expandedLogIndex,
  setExpandedLogIndex,
  transcriptScrollRef,
  handleTranscriptScroll,
}: SessionSidePanelProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <>
      {open && (
        <button
          type="button"
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
          aria-label="Close session panel"
          onClick={onClose}
        />
      )}

      {!open && (
        <button
          type="button"
          onClick={onOpen}
          className="fixed right-0 top-1/2 z-30 flex -translate-y-1/2 flex-col items-center gap-1 rounded-l-2xl border border-r-0 border-white/15 bg-white/[0.08] py-3 pl-2 pr-1.5 text-xs font-medium text-white/85 shadow-[0_8px_32px_rgba(0,0,0,0.35)] backdrop-blur-xl transition-colors hover:bg-white/[0.12]"
          title="Open journaling session"
          aria-label="Open journaling session"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-teal-300/90" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
          <span className="max-w-[2.5rem] text-center leading-tight [writing-mode:vertical-rl] rotate-180 sm:[writing-mode:horizontal-tb] sm:rotate-0 sm:max-w-none">
            Session
          </span>
        </button>
      )}

      <aside
        className={`fixed right-0 top-0 z-50 flex h-full w-[min(100vw,420px)] flex-col border-l border-white/10 bg-[#0f0f18]/85 shadow-2xl backdrop-blur-2xl transition-transform duration-300 ease-out ${
          open ? "translate-x-0" : "translate-x-full pointer-events-none"
        }`}
        aria-hidden={!open}
      >
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex flex-none shrink-0 items-center justify-between gap-2 border-b border-gray-200 px-3 py-2.5 dark:border-gray-700">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">Session</h2>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={onToggleSettings}
                className={`rounded-lg p-2 transition-colors ${
                  settingsExpanded
                    ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
                    : "text-gray-500 hover:bg-gray-100 dark:hover:bg-[#343541]"
                }`}
                title={settingsExpanded ? "Hide settings" : "Show settings"}
                aria-expanded={settingsExpanded}
                aria-label={settingsExpanded ? "Hide settings" : "Show settings"}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                  />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 dark:hover:bg-[#343541]"
                aria-label="Close session panel"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          <div className="flex-none space-y-3 border-b border-gray-200 px-3 py-3 dark:border-gray-700">
            <div className="flex flex-wrap items-center gap-2">
              <ConnectionStatus status={status} className="text-xs shrink-0" />
              <ConnectButton
                status={status}
                onConnect={onConnect}
                onDisconnect={onDisconnect}
                className="min-h-[40px] px-4 py-2 text-sm"
              />
            </div>
            {errorMessage && <p className="text-xs text-red-400">{errorMessage}</p>}
            <div className="space-y-1.5">
              <span className="block text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">Input mode</span>
              <div
                className={`inline-flex w-full max-w-full rounded-full border border-gray-100 bg-gray-50/80 p-1 transition-opacity dark:border-gray-600 dark:bg-[#343541] ${
                  isModeToggleLocked ? "opacity-60" : "opacity-100"
                }`}
              >
                <button
                  type="button"
                  onClick={() => handleInputModeChange("voice")}
                  disabled={isModeToggleLocked}
                  className={`min-h-[40px] flex-1 rounded-full px-3 py-2 text-xs font-medium transition-colors ${
                    inputMode === "voice"
                      ? "bg-emerald-500 text-white shadow-sm"
                      : "text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-[#404040]"
                  }`}
                  aria-pressed={inputMode === "voice"}
                  title={isModeToggleLocked ? "Wait for AI to finish before switching modes" : "Use voice input"}
                >
                  Voice
                </button>
                <button
                  type="button"
                  onClick={() => handleInputModeChange("text")}
                  disabled={isModeToggleLocked}
                  className={`min-h-[40px] flex-1 rounded-full px-3 py-2 text-xs font-medium transition-colors ${
                    inputMode === "text"
                      ? "bg-emerald-500 text-white shadow-sm"
                      : "text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-[#404040]"
                  }`}
                  aria-pressed={inputMode === "text"}
                  title={isModeToggleLocked ? "Wait for AI to finish before switching modes" : "Use text input"}
                >
                  Text
                </button>
              </div>
              {isModeToggleLocked && (
                <p className="text-[11px] text-gray-500 dark:text-gray-400">Input mode locks until AI finishes.</p>
              )}
            </div>
          </div>

          {settingsExpanded && (
            <div className="flex-none overflow-y-auto border-b border-gray-200 p-4 dark:border-gray-700">
              <div>
                <label htmlFor="personaplex-session-mode-panel" className="mb-1.5 block text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">
                  Session mode
                </label>
                <select
                  id="personaplex-session-mode-panel"
                  value={sessionMode}
                  onChange={(e) => setSessionMode(e.target.value as "journal" | "recommendations")}
                  disabled={isConnected}
                  className="w-full rounded-lg border border-gray-100 bg-gray-50/90 px-3 py-2 text-sm text-gray-800 focus:border-emerald-400/80 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-600 dark:bg-[#343541] dark:text-gray-100"
                  aria-label="Interview mode: journal or recommendations"
                >
                  <option value="journal">Interview (journal entries)</option>
                  <option value="recommendations">Interview (consumed media)</option>
                </select>
                {sessionMode === "recommendations" && (
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Same speech-to-text; agent asks about your library and saves short notes for better recommendations.
                  </p>
                )}
              </div>
              <div className="mt-3">
                <label htmlFor="personaplex-voice-panel" className="mb-1.5 block text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">
                  Voice
                </label>
                <select
                  id="personaplex-voice-panel"
                  value={selectedVoiceId}
                  onChange={(e) => setSelectedVoiceId(e.target.value)}
                  disabled={isConnected}
                  className="w-full rounded-lg border border-gray-100 bg-gray-50/90 px-3 py-2 text-sm text-gray-800 focus:border-emerald-400/80 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-600 dark:bg-[#343541] dark:text-gray-100"
                >
                  {voices.map((v) => (
                    <option key={v.voice_id} value={v.voice_id}>
                      {v.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="mt-3 space-y-1.5">
                <label className="block text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">Live transcription</label>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setShowLiveTranscription((v) => !v)}
                    className={`relative h-5 w-10 rounded-full transition-colors duration-200 ease-out ${
                      showLiveTranscription ? "bg-[#10a37f]" : "bg-gray-300 dark:bg-gray-600"
                    }`}
                    aria-pressed={showLiveTranscription}
                    aria-label="Toggle live transcription"
                    disabled={isVoiceMemoMode}
                  >
                    <span
                      className={`absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200 ease-out ${
                        showLiveTranscription ? "translate-x-5" : ""
                      }`}
                    />
                  </button>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {isVoiceMemoMode
                      ? "On mobile, transcription appears after you tap Done."
                      : showLiveTranscription
                        ? "Show words as you speak."
                        : "Only show what you said after you tap Done."}
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className="flex flex-none flex-col items-center justify-center gap-2 border-b border-gray-200 px-3 py-4 dark:border-gray-700">
            <div className="flex flex-col items-center justify-center gap-3">
              <Orb state={orbState} thinkingProgress={thinkingProgress} />
              {inputMode === "voice" && isVoiceMemoMode && isConnected &&
                (isVoiceMemoRecording ? (
                  <button
                    type="button"
                    onClick={stopVoiceMemoRecording}
                    className="rounded-full bg-red-500/80 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-red-500"
                  >
                    Done
                  </button>
                ) : lastPlaybackFailed ? (
                  <button
                    type="button"
                    onClick={playLastFailedPlayback}
                    className="rounded-full bg-[#10a37f] px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-[#0d8c6e]"
                  >
                    Play response
                  </button>
                ) : !isAiSpeaking ? (
                  <button
                    type="button"
                    onClick={startVoiceMemoRecording}
                    className="rounded-full bg-emerald-500/80 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-emerald-500"
                  >
                    Record
                  </button>
                ) : null)}
              {inputMode === "voice" && !isVoiceMemoMode && isConnected && isUserSpeaking && !isAiSpeaking && (
                <button
                  type="button"
                  onClick={commitManual}
                  className="rounded-full bg-[#10a37f] px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-[#0d8c6e]"
                >
                  Done recording
                </button>
              )}
            </div>
            {inputMode === "text" && isConnected && (
              <div className="mt-2 w-full rounded-2xl border border-gray-100 bg-white p-3 shadow-sm dark:rounded-xl dark:border-gray-700 dark:bg-[#343541]">
                <label htmlFor="typed-session-input-panel" className="mb-2 block text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">
                  Type to AI
                </label>
                <div className="flex flex-col gap-2">
                  <textarea
                    id="typed-session-input-panel"
                    value={typedInput}
                    onChange={(e) => setTypedInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendTypedInput();
                      }
                    }}
                    placeholder="Type your message..."
                    rows={4}
                    className="h-24 w-full resize-none overflow-y-auto rounded-lg border border-gray-100 bg-gray-50/90 px-3 py-2 text-sm text-gray-800 focus:border-emerald-400/80 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 dark:border-gray-600 dark:bg-[#343541] dark:text-gray-100"
                  />
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-gray-500 dark:text-gray-400">Send a typed turn to continue the session.</span>
                    <button
                      type="button"
                      onClick={handleSendTypedInput}
                      disabled={!isConnected || !typedInput.trim() || isProcessing || isAiSpeaking}
                      className="rounded-lg bg-[#10a37f] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#0d8c6e] disabled:cursor-not-allowed disabled:opacity-50"
                      title="Send message"
                    >
                      Send
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-none border-0 bg-transparent" aria-label="Conversation transcript">
            <div className="flex-none shrink-0 border-b border-gray-200 px-4 py-2 dark:border-gray-700">
              <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-400">Transcript</h2>
            </div>
            <div
              ref={transcriptScrollRef}
              onScroll={handleTranscriptScroll}
              className="scrollbar min-h-0 flex-1 space-y-3 overflow-y-auto overflow-x-hidden p-4"
            >
              {transcript.length === 0 && !interimTranscript ? (
                <p className="text-sm italic text-gray-500 dark:text-gray-400">
                  {inputMode === "text"
                    ? "Type your message, then press Send. The conversation will appear here."
                    : isVoiceMemoMode
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
                      <div className="max-w-[85%] break-words text-right italic text-gray-800 dark:text-gray-200">
                        <span className="mb-0.5 block font-medium opacity-80">You (speaking...)</span>
                        {interimTranscript}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
