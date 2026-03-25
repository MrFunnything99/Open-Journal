import { usePersonaplexChat } from "../PersonaplexChatContext";

function SendIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
    </svg>
  );
}

/** Shown above the composer while dictating or server transcription runs. */
export function LiveDictationBubble({ className = "" }: { className?: string }) {
  const { micPhase, liveDictationText } = usePersonaplexChat();
  if (micPhase === "idle") return null;
  const body =
    micPhase === "processing"
      ? "Transcribing…"
      : liveDictationText.trim().length > 0
        ? liveDictationText.trim()
        : "Listening…";
  return (
    <div
      className={`rounded-2xl border border-white/15 bg-white/[0.1] px-3 py-2 text-left text-[0.85rem] leading-snug text-white/90 shadow-inner backdrop-blur-sm ${className}`}
      role="status"
      aria-live="polite"
    >
      {body}
      {micPhase === "recording" && (
        <span className="ml-1 inline-block h-2 w-2 animate-pulse rounded-full bg-red-400 align-middle" aria-hidden />
      )}
    </div>
  );
}

export type AskAnythingLayout = "dock" | "rail";

type AskAnythingComposerProps = {
  layout: AskAnythingLayout;
  /** Collapsed sidebar (~72px): icon-only + expand hint */
  railNarrow?: boolean;
  onExpandRail?: () => void;
};

export function AskAnythingComposer({ layout, railNarrow = false, onExpandRail }: AskAnythingComposerProps) {
  const {
    idPrefix,
    draft,
    setDraft,
    sendChat,
    sending,
    micPhase,
    startRecording,
    stopRecording,
    fileInputRef,
    onPickFile,
    composerDisabled,
  } = usePersonaplexChat();

  const shellClass =
    layout === "dock"
      ? "glass-panel flex items-center gap-1 rounded-full pl-3 pr-2 shadow-[0_-4px_32px_rgba(0,0,0,0.35)] transition-shadow md:gap-2 md:pl-4"
      : "glass-panel flex w-full items-center gap-1 rounded-2xl px-2 py-1.5 shadow-[0_4px_24px_rgba(0,0,0,0.25)] transition-shadow";

  if (layout === "rail" && railNarrow) {
    return (
      <div className="flex w-full flex-col gap-1 border-t border-white/10 px-0.5 py-2">
        <button
          type="button"
          onClick={onExpandRail}
          className="rounded-lg py-1 text-center text-[10px] font-medium text-white/45 transition hover:bg-white/5 hover:text-white/75"
        >
          Expand to type
        </button>
        <input
          ref={fileInputRef as React.LegacyRef<HTMLInputElement>}
          type="file"
          accept="audio/*,.mp3,.m4a,.wav,.webm,.ogg,.flac"
          className="hidden"
          onChange={onPickFile}
        />
        <div className="flex flex-col items-stretch gap-1">
          <div className="flex justify-center gap-1">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={composerDisabled}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-white/90 hover:bg-white/10 disabled:opacity-40"
              aria-label="Attach audio"
              title="Attach audio"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
              </svg>
            </button>
            {micPhase === "recording" ? (
              <button
                type="button"
                onClick={stopRecording}
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-red-500 text-white"
                aria-label="Stop recording"
              >
                <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void startRecording()}
                disabled={composerDisabled}
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-white/90 hover:bg-white/10 disabled:opacity-40"
                aria-label="Dictate"
                title="Dictate"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                  />
                </svg>
              </button>
            )}
            <button
              type="button"
              onClick={() => void sendChat()}
              disabled={sending || micPhase !== "idle" || !draft.trim()}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-40"
              aria-label="Send message"
              title="Send"
            >
              <SendIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={shellClass + (micPhase === "recording" ? " ring-2 ring-red-400/50" : "")}>
      <input
        ref={fileInputRef as React.LegacyRef<HTMLInputElement>}
        type="file"
        accept="audio/*,.mp3,.m4a,.wav,.webm,.ogg,.flac"
        className="hidden"
        onChange={onPickFile}
      />
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={composerDisabled}
        className={`flex shrink-0 items-center justify-center rounded-full text-white/90 hover:bg-white/10 disabled:opacity-40 ${
          layout === "rail" ? "h-9 w-9" : "h-11 w-11"
        }`}
        aria-label="Attach audio"
        title="Attach audio"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className={layout === "rail" ? "h-5 w-5" : "h-6 w-6"} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
        </svg>
      </button>

      <textarea
        id={`${idPrefix}-global-composer`}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void sendChat();
          }
        }}
        disabled={composerDisabled}
        rows={layout === "rail" ? 2 : 1}
        placeholder="Ask anything"
        className={`max-h-36 min-w-0 flex-1 resize-none border-0 bg-transparent text-[0.95rem] text-white placeholder:text-white/45 focus:outline-none focus:ring-0 disabled:opacity-50 ${
          layout === "rail" ? "min-h-[44px] py-2 text-[0.9rem] leading-snug" : "min-h-[48px] py-3"
        }`}
      />

      {micPhase === "recording" ? (
        <button
          type="button"
          onClick={stopRecording}
          className={`flex shrink-0 items-center justify-center rounded-full bg-red-500 text-white ${layout === "rail" ? "h-9 w-9" : "h-11 w-11"}`}
          aria-label="Stop recording"
        >
          <span className="h-2.5 w-2.5 rounded-full bg-white animate-pulse" />
        </button>
      ) : (
        <button
          type="button"
          onClick={() => void startRecording()}
          disabled={composerDisabled}
          className={`flex shrink-0 items-center justify-center rounded-full text-white/90 hover:bg-white/10 disabled:opacity-40 ${
            layout === "rail" ? "h-9 w-9" : "h-11 w-11"
          }`}
          aria-label="Dictate"
          title="Dictate"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className={layout === "rail" ? "h-5 w-5" : "h-6 w-6"} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
        </button>
      )}

      <button
        type="button"
        onClick={() => void sendChat()}
        disabled={sending || micPhase !== "idle" || !draft.trim()}
        className={`flex shrink-0 items-center justify-center rounded-full bg-white text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-40 ${
          layout === "rail" ? "h-9 w-9" : "h-11 w-11"
        }`}
        aria-label="Send message"
        title="Send"
      >
        <SendIcon className={layout === "rail" ? "h-4 w-4" : "h-5 w-5"} />
      </button>
    </div>
  );
}

/** Fixed composer for small screens (sidebar composer is desktop-only). */
export function MobileAskComposerDock({ hidden }: { hidden: boolean }) {
  if (hidden) return null;
  return (
    <div
      className="pointer-events-none fixed inset-x-0 bottom-0 z-40 flex justify-center px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-2 md:hidden"
      role="region"
      aria-label="Message composer"
    >
      <div className="pointer-events-auto w-full max-w-3xl">
        <LiveDictationBubble className="mb-2" />
        <AskAnythingComposer layout="dock" />
      </div>
    </div>
  );
}
