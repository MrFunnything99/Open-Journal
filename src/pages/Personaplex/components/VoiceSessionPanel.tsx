import { useCallback, useEffect, useRef, type CSSProperties } from "react";
import type { VoiceSessionState } from "../hooks/useVoiceSession";

export type VoiceSessionPanelProps = {
  voiceState: VoiceSessionState;
  partialTranscript: string;
  fullTranscript: string;
  assistantText: string;
  isMuted: boolean;
  onToggleMute: () => void;
  /** Full teardown: stop audio/mic, reset to idle, return to composer (End + Close). */
  onExitVoiceMode: () => void;
  /** Skip current spoken response if voice mode is enabled in the future. */
  onSkipResponse?: () => void;
};

const RING_COUNT = 4;

function WaveformRings({ state }: { state: VoiceSessionState }) {
  const rings = Array.from({ length: RING_COUNT }, (_, i) => i);
  const active = state === "listening" || state === "speaking";
  const slow = state === "thinking";

  return (
    <div className="pointer-events-none relative flex h-44 w-44 items-center justify-center sm:h-56 sm:w-56">
      {rings.map((i) => {
        const delay = `${i * 0.35}s`;
        const size = 64 + i * 28;
        const animClass = active ? "animate-voice-ring" : slow ? "animate-voice-ring-slow" : "";
        const opacity =
          state === "paused" ? 0.12 : active ? 0.25 - i * 0.04 : slow ? 0.18 - i * 0.03 : 0.08;
        return (
          <span
            key={i}
            className={`absolute rounded-full border border-white/30 ${animClass}`}
            style={
              {
                width: size,
                height: size,
                animationDelay: delay,
                opacity,
                "--ring-scale": state === "speaking" ? 1.18 : 1.08,
              } as CSSProperties
            }
          />
        );
      })}
      <span
        className={`relative z-10 flex h-20 w-20 items-center justify-center rounded-full sm:h-24 sm:w-24 ${
          state === "speaking"
            ? "bg-indigo-500/40"
            : state === "listening"
              ? "bg-emerald-500/30"
              : state === "thinking"
                ? "bg-amber-400/25"
                : "bg-white/10"
        } transition-colors duration-500`}
      >
        <CenterIcon state={state} />
      </span>
    </div>
  );
}

function CenterIcon({ state }: { state: VoiceSessionState }) {
  if (state === "listening") {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-emerald-300 sm:h-10 sm:w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m-4 0h8m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
      </svg>
    );
  }
  if (state === "thinking") {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 animate-spin text-amber-300 sm:h-10 sm:w-10" style={{ animationDuration: "2.5s" }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m8.66-13.5l-.87.5M4.21 16.5l-.87.5M20.66 16.5l-.87-.5M4.21 7.5l-.87-.5M21 12h-1M4 12H3m13.36-5.36l-.7.7M8.34 15.66l-.7.7m9.72 0l-.7-.7M8.34 8.34l-.7-.7" />
      </svg>
    );
  }
  if (state === "speaking") {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-indigo-300 sm:h-10 sm:w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M17.95 6.05a8 8 0 010 11.9M6.5 8H4a1 1 0 00-1 1v6a1 1 0 001 1h2.5l4.5 4V4l-4.5 4z" />
      </svg>
    );
  }
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white/40 sm:h-10 sm:w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M10 9v6m4-6v6" />
    </svg>
  );
}

function statusLabel(state: VoiceSessionState): string {
  switch (state) {
    case "listening":
      return "Listening...";
    case "thinking":
      return "Reflecting...";
    case "speaking":
      return "Speaking...";
    case "paused":
      return "Paused";
    default:
      return "";
  }
}

export function VoiceSessionPanel({
  voiceState,
  partialTranscript,
  fullTranscript,
  assistantText,
  isMuted,
  onToggleMute,
  onExitVoiceMode,
  onSkipResponse,
}: VoiceSessionPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [partialTranscript, assistantText]);

  const handleExit = useCallback(() => {
    onExitVoiceMode();
  }, [onExitVoiceMode]);

  if (voiceState === "idle") return null;

  const showTranscript = voiceState === "listening" || voiceState === "thinking";
  const transcript = showTranscript ? partialTranscript || fullTranscript : "";

  return (
    <div className="fixed inset-0 z-40 flex flex-col items-center bg-[#08080f]/97 backdrop-blur-xl">
      <div className="flex w-full items-center justify-between px-4 py-3">
        <span className="text-xs font-medium uppercase tracking-widest text-white/40">Voice Mode</span>
        <button
          type="button"
          onClick={handleExit}
          className="rounded-lg px-3 py-1.5 text-xs font-medium text-white/60 transition hover:bg-white/10 hover:text-white/90"
        >
          Back to chat
        </button>
      </div>

      <div className="flex flex-1 flex-col items-center justify-center gap-6 px-4">
        <WaveformRings state={voiceState} />

        <p className="text-sm font-medium tracking-wide text-white/50">{statusLabel(voiceState)}</p>

        {showTranscript && transcript && (
          <div
            ref={scrollRef}
            className="max-h-28 w-full max-w-lg overflow-y-auto rounded-xl bg-white/[0.04] px-4 py-3 text-center text-sm leading-relaxed text-white/70"
          >
            {transcript}
          </div>
        )}

        {voiceState === "speaking" && assistantText && (
          <div
            ref={scrollRef}
            className="max-h-40 w-full max-w-lg overflow-y-auto rounded-xl bg-white/[0.04] px-4 py-3 text-center text-[0.9375rem] leading-relaxed text-white/80 animate-in fade-in duration-500"
          >
            {assistantText}
          </div>
        )}

      </div>

      <div className="flex flex-col items-center gap-3 px-4 pb-[max(1.5rem,env(safe-area-inset-bottom))] pt-4">
        {voiceState === "speaking" && onSkipResponse && (
          <button
            type="button"
            onClick={onSkipResponse}
            className="mb-1 flex items-center gap-2 rounded-full bg-white/10 px-5 py-2 text-xs font-medium text-white/70 transition hover:bg-white/20 hover:text-white/90 active:scale-95"
            aria-label="Skip response and start talking"
            title="Skip — stop the assistant and start talking"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 5l7 7-7 7" />
            </svg>
            Skip
          </button>
        )}
        <div className="flex items-center gap-6">
          <button
            type="button"
            onClick={onToggleMute}
            className={`flex h-14 w-14 items-center justify-center rounded-full transition ${
              isMuted
                ? "bg-red-500/20 text-red-300 ring-1 ring-red-400/40"
                : "bg-white/10 text-white/80 hover:bg-white/20"
            }`}
            aria-label={isMuted ? "Unmute" : "Mute"}
            title={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m-4 0h8m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                <line x1="3" y1="3" x2="21" y2="21" strokeWidth={2} strokeLinecap="round" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m-4 0h8m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            )}
          </button>

          <button
            type="button"
            onClick={handleExit}
            className="flex h-14 w-14 items-center justify-center rounded-full bg-red-500/80 text-white shadow-lg shadow-red-500/25 transition hover:bg-red-500"
            aria-label="End voice session and return to chat"
            title="End call"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16 8l-8 8m0-8l8 8" />
            </svg>
          </button>
        </div>
        <button
          type="button"
          onClick={handleExit}
          className="text-xs font-medium text-white/45 underline-offset-2 hover:text-white/70 hover:underline"
        >
          Return to chat
        </button>
      </div>

      <style>{`
        @keyframes voice-ring-pulse {
          0%, 100% { transform: scale(1); opacity: var(--tw-opacity, 0.2); }
          50% { transform: scale(var(--ring-scale, 1.08)); opacity: calc(var(--tw-opacity, 0.2) * 1.6); }
        }
        @keyframes voice-ring-pulse-slow {
          0%, 100% { transform: scale(1); opacity: var(--tw-opacity, 0.15); }
          50% { transform: scale(1.04); opacity: calc(var(--tw-opacity, 0.15) * 1.3); }
        }
        .animate-voice-ring {
          animation: voice-ring-pulse 2s ease-in-out infinite;
        }
        .animate-voice-ring-slow {
          animation: voice-ring-pulse-slow 3.5s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
