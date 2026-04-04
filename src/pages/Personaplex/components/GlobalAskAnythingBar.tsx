import { createPortal } from "react-dom";
import { useEffect, useLayoutEffect, useRef, useState, type MutableRefObject, type RefObject } from "react";
import {
  CHAT_INTERACTION_MODE_META,
  CHAT_INTERACTION_MODES,
  type ChatInteractionMode,
} from "../chatInteractionModes";
import {
  CHAT_COMPLETION_MODEL_OPTIONS,
  type UserSelectableChatModelId,
} from "../chatCompletionModels";
import { usePersonaplexChat } from "../PersonaplexChatContext";

function ModeChipIcon({ mode }: { mode: ChatInteractionMode }) {
  const cls = "h-3.5 w-3.5 shrink-0 opacity-95";
  switch (mode) {
    case "journal":
    case "autobiography":
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75} aria-hidden>
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
          />
        </svg>
      );
    default:
      return (
        <svg className={cls} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75} aria-hidden>
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
          />
        </svg>
      );
  }
}

function composerModeChipClass(mode: ChatInteractionMode, compact: boolean): string {
  const pad = compact ? "px-2 py-0.5" : "px-2.5 py-1";
  const text = compact ? "text-[0.65rem] sm:text-[0.7rem]" : "text-[0.7rem] sm:text-xs";
  const base = `inline-flex max-w-[11rem] shrink-0 items-center gap-1 truncate rounded-full border font-semibold tracking-tight transition-colors hover:brightness-110 sm:max-w-[14rem] ${pad} ${text}`;
  if (mode === "journal") {
    return `${base} border-sky-400/50 bg-sky-500/[0.22] text-sky-100 shadow-[0_0_12px_-4px_rgba(56,189,248,0.45)]`;
  }
  return `${base} border-violet-400/50 bg-violet-600/[0.22] text-violet-100 shadow-[0_0_12px_-4px_rgba(167,139,250,0.4)]`;
}

/** ChatGPT-style mode tag beside +; opens the same menu as + when clicked. */
function ComposerModeChip({
  mode,
  compact = false,
  onOpen,
  disabled,
}: {
  mode: ChatInteractionMode;
  compact?: boolean;
  onOpen: () => void;
  disabled?: boolean;
}) {
  const label = CHAT_INTERACTION_MODE_META[mode].composerChipLabel;
  return (
    <button
      type="button"
      onClick={onOpen}
      disabled={disabled}
      title="Change mode"
      aria-label={`Mode: ${label}. Open mode menu`}
      className={`${composerModeChipClass(mode, compact)} disabled:opacity-40`}
    >
      <ModeChipIcon mode={mode} />
      <span className="truncate">{label}</span>
    </button>
  );
}

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

export type AskAnythingLayout = "dock" | "rail" | "center";

function ComposerChatModelSelect({ idPrefix, embedded }: { idPrefix: string; embedded?: boolean }) {
  const { userChatModel, setUserChatModel, composerDisabled } = usePersonaplexChat();
  return (
    <select
      id={`${idPrefix}-chat-model`}
      aria-label="Chat model"
      value={userChatModel}
      onChange={(e) => setUserChatModel(e.target.value as UserSelectableChatModelId)}
      disabled={composerDisabled}
      title="Model for replies (retrieval + tools use this model)"
      className={
        embedded
          ? "min-w-0 flex-1 rounded-lg border border-white/12 bg-black/40 px-2.5 py-1.5 text-[0.7rem] text-white focus:border-white/25 focus:outline-none focus:ring-1 focus:ring-white/15 disabled:opacity-50 sm:max-w-[16rem] sm:flex-none sm:text-xs"
          : "max-w-[10rem] shrink-0 rounded-full border border-white/15 bg-black/35 px-2 py-1 text-[0.65rem] text-white focus:border-white/30 focus:outline-none disabled:opacity-50 sm:max-w-[11rem] sm:text-xs"
      }
    >
      {CHAT_COMPLETION_MODEL_OPTIONS.map((o) => (
        <option key={o.id} value={o.id}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

type AskAnythingComposerProps = {
  layout: AskAnythingLayout;
  /** Collapsed sidebar (~72px): icon-only + expand hint */
  railNarrow?: boolean;
  onExpandRail?: () => void;
  /** Home — AI-Assisted Journal Mode: hide + menu and mode chip (mode is chosen above). */
  assistedJournalMinimal?: boolean;
  /** Callback to launch voice conversation mode (only shown in autobiography + assistedJournalMinimal). */
  onStartVoiceSession?: () => void;
};

type PlusPlacement = "above" | "below" | "right";

function PlusOptionsMenu({
  open,
  onClose,
  placement,
  anchorRef,
  menuContainerRef,
  fileInputRef,
  composerDisabled,
  mode,
  setMode,
}: {
  open: boolean;
  onClose: () => void;
  placement: PlusPlacement;
  anchorRef: RefObject<HTMLElement | null>;
  /** RefObject<T>.current is T | null — do not use RefObject<T | null>. */
  menuContainerRef: RefObject<HTMLDivElement>;
  fileInputRef: MutableRefObject<HTMLInputElement | null>;
  composerDisabled: boolean;
  mode: ChatInteractionMode;
  setMode: (m: ChatInteractionMode) => void;
}) {
  useLayoutEffect(() => {
    if (!open) return;
    const menu = menuContainerRef.current;
    const anchor = anchorRef.current;
    if (!menu || !anchor) return;

    const pad = 8;
    const gap = 6;

    const place = () => {
      const r = anchor.getBoundingClientRect();
      const mw = menu.offsetWidth;
      const mh = menu.offsetHeight;
      let left = r.left;
      left = Math.max(pad, Math.min(left, window.innerWidth - mw - pad));

      menu.style.position = "fixed";
      menu.style.zIndex = "9999";

      if (placement === "below") {
        let top = r.bottom + gap;
        if (top + mh > window.innerHeight - pad) {
          top = Math.max(pad, window.innerHeight - mh - pad);
        }
        menu.style.top = `${top}px`;
        menu.style.left = `${left}px`;
      } else if (placement === "above") {
        let top = r.top - mh - gap;
        if (top < pad) top = pad;
        menu.style.top = `${top}px`;
        menu.style.left = `${left}px`;
      } else {
        let top = r.top;
        let l = r.right + gap;
        if (l + mw > window.innerWidth - pad) {
          l = Math.max(pad, r.left - mw - gap);
        }
        if (top + mh > window.innerHeight - pad) {
          top = Math.max(pad, window.innerHeight - mh - pad);
        }
        menu.style.top = `${top}px`;
        menu.style.left = `${l}px`;
      }
    };

    place();
    const ro = new ResizeObserver(() => place());
    ro.observe(menu);
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      ro.disconnect();
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, placement, anchorRef, menuContainerRef]);

  if (!open || typeof document === "undefined") return null;

  const panel = (
    <div
      ref={menuContainerRef}
      className="min-w-[min(100vw-2rem,17rem)] max-w-[min(100vw-2rem,20rem)] max-h-[min(70dvh,22rem)] overflow-y-auto overscroll-contain rounded-xl border border-white/15 bg-[#14141f]/95 p-1 pb-1.5 shadow-2xl backdrop-blur-xl"
      role="menu"
      aria-label="Composer options"
    >
      <p className="px-2.5 pb-1 pt-1.5 text-[0.6rem] font-semibold uppercase tracking-[0.14em] text-white/45">Mode</p>
      {CHAT_INTERACTION_MODES.map((m) => {
        const meta = CHAT_INTERACTION_MODE_META[m];
        const sel = mode === m;
        return (
          <button
            key={m}
            type="button"
            role="menuitemradio"
            aria-checked={sel}
            onClick={() => {
              setMode(m);
              onClose();
            }}
            className={`flex w-full flex-col items-stretch rounded-lg px-3 py-2 text-left transition-colors ${
              sel ? "bg-white/[0.12] text-white" : "text-white/85 hover:bg-white/[0.08]"
            }`}
          >
            <span className="flex items-center gap-2 text-sm font-medium">
              {sel ? (
                <span className="text-emerald-400" aria-hidden>
                  ✓
                </span>
              ) : (
                <span className="w-4 shrink-0" aria-hidden />
              )}
              {meta.label}
            </span>
            <span className={`mt-0.5 pl-6 text-[0.7rem] leading-snug ${sel ? "text-white/65" : "text-white/45"}`}>
              {meta.sublabel}
            </span>
          </button>
        );
      })}
      <div className="my-1 border-t border-white/10" role="separator" />
      <button
        type="button"
        role="menuitem"
        disabled={composerDisabled}
        onClick={() => {
          fileInputRef.current?.click();
          onClose();
        }}
        className="flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-sm font-medium text-white/90 transition-colors hover:bg-white/[0.08] disabled:opacity-40"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 shrink-0 opacity-80" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
        </svg>
        Attach audio
      </button>
    </div>
  );

  return createPortal(panel, document.body);
}

export function AskAnythingComposer({
  layout,
  railNarrow = false,
  onExpandRail,
  assistedJournalMinimal = false,
  onStartVoiceSession,
}: AskAnythingComposerProps) {
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
    pendingAudioFile,
    clearPendingAudioFile,
    composerDisabled,
    chatInteractionMode,
    setChatInteractionMode,
  } = usePersonaplexChat();
  const [plusOpen, setPlusOpen] = useState(false);
  const plusWrapRef = useRef<HTMLDivElement>(null);
  const plusMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!plusOpen) return;
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (plusWrapRef.current?.contains(t)) return;
      if (plusMenuRef.current?.contains(t)) return;
      setPlusOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [plusOpen]);

  const showModelFooter = chatInteractionMode === "autobiography";

  const shellOuterClass =
    layout === "dock"
      ? "glass-panel flex w-full flex-col overflow-hidden rounded-[1.75rem] shadow-[0_-4px_32px_rgba(0,0,0,0.35)] transition-shadow"
      : layout === "center"
        ? "glass-panel mx-auto flex w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/12 shadow-[0_8px_32px_rgba(0,0,0,0.28)] transition-shadow"
        : "glass-panel flex w-full flex-col overflow-hidden rounded-2xl shadow-[0_4px_24px_rgba(0,0,0,0.25)] transition-shadow";

  const shellRowClass =
    layout === "dock"
      ? "flex items-center gap-1 pl-3 pr-2 py-1.5 md:gap-2 md:pl-4 md:py-2"
      : layout === "center"
        ? "flex items-center gap-1 px-3 py-2 md:gap-2 md:px-4 md:py-2.5"
        : "flex w-full items-center gap-1 px-2 py-1.5";

  if (layout === "rail" && railNarrow) {
    return (
      <div className="flex w-full flex-col gap-1.5 border-t border-white/10 px-0.5 py-2">
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
        <div className="glass-panel flex flex-col overflow-hidden rounded-xl border border-white/10">
          <div className="relative flex flex-wrap justify-center gap-1 px-1.5 py-2">
            {!assistedJournalMinimal ? (
              <>
                <div className="relative shrink-0" ref={plusWrapRef}>
                  <button
                    type="button"
                    onClick={() => setPlusOpen((o) => !o)}
                    disabled={composerDisabled}
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-white/90 outline-none hover:bg-white/10 disabled:opacity-40"
                    aria-label="More options and modes"
                    aria-expanded={plusOpen}
                    title="Modes and attachments"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                    </svg>
                  </button>
                  <PlusOptionsMenu
                    open={plusOpen}
                    onClose={() => setPlusOpen(false)}
                    placement="right"
                    anchorRef={plusWrapRef}
                    menuContainerRef={plusMenuRef}
                    fileInputRef={fileInputRef}
                    composerDisabled={composerDisabled}
                    mode={chatInteractionMode}
                    setMode={setChatInteractionMode}
                  />
                </div>
                <ComposerModeChip
                  mode={chatInteractionMode}
                  compact
                  onOpen={() => setPlusOpen(true)}
                  disabled={composerDisabled}
                />
              </>
            ) : null}
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
              disabled={sending || micPhase !== "idle" || (!draft.trim() && !pendingAudioFile)}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-40"
              aria-label="Send message"
              title="Send"
            >
              <SendIcon className="h-4 w-4" />
            </button>
          </div>
          {showModelFooter && (
            <div className="flex items-center gap-2 border-t border-white/10 bg-black/25 px-2 py-1.5">
              <span className="shrink-0 text-[0.6rem] font-semibold uppercase tracking-[0.14em] text-white/40">Model</span>
              <ComposerChatModelSelect idPrefix={idPrefix} embedded />
            </div>
          )}
        </div>
      </div>
    );
  }

  const plusPlacement: PlusPlacement = layout === "dock" ? "above" : "below";

  return (
    <div className="flex flex-col">
      {pendingAudioFile && (
        <div className="flex items-center gap-2 rounded-t-2xl border border-b-0 border-white/10 bg-white/[0.04] px-3 py-2">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 shrink-0 text-white/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
          </svg>
          <span className="min-w-0 flex-1 truncate text-xs text-white/80">{pendingAudioFile.name}</span>
          <button
            type="button"
            onClick={clearPendingAudioFile}
            className="shrink-0 rounded p-0.5 text-white/50 hover:bg-white/10 hover:text-white/90"
            aria-label="Remove audio file"
            title="Remove"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}
      <div
        className={
          shellOuterClass +
          (micPhase === "recording" ? " ring-2 ring-red-400/50" : "") +
          (pendingAudioFile ? " rounded-t-none border-t-0" : "")
        }
      >
      <div className={shellRowClass}>
      <input
        ref={fileInputRef as React.LegacyRef<HTMLInputElement>}
        type="file"
        accept="audio/*,.mp3,.m4a,.wav,.webm,.ogg,.flac"
        className="hidden"
        onChange={onPickFile}
      />
      {!assistedJournalMinimal ? (
        <div className="flex shrink-0 items-center gap-1 sm:gap-1.5">
          <div className="relative shrink-0" ref={plusWrapRef}>
            <button
              type="button"
              onClick={() => setPlusOpen((o) => !o)}
              disabled={composerDisabled}
              className={`flex shrink-0 items-center justify-center rounded-full text-white/90 outline-none hover:bg-white/10 disabled:opacity-40 ${
                layout === "rail" ? "h-9 w-9" : "h-11 w-11"
              }`}
              aria-label="Modes and attachments"
              aria-expanded={plusOpen}
              title="Modes and attachments"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className={layout === "rail" ? "h-5 w-5" : "h-6 w-6"} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
              </svg>
            </button>
            <PlusOptionsMenu
              open={plusOpen}
              onClose={() => setPlusOpen(false)}
              placement={plusPlacement}
              anchorRef={plusWrapRef}
              menuContainerRef={plusMenuRef}
              fileInputRef={fileInputRef}
              composerDisabled={composerDisabled}
              mode={chatInteractionMode}
              setMode={setChatInteractionMode}
            />
          </div>
          <ComposerModeChip
            mode={chatInteractionMode}
            compact={layout === "rail"}
            onOpen={() => setPlusOpen(true)}
            disabled={composerDisabled}
          />
        </div>
      ) : null}

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
        placeholder={
          pendingAudioFile
            ? "Add a message or press send to transcribe"
            : chatInteractionMode === "autobiography"
              ? "Chat for AI-assisted journaling…"
              : "Write in your manual journal…"
        }
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

      {/* Voice conversation mode (autobiography + assistedJournalMinimal only) */}
      {assistedJournalMinimal && chatInteractionMode === "autobiography" && onStartVoiceSession && micPhase === "idle" && (
        <button
          type="button"
          onClick={onStartVoiceSession}
          disabled={composerDisabled}
          className={`flex shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500/20 to-emerald-500/20 text-white/90 ring-1 ring-white/15 transition hover:from-indigo-500/30 hover:to-emerald-500/30 hover:ring-white/25 disabled:opacity-40 ${
            layout === "rail" ? "h-9 w-9" : "h-11 w-11"
          }`}
          aria-label="Start voice conversation"
          title="Voice conversation"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className={layout === "rail" ? "h-5 w-5" : "h-[1.35rem] w-[1.35rem]"} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M17.95 6.05a8 8 0 010 11.9M6.5 8H4a1 1 0 00-1 1v6a1 1 0 001 1h2.5l4.5 4V4l-4.5 4z" />
          </svg>
        </button>
      )}

      <button
        type="button"
        onClick={() => void sendChat()}
        disabled={sending || micPhase !== "idle" || (!draft.trim() && !pendingAudioFile)}
        className={`flex shrink-0 items-center justify-center rounded-full bg-white text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-40 ${
          layout === "rail" ? "h-9 w-9" : "h-11 w-11"
        }`}
        aria-label="Send message"
        title="Send"
      >
        <SendIcon className={layout === "rail" ? "h-4 w-4" : "h-5 w-5"} />
      </button>
      </div>
      {showModelFooter && (
        <div
          className={
            layout === "center"
              ? "flex items-center gap-2 border-t border-white/10 bg-black/20 px-3 py-1.5 md:px-4"
              : layout === "rail"
                ? "flex items-center gap-2 border-t border-white/10 bg-black/20 px-2 py-1.5"
                : "flex items-center gap-2 border-t border-white/10 bg-black/25 px-3 py-1.5 md:px-4"
          }
        >
          <span className="shrink-0 text-[0.6rem] font-semibold uppercase tracking-[0.14em] text-white/40">Model</span>
          <ComposerChatModelSelect idPrefix={idPrefix} embedded />
        </div>
      )}
      </div>
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

/**
 * Hides the fixed mobile dock on Home idle so the centered composer + hero are the single entry point.
 */
export function MobileAskComposerDockGate({
  railOpen,
  activeView,
}: {
  railOpen: boolean;
  activeView: string;
}) {
  const { isChatActive, messages, sending } = usePersonaplexChat();
  const homeIdle =
    activeView === "voice_memo" && !isChatActive && messages.length === 0 && !sending;
  /** Home uses only the in-column composers (see VoiceMemoTab). Other tabs may use this dock on mobile. */
  return (
    <MobileAskComposerDock
      hidden={railOpen || homeIdle || activeView === "voice_memo" || activeView === "about"}
    />
  );
}
