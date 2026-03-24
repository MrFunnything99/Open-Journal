import { FC, useCallback, useEffect, useId, useRef, useState } from "react";
import { backendFetch } from "../../../backendApi";
import type { ChatMessage } from "../hooks/useJournalHistory";

type Props = {
  onToast: (msg: string) => void;
  /** Opens the full voice session panel (ChatGPT-style “voice mode” control). */
  onOpenSessionPanel?: () => void;
  /** Persist to The Brain → Knowledge base → Journals (uses date for folder layout + memory ingest). */
  saveEntry?: (transcript: ChatMessage[], dateIso: string) => string;
  syncUnsyncedEntries?: () => Promise<number>;
  /**
   * Chat tab: keep the composer higher (not flush to the bottom) and reserve space beneath it for extra UI.
   */
  elevateComposerLayout?: boolean;
};

type UiMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onloadend = () => {
      const d = r.result as string;
      resolve(d.split(",")[1] ?? "");
    };
    r.onerror = () => reject(new Error("read failed"));
    r.readAsDataURL(blob);
  });
}

/** Transcription APIs use the filename extension to detect container format. */
function defaultFilenameForMicBlob(mimeType: string): string {
  const mt = (mimeType || "").toLowerCase();
  if (mt.includes("mp4") || mt.includes("m4a") || mt.includes("aac") || mt === "audio/mp4") {
    return "dictation.m4a";
  }
  if (mt.includes("webm")) return "dictation.webm";
  if (mt.includes("wav")) return "dictation.wav";
  if (mt.includes("mpeg") || mt.includes("mp3")) return "dictation.mp3";
  if (mt.includes("ogg") || mt.includes("opus")) return "dictation.ogg";
  if (mt.includes("flac")) return "dictation.flac";
  return "dictation.webm";
}

function effectiveRecorderMime(recorder: MediaRecorder): string {
  const t = recorder.mimeType?.trim();
  if (t) return t;
  // Safari (iOS and desktop) often emits MP4/AAC but leaves mimeType empty when WebM isn't used.
  if (typeof navigator !== "undefined" && navigator.vendor === "Apple Computer, Inc.") {
    return "audio/mp4";
  }
  return "audio/webm";
}

const CHAT_TIMEOUT_MS = 90_000;

type HomeInteractionMode = "conversation" | "journal" | "autobiography";

const HOME_MODES: HomeInteractionMode[] = ["conversation", "journal", "autobiography"];

const HOME_MODE_META: Record<
  HomeInteractionMode,
  { label: string; sublabel: string; description: string }
> = {
  conversation: {
    label: "Conversation",
    sublabel: "Chat freely",
    description:
      "Your default AI assistant. Chat freely, ask questions, and explore ideas—enhanced with your personal context.",
  },
  journal: {
    label: "Journal",
    sublabel: "Reflect",
    description:
      "A space for free-flow reflection. Write naturally, process your thoughts, and receive optional AI feedback or gentle structure.",
  },
  autobiography: {
    label: "Autobiography",
    sublabel: "Track your life",
    description:
      "A structured way to track and understand your life. Reflect on your day, habits, goals, and past experiences through guided conversation.",
  },
};

function toDateInputValue(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function toTimeInputValue(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** Local date + time inputs → ISO UTC (matches Brain / ingest `entry.date`). */
function dateAndTimeToIso(dateStr: string, timeStr: string): string {
  const d = dateStr.trim();
  const t = (timeStr.trim() || "00:00").slice(0, 5);
  const dp = d.split("-").map((x) => parseInt(x, 10));
  const tp = t.split(":").map((x) => parseInt(x, 10));
  const y = dp[0]!;
  const mo = dp[1]!;
  const day = dp[2]!;
  const h = tp[0] ?? 0;
  const mi = tp[1] ?? 0;
  if ([y, mo, day, h, mi].some((n) => Number.isNaN(n))) return new Date().toISOString();
  return new Date(y, mo - 1, day, h, mi, 0, 0).toISOString();
}

function WaveformIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="currentColor" aria-hidden>
      <rect x="5" y="10" width="3" height="8" rx="1" />
      <rect x="10.5" y="6" width="3" height="16" rx="1" />
      <rect x="16" y="8" width="3" height="12" rx="1" />
    </svg>
  );
}

export const VoiceMemoTab: FC<Props> = ({
  onToast,
  onOpenSessionPanel,
  saveEntry,
  syncUnsyncedEntries,
  elevateComposerLayout = false,
}) => {
  const idPrefix = useId();
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [micPhase, setMicPhase] = useState<"idle" | "recording" | "processing">("idle");
  const [error, setError] = useState<string | null>(null);
  const [rawTranscript, setRawTranscript] = useState("");
  const [reviewText, setReviewText] = useState("");
  const [validatedJournal, setValidatedJournal] = useState("");
  const [validationFeedback, setValidationFeedback] = useState("");
  const [validationNotes, setValidationNotes] = useState<string[]>([]);
  const [validationModel, setValidationModel] = useState("openai/gpt-5.4");
  const [modelUsed, setModelUsed] = useState("");
  const [validating, setValidating] = useState(false);
  /** Local date/time for journal entry (separate inputs); set when transcript first arrives. */
  const [journalEntryDate, setJournalEntryDate] = useState("");
  const [journalEntryTime, setJournalEntryTime] = useState("");
  const [savingJournal, setSavingJournal] = useState(false);
  /** After first text send: hero fades, composer stays at bottom, sidebar + thread layout. */
  const [isChatActive, setIsChatActive] = useState(false);
  const [homeInteractionMode, setHomeInteractionMode] = useState<HomeInteractionMode>("conversation");
  const listRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, sending, isChatActive]);

  const readAloud = useCallback((text: string) => {
    if (typeof window === "undefined" || !window.speechSynthesis) {
      onToast("Read aloud is not supported in this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1;
    window.speechSynthesis.speak(u);
  }, [onToast]);

  const copyText = useCallback(
    async (text: string) => {
      try {
        await navigator.clipboard.writeText(text);
        onToast("Copied.");
      } catch {
        onToast("Could not copy.");
      }
    },
    [onToast]
  );

  const sendChat = useCallback(async () => {
    const text = draft.trim();
    if (!text || sending) return;
    setIsChatActive(true);
    setError(null);
    setDraft("");
    const userMsg: UiMessage = { id: `u_${Date.now()}`, role: "user", content: text };
    setMessages((m) => [...m, userMsg]);
    setSending(true);

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);

    try {
      const res = await backendFetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          session_id: chatSessionId,
          personalization: 1,
          intrusiveness: 0.5,
          mode: "journal",
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const raw = await res.text();
      let data: {
        error?: string;
        detail?: string;
        response?: string;
        session_id?: string;
      } = {};
      if (raw.trim()) {
        try {
          data = JSON.parse(raw) as typeof data;
        } catch {
          throw new Error(res.ok ? "Invalid response from server" : `Server error (${res.status})`);
        }
      }
      if (!res.ok) {
        throw new Error(data.detail || data.error || `Chat failed (${res.status})`);
      }
      if (!data.response || typeof data.response !== "string" || !data.session_id) {
        throw new Error(data.error || "Invalid chat response");
      }
      setChatSessionId(data.session_id);
      setMessages((m) => [...m, { id: `a_${Date.now()}`, role: "assistant", content: data.response! }]);
    } catch (e) {
      const msg =
        e instanceof Error && e.name === "AbortError"
          ? "Request timed out. Try again."
          : e instanceof Error
            ? e.message
            : "Something went wrong";
      setError(msg);
      onToast(msg);
    } finally {
      setSending(false);
    }
  }, [draft, sending, chatSessionId, onToast]);

  const processMicAudio = useCallback(async (blob: Blob, mimeType: string) => {
    setMicPhase("processing");
    setError(null);
    try {
      const b64 = await blobToBase64(blob);
      // Transcription uses the filename extension to detect format; MediaRecorder blobs have no name.
      const filename =
        blob instanceof File && blob.name?.trim()
          ? blob.name.trim()
          : defaultFilenameForMicBlob(mimeType);
      const res = await backendFetch("/voice-memo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: b64, filename, mime_type: mimeType }),
      });
      const data = (await res.json().catch(() => ({}))) as {
        detail?: string;
        error?: string;
        polished_text?: string;
        raw_transcript?: string;
      };
      if (!res.ok) {
        const d = data.detail;
        const msg =
          typeof d === "string" ? d : typeof data.error === "string" ? data.error : `Request failed (${res.status})`;
        throw new Error(msg);
      }
      const line = (data.polished_text ?? data.raw_transcript ?? "").trim();
      if (line) {
        setIsChatActive(true);
        const capturedAt = new Date();
        const rawBody = ((data.raw_transcript ?? "").trim() || line).trim();
        const reviewBody = (data.raw_transcript ?? line).trim();
        setRawTranscript(rawBody);
        setReviewText(reviewBody);
        setValidatedJournal("");
        setValidationFeedback("");
        setValidationNotes([]);
        setModelUsed("");
        setJournalEntryDate(toDateInputValue(capturedAt));
        setJournalEntryTime(toTimeInputValue(capturedAt));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Transcription failed");
    } finally {
      setMicPhase("idle");
    }
  }, []);

  const runJournalValidation = useCallback(async () => {
    const text = reviewText.trim();
    if (!text || validating) return;
    setValidating(true);
    setError(null);
    try {
      const res = await backendFetch("/journal-validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: validationModel.trim() || undefined }),
      });
      const data = (await res.json().catch(() => ({}))) as {
        detail?: string;
        reformatted_journal?: string;
        feedback?: string;
        validation_notes?: string[];
        model_used?: string;
      };
      if (!res.ok) {
        throw new Error(data.detail || `Validation failed (${res.status})`);
      }
      const reformatted = (data.reformatted_journal || text).trim();
      setValidatedJournal(reformatted);
      setValidationFeedback((data.feedback || "").trim());
      setValidationNotes(Array.isArray(data.validation_notes) ? data.validation_notes.filter((x) => typeof x === "string") : []);
      setModelUsed(typeof data.model_used === "string" && data.model_used.trim() ? data.model_used.trim() : "");
      onToast("Validation complete.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Validation failed";
      setError(msg);
      onToast(msg);
    } finally {
      setValidating(false);
    }
  }, [reviewText, validating, onToast, validationModel]);

  const startAnotherEntry = useCallback(() => {
    setRawTranscript("");
    setReviewText("");
    setValidatedJournal("");
    setValidationFeedback("");
    setValidationNotes([]);
    setModelUsed("");
    setJournalEntryDate("");
    setJournalEntryTime("");
    setError(null);
  }, []);

  const saveToJournal = useCallback(async () => {
    if (!saveEntry || savingJournal) return;
    const body = (validatedJournal.trim() || reviewText.trim());
    if (!body) {
      onToast("Nothing to save yet.");
      return;
    }
    if (!journalEntryDate.trim() || !journalEntryTime.trim()) {
      onToast("Set entry date and time before saving.");
      return;
    }
    const transcript: ChatMessage[] = [{ role: "user", text: body }];
    const dateIso = dateAndTimeToIso(journalEntryDate, journalEntryTime);
    setSavingJournal(true);
    try {
      const id = saveEntry(transcript, dateIso);
      if (!id) {
        onToast("Could not save.");
        return;
      }
      void syncUnsyncedEntries?.();
      onToast("Saved to your journal. Open The Brain → Knowledge base to view.");
    } finally {
      setSavingJournal(false);
    }
  }, [
    saveEntry,
    syncUnsyncedEntries,
    savingJournal,
    validatedJournal,
    reviewText,
    journalEntryDate,
    journalEntryTime,
    onToast,
  ]);

  const startRecording = useCallback(async () => {
    if (micPhase !== "idle" || sending) return;
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "";
      const mr = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const mime = effectiveRecorderMime(mr);
        const blob = new Blob(chunksRef.current, { type: mime });
        mediaRecorderRef.current = null;
        void processMicAudio(blob, mime);
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setMicPhase("recording");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Microphone unavailable");
      setMicPhase("idle");
    }
  }, [micPhase, sending, processMicAudio]);

  const stopRecording = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== "inactive") mr.stop();
    else setMicPhase("idle");
  }, []);

  const onPickFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (!f) return;
      setError(null);
      void processMicAudio(f, f.type || "application/octet-stream");
      e.target.value = "";
    },
    [processMicAudio]
  );

  const composerDisabled = sending || micPhase !== "idle" || validating;

  const hasConversation = messages.length > 0 || sending;

  const composerInner = (
    <>
      <input ref={fileInputRef} type="file" accept="audio/*,.mp3,.m4a,.wav,.webm,.ogg,.flac" className="hidden" onChange={onPickFile} />
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={composerDisabled}
        className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-white/90 hover:bg-white/10 disabled:opacity-40"
        aria-label="Attach audio"
        title="Attach audio"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
        </svg>
      </button>

      <textarea
        id={`${idPrefix}-composer`}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void sendChat();
          }
        }}
        disabled={composerDisabled}
        rows={1}
        placeholder="Ask anything"
        className="max-h-40 min-h-[48px] min-w-0 flex-1 resize-none border-0 bg-transparent py-3 text-[0.95rem] text-white placeholder:text-white/45 focus:outline-none focus:ring-0 disabled:opacity-50"
      />

      {micPhase === "recording" ? (
        <button
          type="button"
          onClick={stopRecording}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-red-500 text-white"
          aria-label="Stop recording"
        >
          <span className="h-2.5 w-2.5 rounded-full bg-white animate-pulse" />
        </button>
      ) : (
        <button
          type="button"
          onClick={startRecording}
          disabled={composerDisabled}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-white/90 hover:bg-white/10 disabled:opacity-40"
          aria-label="Dictate"
          title="Dictate"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        </button>
      )}

      <button
        type="button"
        onClick={() => {
          if (onOpenSessionPanel) {
            onOpenSessionPanel();
            return;
          }
          void startRecording();
        }}
        disabled={sending || micPhase !== "idle"}
        className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-white text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-40"
        aria-label={onOpenSessionPanel ? "Open voice session" : "Voice input"}
        title={onOpenSessionPanel ? "Voice session" : "Voice input"}
      >
        <WaveformIcon className="h-5 w-5" />
      </button>
    </>
  );

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-transparent">
      <div className="flex min-h-0 flex-1 flex-row">
        {isChatActive && (
          <aside
            className="glass-panel mb-3 ml-2 mt-2 hidden w-60 shrink-0 flex-col rounded-2xl border border-white/10 md:mb-4 md:ml-3 md:mt-3 md:flex"
            aria-label="Conversation"
          >
            <div className="border-b border-white/10 px-4 py-3">
              <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-white/50">Chats</p>
              <p className="mt-1 truncate text-sm font-medium text-white">This conversation</p>
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-3">
              <div className="glass-panel-subtle rounded-xl border border-white/10 px-3 py-2.5 text-xs text-white/80">
                Replies stay in this thread.
              </div>
            </div>
          </aside>
        )}

        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div
            ref={listRef}
            className={`relative min-h-0 flex-1 overflow-y-auto ${
              elevateComposerLayout && !isChatActive ? "max-h-[min(52vh,520px)] md:max-h-[min(48vh,560px)]" : ""
            }`}
            role="log"
            aria-live="polite"
          >
            <div
              className={`absolute inset-0 z-0 flex flex-col items-center justify-center px-4 pb-8 pt-4 text-center transition-opacity duration-500 ease-out ${
                isChatActive ? "pointer-events-none opacity-0" : "opacity-100"
              }`}
              aria-hidden={isChatActive}
            >
              <div className="glass-panel max-w-xl rounded-3xl border border-white/10 bg-white/[0.05] px-8 py-10 shadow-[0_8px_40px_rgba(0,0,0,0.25)] backdrop-blur-md">
                <h1 className="max-w-lg text-3xl font-light leading-tight tracking-tight text-white md:text-4xl">
                  The space to be heard.
                </h1>
                <p
                  key={homeInteractionMode}
                  className="mt-10 max-w-md animate-hero-mode-desc text-sm leading-relaxed text-white/70 md:text-[0.95rem]"
                >
                  {HOME_MODE_META[homeInteractionMode].description}
                </p>
              </div>
            </div>

            {(hasConversation || rawTranscript || reviewText) && (
              <div className="relative z-10 mx-auto w-full max-w-[48rem] px-3 py-8 md:px-6">
                {error && (
                  <div className="glass-panel mb-6 rounded-2xl px-4 py-3 text-sm text-red-200">
                    {error}
                  </div>
                )}

                {(rawTranscript || reviewText) && (
                  <div className="glass-panel mb-8 rounded-2xl border border-white/10 px-4 py-4 md:px-5">
                    <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-white/50">Journal audio pipeline</p>
                    <p className="mt-2 text-sm text-white/70">
                      {"Transcribe -> human review/edit -> AI reformat + validation. You can loop this as many times as you want."}
                    </p>
                    {saveEntry && (
                      <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-end">
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-white/60" htmlFor={`${idPrefix}-journal-date`}>
                            Entry date
                          </label>
                          <input
                            id={`${idPrefix}-journal-date`}
                            type="date"
                            value={journalEntryDate}
                            onChange={(e) => setJournalEntryDate(e.target.value)}
                            className="rounded-xl border border-white/15 bg-black/30 px-3 py-2 text-sm text-white focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                          />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-white/60" htmlFor={`${idPrefix}-journal-time`}>
                            Entry time
                          </label>
                          <input
                            id={`${idPrefix}-journal-time`}
                            type="time"
                            value={journalEntryTime}
                            onChange={(e) => setJournalEntryTime(e.target.value)}
                            className="rounded-xl border border-white/15 bg-black/30 px-3 py-2 text-sm text-white focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                          />
                        </div>
                      </div>
                    )}
                    <textarea
                      value={reviewText}
                      onChange={(e) => setReviewText(e.target.value)}
                      rows={8}
                      placeholder="Transcript will appear here..."
                      className="mt-3 w-full resize-y rounded-xl border border-white/15 bg-black/25 px-3 py-3 text-sm text-white placeholder:text-white/40 focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                    />
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <label className="text-xs text-white/60" htmlFor={`${idPrefix}-validation-model`}>
                        Cleanup/feedback model
                      </label>
                      <select
                        id={`${idPrefix}-validation-model`}
                        value={validationModel}
                        onChange={(e) => setValidationModel(e.target.value)}
                        className="rounded-full border border-white/20 bg-black/30 px-3 py-1.5 text-xs text-white focus:border-white/30 focus:outline-none"
                      >
                        <option value="openai/gpt-5.4">openai/gpt-5.4</option>
                        <option value="anthropic/claude-sonnet-4.6">anthropic/claude-sonnet-4.6</option>
                        <option value="anthropic/claude-opus-4.6">anthropic/claude-opus-4.6</option>
                      </select>
                    </div>
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      {saveEntry && (
                        <button
                          type="button"
                          onClick={() => void saveToJournal()}
                          disabled={
                            savingJournal ||
                            !journalEntryDate.trim() ||
                            !journalEntryTime.trim() ||
                            !(validatedJournal.trim() || reviewText.trim())
                          }
                          className="rounded-full border border-white/25 bg-white/10 px-4 py-1.5 text-xs font-medium text-white shadow-sm transition hover:bg-white/15 disabled:opacity-50"
                        >
                          {savingJournal ? "Saving…" : "Save to journal"}
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void runJournalValidation()}
                        disabled={!reviewText.trim() || validating}
                        className="rounded-full bg-white px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-50"
                      >
                        {validating ? "Validating..." : "Run AI validation"}
                      </button>
                      {validatedJournal && (
                        <>
                          <button
                            type="button"
                            onClick={() => setDraft(validatedJournal)}
                            className="rounded-full border border-white/20 px-4 py-1.5 text-xs font-medium text-white/90 transition hover:bg-white/10"
                          >
                            Use as draft
                          </button>
                          <button
                            type="button"
                            onClick={startAnotherEntry}
                            className="rounded-full border border-white/20 px-4 py-1.5 text-xs font-medium text-white/90 transition hover:bg-white/10"
                          >
                            Start another entry
                          </button>
                        </>
                      )}
                    </div>
                    {validatedJournal && (
                      <div className="mt-4 space-y-3 rounded-xl border border-white/10 bg-white/[0.04] p-3">
                        <p className="text-xs font-semibold uppercase tracking-[0.15em] text-white/55">AI reformatted journal</p>
                        <textarea
                          value={validatedJournal}
                          onChange={(e) => setValidatedJournal(e.target.value)}
                          rows={8}
                          className="w-full resize-y rounded-xl border border-white/15 bg-black/25 px-3 py-3 text-sm text-white focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                        />
                        {validationFeedback && (
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-white/55">Feedback</p>
                            <p className="mt-1 whitespace-pre-wrap text-sm text-white/80">{validationFeedback}</p>
                          </div>
                        )}
                        {modelUsed && (
                          <p className="text-xs text-white/50">Model used: {modelUsed}</p>
                        )}
                        {validationNotes.length > 0 && (
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-white/55">Validation notes</p>
                            <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-white/75">
                              {validationNotes.map((note, i) => (
                                <li key={`${i}-${note.slice(0, 18)}`}>{note}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`group mb-10 w-full ${m.role === "user" ? "flex justify-end" : ""}`}
                  >
                    {m.role === "assistant" ? (
                      <div className="glass-panel-subtle max-w-none rounded-2xl border border-white/10 px-5 py-4">
                        <div className="text-[0.95rem] leading-7 text-white/95">
                          <p className="whitespace-pre-wrap break-words">{m.content}</p>
                        </div>
                        <div className="mt-2 flex items-center gap-0.5 text-white/50 opacity-90 transition-opacity group-hover:opacity-100">
                          <button
                            type="button"
                            onClick={() => void copyText(m.content)}
                            className="rounded-lg p-2 hover:bg-white/10 hover:text-white"
                            title="Copy"
                            aria-label="Copy response"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </button>
                          <button
                            type="button"
                            onClick={() => readAloud(m.content)}
                            className="rounded-lg p-2 hover:bg-white/10 hover:text-white"
                            title="Read aloud"
                            aria-label="Read response aloud"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="glass-panel max-w-[min(100%,85%)] rounded-[1.75rem] px-5 py-3 text-[0.95rem] leading-7 text-white/95">
                        <p className="whitespace-pre-wrap break-words">{m.content}</p>
                      </div>
                    )}
                  </div>
                ))}

                {sending && (
                  <div className="glass-panel-subtle mb-10 inline-block rounded-2xl border border-white/10 px-4 py-3 text-[0.95rem] text-white/60">
                    <span className="inline-flex gap-1">
                      <span className="animate-pulse">Thinking</span>
                      <span className="inline-flex gap-0.5">
                        <span className="animate-bounce" style={{ animationDelay: "0ms" }}>.</span>
                        <span className="animate-bounce" style={{ animationDelay: "150ms" }}>.</span>
                        <span className="animate-bounce" style={{ animationDelay: "300ms" }}>.</span>
                      </span>
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div
            className={`flex flex-none flex-col bg-transparent px-3 pt-2 transition-[padding] duration-500 ease-out md:px-4 ${
              elevateComposerLayout ? "pb-2" : "pb-5"
            } ${isChatActive ? "border-t border-white/[0.07]" : ""}`}
          >
            <div className="mx-auto w-full max-w-[48rem]">
              {!isChatActive && error && (
                <div className="glass-panel mb-4 rounded-2xl px-4 py-3 text-sm text-red-200">
                  {error}
                </div>
              )}

              <div
                className={`glass-panel flex items-center gap-1 rounded-full pl-3 pr-2 transition-shadow duration-500 md:gap-2 md:pl-4 ${
                  micPhase === "recording" ? "ring-2 ring-red-400/40" : ""
                } ${isChatActive ? "shadow-[0_-4px_24px_rgba(0,0,0,0.15)]" : ""}`}
              >
                {composerInner}
              </div>

              <div
                className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3"
                role="radiogroup"
                aria-label="How you want to use the assistant"
              >
                {HOME_MODES.map((mode) => {
                  const selected = homeInteractionMode === mode;
                  const meta = HOME_MODE_META[mode];
                  return (
                    <button
                      key={mode}
                      type="button"
                      role="radio"
                      aria-checked={selected}
                      onClick={() => setHomeInteractionMode(mode)}
                      className={`group flex flex-col items-stretch rounded-2xl border px-4 py-3.5 text-left transition-all duration-300 ease-out focus:outline-none focus-visible:ring-2 focus-visible:ring-white/25 focus-visible:ring-offset-2 focus-visible:ring-offset-transparent ${
                        selected
                          ? "border-white/30 bg-white/[0.14] text-white shadow-[0_0_24px_-4px_rgba(255,255,255,0.12)] ring-1 ring-white/15"
                          : "border-white/10 bg-white/[0.04] text-white/45 hover:border-white/20 hover:bg-white/[0.08] hover:text-white/75 hover:shadow-[0_0_20px_-6px_rgba(255,255,255,0.08)]"
                      }`}
                    >
                      <span className={`text-sm font-semibold tracking-tight ${selected ? "text-white" : "text-white/80"}`}>
                        {meta.label}
                      </span>
                      <span
                        className={`mt-1 text-xs font-medium leading-snug transition-colors duration-300 ${
                          selected ? "text-white/75" : "text-white/50 group-hover:text-white/65"
                        }`}
                      >
                        {meta.sublabel}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
            {elevateComposerLayout && !isChatActive ? (
              <div
                className="mx-auto mt-4 w-full max-w-[48rem] flex-none min-h-[6.5rem] sm:min-h-[7.5rem]"
                aria-hidden
              />
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};
