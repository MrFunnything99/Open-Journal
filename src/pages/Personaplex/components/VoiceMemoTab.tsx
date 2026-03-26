import { FC, useCallback, useEffect, useId, useRef, useState } from "react";
import { backendFetch } from "../../../backendApi";
import { CHAT_INTERACTION_MODE_META } from "../chatInteractionModes";
import type { ChatMessage } from "../hooks/useJournalHistory";
import { usePersonaplexChat } from "../PersonaplexChatContext";
import { blobToWavBase64 } from "../utils/audioToWav";
import { AskAnythingComposer, LiveDictationBubble } from "./GlobalAskAnythingBar";

type Props = {
  onToast: (msg: string) => void;
  /** Persist to The Brain → Knowledge base → Journals (uses date for folder layout + memory ingest). */
  saveEntry?: (transcript: ChatMessage[], dateIso: string) => string;
  syncUnsyncedEntries?: () => Promise<number>;
};

function effectiveRecorderMime(recorder: MediaRecorder): string {
  const t = recorder.mimeType?.trim();
  if (t) return t;
  // Safari (iOS and desktop) often emits MP4/AAC but leaves mimeType empty when WebM isn't used.
  if (typeof navigator !== "undefined" && navigator.vendor === "Apple Computer, Inc.") {
    return "audio/mp4";
  }
  return "audio/webm";
}

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

export const VoiceMemoTab: FC<Props> = ({ onToast, saveEntry, syncUnsyncedEntries }) => {
  const {
    messages,
    sending,
    isChatActive,
    chatError,
    setDraft: setGlobalDraft,
    chatInteractionMode,
    journalFileToProcess,
    clearJournalFileToProcess,
  } = usePersonaplexChat();
  const idPrefix = useId();
  const [journalMicPhase, setJournalMicPhase] = useState<"idle" | "recording" | "processing">("idle");
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
  const [journalEditorOpen, setJournalEditorOpen] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, sending, isChatActive]);

  useEffect(() => {
    if (chatInteractionMode !== "journal") setJournalEditorOpen(false);
  }, [chatInteractionMode]);

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

  const processMicAudio = useCallback(async (blob: Blob) => {
    setJournalMicPhase("processing");
    setError(null);
    try {
      // OpenRouter `input_audio` is most reliable with WAV.
      const b64 = await blobToWavBase64(blob);
      const filename = "dictation.wav";
      const isJournal = chatInteractionMode === "journal";
      const res = await backendFetch("/voice-memo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio: b64,
          filename,
          mime_type: "audio/wav",
          journal_mode: isJournal,
        }),
      });
      const data = (await res.json().catch(() => ({}))) as {
        detail?: string;
        error?: string;
        polished_text?: string;
        raw_transcript?: string;
        cleaned_transcript?: string;
      };
      if (!res.ok) {
        const d = data.detail;
        const msg =
          typeof d === "string" ? d : typeof data.error === "string" ? data.error : `Request failed (${res.status})`;
        throw new Error(msg);
      }
      const line = (data.polished_text ?? data.raw_transcript ?? "").trim();
      if (line) {
        const capturedAt = new Date();
        const rawBody = ((data.raw_transcript ?? "").trim() || line).trim();
        setRawTranscript(rawBody);
        const reviewBody = isJournal && data.cleaned_transcript?.trim()
          ? data.cleaned_transcript.trim()
          : rawBody;
        setReviewText(reviewBody);
        if (isJournal) setJournalEditorOpen(true);
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
      setJournalMicPhase("idle");
    }
  }, [chatInteractionMode]);

  useEffect(() => {
    if (journalFileToProcess && chatInteractionMode === "journal") {
      const file = journalFileToProcess;
      clearJournalFileToProcess();
      void processMicAudio(file);
    }
  }, [journalFileToProcess, chatInteractionMode, clearJournalFileToProcess, processMicAudio]);

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
    if (journalMicPhase !== "idle" || sending) return;
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
        void processMicAudio(blob);
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setJournalMicPhase("recording");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Microphone unavailable");
      setJournalMicPhase("idle");
    }
  }, [journalMicPhase, sending, processMicAudio]);

  const stopRecording = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== "inactive") mr.stop();
    else setJournalMicPhase("idle");
  }, []);

  const onPickFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (!f) return;
      setError(null);
      void processMicAudio(f);
      e.target.value = "";
    },
    [processMicAudio]
  );

  const journalRecorderBusy = journalMicPhase !== "idle" || validating;

  const hasConversation = messages.length > 0 || sending;
  /** ChatGPT-style: idle = centered prompt + composer; active = transcript + pinned composer. */
  const showHomeHeroStack =
    !isChatActive && !hasConversation && !rawTranscript && !reviewText;
  const showBottomComposer = !showHomeHeroStack;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-transparent">
      <div className="flex min-h-0 flex-1 flex-row">
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div ref={listRef} className="relative flex min-h-0 flex-1 flex-col overflow-y-auto" role="log" aria-live="polite">
            {showHomeHeroStack && (
              <div className="flex min-h-[min(55vh,560px)] flex-1 flex-col items-center justify-center gap-6 px-4 py-8 text-center sm:min-h-[50vh]">
                <h1 className="max-w-lg text-[1.65rem] font-normal leading-snug tracking-tight text-white sm:text-3xl md:text-[1.75rem]">
                  What can I help with?
                </h1>
                <p
                  key={chatInteractionMode}
                  className="max-w-md animate-hero-mode-desc text-sm leading-relaxed text-white/55 md:text-[0.95rem]"
                >
                  {CHAT_INTERACTION_MODE_META[chatInteractionMode].description}
                </p>
                <div className="w-full max-w-2xl space-y-2 text-left">
                  <LiveDictationBubble />
                  <AskAnythingComposer layout="center" />
                </div>
              </div>
            )}

            {(hasConversation || rawTranscript || reviewText) && (
              <div className="relative z-10 mx-auto w-full max-w-[48rem] px-3 py-8 md:px-6">
                {(error || chatError) && (
                  <div className="glass-panel mb-6 rounded-2xl px-4 py-3 text-sm text-red-200">
                    {error || chatError}
                  </div>
                )}

                {chatInteractionMode === "journal" && (rawTranscript || reviewText) && (
                  <div className="glass-panel mb-8 rounded-2xl border border-white/10 px-4 py-4 md:px-5">
                    <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-white/50">Journal audio pipeline</p>
                    <p className="mt-2 text-sm text-white/70">
                      {"Transcribe → auto-cleanup → review/edit → AI feedback. Click the editor button to open."}
                    </p>
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="audio/*,.mp3,.m4a,.wav,.webm,.ogg,.flac"
                        className="hidden"
                        onChange={onPickFile}
                      />
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={journalRecorderBusy || sending}
                        className="rounded-full border border-white/25 bg-white/10 px-3 py-1.5 text-xs font-medium text-white/90 transition hover:bg-white/15 disabled:opacity-50"
                      >
                        Attach audio
                      </button>
                      {journalMicPhase === "recording" ? (
                        <button
                          type="button"
                          onClick={stopRecording}
                          className="rounded-full bg-red-500/80 px-3 py-1.5 text-xs font-medium text-white"
                        >
                          Stop recording
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => void startRecording()}
                          disabled={journalRecorderBusy || sending}
                          className="rounded-full bg-emerald-500/80 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
                        >
                          {journalMicPhase === "processing" ? "Transcribing…" : "Record"}
                        </button>
                      )}
                    </div>
                    <div className="mt-3 rounded-xl border border-white/10 bg-black/20 p-3">
                      <p className="text-xs text-white/60">Cleaned transcript preview</p>
                      <p className="mt-1 line-clamp-3 whitespace-pre-wrap break-words text-sm text-white/85">
                        {(reviewText || rawTranscript).trim() || "Transcript will appear here..."}
                      </p>
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
                        onClick={() => setJournalEditorOpen(true)}
                        disabled={!(reviewText || rawTranscript).trim()}
                        className="rounded-full bg-white px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-50"
                      >
                        Open editor
                      </button>
                      {validatedJournal && (
                        <>
                          <button
                            type="button"
                            onClick={() => setGlobalDraft(validatedJournal)}
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
                        {(m.agentSteps && m.agentSteps.length > 0) || (m.actions && m.actions.length > 0) || m.retrievalLog ? (
                          <details className="mt-3 rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-white/80">
                            <summary className="cursor-pointer select-none text-xs font-semibold uppercase tracking-[0.15em] text-white/60">
                              Agent activity
                            </summary>
                            <div className="mt-2 space-y-3 text-xs">
                              {m.actions && m.actions.length > 0 ? (
                                <div>
                                  <p className="font-semibold text-white/70">Actions</p>
                                  <ul className="mt-1 list-disc space-y-1 pl-5 text-white/75">
                                    {m.actions.map((a, idx) => (
                                      <li key={`${m.id}-act-${idx}`}>
                                        Opened <span className="font-medium text-white/85">{a.view}</span>
                                        {a.brainSection ? (
                                          <>
                                            {" "}
                                            · <span className="font-medium text-white/85">{a.brainSection}</span>
                                          </>
                                        ) : null}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              ) : null}

                              {m.agentSteps && m.agentSteps.length > 0 ? (
                                <div>
                                  <p className="font-semibold text-white/70">Steps</p>
                                  <ul className="mt-1 list-disc space-y-1 pl-5 text-white/75">
                                    {m.agentSteps
                                      .filter((s) => typeof s?.summary === "string" && (s.summary ?? "").trim())
                                      .slice(0, 12)
                                      .map((s, idx) => {
                                        const kind = typeof s.kind === "string" ? s.kind : "";
                                        const name = typeof s.name === "string" ? s.name : "";
                                        const summary = (s.summary ?? "").trim();
                                        return (
                                          <li key={`${m.id}-step-${idx}`}>
                                            {kind ? <span className="font-medium text-white/85">{kind}</span> : "step"}
                                            {name ? <span className="text-white/55"> · {name}</span> : null}
                                            {": "}
                                            {summary}
                                          </li>
                                        );
                                      })}
                                  </ul>
                                  {m.agentSteps.length > 12 ? (
                                    <p className="mt-1 text-[11px] text-white/50">Showing first 12 steps.</p>
                                  ) : null}
                                </div>
                              ) : null}

                              {m.retrievalLog ? (
                                <div>
                                  <p className="font-semibold text-white/70">Memory context (vector DB)</p>
                                  <pre className="mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-white/10 bg-black/30 p-2 text-[11px] text-white/70">
                                    {m.retrievalLog}
                                  </pre>
                                </div>
                              ) : null}
                            </div>
                          </details>
                        ) : null}
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

          {showBottomComposer && (
            <div className="flex-none border-t border-white/10 bg-[#0a0a12]/90 px-3 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur-md">
              <div className="mx-auto w-full max-w-[48rem] space-y-2">
                <LiveDictationBubble />
                <AskAnythingComposer layout="center" />
              </div>
            </div>
          )}
        </div>
      </div>

      {chatInteractionMode === "journal" && journalEditorOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="mx-4 flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/15 bg-[#121218] shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
              <div>
                <h2 className="text-sm font-semibold text-white">Journal editor</h2>
                <p className="text-xs text-white/50">Review and clean your transcript</p>
              </div>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => void copyText(reviewText)}
                  className="rounded-lg p-2 text-white/60 hover:bg-white/10 hover:text-white"
                  title="Copy"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </button>
                <button
                  type="button"
                  onClick={() => readAloud(reviewText)}
                  className="rounded-lg p-2 text-white/60 hover:bg-white/10 hover:text-white"
                  title="Read aloud"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                  </svg>
                </button>
                <button
                  type="button"
                  onClick={() => setJournalEditorOpen(false)}
                  className="rounded-lg p-2 text-white/60 hover:bg-white/10 hover:text-white"
                  title="Close"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
              {saveEntry && (
                <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-end">
                  <div className="flex flex-col gap-1">
                    <label className="text-xs text-white/60" htmlFor={`${idPrefix}-modal-journal-date`}>
                      Entry date
                    </label>
                    <input
                      id={`${idPrefix}-modal-journal-date`}
                      type="date"
                      value={journalEntryDate}
                      onChange={(e) => setJournalEntryDate(e.target.value)}
                      className="rounded-xl border border-white/15 bg-black/30 px-3 py-2 text-sm text-white focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-xs text-white/60" htmlFor={`${idPrefix}-modal-journal-time`}>
                      Entry time
                    </label>
                    <input
                      id={`${idPrefix}-modal-journal-time`}
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
                rows={10}
                placeholder="Transcript will appear here..."
                className="w-full resize-y rounded-xl border border-white/15 bg-black/25 px-3 py-3 text-sm text-white placeholder:text-white/40 focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
              />

              <div className="flex flex-wrap items-center gap-2">
                <label className="text-xs text-white/60" htmlFor={`${idPrefix}-modal-validation-model`}>
                  Cleanup/feedback model
                </label>
                <select
                  id={`${idPrefix}-modal-validation-model`}
                  value={validationModel}
                  onChange={(e) => setValidationModel(e.target.value)}
                  className="rounded-full border border-white/20 bg-black/30 px-3 py-1.5 text-xs text-white focus:border-white/30 focus:outline-none"
                >
                  <option value="openai/gpt-5.4">openai/gpt-5.4</option>
                  <option value="anthropic/claude-sonnet-4.6">anthropic/claude-sonnet-4.6</option>
                  <option value="anthropic/claude-opus-4.6">anthropic/claude-opus-4.6</option>
                </select>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void runJournalValidation()}
                  disabled={!reviewText.trim() || validating}
                  className="rounded-full bg-white px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-50"
                >
                  {validating ? "Validating..." : "Run AI validation"}
                </button>
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
                {validatedJournal && (
                  <>
                    <button
                      type="button"
                      onClick={() => { setGlobalDraft(validatedJournal); setJournalEditorOpen(false); }}
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
                <div className="space-y-3 rounded-xl border border-white/10 bg-white/[0.04] p-3">
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
          </div>
        </div>
      )}
    </div>
  );
};
