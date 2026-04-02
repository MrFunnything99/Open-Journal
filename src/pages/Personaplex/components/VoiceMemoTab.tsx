import { FC, useCallback, useEffect, useId, useRef, useState } from "react";
import { backendFetch } from "../../../backendApi";
import { CHAT_INTERACTION_MODE_META } from "../chatInteractionModes";
import type { ChatMessage } from "../hooks/useJournalHistory";
import { personaplexChatToJournalTranscript, usePersonaplexChat } from "../PersonaplexChatContext";
import { blobToBase64, blobToWavBase64 } from "../utils/audioToWav";
import { AskAnythingComposer, LiveDictationBubble } from "./GlobalAskAnythingBar";
import { playChatReadAloud } from "../utils/chatReadAloud";

type Props = {
  onToast: (msg: string) => void;
  saveEntry?: (transcript: ChatMessage[], dateIso: string, source?: "journal" | "conversation") => string;
  syncUnsyncedEntries?: () => Promise<number>;
};

function effectiveRecorderMime(recorder: MediaRecorder): string {
  const t = recorder.mimeType?.trim();
  if (t) return t;
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

/** Insert transcribed speech into the journal editor with readable spacing (multi-segment dictation). */
function insertTranscriptSegment(existing: string, segment: string, insertAt?: number | null): string {
  const seg = segment.trim();
  if (!seg) return existing;

  const len = existing.length;
  const at =
    insertAt == null || Number.isNaN(insertAt)
      ? len
      : Math.max(0, Math.min(Math.floor(insertAt), len));
  const before = existing.slice(0, at);
  const after = existing.slice(at);

  let glueLeft = "";
  if (before.trim().length > 0) {
    if (before.endsWith("\n\n")) glueLeft = "";
    else if (before.endsWith("\n")) glueLeft = "\n";
    else glueLeft = "\n\n";
  }

  let glueRight = "";
  if (after.trim().length > 0) {
    glueRight = after.startsWith("\n") ? "" : "\n\n";
  }

  return `${before}${glueLeft}${seg}${glueRight}${after}`;
}

export const VoiceMemoTab: FC<Props> = ({ onToast, saveEntry, syncUnsyncedEntries }) => {
  const {
    messages,
    sending,
    isChatActive,
    chatError,
    chatInteractionMode,
    setChatInteractionMode,
    journalFileToProcess,
    clearJournalFileToProcess,
    journalTextToProcess,
    clearJournalTextToProcess,
    resetAssistedWorkspace,
  } = usePersonaplexChat();
  const idPrefix = useId();
  const [journalMicPhase, setJournalMicPhase] = useState<"idle" | "recording" | "processing">("idle");
  const [error, setError] = useState<string | null>(null);
  const [rawTranscript, setRawTranscript] = useState("");
  const [reviewText, setReviewText] = useState("");
  const [feedbackModel, setFeedbackModel] = useState("openai/gpt-5.4");
  const [gettingFeedback, setGettingFeedback] = useState(false);
  const [journalCleanupBusy, setJournalCleanupBusy] = useState(false);
  const [journalEntryDate, setJournalEntryDate] = useState("");
  const [journalEntryTime, setJournalEntryTime] = useState("");
  const [savingJournal, setSavingJournal] = useState(false);
  const [journalMessages, setJournalMessages] = useState<ChatMessage[]>([]);
  const [readAloudBusy, setReadAloudBusy] = useState(false);

  // --- AI-Assisted Finalizer (distinct from Manual Journal editor) ---
  const [finalizerOpen, setFinalizerOpen] = useState(false);
  const [finalizerText, setFinalizerText] = useState("");
  const [finalizerMessages, setFinalizerMessages] = useState<ChatMessage[]>([]);
  const [finalizerDate, setFinalizerDate] = useState("");
  const [finalizerTime, setFinalizerTime] = useState("");
  const [finalizerSaving, setFinalizerSaving] = useState(false);
  const [finalizerCleanupBusy, setFinalizerCleanupBusy] = useState(false);

  const listRef = useRef<HTMLDivElement>(null);
  const journalScrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const journalTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const reviewTextRef = useRef(reviewText);
  reviewTextRef.current = reviewText;
  /** Insert offset for the next journal voice/file transcription (captured when record/file pick starts). */
  const journalVoiceInsertPosRef = useRef<number | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const captureJournalVoiceInsertPosition = useCallback(() => {
    const ta = journalTextareaRef.current;
    if (ta && document.activeElement === ta) {
      journalVoiceInsertPosRef.current = ta.selectionStart;
    } else {
      journalVoiceInsertPosRef.current = reviewTextRef.current.length;
    }
  }, []);

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, sending, isChatActive]);

  useEffect(() => {
    const el = journalScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [journalMessages, gettingFeedback]);

  useEffect(() => {
    if (chatInteractionMode === "learning") {
      setChatInteractionMode("journal");
    }
  }, [chatInteractionMode, setChatInteractionMode]);

  const handoffAssistedToJournal = useCallback(() => {
    if (chatInteractionMode !== "autobiography" || sending || messages.length === 0) return;
    const transcript = personaplexChatToJournalTranscript(messages);
    const fullText = transcript.map((m) => {
      const label = m.role === "user" ? "You" : "AI";
      return `### ${label}\n\n${m.text}`;
    }).join("\n\n---\n\n");
    setFinalizerMessages(transcript);
    setFinalizerText(fullText);
    const now = new Date();
    setFinalizerDate(toDateInputValue(now));
    setFinalizerTime(toTimeInputValue(now));
    setFinalizerOpen(true);
  }, [chatInteractionMode, sending, messages]);

  const finalizerDismiss = useCallback(() => {
    setFinalizerOpen(false);
    setFinalizerText("");
    setFinalizerMessages([]);
    setFinalizerDate("");
    setFinalizerTime("");
  }, []);

  const finalizerSave = useCallback(async () => {
    if (!saveEntry || finalizerSaving) return;
    if (!finalizerDate.trim() || !finalizerTime.trim()) {
      onToast("Set entry date and time before saving.");
      return;
    }
    const edited = finalizerText.trim();
    const transcript: ChatMessage[] = edited
      ? [{ role: "user" as const, text: edited }]
      : finalizerMessages;
    if (transcript.length === 0) {
      onToast("Nothing to save.");
      return;
    }
    const dateIso = dateAndTimeToIso(finalizerDate, finalizerTime);
    setFinalizerSaving(true);
    try {
      const id = saveEntry(transcript, dateIso, "conversation");
      if (!id) { onToast("Could not save."); return; }
      void syncUnsyncedEntries?.();
      onToast("Saved to your journal.");
      finalizerDismiss();
      resetAssistedWorkspace();
    } finally {
      setFinalizerSaving(false);
    }
  }, [
    saveEntry, syncUnsyncedEntries, finalizerSaving,
    finalizerText, finalizerMessages,
    finalizerDate, finalizerTime,
    onToast, finalizerDismiss, resetAssistedWorkspace,
  ]);

  const finalizerCleanup = useCallback(async () => {
    const text = finalizerText.trim();
    if (!text || finalizerCleanupBusy) return;
    setFinalizerCleanupBusy(true);
    try {
      const res = await backendFetch("/journal-cleanup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = (await res.json().catch(() => ({}))) as { detail?: string; cleaned_text?: string };
      if (!res.ok) {
        const d = data.detail;
        throw new Error(typeof d === "string" ? d : `Correction failed (${res.status})`);
      }
      const cleaned = (data.cleaned_text ?? "").trim();
      if (!cleaned) throw new Error("No corrected text returned.");
      setFinalizerText(cleaned);
      onToast("Spelling and formatting applied.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Correction failed";
      onToast(msg);
    } finally {
      setFinalizerCleanupBusy(false);
    }
  }, [finalizerText, finalizerCleanupBusy, onToast]);

  const readAloud = useCallback(
    (text: string) => {
      void playChatReadAloud(text, onToast, { onLoading: setReadAloudBusy });
    },
    [onToast],
  );

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
      const isUploadedFile = blob instanceof File && blob.name?.trim();
      let b64: string;
      let filename: string;
      let mimeType: string;
      if (isUploadedFile) {
        b64 = await blobToBase64(blob);
        filename = (blob as File).name;
        mimeType = blob.type || "audio/mpeg";
      } else {
        b64 = await blobToWavBase64(blob);
        filename = "dictation.wav";
        mimeType = "audio/wav";
      }
      const isJournal = chatInteractionMode === "journal";
      const res = await backendFetch("/voice-memo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio: b64,
          filename,
          mime_type: mimeType,
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
        // Manual Journal Mode: show raw transcript only; polish via "AI Spelling Correction/Reformatting".
        const reviewBody = !isJournal && data.cleaned_transcript?.trim()
          ? data.cleaned_transcript.trim()
          : rawBody;

        if (isJournal) {
          const pos = journalVoiceInsertPosRef.current;
          journalVoiceInsertPosRef.current = null;
          setReviewText((prev) => {
            const next = insertTranscriptSegment(prev, reviewBody, pos ?? undefined);
            setRawTranscript(next);
            return next;
          });
        } else {
          setRawTranscript(rawBody);
          setReviewText(reviewBody);
        }
        if (!journalEntryDate) setJournalEntryDate(toDateInputValue(capturedAt));
        if (!journalEntryTime) setJournalEntryTime(toTimeInputValue(capturedAt));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Transcription failed");
    } finally {
      setJournalMicPhase("idle");
    }
  }, [chatInteractionMode, journalEntryDate, journalEntryTime]);

  useEffect(() => {
    if (journalFileToProcess && chatInteractionMode === "journal") {
      const file = journalFileToProcess;
      clearJournalFileToProcess();
      journalVoiceInsertPosRef.current = reviewTextRef.current.length;
      void processMicAudio(file);
    }
  }, [journalFileToProcess, chatInteractionMode, clearJournalFileToProcess, processMicAudio]);

  useEffect(() => {
    if (journalTextToProcess && chatInteractionMode === "journal") {
      const text = journalTextToProcess;
      clearJournalTextToProcess();
      const now = new Date();
      setReviewText(text);
      setRawTranscript(text);
      if (!journalEntryDate) setJournalEntryDate(toDateInputValue(now));
      if (!journalEntryTime) setJournalEntryTime(toTimeInputValue(now));
    }
  }, [journalTextToProcess, chatInteractionMode, clearJournalTextToProcess, journalEntryDate, journalEntryTime]);

  const getJournalFeedback = useCallback(async () => {
    const text = reviewText.trim();
    if (!text || gettingFeedback) return;

    setJournalMessages((prev) => [...prev, { role: "user", text }]);
    setReviewText("");
    setRawTranscript("");
    setGettingFeedback(true);
    setError(null);

    try {
      const res = await backendFetch("/journal-validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: feedbackModel.trim() || undefined }),
      });
      const data = (await res.json().catch(() => ({}))) as {
        detail?: string;
        reformatted_journal?: string;
        feedback?: string;
        validation_notes?: string[];
        model_used?: string;
      };
      if (!res.ok) {
        throw new Error(data.detail || `Feedback failed (${res.status})`);
      }
      const parts: string[] = [];
      const feedback = (data.feedback || "").trim();
      if (feedback) parts.push(feedback);
      const notes = Array.isArray(data.validation_notes)
        ? data.validation_notes.filter((x) => typeof x === "string" && x.trim())
        : [];
      if (notes.length > 0) {
        parts.push(notes.map((n) => `- ${n}`).join("\n"));
      }
      const aiText = parts.join("\n\n") || "Looks good — no notes.";
      setJournalMessages((prev) => [...prev, { role: "ai", text: aiText }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Feedback failed";
      setError(msg);
      onToast(msg);
      setJournalMessages((prev) => [...prev, { role: "ai", text: `Error: ${msg}` }]);
    } finally {
      setGettingFeedback(false);
    }
  }, [reviewText, gettingFeedback, onToast, feedbackModel]);

  const runJournalCleanup = useCallback(async () => {
    const text = reviewText.trim();
    if (!text || journalCleanupBusy || gettingFeedback) return;

    setJournalCleanupBusy(true);
    setError(null);
    try {
      const res = await backendFetch("/journal-cleanup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: feedbackModel.trim() || undefined }),
      });
      const data = (await res.json().catch(() => ({}))) as {
        detail?: string;
        cleaned_text?: string;
      };
      if (!res.ok) {
        const d = data.detail;
        throw new Error(typeof d === "string" ? d : `Correction failed (${res.status})`);
      }
      const cleaned = (data.cleaned_text ?? "").trim();
      if (!cleaned) throw new Error("No corrected text returned.");
      setReviewText(cleaned);
      onToast("Spelling and formatting applied.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Correction failed";
      setError(msg);
      onToast(msg);
    } finally {
      setJournalCleanupBusy(false);
    }
  }, [reviewText, journalCleanupBusy, gettingFeedback, feedbackModel, onToast]);

  const startAnotherEntry = useCallback(() => {
    setRawTranscript("");
    setReviewText("");
    setJournalMessages([]);
    setJournalEntryDate("");
    setJournalEntryTime("");
    setError(null);
  }, []);

  const saveToJournal = useCallback(async () => {
    if (!saveEntry || savingJournal) return;
    const pendingText = reviewText.trim();
    const hasMessages = journalMessages.length > 0;
    if (!pendingText && !hasMessages) {
      onToast("Nothing to save yet.");
      return;
    }
    if (!journalEntryDate.trim() || !journalEntryTime.trim()) {
      onToast("Set entry date and time before saving.");
      return;
    }
    const transcript: ChatMessage[] = [...journalMessages];
    if (pendingText) {
      transcript.push({ role: "user", text: pendingText });
    }
    if (transcript.length === 0) {
      onToast("Nothing to save yet.");
      return;
    }
    const dateIso = dateAndTimeToIso(journalEntryDate, journalEntryTime);
    setSavingJournal(true);
    try {
      const id = saveEntry(transcript, dateIso, "journal");
      if (!id) {
        onToast("Could not save.");
        return;
      }
      void syncUnsyncedEntries?.();
      onToast("Saved to your journal.");
      startAnotherEntry();
      resetAssistedWorkspace();
    } finally {
      setSavingJournal(false);
    }
  }, [
    saveEntry, syncUnsyncedEntries, savingJournal,
    journalMessages, reviewText,
    journalEntryDate, journalEntryTime,
    onToast, startAnotherEntry, resetAssistedWorkspace,
  ]);

  const startRecording = useCallback(async () => {
    if (journalMicPhase !== "idle" || sending) return;
    setError(null);
    if (chatInteractionMode === "journal") {
      captureJournalVoiceInsertPosition();
    }
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
  }, [journalMicPhase, sending, processMicAudio, chatInteractionMode, captureJournalVoiceInsertPosition]);

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
      if (chatInteractionMode === "journal") {
        captureJournalVoiceInsertPosition();
      }
      void processMicAudio(f);
      e.target.value = "";
    },
    [processMicAudio, chatInteractionMode, captureJournalVoiceInsertPosition]
  );

  const journalRecorderBusy = journalMicPhase !== "idle" || gettingFeedback || journalCleanupBusy;
  const journalProcessing = chatInteractionMode === "journal" && journalMicPhase === "processing";
  const hasConversation = messages.length > 0 || sending;
  const assistedHero =
    chatInteractionMode === "autobiography" && !isChatActive && !hasConversation;
  const showAssistedBottomComposer =
    chatInteractionMode === "autobiography" && !assistedHero && !finalizerOpen;

  const errorBanner = (error || chatError) && (
    <div className="glass-panel mb-6 rounded-2xl px-4 py-3 text-sm text-red-200">{error || chatError}</div>
  );

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-transparent">
      <div
        className="flex flex-none flex-wrap items-center justify-center gap-1.5 border-b border-white/[0.08] bg-[#0a0a12]/88 px-3 py-2 backdrop-blur-md sm:gap-2 sm:px-4"
        role="tablist"
        aria-label="Manual Journal Mode and AI-Assisted Journal Mode"
      >
        <button
          type="button"
          role="tab"
          aria-selected={chatInteractionMode === "autobiography"}
          onClick={() => setChatInteractionMode("autobiography")}
          className={`rounded-full px-3 py-1.5 text-xs font-medium transition sm:px-4 sm:text-[0.8rem] ${
            chatInteractionMode === "autobiography"
              ? "bg-white text-gray-900 shadow-sm"
              : "text-white/65 hover:bg-white/10 hover:text-white"
          }`}
        >
          {CHAT_INTERACTION_MODE_META.autobiography.label}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={chatInteractionMode === "journal"}
          onClick={() => setChatInteractionMode("journal")}
          className={`rounded-full px-3 py-1.5 text-xs font-medium transition sm:px-4 sm:text-[0.8rem] ${
            chatInteractionMode === "journal"
              ? "bg-white text-gray-900 shadow-sm"
              : "text-white/65 hover:bg-white/10 hover:text-white"
          }`}
        >
          {CHAT_INTERACTION_MODE_META.journal.label}
        </button>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div ref={listRef} className="relative flex min-h-0 flex-1 flex-col overflow-y-auto" role="log" aria-live="polite">
          {chatInteractionMode === "journal" && (
            <div className="relative z-10 mx-auto w-full max-w-[48rem] px-3 py-6 md:px-6 md:py-8">
              {errorBanner}

              <div className="glass-panel rounded-2xl border border-white/10 px-4 py-4 md:px-5">
                    {/* Header */}
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-white/50">Manual Journal Mode</p>
                        <p className="mt-1 text-sm text-white/70">
                          Write directly in the journal box below — type, record, or attach audio.
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-1">
                        {journalMessages.length > 0 && (
                          <button
                            type="button"
                            onClick={startAnotherEntry}
                            className="rounded-lg px-2 py-1 text-[0.65rem] font-medium text-white/50 hover:bg-white/10 hover:text-white"
                            title="Clear and start fresh"
                          >
                            New entry
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Audio controls + date/time */}
                    <div className="mt-3 flex flex-wrap items-end gap-3">
                      <div className="flex flex-wrap items-center gap-2">
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
                      {saveEntry && (
                        <div className="flex flex-wrap items-end gap-2">
                          <div className="flex flex-col gap-0.5">
                            <label className="text-[0.6rem] text-white/45" htmlFor={`${idPrefix}-jdate`}>Date</label>
                            <input
                              id={`${idPrefix}-jdate`}
                              type="date"
                              value={journalEntryDate}
                              onChange={(e) => setJournalEntryDate(e.target.value)}
                              className="rounded-lg border border-white/15 bg-black/30 px-2 py-1 text-xs text-white focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                            />
                          </div>
                          <div className="flex flex-col gap-0.5">
                            <label className="text-[0.6rem] text-white/45" htmlFor={`${idPrefix}-jtime`}>Time</label>
                            <input
                              id={`${idPrefix}-jtime`}
                              type="time"
                              value={journalEntryTime}
                              onChange={(e) => setJournalEntryTime(e.target.value)}
                              className="rounded-lg border border-white/15 bg-black/30 px-2 py-1 text-xs text-white focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                            />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Conversation thread */}
                    {journalMessages.length > 0 && (
                      <div
                        ref={journalScrollRef}
                        className="mt-4 max-h-[50vh] space-y-4 overflow-y-auto rounded-xl border border-white/10 bg-black/15 p-3"
                      >
                        {journalMessages.map((m, i) => (
                          <div
                            key={`jm-${i}`}
                            className={`w-full ${m.role === "user" ? "flex justify-end" : ""}`}
                          >
                            {m.role === "ai" ? (
                              <div className="max-w-[95%] rounded-2xl border border-white/10 bg-white/[0.06] px-4 py-3">
                                <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-white/90">{m.text}</p>
                                <div className="mt-1.5 flex items-center gap-0.5 text-white/40">
                                  <button
                                    type="button"
                                    onClick={() => void copyText(m.text)}
                                    className="rounded-md p-1 hover:bg-white/10 hover:text-white"
                                    title="Copy"
                                  >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                      <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                    </svg>
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => readAloud(m.text)}
                                    disabled={readAloudBusy}
                                    aria-busy={readAloudBusy}
                                    className="rounded-md p-1 hover:bg-white/10 hover:text-white disabled:pointer-events-none disabled:opacity-35"
                                    title={readAloudBusy ? "Loading audio…" : "Read aloud"}
                                  >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                                      <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                                    </svg>
                                  </button>
                                </div>
                              </div>
                            ) : (
                              <div className="max-w-[85%] rounded-2xl bg-white/[0.12] px-4 py-3">
                                <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-white/95">{m.text}</p>
                              </div>
                            )}
                          </div>
                        ))}
                        {gettingFeedback && (
                          <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.06] px-4 py-3 text-sm text-white/60">
                            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                            Thinking…
                          </div>
                        )}
                      </div>
                    )}

                    {/* Transcribing indicator (before any text arrives) */}
                    {journalProcessing && !(reviewText || rawTranscript).trim() && journalMessages.length === 0 && (
                      <div className="mt-3 flex items-center gap-2 rounded-xl border border-white/10 bg-black/20 p-3 text-sm text-white/70">
                        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                        Transcribing audio…
                      </div>
                    )}

                    {/* Editor textarea */}
                    {(reviewText || rawTranscript || !journalProcessing) && (
                      <textarea
                        ref={journalTextareaRef}
                        value={reviewText}
                        onChange={(e) => setReviewText(e.target.value)}
                        rows={journalMessages.length > 0 ? 6 : 12}
                        placeholder={journalMessages.length > 0 ? "Continue journaling or record more audio…" : "Transcript will appear here — or type directly…"}
                        className="mt-3 w-full resize-y rounded-xl border border-white/15 bg-black/25 px-3 py-3 text-sm leading-relaxed text-white placeholder:text-white/40 focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                        style={{ minHeight: journalMessages.length > 0 ? "6rem" : "12rem" }}
                      />
                    )}

                    {/* Action buttons */}
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      {saveEntry && (
                        <button
                          type="button"
                          onClick={() => void saveToJournal()}
                          disabled={
                            savingJournal ||
                            journalCleanupBusy ||
                            !journalEntryDate.trim() ||
                            !journalEntryTime.trim() ||
                            (!reviewText.trim() && journalMessages.length === 0)
                          }
                          className="rounded-full bg-white px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-50"
                        >
                          {savingJournal ? "Saving…" : "Save to journal"}
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void runJournalCleanup()}
                        disabled={!reviewText.trim() || journalCleanupBusy || gettingFeedback}
                        className="rounded-full border border-white/25 bg-white/10 px-4 py-1.5 text-xs font-medium text-white shadow-sm transition hover:bg-white/15 disabled:opacity-50"
                      >
                        {journalCleanupBusy ? "Applying…" : "AI Spelling Correction/Reformatting"}
                      </button>
                      <button
                        type="button"
                        onClick={() => void getJournalFeedback()}
                        disabled={!reviewText.trim() || gettingFeedback || journalCleanupBusy}
                        className="rounded-full border border-white/25 bg-white/10 px-4 py-1.5 text-xs font-medium text-white shadow-sm transition hover:bg-white/15 disabled:opacity-50"
                      >
                        {gettingFeedback ? "Getting feedback…" : "Get feedback"}
                      </button>
                      <div className="flex items-center gap-1.5">
                        <label className="text-[0.65rem] text-white/45" htmlFor={`${idPrefix}-fbmodel`}>Model</label>
                        <select
                          id={`${idPrefix}-fbmodel`}
                          value={feedbackModel}
                          onChange={(e) => setFeedbackModel(e.target.value)}
                          className="rounded-full border border-white/15 bg-black/30 px-2 py-1 text-[0.65rem] text-white focus:border-white/25 focus:outline-none"
                        >
                          <option value="openai/gpt-5.4">gpt-5.4</option>
                          <option value="anthropic/claude-sonnet-4.6">claude-sonnet-4.6</option>
                          <option value="anthropic/claude-opus-4.6">claude-opus-4.6</option>
                        </select>
                      </div>
                    </div>
                  </div>
              </div>
          )}

          {chatInteractionMode === "autobiography" && finalizerOpen && (
            <div className="relative z-10 mx-auto w-full max-w-[48rem] px-3 py-6 md:px-6 md:py-8">
              {errorBanner}

              <div className="glass-panel rounded-2xl border border-white/10 px-4 py-4 md:px-5">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-white/50">
                      AI-Assisted Journal
                    </p>
                    <p className="mt-1 text-sm text-white/70">
                      Polish your reflection, then save.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={finalizerDismiss}
                    className="rounded-lg px-2 py-1 text-[0.65rem] font-medium text-white/50 hover:bg-white/10 hover:text-white"
                  >
                    Cancel
                  </button>
                </div>

                <div className="mt-3 flex flex-wrap items-end gap-3">
                  <div className="flex flex-wrap items-end gap-2">
                    <div className="flex flex-col gap-0.5">
                      <label className="text-[0.6rem] text-white/45" htmlFor={`${idPrefix}-fdate`}>
                        Date
                      </label>
                      <input
                        id={`${idPrefix}-fdate`}
                        type="date"
                        value={finalizerDate}
                        onChange={(e) => setFinalizerDate(e.target.value)}
                        className="rounded-lg border border-white/15 bg-black/30 px-2 py-1 text-xs text-white focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                      />
                    </div>
                    <div className="flex flex-col gap-0.5">
                      <label className="text-[0.6rem] text-white/45" htmlFor={`${idPrefix}-ftime`}>
                        Time
                      </label>
                      <input
                        id={`${idPrefix}-ftime`}
                        type="time"
                        value={finalizerTime}
                        onChange={(e) => setFinalizerTime(e.target.value)}
                        className="rounded-lg border border-white/15 bg-black/30 px-2 py-1 text-xs text-white focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                      />
                    </div>
                  </div>
                </div>

                <textarea
                  value={finalizerText}
                  onChange={(e) => setFinalizerText(e.target.value)}
                  rows={12}
                  placeholder="Your AI-assisted reflection…"
                  className="mt-3 w-full resize-y rounded-xl border border-white/15 bg-black/25 px-3 py-3 text-sm leading-relaxed text-white placeholder:text-white/40 focus:border-white/25 focus:outline-none focus:ring-2 focus:ring-white/10"
                  style={{ minHeight: "12rem" }}
                />

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void finalizerSave()}
                    disabled={
                      finalizerSaving ||
                      finalizerCleanupBusy ||
                      !finalizerDate.trim() ||
                      !finalizerTime.trim() ||
                      (!finalizerText.trim() && finalizerMessages.length === 0)
                    }
                    className="rounded-full bg-white px-4 py-1.5 text-xs font-medium text-gray-900 shadow-sm transition hover:bg-white/90 disabled:opacity-50"
                  >
                    {finalizerSaving ? "Saving…" : "Save to journal"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void finalizerCleanup()}
                    disabled={!finalizerText.trim() || finalizerCleanupBusy || finalizerSaving}
                    className="rounded-full border border-white/25 bg-white/10 px-4 py-1.5 text-xs font-medium text-white shadow-sm transition hover:bg-white/15 disabled:opacity-50"
                  >
                    {finalizerCleanupBusy ? "Applying…" : "AI Spelling Correction/Reformatting"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {chatInteractionMode === "autobiography" && !finalizerOpen && assistedHero && (
            <div className="flex min-h-[min(55vh,560px)] flex-1 flex-col items-center justify-center gap-6 px-4 py-8 text-center sm:min-h-[50vh]">
              <h1 className="max-w-lg text-[1.65rem] font-normal leading-snug tracking-tight text-white sm:text-3xl md:text-[1.75rem]">
                What can I help with?
              </h1>
              <p className="max-w-md animate-hero-mode-desc text-sm leading-relaxed text-white/55 md:text-[0.95rem]">
                {CHAT_INTERACTION_MODE_META.autobiography.description}
              </p>
              <div className="w-full max-w-2xl space-y-1.5 text-left">
                <LiveDictationBubble />
                <AskAnythingComposer layout="center" assistedJournalMinimal />
              </div>
            </div>
          )}

          {chatInteractionMode === "autobiography" && !finalizerOpen && !assistedHero && (
            <div className="relative z-10 mx-auto w-full max-w-[48rem] px-3 py-8 md:px-6">
              {errorBanner}

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
                                            {" "}· <span className="font-medium text-white/85">{a.brainSection}</span>
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
                            disabled={readAloudBusy}
                            aria-busy={readAloudBusy}
                            className="rounded-lg p-2 hover:bg-white/10 hover:text-white disabled:pointer-events-none disabled:opacity-35"
                            title={readAloudBusy ? "Loading audio…" : "Read aloud"}
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

        {showAssistedBottomComposer && (
          <div className="flex-none border-t border-white/10 bg-[#0a0a12]/92 px-3 py-2.5 pb-[max(0.65rem,env(safe-area-inset-bottom))] backdrop-blur-md">
            <div className="mx-auto w-full max-w-[48rem] space-y-2">
              {messages.length > 0 && (
                <button
                  type="button"
                  onClick={handoffAssistedToJournal}
                  disabled={sending}
                  className="w-full rounded-2xl bg-[#10a37f] px-4 py-3.5 text-center text-sm font-semibold tracking-tight text-white shadow-lg shadow-[#10a37f]/20 transition hover:bg-[#0d8c6e] disabled:pointer-events-none disabled:opacity-45"
                >
                  Save to Journal
                </button>
              )}
              <LiveDictationBubble />
              <AskAnythingComposer layout="center" assistedJournalMinimal />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
