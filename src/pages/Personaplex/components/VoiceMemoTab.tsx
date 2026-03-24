import { FC, useCallback, useEffect, useId, useRef, useState } from "react";
import { backendFetch } from "../../../backendApi";

type Props = {
  onToast: (msg: string) => void;
  /** Opens the full voice session panel (ChatGPT-style “voice mode” control). */
  onOpenSessionPanel?: () => void;
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

const CHAT_TIMEOUT_MS = 90_000;

function WaveformIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="currentColor" aria-hidden>
      <rect x="5" y="10" width="3" height="8" rx="1" />
      <rect x="10.5" y="6" width="3" height="16" rx="1" />
      <rect x="16" y="8" width="3" height="12" rx="1" />
    </svg>
  );
}

export const VoiceMemoTab: FC<Props> = ({ onToast, onOpenSessionPanel }) => {
  const idPrefix = useId();
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [micPhase, setMicPhase] = useState<"idle" | "recording" | "processing">("idle");
  const [error, setError] = useState<string | null>(null);
  /** After first text send: hero fades, composer stays at bottom, sidebar + thread layout. */
  const [isChatActive, setIsChatActive] = useState(false);
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
      const res = await backendFetch("/voice-memo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: b64, filename: "dictation.webm", mime_type: mimeType }),
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
        setDraft((prev) => (prev ? `${prev.trim()}\n${line}` : line));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Transcription failed");
    } finally {
      setMicPhase("idle");
    }
  }, []);

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
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        mediaRecorderRef.current = null;
        void processMicAudio(blob, blob.type || "audio/webm");
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

  const composerDisabled = sending || micPhase !== "idle";

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
            className="relative min-h-0 flex-1 overflow-y-auto"
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
                <p className="mt-10 max-w-md text-sm leading-relaxed text-white/60">
                  SelfMeridian is your private space to reflect, grow, and cultivate a deeper connection with yourself.
                </p>
              </div>
            </div>

            {hasConversation && (
              <div className="relative z-10 mx-auto w-full max-w-[48rem] px-3 py-8 md:px-6">
                {error && (
                  <div className="glass-panel mb-6 rounded-2xl px-4 py-3 text-sm text-red-200">
                    {error}
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
            className={`flex flex-none bg-transparent px-3 pb-5 pt-2 transition-[padding] duration-500 ease-out md:px-4 ${
              isChatActive ? "border-t border-white/[0.07]" : ""
            }`}
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
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
