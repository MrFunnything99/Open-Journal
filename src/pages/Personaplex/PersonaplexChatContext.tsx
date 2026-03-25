import {
  createContext,
  useCallback,
  useContext,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

/** Allowlisted client action from journal /chat (backend filters unknown types). */
export type PersonaplexNavigateAction = {
  type: "navigate";
  view: "voice_memo" | "journal" | "brain" | "recommendations";
  brainSection?: "knowledgeBase" | "calendar";
};

function parseNavigateActions(raw: unknown): PersonaplexNavigateAction[] {
  if (!Array.isArray(raw)) return [];
  const out: PersonaplexNavigateAction[] = [];
  const views = new Set<string>(["voice_memo", "journal", "brain", "recommendations"]);
  const brains = new Set<string>(["knowledgeBase", "calendar"]);
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const o = item as Record<string, unknown>;
    if (o.type !== "navigate") continue;
    const view = o.view;
    if (typeof view !== "string" || !views.has(view)) continue;
    const nav: PersonaplexNavigateAction = {
      type: "navigate",
      view: view as PersonaplexNavigateAction["view"],
    };
    if (o.brainSection != null && typeof o.brainSection === "string" && brains.has(o.brainSection)) {
      nav.brainSection = o.brainSection as PersonaplexNavigateAction["brainSection"];
    }
    out.push(nav);
  }
  return out;
}
import { backendFetch } from "../../backendApi";

const CHAT_TIMEOUT_MS = 90_000;

export type PersonaplexChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  retrievalLog?: string;
  agentSteps?: Array<{ kind?: string; name?: string; summary?: string }>;
  actions?: PersonaplexNavigateAction[];
};

export type AgentActivityEntry = {
  id: string;
  ts: number;
  kind: "user" | "assistant" | "retrieval" | "tool" | "system";
  summary: string;
  detail?: string;
};

export type ChatRecentEntry = {
  sessionId: string;
  title: string;
  updatedAt: number;
};

const RECENTS_STORAGE_KEY = "personaplex-chat-recents";
const MAX_RECENTS = 14;
const MAX_ACTIVITY = 60;

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

export function defaultFilenameForMicBlob(mimeType: string): string {
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
  if (typeof navigator !== "undefined" && navigator.vendor === "Apple Computer, Inc.") {
    return "audio/mp4";
  }
  return "audio/webm";
}

function readRecentsFromStorage(): ChatRecentEntry[] {
  try {
    const raw = localStorage.getItem(RECENTS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (x): x is ChatRecentEntry =>
          x &&
          typeof x === "object" &&
          typeof (x as ChatRecentEntry).sessionId === "string" &&
          typeof (x as ChatRecentEntry).title === "string" &&
          typeof (x as ChatRecentEntry).updatedAt === "number"
      )
      .slice(0, MAX_RECENTS);
  } catch {
    return [];
  }
}

function writeRecentsToStorage(entries: ChatRecentEntry[]) {
  try {
    localStorage.setItem(RECENTS_STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_RECENTS)));
  } catch {
    /* ignore */
  }
}

/** Minimal typing for browser Speech Recognition (Chrome / Safari). */
type SpeechRecInstance = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((this: SpeechRecInstance, ev: Event) => void) | null;
  onerror: ((this: SpeechRecInstance, ev: Event) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecResult = { isFinal: boolean; 0: { transcript: string } };
type SpeechRecEvent = Event & { resultIndex: number; results: ArrayLike<SpeechRecResult> & { length: number } };

function getSpeechRecognitionCtor(): (new () => SpeechRecInstance) | undefined {
  if (typeof window === "undefined") return undefined;
  const w = window as unknown as {
    SpeechRecognition?: new () => SpeechRecInstance;
    webkitSpeechRecognition?: new () => SpeechRecInstance;
  };
  return w.SpeechRecognition || w.webkitSpeechRecognition;
}

type PersonaplexChatContextValue = {
  idPrefix: string;
  messages: PersonaplexChatMessage[];
  draft: string;
  setDraft: (v: string | ((p: string) => string)) => void;
  chatSessionId: string | null;
  setChatSessionId: (v: string | null) => void;
  sending: boolean;
  micPhase: "idle" | "recording" | "processing";
  /** Live caption while dictating (browser speech recognition when available). */
  liveDictationText: string;
  chatError: string | null;
  setChatError: (v: string | null) => void;
  isChatActive: boolean;
  sendChat: () => Promise<void>;
  transcribeBlob: (blob: Blob, mimeType: string) => Promise<{ polished: string; raw: string } | null>;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  fileInputRef: React.MutableRefObject<HTMLInputElement | null>;
  onPickFile: (e: React.ChangeEvent<HTMLInputElement>) => void;
  composerDisabled: boolean;
  activityLog: AgentActivityEntry[];
  clearActivityLog: () => void;
  chatRecents: ChatRecentEntry[];
  loadRecentSession: (sessionId: string) => void;
  newChat: () => void;
};

const PersonaplexChatContext = createContext<PersonaplexChatContextValue | null>(null);

export function usePersonaplexChat(): PersonaplexChatContextValue {
  const ctx = useContext(PersonaplexChatContext);
  if (!ctx) {
    throw new Error("usePersonaplexChat must be used within PersonaplexChatProvider");
  }
  return ctx;
}

export function PersonaplexChatProvider({
  children,
  onToast,
  onAgentAction,
}: {
  children: ReactNode;
  onToast: (msg: string) => void;
  /** Runs after a successful /chat when the server returned allowlisted UI actions (e.g. navigate). */
  onAgentAction?: (actions: PersonaplexNavigateAction[]) => void;
}) {
  const idPrefix = useId();
  const onAgentActionRef = useRef(onAgentAction);
  onAgentActionRef.current = onAgentAction;
  const [messages, setMessages] = useState<PersonaplexChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [micPhase, setMicPhase] = useState<"idle" | "recording" | "processing">("idle");
  const [liveDictationText, setLiveDictationText] = useState("");
  const [chatError, setChatError] = useState<string | null>(null);
  const [isChatActive, setIsChatActive] = useState(false);
  const [activityLog, setActivityLog] = useState<AgentActivityEntry[]>([]);
  const [chatRecents, setChatRecents] = useState<ChatRecentEntry[]>(() =>
    typeof window !== "undefined" ? readRecentsFromStorage() : []
  );

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const speechRecRef = useRef<SpeechRecInstance | null>(null);
  const speechSessionActiveRef = useRef(false);
  const speechFinalRef = useRef("");

  const stopSpeechRecognition = useCallback(() => {
    speechSessionActiveRef.current = false;
    const rec = speechRecRef.current;
    speechRecRef.current = null;
    if (!rec) return;
    rec.onresult = null;
    rec.onerror = null;
    rec.onend = null;
    try {
      rec.stop();
    } catch {
      /* ignore */
    }
  }, []);

  const beginLiveSpeechCaption = useCallback(() => {
    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) return;
    speechSessionActiveRef.current = true;
    speechFinalRef.current = "";
    setLiveDictationText("");

    const wire = (rec: SpeechRecInstance) => {
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = typeof navigator !== "undefined" && navigator.language ? navigator.language : "en-US";
      rec.onresult = (ev: Event) => {
        const event = ev as SpeechRecEvent;
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          const piece = result[0]?.transcript ?? "";
          if (result.isFinal) speechFinalRef.current += piece;
          else interim += piece;
        }
        setLiveDictationText((speechFinalRef.current + interim).trim());
      };
      rec.onerror = () => {
        /* best-effort caption; MediaRecorder is canonical */
      };
      rec.onend = () => {
        if (!speechSessionActiveRef.current) return;
        if (mediaRecorderRef.current?.state !== "recording") return;
        speechRecRef.current = null;
        try {
          const next = new Ctor();
          speechRecRef.current = next;
          wire(next);
          next.start();
        } catch {
          /* ignore */
        }
      };
    };

    try {
      const rec = new Ctor();
      speechRecRef.current = rec;
      wire(rec);
      rec.start();
    } catch {
      speechRecRef.current = null;
    }
  }, []);

  const pushActivity = useCallback((entry: Omit<AgentActivityEntry, "id" | "ts">) => {
    const full: AgentActivityEntry = {
      ...entry,
      id: `a_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      ts: Date.now(),
    };
    setActivityLog((prev) => [...prev.slice(-(MAX_ACTIVITY - 1)), full]);
  }, []);

  const bumpRecents = useCallback((sessionId: string, userLine: string) => {
    const title = userLine.trim().slice(0, 56) || "Chat";
    const now = Date.now();
    setChatRecents((prev) => {
      const next = [{ sessionId, title, updatedAt: now }, ...prev.filter((r) => r.sessionId !== sessionId)];
      writeRecentsToStorage(next);
      return next.slice(0, MAX_RECENTS);
    });
  }, []);

  const transcribeBlob = useCallback(
    async (blob: Blob, mimeType: string) => {
      setMicPhase("processing");
      setChatError(null);
      try {
        const b64 = await blobToBase64(blob);
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
        const raw = ((data.raw_transcript ?? "").trim() || line).trim();
        if (!line) return null;
        return { polished: line, raw: raw || line };
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Transcription failed";
        setChatError(msg);
        onToast(msg);
        return null;
      } finally {
        setMicPhase("idle");
        setLiveDictationText("");
      }
    },
    [onToast]
  );

  const sendChat = useCallback(async () => {
    const text = draft.trim();
    if (!text || sending) return;
    setIsChatActive(true);
    setChatError(null);
    setDraft("");
    setMessages((m) => [...m, { id: `u_${Date.now()}`, role: "user", content: text }]);
    pushActivity({ kind: "user", summary: `You: ${text.length > 120 ? `${text.slice(0, 120)}…` : text}` });
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
      const rawText = await res.text();
      let data: {
        error?: string;
        detail?: string;
        response?: string;
        session_id?: string;
        library_items_added?: number;
        retrieval_log?: string;
        agent_steps?: Array<{ kind?: string; name?: string; summary?: string }>;
        actions?: unknown[];
      } = {};
      if (rawText.trim()) {
        try {
          data = JSON.parse(rawText) as typeof data;
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
      bumpRecents(data.session_id, text);

      const steps = Array.isArray(data.agent_steps) ? data.agent_steps : [];
      for (const s of steps) {
        const kind = (s.kind || "").toLowerCase();
        const summary = typeof s.summary === "string" ? s.summary : "";
        if (!summary) continue;
        if (kind === "retrieval") {
          pushActivity({ kind: "retrieval", summary });
        } else if (kind === "tool") {
          const name = typeof s.name === "string" ? s.name : "tool";
          pushActivity({ kind: "tool", summary, detail: name });
        } else {
          pushActivity({ kind: "system", summary });
        }
      }

      const retrievalLog =
        typeof data.retrieval_log === "string" && data.retrieval_log.trim() ? data.retrieval_log.trim() : undefined;

      const navActions = parseNavigateActions(data.actions);
      setMessages((m) => [
        ...m,
        {
          id: `a_${Date.now()}`,
          role: "assistant",
          content: data.response!,
          retrievalLog,
          agentSteps: steps.length > 0 ? steps : undefined,
          actions: navActions.length > 0 ? navActions : undefined,
        },
      ]);

      const replyPreview =
        data.response!.length > 160 ? `${data.response!.slice(0, 160)}…` : data.response!;
      pushActivity({ kind: "assistant", summary: `Assistant: ${replyPreview}` });

      const libN =
        typeof data.library_items_added === "number" && data.library_items_added > 0
          ? Math.floor(data.library_items_added)
          : 0;
      if (libN > 0) {
        onToast(libN === 1 ? "Added 1 item to your Library." : `Added ${libN} items to your Library.`);
      }

      if (navActions.length > 0) {
        onAgentActionRef.current?.(navActions);
        for (const na of navActions) {
          if (na.type === "navigate") {
            pushActivity({
              kind: "system",
              summary: `Opened ${na.view}${na.brainSection ? ` · ${na.brainSection}` : ""}`,
            });
          }
        }
      }
    } catch (e) {
      const msg =
        e instanceof Error && e.name === "AbortError"
          ? "Request timed out. Try again."
          : e instanceof Error
            ? e.message
            : "Something went wrong";
      setChatError(msg);
      onToast(msg);
      pushActivity({ kind: "system", summary: `Error: ${msg}` });
    } finally {
      setSending(false);
    }
  }, [draft, sending, chatSessionId, onToast, pushActivity, bumpRecents]);

  const startRecording = useCallback(async () => {
    if (micPhase !== "idle" || sending) return;
    setChatError(null);
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
        const mimeDone = effectiveRecorderMime(mr);
        const blob = new Blob(chunksRef.current, { type: mimeDone });
        mediaRecorderRef.current = null;
        void (async () => {
          const t = await transcribeBlob(blob, mimeDone);
          if (t) {
            setIsChatActive(true);
            setDraft((d) => (d.trim() ? `${d.trim()}\n\n${t.polished}` : t.polished));
            pushActivity({ kind: "system", summary: "Transcribed voice into your message — edit or send." });
          }
        })();
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setMicPhase("recording");
      beginLiveSpeechCaption();
    } catch (e) {
      stopSpeechRecognition();
      setLiveDictationText("");
      setChatError(e instanceof Error ? e.message : "Microphone unavailable");
      setMicPhase("idle");
    }
  }, [micPhase, sending, transcribeBlob, pushActivity, beginLiveSpeechCaption, stopSpeechRecognition]);

  const stopRecording = useCallback(() => {
    stopSpeechRecognition();
    setLiveDictationText("");
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== "inactive") mr.stop();
    else setMicPhase("idle");
  }, [stopSpeechRecognition]);

  const onPickFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (!f) return;
      setChatError(null);
      void (async () => {
        const t = await transcribeBlob(f, f.type || "application/octet-stream");
        if (t) {
          setIsChatActive(true);
          setDraft((d) => (d.trim() ? `${d.trim()}\n\n${t.polished}` : t.polished));
          pushActivity({ kind: "system", summary: "Transcribed audio file into your message." });
        }
      })();
      e.target.value = "";
    },
    [transcribeBlob, pushActivity]
  );

  const composerDisabled = sending || micPhase !== "idle";

  const clearActivityLog = useCallback(() => setActivityLog([]), []);

  const loadRecentSession = useCallback(
    (sessionId: string) => {
      setChatSessionId(sessionId);
      setMessages([]);
      setIsChatActive(false);
      setDraft("");
      pushActivity({ kind: "system", summary: "Switched to a saved chat session." });
      onToast("Session loaded — continue the conversation.");
    },
    [onToast, pushActivity]
  );

  const newChat = useCallback(() => {
    setChatSessionId(null);
    setMessages([]);
    setDraft("");
    setLiveDictationText("");
    setIsChatActive(false);
    setChatError(null);
    pushActivity({ kind: "system", summary: "Started a new chat." });
  }, [pushActivity]);

  const value = useMemo(
    () => ({
      idPrefix,
      messages,
      draft,
      setDraft,
      chatSessionId,
      setChatSessionId,
      sending,
      micPhase,
      liveDictationText,
      chatError,
      setChatError,
      isChatActive,
      sendChat,
      transcribeBlob,
      startRecording,
      stopRecording,
      fileInputRef,
      onPickFile,
      composerDisabled,
      activityLog,
      clearActivityLog,
      chatRecents,
      loadRecentSession,
      newChat,
    }),
    [
      idPrefix,
      messages,
      draft,
      chatSessionId,
      sending,
      micPhase,
      liveDictationText,
      chatError,
      isChatActive,
      sendChat,
      transcribeBlob,
      startRecording,
      stopRecording,
      composerDisabled,
      activityLog,
      chatRecents,
      loadRecentSession,
      newChat,
      clearActivityLog,
    ]
  );

  return <PersonaplexChatContext.Provider value={value}>{children}</PersonaplexChatContext.Provider>;
}
