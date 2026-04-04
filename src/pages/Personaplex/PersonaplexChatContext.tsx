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
import type { ChatMessage } from "./hooks/useJournalHistory";
import type { ChatInteractionMode } from "./chatInteractionModes";
import {
  DEFAULT_USER_CHAT_MODEL,
  readStoredUserChatModel,
  type UserSelectableChatModelId,
  USER_CHAT_MODEL_STORAGE_KEY,
} from "./chatCompletionModels";
import { blobToBase64, blobToWavBase64 } from "./utils/audioToWav";

const CHAT_TIMEOUT_MS = 190_000;
/** Sent for API compatibility; server always uses full memory retrieval (1.0) for graph /chat. */
const CHAT_PERSONALIZATION_FULL = 1;

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

const MAX_ACTIVITY = 60;

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

/** Map Personaplex /chat messages to journal transcript format (AI-Assisted Journal handoff / tab-close save). */
export function personaplexChatToJournalTranscript(messages: PersonaplexChatMessage[]): ChatMessage[] {
  return messages.map((m) => ({
    role: m.role === "user" ? "user" : "ai",
    text: m.content,
  }));
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
  /** Send arbitrary text directly (bypasses draft state). Used by voice mode. */
  sendChatWithText: (text: string) => Promise<void>;
  transcribeBlob: (blob: Blob, mimeType: string) => Promise<{ polished: string; raw: string } | null>;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  fileInputRef: React.MutableRefObject<HTMLInputElement | null>;
  onPickFile: (e: React.ChangeEvent<HTMLInputElement>) => void;
  pendingAudioFile: File | null;
  clearPendingAudioFile: () => void;
  /** Set by sendChat when journal mode + pending file; VoiceMemoTab watches and processes via journal pipeline. */
  journalFileToProcess: File | null;
  clearJournalFileToProcess: () => void;
  /** Set by sendChat when journal mode + text only; VoiceMemoTab watches and populates the review editor. */
  journalTextToProcess: string | null;
  clearJournalTextToProcess: () => void;
  composerDisabled: boolean;
  activityLog: AgentActivityEntry[];
  clearActivityLog: () => void;
  newChat: () => void;
  /** Clear assisted /chat workspace (after handoff to journal or intentional new reflection). */
  resetAssistedWorkspace: () => void;
  chatInteractionMode: ChatInteractionMode;
  setChatInteractionMode: (m: ChatInteractionMode) => void;
  /** OpenRouter model for AI-Assisted Journal Mode only (/chat with tools + model picker). */
  userChatModel: UserSelectableChatModelId;
  setUserChatModel: (id: UserSelectableChatModelId) => void;
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
  const [chatInteractionMode, setChatInteractionMode] = useState<ChatInteractionMode>("journal");
  const [userChatModel, setUserChatModelState] = useState<UserSelectableChatModelId>(() =>
    typeof window !== "undefined" ? readStoredUserChatModel() : DEFAULT_USER_CHAT_MODEL
  );
  const [activityLog, setActivityLog] = useState<AgentActivityEntry[]>([]);

  const [pendingAudioFile, setPendingAudioFile] = useState<File | null>(null);
  const [journalFileToProcess, setJournalFileToProcess] = useState<File | null>(null);
  const [journalTextToProcess, setJournalTextToProcess] = useState<string | null>(null);
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

  const transcribeBlob = useCallback(
    async (blob: Blob) => {
      setMicPhase("processing");
      setChatError(null);
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
    let text = draft.trim();
    const audioFile = pendingAudioFile;

    if (!text && !audioFile) return;
    if (sending) return;

    // Journal mode + pending audio → hand off to VoiceMemoTab's journal pipeline
    if (chatInteractionMode === "journal" && audioFile) {
      setPendingAudioFile(null);
      setDraft("");
      setJournalFileToProcess(audioFile);
      return;
    }

    // Journal mode + text only → hand off to VoiceMemoTab's journal panel
    if (chatInteractionMode === "journal" && text && !audioFile) {
      setDraft("");
      setJournalTextToProcess(text);
      return;
    }

    setIsChatActive(true);
    setChatError(null);
    setDraft("");
    if (audioFile) setPendingAudioFile(null);

    if (audioFile) {
      pushActivity({ kind: "system", summary: `Transcribing attached audio: ${audioFile.name}` });
      setSending(true);
      const t = await transcribeBlob(audioFile);
      if (t) {
        text = text ? `${text}\n\n${t.polished}` : t.polished;
        pushActivity({ kind: "system", summary: "Transcribed audio file." });
      } else {
        setSending(false);
        return;
      }
    }

    if (!text) { setSending(false); return; }

    setMessages((m) => [...m, { id: `u_${Date.now()}`, role: "user", content: text }]);
    pushActivity({ kind: "user", summary: `You: ${text.length > 120 ? `${text.slice(0, 120)}…` : text}` });
    if (!audioFile) setSending(true);

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);

    try {
      const payload: Record<string, unknown> = {
        text,
        session_id: chatSessionId,
        personalization: CHAT_PERSONALIZATION_FULL,
        intrusiveness: 0.5,
        mode: chatInteractionMode,
      };
      if (chatInteractionMode === "autobiography") {
        payload.openrouter_model = userChatModel;
      }
      const res = await backendFetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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
  }, [
    draft,
    sending,
    chatSessionId,
    onToast,
    pushActivity,
    pendingAudioFile,
    transcribeBlob,
    chatInteractionMode,
    userChatModel,
  ]);

  const sendChatWithText = useCallback(async (text: string) => {
    const t = text.trim();
    if (!t || sending) return;

    setIsChatActive(true);
    setChatError(null);
    setMessages((m) => [...m, { id: `u_${Date.now()}`, role: "user", content: t }]);
    pushActivity({ kind: "user", summary: `You: ${t.length > 120 ? `${t.slice(0, 120)}…` : t}` });
    setSending(true);

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);

    try {
      const payload: Record<string, unknown> = {
        text: t,
        session_id: chatSessionId,
        personalization: CHAT_PERSONALIZATION_FULL,
        intrusiveness: 0.5,
        mode: "autobiography",
        openrouter_model: userChatModel,
      };
      const res = await backendFetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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
  }, [sending, chatSessionId, onToast, pushActivity, userChatModel]);

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
          const t = await transcribeBlob(blob);
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
      setPendingAudioFile(f);
      e.target.value = "";
    },
    []
  );

  const clearPendingAudioFile = useCallback(() => setPendingAudioFile(null), []);
  const clearJournalFileToProcess = useCallback(() => setJournalFileToProcess(null), []);
  const clearJournalTextToProcess = useCallback(() => setJournalTextToProcess(null), []);

  const setUserChatModel = useCallback((id: UserSelectableChatModelId) => {
    setUserChatModelState(id);
    try {
      localStorage.setItem(USER_CHAT_MODEL_STORAGE_KEY, id);
    } catch {
      /* ignore */
    }
  }, []);

  const composerDisabled = sending || micPhase !== "idle";

  const clearActivityLog = useCallback(() => setActivityLog([]), []);

  const newChat = useCallback(() => {
    setChatSessionId(null);
    setMessages([]);
    setDraft("");
    setLiveDictationText("");
    setIsChatActive(false);
    setChatError(null);
    pushActivity({ kind: "system", summary: "Started a new chat." });
  }, [pushActivity]);

  const resetAssistedWorkspace = useCallback(() => {
    setChatSessionId(null);
    setMessages([]);
    setDraft("");
    setLiveDictationText("");
    setIsChatActive(false);
    setChatError(null);
    setPendingAudioFile(null);
  }, []);

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
      sendChatWithText,
      transcribeBlob,
      startRecording,
      stopRecording,
      fileInputRef,
      onPickFile,
      pendingAudioFile,
      clearPendingAudioFile,
      journalFileToProcess,
      clearJournalFileToProcess,
      journalTextToProcess,
      clearJournalTextToProcess,
      composerDisabled,
      activityLog,
      clearActivityLog,
      newChat,
      resetAssistedWorkspace,
      chatInteractionMode,
      setChatInteractionMode,
      userChatModel,
      setUserChatModel,
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
      sendChatWithText,
      transcribeBlob,
      startRecording,
      stopRecording,
      composerDisabled,
      pendingAudioFile,
      journalFileToProcess,
      journalTextToProcess,
      activityLog,
      newChat,
      resetAssistedWorkspace,
      clearActivityLog,
      chatInteractionMode,
      userChatModel,
      setUserChatModel,
    ]
  );

  return <PersonaplexChatContext.Provider value={value}>{children}</PersonaplexChatContext.Provider>;
}
