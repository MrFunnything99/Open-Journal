import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
  type ReactNode,
} from "react";

/** Allowlisted client action from journal /chat (backend filters unknown types). */
export type PersonaplexNavigateAction = {
  type: "navigate";
  view: "voice_memo" | "journal" | "brain";
  brainSection?: "knowledgeBase";
};

function parseNavigateActions(raw: unknown): PersonaplexNavigateAction[] {
  if (!Array.isArray(raw)) return [];
  const out: PersonaplexNavigateAction[] = [];
  const views = new Set<string>(["voice_memo", "journal", "brain"]);
  const brains = new Set<string>(["knowledgeBase"]);
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
  readStoredUserChatModel,
  USER_CHAT_MODEL_STORAGE_KEY,
  type UserSelectableChatModelId,
} from "./chatCompletionModels";
import { blobToBase64, micBlobToTranscriptionPayload } from "./utils/audioToWav";
import {
  attachDictationLevelMonitor,
  DICTATION_MIC_CONSTRAINTS,
  shouldDiscardDictationRecording,
  type DictationLevelMonitor,
} from "./utils/dictationRecordingMonitor";
import { stopReadAloudAndCooldown } from "./utils/chatReadAloud";
import { transcriptLikelyEchoesAssistantText } from "./utils/transcriptionEchoGuard";

const CHAT_TIMEOUT_MS = 190_000;
/** Sent for API compatibility; server always uses full memory retrieval (1.0) for graph /chat. */
const CHAT_PERSONALIZATION_FULL = 1;

/** Local time + daypart for Assisted Journal (/chat autobiography) secondary check-ins. */
function buildAssistedJournalClientTimeContext(): string {
  const now = new Date();
  const h = now.getHours();
  const daypart =
    h >= 5 && h < 12 ? "morning" : h >= 12 && h < 17 ? "afternoon" : h >= 17 && h < 21 ? "evening" : "night";
  const local = new Intl.DateTimeFormat(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(now);
  return `User local: ${local}. For optional secondary check-ins, treat this as ${daypart}.`;
}

export type PersonaplexChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  retrievalLog?: string;
  agentSteps?: Array<{ kind?: string; name?: string; summary?: string }>;
  actions?: PersonaplexNavigateAction[];
};

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
  newChat: () => void;
  /** Clear assisted /chat workspace (after handoff to journal or intentional new reflection). */
  resetAssistedWorkspace: () => void;
  chatInteractionMode: ChatInteractionMode;
  setChatInteractionMode: (m: ChatInteractionMode) => void;
  /** Tinfoil model for AI-Assisted Journal Mode /chat (allowlisted on server). */
  userChatModel: UserSelectableChatModelId;
  setUserChatModel: (m: UserSelectableChatModelId) => void;
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
  chatWorkspaceResetRef,
}: {
  children: ReactNode;
  onToast: (msg: string) => void;
  /** Runs after a successful /chat when the server returned allowlisted UI actions (e.g. navigate). */
  onAgentAction?: (actions: PersonaplexNavigateAction[]) => void;
  /** Parent can call ref.current() to clear in-memory chat / assisted journal workspace (e.g. Start fresh). */
  chatWorkspaceResetRef?: MutableRefObject<(() => void) | null>;
}) {
  const idPrefix = useId();
  const onAgentActionRef = useRef(onAgentAction);
  onAgentActionRef.current = onAgentAction;
  const [messages, setMessages] = useState<PersonaplexChatMessage[]>([]);
  const messagesRef = useRef<PersonaplexChatMessage[]>([]);
  messagesRef.current = messages;
  const [draft, setDraft] = useState("");
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [micPhase, setMicPhase] = useState<"idle" | "recording" | "processing">("idle");
  const [liveDictationText, setLiveDictationText] = useState("");
  const [chatError, setChatError] = useState<string | null>(null);
  const [isChatActive, setIsChatActive] = useState(false);
  const [chatInteractionMode, setChatInteractionMode] = useState<ChatInteractionMode>("autobiography");
  const [userChatModel, setUserChatModel] = useState<UserSelectableChatModelId>(() => readStoredUserChatModel());

  useEffect(() => {
    try {
      localStorage.setItem(USER_CHAT_MODEL_STORAGE_KEY, userChatModel);
    } catch {
      /* ignore */
    }
  }, [userChatModel]);

  const [pendingAudioFile, setPendingAudioFile] = useState<File | null>(null);
  const [journalFileToProcess, setJournalFileToProcess] = useState<File | null>(null);
  const [journalTextToProcess, setJournalTextToProcess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const dictationMonitorRef = useRef<DictationLevelMonitor | null>(null);
  const recordStartRef = useRef(0);
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
          const mic = await micBlobToTranscriptionPayload(blob);
          b64 = mic.b64;
          filename = mic.filename;
          mimeType = mic.mimeType;
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
        const lastAsst = [...messagesRef.current].reverse().find((m) => m.role === "assistant");
        if (
          lastAsst?.content &&
          (transcriptLikelyEchoesAssistantText(line, lastAsst.content) ||
            transcriptLikelyEchoesAssistantText(raw, lastAsst.content))
        ) {
          onToast(
            "Transcription matched the assistant's last reply instead of your voice — try again with speakers lower or after read-aloud stops."
          );
          return null;
        }
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
      setSending(true);
      const t = await transcribeBlob(audioFile);
      if (t) {
        text = text ? `${text}\n\n${t.polished}` : t.polished;
      } else {
        setSending(false);
        return;
      }
    }

    if (!text) { setSending(false); return; }

    setMessages((m) => [...m, { id: `u_${Date.now()}`, role: "user", content: text }]);
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
        payload.tinfoil_model = userChatModel;
        payload.client_time_context = buildAssistedJournalClientTimeContext();
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
        retrieval_log?: string;
        agent_steps?: Array<{ kind?: string; name?: string; summary?: string }>;
        actions?: unknown[];
        library_items_added?: number;
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

      const libN =
        typeof data.library_items_added === "number" && data.library_items_added > 0
          ? Math.floor(data.library_items_added)
          : 0;
      if (libN > 0) {
        onToast(libN === 1 ? "Added 1 item to your Library." : `Added ${libN} items to your Library.`);
      }

      if (navActions.length > 0) {
        onAgentActionRef.current?.(navActions);
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
    } finally {
      setSending(false);
    }
  }, [
    draft,
    sending,
    chatSessionId,
    onToast,
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
        tinfoil_model: userChatModel,
        client_time_context: buildAssistedJournalClientTimeContext(),
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
        retrieval_log?: string;
        agent_steps?: Array<{ kind?: string; name?: string; summary?: string }>;
        actions?: unknown[];
        library_items_added?: number;
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

      const libN =
        typeof data.library_items_added === "number" && data.library_items_added > 0
          ? Math.floor(data.library_items_added)
          : 0;
      if (libN > 0) {
        onToast(libN === 1 ? "Added 1 item to your Library." : `Added ${libN} items to your Library.`);
      }

      if (navActions.length > 0) {
        onAgentActionRef.current?.(navActions);
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
    } finally {
      setSending(false);
    }
  }, [sending, chatSessionId, onToast, userChatModel]);

  const startRecording = useCallback(async () => {
    if (micPhase !== "idle" || sending) return;
    setChatError(null);
    await stopReadAloudAndCooldown();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: DICTATION_MIC_CONSTRAINTS });
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
        const mon = dictationMonitorRef.current;
        dictationMonitorRef.current = null;
        const peakRms = mon ? mon.getMaxRms() : 1;
        mon?.stop();
        const elapsed = Date.now() - recordStartRef.current;
        if (shouldDiscardDictationRecording(elapsed, peakRms)) {
          setMicPhase("idle");
          setLiveDictationText("");
          return;
        }
        void (async () => {
          const t = await transcribeBlob(blob);
          if (t) {
            setDraft((d) => (d.trim() ? `${d.trim()}\n\n${t.polished}` : t.polished));
          }
        })();
      };
      mr.start();
      recordStartRef.current = Date.now();
      dictationMonitorRef.current = attachDictationLevelMonitor(stream);
      mediaRecorderRef.current = mr;
      setMicPhase("recording");
      beginLiveSpeechCaption();
    } catch (e) {
      stopSpeechRecognition();
      setLiveDictationText("");
      setChatError(e instanceof Error ? e.message : "Microphone unavailable");
      setMicPhase("idle");
    }
  }, [micPhase, sending, transcribeBlob, beginLiveSpeechCaption, stopSpeechRecognition]);

  const stopRecording = useCallback(() => {
    stopSpeechRecognition();
    setLiveDictationText("");
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== "inactive") mr.stop();
    else {
      dictationMonitorRef.current?.stop();
      dictationMonitorRef.current = null;
      setMicPhase("idle");
    }
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

  const composerDisabled = sending || micPhase !== "idle";

  const newChat = useCallback(() => {
    setChatSessionId(null);
    setMessages([]);
    setDraft("");
    setLiveDictationText("");
    setIsChatActive(false);
    setChatError(null);
  }, []);

  const resetAssistedWorkspace = useCallback(() => {
    setChatSessionId(null);
    setMessages([]);
    setDraft("");
    setLiveDictationText("");
    setIsChatActive(false);
    setChatError(null);
    setPendingAudioFile(null);
  }, []);

  const resetChatWorkspaceForStartFresh = useCallback(() => {
    resetAssistedWorkspace();
    setJournalFileToProcess(null);
    setJournalTextToProcess(null);
    setMicPhase("idle");
    stopSpeechRecognition();
  }, [resetAssistedWorkspace, stopSpeechRecognition]);

  useEffect(() => {
    if (!chatWorkspaceResetRef) return;
    chatWorkspaceResetRef.current = resetChatWorkspaceForStartFresh;
    return () => {
      chatWorkspaceResetRef.current = null;
    };
  }, [chatWorkspaceResetRef, resetChatWorkspaceForStartFresh]);

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
      newChat,
      resetAssistedWorkspace,
      chatInteractionMode,
      userChatModel,
      setUserChatModel,
    ]
  );

  return <PersonaplexChatContext.Provider value={value}>{children}</PersonaplexChatContext.Provider>;
}
