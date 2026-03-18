import { useCallback, useRef, useState } from "react";
import { blobToWavBase64 } from "../utils/audioToWav";

export type PersonaplexConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

export type VoiceSettings = {
  stability?: number;
  similarity_boost?: number;
  style?: number;
  speed?: number;
};

export type SessionMode = "journal" | "recommendations" | "extreme" | "therapy";

export type UsePersonaplexSessionOptions = {
  systemPrompt: string;
  selectedVoiceId: string;
  manualMode?: boolean;
  personalization: number;
  intrusiveness?: number;
  /** "journal" = regular interview; "recommendations" = talk about your books, save notes */
  sessionMode?: SessionMode;
  voiceSettings?: VoiceSettings;
  onTranscriptUpdate: (updater: (prev: TranscriptEntry[]) => TranscriptEntry[]) => void;
  onInterimTranscript: (text: string) => void;
  /** Called when the agent saves a note in recommendations mode */
  onNotesSaved?: (notes: { item_id: string; note: string }[]) => void;
  /** If true, show live partial transcription while the user is speaking (desktop flow). If false, only show text after the turn is committed. */
  showLiveTranscription?: boolean;
};

import { backendFetch } from "../../../backendApi";

export type TranscriptEntry = { role: "user" | "ai"; text: string; retrievalLog?: string };

const CHAT_FETCH_TIMEOUT_MS = 90_000;

async function fetchInterviewerQuestion(
  text: string,
  sessionId: string | null,
  personalization: number,
  intrusiveness: number,
  mode: "journal" | "recommendations" | "extreme" | "therapy"
): Promise<{ question: string; sessionId: string; retrievalLog?: string; notesSaved?: { item_id: string; note: string }[] }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), CHAT_FETCH_TIMEOUT_MS);
  const res = await backendFetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      session_id: sessionId,
      personalization,
      intrusiveness,
      mode,
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
    retrieval_log?: string;
    notes_saved?: { item_id: string; note: string }[];
  } = {};
  if (rawText.trim()) {
    try {
      data = JSON.parse(rawText) as typeof data;
    } catch {
      const snippet = rawText.slice(0, 80).replace(/\s+/g, " ");
      throw new Error(
        res.status === 404
          ? "Backend not found. Make sure the Python backend is running."
          : res.ok
            ? "Invalid response from server"
            : `Server error (${res.status}): ${snippet || res.statusText}`
      );
    }
  }

  if (!res.ok) {
    throw new Error(data.detail || data.error || `Interviewer API failed (${res.status})`);
  }

  if (
    !data.response ||
    typeof data.response !== "string" ||
    !data.session_id ||
    typeof data.session_id !== "string"
  ) {
    throw new Error(data.error || "Invalid response from backend");
  }

  return {
    question: data.response,
    sessionId: data.session_id,
    retrievalLog: typeof data.retrieval_log === "string" ? data.retrieval_log : undefined,
    notesSaved: Array.isArray(data.notes_saved) ? data.notes_saved : undefined,
  };
}

async function fetchVoiceAudio(
  text: string,
  voiceId: string,
  voiceSettings?: VoiceSettings
): Promise<{ base64: string; format: string }> {
  const res = await fetch("/api/voice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voiceId, voice_settings: voiceSettings ?? undefined }),
  });

  const rawText = await res.text();
  let data: { error?: string; audio?: string; format?: string } = {};
  if (rawText.trim()) {
    try {
      data = JSON.parse(rawText) as {
        error?: string;
        audio?: string;
        format?: string;
      };
    } catch {
      const snippet = rawText.slice(0, 80).replace(/\s+/g, " ");
      throw new Error(
        res.status === 404
          ? "Voice API not found. Run 'npm run dev' (starts both Vite + API server). If port 3001 is in use, add API_PORT=3002 and VITE_API_URL=http://localhost:3002 to .env"
          : res.ok
            ? "Invalid response from Voice API"
            : `Voice API error (${res.status}): ${snippet || res.statusText}`
      );
    }
  }

  if (!res.ok) {
    throw new Error(data.error || `Voice API failed (${res.status})`);
  }

  if (!data.audio || typeof data.audio !== "string") {
    throw new Error(data.error || "No audio in response");
  }

  return { base64: data.audio, format: data.format ?? "mp3" };
}

async function fetchScribeToken(): Promise<string> {
  const res = await fetch("/api/scribe-token");
  const rawText = await res.text();
  let data: { error?: string; token?: string } = {};
  if (rawText.trim()) {
    try {
      data = JSON.parse(rawText) as { error?: string; token?: string };
    } catch {
      throw new Error(
        res.status === 404
          ? "Scribe token API not found. Run 'npm run dev'."
          : `Scribe token API error (${res.status})`
      );
    }
  }

  if (!res.ok) {
    throw new Error(data.error || `Scribe token API failed (${res.status})`);
  }

  if (!data.token || typeof data.token !== "string") {
    throw new Error(data.error || "No token in response");
  }

  return data.token;
}

async function fetchTranscribe(audioBase64: string): Promise<string> {
  const res = await fetch("/api/transcribe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio: audioBase64 }),
  });
  const rawText = await res.text();
  let data: { error?: string; text?: string } = {};
  if (rawText.trim()) {
    try {
      data = JSON.parse(rawText) as { error?: string; text?: string };
    } catch {
      throw new Error(res.ok ? "Invalid transcription response" : `Transcription failed (${res.status})`);
    }
  }
  if (!res.ok) {
    throw new Error(data.error ?? `Transcription failed (${res.status})`);
  }
  return (data.text ?? "").trim();
}

function float32ToPcmBase64(float32: Float32Array, targetRate?: number, sourceRate?: number): string {
  let samples = float32;
  if (targetRate && sourceRate && targetRate !== sourceRate) {
    const ratio = sourceRate / targetRate;
    const outLen = Math.floor(float32.length / ratio);
    const resampled = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, float32.length - 1);
      const frac = srcIdx - lo;
      resampled[i] = (float32[lo] ?? 0) * (1 - frac) + (float32[hi] ?? 0) * frac;
    }
    samples = resampled;
  }
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i] ?? 0));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}

const SCRIBE_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime";
const SCRIBE_MODEL = "scribe_v2_realtime";

function createSilentPcmBase64(numSamples: number): string {
  const pcm = new Int16Array(numSamples);
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}

export const usePersonaplexSession = ({
  systemPrompt: _systemPrompt,
  selectedVoiceId,
  manualMode = false,
  personalization,
  intrusiveness = 0.5,
  sessionMode = "journal",
  voiceSettings,
  onTranscriptUpdate,
  onInterimTranscript,
  onNotesSaved,
  showLiveTranscription = true,
}: UsePersonaplexSessionOptions) => {
  const [status, setStatus] = useState<PersonaplexConnectionStatus>("disconnected");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [isVoiceMemoRecording, setIsVoiceMemoRecording] = useState(false);
  const [lastPlaybackFailed, setLastPlaybackFailed] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const voiceMemoStreamRef = useRef<MediaStream | null>(null);
  const voiceMemoChunksRef = useRef<Blob[]>([]);
  const lastFailedPlaybackRef = useRef<{ blob: Blob; mime: string } | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const currentPlaybackSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const transcriptRef = useRef<TranscriptEntry[]>([]);
  const isProcessingRef = useRef(false);
  const isListeningRef = useRef(false);
  const startRecordingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isAiSpeakingRef = useRef(false);
  const targetRateRef = useRef<number>(16000);
  const manualBufferRef = useRef<string[]>([]);
  const pendingManualCommitRef = useRef(false);
  const pendingManualCommitTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const manualModeRef = useRef(manualMode);
  manualModeRef.current = manualMode;
  const isConnectedRef = useRef(false);
  const pendingReconnectRef = useRef(false);
  const backendSessionIdRef = useRef<string | null>(null);
  const startRecordingAfterAIRef = useRef<((playbackFailed?: boolean) => void) | null>(null);
  const speakRef = useRef<((text: string) => Promise<void>) | null>(null);

  const isVoiceMemoMode = typeof navigator !== "undefined" && (/iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1));
  const POST_AI_LISTEN_DELAY_MS = isVoiceMemoMode ? 1800 : 700;
  const DEBUG_LOG = false;
  const log = (...args: unknown[]) => DEBUG_LOG && console.log("[Personaplex]", ...args);

  const processUserInput = useCallback(
    async (userText: string) => {
      if (isProcessingRef.current || !userText.trim()) return;
      log("processUserInput called", { text: userText.slice(0, 50) });
      isProcessingRef.current = true;
      setIsProcessing(true);

      const nextWithUser: TranscriptEntry[] = [...transcriptRef.current, { role: "user", text: userText }];
      transcriptRef.current = nextWithUser;
      onTranscriptUpdate(() => nextWithUser);
      onInterimTranscript("");

      try {
        log("Fetching /chat...");
        const { question, sessionId, retrievalLog, notesSaved } = await fetchInterviewerQuestion(
          userText,
          backendSessionIdRef.current,
          personalization,
          intrusiveness,
          sessionMode
        );
        log("Got response, speaking...");
        backendSessionIdRef.current = sessionId;
        if (notesSaved?.length && onNotesSaved) onNotesSaved(notesSaved);
        const nextWithAi: TranscriptEntry[] = [...nextWithUser, { role: "ai", text: question, ...(retrievalLog != null ? { retrievalLog } : {}) }];
        transcriptRef.current = nextWithAi;
        onTranscriptUpdate(() => nextWithAi);
        if (speakRef.current) {
          await speakRef.current(question);
        } else {
          log("speakRef.current is null, reopening mic");
          if (!isVoiceMemoMode && isConnectedRef.current && startRecordingAfterAIRef.current) {
            startRecordingAfterAIRef.current(true);
          }
        }
      } catch (err) {
        console.error("[Personaplex] Interviewer API error:", err);
        const message =
          err instanceof Error
            ? (err.name === "AbortError" ? "Request timed out. Check the backend and try again." : err.message)
            : "API error";
        setErrorMessage(message);
        setStatus("error");
        if (!isVoiceMemoMode && isConnectedRef.current && startRecordingAfterAIRef.current) {
          startRecordingAfterAIRef.current(true);
        }
      } finally {
        isProcessingRef.current = false;
        setIsProcessing(false);
      }
    },
    [personalization, intrusiveness, sessionMode, onTranscriptUpdate, onInterimTranscript, onNotesSaved]
  );

  const speakWithVoiceApi = useCallback(
    (text: string, onDone: (playbackFailed?: boolean) => void, setError: (msg: string | null) => void) => {
      if (!text.trim()) {
        onDone();
        return;
      }

      const done = (playbackFailed?: boolean) => {
        isAiSpeakingRef.current = false;
        setIsAiSpeaking(false);
        if (playbackFailed && isVoiceMemoMode) setLastPlaybackFailed(true);
        log("AI finished speaking:", text.slice(0, 100) + (text.length > 100 ? "..." : ""), playbackFailed ? "(playback failed)" : "");
        onDone(playbackFailed);
      };

      isAiSpeakingRef.current = true;
      setIsAiSpeaking(true);
      if (isVoiceMemoMode) {
        lastFailedPlaybackRef.current = null;
        setLastPlaybackFailed(false);
      }
      log("AI started speaking:", text);
      setError(null);

      fetchVoiceAudio(text, selectedVoiceId, voiceSettings)
        .then(async ({ base64, format }) => {
          const mime = format === "mp3" ? "audio/mpeg" : "audio/wav";
          const binary = atob(base64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          const blob = new Blob([bytes], { type: mime });
          const blobUrl = URL.createObjectURL(blob);

          const playWithHtmlAudio = () => {
            if (isVoiceMemoMode) lastFailedPlaybackRef.current = { blob, mime };
            const audioEl = new Audio();
            currentAudioRef.current = audioEl;
            audioEl.onended = () => {
              URL.revokeObjectURL(blobUrl);
              currentAudioRef.current = null;
              if (isVoiceMemoMode) lastFailedPlaybackRef.current = null;
              done();
            };
            audioEl.onerror = () => {
              URL.revokeObjectURL(blobUrl);
              currentAudioRef.current = null;
              if (isVoiceMemoMode) setError("Audio playback failed. Tap Play to hear.");
              else setError("Audio playback failed");
              done(true);
            };
            audioEl.src = blobUrl;
            audioEl.play().catch(() => {
              URL.revokeObjectURL(blobUrl);
              currentAudioRef.current = null;
              if (isVoiceMemoMode) setError("Audio playback failed. Tap Play to hear.");
              else setError("Audio playback failed");
              done(true);
            });
          };

          const ctx = playbackContextRef.current;
          if (!isVoiceMemoMode && ctx) {
            try {
              await Promise.race([ctx.resume(), new Promise((_, r) => setTimeout(() => r(new Error("resume timeout")), 3000))]);
              const buffer = await ctx.decodeAudioData(bytes.buffer.slice(0, bytes.byteLength));
              const source = ctx.createBufferSource();
              source.buffer = buffer;
              source.connect(ctx.destination);
              currentPlaybackSourceRef.current = source;
              source.onended = () => {
                currentPlaybackSourceRef.current = null;
                done();
              };
              source.start(0);
              URL.revokeObjectURL(blobUrl);
            } catch (e) {
              console.warn("[Personaplex] Web Audio failed, trying HTML Audio:", e);
              playWithHtmlAudio();
            }
          } else {
            playWithHtmlAudio();
          }
        })
        .catch((err) => {
          const msg = err instanceof Error ? err.message : "Voice API failed";
          setError(msg);
          console.error("[Personaplex] Voice API error:", err);
          done(true);
        });
    },
    [selectedVoiceId]
  );

  const stopAiPlayback = useCallback(() => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
    if (currentPlaybackSourceRef.current) {
      try {
        currentPlaybackSourceRef.current.stop();
      } catch {
        /* already stopped */
      }
      currentPlaybackSourceRef.current = null;
    }
    isAiSpeakingRef.current = false;
    setIsAiSpeaking(false);
    log("AI playback stopped");
  }, []);

  const stopRecording = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
      wsRef.current = null;
    }
    const proc = processorRef.current;
    if (proc) {
      proc.disconnect();
      processorRef.current = null;
    }
    const src = sourceRef.current;
    if (src) {
      src.disconnect();
      sourceRef.current = null;
    }
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    audioContextRef.current?.close();
    audioContextRef.current = null;
    isListeningRef.current = false;
    setIsUserSpeaking(false);
  }, []);

  const startRecording = useCallback(() => {
    if (!isConnectedRef.current || isProcessingRef.current || isListeningRef.current) return;

    const start = async () => {
      try {
        let stream: MediaStream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            },
          });
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        streamRef.current = stream;

        const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const ctx = new AudioContextClass();
        audioContextRef.current = ctx;

        const sourceRate = ctx.sampleRate;
        // Force 16kHz for best mobile compatibility - ElevenLabs recommends it for speech
        const targetRate = 16000;
        const audioFormat = "pcm_16000";

        const token = await fetchScribeToken();
        targetRateRef.current = targetRate;

        // Manual turn-taking: user ends turn with button, no hands-free phrases.
        const commitStrategy = "manual";
        log("WebSocket params: commit_strategy =", commitStrategy, "sourceRate =", sourceRate);
        const params = new URLSearchParams({
          token,
          model_id: SCRIBE_MODEL,
          commit_strategy: commitStrategy,
          audio_format: audioFormat,
          language_code: "en",
        });
        const ws = new WebSocket(`${SCRIBE_WS_URL}?${params}`);
        wsRef.current = ws;

        ws.onerror = () => {
          setErrorMessage("Live transcription connection failed");
          setStatus("error");
        };

        ws.onclose = () => {
          wsRef.current = null;
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data) as { message_type?: string; text?: string; error?: string };
            const type = msg.message_type;

            if (type === "partial_transcript" && typeof msg.text === "string") {
              // While user is speaking, optionally show the latest partial text as live transcription.
              if (!isAiSpeakingRef.current && showLiveTranscription) {
                onInterimTranscript(msg.text);
              }
            } else if (type === "committed_transcript" && typeof msg.text === "string") {
              const text = msg.text.trim();
              if (text) {
                // Manual mode: buffer committed chunks until the user taps Done, then send once.
                if (isAiSpeakingRef.current) {
                  log("IGNORED committed_transcript (AI still speaking):", text.slice(0, 50));
                  return;
                }
                manualBufferRef.current.push(text);
                if (pendingManualCommitRef.current) {
                  if (pendingManualCommitTimeoutRef.current) {
                    clearTimeout(pendingManualCommitTimeoutRef.current);
                    pendingManualCommitTimeoutRef.current = null;
                  }
                  const fullText = manualBufferRef.current.join(" ").trim();
                  manualBufferRef.current = [];
                  pendingManualCommitRef.current = false;
                  stopRecording();
                  if (fullText) {
                    log("Processing (user clicked Done):", fullText.slice(0, 50));
                    processUserInput(fullText);
                  } else {
                    onInterimTranscript("");
                  }
                } else {
                  log("Buffered chunk (waiting for Done click)");
                  if (showLiveTranscription) {
                    onInterimTranscript(manualBufferRef.current.join(" "));
                  }
                }
              }
            } else if (type === "error" || type === "auth_error" || type === "quota_exceeded") {
              const err = msg.error ?? "Transcription error";
              console.error("[Personaplex] Scribe error:", err);
              setErrorMessage(err);
              stopRecording();
              if (isAiSpeakingRef.current) {
                pendingReconnectRef.current = true;
              } else {
                startRecording();
              }
            }
          } catch {
            // ignore parse errors
          }
        };

        ws.onopen = () => {
          setErrorMessage(null);
          try {
            const source = ctx.createMediaStreamSource(stream);
            sourceRef.current = source;

            const bufferSize = 4096;
            const processor = ctx.createScriptProcessor(bufferSize, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
              const w = wsRef.current;
              if (!w || w.readyState !== WebSocket.OPEN) return;

              const input = e.inputBuffer.getChannelData(0);
              const base64 = float32ToPcmBase64(input, targetRate, sourceRate);

              w.send(
                JSON.stringify({
                  message_type: "input_audio_chunk",
                  audio_base_64: base64,
                  sample_rate: targetRate,
                  commit: false,
                })
              );
            };

            source.connect(processor);
            processor.connect(ctx.destination);

            isListeningRef.current = true;
            setIsUserSpeaking(true);
            manualBufferRef.current = [];
            onInterimTranscript("Listening...");
          } catch (err) {
            console.error("[Personaplex] Mic access error:", err);
            setErrorMessage("Microphone access denied or unavailable");
            setStatus("error");
            ws.close();
          }
        };
      } catch (err) {
        console.error("[Personaplex] Scribe token error:", err);
        setErrorMessage(err instanceof Error ? err.message : "Could not start live transcription");
        setStatus("error");
      }
    };

    start();
  }, [processUserInput, onInterimTranscript, stopRecording, stopAiPlayback, showLiveTranscription]);

  const commitManual = useCallback(() => {
    const w = wsRef.current;
    if (!w || w.readyState !== WebSocket.OPEN || !isListeningRef.current) return;
    if (isAiSpeakingRef.current) return;

    pendingManualCommitRef.current = true;

    const rate = targetRateRef.current;
    const silentChunk = createSilentPcmBase64(1024);
    w.send(
      JSON.stringify({
        message_type: "input_audio_chunk",
        audio_base_64: silentChunk,
        sample_rate: rate,
        commit: true,
      })
    );
    log("Manual commit sent (buffered chunks:", manualBufferRef.current.length, ")");

    pendingManualCommitTimeoutRef.current = setTimeout(() => {
      pendingManualCommitTimeoutRef.current = null;
      if (pendingManualCommitRef.current && manualBufferRef.current.length > 0) {
        pendingManualCommitRef.current = false;
        const fullText = manualBufferRef.current.join(" ").trim();
        manualBufferRef.current = [];
        stopRecording();
        if (fullText) processUserInput(fullText);
      }
    }, 1500);
  }, [stopRecording, processUserInput]);

  const startVoiceMemoRecording = useCallback(async () => {
    if (!isConnectedRef.current || isProcessingRef.current || isVoiceMemoRecording) return;
    try {
      const silent = new Audio("data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=");
      silent.volume = 0;
      silent.play().catch(() => {});
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      voiceMemoStreamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      voiceMemoChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) voiceMemoChunksRef.current.push(e.data);
      };
      recorder.start();
      setIsVoiceMemoRecording(true);
      setErrorMessage(null);
    } catch (err) {
      console.error("[Personaplex] Voice memo start error:", err);
      setErrorMessage("Microphone access denied or unavailable");
    }
  }, [isVoiceMemoRecording]);

  const playLastFailedPlayback = useCallback(() => {
    const pending = lastFailedPlaybackRef.current;
    if (!pending) return;
    lastFailedPlaybackRef.current = null;
    setLastPlaybackFailed(false);
    setErrorMessage(null);
    const url = URL.createObjectURL(pending.blob);
    const audioEl = new Audio(url);
    audioEl.onended = () => {
      URL.revokeObjectURL(url);
    };
    audioEl.onerror = () => {
      URL.revokeObjectURL(url);
      setErrorMessage("Playback failed");
    };
    audioEl.play().catch(() => {
      URL.revokeObjectURL(url);
      setErrorMessage("Playback failed");
    });
  }, []);

  const stopVoiceMemoRecording = useCallback(async () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") return;
    const chunks = voiceMemoChunksRef.current;
    return new Promise<void>((resolve) => {
      recorder.onstop = async () => {
        mediaRecorderRef.current = null;
        const stream = voiceMemoStreamRef.current;
        voiceMemoStreamRef.current = null;
        setIsVoiceMemoRecording(false);
        stream?.getTracks().forEach((t) => t.stop());
        if (!isConnectedRef.current) {
          resolve();
          return;
        }
        if (chunks.length === 0) {
          setErrorMessage("Recording too short");
          resolve();
          return;
        }
        try {
          const blob = new Blob(chunks, { type: recorder.mimeType });
          const base64 = await blobToWavBase64(blob);
          const text = await fetchTranscribe(base64);
          if (text && isConnectedRef.current) {
            onInterimTranscript("");
            processUserInput(text);
          } else if (!text) {
            setErrorMessage("No speech detected");
          }
        } catch (err) {
          console.error("[Personaplex] Voice memo transcribe error:", err);
          setErrorMessage(err instanceof Error ? err.message : "Transcription failed");
        }
        resolve();
      };
      recorder.stop();
    });
  }, [onInterimTranscript, processUserInput]);

  const startRecordingAfterAI = useCallback((playbackFailed?: boolean) => {
    if (!isConnectedRef.current) return;
    if (isVoiceMemoMode) return;
    if (isListeningRef.current) return; // mic never stopped; no need to start again
    const wasPendingReconnect = pendingReconnectRef.current;
    if (pendingReconnectRef.current) pendingReconnectRef.current = false;
    const delay = wasPendingReconnect ? 0 : (playbackFailed ? 2500 : POST_AI_LISTEN_DELAY_MS);
    log("Scheduling startRecording in", delay, "ms", wasPendingReconnect ? "(pending reconnect)" : playbackFailed ? "(playback failed)" : "");
    startRecordingTimeoutRef.current = setTimeout(() => {
      startRecordingTimeoutRef.current = null;
      if (!isConnectedRef.current) return;
      log("Starting recording (mic open)");
      startRecording();
    }, delay);
  }, [startRecording]);
  startRecordingAfterAIRef.current = startRecordingAfterAI;

  const speak = useCallback(
    async (text: string) => {
      if (!text.trim()) {
        startRecording();
        return;
      }

      if (currentAudioRef.current) {
        log("Pausing current AI audio (starting new response)");
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
      if (currentPlaybackSourceRef.current) {
        try {
          currentPlaybackSourceRef.current.stop();
        } catch {
          /* already stopped */
        }
        currentPlaybackSourceRef.current = null;
      }

      speakWithVoiceApi(text, startRecordingAfterAI, setErrorMessage);
    },
    [speakWithVoiceApi, startRecordingAfterAI, startRecording]
  );
  speakRef.current = speak;

  const connect = useCallback(() => {
    log("Connect");
    isConnectedRef.current = true;
    setStatus("connecting");
    setErrorMessage(null);
    transcriptRef.current = [];

    const PlaybackCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new PlaybackCtx();
    playbackContextRef.current = ctx;
    ctx.resume().catch(() => {});

    if (isVoiceMemoMode) {
      const silent = new Audio("data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=");
      silent.volume = 0;
      silent.play().catch(() => {});
    }

    setStatus("connected");
    speak("Hello, I am your Selfmeridian assistant. How can I help you?");
  }, [speak]);

  const disconnect = useCallback(() => {
    log("Disconnect");
    const sessionId = backendSessionIdRef.current;
    if (sessionId) {
      backendFetch("/end-session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(() => {});
      backendSessionIdRef.current = null;
    }
    isConnectedRef.current = false;
    if (pendingManualCommitTimeoutRef.current) {
      clearTimeout(pendingManualCommitTimeoutRef.current);
      pendingManualCommitTimeoutRef.current = null;
    }
    if (startRecordingTimeoutRef.current) {
      clearTimeout(startRecordingTimeoutRef.current);
      startRecordingTimeoutRef.current = null;
    }
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    if (currentPlaybackSourceRef.current) {
      try {
        currentPlaybackSourceRef.current.stop();
      } catch {
        /* already stopped */
      }
      currentPlaybackSourceRef.current = null;
    }
    if (playbackContextRef.current) {
      playbackContextRef.current.close().catch(() => {});
      playbackContextRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      voiceMemoStreamRef.current?.getTracks().forEach((t) => t.stop());
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
      voiceMemoStreamRef.current = null;
      setIsVoiceMemoRecording(false);
    }
    stopRecording();
    setStatus("disconnected");
    setErrorMessage(null);
    setIsUserSpeaking(false);
    setIsAiSpeaking(false);
    isAiSpeakingRef.current = false;
    isProcessingRef.current = false;
    isListeningRef.current = false;
    manualBufferRef.current = [];
    pendingManualCommitRef.current = false;
    pendingReconnectRef.current = false;
    lastFailedPlaybackRef.current = null;
    setLastPlaybackFailed(false);
  }, [stopRecording]);

  return {
    status,
    errorMessage,
    isProcessing,
    connect,
    disconnect,
    commitManual,
    isConnected: status === "connected",
    isUserSpeaking,
    isAiSpeaking,
    isVoiceMemoMode,
    isVoiceMemoRecording,
    startVoiceMemoRecording,
    stopVoiceMemoRecording,
    lastPlaybackFailed,
    playLastFailedPlayback,
  };
};
