import { useCallback, useRef, useState } from "react";

export type PersonaplexConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

export type UsePersonaplexSessionOptions = {
  systemPrompt: string;
  selectedVoiceId: string;
  manualMode?: boolean;
  onTranscriptUpdate: (updater: (prev: Array<{ role: "user" | "ai"; text: string }>) => Array<{ role: "user" | "ai"; text: string }>) => void;
  onInterimTranscript: (text: string) => void;
};

async function fetchInterviewerQuestion(
  systemPrompt: string,
  messages: Array<{ role: "user" | "ai"; text: string }>
): Promise<string> {
  const res = await fetch("/api/interviewer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ systemPrompt, messages }),
  });

  const rawText = await res.text();
  let data: { error?: string; question?: string } = {};
  if (rawText.trim()) {
    try {
      data = JSON.parse(rawText) as { error?: string; question?: string };
    } catch {
      const snippet = rawText.slice(0, 80).replace(/\s+/g, " ");
      throw new Error(
        res.status === 404
          ? "API route not found. Use 'vercel dev' for local dev with API."
          : res.ok
            ? "Invalid response from server"
            : `Server error (${res.status}): ${snippet || res.statusText}`
      );
    }
  }

  if (!res.ok) {
    throw new Error(data.error || `Interviewer API failed (${res.status})`);
  }

  if (!data.question || typeof data.question !== "string") {
    throw new Error(data.error || "No question in response");
  }

  return data.question;
}

async function fetchVoiceAudio(
  text: string,
  voiceId: string
): Promise<{ base64: string; format: string }> {
  const res = await fetch("/api/voice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voiceId }),
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
  systemPrompt,
  selectedVoiceId,
  manualMode = false,
  onTranscriptUpdate,
  onInterimTranscript,
}: UsePersonaplexSessionOptions) => {
  const [status, setStatus] = useState<PersonaplexConnectionStatus>("disconnected");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const currentPlaybackSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const transcriptRef = useRef<Array<{ role: "user" | "ai"; text: string }>>([]);
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

  const POST_AI_LISTEN_DELAY_MS = 700;
  const DEBUG_LOG = true; // Set to false to disable session logs
  const log = (...args: unknown[]) => DEBUG_LOG && console.log("[Personaplex]", ...args);

  const processUserInput = useCallback(
    async (userText: string) => {
      if (isProcessingRef.current || !userText.trim()) return;
      log("processUserInput called", { text: userText.slice(0, 50) });
      isProcessingRef.current = true;
      stopRecording();

      const nextWithUser = [...transcriptRef.current, { role: "user" as const, text: userText }];
      transcriptRef.current = nextWithUser;
      onTranscriptUpdate(() => nextWithUser);
      onInterimTranscript("");

      try {
        const question = await fetchInterviewerQuestion(
          systemPrompt,
          nextWithUser
        );
        const nextWithAi = [...nextWithUser, { role: "ai" as const, text: question }];
        transcriptRef.current = nextWithAi;
        onTranscriptUpdate(() => nextWithAi);
        await speak(question);
      } catch (err) {
        console.error("[Personaplex] Interviewer API error:", err);
        setErrorMessage(err instanceof Error ? err.message : "API error");
        setStatus("error");
        isProcessingRef.current = false;
      } finally {
        isProcessingRef.current = false;
      }
    },
    [systemPrompt, onTranscriptUpdate, onInterimTranscript]
  );

  const speakWithVoiceApi = useCallback(
    (text: string, onDone: () => void, setError: (msg: string | null) => void) => {
      if (!text.trim()) {
        onDone();
        return;
      }

      const done = () => {
        isAiSpeakingRef.current = false;
        setIsAiSpeaking(false);
        log("AI finished speaking:", text.slice(0, 100) + (text.length > 100 ? "..." : ""));
        onDone();
      };

      isAiSpeakingRef.current = true;
      setIsAiSpeaking(true);
      log("AI started speaking:", text);
      setError(null);

      fetchVoiceAudio(text, selectedVoiceId)
        .then(async ({ base64, format }) => {
          const mime = format === "mp3" ? "audio/mpeg" : "audio/wav";
          const binary = atob(base64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          const blob = new Blob([bytes], { type: mime });
          const blobUrl = URL.createObjectURL(blob);

          const playWithHtmlAudio = () => {
            const audioEl = new Audio();
            currentAudioRef.current = audioEl;
            audioEl.playsInline = true;
            audioEl.onended = () => {
              URL.revokeObjectURL(blobUrl);
              currentAudioRef.current = null;
              done();
            };
            audioEl.onerror = () => {
              URL.revokeObjectURL(blobUrl);
              currentAudioRef.current = null;
              setError("Audio playback failed");
              done();
            };
            audioEl.src = blobUrl;
            audioEl.play().catch(() => {
              URL.revokeObjectURL(blobUrl);
              currentAudioRef.current = null;
              setError("Audio playback failed");
              done();
            });
          };

          const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
          const ctx = playbackContextRef.current;
          if (!isIOS && ctx) {
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
          done();
        });
    },
    [selectedVoiceId]
  );

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
  }, []);

  const startRecording = useCallback(() => {
    if (isProcessingRef.current || isListeningRef.current) return;

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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

        const commitStrategy = manualMode ? "manual" : "vad";
        log("WebSocket params: commit_strategy =", commitStrategy, "sourceRate =", sourceRate);
        const params = new URLSearchParams({
          token,
          model_id: SCRIBE_MODEL,
          commit_strategy: commitStrategy,
          audio_format: audioFormat,
          language_code: "en",
        });
        if (!manualMode) {
          params.set("vad_silence_threshold_secs", "2.0");
          params.set("vad_threshold", "0.55");
          params.set("min_speech_duration_ms", "250");
        }
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
              if (manualModeRef.current && manualBufferRef.current.length > 0) {
                const accumulated = manualBufferRef.current.join(" ") + " " + msg.text;
                onInterimTranscript(accumulated);
              } else {
                onInterimTranscript(msg.text);
              }
            } else if (type === "committed_transcript" && typeof msg.text === "string") {
              const text = msg.text.trim();
              if (text) {
                if (isAiSpeakingRef.current) {
                  log("IGNORED committed_transcript (AI still speaking):", text.slice(0, 50));
                  return;
                }
                log("committed_transcript received:", text.slice(0, 50), "manualMode:", manualModeRef.current);

                if (manualModeRef.current) {
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
                      log("Processing (user clicked Done)");
                      processUserInput(fullText);
                    }
                  } else {
                    log("Buffered chunk (waiting for Done click)");
                    onInterimTranscript(manualBufferRef.current.join(" "));
                  }
                } else {
                  log("Processing (VAD mode)");
                  stopRecording();
                  processUserInput(text);
                }
              }
            } else if (type === "error" || type === "auth_error" || type === "quota_exceeded") {
              const err = msg.error ?? "Transcription error";
              console.error("[Personaplex] Scribe error:", err);
              setErrorMessage(err);
              stopRecording();
              startRecording();
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
            if (manualModeRef.current) manualBufferRef.current = [];
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
  }, [processUserInput, onInterimTranscript, stopRecording, manualMode]);

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

  const startRecordingAfterAI = useCallback(() => {
    log("Scheduling startRecording in", POST_AI_LISTEN_DELAY_MS, "ms");
    startRecordingTimeoutRef.current = setTimeout(() => {
      startRecordingTimeoutRef.current = null;
      log("Starting recording (mic open)");
      startRecording();
    }, POST_AI_LISTEN_DELAY_MS);
  }, [startRecording]);

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

  const connect = useCallback(() => {
    log("Connect");
    setStatus("connecting");
    setErrorMessage(null);
    transcriptRef.current = [];

    const PlaybackCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new PlaybackCtx();
    playbackContextRef.current = ctx;
    ctx.resume().catch(() => {});

    setStatus("connected");
    speak("Hello, I am your OpenJournal assistant. How can I help you?");
  }, [speak]);

  const disconnect = useCallback(() => {
    log("Disconnect");
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
  }, [stopRecording]);

  return {
    status,
    errorMessage,
    connect,
    disconnect,
    commitManual,
    isConnected: status === "connected",
    isUserSpeaking,
    isAiSpeaking,
  };
};
