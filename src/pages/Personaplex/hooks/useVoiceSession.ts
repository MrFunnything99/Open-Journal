import { useCallback, useEffect, useRef, useState } from "react";
import { backendFetch } from "../../../backendApi";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type VoiceSessionState = "idle" | "listening" | "thinking" | "speaking" | "paused";

type SpeechRecInstance = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((this: SpeechRecInstance, ev: Event) => void) | null;
  onerror: ((this: SpeechRecInstance, ev: Event) => void) | null;
  onend: (() => void) | null;
  onsoundstart: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechRecAlternative = { transcript: string; confidence?: number };
type SpeechRecResult = { isFinal: boolean; 0: SpeechRecAlternative; length: number };
type SpeechRecEvent = Event & {
  resultIndex: number;
  results: ArrayLike<SpeechRecResult> & { length: number };
};

function getSpeechRecCtor(): (new () => SpeechRecInstance) | undefined {
  if (typeof window === "undefined") return undefined;
  const w = window as unknown as {
    SpeechRecognition?: new () => SpeechRecInstance;
    webkitSpeechRecognition?: new () => SpeechRecInstance;
  };
  return w.SpeechRecognition || w.webkitSpeechRecognition;
}

// ---------------------------------------------------------------------------
// Barge-in while TTS: avoid false triggers from echo / noise
// ---------------------------------------------------------------------------

/** Min time RMS stays above threshold before it counts toward barge-in. */
const BARGE_IN_RMS_MS = 400;
/** Min time we have considered “confident” speech hypothesis before stopping TTS. */
const BARGE_IN_SPEECH_MS = 450;
/** After TTS starts, ignore barge-in briefly (transients / echo). */
const BARGE_IN_COOLDOWN_MS = 500;
/** Chrome exposes confidence on alternatives; require this when present. */
const BARGE_IN_CONFIDENCE = 0.85;
/** RMS ~ average normalized sample magnitude; tune for quiet rooms vs noisy. */
const RMS_THRESHOLD = 0.018;
const RMS_HYSTERESIS = 0.65;

function rmsFromTimeDomain(data: Uint8Array): number {
  if (data.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i]! - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / data.length);
}

/**
 * True if this recognition slice should count toward intentional barge-in
 * (not used alone — also need sustained RMS).
 */
function speechHypothesisQualifiesForBargeIn(
  transcript: string,
  confidence: number | undefined,
  isFinal: boolean,
): boolean {
  const t = transcript.trim();
  if (t.length < 5) return false;

  if (confidence !== undefined && !Number.isNaN(confidence)) {
    if (confidence < BARGE_IN_CONFIDENCE) return false;
    return isFinal || t.length >= 10;
  }

  // No confidence (e.g. Safari): require stronger text signal
  if (isFinal) return t.length >= 8;
  return t.length >= 22;
}

// ---------------------------------------------------------------------------
// TTS helper (mirrors chatReadAloud.ts but returns the Audio element)
// ---------------------------------------------------------------------------

function mimeForTtsFormat(format: string | undefined): string {
  switch ((format || "mp3").toLowerCase()) {
    case "opus":
      return "audio/ogg; codecs=opus";
    case "wav":
      return "audio/wav";
    case "flac":
      return "audio/flac";
    default:
      return "audio/mpeg";
  }
}

async function fetchTtsAudio(text: string): Promise<HTMLAudioElement | null> {
  const t = text.trim().slice(0, 12_000);
  if (!t) return null;

  const res = await backendFetch("/voice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: t }),
  });

  const data = (await res.json().catch(() => ({}))) as {
    audio?: string;
    format?: string;
    playback_rate?: number;
  };
  if (!res.ok || !data.audio) return null;

  const bin = atob(data.audio);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

  const blob = new Blob([bytes], { type: mimeForTtsFormat(data.format) });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  const pr =
    typeof data.playback_rate === "number" && data.playback_rate > 0
      ? data.playback_rate
      : 1;
  audio.playbackRate = Math.min(4, Math.max(0.25, pr));

  const cleanup = () => URL.revokeObjectURL(url);
  audio.addEventListener("ended", cleanup, { once: true });
  audio.addEventListener("error", cleanup, { once: true });

  return audio;
}

// ---------------------------------------------------------------------------
// Silence detection timer (ms)
// ---------------------------------------------------------------------------

const SILENCE_TIMEOUT_MS = 2500;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export type UseVoiceSessionOpts = {
  onSendTranscript: (text: string) => Promise<void>;
};

export type UseVoiceSessionReturn = {
  voiceState: VoiceSessionState;
  partialTranscript: string;
  fullTranscript: string;
  assistantText: string;
  startSession: () => void;
  endSession: () => void;
  toggleMute: () => void;
  isMuted: boolean;
  speakResponse: (text: string) => Promise<void>;
};

export function useVoiceSession({
  onSendTranscript,
}: UseVoiceSessionOpts): UseVoiceSessionReturn {
  const [voiceState, setVoiceState] = useState<VoiceSessionState>("idle");
  const [partialTranscript, setPartialTranscript] = useState("");
  const [fullTranscript, setFullTranscript] = useState("");
  const [assistantText, setAssistantText] = useState("");
  const [isMuted, setIsMuted] = useState(false);

  const speechRecRef = useRef<SpeechRecInstance | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const activeRef = useRef(false);
  const accumulatedFinalRef = useRef("");
  const onSendRef = useRef(onSendTranscript);
  onSendRef.current = onSendTranscript;
  const stateRef = useRef<VoiceSessionState>("idle");

  const micStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const vadFrameRef = useRef<number | null>(null);
  const timeDomainDataRef = useRef<Uint8Array | null>(null);

  const rmsHighSinceRef = useRef<number | null>(null);
  const speechQualifySinceRef = useRef<number | null>(null);
  const ttsPlaybackStartRef = useRef<number>(0);

  useEffect(() => {
    stateRef.current = voiceState;
  }, [voiceState]);

  const clearSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current != null) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }, []);

  const stopVadLoop = useCallback(() => {
    if (vadFrameRef.current != null) {
      cancelAnimationFrame(vadFrameRef.current);
      vadFrameRef.current = null;
    }
    rmsHighSinceRef.current = null;
    speechQualifySinceRef.current = null;
  }, []);

  const resetBargeInGates = useCallback(() => {
    rmsHighSinceRef.current = null;
    speechQualifySinceRef.current = null;
  }, []);

  const stopTts = useCallback(() => {
    const a = ttsAudioRef.current;
    if (a) {
      a.pause();
      a.src = "";
      ttsAudioRef.current = null;
    }
    resetBargeInGates();
    stopVadLoop();
  }, [resetBargeInGates, stopVadLoop]);

  const stopSpeechRec = useCallback(() => {
    const rec = speechRecRef.current;
    speechRecRef.current = null;
    if (!rec) return;
    rec.onresult = null;
    rec.onerror = null;
    rec.onend = null;
    rec.onsoundstart = null;
    try {
      rec.abort();
    } catch {
      /* ignore */
    }
  }, []);

  const releaseMicStream = useCallback(() => {
    stopVadLoop();
    const ctx = audioContextRef.current;
    if (ctx) {
      void ctx.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    timeDomainDataRef.current = null;
    const s = micStreamRef.current;
    if (s) {
      s.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
  }, [stopVadLoop]);

  const flushTranscript = useCallback(() => {
    clearSilenceTimer();
    const text = accumulatedFinalRef.current.trim();
    if (!text) return;

    setFullTranscript(text);
    setPartialTranscript("");
    accumulatedFinalRef.current = "";

    stopSpeechRec();

    setVoiceState("thinking");
    setAssistantText("");

    void onSendRef.current(text);
  }, [clearSilenceTimer, stopSpeechRec]);

  const resetSilenceTimer = useCallback(() => {
    clearSilenceTimer();
    silenceTimerRef.current = setTimeout(() => {
      if (stateRef.current === "listening" && accumulatedFinalRef.current.trim()) {
        flushTranscript();
      }
    }, SILENCE_TIMEOUT_MS);
  }, [clearSilenceTimer, flushTranscript]);

  const tryBargeIn = useCallback(() => {
    if (stateRef.current !== "speaking" || !activeRef.current) return;
    const now = performance.now();
    if (now - ttsPlaybackStartRef.current < BARGE_IN_COOLDOWN_MS) return;

    const rmsSince = rmsHighSinceRef.current;
    const speechSince = speechQualifySinceRef.current;
    if (rmsSince == null || speechSince == null) return;

    if (now - rmsSince >= BARGE_IN_RMS_MS && now - speechSince >= BARGE_IN_SPEECH_MS) {
      stopTts();
      setVoiceState("listening");
      resetBargeInGates();
      // SpeechRecognition was already running for barge-in; no restart needed.
    }
  }, [resetBargeInGates, stopTts]);

  const runVadFrame = useCallback(() => {
    const analyser = analyserRef.current;
    const buf = timeDomainDataRef.current;
    if (!analyser || !buf || stateRef.current !== "speaking" || !activeRef.current) {
      vadFrameRef.current = null;
      return;
    }

    analyser.getByteTimeDomainData(buf as Uint8Array<ArrayBuffer>);
    const rms = rmsFromTimeDomain(buf);
    const now = performance.now();

    const high = rms >= RMS_THRESHOLD;
    const low = rms < RMS_THRESHOLD * RMS_HYSTERESIS;

    if (high) {
      if (rmsHighSinceRef.current == null) rmsHighSinceRef.current = now;
    } else if (low) {
      rmsHighSinceRef.current = null;
      speechQualifySinceRef.current = null;
    }

    tryBargeIn();

    vadFrameRef.current = requestAnimationFrame(runVadFrame);
  }, [tryBargeIn]);

  const startVadWhileSpeaking = useCallback(() => {
    stopVadLoop();
    resetBargeInGates();
    ttsPlaybackStartRef.current = performance.now();

    const analyser = analyserRef.current;
    if (!analyser) return;

    const size = analyser.fftSize;
    timeDomainDataRef.current = new Uint8Array(size);
    vadFrameRef.current = requestAnimationFrame(runVadFrame);
  }, [resetBargeInGates, runVadFrame, stopVadLoop]);

  const startSpeechRec = useCallback(() => {
    stopSpeechRec();
    const Ctor = getSpeechRecCtor();
    if (!Ctor) return;

    const createAndWire = () => {
      const rec = new Ctor();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang =
        typeof navigator !== "undefined" && navigator.language
          ? navigator.language
          : "en-US";

      rec.onresult = (ev: Event) => {
        const event = ev as SpeechRecEvent;
        let interim = "";
        let bargeInCandidate = false;

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const r = event.results[i]!;
          const alt = r[0] ?? { transcript: "" };
          const piece = alt.transcript ?? "";
          const conf =
            typeof alt.confidence === "number" && !Number.isNaN(alt.confidence)
              ? alt.confidence
              : undefined;

          if (r.isFinal) {
            accumulatedFinalRef.current += piece;
            resetSilenceTimer();
          } else {
            interim += piece;
          }

          const chunk = piece.trim();
          if (
            stateRef.current === "speaking" &&
            chunk &&
            speechHypothesisQualifiesForBargeIn(chunk, conf, r.isFinal)
          ) {
            bargeInCandidate = true;
          }
        }

        setPartialTranscript((accumulatedFinalRef.current + interim).trim());

        if (stateRef.current === "speaking" && bargeInCandidate) {
          const now = performance.now();
          if (speechQualifySinceRef.current == null) speechQualifySinceRef.current = now;
          tryBargeIn();
        } else if (stateRef.current !== "speaking") {
          speechQualifySinceRef.current = null;
        }
      };

      // Do not use onsoundstart for TTS interruption — it fires on echo/noise.
      rec.onsoundstart = null;

      rec.onerror = () => {
        /* best-effort; let onend handle restart */
      };

      rec.onend = () => {
        if (!activeRef.current) return;
        // During TTS, keep recognition alive for intentional barge-in (engine often stops sessions).
        if (stateRef.current === "speaking") {
          speechRecRef.current = null;
          try {
            const next = createAndWire();
            speechRecRef.current = next;
            next.start();
          } catch {
            /* give up */
          }
          return;
        }
        if (stateRef.current !== "listening") return;
        speechRecRef.current = null;
        try {
          const next = createAndWire();
          speechRecRef.current = next;
          next.start();
        } catch {
          /* give up */
        }
      };

      return rec;
    };

    try {
      const rec = createAndWire();
      speechRecRef.current = rec;
      rec.start();
    } catch {
      /* SpeechRecognition unavailable */
    }
  }, [stopSpeechRec, resetSilenceTimer, tryBargeIn]);

  const ensureMicStream = useCallback(async (): Promise<boolean> => {
    if (micStreamRef.current) return true;
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      return false;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      micStreamRef.current = stream;

      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) return true;

      const ctx = new Ctx();
      audioContextRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.35;
      source.connect(analyser);
      analyserRef.current = analyser;
      return true;
    } catch {
      return false;
    }
  }, []);

  const speakResponse = useCallback(
    async (text: string) => {
      if (!activeRef.current) return;
      setAssistantText(text);
      setVoiceState("speaking");

      try {
        const audio = await fetchTtsAudio(text);
        if (!activeRef.current) return;
        if (!audio) {
          setVoiceState("listening");
          startSpeechRec();
          return;
        }

        ttsAudioRef.current = audio;

        void ensureMicStream().then(() => {
          if (stateRef.current === "speaking" && activeRef.current) {
            startVadWhileSpeaking();
          }
        });

        startSpeechRec();

        audio.addEventListener(
          "ended",
          () => {
            ttsAudioRef.current = null;
            stopVadLoop();
            resetBargeInGates();
            if (!activeRef.current) return;
            setVoiceState("listening");
          },
          { once: true },
        );

        audio.addEventListener(
          "error",
          () => {
            ttsAudioRef.current = null;
            stopVadLoop();
            resetBargeInGates();
            if (!activeRef.current) return;
            setVoiceState("listening");
          },
          { once: true },
        );

        if (activeRef.current) {
          await audio.play();
        }
      } catch {
        if (activeRef.current) {
          setVoiceState("listening");
          startSpeechRec();
        }
      }
    },
    [ensureMicStream, resetBargeInGates, startSpeechRec, startVadWhileSpeaking, stopVadLoop],
  );

  const startSession = useCallback(async () => {
    if (activeRef.current) return;
    activeRef.current = true;
    accumulatedFinalRef.current = "";
    setPartialTranscript("");
    setFullTranscript("");
    setAssistantText("");
    setIsMuted(false);
    void ensureMicStream();
    setVoiceState("listening");
    startSpeechRec();
  }, [ensureMicStream, startSpeechRec]);

  const endSession = useCallback(() => {
    activeRef.current = false;
    clearSilenceTimer();
    stopSpeechRec();
    stopTts();
    releaseMicStream();
    accumulatedFinalRef.current = "";
    setPartialTranscript("");
    setFullTranscript("");
    setAssistantText("");
    setIsMuted(false);
    if (typeof window !== "undefined" && window.speechSynthesis) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        /* ignore */
      }
    }
    setVoiceState("idle");
  }, [clearSilenceTimer, releaseMicStream, stopSpeechRec, stopTts]);

  const toggleMute = useCallback(() => {
    if (!activeRef.current) return;
    setIsMuted((prev) => {
      const next = !prev;
      if (next) {
        stopSpeechRec();
        clearSilenceTimer();
        stopVadLoop();
        setVoiceState("paused");
      } else {
        setVoiceState("listening");
        startSpeechRec();
      }
      return next;
    });
  }, [stopSpeechRec, clearSilenceTimer, stopVadLoop, startSpeechRec]);

  useEffect(() => {
    return () => {
      activeRef.current = false;
      clearSilenceTimer();
      stopSpeechRec();
      stopTts();
      releaseMicStream();
    };
  }, [clearSilenceTimer, releaseMicStream, stopSpeechRec, stopTts]);

  return {
    voiceState,
    partialTranscript,
    fullTranscript,
    assistantText,
    startSession,
    endSession,
    toggleMute,
    isMuted,
    speakResponse,
  };
}
