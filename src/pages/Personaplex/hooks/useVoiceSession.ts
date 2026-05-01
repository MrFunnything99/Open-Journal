import { useCallback, useState } from "react";

export type VoiceSessionState = "idle" | "listening" | "thinking" | "speaking" | "paused";

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
  skipResponse: () => void;
  isMuted: boolean;
  speakResponse: (text: string) => Promise<void>;
};

export function useVoiceSession({ onSendTranscript: _onSendTranscript }: UseVoiceSessionOpts): UseVoiceSessionReturn {
  const [isMuted, setIsMuted] = useState(false);

  const startSession = useCallback(() => {
    /* Voice-to-voice is disabled until Tinfoil exposes TTS. */
  }, []);

  const endSession = useCallback(() => {
    /* no-op */
  }, []);

  const toggleMute = useCallback(() => {
    setIsMuted((v) => !v);
  }, []);

  const skipResponse = useCallback(() => {
    /* no-op */
  }, []);

  const speakResponse = useCallback(async (_text: string) => {
    /* no-op: TTS disabled */
  }, []);

  return {
    voiceState: "idle",
    partialTranscript: "",
    fullTranscript: "",
    assistantText: "",
    startSession,
    endSession,
    toggleMute,
    skipResponse,
    isMuted,
    speakResponse,
  };
}
