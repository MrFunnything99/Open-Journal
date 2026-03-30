import { backendFetch } from "../../../backendApi";

let currentAudio: HTMLAudioElement | null = null;
let currentObjectUrl: string | null = null;

/** True while a TTS request is in flight (globally, one at a time). */
let readAloudRequestInFlight = false;

function stopChatReadAloud() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = "";
    currentAudio = null;
  }
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
}

function mimeForTtsFormat(format: string | undefined): string {
  const f = (format || "mp3").toLowerCase();
  switch (f) {
    case "opus":
      return "audio/ogg; codecs=opus";
    case "mp3":
      return "audio/mpeg";
    case "wav":
      return "audio/wav";
    case "flac":
      return "audio/flac";
    default:
      return "audio/mpeg";
  }
}

export type PlayChatReadAloudOptions = {
  /** Called when the TTS request starts and when it finishes (success, error, or blocked playback). */
  onLoading?: (loading: boolean) => void;
};

/**
 * Play assistant/user text via backend TTS (Mistral Voxtral when configured, else ElevenLabs).
 */
export async function playChatReadAloud(
  text: string,
  onError: (message: string) => void,
  opts?: PlayChatReadAloudOptions,
): Promise<void> {
  const t = text.trim();
  if (!t) {
    onError("Nothing to read.");
    return;
  }
  if (readAloudRequestInFlight) {
    return;
  }
  readAloudRequestInFlight = true;
  opts?.onLoading?.(true);
  stopChatReadAloud();

  try {
    let res: Response;
    try {
      res = await backendFetch("/voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t.slice(0, 12_000) }),
      });
    } catch {
      onError("Could not reach the server for read aloud.");
      return;
    }

    const data = (await res.json().catch(() => ({}))) as {
      detail?: unknown;
      audio?: string;
      format?: string;
      playback_rate?: number;
    };
    if (!res.ok) {
      const detail = data.detail;
      const msg =
        typeof detail === "string"
          ? detail
          : detail && typeof detail === "object" && "message" in detail && typeof (detail as { message: unknown }).message === "string"
            ? (detail as { message: string }).message
            : `Read aloud failed (${res.status}).`;
      onError(msg);
      return;
    }

    const b64 = typeof data.audio === "string" ? data.audio : "";
    if (!b64) {
      onError("No audio returned.");
      return;
    }

    let bytes: Uint8Array;
    try {
      const bin = atob(b64);
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    } catch {
      onError("Invalid audio data.");
      return;
    }

    const blob = new Blob([bytes], { type: mimeForTtsFormat(data.format) });
    const url = URL.createObjectURL(blob);
    currentObjectUrl = url;
    const audio = new Audio(url);
    currentAudio = audio;
    const pr = typeof data.playback_rate === "number" && data.playback_rate > 0 ? data.playback_rate : 1;
    audio.playbackRate = Math.min(4, Math.max(0.25, pr));
    audio.addEventListener("ended", () => {
      stopChatReadAloud();
    });
    try {
      await audio.play();
    } catch {
      stopChatReadAloud();
      onError("Playback was blocked or failed.");
    }
  } finally {
    readAloudRequestInFlight = false;
    opts?.onLoading?.(false);
  }
}

export function stopReadAloudPlayback() {
  stopChatReadAloud();
}
