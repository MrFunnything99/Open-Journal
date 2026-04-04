/// <reference types="node" />

const MISTRAL_SPEECH_URL = "https://api.mistral.ai/v1/audio/speech";
const DEFAULT_VOICE = "en_paul_neutral";
const DEFAULT_MODEL = "voxtral-mini-tts-2603";

export async function POST(request: Request) {
  const jsonResponse = (
    body: { error?: string; audio?: string; format?: string; provider?: string; playback_rate?: number },
    status: number
  ) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  try {
    const apiKey = process.env.MISTRAL_API_KEY?.trim();
    if (!apiKey) {
      return jsonResponse({ error: "MISTRAL_API_KEY is not configured" }, 500);
    }

    let body: { text?: string; voiceId?: string };
    try {
      body = (await request.json()) as typeof body;
    } catch {
      return jsonResponse({ error: "Invalid JSON in request body" }, 400);
    }

    const text = body?.text;
    if (!text || typeof text !== "string" || !text.trim()) {
      return jsonResponse({ error: "text is required" }, 400);
    }

    const inp = text.trim().slice(0, 12_000);
    const voiceId =
      (body?.voiceId && String(body.voiceId).trim()) ||
      process.env.MISTRAL_TTS_VOICE_ID?.trim() ||
      DEFAULT_VOICE;
    const model = process.env.MISTRAL_TTS_MODEL?.trim() || DEFAULT_MODEL;
    let fmt = (process.env.MISTRAL_TTS_RESPONSE_FORMAT || "opus").toLowerCase();
    if (!["opus", "mp3", "wav", "flac", "pcm"].includes(fmt)) fmt = "opus";

    const res = await fetch(MISTRAL_SPEECH_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        input: inp,
        voice_id: voiceId,
        response_format: fmt,
      }),
    });

    const rawText = await res.text();
    let data: { audio_data?: string; detail?: unknown; message?: string } = {};
    if (rawText.trim()) {
      try {
        data = JSON.parse(rawText) as typeof data;
      } catch {
        return jsonResponse({ error: rawText.slice(0, 200) || `Mistral TTS failed (${res.status})` }, 500);
      }
    }

    if (!res.ok) {
      const d = data.detail;
      const msg =
        typeof d === "string"
          ? d
          : d && typeof d === "object" && "message" in d
            ? String((d as { message?: string }).message)
            : data.message || rawText.slice(0, 200) || `Mistral TTS failed (${res.status})`;
      return jsonResponse({ error: msg }, 500);
    }

    const audioB64 = data.audio_data;
    if (!audioB64 || typeof audioB64 !== "string") {
      return jsonResponse({ error: "Mistral TTS returned no audio_data" }, 500);
    }

    let playbackRate = 1.05;
    try {
      const r = parseFloat(process.env.MISTRAL_TTS_PLAYBACK_RATE || "1.05");
      if (!Number.isNaN(r) && r >= 0.25 && r <= 4) playbackRate = r;
    } catch {
      /* keep default */
    }

    return jsonResponse(
      {
        audio: audioB64,
        format: fmt,
        provider: "mistral",
        playback_rate: playbackRate,
      },
      200
    );
  } catch (err) {
    console.error("[voice] Error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return jsonResponse({ error: message }, 500);
  }
}
