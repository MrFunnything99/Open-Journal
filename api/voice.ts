/// <reference types="node" />

const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech";
const DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // Rachel
const DEFAULT_MODEL = "eleven_multilingual_v2";
const OUTPUT_FORMAT = "mp3_44100_128";

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

export async function POST(request: Request) {
  const jsonResponse = (body: { error?: string; audio?: string }, status: number) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  try {
    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) {
      return jsonResponse({ error: "ELEVENLABS_API_KEY is not configured" }, 500);
    }

    type VoiceSettings = { stability?: number; similarity_boost?: number; style?: number; speed?: number };
    let body: { text?: string; voiceId?: string; voice_settings?: VoiceSettings };
    try {
      body = (await request.json()) as typeof body;
    } catch {
      return jsonResponse({ error: "Invalid JSON in request body" }, 400);
    }

    const text = body?.text;
    if (!text || typeof text !== "string" || !text.trim()) {
      return jsonResponse({ error: "text is required" }, 400);
    }

    const voiceId = body?.voiceId ?? DEFAULT_VOICE_ID;

    const raw = body?.voice_settings ?? {};
    const stability = clamp(
      typeof raw.stability === "number" ? raw.stability : 0.5,
      0,
      1
    );
    const similarity_boost = clamp(
      typeof raw.similarity_boost === "number" ? raw.similarity_boost : 0.75,
      0,
      1
    );
    const style = clamp(typeof raw.style === "number" ? raw.style : 0,
      0,
      1
    );
    const speed = clamp(
      typeof raw.speed === "number" ? raw.speed : 1,
      0.5,
      2
    );

    const res = await fetch(`${ELEVENLABS_API_URL}/${voiceId}`, {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      body: JSON.stringify({
        text: text.trim(),
        model_id: DEFAULT_MODEL,
        output_format: OUTPUT_FORMAT,
        voice_settings: { stability, similarity_boost, style, speed },
      }),
    });

    if (!res.ok) {
      const rawText = await res.text();
      let errMsg = `ElevenLabs request failed (${res.status})`;
      if (rawText.trim()) {
        try {
          const err = JSON.parse(rawText) as { detail?: { message?: string }; message?: string };
          errMsg = err.detail?.message ?? err.message ?? rawText.slice(0, 200);
        } catch {
          errMsg = rawText.slice(0, 200);
        }
      }
      return jsonResponse({ error: errMsg }, 500);
    }

    const arrayBuffer = await res.arrayBuffer();
    const base64 = Buffer.from(arrayBuffer).toString("base64");

    return jsonResponse({ audio: base64, format: "mp3" }, 200);
  } catch (err) {
    console.error("[voice] Error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return jsonResponse({ error: message }, 500);
  }
}
