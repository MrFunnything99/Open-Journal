/// <reference types="node" />

const MISTRAL_VOICES_URL = "https://api.mistral.ai/v1/audio/voices?limit=50&offset=0";

const FALLBACK_VOICES: Array<{ voice_id: string; name: string }> = [
  { voice_id: "en_paul_neutral", name: "Paul (neutral)" },
];

export async function GET() {
  const jsonResponse = (body: unknown, status: number) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  try {
    const apiKey = process.env.MISTRAL_API_KEY?.trim();
    if (!apiKey) {
      return jsonResponse({ voices: FALLBACK_VOICES, provider: "fallback" }, 200);
    }

    const res = await fetch(MISTRAL_VOICES_URL, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });

    if (!res.ok) {
      return jsonResponse({ voices: FALLBACK_VOICES, provider: "fallback" }, 200);
    }

    const data = (await res.json()) as { items?: Array<{ id?: string; name?: string }> };
    const items = data.items ?? [];
    const voices = items
      .filter((v) => v && typeof v.id === "string" && v.id.trim())
      .map((v) => ({ voice_id: v.id!.trim(), name: (v.name || "Voice").trim() || "Voice" }));

    if (voices.length === 0) {
      return jsonResponse({ voices: FALLBACK_VOICES, provider: "fallback" }, 200);
    }

    return jsonResponse({ voices, provider: "mistral" }, 200);
  } catch {
    return jsonResponse({ voices: FALLBACK_VOICES, provider: "fallback" }, 200);
  }
}
