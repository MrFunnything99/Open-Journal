/// <reference types="node" />

const ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices";

/** Fallback premade voices if API fails */
const FALLBACK_VOICES: Array<{ voice_id: string; name: string }> = [
  { voice_id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel" },
  { voice_id: "pNInz6obpgDQGcFmaJgB", name: "Adam" },
  { voice_id: "EXAVITQu4vr4xnSDxMaL", name: "Bella" },
  { voice_id: "ErXwobaYiN019PkySvjV", name: "Antoni" },
  { voice_id: "MF3mGyEYCl7XYWbV9V6O", name: "Elli" },
  { voice_id: "TxGEqnHWrfWFTfGW9XjX", name: "Josh" },
  { voice_id: "VR6AewLTigWG4xSOukaG", name: "Arnold" },
  { voice_id: "onwK4e9ZLuTAKqWW03F9", name: "Domi" },
  { voice_id: "N2lVS1w4EtoT3dr4eOWO", name: "Sam" },
];

export async function GET() {
  const jsonResponse = (body: unknown, status: number) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  try {
    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) {
      return jsonResponse({ voices: FALLBACK_VOICES }, 200);
    }

    const res = await fetch(ELEVENLABS_VOICES_URL, {
      headers: { "xi-api-key": apiKey },
    });

    if (!res.ok) {
      return jsonResponse({ voices: FALLBACK_VOICES }, 200);
    }

    const data = (await res.json()) as {
      voices?: Array<{ voice_id?: string; id?: string; name: string }>;
    };
    const rawVoices = data.voices ?? FALLBACK_VOICES;
    const voices = rawVoices
      .map((v) => ({
        voice_id: v.voice_id ?? v.id ?? "",
        name: v.name,
      }))
      .filter((v) => v.voice_id);

    const hasRachel = voices.some((v) => v.voice_id === "21m00Tcm4TlvDq8ikWAM");
    const voicesWithDefault = hasRachel
      ? voices
      : [{ voice_id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel" }, ...voices];

    return jsonResponse({ voices: voicesWithDefault }, 200);
  } catch {
    return jsonResponse({ voices: FALLBACK_VOICES }, 200);
  }
}
