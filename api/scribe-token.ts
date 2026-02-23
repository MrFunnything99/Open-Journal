/**
 * Returns a single-use token for ElevenLabs Realtime Speech-to-Text WebSocket.
 * Client uses this token to connect directly to wss://api.elevenlabs.io/v1/speech-to-text/realtime
 * without exposing the API key.
 */
const ELEVENLABS_TOKEN_URL = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe";

export async function GET() {
  const jsonResponse = (body: { error?: string; token?: string }, status: number) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  try {
    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) {
      return jsonResponse({ error: "ELEVENLABS_API_KEY is not configured" }, 500);
    }

    const res = await fetch(ELEVENLABS_TOKEN_URL, {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
      },
    });

    const rawText = await res.text();
    let data: { token?: string; detail?: { message?: string } } = {};
    if (rawText.trim()) {
      try {
        data = JSON.parse(rawText) as typeof data;
      } catch {
        console.error("[scribe-token] ElevenLabs returned non-JSON:", rawText.slice(0, 200));
      }
    }

    if (!res.ok) {
      const errMsg =
        (typeof data.detail === "object" && data.detail?.message) ??
        (rawText.trim() ? rawText.slice(0, 200) : null) ??
        `ElevenLabs token failed (${res.status})`;
      console.error("[scribe-token] ElevenLabs error:", res.status, errMsg);
      return jsonResponse({ error: errMsg }, 500);
    }

    const token = data.token;
    if (!token || typeof token !== "string") {
      return jsonResponse({ error: "No token in response" }, 500);
    }

    return jsonResponse({ token }, 200);
  } catch (err) {
    console.error("[scribe-token] Error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return jsonResponse({ error: message }, 500);
  }
}
