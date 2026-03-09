/// <reference types="node" />

import FormData from "form-data";
import https from "node:https";

const ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text";
const SCRIBE_MODEL = "scribe_v2";

export async function POST(request: Request) {
  const jsonResponse = (body: { error?: string; text?: string }, status: number) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  try {
    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) {
      return jsonResponse({ error: "ELEVENLABS_API_KEY is not configured" }, 500);
    }

    let body: { audio?: string; format?: string };
    try {
      body = (await request.json()) as { audio?: string; format?: string };
    } catch {
      return jsonResponse({ error: "Invalid JSON in request body" }, 400);
    }

    const audioBase64 = body?.audio;
    if (!audioBase64 || typeof audioBase64 !== "string" || !audioBase64.trim()) {
      return jsonResponse({ error: "audio (base64) is required" }, 400);
    }

    const buffer = Buffer.from(audioBase64.trim(), "base64");
    if (buffer.length < 100) {
      return jsonResponse({ error: "Audio too short to transcribe" }, 400);
    }

    const form = new FormData();
    form.append("file", buffer, { filename: "audio.wav", contentType: "audio/wav" });
    form.append("model_id", SCRIBE_MODEL);

    const { rawText, statusCode } = await new Promise<{ rawText: string; statusCode: number }>(
      (resolve, reject) => {
        const url = new URL(ELEVENLABS_STT_URL);
        const req = https.request(
          {
            hostname: url.hostname,
            path: url.pathname,
            method: "POST",
            headers: {
              "xi-api-key": apiKey,
              ...form.getHeaders(),
            },
          },
          (res) => {
            const chunks: Buffer[] = [];
            res.on("data", (chunk) => chunks.push(chunk));
            res.on("end", () =>
              resolve({
                rawText: Buffer.concat(chunks).toString("utf8"),
                statusCode: res.statusCode ?? 0,
              })
            );
            res.on("error", reject);
          }
        );
        req.on("error", reject);
        form.pipe(req);
      }
    );

    let data: Record<string, unknown> = {};
    if (rawText.trim()) {
      try {
        data = JSON.parse(rawText) as Record<string, unknown>;
      } catch {
        console.error("[transcribe] ElevenLabs returned non-JSON:", rawText.slice(0, 300));
      }
    }

    if (statusCode < 200 || statusCode >= 300) {
      const errDetail = data?.detail as { message?: string } | string | undefined;
      const errMsg =
        (typeof errDetail === "object" && errDetail?.message) ??
        (typeof errDetail === "string" ? errDetail : null) ??
        (data?.message as string) ??
        (rawText.trim() ? rawText.slice(0, 300) : null) ??
        `ElevenLabs STT failed (${statusCode})`;
      console.error("[transcribe] ElevenLabs error:", statusCode, errMsg);
      return jsonResponse({ error: errMsg }, 500);
    }

    const text = data?.text;
    const transcript = text != null ? String(text).trim() : "";

    return jsonResponse({ text: transcript }, 200);
  } catch (err) {
    console.error("[transcribe] Error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return jsonResponse({ error: message }, 500);
  }
}
