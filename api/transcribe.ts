/// <reference types="node" />

import FormData from "form-data";
import https from "node:https";

const TINFOIL_TRANSCRIPTION_URL = "https://inference.tinfoil.sh/v1/audio/transcriptions";
const DEFAULT_TINFOIL_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo";

function filenameFromHint(hint: string | undefined): string {
  const h = (hint || "").trim().toLowerCase().replace(/^\./, "");
  if (h === "mp3") return "audio.mp3";
  return "audio.wav";
}

export async function POST(request: Request) {
  const jsonResponse = (body: { error?: string; text?: string }, status: number) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

  try {
    const apiKey = process.env.TINFOIL_API_KEY?.trim();
    if (!apiKey) return jsonResponse({ error: "Configure TINFOIL_API_KEY for transcription." }, 500);

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
    if (buffer.length < 100) return jsonResponse({ error: "Audio too short to transcribe" }, 400);

    const filename = filenameFromHint(body?.format);
    const model = process.env.TINFOIL_TRANSCRIPTION_MODEL?.trim() || DEFAULT_TINFOIL_TRANSCRIPTION_MODEL;
    const form = new FormData();
    form.append("model", model);
    form.append("file", buffer, {
      filename,
      contentType: filename.endsWith(".mp3") ? "audio/mpeg" : "audio/wav",
    });

    const { rawText, statusCode } = await new Promise<{ rawText: string; statusCode: number }>((resolve, reject) => {
      const url = new URL(TINFOIL_TRANSCRIPTION_URL);
      const req = https.request(
        {
          hostname: url.hostname,
          path: url.pathname,
          method: "POST",
          headers: { Authorization: `Bearer ${apiKey}`, ...form.getHeaders() },
        },
        (res) => {
          const chunks: Buffer[] = [];
          res.on("data", (chunk) => chunks.push(chunk));
          res.on("end", () => resolve({ rawText: Buffer.concat(chunks).toString("utf8"), statusCode: res.statusCode ?? 0 }));
          res.on("error", reject);
        }
      );
      req.on("error", reject);
      form.pipe(req);
    });

    let data: Record<string, unknown> = {};
    if (rawText.trim()) {
      try {
        data = JSON.parse(rawText) as Record<string, unknown>;
      } catch {
        if (statusCode >= 200 && statusCode < 300) return jsonResponse({ text: rawText.trim() }, 200);
      }
    }

    if (statusCode < 200 || statusCode >= 300) {
      const err = data.error as { message?: string } | string | undefined;
      const msg = typeof err === "string" ? err : err?.message || rawText.slice(0, 300) || `Tinfoil transcription failed (${statusCode})`;
      return jsonResponse({ error: msg }, 500);
    }

    const text = data.text;
    return jsonResponse({ text: text != null ? String(text).trim() : "" }, 200);
  } catch (err) {
    console.error("[transcribe] Error:", err);
    return jsonResponse({ error: err instanceof Error ? err.message : "Unknown error" }, 500);
  }
}
