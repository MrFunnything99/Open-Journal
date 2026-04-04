/// <reference types="node" />

import FormData from "form-data";
import https from "node:https";

const OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions";
const DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe";
const OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL = "openai/gpt-audio-mini";

function openRouterFormatFromHint(hint: string | undefined): string {
  const h = (hint || "").trim().toLowerCase().replace(/^\./, "");
  const allowed = new Set(["wav", "mp3", "aac", "ogg", "flac", "m4a", "aiff", "pcm16", "pcm24", "webm"]);
  if (allowed.has(h)) return h;
  return "wav";
}

function assistantTextFromCompletion(data: unknown): string {
  const d = data as { choices?: Array<{ message?: { content?: unknown } }> };
  const content = d?.choices?.[0]?.message?.content;
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "object" && block !== null && (block as { type?: string }).type === "text") {
          return String((block as { text?: string }).text ?? "");
        }
        return "";
      })
      .join("")
      .trim();
  }
  return "";
}

export async function POST(request: Request) {
  const jsonResponse = (body: { error?: string; text?: string }, status: number) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });

  try {
    const openrouterKey = process.env.OPENROUTER_API_KEY?.trim();
    const openaiKey = process.env.OPENAI_API_KEY?.trim();
    if (!openrouterKey && !openaiKey) {
      return jsonResponse(
        {
          error:
            "Configure OPENROUTER_API_KEY (openai/gpt-audio-mini STT) or OPENAI_API_KEY.",
        },
        500
      );
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

    if (openrouterKey) {
      const model =
        process.env.OPENROUTER_TRANSCRIPTION_MODEL?.trim() || DEFAULT_OPENROUTER_TRANSCRIPTION_MODEL;
      const format = openRouterFormatFromHint(body?.format);
      const payload = JSON.stringify({
        model,
        temperature: 0,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Transcribe this audio verbatim. Reply with the transcript only — no preamble or quotes.",
              },
              {
                type: "input_audio",
                input_audio: {
                  data: buffer.toString("base64"),
                  format,
                },
              },
            ],
          },
        ],
      });
      const res = await fetch(OPENROUTER_CHAT_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${openrouterKey}`,
          "Content-Type": "application/json",
          "HTTP-Referer": process.env.OPENROUTER_REFERER || "https://selfmeridian.local",
          "X-Title": process.env.OPENROUTER_TITLE || "SelfMeridian",
        },
        body: payload,
      });
      const rawText = await res.text();
      let data: Record<string, unknown> = {};
      if (rawText.trim()) {
        try {
          data = JSON.parse(rawText) as Record<string, unknown>;
        } catch {
          console.error("[transcribe] OpenRouter returned non-JSON:", rawText.slice(0, 300));
        }
      }
      if (!res.ok) {
        const errObj = data?.error as { message?: string } | string | undefined;
        const errMsg =
          (typeof errObj === "object" && errObj?.message) ??
          (typeof errObj === "string" ? errObj : null) ??
          (data?.message as string) ??
          (rawText.trim() ? rawText.slice(0, 400) : null) ??
          `OpenRouter transcription failed (${res.status})`;
        console.error("[transcribe] OpenRouter error:", res.status, errMsg);
        return jsonResponse({ error: String(errMsg) }, 500);
      }
      const transcript = assistantTextFromCompletion(data);
      return jsonResponse({ text: transcript }, 200);
    }

    const model = process.env.OPENAI_TRANSCRIPTION_MODEL?.trim() || DEFAULT_OPENAI_TRANSCRIPTION_MODEL;
    const form = new FormData();
    form.append("model", model);
    form.append("file", buffer, { filename: "audio.wav", contentType: "audio/wav" });

    const { rawText, statusCode } = await new Promise<{ rawText: string; statusCode: number }>(
      (resolve, reject) => {
        const url = new URL(OPENAI_TRANSCRIPTION_URL);
        const req = https.request(
          {
            hostname: url.hostname,
            path: url.pathname,
            method: "POST",
            headers: {
              Authorization: `Bearer ${openaiKey!}`,
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
        console.error("[transcribe] OpenAI returned non-JSON:", rawText.slice(0, 300));
      }
    }

    if (statusCode < 200 || statusCode >= 300) {
      const errObj = data?.error as { message?: string } | undefined;
      const errMsg =
        (typeof errObj === "object" && errObj?.message) ??
        (data?.message as string) ??
        (rawText.trim() ? rawText.slice(0, 300) : null) ??
        `OpenAI transcription failed (${statusCode})`;
      console.error("[transcribe] OpenAI error:", statusCode, errMsg);
      return jsonResponse({ error: String(errMsg) }, 500);
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
