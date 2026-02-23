/// <reference types="node" />

const OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions";
const INTERVIEWER_MODEL = "google/gemini-3.1-pro-preview";

type ChatMessage = { role: "user" | "ai"; text: string };

export async function POST(request: Request) {
  const jsonResponse = (body: { error?: string; question?: string }, status: number) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });

  try {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      return jsonResponse({ error: "OPENROUTER_API_KEY is not configured" }, 500);
    }

    let body: { systemPrompt?: string; messages?: ChatMessage[] };
    try {
      body = (await request.json()) as { systemPrompt?: string; messages?: ChatMessage[] };
    } catch {
      return jsonResponse({ error: "Invalid JSON in request body" }, 400);
    }
    const { systemPrompt, messages } = body as {
      systemPrompt: string;
      messages: ChatMessage[];
    };

    if (!systemPrompt || typeof systemPrompt !== "string") {
      return jsonResponse({ error: "systemPrompt is required" }, 400);
    }

    if (!Array.isArray(messages) || messages.length === 0) {
      return jsonResponse({ error: "messages array is required and must not be empty" }, 400);
    }

    const chatMessages = [
      { role: "system" as const, content: systemPrompt },
      ...messages.map((m) => ({
        role: m.role === "user" ? ("user" as const) : ("assistant" as const),
        content: m.text,
      })),
    ];

    const res = await fetch(OPENROUTER_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: INTERVIEWER_MODEL,
        messages: chatMessages,
        max_tokens: 4096,
      }),
    });

    const rawText = await res.text();
    let data: Record<string, unknown> = {};
    if (rawText.trim()) {
      try {
        data = JSON.parse(rawText) as Record<string, unknown>;
      } catch {
        console.error("[interviewer] OpenRouter returned non-JSON:", rawText.slice(0, 200));
      }
    }

    if (!res.ok) {
      const errMsg =
        (data?.error as { message?: string })?.message ??
        (data?.error as string) ??
        (rawText.trim() ? rawText.slice(0, 200) : null) ??
        `OpenRouter request failed (${res.status})`;
      return jsonResponse({ error: String(errMsg) }, 500);
    }

    const choices = data?.choices as Array<{
      message?: { content?: string | Array<{ type?: string; text?: string }> };
      finish_reason?: string;
    }> | undefined;
    const choice = choices?.[0];
    const finishReason = choice?.finish_reason;
    const content = choice?.message?.content;

    let text: string;
    if (typeof content === "string") {
      text = content;
    } else if (Array.isArray(content)) {
      text = content
        .map((block) => (block && typeof block.text === "string" ? block.text : ""))
        .join("");
    } else {
      return jsonResponse({ error: "No content generated" }, 500);
    }

    if (!text || !text.trim()) {
      return jsonResponse({ error: "No content generated" }, 500);
    }

    const trimmed = text.trim();
    if (finishReason === "length") {
      console.warn("[interviewer] Model hit token limit (finish_reason: length), response may be truncated");
    }
    console.log("[interviewer] Response:", { finish_reason: finishReason, length: trimmed.length, preview: trimmed.slice(0, 80) });

    return jsonResponse({ question: trimmed }, 200);
  } catch (err) {
    console.error("[interviewer] Error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return jsonResponse({ error: message }, 500);
  }
}
