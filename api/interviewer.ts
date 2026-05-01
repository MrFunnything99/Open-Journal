/// <reference types="node" />

const TINFOIL_CHAT_URL = "https://inference.tinfoil.sh/v1/chat/completions";
const INTERVIEWER_MODEL = process.env.TINFOIL_INTERVIEWER_MODEL?.trim() || process.env.TINFOIL_CHAT_MODEL?.trim() || "kimi-k2-6";

type ChatMessage = { role: "user" | "ai"; text: string };

function assistantText(data: Record<string, unknown>): string {
  const choices = data.choices as Array<{ message?: { content?: unknown } }> | undefined;
  const content = choices?.[0]?.message?.content;
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) return content.map((b) => (b && typeof b === "object" && "text" in b ? String((b as { text?: unknown }).text ?? "") : "")).join("").trim();
  return "";
}

export async function POST(request: Request) {
  const jsonResponse = (body: { error?: string; question?: string }, status: number) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json; charset=utf-8" } });

  try {
    const apiKey = process.env.TINFOIL_API_KEY?.trim();
    if (!apiKey) return jsonResponse({ error: "TINFOIL_API_KEY is not configured" }, 500);

    let body: { systemPrompt?: string; messages?: ChatMessage[] };
    try {
      body = (await request.json()) as { systemPrompt?: string; messages?: ChatMessage[] };
    } catch {
      return jsonResponse({ error: "Invalid JSON in request body" }, 400);
    }

    const { systemPrompt, messages } = body;
    if (!systemPrompt || typeof systemPrompt !== "string") return jsonResponse({ error: "systemPrompt is required" }, 400);
    if (!Array.isArray(messages) || messages.length === 0) return jsonResponse({ error: "messages array is required and must not be empty" }, 400);

    const chatMessages = [
      { role: "system" as const, content: systemPrompt },
      ...messages.map((m) => ({ role: m.role === "user" ? ("user" as const) : ("assistant" as const), content: m.text })),
    ];

    const res = await fetch(TINFOIL_CHAT_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: INTERVIEWER_MODEL, messages: chatMessages, max_tokens: 8192 }),
    });

    const rawText = await res.text();
    const data = rawText.trim() ? (JSON.parse(rawText) as Record<string, unknown>) : {};
    if (!res.ok) {
      const err = data.error as { message?: string } | string | undefined;
      const msg = typeof err === "string" ? err : err?.message || rawText.slice(0, 200) || `Tinfoil request failed (${res.status})`;
      return jsonResponse({ error: msg }, 500);
    }

    const text = assistantText(data);
    if (!text) return jsonResponse({ error: "No content generated" }, 500);
    return jsonResponse({ question: text }, 200);
  } catch (err) {
    console.error("[interviewer] Error:", err);
    return jsonResponse({ error: err instanceof Error ? err.message : "Unknown error" }, 500);
  }
}
