/// <reference types="node" />

const TINFOIL_CHAT_URL = "https://inference.tinfoil.sh/v1/chat/completions";
const NARRATOR_MODEL = process.env.TINFOIL_NARRATOR_MODEL?.trim() || process.env.TINFOIL_CHAT_MODEL?.trim() || "kimi-k2-6";

const SYSTEM_PROMPT = `You are an expert editor. You will be given a transcript of a conversation between a User and a Selfmeridian assistant.
TASK: Rewrite this conversation into a single, beautifully formatted, cohesive, first-person journal entry written from the User's perspective. The output should read like a polished diary entry.
RULES:
- Remove all AI questions and filler.
- Merge the User's answers into a flowing narrative.
- Capture the emotional tone.
- Do not add fictional details, but you can smooth out transitions.
- The output should look like a diary entry starting with 'Today...'`;

type ChatMessage = { role: "user" | "ai"; text: string };

function formatTranscriptForPrompt(messages: ChatMessage[]): string {
  return messages.map((m) => `${m.role === "user" ? "User" : "AI"}: ${m.text}`).join("\n\n");
}

export async function POST(request: Request) {
  const jsonResponse = (body: { error?: string; text?: string }, status: number) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

  try {
    const apiKey = process.env.TINFOIL_API_KEY?.trim();
    if (!apiKey) return jsonResponse({ error: "TINFOIL_API_KEY is not configured" }, 500);

    const body = (await request.json()) as { messages?: ChatMessage[] };
    const messages = body.messages;
    if (!Array.isArray(messages) || messages.length === 0) return jsonResponse({ error: "messages array is required and must not be empty" }, 400);

    const res = await fetch(TINFOIL_CHAT_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: NARRATOR_MODEL,
        max_tokens: 8192,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: `Here is the conversation transcript to rewrite:\n\n${formatTranscriptForPrompt(messages)}` },
        ],
      }),
    });

    const data = (await res.json()) as { error?: { message?: string } | string; choices?: Array<{ message?: { content?: string } }> };
    if (!res.ok) {
      const err = data.error;
      return jsonResponse({ error: typeof err === "string" ? err : err?.message || "Tinfoil request failed" }, res.status);
    }
    const text = data.choices?.[0]?.message?.content;
    if (!text || typeof text !== "string") return jsonResponse({ error: "No content generated" }, 500);
    return jsonResponse({ text: text.trim() }, 200);
  } catch (err) {
    console.error("[reformat] Error:", err);
    return jsonResponse({ error: err instanceof Error ? err.message : "Unknown error" }, 500);
  }
}
