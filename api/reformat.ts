/// <reference types="node" />

const OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions";
const NARRATOR_MODEL = "google/gemini-3.1-pro-preview";

const SYSTEM_PROMPT = `You are an expert editor. You will be given a transcript of a conversation between a User and an OpenJournal Assistant.
TASK: Rewrite this conversation into a single, beautifully formatted, cohesive, first-person journal entry written from the User's perspective. The output should read like a polished diary entry.
RULES:
- Remove all AI questions and filler.
- Merge the User's answers into a flowing narrative.
- Capture the emotional tone (e.g., if the user was angry, write the entry with that emotion).
- Do not add fictional details, but you can smooth out transitions.
- The output should look like a diary entry starting with 'Today...'`;

type ChatMessage = { role: "user" | "ai"; text: string };

function formatTranscriptForPrompt(messages: ChatMessage[]): string {
  return messages
    .map((m) => `${m.role === "user" ? "User" : "AI"}: ${m.text}`)
    .join("\n\n");
}

export async function POST(request: Request) {
  try {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      return new Response(
        JSON.stringify({ error: "OPENROUTER_API_KEY is not configured" }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    const body = await request.json();
    const messages: ChatMessage[] = body.messages;

    if (!Array.isArray(messages) || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: "messages array is required and must not be empty" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const transcript = formatTranscriptForPrompt(messages);

    const res = await fetch(OPENROUTER_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: NARRATOR_MODEL,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          {
            role: "user",
            content: `Here is the conversation transcript to rewrite:\n\n${transcript}`,
          },
        ],
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      const errMsg = data.error?.message ?? data.error ?? "OpenRouter request failed";
      return new Response(
        JSON.stringify({ error: errMsg }),
        { status: res.status, headers: { "Content-Type": "application/json" } }
      );
    }

    const text = data.choices?.[0]?.message?.content;
    if (!text || typeof text !== "string") {
      return new Response(
        JSON.stringify({ error: "No content generated" }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    return new Response(
      JSON.stringify({ text: text.trim() }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } catch (err) {
    console.error("[reformat] Error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return new Response(
      JSON.stringify({ error: message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
