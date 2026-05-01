/// <reference types="node" />

export async function POST() {
  return new Response(JSON.stringify({ error: "Text-to-speech is unavailable in the Tinfoil-only build." }), {
    status: 501,
    headers: { "Content-Type": "application/json" },
  });
}
