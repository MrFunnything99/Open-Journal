/// <reference types="node" />

export async function GET() {
  return new Response(JSON.stringify({ voices: [], provider: "none" }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
