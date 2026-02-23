/**
 * Local API server for development. Proxies /api/* to our serverless handlers.
 * Vite proxies /api to http://localhost:3001 when running npm run dev.
 */
import { createServer } from "node:http";
import { config } from "dotenv";

config({ path: ".env" });

const PORT = Number(process.env.API_PORT ?? process.env.PORT ?? 3001);

function getBody(req: import("node:http").IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function handleRequest(
  req: import("node:http").IncomingMessage,
  res: import("node:http").ServerResponse
): Promise<void> {
  const url = req.url ?? "/";
  const method = req.method ?? "GET";

  const pathname = url.split("?")[0];
  if (pathname === "/api/interviewer" && method === "POST") {
    try {
      const { POST } = await import("../api/interviewer");
      const body = await getBody(req);
      const request = new Request(`http://localhost${url}`, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body || undefined,
      });
      const response = await POST(request);
      res.writeHead(response.status, {
        "Content-Type": response.headers.get("Content-Type") ?? "application/json",
      });
      res.end(await response.text());
    } catch (err) {
      console.error("[api-server] /api/interviewer error:", err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  if (pathname === "/api/voice" && method === "POST") {
    try {
      const { POST } = await import("../api/voice");
      const body = await getBody(req);
      const request = new Request(`http://localhost${url}`, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body || undefined,
      });
      const response = await POST(request);
      res.writeHead(response.status, {
        "Content-Type": response.headers.get("Content-Type") ?? "application/json",
      });
      res.end(await response.text());
    } catch (err) {
      console.error("[api-server] /api/voice error:", err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  if (pathname === "/api/voices" && method === "GET") {
    try {
      const { GET } = await import("../api/voices");
      const request = new Request(`http://localhost${url}`);
      const response = await GET(request);
      res.writeHead(response.status, {
        "Content-Type": response.headers.get("Content-Type") ?? "application/json",
      });
      res.end(await response.text());
    } catch (err) {
      console.error("[api-server] /api/voices error:", err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  if (pathname === "/api/scribe-token" && method === "GET") {
    try {
      const { GET } = await import("../api/scribe-token");
      const request = new Request(`http://localhost${url}`);
      const response = await GET(request);
      res.writeHead(response.status, {
        "Content-Type": response.headers.get("Content-Type") ?? "application/json",
      });
      res.end(await response.text());
    } catch (err) {
      console.error("[api-server] /api/scribe-token error:", err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  if (pathname === "/api/transcribe" && method === "POST") {
    try {
      const { POST } = await import("../api/transcribe");
      const body = await getBody(req);
      const request = new Request(`http://localhost${url}`, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body || undefined,
      });
      const response = await POST(request);
      res.writeHead(response.status, {
        "Content-Type": response.headers.get("Content-Type") ?? "application/json",
      });
      res.end(await response.text());
    } catch (err) {
      console.error("[api-server] /api/transcribe error:", err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  if (pathname === "/api/reformat" && method === "POST") {
    try {
      const { POST } = await import("../api/reformat");
      const body = await getBody(req);
      const request = new Request(`http://localhost${url}`, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body || undefined,
      });
      const response = await POST(request);
      res.writeHead(response.status, {
        "Content-Type": response.headers.get("Content-Type") ?? "application/json",
      });
      res.end(await response.text());
    } catch (err) {
      console.error("[api-server] /api/reformat error:", err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not found" }));
}

const server = createServer(async (req, res) => {
  try {
    await handleRequest(req, res);
  } catch (err) {
    console.error("[api-server] Error:", err);
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Internal server error" }));
  }
});

server.on("error", (err: NodeJS.ErrnoException) => {
  if (err.code === "EADDRINUSE") {
    console.error(`[api-server] Port ${PORT} is in use. Set API_PORT=3002 in .env and VITE_API_URL=http://localhost:3002`);
  }
});

server.listen(PORT, () => {
  console.log(`[api-server] API routes at http://localhost:${PORT}/api/*`);
});
