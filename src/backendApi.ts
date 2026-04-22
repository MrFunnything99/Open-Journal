/**
 * API client for monolith: same-origin /api/*, two-token auth (access in header, refresh in cookie).
 * On 401, silently call /api/refresh and retry the request.
 */
const API_BASE = ""; // same origin; paths get /api prefix below
const TOKEN_KEY = "open_journal_access_token";
const ANON_ID_KEY = "open_journal_anon_id";
const ANON_TS_KEY = "open_journal_anon_ts";
/** Keep stable across tabs and restarts so /ingest-history and /chat share the same vec-store namespace. */
const ANON_TTL_MS = 10 * 365 * 24 * 60 * 60 * 1000; // 10 years (effectively permanent until user clears site data)

function apiPath(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  return path.startsWith("http") ? path : `${API_BASE}/api${p}`;
}

export function getBackendUrl(): string {
  if (typeof window === "undefined") return "";
  return window.location.origin;
}

export function getStoredToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setStoredToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {}
}

/** Decode JWT payload (for display only). Supports access token with email or username. */
export function decodeJwtPayload(token: string): { sub?: number; email?: string; username?: string } | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const raw = atob(parts[1].replace(/-/g, "+").replace(/_/g, "/"));
    const payload = JSON.parse(raw) as { sub?: number; email?: string; username?: string };
    return payload?.sub != null ? payload : null;
  } catch {
    return null;
  }
}

export function getAnonymousInstanceId(): string {
  try {
    const read = (): { id: string | null; ts: string | null } => ({
      id: localStorage.getItem(ANON_ID_KEY),
      ts: localStorage.getItem(ANON_TS_KEY),
    });
    let { id, ts } = read();
    // One-time migration from sessionStorage (older builds used per-tab ids + short TTL).
    if (!id) {
      const sid = sessionStorage.getItem(ANON_ID_KEY);
      const sts = sessionStorage.getItem(ANON_TS_KEY);
      if (sid) {
        localStorage.setItem(ANON_ID_KEY, sid);
        localStorage.setItem(ANON_TS_KEY, sts || String(Date.now()));
        try {
          sessionStorage.removeItem(ANON_ID_KEY);
          sessionStorage.removeItem(ANON_TS_KEY);
        } catch {
          /* ignore */
        }
        id = sid;
        ts = sts || String(Date.now());
      }
    }
    const t = ts ? parseInt(ts, 10) : 0;
    if (id && !isNaN(t) && Date.now() - t < ANON_TTL_MS) return id;
    const newId = crypto.randomUUID?.() ?? `anon_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
    localStorage.setItem(ANON_ID_KEY, newId);
    localStorage.setItem(ANON_TS_KEY, String(Date.now()));
    return newId;
  } catch {
    return "";
  }
}

/** Call /api/refresh to get new access token; returns new token or null. */
async function refreshAccessToken(): Promise<string | null> {
  const res = await fetch(apiPath("/refresh"), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) return null;
  const data = await res.json().catch(() => null);
  const token = data?.access_token ?? null;
  if (token) setStoredToken(token);
  return token;
}

/**
 * Fetch with access token. On 401, tries /api/refresh once and retries the request.
 */
export async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = apiPath(path);
  const headers = new Headers(init?.headers);
  const token = getStoredToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  /** Memory (/chat retrieval, Brain, journals) is keyed by X-Instance-ID; send it whenever we have one, even when logged in. */
  const anonId = getAnonymousInstanceId();
  if (anonId) headers.set("X-Instance-ID", anonId);

  let res = await fetch(url, { ...init, headers, credentials: "include" } as RequestInit);

  if (res.status === 401) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      headers.set("Authorization", `Bearer ${newToken}`);
      res = await fetch(url, { ...init, headers, credentials: "include" });
    } else {
      setStoredToken(null);
    }
  }
  return res;
}

// --- Two-token auth (email + password) ---
export type ApiAuthResponse = { access_token: string; user_id: number; email: string };

export async function apiRegister(email: string, password: string): Promise<ApiAuthResponse> {
  const res = await fetch(apiPath("/register"), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Register failed (${res.status})`);
  const out = data as ApiAuthResponse;
  setStoredToken(out.access_token);
  return out;
}

export async function apiLogin(email: string, password: string): Promise<ApiAuthResponse> {
  const res = await fetch(apiPath("/login"), {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Login failed (${res.status})`);
  const out = data as ApiAuthResponse;
  setStoredToken(out.access_token);
  return out;
}

export async function apiLogout(): Promise<void> {
  try {
    await fetch(apiPath("/logout"), { method: "POST", credentials: "include" });
  } catch {
    /* ignore */
  }
  setStoredToken(null);
}

/** Current user from access token or /api/me. */
export async function authMe(): Promise<{ user_id: number; email: string; username?: string } | null> {
  const token = getStoredToken();
  if (!token) return null;
  const res = await backendFetch("/me", { credentials: "include" });
  if (!res.ok) {
    if (res.status === 401) setStoredToken(null);
    return null;
  }
  const data = await res.json().catch(() => null);
  if (!data?.user_id) return null;
  return {
    user_id: data.user_id,
    email: data.email ?? "",
    username: data.username ?? data.email ?? "",
  };
}

export type SemanticMemoryCategory = "book" | "podcast" | "research_article";

export type SemanticConsumedItem = {
  id: number;
  category: SemanticMemoryCategory;
  title: string;
  creator_or_source: string;
  notes: string;
  consumed_on: string | null;
  created_at: string;
  updated_at: string;
};

export async function apiSemanticConsumedList(category?: SemanticMemoryCategory): Promise<SemanticConsumedItem[]> {
  const qs = category ? `?category=${encodeURIComponent(category)}` : "";
  const res = await backendFetch(`/semantic-memory/consumed${qs}`);
  if (!res.ok) throw new Error(`Failed to load semantic memory (${res.status})`);
  const data = (await res.json().catch(() => ({}))) as { items?: SemanticConsumedItem[] };
  return Array.isArray(data.items) ? data.items : [];
}

export async function apiSemanticConsumedCreate(payload: {
  category: SemanticMemoryCategory;
  title: string;
  creator_or_source?: string;
  notes?: string;
  consumed_on?: string | null;
}): Promise<boolean> {
  const res = await backendFetch("/semantic-memory/consumed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.ok;
}

export async function apiSemanticConsumedUpdate(
  id: number,
  payload: {
    category: SemanticMemoryCategory;
    title: string;
    creator_or_source?: string;
    notes?: string;
    consumed_on?: string | null;
  }
): Promise<boolean> {
  const res = await backendFetch(`/semantic-memory/consumed/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.ok;
}

export async function apiSemanticConsumedDelete(id: number): Promise<boolean> {
  const res = await backendFetch(`/semantic-memory/consumed/${id}`, {
    method: "DELETE",
  });
  return res.ok;
}
