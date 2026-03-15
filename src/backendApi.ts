/**
 * Backend API: base URL, auth (optional login), and instance scoping.
 * - Logged in: JWT sent, backend uses user_id → data persists.
 * - Anonymous: no JWT; ephemeral instance ID (1hr TTL) → data forgotten after 1 hour.
 */
const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

const TOKEN_KEY = "open_journal_token";
const ANON_ID_KEY = "open_journal_anon_id";
const ANON_TS_KEY = "open_journal_anon_ts";
const ANON_TTL_MS = 60 * 60 * 1000; // 1 hour

export function getBackendUrl(): string {
  return BACKEND_URL;
}

/** Return stored JWT or null. */
export function getStoredToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

/** Decode JWT payload without verification (for optimistic display only). Returns sub + username if present. */
export function decodeJwtPayload(token: string): { sub?: number; username?: string } | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const raw = atob(parts[1].replace(/-/g, "+").replace(/_/g, "/"));
    const payload = JSON.parse(raw) as { sub?: number; username?: string };
    return payload?.sub != null ? { sub: payload.sub, username: payload.username ?? "" } : null;
  } catch {
    return null;
  }
}

/** Store token after login/register; clear on logout. */
export function setStoredToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {}
}

/** Ephemeral instance ID for anonymous users: same for 1 hour, then new (data effectively forgotten). */
export function getAnonymousInstanceId(): string {
  try {
    const id = sessionStorage.getItem(ANON_ID_KEY);
    const ts = sessionStorage.getItem(ANON_TS_KEY);
    const t = ts ? parseInt(ts, 10) : 0;
    if (id && !isNaN(t) && Date.now() - t < ANON_TTL_MS) {
      return id;
    }
    const newId = crypto.randomUUID?.() ?? `anon_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
    sessionStorage.setItem(ANON_ID_KEY, newId);
    sessionStorage.setItem(ANON_TS_KEY, String(Date.now()));
    return newId;
  } catch {
    return "";
  }
}

/** Fetch from backend: Authorization if logged in, else X-Instance-ID (ephemeral 1hr for anonymous). */
export async function backendFetch(
  path: string,
  init?: RequestInit
): Promise<Response> {
  const url = path.startsWith("http") ? path : `${BACKEND_URL}${path}`;
  const headers = new Headers(init?.headers);
  const token = getStoredToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  } else {
    const instanceId = getAnonymousInstanceId();
    if (instanceId) headers.set("X-Instance-ID", instanceId);
  }
  return fetch(url, { ...init, headers });
}

/** Auth API: register and login return { token, user_id, username }. */
export type AuthResponse = { token: string; user_id: number; username: string };

export async function authRegister(username: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${BACKEND_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username.trim(), password }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Register failed (${res.status})`);
  return data as AuthResponse;
}

export async function authLogin(username: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${BACKEND_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username.trim(), password }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Login failed (${res.status})`);
  return data as AuthResponse;
}

export async function authMe(): Promise<{ user_id: number; username: string } | null> {
  const token = getStoredToken();
  if (!token) return null;
  const res = await fetch(`${BACKEND_URL}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 401) setStoredToken(null);
    return null;
  }
  const data = await res.json().catch(() => null);
  return data;
}
