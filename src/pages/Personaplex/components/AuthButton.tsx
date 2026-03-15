import { FC, useCallback, useState } from "react";
import { createPortal } from "react-dom";
import {
  authLogin,
  authRegister,
  setStoredToken,
  type AuthResponse,
} from "../../../backendApi";

export type AuthUser = { username: string; user_id: number };

type AuthButtonProps = {
  user: AuthUser | null;
  onUserChange: (user: AuthUser | null) => void;
  className?: string;
};

export const AuthButton: FC<AuthButtonProps> = ({
  user,
  onUserChange,
  className = "",
}) => {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError("");
      if (!username.trim() || !password) {
        setError("Username and password required.");
        return;
      }
      if (password.length < 6) {
        setError("Password must be at least 6 characters.");
        return;
      }
      setLoading(true);
      try {
        const res: AuthResponse =
          mode === "login"
            ? await authLogin(username.trim(), password)
            : await authRegister(username.trim(), password);
        setStoredToken(res.token);
        onUserChange({ username: res.username, user_id: res.user_id });
        setOpen(false);
        setUsername("");
        setPassword("");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      } finally {
        setLoading(false);
      }
    },
    [mode, username, password, onUserChange]
  );

  const handleLogout = useCallback(() => {
    setStoredToken(null);
    onUserChange(null);
    setOpen(false);
  }, [onUserChange]);

  if (user) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <span className="text-sm text-slate-400 truncate max-w-[120px]" title={user.username}>
          {user.username}
        </span>
        <button
          type="button"
          onClick={handleLogout}
          className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-300 hover:bg-slate-700/50 transition-colors"
        >
          Log out
        </button>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => {
          setOpen((o) => !o);
          setError("");
          setMode("login");
        }}
        className="px-3 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-300 hover:bg-slate-700/50 transition-colors border border-slate-600/50"
      >
        Log in
      </button>
      {open &&
        createPortal(
          <>
            <div
              className="fixed inset-0 z-[100]"
              aria-hidden
              onClick={() => setOpen(false)}
            />
            <div
              className="fixed left-1/2 top-24 z-[101] w-72 -translate-x-1/2 rounded-xl bg-slate-800 border border-slate-600 shadow-xl p-4"
              role="dialog"
              aria-label="Log in or register"
            >
              <p className="text-xs text-slate-500 mb-3">
                Log in to keep your data. No email. Without logging in, data is forgotten after 1 hour.
              </p>
              <form onSubmit={handleSubmit} className="space-y-3">
                <input
                  type="text"
                  placeholder="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-600 text-slate-200 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-600 text-slate-200 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
                {error && (
                  <p className="text-xs text-red-400">{error}</p>
                )}
                <div className="flex gap-2">
                  <button
                    type="submit"
                    disabled={loading}
                    className="flex-1 px-3 py-2 rounded-lg bg-violet-600 text-white text-sm font-medium hover:bg-violet-500 disabled:opacity-50"
                  >
                    {loading ? "..." : mode === "login" ? "Log in" : "Register"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setMode((m) => (m === "login" ? "register" : "login"));
                      setError("");
                    }}
                    className="px-3 py-2 rounded-lg text-slate-400 text-sm hover:text-slate-300"
                  >
                    {mode === "login" ? "Register" : "Log in"}
                  </button>
                </div>
              </form>
            </div>
          </>,
          document.body
        )}
    </div>
  );
};
