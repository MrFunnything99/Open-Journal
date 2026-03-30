import { useCallback, useEffect, useRef, useState } from "react";
import { usePersonaplexChat } from "../PersonaplexChatContext";

/**
 * Desktop-only chat history column on the right edge of Home.
 *
 * Behaviour and width limits are ported from Open WebUI’s `Sidebar.svelte`
 * (see https://github.com/open-webui/open-webui — MIN_WIDTH / MAX_WIDTH, persisted width,
 * collapse vs expanded). This is a React implementation, not Svelte source.
 */
const MIN_WIDTH = 220;
const MAX_WIDTH = 480;
const WIDTH_STORAGE = "homeChatSidebarWidthPx";
const OPEN_STORAGE = "homeChatSidebarOpen";

type Props = {
  /** Only mounted on the Home (voice_memo) screen; hidden elsewhere. */
  active: boolean;
};

export function HomeChatSidebar({ active }: Props) {
  const { chatRecents, chatSessionId, loadRecentSession, newChat } = usePersonaplexChat();

  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem(OPEN_STORAGE) !== "false";
    } catch {
      return true;
    }
  });

  const [width, setWidth] = useState(() => {
    try {
      const n = Number(localStorage.getItem(WIDTH_STORAGE));
      if (!Number.isNaN(n) && n >= MIN_WIDTH && n <= MAX_WIDTH) return n;
    } catch {
      /* ignore */
    }
    return 260;
  });

  useEffect(() => {
    try {
      localStorage.setItem(OPEN_STORAGE, open ? "true" : "false");
    } catch {
      /* ignore */
    }
  }, [open]);

  useEffect(() => {
    try {
      localStorage.setItem(WIDTH_STORAGE, String(width));
    } catch {
      /* ignore */
    }
  }, [width]);

  const resizing = useRef(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(260);

  const onResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizing.current = true;
    startXRef.current = e.clientX;
    startWidthRef.current = width;
    document.body.style.userSelect = "none";
  }, [width]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!resizing.current) return;
      const dx = startXRef.current - e.clientX;
      const next = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidthRef.current + dx));
      setWidth(next);
    };
    const onUp = () => {
      if (!resizing.current) return;
      resizing.current = false;
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const startNewChat = useCallback(() => {
    newChat();
  }, [newChat]);

  if (!active) return null;

  return (
    <div className="hidden h-full shrink-0 md:flex">
      {!open && (
        <div
          className="flex h-full w-[52px] flex-col items-center gap-2 border-l border-white/10 bg-[#0c0c12]/95 py-3 backdrop-blur-xl"
          aria-label="Chat sidebar collapsed"
        >
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="rounded-lg p-2.5 text-white/70 transition hover:bg-white/10 hover:text-white"
            title="Open sidebar"
            aria-label="Open sidebar"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          </button>
          <button
            type="button"
            onClick={startNewChat}
            className="rounded-lg p-2.5 text-white/70 transition hover:bg-white/10 hover:text-white"
            title="New chat"
            aria-label="New chat"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </div>
      )}

      {open && (
        <aside
          style={{ width }}
          className="relative flex h-full min-w-0 flex-col border-l border-white/10 bg-[#0c0c12]/95 text-white backdrop-blur-xl"
          aria-label="Chat history"
        >
          <div
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize sidebar"
            className="absolute left-0 top-0 z-10 h-full w-1.5 -translate-x-1/2 cursor-col-resize hover:bg-white/15"
            onMouseDown={onResizeStart}
          />

          <div className="flex flex-none items-center gap-1 border-b border-white/10 px-2 py-2 pl-3">
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-lg p-2 text-white/65 transition hover:bg-white/10 hover:text-white"
              title="Close sidebar"
              aria-label="Close sidebar"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
              </svg>
            </button>
            <span className="min-w-0 flex-1 truncate px-1 text-xs font-semibold uppercase tracking-[0.14em] text-white/55">
              Chats
            </span>
          </div>

          <div className="flex-none p-2">
            <button
              type="button"
              onClick={startNewChat}
              className="flex w-full items-center gap-2 rounded-xl border border-white/[0.14] bg-white/[0.07] px-3 py-2.5 text-left text-sm font-medium text-white shadow-sm transition hover:bg-white/[0.11]"
            >
              <svg className="h-5 w-5 shrink-0 text-white/85" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
              </svg>
              New chat
            </button>
          </div>

          <nav className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain px-2 pb-3" aria-label="Recent chats">
            {chatRecents.length === 0 ? (
              <p className="px-2 py-3 text-xs leading-relaxed text-white/40">No chats yet. Send a message to create one.</p>
            ) : (
              <ul className="space-y-0.5">
                {chatRecents.map((r) => {
                  const isActive = chatSessionId === r.sessionId;
                  return (
                    <li key={r.sessionId}>
                      <button
                        type="button"
                        onClick={() => loadRecentSession(r.sessionId)}
                        className={`w-full truncate rounded-lg px-2.5 py-2 text-left text-[0.8125rem] transition ${
                          isActive
                            ? "bg-white/[0.14] font-medium text-white ring-1 ring-white/12"
                            : "text-white/75 hover:bg-white/[0.08] hover:text-white"
                        }`}
                      >
                        {r.title}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </nav>
        </aside>
      )}
    </div>
  );
}
