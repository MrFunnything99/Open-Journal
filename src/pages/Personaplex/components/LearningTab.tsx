import { FC, useCallback, useEffect, useRef, useState } from "react";
import { backendFetch } from "../../../backendApi";
import { playChatReadAloud } from "../utils/chatReadAloud";
import { usePersonaplexChat } from "../PersonaplexChatContext";
import { AskAnythingComposer, LiveDictationBubble } from "./GlobalAskAnythingBar";

type Props = {
  onToast: (msg: string) => void;
};

type DailyArticle = {
  title: string;
  url: string;
  hook: string;
  snippet: string;
  date: string;
  status: string;
  has_article: boolean;
  themes?: string[];
  theme_notes?: { theme: string; notes: string }[];
  theme_model?: string;
  article_model?: string;
};

type TabPhase = "loading" | "article" | "reflection";

export const LearningTab: FC<Props> = ({ onToast }) => {
  const {
    messages,
    sending,
    setChatInteractionMode,
    newChat,
  } = usePersonaplexChat();

  const [phase, setPhase] = useState<TabPhase>("loading");
  const [article, setArticle] = useState<DailyArticle | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [regenerating, setRegenerating] = useState(false);
  const [readAloudBusy, setReadAloudBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const fetchArticle = useCallback(
    async (force: boolean) => {
      try {
        setLoadError(null);
        if (force) setRegenerating(true);
        else setPhase("loading");

        const endpoint = force ? "/learning/regenerate" : "/learning/today";
        const opts: RequestInit = force ? { method: "POST" } : {};
        const res = await backendFetch(endpoint, opts);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: DailyArticle = await res.json();
        setArticle(data);

        if (data.status === "reflected") {
          setPhase("reflection");
          setChatInteractionMode("learning");
        } else {
          setPhase("article");
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Failed to load article";
        setLoadError(msg);
        setPhase("article");
      } finally {
        setRegenerating(false);
      }
    },
    [setChatInteractionMode]
  );

  useEffect(() => {
    void fetchArticle(false);
  }, [fetchArticle]);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, sending]);

  const handleStartReflection = useCallback(() => {
    newChat();
    setChatInteractionMode("learning");
    setPhase("reflection");
    if (article) {
      void backendFetch("/learning/status", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "read" }),
      });
    }
  }, [newChat, setChatInteractionMode, article]);

  const handleNotInterested = useCallback(() => {
    if (regenerating) return;
    void fetchArticle(true);
  }, [fetchArticle, regenerating]);

  const readAloud = useCallback(
    (text: string) => {
      void playChatReadAloud(text, onToast, { onLoading: setReadAloudBusy });
    },
    [onToast],
  );

  const copyText = useCallback(
    async (text: string) => {
      try {
        await navigator.clipboard.writeText(text);
        onToast("Copied.");
      } catch {
        onToast("Could not copy.");
      }
    },
    [onToast]
  );

  const hasConversation = messages.length > 0 || sending;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-transparent">
      <div className="flex min-h-0 flex-1 flex-row">
        <div className="flex min-h-0 flex-1 flex-col">
          <div
            ref={listRef}
            className="relative flex min-h-0 flex-1 flex-col overflow-y-auto"
            role="log"
            aria-live="polite"
          >
            {/* Loading state */}
            {phase === "loading" && (
              <div className="flex flex-1 items-center justify-center">
                <div className="flex flex-col items-center gap-4 text-white/60">
                  <svg
                    className="h-8 w-8 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  <p className="text-sm">Finding today's article...</p>
                </div>
              </div>
            )}

            {/* Article card */}
            {phase === "article" && (
              <div className="flex flex-1 items-center justify-center px-4">
                <div className="w-full max-w-xl">
                  {loadError ? (
                    <div className="glass-panel rounded-2xl border border-red-500/30 px-6 py-8 text-center">
                      <p className="text-red-400">{loadError}</p>
                      <button
                        type="button"
                        onClick={() => void fetchArticle(false)}
                        className="mt-4 rounded-xl bg-white/10 px-5 py-2 text-sm font-medium text-white/80 transition hover:bg-white/15"
                      >
                        Retry
                      </button>
                    </div>
                  ) : article && !article.has_article ? (
                    <div className="glass-panel rounded-2xl border border-white/10 px-6 py-8 text-center">
                      <div className="mb-3 text-3xl">📚</div>
                      <p className="text-white/70">{article.hook || "No article available today."}</p>
                    </div>
                  ) : article ? (
                    <div className="glass-panel rounded-2xl border border-white/10 px-6 py-8">
                      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-white/40">
                        Today's read
                      </p>
                      <h2 className="mb-3 text-xl font-bold leading-snug text-white/95">
                        {article.title}
                      </h2>
                      <p className="mb-6 text-[0.95rem] leading-relaxed text-white/70">
                        {article.hook}
                      </p>
                      <div className="flex flex-wrap items-center gap-3">
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-2 rounded-xl bg-white/[0.12] px-5 py-2.5 text-sm font-medium text-white/90 transition hover:bg-white/20"
                        >
                          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                          Read Article
                        </a>
                        <button
                          type="button"
                          onClick={handleStartReflection}
                          className="inline-flex items-center gap-2 rounded-xl bg-indigo-600/80 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-indigo-600"
                        >
                          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                          </svg>
                          I've read it — reflect
                        </button>
                      </div>
                      <div className="mt-6 border-t border-white/10 pt-4 flex flex-wrap items-center justify-between gap-2">
                        <button
                          type="button"
                          onClick={handleNotInterested}
                          disabled={regenerating}
                          className="text-xs text-white/40 transition hover:text-white/60 disabled:opacity-50"
                        >
                          {regenerating ? "Finding another..." : "Not interested — find another"}
                        </button>
                        {(article.themes?.length || article.theme_model) && (
                          <details className="text-right">
                            <summary className="cursor-pointer list-none text-[0.6rem] text-white/25 hover:text-white/45">
                              synthesis log
                            </summary>
                            <div className="mt-2 rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-left text-[0.65rem] text-white/50">
                              {article.theme_model && (
                                <p className="mb-1">
                                  <span className="text-white/35">theme model: </span>{article.theme_model}
                                </p>
                              )}
                              {article.article_model && (
                                <p className="mb-1">
                                  <span className="text-white/35">article model: </span>{article.article_model}
                                </p>
                              )}
                              {article.themes && article.themes.length > 0 && (
                                <p className="mb-1">
                                  <span className="text-white/35">themes: </span>{article.themes.join(", ")}
                                </p>
                              )}
                              {article.theme_notes && article.theme_notes.length > 0 && (
                                <div className="mt-2 space-y-2 border-t border-white/10 pt-2">
                                  {article.theme_notes.map((n, i) => (
                                    <div key={i}>
                                      <p className="font-medium text-white/60">{n.theme}</p>
                                      <p className="text-white/40">{n.notes}</p>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          </details>
                        )}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            )}

            {/* Reflection chat */}
            {phase === "reflection" && (
              <div className="relative z-10 mx-auto w-full max-w-[48rem] px-3 py-8 md:px-6">
                {article && (
                  <div className="mb-8 rounded-xl border border-white/10 bg-white/[0.04] px-5 py-4">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-[0.15em] text-white/40">
                      Reflecting on
                    </p>
                    <p className="text-sm font-medium text-white/80">{article.title}</p>
                  </div>
                )}

                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`group mb-10 w-full ${m.role === "user" ? "flex justify-end" : ""}`}
                  >
                    {m.role === "assistant" ? (
                      <div className="glass-panel-subtle max-w-none rounded-2xl border border-white/10 px-5 py-4">
                        <div className="text-[0.95rem] leading-7 text-white/95">
                          <p className="whitespace-pre-wrap break-words">{m.content}</p>
                        </div>
                        <div className="mt-2 flex items-center gap-0.5 text-white/50 opacity-90 transition-opacity group-hover:opacity-100">
                          <button
                            type="button"
                            onClick={() => void copyText(m.content)}
                            className="rounded-lg p-2 hover:bg-white/10 hover:text-white"
                            title="Copy"
                            aria-label="Copy response"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </button>
                          <button
                            type="button"
                            onClick={() => readAloud(m.content)}
                            disabled={readAloudBusy}
                            aria-busy={readAloudBusy}
                            className="rounded-lg p-2 hover:bg-white/10 hover:text-white disabled:pointer-events-none disabled:opacity-35"
                            title={readAloudBusy ? "Loading audio…" : "Read aloud"}
                            aria-label="Read response aloud"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="glass-panel max-w-[min(100%,85%)] rounded-[1.75rem] px-5 py-3 text-[0.95rem] leading-7 text-white/95">
                        <p className="whitespace-pre-wrap break-words">{m.content}</p>
                      </div>
                    )}
                  </div>
                ))}

                {sending && (
                  <div className="glass-panel-subtle mb-10 inline-block rounded-2xl border border-white/10 px-4 py-3 text-[0.95rem] text-white/60">
                    <span className="inline-flex gap-1">
                      <span className="animate-pulse">Thinking</span>
                      <span className="inline-flex gap-0.5">
                        <span className="animate-bounce" style={{ animationDelay: "0ms" }}>.</span>
                        <span className="animate-bounce" style={{ animationDelay: "150ms" }}>.</span>
                        <span className="animate-bounce" style={{ animationDelay: "300ms" }}>.</span>
                      </span>
                    </span>
                  </div>
                )}

                {!hasConversation && (
                  <p className="text-center text-sm text-white/40">
                    Share your thoughts on the article to begin the reflection.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Bottom composer — visible in reflection phase */}
          {phase === "reflection" && (
            <div className="flex-none border-t border-white/10 bg-[#0a0a12]/90 px-3 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur-md">
              <div className="mx-auto w-full max-w-[48rem] space-y-2">
                <LiveDictationBubble />
                <AskAnythingComposer layout="center" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
