import type { Dispatch, ReactNode, SetStateAction } from "react";
import { usePersonaplexChat, type AgentActivityEntry } from "../PersonaplexChatContext";
import { ThemeToggle } from "../../../components/ThemeToggle";
import { useTheme } from "../../../hooks/useTheme";
import { AskAnythingComposer, LiveDictationBubble } from "./GlobalAskAnythingBar";

export type PersonaplexView = "voice_memo" | "brain" | "recommendations" | "journal";

type Props = {
  /** Expanded width on desktop (Open WebUI–style rail). */
  expanded: boolean;
  setExpanded: (v: boolean) => void;
  mobileOpen: boolean;
  setMobileOpen: (v: boolean) => void;
  view: PersonaplexView;
  setView: Dispatch<SetStateAction<PersonaplexView>>;
};

function sectionLabel(text: string) {
  return (
    <p className="px-2 pb-1.5 pt-3 text-[0.6rem] font-semibold uppercase tracking-[0.18em] text-white/40">{text}</p>
  );
}

function navRow(
  active: boolean,
  onClick: () => void,
  label: string,
  icon: ReactNode,
  narrow: boolean
) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      className={`flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2.5 text-left text-sm font-medium transition-colors ${
        active ? "bg-white/[0.14] text-white shadow-sm ring-1 ring-white/10" : "text-white/70 hover:bg-white/10 hover:text-white"
      }`}
    >
      <span className="flex h-8 w-8 shrink-0 items-center justify-center text-white/85">{icon}</span>
      {!narrow && <span className="min-w-0 truncate">{label}</span>}
    </button>
  );
}

function activityIcon(kind: AgentActivityEntry["kind"]) {
  switch (kind) {
    case "user":
      return "→";
    case "assistant":
      return "◆";
    case "retrieval":
      return "⌕";
    case "tool":
      return "⚙";
    default:
      return "·";
  }
}

export function PersonaplexLeftRail({
  expanded,
  setExpanded,
  mobileOpen,
  setMobileOpen,
  view,
  setView,
}: Props) {
  const { mode, toggle } = useTheme();
  const { activityLog, chatRecents, loadRecentSession, newChat, clearActivityLog } = usePersonaplexChat();
  const narrow = !expanded;

  const closeMobile = () => setMobileOpen(false);

  const railColumn = (showComposer: boolean) => (
    <div className="flex h-full min-h-0 flex-col border-r border-white/10 bg-[#0c0c12]/92 text-white backdrop-blur-xl">
      <div className="flex flex-none items-center gap-1 border-b border-white/10 px-2 py-2.5">
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="rounded-lg p-2 text-white/70 transition hover:bg-white/10 hover:text-white"
          title={expanded ? "Collapse sidebar" : "Expand sidebar"}
          aria-expanded={expanded}
          aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h10M4 18h16" />
          </svg>
        </button>
        {!narrow && (
          <span className="flex-1 truncate pl-1 text-xs font-semibold uppercase tracking-[0.2em] text-white/80">
            Selfmeridian
          </span>
        )}
        <button
          type="button"
          onClick={() => {
            newChat();
            closeMobile();
          }}
          className="rounded-lg p-2 text-emerald-300/90 transition hover:bg-white/10"
          title="New chat"
          aria-label="New chat"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
      </div>

      <nav className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain px-1.5 pb-3" aria-label="Primary">
        {sectionLabel("Navigate")}
        {navRow(
          view === "voice_memo",
          () => {
            setView("voice_memo");
            closeMobile();
          },
          "Home",
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
          </svg>,
          narrow
        )}
        {navRow(
          view === "journal",
          () => {
            setView("journal");
            closeMobile();
          },
          "Chat",
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>,
          narrow
        )}
        {navRow(
          view === "brain",
          () => {
            setView("brain");
            closeMobile();
          },
          "The Brain",
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>,
          narrow
        )}
        {navRow(
          view === "recommendations",
          () => {
            setView("recommendations");
            closeMobile();
          },
          "Recommendations",
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
          </svg>,
          narrow
        )}

        {!narrow && chatRecents.length > 0 && (
          <>
            {sectionLabel("Recents")}
            <ul className="space-y-0.5">
              {chatRecents.map((r) => (
                <li key={r.sessionId}>
                  <button
                    type="button"
                    onClick={() => {
                      loadRecentSession(r.sessionId);
                      closeMobile();
                    }}
                    className="w-full truncate rounded-lg px-2.5 py-2 text-left text-xs text-white/75 transition hover:bg-white/10 hover:text-white"
                  >
                    {r.title}
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}

        {!narrow && (
          <>
            {sectionLabel("Activity")}
            <div className="flex items-center justify-between gap-2 px-2 pb-1">
              <span className="text-[10px] text-white/35">Agent steps</span>
              {activityLog.length > 0 && (
                <button type="button" className="text-[10px] text-emerald-400/90 hover:underline" onClick={clearActivityLog}>
                  Clear
                </button>
              )}
            </div>
            <ul className="max-h-48 space-y-1 overflow-y-auto px-1 text-[11px] leading-snug text-white/60">
              {activityLog.length === 0 ? (
                <li className="px-2 py-2 italic text-white/35">No activity yet — use the composer at the bottom of the sidebar.</li>
              ) : (
                [...activityLog].reverse().map((e) => (
                  <li key={e.id} className="rounded-lg bg-white/[0.04] px-2 py-1.5">
                    <span className="mr-1 opacity-60">{activityIcon(e.kind)}</span>
                    {e.summary}
                  </li>
                ))
              )}
            </ul>
          </>
        )}
      </nav>

      {showComposer && (
        <div className="flex flex-none flex-col gap-1.5 border-t border-white/10 bg-[#080810]/95 px-1.5 pb-1.5 pt-2">
          <LiveDictationBubble />
          <AskAnythingComposer layout="rail" railNarrow={narrow} onExpandRail={() => setExpanded(true)} />
        </div>
      )}

      <div className="flex flex-none items-center justify-center gap-1 border-t border-white/10 p-2">
        <ThemeToggle mode={mode} onToggle={toggle} className="border-white/15 bg-white/10 text-white hover:bg-white/15" />
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop rail */}
      <aside
        className={`hidden h-full shrink-0 transition-[width] duration-200 ease-out md:flex md:flex-col ${
          expanded ? "w-[260px]" : "w-[72px]"
        }`}
        aria-label="App sidebar"
      >
        {railColumn(true)}
      </aside>

      {/* Mobile drawer */}
      <div className="md:hidden">
        {mobileOpen && (
          <button
            type="button"
            className="fixed inset-0 z-40 bg-black/50"
            aria-label="Close menu"
            onClick={() => setMobileOpen(false)}
          />
        )}
        <aside
          className={`fixed left-0 top-0 z-50 h-full w-[min(88vw,280px)] shadow-2xl transition-transform duration-200 ease-out ${
            mobileOpen ? "translate-x-0" : "-translate-x-full pointer-events-none"
          }`}
          aria-hidden={!mobileOpen}
        >
          {railColumn(false)}
        </aside>
      </div>
    </>
  );
}
