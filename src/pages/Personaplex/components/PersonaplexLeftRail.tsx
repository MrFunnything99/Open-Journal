import type { Dispatch, ReactNode, SetStateAction } from "react";

export type PersonaplexView = "voice_memo" | "brain" | "semantic_memory";

type Props = {
  /** Expanded width on desktop (Open WebUI–style rail). */
  expanded: boolean;
  setExpanded: (v: boolean) => void;
  mobileOpen: boolean;
  setMobileOpen: (v: boolean) => void;
  view: PersonaplexView;
  setView: Dispatch<SetStateAction<PersonaplexView>>;
};

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

export function PersonaplexLeftRail({
  expanded,
  setExpanded,
  mobileOpen,
  setMobileOpen,
  view,
  setView,
}: Props) {
  const narrow = !expanded;

  const closeMobile = () => setMobileOpen(false);

  const railColumn = (
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
      </div>

      <nav className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain px-1.5 pb-3 pt-2" aria-label="Primary">
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
          view === "semantic_memory",
          () => {
            setView("semantic_memory");
            closeMobile();
          },
          "Semantic Memory",
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6M7 4h10a2 2 0 012 2v12a2 2 0 01-2 2H7a2 2 0 01-2-2V6a2 2 0 012-2z" />
          </svg>,
          narrow
        )}
      </nav>
    </div>
  );

  return (
    <>
      <aside
        className={`hidden h-full shrink-0 transition-[width] duration-200 ease-out md:flex md:flex-col ${
          expanded ? "w-[260px]" : "w-[72px]"
        }`}
        aria-label="App sidebar"
      >
        {railColumn}
      </aside>

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
          {railColumn}
        </aside>
      </div>
    </>
  );
}
