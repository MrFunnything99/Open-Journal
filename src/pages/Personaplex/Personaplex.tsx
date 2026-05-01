import { useCallback, useEffect, useRef, useState } from "react";
import { backendFetch } from "../../backendApi";
import type { ChatMessage } from "./hooks/useJournalHistory";
import { JOURNAL_HISTORY_STORAGE_KEY, parseKnowledgeBaseFile, useJournalHistory } from "./hooks/useJournalHistory";
import {
  buildJournalsDownloadZip,
  extractJournalEntriesFromMarkdownDump,
  listJournalPathsInKnowledgeBaseZip,
  parseKnowledgeBaseMarkdownZip,
} from "./knowledgeBaseMarkdownZip";
import { BrainLayout } from "./components/BrainLayout";
import { SemanticMemoryLayout } from "./components/SemanticMemoryLayout";
import { VoiceMemoTab } from "./components/VoiceMemoTab";

import { PersonaplexChatProvider, type PersonaplexNavigateAction } from "./PersonaplexChatContext";
import { MobileAskComposerDockGate } from "./components/GlobalAskAnythingBar";
import { PersonaplexGithubLink } from "./components/PersonaplexGithubLink";
import { PersonaplexLeftRail, type PersonaplexView } from "./components/PersonaplexLeftRail";
import { AssistedJournalUnloadSync } from "./components/AssistedJournalUnloadSync";

const JOURNAL_FILENAME_DATE_CHUNK = 80;

/** Canonical YYYY-MM-DD from flexible string, or null. */
function parseYyyyMmDdFlexible(raw: string | null | undefined): string | null {
  if (raw == null || typeof raw !== "string") return null;
  const s = raw.trim().split("T")[0].split(" ")[0];
  const m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (!m) return null;
  const y = parseInt(m[1], 10);
  const mo = parseInt(m[2], 10);
  const d = parseInt(m[3], 10);
  const dt = new Date(y, mo - 1, d, 12, 0, 0, 0);
  if (dt.getFullYear() !== y || dt.getMonth() !== mo - 1 || dt.getDate() !== d) return null;
  return `${y}-${String(mo).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
}

/** Local-calendar noon ISO from YYYY-MM-DD. */
function localNoonIsoFromYmd(ymd: string): string {
  const m = ymd.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return new Date().toISOString();
  const y = parseInt(m[1], 10);
  const mo = parseInt(m[2], 10);
  const d = parseInt(m[3], 10);
  const dt = new Date(y, mo - 1, d, 12, 0, 0, 0);
  return dt.toISOString();
}

/** 4-digit year or 2-digit (00–69 → 20xx, 70–99 → 19xx), aligned with backend. */
function yearFromFilenameToken(t: string): number | null {
  if (!/^\d+$/.test(t)) return null;
  if (t.length === 4) {
    const y = parseInt(t, 10);
    return y >= 1900 && y <= 2100 ? y : null;
  }
  if (t.length === 2) {
    const yy = parseInt(t, 10);
    return yy >= 70 ? 1900 + yy : 2000 + yy;
  }
  return null;
}

/** Path/name-only date (matches backend fallback) when API is unreachable or returns null. */
function journalPathYmdFallback(path: string): string | null {
  const base = path.replace(/\\/g, "/").split("/").pop() ?? path;
  const stem = base.replace(/\.(md|txt|markdown)$/i, "").trim();
  let g = stem.match(/^(\d{1,2})[-_.](\d{1,2})[-_.](\d{4}|\d{2})$/);
  if (g) {
    const y = yearFromFilenameToken(g[3]);
    if (y != null) return parseYyyyMmDdFlexible(`${y}-${g[1]}-${g[2]}`);
  }
  g = stem.match(/^(\d{4}|\d{2})[-_.](\d{1,2})[-_.](\d{1,2})$/);
  if (g) {
    const y = yearFromFilenameToken(g[1]);
    if (y != null) return parseYyyyMmDdFlexible(`${y}-${g[2]}-${g[3]}`);
  }
  g = stem.match(/^(\d{4})(\d{2})(\d{2})$/);
  if (g) return parseYyyyMmDdFlexible(`${g[1]}-${g[2]}-${g[3]}`);
  g = stem.match(/^(\d{2})(\d{2})(\d{2})$/);
  if (g) {
    const y = yearFromFilenameToken(g[1]);
    if (y != null) return parseYyyyMmDdFlexible(`${y}-${parseInt(g[2], 10)}-${parseInt(g[3], 10)}`);
  }
  const inBase = base.match(/(?:^|[^\d])(\d{1,2})[-_.](\d{1,2})[-_.](\d{4}|\d{2})(?:[^\d]|$)/);
  if (inBase) {
    const y = yearFromFilenameToken(inBase[3]);
    if (y != null) return parseYyyyMmDdFlexible(`${y}-${inBase[1]}-${inBase[2]}`);
  }
  return null;
}

function resolveFilingIsoFromApiAndPath(apiDay: string | null | undefined, path: string, fallbackNow: string): string {
  const fromApi = parseYyyyMmDdFlexible(
    apiDay != null && String(apiDay).trim() !== "" && String(apiDay).toLowerCase() !== "null" ? String(apiDay) : null
  );
  if (fromApi) return localNoonIsoFromYmd(fromApi);
  const fromPath = journalPathYmdFallback(path);
  if (fromPath) return localNoonIsoFromYmd(fromPath);
  return fallbackNow;
}

/** Tinfoil: paths/filenames only — no file contents. */
async function inferJournalPathDatesMap(paths: string[]): Promise<Map<string, string>> {
  const map = new Map<string, string>();
  if (paths.length === 0) return map;
  const fallback = new Date().toISOString();
  const unique = [...new Set(paths)];
  for (let i = 0; i < unique.length; i += JOURNAL_FILENAME_DATE_CHUNK) {
    const slice = unique.slice(i, i + JOURNAL_FILENAME_DATE_CHUNK);
    try {
      const r = await backendFetch("/infer-journal-filename-dates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: slice }),
      });
      const data = r.ok ? ((await r.json()) as { dates?: (string | null)[] }) : {};
      const dates = Array.isArray(data.dates) ? data.dates : [];
      slice.forEach((p, j) => {
        map.set(p, resolveFilingIsoFromApiAndPath(j < dates.length ? dates[j] : null, p, fallback));
      });
    } catch {
      slice.forEach((p) => map.set(p, resolveFilingIsoFromApiAndPath(null, p, fallback)));
    }
  }
  return map;
}

const KB_UPLOAD_CONFIRM_MESSAGE =
  "Uploading a knowledge base replaces everything for this session.\n\n" +
  "• Server: all journal embeddings are deleted, then rebuilt from this file.\n" +
  "• This device: manual journals and AI-assisted journals are replaced by the import (not merged).\n\n" +
  "Stay online until syncing finishes. This cannot be undone.\n\n" +
  "Continue?";

const JOURNAL_FOLDER_UPLOAD_CONFIRM_MESSAGE =
  "Import journal files from the selected folder or chosen files? New entries will be added to your knowledge base.";

export const Personaplex = () => {
  const [view, setView] = useState<PersonaplexView>("voice_memo");
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [railExpanded, setRailExpanded] = useState(true);
  const [mobileRailOpen, setMobileRailOpen] = useState(false);
  const chatToast = useCallback((msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  }, []);
  const handleChatAgentAction = useCallback(
    (actions: PersonaplexNavigateAction[]) => {
      let navigated = false;
      for (const a of actions) {
        if (a.type !== "navigate") continue;
        const target: PersonaplexView = a.view === "journal" ? "voice_memo" : a.view;
        setView(target);
        navigated = true;
      }
      if (navigated) chatToast("Opened the screen you asked for.");
    },
    [chatToast]
  );

  const {
    entries,
    saveEntry,
    syncUnsyncedEntries,
    deleteEntry,
    updateJournalEntry,
    getFormattedDate,
    importEntriesReplaceAll,
    clearHistory,
  } = useJournalHistory();

  const chatWorkspaceResetRef = useRef<(() => void) | null>(null);

  const prepareKnowledgeBaseUpload = useCallback(() => window.confirm(KB_UPLOAD_CONFIRM_MESSAGE), []);

  const prepareJournalDumpUpload = useCallback(() => window.confirm(JOURNAL_FOLDER_UPLOAD_CONFIRM_MESSAGE), []);

  const resetServerKnowledgeBaseMemory = useCallback(async (): Promise<boolean> => {
    try {
      const r = await backendFetch("/memory-reset-knowledge-base-import", { method: "POST" });
      const data = (await r.json().catch(() => ({}))) as { ok?: boolean };
      return r.ok && data.ok !== false;
    } catch {
      return false;
    }
  }, []);

  const handleStartFreshPersonaplex = useCallback(async () => {
    if (
      !window.confirm(
        "Start fresh?\n\n" +
          "• Server: deletes journal embeddings for this device/instance.\n" +
          "• This browser: clears saved journals and the in-progress home chat.\n\n" +
          "This cannot be undone.",
      )
    ) {
      return;
    }
    const ok = await resetServerKnowledgeBaseMemory();
    if (!ok) {
      setToastMessage("Could not wipe server memory. Your local data was not changed.");
      setTimeout(() => setToastMessage(null), 5000);
      return;
    }
    try {
      localStorage.removeItem(JOURNAL_HISTORY_STORAGE_KEY);
    } catch {
      /* ignore */
    }
    clearHistory();
    chatWorkspaceResetRef.current?.();
    setToastMessage("Started fresh: server memory wiped and local Personaplex data cleared.");
    setTimeout(() => setToastMessage(null), 5000);
  }, [clearHistory, resetServerKnowledgeBaseMemory]);

  const handleDownloadJournals = useCallback(async () => {
    try {
      const blob = await buildJournalsDownloadZip(entries);
      const a = document.createElement("a");
      const url = URL.createObjectURL(blob);
      a.href = url;
      a.download = `selfmeridian-journals-${new Date().toISOString().slice(0, 10)}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      setToastMessage("Downloaded Journals (.zip).");
      setTimeout(() => setToastMessage(null), 4000);
    } catch {
      setToastMessage("Download failed.");
      setTimeout(() => setToastMessage(null), 3000);
    }
  }, [entries]);

  /** Legacy knowledge-base .zip / .json import (`journals/` + `conversations/` under selfmeridian-knowledge-base). */
  const handleImportKnowledgeBaseFile = useCallback(
    async (file: File) => {
      const applyKnowledgeBaseImport = async (exportPayload: {
        version: number;
        exportedAt: string;
        entries: (typeof entries)[number][];
      }) => {
        if (!(await resetServerKnowledgeBaseMemory())) {
          setToastMessage("Could not reset server memory. Check your connection and try again.");
          setTimeout(() => setToastMessage(null), 6000);
          return;
        }
        const nEntries = importEntriesReplaceAll({
          version: exportPayload.version,
          exportedAt: exportPayload.exportedAt,
          entries: exportPayload.entries,
        });
        let synced = await syncUnsyncedEntries();
        // Retry once after a brief delay to smooth over transient backend restarts.
        if (synced < nEntries) {
          await new Promise((resolve) => setTimeout(resolve, 900));
          synced += await syncUnsyncedEntries();
        }
        if (synced < nEntries) {
          const pending = Math.max(0, nEntries - synced);
          setToastMessage(
            `Knowledge base replaced: ${nEntries} journal entr${nEntries === 1 ? "y" : "ies"} saved locally; ${synced} synced to server, ${pending} pending. Keep backend running and re-import or add/edit an entry to trigger another sync.`
          );
        } else {
          setToastMessage(
            `Knowledge base replaced: ${nEntries} journal entr${nEntries === 1 ? "y" : "ies"}. ${synced} synced to server memory.`
          );
        }
        setTimeout(() => setToastMessage(null), 6500);
      };

      try {
        const looksZip =
          file.name.toLowerCase().endsWith(".zip") ||
          file.type === "application/zip" ||
          file.type === "application/x-zip-compressed";

        if (looksZip) {
          const journalPaths = await listJournalPathsInKnowledgeBaseZip(file);
          const dateMap = await inferJournalPathDatesMap(journalPaths);
          const parsed = await parseKnowledgeBaseMarkdownZip(file, dateMap);
          if (!parsed) {
            setToastMessage("That ZIP isn’t a valid Markdown knowledge base (expected journals/ and conversations/).");
            setTimeout(() => setToastMessage(null), 5000);
            return;
          }
          await applyKnowledgeBaseImport({ version: 1, exportedAt: new Date().toISOString(), entries: parsed.entries });
          return;
        }

        const text = await file.text();
        const parsed = parseKnowledgeBaseFile(text);
        if (!parsed) {
          setToastMessage("Use a .zip of Markdown folders, or a legacy .json backup.");
          setTimeout(() => setToastMessage(null), 4000);
          return;
        }
        if (parsed.kind === "journalsOnly") {
          await applyKnowledgeBaseImport(parsed.data);
          return;
        }
        await applyKnowledgeBaseImport({
          version: parsed.data.version,
          exportedAt: parsed.data.exportedAt,
          entries: parsed.data.entries,
        });
      } catch {
        setToastMessage("Could not read that file.");
        setTimeout(() => setToastMessage(null), 4000);
      }
    },
    [importEntriesReplaceAll, resetServerKnowledgeBaseMemory, syncUnsyncedEntries]
  );

  const handleImportJournalDumpFolder = useCallback(
    async (files: FileList) => {
      const list = Array.from(files).filter((f) => /\.(md|txt)$/i.test(f.name));
      if (list.length === 0) {
        setToastMessage("No .md or .txt files found in that folder.");
        setTimeout(() => setToastMessage(null), 3500);
        return;
      }

      const paths = list.map((f) => (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name);
      const dateMap = await inferJournalPathDatesMap(paths);

      type Row = { date: string | null; transcript: ChatMessage[] };
      const rows: Row[] = [];
      for (const f of list) {
        const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
        const text = (await f.text()).trim();
        if (!text) continue;
        const entryDateIso = dateMap.get(rel) ?? new Date().toISOString();
        for (const item of extractJournalEntriesFromMarkdownDump(rel, text, entryDateIso)) {
          rows.push(item);
        }
      }

      rows.sort((a, b) => {
        const ta = a.date ? Date.parse(a.date) : 0;
        const tb = b.date ? Date.parse(b.date) : 0;
        return ta - tb;
      });

      let imported = 0;
      for (const row of rows) {
        const id = saveEntry(row.transcript, row.date ?? undefined);
        if (id) imported += 1;
      }

      if (imported === 0) {
        setToastMessage("No valid journal content found.");
      } else {
        setToastMessage(`Imported ${imported} journal entr${imported === 1 ? "y" : "ies"} from folder.`);
      }
      setTimeout(() => setToastMessage(null), 4500);
    },
    [saveEntry]
  );

  const hasRunInitialSyncRef = useRef(false);
  useEffect(() => {
    if (entries.length === 0 || hasRunInitialSyncRef.current) return;
    hasRunInitialSyncRef.current = true;
    syncUnsyncedEntries();
  }, [entries.length, syncUnsyncedEntries]);

  return (
    <PersonaplexChatProvider onToast={chatToast} onAgentAction={handleChatAgentAction} chatWorkspaceResetRef={chatWorkspaceResetRef}>
      <AssistedJournalUnloadSync />
    <div className="relative flex h-screen w-full flex-row overflow-hidden bg-[#0a0a12] font-sans text-white antialiased">
      {/* Ethereal mesh + animated ambient orbs (pointer-events none on layer) */}
      <div className="pointer-events-none fixed inset-0 z-0" aria-hidden>
        <div className="absolute inset-0 bg-gradient-to-br from-[#0c1228] via-[#0d1f2d] to-[#1a0f2e]" />
        <div
          className="absolute inset-0 opacity-90"
          style={{
            background:
              "radial-gradient(ellipse 85% 55% at 50% -15%, rgba(45, 212, 191, 0.22), transparent 55%), radial-gradient(ellipse 55% 45% at 100% 45%, rgba(168, 85, 247, 0.2), transparent 50%), radial-gradient(ellipse 50% 40% at 0% 85%, rgba(59, 130, 246, 0.14), transparent 50%)",
          }}
        />
        <div
          className="ambient-orb ambient-orb-1 absolute left-[6%] top-[10%] h-96 w-96 rounded-full blur-[100px]"
          style={{
            background:
              "radial-gradient(circle at 35% 35%, rgba(94, 234, 212, 0.45), rgba(45, 212, 191, 0.12) 45%, transparent 70%)",
          }}
        />
        <div
          className="ambient-orb ambient-orb-2 absolute bottom-[6%] right-[2%] h-96 w-96 rounded-full blur-[110px]"
          style={{
            background:
              "radial-gradient(circle at 40% 40%, rgba(216, 180, 254, 0.4), rgba(168, 85, 247, 0.14) 50%, transparent 72%)",
          }}
        />
        <div
          className="ambient-orb ambient-orb-3 absolute right-[18%] top-[32%] h-96 w-96 rounded-full blur-[95px]"
          style={{
            background:
              "radial-gradient(circle at 30% 50%, rgba(129, 140, 248, 0.38), rgba(99, 102, 241, 0.1) 48%, transparent 70%)",
          }}
        />
        <div
          className="ambient-orb ambient-orb-4 absolute left-[28%] bottom-[22%] h-80 w-80 rounded-full blur-[100px] md:h-96 md:w-96"
          style={{
            background:
              "radial-gradient(circle at 50% 50%, rgba(167, 139, 250, 0.32), rgba(192, 132, 252, 0.1) 55%, transparent 72%)",
          }}
        />
      </div>

      <PersonaplexLeftRail
        expanded={railExpanded}
        setExpanded={setRailExpanded}
        mobileOpen={mobileRailOpen}
        setMobileOpen={setMobileRailOpen}
        view={view}
        setView={setView}
      />

      <div className="relative z-10 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      {/* Header — centered brand; nav lives in left rail (Open WebUI–style shell) */}
      <header className="relative z-20 flex-none border-b border-white/[0.06] px-4 py-2.5 sm:px-6 sm:py-3">
        <div className="flex w-full items-center gap-3">
          <div className="flex min-w-0 flex-1 justify-start">
            <button
              type="button"
                    className="rounded-full border border-white/15 bg-white/10 p-2.5 text-white shadow-sm backdrop-blur-md md:hidden"
              onClick={() => setMobileRailOpen(true)}
              aria-label="Open menu"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
          <div className="glass-panel flex shrink-0 items-center justify-center rounded-full px-5 py-2 text-center shadow-sm">
            <h1 className="whitespace-nowrap text-xs font-medium uppercase tracking-[0.22em] text-white sm:text-sm">
              Selfmeridian
            </h1>
          </div>
          <div className="min-w-0 flex-1" />
        </div>
      </header>

      {toastMessage && (
        <div
          className="fixed bottom-24 left-1/2 z-50 -translate-x-1/2 rounded-xl border border-white/10 bg-white/10 px-4 py-2 text-sm font-medium text-white shadow-lg backdrop-blur-xl"
          role="status"
        >
          {toastMessage}
        </div>
      )}

      {/* Main content */}
      <main className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden opacity-100 transition-opacity duration-300">
          {view === "voice_memo" && (
            <VoiceMemoTab onToast={chatToast} saveEntry={saveEntry} syncUnsyncedEntries={syncUnsyncedEntries} />
          )}
          {view === "brain" && (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <div className="flex-none border-b border-white/10 px-4 py-3 md:px-5">
                <h2 className="text-sm font-medium text-[#E8F1F5] md:text-base">Knowledge base</h2>
                <p className="mt-1 text-xs text-[#9BB1BE] md:text-sm">
                  Manual journals and AI-assisted journals.
                </p>
              </div>
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                <BrainLayout
                  entries={entries}
                  onDeleteEntry={deleteEntry}
                  onUpdateJournalEntry={updateJournalEntry}
                  syncUnsyncedEntries={syncUnsyncedEntries}
                  getFormattedDate={getFormattedDate}
                  onToast={(msg) => {
                    setToastMessage(msg);
                    setTimeout(() => setToastMessage(null), 3000);
                  }}
                  onDownloadJournals={handleDownloadJournals}
                  onImportKnowledgeBaseFile={handleImportKnowledgeBaseFile}
                  onPrepareKnowledgeBaseUpload={prepareKnowledgeBaseUpload}
                  onImportJournalDumpFolder={handleImportJournalDumpFolder}
                  onPrepareJournalDumpUpload={prepareJournalDumpUpload}
                  onStartFresh={handleStartFreshPersonaplex}
                />
              </div>
            </div>
          )}
          {view === "semantic_memory" && (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <div className="flex-none border-b border-white/10 px-4 py-3 md:px-5">
                <h2 className="text-sm font-medium text-[#E8F1F5] md:text-base">Semantic memory</h2>
                <p className="mt-1 text-xs text-[#9BB1BE] md:text-sm">
                  Books, podcasts, and research articles you've consumed.
                </p>
              </div>
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                <SemanticMemoryLayout
                  onToast={(msg) => {
                    setToastMessage(msg);
                    setTimeout(() => setToastMessage(null), 3000);
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="pointer-events-none relative z-10 flex-shrink-0 px-4 pb-[calc(4.5rem+env(safe-area-inset-bottom))] pt-1 text-center md:pb-5">
        <PersonaplexGithubLink className="pointer-events-auto" />
      </footer>

      <MobileAskComposerDockGate railOpen={mobileRailOpen} activeView={view} />
      </div>
    </div>
    </PersonaplexChatProvider>
  );
};
