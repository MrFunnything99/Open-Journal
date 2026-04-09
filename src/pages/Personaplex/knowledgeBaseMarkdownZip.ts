import JSZip from "jszip";
import type { ChatMessage, JournalEntry } from "./hooks/useJournalHistory";

const ROOT = "selfmeridian-knowledge-base";

/** Split marker between multiple entries in one daily Markdown file (export/import round-trip). */
export const KNOWLEDGE_BASE_ENTRY_BOUNDARY = "<!-- SelfMeridian:entry-boundary -->";

function isConversationEntry(e: JournalEntry): boolean {
  return e.entrySource === "conversation";
}

/** Local calendar date key `YYYY-MM-DD` for grouping exports and sidebar. */
export function localCalendarDayKey(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "1970-01-01";
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** Heading for a `YYYY-MM-DD` key (Brain sidebar day groups). */
export function formatCalendarDayHeading(dayKey: string): string {
  const parts = dayKey.split("-").map((x) => parseInt(x, 10));
  if (parts.length !== 3 || parts.some((n) => Number.isNaN(n))) return dayKey;
  const [y, mo, d] = parts;
  return new Date(y!, mo! - 1, d!).toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

/** Same Year → Month → entries bucketing as The Brain explorer sidebar (descending years/months). */
export type YearMonthTree = {
  year: number;
  months: { month: number; monthLabel: string; entries: JournalEntry[] }[];
}[];

export function buildYearMonthTree(sorted: JournalEntry[]): YearMonthTree {
  const byYear = new Map<number, Map<number, JournalEntry[]>>();
  for (const e of sorted) {
    const d = new Date(e.date);
    const y = d.getFullYear();
    const m = d.getMonth();
    if (!byYear.has(y)) byYear.set(y, new Map());
    const ym = byYear.get(y)!;
    if (!ym.has(m)) ym.set(m, []);
    ym.get(m)!.push(e);
  }
  const years = [...byYear.keys()].sort((a, b) => b - a);
  return years.map((year) => {
    const monthsMap = byYear.get(year)!;
    const months = [...monthsMap.keys()].sort((a, b) => b - a);
    return {
      year,
      months: months.map((month) => ({
        month,
        monthLabel: new Date(year, month, 1).toLocaleString("en-US", { month: "long" }),
        entries: monthsMap.get(month)!,
      })),
    };
  });
}

/** Calendar-day sub-groups within a month — same ordering as explorer day rows. */
export function groupJournalMonthByCalendarDay(
  monthEntries: JournalEntry[]
): { dayKey: string; entries: JournalEntry[] }[] {
  const m = new Map<string, JournalEntry[]>();
  for (const e of monthEntries) {
    const k = localCalendarDayKey(e.date);
    if (!m.has(k)) m.set(k, []);
    m.get(k)!.push(e);
  }
  const keys = [...m.keys()].sort();
  return keys.map((dayKey) => ({
    dayKey,
    entries: (m.get(dayKey) ?? []).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
  }));
}

function dayBundleRelPath(iso: string, section: "journals" | "conversations"): string {
  const d = new Date(iso);
  const year = d.getFullYear();
  const monthName = d.toLocaleString("en-US", { month: "long" });
  const dk = localCalendarDayKey(iso);
  return `${section}/${year}/${monthName}/${dk}.md`;
}

function bucketEntriesByLocalDay(entries: JournalEntry[], section: "journals" | "conversations"): Map<string, JournalEntry[]> {
  const map = new Map<string, JournalEntry[]>();
  for (const e of entries) {
    const path = dayBundleRelPath(e.date, section);
    if (!map.has(path)) map.set(path, []);
    map.get(path)!.push(e);
  }
  for (const list of map.values()) {
    list.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }
  return map;
}

function formatRecordedForMarkdown(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

function humanReadableTranscript(e: JournalEntry): string {
  return e.fullTranscript
    .map((m) => {
      const role = m.role === "user" ? "You" : "AI";
      let block = `### ${role}\n\n${m.text}`;
      if (m.role === "ai" && m.retrievalLog) {
        block += `\n\n**Memory context (vector DB)**\n\n\`\`\`\n${m.retrievalLog}\n\`\`\``;
      }
      return block;
    })
    .join("\n\n");
}

function entryToMarkdown(e: JournalEntry): string {
  const src = e.entrySource === "conversation" ? "conversation" : "journal";
  const fm = [
    "---",
    `id: ${e.id}`,
    `date: ${e.date}`,
    `entry_source: ${src}`,
    `synced_to_memory: ${e.syncedToMemory ? "true" : "false"}`,
    "---",
    "",
    `**Recorded:** ${formatRecordedForMarkdown(e.date)}`,
    "",
    "## Transcript",
    "",
    humanReadableTranscript(e),
    "",
    "",
    "<!-- SelfMeridian: full transcript JSON for re-import -->",
    "```json",
    JSON.stringify(e.fullTranscript, null, 2),
    "```",
    "",
  ].join("\n");
  return fm;
}

function readme(): string {
  return [
    "# SelfMeridian knowledge base",
    "",
    "This archive mirrors **The Brain → Knowledge base** layout:",
    "",
    "- `journals/<year>/<Month>/YYYY-MM-DD.md` — **Manual Journal Mode** saves; one file per calendar day; multiple sessions in time order, separated by `<!-- SelfMeridian:entry-boundary -->`",
    "- `conversations/<year>/<Month>/` — **AI-Assisted Journal Mode** saves; same one-file-per-day layout",
    "",
    "Each daily file contains one or more entries (YAML front matter + transcript each). Entries are separated by `<!-- SelfMeridian:entry-boundary -->`. Transcript JSON for re-import is at the end of each entry block.",
    "",
  ].join("\n");
}

/**
 * Legacy export layout: `selfmeridian-knowledge-base/` with `journals/` and `conversations/` day-combined `.md` files.
 * Kept for compatibility and importers; UI uses {@link buildJournalsDownloadZip} instead.
 */
export async function buildKnowledgeBaseMarkdownZip(entries: JournalEntry[]): Promise<Blob> {
  const zip = new JSZip();
  const root = zip.folder(ROOT);
  if (!root) throw new Error("Failed to create zip root");

  root.file("README.md", readme());

  const journals = entries.filter((e) => !isConversationEntry(e));
  const conversations = entries.filter(isConversationEntry);

  for (const [path, dayEntries] of bucketEntriesByLocalDay(journals, "journals")) {
    const combined = dayEntries.map((e) => entryToMarkdown(e)).join(`\n\n${KNOWLEDGE_BASE_ENTRY_BOUNDARY}\n\n`);
    root.file(path, combined);
  }

  for (const [path, dayEntries] of bucketEntriesByLocalDay(conversations, "conversations")) {
    const combined = dayEntries.map((e) => entryToMarkdown(e)).join(`\n\n${KNOWLEDGE_BASE_ENTRY_BOUNDARY}\n\n`);
    root.file(path, combined);
  }

  return zip.generateAsync({ type: "blob", compression: "DEFLATE" });
}

const JOURNALS_EXPORT_ROOT = "Journals";
const JOURNALS_MANUAL_DIR = "Manual Journals";
const JOURNALS_ASSISTED_DIR = "AI-Assisted Journals";

function safeJournalEntryFileName(id: string): string {
  return `${id.replace(/[/\\:?*"<>|]/g, "_")}.md`;
}

/**
 * One `.md` per entry under explorer-aligned paths, using {@link buildYearMonthTree} + {@link groupJournalMonthByCalendarDay}.
 * Each file body uses the same `entryToMarkdown` pipeline as the legacy knowledge-base zip (YAML + transcript + JSON block).
 */
export async function buildJournalsDownloadZip(entries: JournalEntry[]): Promise<Blob> {
  const zip = new JSZip();
  const root = zip.folder(JOURNALS_EXPORT_ROOT);
  if (!root) throw new Error("Failed to create zip root");

  const journalSorted = entries
    .filter((e) => !isConversationEntry(e))
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  const conversationSorted = entries
    .filter(isConversationEntry)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  function writeExplorerTree(base: JSZip, tree: YearMonthTree): void {
    for (const { year, months } of tree) {
      const yearFolder = base.folder(String(year));
      if (!yearFolder) continue;
      for (const { monthLabel, entries: monthEntries } of months) {
        const monthFolder = yearFolder.folder(monthLabel);
        if (!monthFolder) continue;
        for (const { dayKey, entries: dayEntries } of groupJournalMonthByCalendarDay(monthEntries)) {
          const dayFolder = monthFolder.folder(dayKey);
          if (!dayFolder) continue;
          for (const e of dayEntries) {
            dayFolder.file(safeJournalEntryFileName(e.id), entryToMarkdown(e));
          }
        }
      }
    }
  }

  const manualRoot = root.folder(JOURNALS_MANUAL_DIR);
  const assistedRoot = root.folder(JOURNALS_ASSISTED_DIR);
  if (manualRoot) writeExplorerTree(manualRoot, buildYearMonthTree(journalSorted));
  if (assistedRoot) writeExplorerTree(assistedRoot, buildYearMonthTree(conversationSorted));

  return zip.generateAsync({ type: "blob", compression: "DEFLATE" });
}

type SimpleFront = Record<string, string>;

function parseSimpleFrontmatter(md: string): { front: SimpleFront; body: string } {
  const t = md.trim();
  if (!t.startsWith("---")) return { front: {}, body: md };
  const end = t.indexOf("\n---", 3);
  if (end === -1) return { front: {}, body: md };
  const block = t.slice(3, end).trim();
  const body = t.slice(end + 4).trim();
  const front: SimpleFront = {};
  for (const line of block.split("\n")) {
    const m = line.match(/^([a-zA-Z0-9_]+):\s*(.*)$/);
    if (!m) continue;
    let v = m[2].trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
      try {
        v = JSON.parse(v.startsWith("'") ? `"${v.slice(1, -1).replace(/\\'/g, "'")}"` : v);
      } catch {
        /* keep raw */
      }
    }
    front[m[1]] = v;
  }
  return { front, body };
}

/** When there is no ```json transcript block, use markdown body minus boilerplate. */
function fallbackUserTranscriptFromBody(body: string): string {
  let b = body.replace(/^\s*\*\*Recorded:\*\*[^\n]*\n*/i, "").trim();
  b = b.replace(/^\s*##\s+Transcript\s*\n/i, "").trim();
  b = b.replace(/```json\s*[\s\S]*?```/g, "").trim();
  return b;
}

/**
 * Parse one or more journal entries from a dumped .md file (folder upload).
 * `entryDateIso` is the filing timestamp (from server filename inference); body is not used for dates.
 */
export function extractJournalEntriesFromMarkdownDump(
  _relativePath: string,
  rawText: string,
  entryDateIso: string
): Array<{ date: string | null; transcript: ChatMessage[] }> {
  const text = rawText.trim();
  if (!text) return [];
  const fileDate = entryDateIso;
  const segments = text.includes(KNOWLEDGE_BASE_ENTRY_BOUNDARY)
    ? text.split(KNOWLEDGE_BASE_ENTRY_BOUNDARY).map((s) => s.trim()).filter(Boolean)
    : [text];
  const out: Array<{ date: string | null; transcript: ChatMessage[] }> = [];

  for (const seg of segments) {
    const { body } = parseSimpleFrontmatter(seg);
    const dateStr = fileDate;
    let transcript = parseTranscriptJson(body);
    if (!transcript) {
      const userText = fallbackUserTranscriptFromBody(body);
      if (!userText.trim()) continue;
      transcript = [{ role: "user", text: userText.trim() }];
    }
    out.push({ date: dateStr, transcript });
  }
  return out;
}

function parseTranscriptJson(body: string): ChatMessage[] | null {
  const m = body.match(/```json\s*([\s\S]*?)```/);
  if (!m) return null;
  try {
    const raw = JSON.parse(m[1].trim()) as unknown;
    if (!Array.isArray(raw)) return null;
    const out: ChatMessage[] = [];
    for (const row of raw) {
      if (row == null || typeof row !== "object") continue;
      const o = row as Record<string, unknown>;
      if (o.role !== "user" && o.role !== "ai") continue;
      if (typeof o.text !== "string") continue;
      const retrievalLog = typeof o.retrievalLog === "string" ? o.retrievalLog : undefined;
      out.push({ role: o.role, text: o.text, retrievalLog });
    }
    return out.length > 0 ? out : null;
  } catch {
    return null;
  }
}

function journalEntryFromMarkdown(
  body: string,
  front: SimpleFront,
  pathHint: "journal" | "conversation" | null,
  entryDateIso: string
): JournalEntry | null {
  const transcript = parseTranscriptJson(body);
  if (!transcript) return null;
  const id = typeof front.id === "string" ? front.id : `import-${Date.now()}`;
  const date = entryDateIso;
  let entrySource: "journal" | "conversation" | undefined;
  if (front.entry_source === "conversation") entrySource = "conversation";
  else if (front.entry_source === "journal") entrySource = "journal";
  else if (pathHint === "conversation") entrySource = "conversation";
  else entrySource = "journal";
  const preview = transcript.map((m) => m.text).join(" ").slice(0, 120);
  const syncedToMemory = front.synced_to_memory === "true";
  return {
    id,
    date,
    preview: preview.length >= 120 ? preview + "…" : preview,
    fullTranscript: transcript,
    syncedToMemory,
    entrySource,
  };
}

function normalizePath(p: string): string {
  return p.replace(/\\/g, "/").replace(/^\/+/, "");
}

function stripRootPrefix(p: string): string {
  let x = normalizePath(p);
  if (x.startsWith(`${ROOT}/`)) x = x.slice(ROOT.length + 1);
  else if (x === ROOT) return "";
  return x;
}

function pathEntryKind(rel: string): "journal" | "conversation" | "skip" {
  const p = normalizePath(rel);
  if (p === "README.md" || !p.endsWith(".md")) return "skip";
  if (p.startsWith("journals/")) return "journal";
  if (p.startsWith("conversations/")) return "conversation";
  if (p.startsWith("library/")) return "skip";
  return "skip";
}

/** Relative paths for journal/conversation `.md` files (for `/infer-journal-filename-dates`). */
export async function listJournalPathsInKnowledgeBaseZip(file: File): Promise<string[]> {
  try {
    const zip = await JSZip.loadAsync(await file.arrayBuffer());
    const out = new Set<string>();
    for (const relPath of Object.keys(zip.files)) {
      const z = zip.files[relPath];
      if (!z || z.dir) continue;
      const rel = stripRootPrefix(relPath);
      if (!rel.toLowerCase().endsWith(".md")) continue;
      const kind = pathEntryKind(rel);
      if (kind === "journal" || kind === "conversation") out.add(rel);
    }
    return [...out];
  } catch {
    return [];
  }
}

export async function parseKnowledgeBaseMarkdownZip(
  file: File,
  journalPathDates: Map<string, string>
): Promise<{ entries: JournalEntry[] } | null> {
  try {
    const zip = await JSZip.loadAsync(await file.arrayBuffer());
    const entries: JournalEntry[] = [];

    const tasks: Promise<void>[] = [];
    for (const relPath of Object.keys(zip.files)) {
      const entry = zip.files[relPath];
      if (!entry || entry.dir) continue;
      const rel = stripRootPrefix(relPath);
      if (!rel.toLowerCase().endsWith(".md")) continue;

      tasks.push(
        (async () => {
          const text = await entry.async("string");
          const kind = pathEntryKind(rel);
          if (kind === "skip") return;

          if (kind === "journal" || kind === "conversation") {
            const entryDateIso = journalPathDates.get(rel) ?? new Date().toISOString();
            const segments = text.includes(KNOWLEDGE_BASE_ENTRY_BOUNDARY)
              ? text
                  .split(KNOWLEDGE_BASE_ENTRY_BOUNDARY)
                  .map((s) => s.trim())
                  .filter(Boolean)
              : [text];
            for (const seg of segments) {
              const parsed = parseSimpleFrontmatter(seg);
              const je = journalEntryFromMarkdown(
                parsed.body,
                parsed.front,
                kind === "conversation" ? "conversation" : "journal",
                entryDateIso
              );
              if (je) entries.push(je);
            }
          }
        })()
      );
    }

    await Promise.all(tasks);

    if (entries.length === 0) return null;
    return { entries };
  } catch {
    return null;
  }
}
