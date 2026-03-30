import JSZip from "jszip";
import type { ChatMessage, JournalEntry, KnowledgeBaseLibrarySnapshot } from "./hooks/useJournalHistory";

const ROOT = "selfmeridian-knowledge-base";

/** Split marker between multiple entries in one daily Markdown file (export/import round-trip). */
export const KNOWLEDGE_BASE_ENTRY_BOUNDARY = "<!-- SelfMeridian:entry-boundary -->";

function isConversationEntry(e: JournalEntry): boolean {
  return e.entrySource === "conversation";
}

function slugFilePart(s: string): string {
  return s.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "entry";
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

function yamlScalar(s: string): string {
  if (s === "") return '""';
  if (/[\n":]/.test(s) || s.trim() !== s) return JSON.stringify(s);
  return s;
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

function libraryItemToMarkdown(
  item: KnowledgeBaseLibrarySnapshot["books"][number],
  category: keyof KnowledgeBaseLibrarySnapshot
): string {
  const lines = [
    "---",
    `id: ${item.id}`,
    `title: ${yamlScalar(item.title)}`,
    item.author ? `author: ${yamlScalar(item.author)}` : "author: ",
    item.date_completed ? `date_completed: ${yamlScalar(item.date_completed)}` : "date_completed: ",
    `category: ${category}`,
    "---",
    "",
    item.note?.trim() ? item.note.trim() : "_No notes._",
    "",
  ];
  return lines.join("\n");
}

function libraryFileName(item: KnowledgeBaseLibrarySnapshot["books"][number]): string {
  const t = slugFilePart(item.title).slice(0, 60);
  const tail = item.id.replace(/[^a-zA-Z0-9-_]/g, "").slice(-10);
  return `${t}_${tail}.md`;
}

function readme(): string {
  return [
    "# SelfMeridian knowledge base",
    "",
    "This archive mirrors **The Brain → Knowledge base** layout:",
    "",
    "- `journals/<year>/<Month>/YYYY-MM-DD.md` — one file per calendar day; multiple sessions appear in time order, separated by `<!-- SelfMeridian:entry-boundary -->`",
    "- `conversations/<year>/<Month>/` — same one-file-per-day layout for live AI chats",
    "- `library/books|podcasts|articles|research/` — media library notes",
    "",
    "Each daily file contains one or more entries (YAML front matter + transcript each). Entries are separated by `<!-- SelfMeridian:entry-boundary -->`. Transcript JSON for re-import is at the end of each entry block.",
    "",
  ].join("\n");
}

export async function buildKnowledgeBaseMarkdownZip(
  entries: JournalEntry[],
  library: KnowledgeBaseLibrarySnapshot
): Promise<Blob> {
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

  (["books", "podcasts", "articles", "research"] as const).forEach((cat) => {
    for (const item of library[cat]) {
      root.file(`library/${cat}/${libraryFileName(item)}`, libraryItemToMarkdown(item, cat));
    }
  });

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

/** YYYY-MM-DD (and common variants) anywhere in a relative path or file name. */
function parseIsoLikeDateFromPath(rel: string): string | null {
  const normalized = rel.replace(/\\/g, "/");
  const patterns = [
    /(20\d{2})-(\d{2})-(\d{2})/,
    /(20\d{2})_(\d{2})_(\d{2})/,
    /(20\d{2})\.(\d{2})\.(\d{2})/,
  ];
  for (const pat of patterns) {
    const m = normalized.match(pat);
    if (m) {
      const iso = `${m[1]}-${m[2]}-${m[3]}T12:00:00.000Z`;
      if (!Number.isNaN(Date.parse(iso))) return iso;
    }
  }
  const compact = normalized.match(/(?:^|[^\d])(20\d{2})(\d{2})(\d{2})(?:[^\d]|$)/);
  if (compact) {
    const iso = `${compact[1]}-${compact[2]}-${compact[3]}T12:00:00.000Z`;
    if (!Number.isNaN(Date.parse(iso))) return iso;
  }
  return null;
}

function parseRecordedLineDate(body: string): string | null {
  const m = body.match(/\*\*Recorded:\*\*\s*([^\n]+)/);
  if (!m) return null;
  const parsed = Date.parse(m[1].trim());
  if (Number.isNaN(parsed)) return null;
  return new Date(parsed).toISOString();
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
 * Uses YAML `date:`, then **Recorded:**, then path/filename — same precedence as ZIP import.
 */
export function extractJournalEntriesFromMarkdownDump(
  relativePath: string,
  rawText: string
): Array<{ date: string | null; transcript: ChatMessage[] }> {
  const text = rawText.trim();
  if (!text) return [];
  const pathDate = parseIsoLikeDateFromPath(relativePath);
  const segments = text.includes(KNOWLEDGE_BASE_ENTRY_BOUNDARY)
    ? text.split(KNOWLEDGE_BASE_ENTRY_BOUNDARY).map((s) => s.trim()).filter(Boolean)
    : [text];
  const out: Array<{ date: string | null; transcript: ChatMessage[] }> = [];

  for (const seg of segments) {
    const { front, body } = parseSimpleFrontmatter(seg);
    let dateStr: string | null = null;
    if (typeof front.date === "string" && !Number.isNaN(Date.parse(front.date))) {
      dateStr = front.date;
    } else {
      const rec = parseRecordedLineDate(body);
      if (rec) dateStr = rec;
      else if (pathDate) dateStr = pathDate;
    }
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
  pathHint: "journal" | "conversation" | null
): JournalEntry | null {
  const transcript = parseTranscriptJson(body);
  if (!transcript) return null;
  const id = typeof front.id === "string" ? front.id : `import-${Date.now()}`;
  const date = typeof front.date === "string" && !Number.isNaN(Date.parse(front.date)) ? front.date : new Date().toISOString();
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

function libraryItemFromMarkdown(body: string, front: SimpleFront) {
  const title = typeof front.title === "string" ? front.title.trim() : "";
  if (!title) return null;
  const id = typeof front.id === "string" ? front.id : `lib-${Date.now()}`;
  let note = body.trim();
  if (note === "_No notes._") note = "";
  return {
    id,
    title,
    author: front.author?.trim() || undefined,
    date_completed: front.date_completed?.trim() || undefined,
    note: note || undefined,
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

function pathEntryKind(rel: string): "journal" | "conversation" | "library" | "skip" {
  const p = normalizePath(rel);
  if (p === "README.md" || !p.endsWith(".md")) return "skip";
  if (p.startsWith("journals/")) return "journal";
  if (p.startsWith("conversations/")) return "conversation";
  if (p.startsWith("library/")) return "library";
  return "skip";
}

function libraryCategoryFromPath(rel: string): keyof KnowledgeBaseLibrarySnapshot | null {
  const p = normalizePath(rel);
  const m = p.match(/^library\/(books|podcasts|articles|research)\//);
  if (!m) return null;
  return m[1] as keyof KnowledgeBaseLibrarySnapshot;
}

export async function parseKnowledgeBaseMarkdownZip(file: File): Promise<{
  entries: JournalEntry[];
  library: KnowledgeBaseLibrarySnapshot;
} | null> {
  try {
    const zip = await JSZip.loadAsync(await file.arrayBuffer());
    const entries: JournalEntry[] = [];
    const library: KnowledgeBaseLibrarySnapshot = { books: [], podcasts: [], articles: [], research: [] };

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

          const { front, body } = parseSimpleFrontmatter(text);

          if (kind === "journal" || kind === "conversation") {
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
                kind === "conversation" ? "conversation" : "journal"
              );
              if (je) entries.push(je);
            }
            return;
          }

          if (kind === "library") {
            const cat = libraryCategoryFromPath(rel);
            if (!cat) return;
            const item = libraryItemFromMarkdown(body, front);
            if (item) library[cat].push(item);
          }
        })()
      );
    }

    await Promise.all(tasks);

    if (entries.length === 0 && library.books.length + library.podcasts.length + library.articles.length + library.research.length === 0) {
      return null;
    }
    return { entries, library };
  } catch {
    return null;
  }
}
