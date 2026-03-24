import JSZip from "jszip";
import type { ChatMessage, JournalEntry, KnowledgeBaseLibrarySnapshot } from "./hooks/useJournalHistory";

const ROOT = "selfmeridian-knowledge-base";

function isConversationEntry(e: JournalEntry): boolean {
  return e.entrySource === "conversation";
}

function slugFilePart(s: string): string {
  return s.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "entry";
}

function entryRelPath(e: JournalEntry, getFormattedDate: (e: JournalEntry) => string, section: "journals" | "conversations"): string {
  const d = new Date(e.date);
  const year = d.getFullYear();
  const monthName = d.toLocaleString("en-US", { month: "long" });
  const label = slugFilePart(getFormattedDate(e));
  const idTail = slugFilePart(e.id).slice(-14);
  return `${section}/${year}/${monthName}/${label}_${idTail}.md`;
}

function yamlScalar(s: string): string {
  if (s === "") return '""';
  if (/[\n":]/.test(s) || s.trim() !== s) return JSON.stringify(s);
  return s;
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
    "- `journals/<year>/<Month>/` — journal sessions (not from live AI chat)",
    "- `conversations/<year>/<Month>/` — conversation transcripts",
    "- `library/books|podcasts|articles|research/` — media library notes",
    "",
    "Each entry is Markdown with YAML front matter. Transcript data for round-trip import is stored in the trailing `json` code block.",
    "",
  ].join("\n");
}

export async function buildKnowledgeBaseMarkdownZip(
  entries: JournalEntry[],
  library: KnowledgeBaseLibrarySnapshot,
  getFormattedDate: (e: JournalEntry) => string
): Promise<Blob> {
  const zip = new JSZip();
  const root = zip.folder(ROOT);
  if (!root) throw new Error("Failed to create zip root");

  root.file("README.md", readme());

  for (const e of entries) {
    const section = isConversationEntry(e) ? "conversations" : "journals";
    const path = entryRelPath(e, getFormattedDate, section);
    root.file(path, entryToMarkdown(e));
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
            const je = journalEntryFromMarkdown(body, front, kind === "conversation" ? "conversation" : "journal");
            if (je) entries.push(je);
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
