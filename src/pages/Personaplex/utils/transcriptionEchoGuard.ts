/**
 * Detect when STT output is effectively a copy of the last assistant message (speaker bleed,
 * silent audio hallucination, or corrupt capture). Uses prefix-token alignment (partial echoes)
 * plus normalized overlap heuristics.
 */
function normalizeForEchoCompare(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokens(s: string): string[] {
  return normalizeForEchoCompare(s)
    .split(" ")
    .map((w) => w.trim())
    .filter(Boolean);
}

/** How many leading words of the transcript match the assistant, word-for-word in order. */
function consecutivePrefixWordMatches(transcript: string, assistantText: string): { matches: number; transcriptWords: number } {
  const tw = tokens(transcript);
  const aw = tokens(assistantText);
  if (tw.length === 0 || aw.length === 0) return { matches: 0, transcriptWords: tw.length };
  let i = 0;
  while (i < tw.length && i < aw.length && tw[i] === aw[i]) {
    i += 1;
  }
  return { matches: i, transcriptWords: tw.length };
}

/**
 * True when the transcript clearly tracks the assistant's opening (common for speaker bleed /
 * STT hallucination on garbage audio).
 */
function transcriptTracksAssistantPrefix(transcript: string, assistantText: string): boolean {
  const { matches: cp, transcriptWords: n } = consecutivePrefixWordMatches(transcript, assistantText);
  if (n === 0 || cp < 3) return false;
  // Strong: first 4+ words identical in order (handles short partial echoes).
  if (cp >= 4) return true;
  // Medium transcript: most of it is the same opening phrase as the assistant.
  if (cp / n >= 0.72) return true;
  // Short user utterance: 3-word prefix match to assistant is suspicious (e.g. "yeah you mentioned").
  if (n <= 5 && cp >= 3) return true;
  return false;
}

export function transcriptLikelyEchoesAssistantText(transcript: string, assistantText: string): boolean {
  const t = normalizeForEchoCompare(transcript);
  const a = normalizeForEchoCompare(assistantText);
  if (t.length < 8 || a.length < 16) return false;
  if (t === a) return true;
  if (transcriptTracksAssistantPrefix(transcript, assistantText)) return true;

  const shorter = t.length <= a.length ? t : a;
  const longer = t.length <= a.length ? a : t;
  if (longer.includes(shorter) && shorter.length >= 20 && shorter.length / longer.length >= 0.75) {
    return true;
  }
  const tw = t.split(" ").filter((w) => w.length > 2);
  const aset = new Set(a.split(" ").filter((w) => w.length > 2));
  if (tw.length < 3) return false;
  let hit = 0;
  for (const w of tw) {
    if (aset.has(w)) hit += 1;
  }
  const ratio = hit / tw.length;
  // Short echoes: fewer total words but nearly all appear in the assistant reply.
  if (tw.length <= 8 && ratio >= 0.75) return true;
  if (tw.length >= 9 && ratio >= 0.82) return true;
  return false;
}
