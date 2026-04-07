"""Learning Tab — daily article selection and reflection orchestration."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from urllib.parse import urlparse

from decision_logger import DecisionLogger
import vec_store

# Paywalled hosts to skip when picking open-web articles (mirrors former library.py list).
_PAYWALLED_ARTICLE_DOMAINS = (
    "nytimes.com",
    "wsj.com",
    "washingtontimes.com",
    "washingtonpost.com",
    "theatlantic.com",
    "newyorker.com",
    "economist.com",
    "ft.com",
    "bloomberg.com",
    "barrons.com",
    "latimes.com",
    "bostonglobe.com",
    "chicagotribune.com",
    "harpers.org",
    "medium.com",
    "substack.com",
)


def _is_paywalled_domain(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return False
        for domain in _PAYWALLED_ARTICLE_DOMAINS:
            if host == domain or host.endswith("." + domain):
                return True
        return False
    except Exception:
        return False

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_LEARNING_MODEL = "anthropic/claude-opus-4.6"
_DEFAULT_THEME_MODEL = "openai/gpt-5.4"

_SYNTHESIS_LOGGER_NAME = "selfmeridian.learning_synthesis"


def _get_learning_synthesis_logger() -> logging.Logger:
    """File logger for full Grok theme synthesis (raw + parsed). Idempotent handler attach."""
    lg = logging.getLogger(_SYNTHESIS_LOGGER_NAME)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    lg.propagate = False
    env_path = (os.getenv("LEARNING_SYNTHESIS_LOG") or "").strip()
    if env_path.lower() in ("0", "false", "off", "disable", "none"):
        lg.addHandler(logging.NullHandler())
        return lg
    path = env_path or str(Path(__file__).resolve().parent / "logs" / "learning_synthesis.log")
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        lg.addHandler(fh)
    except OSError as e:
        print(f"[backend] Learning: could not open synthesis log {path}: {e}", flush=True)
        lg.addHandler(logging.NullHandler())
    return lg


def _log_theme_synthesis(
    instance_id: str,
    source: str,
    raw_response: str,
    brief: dict | None,
    parse_ok: bool,
) -> None:
    """Append one JSON record with full model output and normalized brief (never raises)."""
    try:
        lg = _get_learning_synthesis_logger()
        payload = {
            "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "instance_id": instance_id or "",
            "source": source,
            "parse_ok": parse_ok,
            "theme_model": _theme_model(),
            "raw_response": raw_response or "",
            "normalized_brief": brief,
        }
        lg.info(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[backend] Learning: synthesis log failed: {e}", flush=True)

_SPAM_TITLE_PATTERNS = re.compile(
    r"unveiling|journey of discovery|transform your|life.changing|"
    r"you won.t believe|secrets? (to|of)|shocking|amazing tips|"
    r"click here|sponsored|advertis|buy now|"
    r"essay example|sample essay|write (an?|your) essay|essay writing|"
    r"homework help|assignment help|term paper|coursework",
    re.IGNORECASE,
)

_JUNK_DOMAINS = {
    "edubirdie.com", "hub.edubirdie.com",
    "papersowl.com", "hub.papersowl.com",
    "ipl.org",
    "essaymechanic.com",
    "aithor.com",
    "gradesfixer.com",
    "studymoose.com",
    "bartleby.com",
    "coursehero.com",
    "chegg.com",
    "brainly.com",
    "123helpme.com",
    "essayshark.com",
    "studycorgi.com",
    "ivypanda.com",
    "freebooksummary.com",
    "getyourselfintocollege.com",
    "writingbros.com",
    "phdessay.com",
    "enotes.com",
    "ukessays.com",
    "essaypro.com",
    "graduateway.com",
    "speedypaper.com",
    "wowessays.com",
    "samploon.com",
    "nerdyseal.com",
    "studydriver.com",
}


def _learning_model() -> str:
    """OpenRouter slug for article search, pick, and reflection chat (Claude / graph.py)."""
    return (os.getenv("OPENROUTER_LEARNING_MODEL") or _DEFAULT_LEARNING_MODEL).strip()


def _theme_model() -> str:
    """GPT-5.4 via OpenRouter: bookends journals → per-theme notes for Claude."""
    return (os.getenv("OPENROUTER_LEARNING_GROK_MODEL") or _DEFAULT_THEME_MODEL).strip()


def _model_supports_openrouter_reasoning(model: str) -> bool:
    m = model.lower()
    return m.startswith("x-ai/") or "/grok" in m


def _openrouter_key() -> str:
    return (os.getenv("OPENROUTER_API_KEY") or "").strip()


def _openrouter_call(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    timeout_sec: float = 45.0,
    reasoning: bool = False,
    web_plugin: bool = False,
) -> tuple[str, dict]:
    """Call OpenRouter chat completions. Returns (content, full_response_data).

    When reasoning=True, enables reasoning tokens for xAI/Grok models only.
    When web_plugin=True, enables OpenRouter web search (native for Anthropic).
    """
    key = _openrouter_key()
    if not key:
        return "", {}

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning and _model_supports_openrouter_reasoning(model):
        payload["reasoning"] = {"enabled": True}
    if web_plugin and ":online" not in model.lower():
        payload["plugins"] = [{"id": "web", "max_results": 8}]

    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://selfmeridian.local"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "SelfMeridian"),
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = ""
        try:
            err = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        print(f"[backend] Learning OpenRouter error: {e.code} {err}", flush=True)
        return "", {}
    except Exception as e:
        print(f"[backend] Learning OpenRouter error: {e}", flush=True)
        return "", {}

    choices = data.get("choices") or []
    if not choices:
        return "", data
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    return content.strip(), data


def _verified_urls_from_openrouter(data: dict) -> list[dict]:
    """URLs from OpenRouter web plugin / citations only (not model-invented links)."""
    seen: set[str] = set()
    out: list[dict] = []

    for choice in data.get("choices") or []:
        msg = choice.get("message") or {}
        for ann in msg.get("annotations") or []:
            if ann.get("type") != "url_citation":
                continue
            uc = ann.get("url_citation") or {}
            url = (uc.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({
                "url": url,
                "title": (uc.get("title") or "").strip(),
                "snippet": (uc.get("content") or "").strip(),
            })

    for url in data.get("citations") or []:
        url = (url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"url": url, "title": "", "snippet": ""})

    return out


def _today_str() -> str:
    return date.today().isoformat()


def _journal_entries_any_instance(start_date: str, end_date: str, limit: int = 20) -> list[dict]:
    """Query journal entries across ALL instances (no instance_id filter)."""
    import sqlite3
    conn = vec_store._get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, raw_text, session_id, entry_date, created_at
            FROM journal_entries
            WHERE date(entry_date) >= date(?) AND date(entry_date) <= date(?)
            ORDER BY entry_date DESC, id DESC
            LIMIT ?
            """,
            [start_date[:10], end_date[:10], limit],
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "id": r[0],
            "document": r[1] or "",
            "session_id": r[2] or "",
            "timestamp": r[3] or "",
            "created_at": r[4] if len(r) > 4 else None,
        }
        for r in rows
    ]


def _format_learning_entry(label: str, e: dict) -> str:
    ts = e.get("timestamp") or e.get("created_at") or ""
    doc = (e.get("document") or "").strip()
    if not doc:
        return ""
    return f"{label}\nDate: {ts}\n{doc[:12000]}"


def _extract_json_object(raw: str) -> dict | None:
    """Parse first top-level JSON object from model text (handles nested arrays)."""
    if not raw or not isinstance(raw, str):
        return None
    t = raw.strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else t
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(t[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_grok_brief(obj: dict) -> dict | None:
    """Build canonical brief: theme_notes[{theme, notes}], themes[], optional summary for legacy."""
    if not isinstance(obj, dict):
        return None

    tn_raw = obj.get("theme_notes")
    if isinstance(tn_raw, list) and tn_raw:
        theme_notes: list[dict] = []
        for item in tn_raw[:6]:
            if not isinstance(item, dict):
                continue
            label = (
                item.get("theme")
                or item.get("label")
                or item.get("title")
                or ""
            )
            label = str(label).strip()
            notes = (
                item.get("notes")
                or item.get("text")
                or item.get("paragraph")
                or ""
            )
            notes = str(notes).strip()
            if not label or len(notes) < 25:
                continue
            theme_notes.append({"theme": label, "notes": notes})
        if len(theme_notes) < 3:
            return None
        theme_notes = theme_notes[:4]
        themes = [x["theme"] for x in theme_notes]
        return {
            "theme_notes": theme_notes,
            "themes": themes,
            "summary": "",  # deprecated; Opus uses theme_notes only
        }

    # Legacy: one synthesis paragraph + theme labels (same summary under each label for Opus)
    summary = str(obj.get("summary", "")).strip()
    themes_raw = obj.get("themes")
    if not isinstance(themes_raw, list) or len(summary) < 12:
        return None
    themes = [str(x).strip() for x in themes_raw if str(x).strip()][:4]
    if len(themes) < 3:
        return None
    theme_notes = [{"theme": th, "notes": summary} for th in themes]
    return {"theme_notes": theme_notes, "themes": themes, "summary": summary}


def _parse_grok_brief(raw: str) -> dict | None:
    obj = _extract_json_object(raw)
    if not obj:
        return None
    return _normalize_grok_brief(obj)


def _brief_for_opus(brief: dict) -> str:
    """Format Grok per-theme writeups for Claude search/pick prompts."""
    lines: list[str] = []
    for item in brief.get("theme_notes") or []:
        if not isinstance(item, dict):
            continue
        th = (item.get("theme") or "").strip()
        notes = (item.get("notes") or "").strip()
        if th and notes:
            lines.append(f"**{th}**\n{notes}")
    if lines:
        return "\n\n".join(lines)
    summary = (brief.get("summary") or "").strip()
    themes = brief.get("themes") or []
    if summary and themes:
        return f"Themes: {', '.join(themes)}\n\n{summary}"
    return ", ".join(themes) if themes else ""


def _grok_synthesize_journal_bookends(
    instance_id: str, recent_entries: list[dict]
) -> dict | None:
    """GPT-5.4 reads the 3 most recent journal entries; returns 3-4 themes with 2-3 sentences each."""
    recent = [e for e in recent_entries if (e.get("document") or "").strip()]
    if not recent:
        return None

    parts = ["THREE MOST RECENT JOURNAL ENTRIES (what is on their mind now):"]
    for i, e in enumerate(recent, 1):
        ts = e.get("timestamp") or e.get("created_at") or ""
        doc = (e.get("document") or "").strip()[:30000]
        parts.append(f"\n--- Entry {i} (date: {ts}) ---\n{doc}")
    blocks = ["\n".join(parts)]

    prompt = (
        "You are preparing a brief for another AI (Claude) that will find one high-quality article for this journaler.\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        "Instructions:\n"
        "1. Read the recent entries carefully. Identify what is on their mind right now.\n"
        "2. Identify exactly 3 or 4 distinct themes (short labels, 2-6 words each). Be concrete — not generic "
        "like 'personal growth'.\n"
        "3. For EACH theme, write 2-3 sentences grounded in what they actually wrote. "
        "Do not write one overall essay — each theme gets its own mini write-up. No bullet points inside notes.\n\n"
        'Respond with ONLY valid JSON in this exact shape:\n'
        '{"theme_notes": [\n'
        '  {"theme": "short label", "notes": "Two or three sentences about this theme for this person."},\n'
        '  {"theme": "...", "notes": "..."}\n'
        "]}\n"
        "Include 3 or 4 objects in theme_notes."
    )
    raw, _ = _openrouter_call(
        prompt,
        model=_theme_model(),
        temperature=0.35,
        max_tokens=2048,
        timeout_sec=90,
    )
    brief = _parse_grok_brief(raw)
    _log_theme_synthesis(instance_id, "journal_bookends", raw or "", brief, brief is not None)
    print(
        f"[backend] Learning: theme synthesis — themes={brief.get('themes') if brief else None} "
        f"(full output: logs/learning_synthesis.log or LEARNING_SYNTHESIS_LOG)",
        flush=True,
    )
    return brief


def _grok_synthesize_consumed_media(instance_id: str, consumed_rows: list[dict]) -> dict | None:
    """When there are no journal entries, infer per-theme notes from consumed media (Grok)."""
    if not consumed_rows:
        return None
    titles = [r.get("title", "") for r in consumed_rows[:30] if r.get("title")]
    subjects = []
    for r in consumed_rows[:30]:
        s = r.get("subjects") or ""
        if s:
            subjects.append(s)
    media_summary = "; ".join(titles[:15])
    if subjects:
        media_summary += "\nSubjects: " + "; ".join(subjects[:10])
    if not media_summary.strip():
        return None

    prompt = (
        "This user has no journal text yet, but we know media they consumed. "
        "Prepare a brief for Claude, which will find one substantive article.\n\n"
        f"Media: {media_summary}\n\n"
        "1. Identify exactly 3 or 4 themes (short labels, 2-6 words), inferred from this media — not generic.\n"
        "2. For EACH theme, write 2-3 sentences on why it likely matters to this person given what they consumed.\n\n"
        'Respond with ONLY valid JSON:\n'
        '{"theme_notes": [\n'
        '  {"theme": "label", "notes": "Two or three sentences."}\n'
        "]}\n"
        "Include 3 or 4 objects."
    )
    raw, _ = _openrouter_call(
        prompt,
        model=_theme_model(),
        temperature=0.35,
        max_tokens=1536,
        timeout_sec=60,
    )
    brief = _parse_grok_brief(raw)
    _log_theme_synthesis(instance_id, "consumed_media", raw or "", brief, brief is not None)
    return brief


def _fallback_brief_no_signal() -> dict:
    theme_notes = [
        {
            "theme": "intellectual curiosity",
            "notes": "Little is known about their specific situation. They may still enjoy ideas that reward close reading and surprise.",
        },
        {
            "theme": "human behavior and institutions",
            "notes": "Thoughtful essays on how people and systems shape each other often resonate without deep personal context.",
        },
        {
            "theme": "meaning, work, and rest",
            "notes": "Many readers connect to pieces that examine purpose and sustainability without prescribing a single lifestyle.",
        },
        {
            "theme": "mental life and self-understanding",
            "notes": "Psychology- and philosophy-informed journalism can open reflection even when journals are sparse.",
        },
    ]
    return {
        "theme_notes": theme_notes,
        "themes": [x["theme"] for x in theme_notes],
        "summary": "",
    }


def _build_learning_brief(instance_id: str, consumed_rows: list[dict]) -> dict:
    """GPT-5.4 recent journals → per-theme notes, or consumed-media synthesis, or static fallback."""
    recent = vec_store.journal_learning_bookends(instance_id)
    if recent:
        brief = _grok_synthesize_journal_bookends(instance_id, recent)
        if brief:
            return brief
        brief2 = _grok_synthesize_consumed_media(instance_id, consumed_rows)
        if brief2:
            return brief2
        fb = _fallback_brief_no_signal()
        _log_theme_synthesis(instance_id, "fallback_no_signal", "", fb, True)
        return fb
    brief = _grok_synthesize_consumed_media(instance_id, consumed_rows)
    if brief:
        return brief
    fb = _fallback_brief_no_signal()
    _log_theme_synthesis(instance_id, "fallback_no_signal", "", fb, True)
    return fb


def _recent_journal_context(instance_id: str, num_days: int = 4) -> str:
    """Pull recent journal entries as a combined text block.

    Tries progressively wider scopes to handle instance_id rotation:
    1. Current instance + last num_days
    2. Current instance + last 365 days
    3. ALL instances + last 365 days
    """
    end = date.today()

    for scope_id, days_back, use_any in [
        (instance_id, num_days, False),
        (instance_id, 365, False),
        (instance_id, 365, True),
    ]:
        start = end - timedelta(days=days_back)
        if use_any:
            entries = _journal_entries_any_instance(start.isoformat(), end.isoformat(), limit=20)
        else:
            entries = vec_store.journal_entries_for_date_range(
                scope_id, start.isoformat(), end.isoformat(), limit=20
            )
        if entries:
            parts: list[str] = []
            for e in entries:
                ts = e.get("timestamp") or e.get("created_at") or ""
                doc = (e.get("document") or "").strip()
                if doc:
                    parts.append(f"[{ts}]\n{doc}")
            if parts:
                if use_any:
                    print(f"[backend] Learning: found {len(parts)} journal entries via cross-instance fallback ({days_back}d)", flush=True)
                else:
                    print(f"[backend] Learning: found {len(parts)} journal entries (instance={scope_id[:12]}..., {days_back}d)", flush=True)
                return "\n\n---\n\n".join(parts)
    return ""


def _consumed_context(_instance_id: str) -> tuple[str, list[dict]]:
    """No library/consumed store; empty overlap string and rows (learning uses journals only)."""
    return "", []


def _search_articles_opus(
    brief: dict, consumed_lower: str
) -> tuple[list[dict], list[str]]:
    """Claude + web search using Grok's per-theme notes (2-3 sentences each).

    Returns (candidates, search_queries_used). Only URLs present in API citations
    / annotations are trusted — never scrape URLs from prose alone.
    """
    if not _openrouter_key():
        print("[backend] Learning: no OpenRouter key — cannot run web search", flush=True)
        return [], []

    themes = brief.get("themes") or []
    themes_str = ", ".join(themes)
    opus_brief = _brief_for_opus(brief).strip()

    context_block = ""
    if opus_brief:
        context_block = (
            "\nAnother model read their earliest and latest journal entries (or their media list) and wrote "
            "this per-theme brief — use it as your guide for relevance:\n\n"
            f"{opus_brief}\n"
        )

    prompt = (
        f"Find 5-8 thought-provoking, substantive articles aligned with these themes: {themes_str}\n"
        f"{context_block}\n"
        "Requirements:\n"
        "- Real published articles from quality outlets (Aeon, Nautilus, The Conversation, "
        "NPR, Psyche, HBR, Scientific American, Wired, The Guardian, BBC Future, "
        "MIT Technology Review, Psychology Today, The Atlantic, etc.)\n"
        "- Long-form journalism, essays, or research summaries — NOT listicles, "
        "homework sites, or essay-writing services\n"
        "- Articles that would expand someone's thinking, not just echo common advice\n"
        "- Each article should be from a different source if possible\n\n"
        "In your reply, cite each source using markdown links [title](url) that match "
        "pages you actually found via search. One short line per article.\n"
    )

    content, data = _openrouter_call(
        prompt,
        model=_learning_model(),
        temperature=0.3,
        max_tokens=2048,
        timeout_sec=90,
        web_plugin=True,
    )
    if not data:
        print("[backend] Learning: OpenRouter web search returned no data", flush=True)
        return [], [themes_str] if themes_str else []

    citation_rows = _verified_urls_from_openrouter(data)
    print(f"[backend] Learning: web search returned {len(citation_rows)} verified URLs", flush=True)

    if not citation_rows:
        print("[backend] Learning: 0 API citations — skipping prose URLs", flush=True)
        return [], [themes_str]

    candidates: list[dict] = []
    seen_urls: set[str] = set()

    for row in citation_rows:
        url = row.get("url") or ""
        if not url or url in seen_urls:
            continue
        if _is_paywalled_domain(url) or _is_junk_domain(url):
            print(f"[backend] Learning: filtered citation: {url}", flush=True)
            continue
        if url.lower() in consumed_lower:
            continue
        seen_urls.add(url)

        title = (row.get("title") or "").strip() or _extract_title_for_url(url, content)
        snippet = (row.get("snippet") or "").strip()
        if _is_spam_title(title):
            continue

        candidates.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    print(f"[backend] Learning: {len(candidates)} candidates from verified citations", flush=True)
    return candidates[:8], [themes_str]


def _extract_title_for_url(url: str, content: str) -> str:
    """Try to extract a title for a URL from the model response content."""
    patterns = [
        re.compile(r'\[([^\]]+)\]\(' + re.escape(url) + r'\)'),
        re.compile(r'\d+\.\s*\[([^\]]+)\]\(' + re.escape(url) + r'\)'),
    ]
    for pat in patterns:
        m = pat.search(content)
        if m:
            return m.group(1).strip()
    from urllib.parse import urlparse
    path = urlparse(url).path.strip("/").split("/")[-1] if url else ""
    return path.replace("-", " ").replace("_", " ").title() if path else url


def _is_spam_title(title: str) -> bool:
    """Reject clickbait/SEO-spam titles."""
    if not title:
        return True
    if _SPAM_TITLE_PATTERNS.search(title):
        return True
    if title.isupper() and len(title) > 20:
        return True
    return False


def _is_junk_domain(url: str) -> bool:
    """Reject essay mills and homework-help sites."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        for junk in _JUNK_DOMAINS:
            if host == junk or host.endswith("." + junk):
                return True
    except Exception:
        pass
    return False


def _pick_best_article(candidates: list[dict], brief: dict) -> dict | None:
    """Claude picks the best article using Grok's per-theme notes."""
    if not candidates:
        return None

    candidates_text = ""
    for i, c in enumerate(candidates):
        candidates_text += (
            f"\n{i+1}. Title: {c.get('title', '')}\n"
            f"   URL: {c.get('url', '')}\n"
            f"   Snippet: {c.get('snippet', '')[:300]}\n"
        )

    themes = brief.get("themes") or []
    opus_brief = _brief_for_opus(brief).strip()
    context_block = ""
    if opus_brief:
        context_block = f"User context (per-theme notes from their journals or media):\n{opus_brief[:6000]}\n\n"

    prompt = (
        "You are selecting the single best article for a user's daily learning.\n\n"
        f"User themes: {', '.join(themes)}\n\n"
        f"{context_block}"
        f"Candidate articles:{candidates_text}\n\n"
        "Pick the SINGLE best article. It should:\n"
        "- Be a real, substantive article (not a listicle or SEO page)\n"
        "- Expand their thinking, not just echo what they already know\n"
        "- Be from a reputable source\n\n"
        "Respond with ONLY this JSON object (keep the hook under 20 words):\n"
        '{"pick": 1, "hook": "short hook sentence"}\n'
    )
    raw, _ = _openrouter_call(
        prompt, model=_learning_model(), temperature=0.4, max_tokens=512,
        timeout_sec=45, reasoning=True,
    )
    parsed = _parse_json_object(raw)
    if parsed:
        idx = int(parsed.get("pick", 1)) - 1
        hook = str(parsed.get("hook", "")).strip()
        if 0 <= idx < len(candidates):
            result = dict(candidates[idx])
            result["hook"] = hook or "This connects to themes you've been exploring."
            return result

    result = dict(candidates[0])
    result["hook"] = "This article connects to themes you've been exploring recently."
    return result


def _title_from_user_pasted_url(url: str) -> str:
    """Readable default title from a user-pasted article or podcast URL (no network fetch)."""
    from urllib.parse import unquote, urlparse

    p = urlparse(url.strip())
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if "spotify.com" in host:
        return "Podcast (Spotify)"
    if "podcasts.apple.com" in host or "itunes.apple.com" in host:
        return "Podcast (Apple Podcasts)"
    if "overcast.fm" in host:
        return "Podcast (Overcast)"
    if "youtube.com" in host or "youtu.be" in host:
        return "YouTube"
    if "substack.com" in host:
        return "Article (Substack)"
    path = unquote((p.path or "").rstrip("/"))
    seg = path.split("/")[-1] if path else ""
    if seg and len(seg) < 120 and "?" not in seg:
        pretty = seg.replace("-", " ").replace("_", " ").strip()
        if pretty and not pretty.isdigit():
            return pretty[:200]
    return (host or "Saved link").replace("www.", "")[:120] or "Saved link"


def register_user_provided_learning_url(instance_id: str, url: str) -> dict:
    """
    Store a user-pasted article or podcast URL as today's learning item (replaces curated pick for that date).
    Reuses journal / media theme brief so reflection chat still bridges to their journals.
    """
    from urllib.parse import urlparse

    raw = (url or "").strip()
    if not raw:
        raise ValueError("URL is required")
    if len(raw) > 2048:
        raise ValueError("URL is too long")
    parsed = urlparse(raw)
    if (parsed.scheme or "").lower() not in ("http", "https"):
        raise ValueError("Link must start with http:// or https://")
    if not (parsed.netloc or "").strip():
        raise ValueError("Please enter a valid URL")
    if _is_junk_domain(raw):
        raise ValueError("That domain is not allowed for learning links.")

    today = _today_str()
    _consumed_lower, consumed_rows = _consumed_context(instance_id)
    brief = _build_learning_brief(instance_id, consumed_rows)
    themes = brief.get("themes") or []
    theme_notes = brief.get("theme_notes") or []

    title = _title_from_user_pasted_url(raw)
    hook = (
        "You shared this link. Reflect on it here—the assistant will connect the conversation to your journals "
        "when you move to reflection."
    )

    vec_store.daily_article_upsert(
        instance_id=instance_id,
        entry_date=today,
        article_title=title,
        article_url=raw,
        article_snippet="",
        hook=hook,
        candidates_json=json.dumps([{"title": title, "url": raw, "source": "user_pasted"}]),
        themes_json=json.dumps(
            {
                "themes": themes,
                "theme_notes": theme_notes,
                "summary": brief.get("summary") or "",
            },
            default=str,
        ),
        search_queries_json=json.dumps(["user_pasted_link"]),
        status="pending",
    )

    return {
        "title": title,
        "url": raw,
        "hook": hook,
        "snippet": "",
        "date": today,
        "status": "pending",
        "has_article": True,
        "themes": themes,
        "theme_notes": theme_notes,
        "theme_model": _theme_model(),
        "article_model": "user_pasted_link",
    }


def _parse_json_object(text: str) -> dict | None:
    """Robustly extract a JSON object from LLM output."""
    if not text or not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else t
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*"pick"\s*:\s*\d+[^{}]*"hook"\s*:\s*"[^"]*"[^{}]*\}', t, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    match2 = re.search(r'\{.*\}', t, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(0))
        except json.JSONDecodeError:
            pass
    print(f"[backend] Learning: could not parse LLM pick response: {t[:200]}", flush=True)
    return None


def generate_daily_article(instance_id: str, force: bool = False) -> dict:
    """
    Main orchestration: generate today's article or return the cached one.

    Returns dict with keys: title, url, hook, snippet, date, status, has_article
    """
    today = _today_str()
    t0 = time.time()

    if not force:
        existing = vec_store.daily_article_get(instance_id, today)
        if existing:
            return {
                "title": existing["article_title"] or "",
                "url": existing["article_url"] or "",
                "hook": existing["hook"] or "",
                "snippet": existing["article_snippet"] or "",
                "date": existing["entry_date"],
                "status": existing["status"],
                "has_article": True,
            }

    consumed_lower, consumed_rows = _consumed_context(instance_id)
    brief = _build_learning_brief(instance_id, consumed_rows)
    themes = brief.get("themes") or []
    print(f"[backend] Learning: {_theme_model()}→{_learning_model()} brief themes: {themes}", flush=True)

    candidates, queries = _search_articles_opus(brief, consumed_lower)
    print(f"[backend] Learning: search returned {len(candidates)} candidates", flush=True)

    if not candidates:
        return {
            "title": "",
            "url": "",
            "hook": "No suitable articles found today. Check back tomorrow!",
            "snippet": "",
            "date": today,
            "status": "pending",
            "has_article": False,
        }

    winner = _pick_best_article(candidates, brief)
    if not winner:
        return {
            "title": "",
            "url": "",
            "hook": "Could not select an article today.",
            "snippet": "",
            "date": today,
            "status": "pending",
            "has_article": False,
        }

    elapsed_ms = int((time.time() - t0) * 1000)

    vec_store.daily_article_upsert(
        instance_id=instance_id,
        entry_date=today,
        article_title=winner.get("title", ""),
        article_url=winner.get("url", ""),
        article_snippet=winner.get("snippet", ""),
        hook=winner.get("hook", ""),
        candidates_json=json.dumps(candidates, default=str),
        themes_json=json.dumps(
            {
                "themes": themes,
                "theme_notes": brief.get("theme_notes") or [],
                "summary": brief.get("summary") or "",
            },
            default=str,
        ),
        search_queries_json=json.dumps(queries),
        status="pending",
    )

    DecisionLogger._write(
        instance_id=instance_id,
        action_type="daily_article_selection",
        input_summary=f"theme_brief_themes={json.dumps(themes)}, queries={json.dumps(queries)}",
        retrieved_items=f"{len(candidates)} candidates from {_learning_model()} + web (brief via {_theme_model()})",
        final_output=json.dumps({"title": winner.get("title"), "url": winner.get("url"), "hook": winner.get("hook")}),
        reasoning_notes=f"Theme synthesis → Claude search; picked from {len(candidates)} web-grounded candidates",
        duration_ms=elapsed_ms,
        model_used=f"{_theme_model()} + {_learning_model()}",
        search_api_calls=json.dumps(queries),
    )

    print(
        f"[backend] Learning: selected '{winner.get('title')}' ({winner.get('url')}) in {elapsed_ms}ms",
        flush=True,
    )

    theme_notes = brief.get("theme_notes") or []
    return {
        "title": winner.get("title", ""),
        "url": winner.get("url", ""),
        "hook": winner.get("hook", ""),
        "snippet": winner.get("snippet", ""),
        "date": today,
        "status": "pending",
        "has_article": True,
        "themes": themes,
        "theme_notes": theme_notes,
        "theme_model": _theme_model(),
        "article_model": _learning_model(),
    }


def get_learning_system_prompt(instance_id: str, article_data: dict) -> str:
    """
    Build the system prompt for the learning reflection conversation.

    This gets used in graph.py when mode == "learning".
    """
    journal_context = _recent_journal_context(instance_id)
    title = article_data.get("title") or article_data.get("article_title") or ""
    url = article_data.get("url") or article_data.get("article_url") or ""
    snippet = article_data.get("snippet") or article_data.get("article_snippet") or ""
    hook = article_data.get("hook") or ""
    themes_raw = article_data.get("themes_json") or "[]"
    theme_labels: list[str] = []
    grok_summary = ""
    theme_notes_stored: list = []
    try:
        parsed = json.loads(themes_raw) if isinstance(themes_raw, str) else themes_raw
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        theme_labels = [str(x).strip() for x in (parsed.get("themes") or []) if x]
        grok_summary = str(parsed.get("summary", "")).strip()
        tn = parsed.get("theme_notes")
        if isinstance(tn, list):
            theme_notes_stored = tn
    elif isinstance(parsed, list):
        theme_labels = [str(x).strip() for x in parsed if x]

    themes_line = ", ".join(theme_labels) if theme_labels else "general personal growth"
    brief_parts: list[str] = []
    for item in theme_notes_stored:
        if not isinstance(item, dict):
            continue
        th = str(item.get("theme") or item.get("label") or "").strip()
        notes = str(item.get("notes") or item.get("text") or "").strip()
        if th and notes:
            brief_parts.append(f"• {th}: {notes}")
    if brief_parts:
        brief_line = "\n".join(brief_parts)
    elif grok_summary:
        brief_line = grok_summary
    else:
        brief_line = "(no per-theme brief stored for this pick)"

    return f"""You are a thoughtful learning companion helping the user reflect on an article, essay, or episode they read or listened to (they may have opened it via a pasted link).

ARTICLE / MEDIA:
- Title: {title}
- URL: {url}
- Snippet: {snippet[:500]}
- Why it was selected: {hook}

LEARNING CONTEXT (per-theme notes from their earliest + most recent journals, or from media): {brief_line}
USER'S LEARNING THEMES: {themes_line}

USER'S RECENT JOURNAL CONTEXT:
{journal_context[:3000]}

YOUR APPROACH:
1. Start by asking: "What stood out to you from the article?"
   - This is open-ended. Let them share their initial reaction.
   - If they say it wasn't interesting or they didn't read it, acknowledge that gracefully and note their feedback.

2. After their first response, generate 1-2 bridging reflection questions:
   - Each question MUST connect a specific idea from the article to something specific the user has written about in their journal
   - Do NOT ask generic questions like "How does this apply to your life?"
   - Instead, reference concrete details: specific situations, feelings, or decisions they've journaled about
   - Example format: "The article argues [specific claim]. You've written about [specific journal reference]. Do you see a connection?"

3. Keep the conversation natural and concise. 2-3 exchanges total is ideal.

RULES:
- Do NOT summarize the article for the user. They should have read it themselves.
- Do NOT ask more than 2 reflection questions per exchange.
- Be warm but intellectually substantive.
- If the user gives shallow responses, gently push deeper with a more specific question.
- Reference their journal content to make it personal, but don't quote it verbatim — paraphrase naturally."""
