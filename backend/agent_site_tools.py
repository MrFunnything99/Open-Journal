"""
Server-side actions the journal chat agent may invoke via OpenRouter tool calls.

API inventory (related FastAPI routes):
- Library: GET/PATCH/DELETE /library/{id}, POST /library-notes, POST /library/bulk-import
- Recommendations consumed: POST /recommendations/consumed -> add_consumed()
- Calendar: POST /calendar-day-summary (read-only AI summary; no subtitle persistence in backend)
- Memory wipe / KB reset: POST /memory-wipe, POST /memory-reset-knowledge-base-import (NOT exposed to agent)

Tools implemented here:
- update_library_item — PATCH-equivalent via update_consumed (by id or title search)
- mark_recommendation_consumed — same persistence as POST /recommendations/consumed
- navigate_ui — no server mutation; returns a client action envelope for the SPA allowlist

Guardrails: allowlisted tool names only; payload size limits; structured logging.
"""
from __future__ import annotations

import re
from typing import Any

from library import add_consumed, list_consumed, update_consumed

_MAX_NOTE_LEN = 2000
_MAX_TITLE_QUERY = 240
_MAX_LOG_ARG_LEN = 400


def _norm_title(s: str) -> str:
    t = (s or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def find_consumed_item_by_title_query(title_query: str, instance_id: str) -> dict[str, str] | None:
    """Return {\"id\",\"title\",\"media_type\"} or None. media_type is book|podcast|article|research."""
    q = _norm_title(title_query)
    if len(q) < 2:
        return None
    data = list_consumed(200, instance_id)
    best: dict[str, str] | None = None
    best_score = 0
    type_keys = ("books", "podcasts", "articles", "research")
    media_map = {"books": "book", "podcasts": "podcast", "articles": "article", "research": "research"}
    for key in type_keys:
        for item in data.get(key) or []:
            tid = (item.get("id") or "").strip()
            tit = (item.get("title") or "").strip()
            if not tid or not tit:
                continue
            tn = _norm_title(tit)
            if q == tn:
                return {"id": tid, "title": tit, "media_type": media_map[key]}
            score = 0
            if q in tn:
                score = len(q)
            elif tn in q:
                score = len(tn)
            else:
                # token overlap
                q_tokens = set(q.split())
                t_tokens = set(tn.split())
                if q_tokens and t_tokens:
                    inter = len(q_tokens & t_tokens)
                    if inter:
                        score = inter
            if score > best_score:
                best_score = score
                best = {"id": tid, "title": tit, "media_type": media_map[key]}
    if best_score == 0:
        return None
    return best


def tool_update_library_item(raw_args: object, instance_id: str) -> tuple[dict[str, Any], str]:
    """
    Apply library note/date update. Returns (result_json_for_llm, agent_step_summary).
    """
    if not isinstance(raw_args, dict):
        return {"ok": False, "error": "invalid_args"}, "Library update failed: invalid arguments"
    item_id = (raw_args.get("item_id") or "").strip()
    title_query = (raw_args.get("title_query") or "").strip()[: _MAX_TITLE_QUERY]
    has_note = "note" in raw_args
    has_date = "date_completed" in raw_args
    note_s = None if not has_note else str(raw_args.get("note") or "").strip()[:_MAX_NOTE_LEN]
    date_s = None if not has_date else str(raw_args.get("date_completed") or "").strip()[:50]

    if not item_id and not title_query:
        return {"ok": False, "error": "need_item_id_or_title_query"}, "Library update skipped: need item id or title to search"
    if not has_note and not has_date:
        return {"ok": False, "error": "no_fields_to_update"}, "Library update skipped: provide note and/or date_completed"

    resolved_id = item_id
    resolved_title = ""
    if not resolved_id and title_query:
        hit = find_consumed_item_by_title_query(title_query, instance_id)
        if not hit:
            return {"ok": False, "error": "item_not_found", "title_query": title_query}, f"No library item matched “{title_query[:60]}”"
        resolved_id = hit["id"]
        resolved_title = hit["title"]

    ok = update_consumed(
        resolved_id,
        date_completed=date_s,
        note=note_s,
        instance_id=instance_id,
    )
    if not ok:
        return {"ok": False, "error": "update_failed", "item_id": resolved_id}, "Library update failed (item missing or wrong instance)"
    parts = ["Updated library item"]
    if resolved_title:
        parts[0] += f" “{resolved_title[:80]}{'…' if len(resolved_title) > 80 else ''}”"
    if note_s is not None:
        parts.append("note set")
    if date_s is not None:
        parts.append(f"date {date_s}")
    summ = " — ".join(parts[:3])
    return {"ok": True, "item_id": resolved_id, "title": resolved_title or None}, summ


def tool_mark_recommendation_consumed(raw_args: object, instance_id: str) -> tuple[dict[str, Any], str]:
    if not isinstance(raw_args, dict):
        return {"ok": False, "error": "invalid_args"}, "Mark consumed failed: invalid arguments"
    ctype = str(raw_args.get("type", "article")).lower().strip()
    if ctype not in ("book", "podcast", "article", "research"):
        ctype = "article"
    title = (raw_args.get("title") or "").strip()
    if not title:
        return {"ok": False, "error": "title_required"}, "Mark consumed skipped: title required"
    author = (raw_args.get("author") or "").strip() or None
    url = (raw_args.get("url") or "").strip() or None
    liked = raw_args.get("liked", True)
    if isinstance(liked, str):
        liked = liked.lower() in ("1", "true", "yes", "y")
    liked = bool(liked)
    try:
        add_consumed(ctype, title, author=author, url=url, liked=liked, instance_id=instance_id)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}, f"Could not mark consumed: {str(e)[:60]}"
    summ = f"Recorded as consumed: {ctype} “{title[:72]}{'…' if len(title) > 72 else ''}”"
    return {"ok": True, "type": ctype, "title": title}, summ


_ALLOWED_VIEWS = frozenset({"home", "chat", "brain", "recommendations"})
_ALLOWED_BRAIN = frozenset({"knowledgeBase", "calendar"})


def tool_navigate_ui(raw_args: object) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    """
    Returns (result_for_llm, agent_step_summary, client_action_or_none).
    client_action shape: { \"type\": \"navigate\", \"view\": PersonaplexView, ... }
    """
    if not isinstance(raw_args, dict):
        return {"ok": False, "error": "invalid_args"}, "Navigate failed: invalid arguments", None
    view = str(raw_args.get("view", "")).strip().lower()
    # accept synonyms
    if view == "journal":
        view = "chat"
    if view not in _ALLOWED_VIEWS:
        return {"ok": False, "error": "invalid_view"}, f"Navigate rejected: unknown view “{view}”", None
    brain_section = raw_args.get("brain_section")
    brain_s = None
    if view in ("brain",) and brain_section is not None:
        bs = str(brain_section).strip()
        if bs not in _ALLOWED_BRAIN:
            return {"ok": False, "error": "invalid_brain_section"}, "Navigate rejected: invalid brain section", None
        brain_s = bs
    view_map = {
        "home": "voice_memo",
        "chat": "journal",
        "brain": "brain",
        "recommendations": "recommendations",
    }
    client_action: dict[str, Any] = {"type": "navigate", "view": view_map[view]}
    if brain_s:
        client_action["brainSection"] = brain_s
    label = view if view != "home" else "home"
    summ = f"UI: open {label}"
    if brain_s:
        summ += f" / {brain_s}"
    return {"ok": True, "navigated_to": view}, summ, client_action


def log_tool_invocation(name: str, instance_id: str, args_preview: str) -> None:
    inst = (instance_id or "")[:12]
    preview = (args_preview or "")[:_MAX_LOG_ARG_LEN]
    print(f"[agent_tool] name={name} instance_id={inst!r} args_preview={preview!r}")
