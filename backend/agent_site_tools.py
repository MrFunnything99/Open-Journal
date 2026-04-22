"""
Server-side actions the journal chat agent may invoke via OpenRouter tool calls.

Tools implemented here:
- update_library_item — update a Semantic Memory (consumed_media) row by id or title search
- navigate_ui — no server mutation; returns a client action envelope for the SPA allowlist

Guardrails: allowlisted tool names only; payload size limits; structured logging.
"""
from __future__ import annotations

import re
from typing import Any

from library import list_consumed_media, update_consumed_media

_MAX_NOTE_LEN = 2000
_MAX_TITLE_QUERY = 240
_MAX_LOG_ARG_LEN = 400

_ALLOWED_VIEWS = frozenset({"home", "chat", "brain"})
_ALLOWED_BRAIN = frozenset({"knowledgeBase"})


def _norm_title(s: str) -> str:
    t = (s or "").lower().strip()
    return re.sub(r"\s+", " ", t)


def find_consumed_media_row_by_title_query(title_query: str, instance_id: str) -> dict | None:
    q = _norm_title(title_query)
    if len(q) < 2:
        return None
    rows = list_consumed_media(instance_id)
    best: dict | None = None
    best_score = 0
    for r in rows:
        tit = (r.get("title") or "").strip()
        if not tit:
            continue
        tn = _norm_title(tit)
        if q == tn:
            return r
        score = 0
        if q in tn:
            score = len(q)
        elif tn in q:
            score = len(tn)
        else:
            q_tokens = set(q.split())
            t_tokens = set(tn.split())
            if q_tokens and t_tokens:
                inter = len(q_tokens & t_tokens)
                if inter:
                    score = inter
        if score > best_score:
            best_score = score
            best = r
    return best if best_score > 0 else None


def tool_update_library_item(raw_args: object, instance_id: str) -> tuple[dict[str, Any], str]:
    """Update Semantic Memory metadata (notes, completion date, title, creator, URL in notes)."""
    if not isinstance(raw_args, dict):
        return {"ok": False, "error": "invalid_args"}, "Library update failed: invalid arguments"
    item_id_raw = (raw_args.get("item_id") or "").strip()
    title_query = (raw_args.get("title_query") or "").strip()[:_MAX_TITLE_QUERY]
    has_note = "note" in raw_args
    has_date = "date_completed" in raw_args
    has_title = "new_title" in raw_args
    has_author = "new_author" in raw_args
    has_url = "new_url" in raw_args
    note_s = None if not has_note else str(raw_args.get("note") or "").strip()[:_MAX_NOTE_LEN]
    date_s = None if not has_date else str(raw_args.get("date_completed") or "").strip()[:50]
    new_title = None if not has_title else str(raw_args.get("new_title") or "").strip()[:500]
    new_author = None if not has_author else str(raw_args.get("new_author") or "").strip()[:300]
    new_url = None if not has_url else str(raw_args.get("new_url") or "").strip()[:500]

    if not item_id_raw and not title_query:
        return {"ok": False, "error": "need_item_id_or_title_query"}, (
            "Library update skipped: need item id or title to search"
        )
    if not (has_note or has_date or has_title or has_author or has_url):
        return {"ok": False, "error": "no_fields_to_update"}, (
            "Library update skipped: provide note, date_completed, new_title, new_author, and/or new_url"
        )

    existing: dict | None = None
    if item_id_raw:
        try:
            iid = int(item_id_raw)
        except ValueError:
            iid = None
        if iid is not None:
            for r in list_consumed_media(instance_id):
                if int(r.get("id") or 0) == iid:
                    existing = r
                    break
    if existing is None and title_query:
        existing = find_consumed_media_row_by_title_query(title_query, instance_id)

    if not existing:
        return {"ok": False, "error": "item_not_found"}, (
            f"No library item matched “{title_query[:60]}”" if title_query else "Library item not found"
        )

    eid = int(existing["id"])
    cat = (existing.get("category") or "").strip()
    title_cur = (existing.get("title") or "").strip()
    author_cur = (existing.get("creator_or_source") or "").strip() or None
    notes_cur = (existing.get("notes") or "").strip()

    final_title = new_title if has_title and new_title else title_cur
    final_author = new_author if has_author else author_cur
    if has_author and new_author == "":
        final_author = None

    if has_note:
        final_notes = note_s or ""
    else:
        final_notes = notes_cur

    if has_url and new_url and new_url.strip():
        extra = f"URL: {new_url.strip()}"
        if extra not in final_notes:
            final_notes = f"{final_notes}\n{extra}".strip() if final_notes else extra

    final_consumed_on = date_s if has_date else (existing.get("consumed_on") or None)
    if has_date and date_s == "":
        final_consumed_on = None

    ok = update_consumed_media(
        instance_id=instance_id,
        item_id=eid,
        category=cat,
        title=final_title,
        creator_or_source=final_author,
        notes=final_notes or None,
        consumed_on=final_consumed_on,
    )
    if not ok:
        return {"ok": False, "error": "update_failed", "item_id": eid}, "Library update failed"

    parts = ["Updated library item"]
    if title_cur:
        parts[0] += f" “{title_cur[:80]}{'…' if len(title_cur) > 80 else ''}”"
    if note_s is not None:
        parts.append("note set")
    if date_s is not None:
        parts.append(f"date {date_s}")
    if new_title is not None and str(new_title).strip():
        parts.append("title updated")
    if new_author is not None:
        parts.append("author updated")
    if new_url is not None:
        parts.append("url updated")
    summ = " — ".join(parts[:3])
    return {"ok": True, "item_id": str(eid), "title": title_cur or None}, summ


def tool_navigate_ui(raw_args: object) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    """
    Returns (result_for_llm, agent_step_summary, client_action_or_none).
    client_action shape: { "type": "navigate", "view": PersonaplexView, ... }
    """
    if not isinstance(raw_args, dict):
        return {"ok": False, "error": "invalid_args"}, "Navigate failed: invalid arguments", None
    view = str(raw_args.get("view", "")).strip().lower()
    if view == "journal":
        view = "chat"
    if view not in _ALLOWED_VIEWS:
        return {"ok": False, "error": "invalid_view"}, f"Navigate rejected: unknown view \u201c{view}\u201d", None
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
