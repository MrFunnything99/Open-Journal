"""
Server-side actions the journal chat agent may invoke via OpenRouter tool calls.

Tools implemented here:
- navigate_ui — no server mutation; returns a client action envelope for the SPA allowlist

Guardrails: allowlisted tool names only; payload size limits; structured logging.
"""
from __future__ import annotations

from typing import Any


_MAX_LOG_ARG_LEN = 400

_ALLOWED_VIEWS = frozenset({"home", "chat", "brain"})
_ALLOWED_BRAIN = frozenset({"knowledgeBase", "calendar"})


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
