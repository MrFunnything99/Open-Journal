"""
LangGraph for Selfmeridian: Interviewer (chat) and Librarian (end-session).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from library import (
    apply_library_tool_items,
    get_relevant_context_dual,
    ingest_journal_entry,
    resolve_books_via_openlibrary,
    save_resolved_books,
)
from agent_site_tools import (
    log_tool_invocation,
    tool_navigate_ui,
    tool_update_library_item,
)

# When True, /chat injects sqlite-vec journal context via get_relevant_context_dual (all modes with personalization > 0).
_GRAPH_RETRIEVAL_ENABLED = True

# State: list of messages + session_id for Librarian + personalization + intrusiveness + mode + optional retrieval log + instance_id
class JournalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    personalization: float
    intrusiveness: NotRequired[float]
    mode: NotRequired[str]  # "journal" | "conversation" | "autobiography" (UI: Assisted Journal; recommendations in main)
    # OpenRouter model id (allowlisted); used only for conversation + autobiography when set
    openrouter_model: NotRequired[str]
    retrieval_log: NotRequired[str]
    last_transcript: NotRequired[str]
    last_summary: NotRequired[str]
    last_facts: NotRequired[list]
    instance_id: NotRequired[str]  # X-Instance-ID for per-device data isolation
    agent_steps: NotRequired[list]  # UI: retrieval + tool summaries for this turn
    client_actions: NotRequired[list]  # UI: allowlisted navigate actions from navigate_ui tool
    # Optional: browser-built local time + daypart for Assisted Journal secondary check-ins
    client_time_context: NotRequired[str]
    library_items_added: NotRequired[int]  # chat agent saved N items to Semantic Memory (consumed_media)


DEFAULT_OPENROUTER_CHAT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_OPENROUTER_PRIMARY_MODEL = "anthropic/claude-opus-4.6"
DEFAULT_OPENROUTER_CHAT_FALLBACK_MODEL = DEFAULT_OPENROUTER_PRIMARY_MODEL
DEFAULT_OPENROUTER_CONVERSATION_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_ASSISTED_JOURNAL_MODEL = DEFAULT_OPENROUTER_PRIMARY_MODEL

# Allowlist for client-provided OpenRouter id (must match frontend CHAT_COMPLETION_MODEL_OPTIONS)
USER_SELECTABLE_CHAT_MODELS: frozenset[str] = frozenset(
    {
        DEFAULT_OPENROUTER_PRIMARY_MODEL,
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
    }
)


def _user_pick_openrouter_model(mode: str, requested: str | None) -> str | None:
    if mode not in ("conversation", "autobiography"):
        return None
    rid = (requested or "").strip()
    if not rid or rid not in USER_SELECTABLE_CHAT_MODELS:
        return None
    return rid


def _reasoning_extra_body_for_model(model: str) -> dict | None:
    ml = model.lower()
    if ml.startswith("x-ai/") or "/grok" in ml:
        return {"reasoning": {"enabled": True}}
    return None


def _openrouter_chat_client_and_model():
    """OpenAI-compatible client pointed at OpenRouter for /chat (journal interviewer)."""
    from openai import OpenAI

    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is required for the chat interviewer (/chat). "
            "Get a key at https://openrouter.ai/keys"
        )
    model = (os.getenv("OPENROUTER_CHAT_MODEL") or DEFAULT_OPENROUTER_CHAT_MODEL).strip()
    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://selfmeridian.local"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "SelfMeridian"),
        },
    )
    return client, model


def _openrouter_chat_client_models():
    """
    Return (client, primary_model, fallback_model) for /chat.
    Primary should be cheap + good at tool calling; fallback should be stronger for ambiguity/errors.
    """
    client, primary = _openrouter_chat_client_and_model()
    fallback = (os.getenv("OPENROUTER_CHAT_FALLBACK_MODEL") or DEFAULT_OPENROUTER_CHAT_FALLBACK_MODEL).strip()
    if not fallback:
        fallback = DEFAULT_OPENROUTER_CHAT_FALLBACK_MODEL
    if fallback == primary:
        fallback = DEFAULT_OPENROUTER_CHAT_FALLBACK_MODEL
    return client, primary, fallback


NAVIGATE_UI_TOOL = {
    "type": "function",
    "function": {
        "name": "navigate_ui",
        "description": (
            "Switch the user's main app screen when they ask to open a section (e.g. go to Brain, open Chat). "
            "Does not change server data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["home", "chat", "brain"],
                    "description": "home=journal home; chat=full chat; brain=knowledge hub",
                },
                "brain_section": {
                    "type": "string",
                    "enum": ["knowledgeBase"],
                    "description": "When view is brain, optional subsection to open",
                },
            },
            "required": ["view"],
        },
    },
}

LIBRARY_ITEMS_TOOL = {
    "type": "function",
    "function": {
        "name": "add_library_items",
        "description": (
            "Save podcasts, articles, or research papers the user has finished (or wants logged) into their "
            "Library / Semantic Memory. For BOOKS, always use extract_books_read instead — it normalizes "
            "title/author. Use when they enumerate titles or ask to add what they listened to or read (non-book). "
            "Leave url empty unless they gave a real link. "
            "Do NOT use for hypothetical picks unless they clearly want them saved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["book", "podcast", "article", "research"],
                                "description": "Media category",
                            },
                            "title": {"type": "string", "description": "Title of the work or episode"},
                            "author": {
                                "type": "string",
                                "description": "Author, show name, venue, or first author; omit if unknown",
                            },
                            "url": {
                                "type": "string",
                                "description": "Canonical URL if provided; otherwise empty string",
                            },
                            "liked": {
                                "type": "boolean",
                                "description": "True if they enjoyed or neutral; false if they disliked",
                            },
                            "note": {
                                "type": "string",
                                "description": "Optional short note from the user",
                            },
                        },
                        "required": ["type", "title"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}

UPDATE_LIBRARY_ITEM_TOOL = {
    "type": "function",
    "function": {
        "name": "update_library_item",
        "description": (
            "Update an existing Library / Semantic Memory item (note, completion date, title, author, url). "
            "Use when the user fixes a title, changes a note, or sets a completion date for something already saved. "
            "Provide item_id if known; otherwise title_query to find a match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "Numeric row id if known",
                },
                "title_query": {
                    "type": "string",
                    "description": "Substring or full title when item_id is unknown",
                },
                "note": {"type": "string", "description": "New note; include only if updating the note"},
                "date_completed": {
                    "type": "string",
                    "description": "YYYY-MM-DD; include only if updating completion date",
                },
                "new_title": {"type": "string", "description": "Corrected title; only if changing title"},
                "new_author": {"type": "string", "description": "Corrected author; only if changing author"},
                "new_url": {"type": "string", "description": "Corrected URL; only if changing URL"},
            },
        },
    },
}

EXTRACT_BOOKS_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_books_read",
        "description": (
            "Extract books the user mentions having read, finished, or consumed. "
            "Use when they say they read or finished a book — even casually. "
            "Provide raw_title (and raw_author if known); the backend normalizes spelling. "
            "Include a short note for their opinion if they gave one. "
            "Do NOT use add_library_items for books — always use this tool for books."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "books": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "raw_title": {"type": "string", "description": "Title as the user said it"},
                            "raw_author": {
                                "type": "string",
                                "description": "Author as the user said it; empty if unknown",
                            },
                            "liked": {
                                "type": "boolean",
                                "description": "True if they enjoyed or neutral; false if they disliked",
                            },
                            "note": {
                                "type": "string",
                                "description": "Short opinion or comment about the book",
                            },
                        },
                        "required": ["raw_title"],
                    },
                }
            },
            "required": ["books"],
        },
    },
}

_CHAT_TOOLS = [
    LIBRARY_ITEMS_TOOL,
    UPDATE_LIBRARY_ITEM_TOOL,
    EXTRACT_BOOKS_READ_TOOL,
    NAVIGATE_UI_TOOL,
]

_ASSISTED_JOURNAL_PROMPT = (
    "You are a journaling companion. Your role is to help the user reflect, not to advise, fix, or analyze from a distance.\n\n"
    "Current time / setting: {local_time}.\n\n"
    "You are not a general-purpose chatbot or task assistant. You are a reflective journaling companion. Your default tone is calm, curious, and slightly slower than a typical chat app.\n\n"
    "Each turn, do two things:\n\n"
    "Mirror briefly. Reflect back what they said in a sentence — the substance and the feeling underneath it. "
    "Show you actually heard them. Don't parrot; distill.\n"
    "Ask one question. Just one. Early in a conversation, keep questions broad and open "
    "(e.g. \"What's been sitting with you today?\", \"Where does your mind keep going back to?\"). "
    "As context builds, let questions get more specific and probing — surface a tension, a contradiction, "
    "an assumption they didn't notice they were making, or an angle they haven't considered. "
    "The best questions help them see something they wouldn't have seen alone.\n\n"
    "Opening behavior. If the user's first message is a greeting, a vague opener, or a bare topic ('hey', 'good afternoon', 'let's talk about work'), do not launch into deep reflection and do not respond like a general chatbot. Greet back briefly in a grounded, journal-assistant register, then offer one open invitation to begin — something like 'what's sitting with you right now?' or 'where would you like to start?'. Keep it warm and unhurried. Save mirroring for when there's actually something to mirror.\n\n"
    "If they name a topic without context ('let's talk about my job'), don't assume what about it matters. Ask what draws them to it today.\n\n"
    "Guidelines:\n\n"
    "Build before you dig. Do not ask probing or tension-surfacing questions until at least the third user turn, or until the user has shared something with clear emotional weight. Early turns stay broad and open.\n"
    "Follow their thread, not yours. If they shift, shift with them.\n"
    "No advice, no reframes-as-solutions, no \"have you tried.\" You're a mirror with curiosity, not a coach.\n"
    "Keep responses short. Two or three sentences, usually. White space is part of the work.\n"
    "Match their register. If they're casual, be casual. If they're raw, don't tidy them up."
)


def _messages_to_openai_dicts(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI chat format (system/user/assistant text only)."""
    out: list[dict] = []
    for m in messages:
        content = getattr(m, "content", str(m))
        if isinstance(content, list):
            content = " ".join(
                c.get("text", str(c)) for c in content if isinstance(c, dict)
            )
        content = (content or "").strip()
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": content})
    return out


def _assistant_message_to_dict(msg) -> dict:
    """OpenAI SDK assistant message including optional tool_calls."""
    d: dict = {"role": "assistant"}
    tcalls = getattr(msg, "tool_calls", None)
    if tcalls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tcalls
        ]
        # Tool-only turns often have null content; APIs expect explicit null, not omission.
        d["content"] = msg.content if msg.content else None
    else:
        d["content"] = msg.content or ""
    return d


def _interviewer_run_with_tools(
    client,
    model: str,
    oai_messages: list[dict],
    instance_id: str,
    *,
    max_tool_rounds: int = 5,
    extra_body: dict | None = None,
    temperature: float | None = None,
) -> tuple[AIMessage, int, list[dict], list[dict]]:
    """
    Multi-turn chat completion (OpenRouter) with allowlisted site tools.
    Returns (assistant message, library_items_saved_this_turn, agent_steps, client_actions).
    """
    total_saved = 0
    agent_steps: list[dict] = []
    client_actions: list[dict] = []
    rounds = 0
    fallback_model = (os.getenv("OPENROUTER_CHAT_FALLBACK_MODEL") or DEFAULT_OPENROUTER_CHAT_FALLBACK_MODEL).strip()
    if not fallback_model:
        fallback_model = DEFAULT_OPENROUTER_CHAT_FALLBACK_MODEL
    active_model = model
    escalated = False
    while rounds < max_tool_rounds:
        rounds += 1
        try:
            create_kwargs: dict = dict(
                model=active_model,
                messages=oai_messages,
                tools=_CHAT_TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            )
            if temperature is not None:
                create_kwargs["temperature"] = float(temperature)
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**create_kwargs)
        except Exception as e:
            # If the cheap model fails (timeouts, provider errors, tool-call glitches), retry once on fallback.
            if (not escalated) and fallback_model and active_model != fallback_model:
                escalated = True
                active_model = fallback_model
                agent_steps.append(
                    {
                        "kind": "system",
                        "summary": f"Escalated to a stronger model after an error ({str(e)[:80]})",
                    }
                )
                continue
            raise
        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            text = (msg.content or "").strip()
            return AIMessage(content=text), total_saved, agent_steps, client_actions

        oai_messages.append(_assistant_message_to_dict(msg))
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            log_tool_invocation(name, instance_id, json.dumps(args)[:400])

            if name == "add_library_items":
                try:
                    items = args.get("items", [])
                    n, labels = apply_library_tool_items(items, instance_id)
                    total_saved += n
                    result = {"ok": True, "items_added": n, "saved": labels[:24]}
                    preview = ", ".join(labels[:4]) if labels else ""
                    if n > 0:
                        summ = f"Saved {n} item(s) to Library"
                        if preview:
                            summ += f": {preview}{'…' if len(labels) > 4 else ''}"
                    else:
                        summ = "Library tool ran (no new items)"
                    agent_steps.append({"kind": "tool", "name": name, "summary": summ})
                except Exception as e:
                    result = {"ok": False, "error": str(e)[:240]}
                    agent_steps.append({
                        "kind": "tool",
                        "name": name,
                        "summary": f"Library save error: {str(e)[:80]}",
                    })
            elif name == "update_library_item":
                result, summ = tool_update_library_item(args, instance_id)
                agent_steps.append({"kind": "tool", "name": name, "summary": summ})
            elif name == "extract_books_read":
                try:
                    raw_books = args.get("books", [])
                    resolved = resolve_books_via_openlibrary(raw_books)
                    n, labels = save_resolved_books(resolved, instance_id)
                    total_saved += n
                    result = {"ok": True, "books_saved": n, "saved": labels[:24]}
                    preview = ", ".join(labels[:4]) if labels else ""
                    if n > 0:
                        summ = f"Verified & saved {n} book(s) to Library"
                        if preview:
                            summ += f": {preview}{'…' if len(labels) > 4 else ''}"
                    else:
                        summ = "Book extraction ran (no new items)"
                    agent_steps.append({"kind": "tool", "name": name, "summary": summ})
                except Exception as e:
                    result = {"ok": False, "error": str(e)[:240]}
                    agent_steps.append({
                        "kind": "tool",
                        "name": name,
                        "summary": f"Book extraction error: {str(e)[:80]}",
                    })
            elif name == "navigate_ui":
                result, summ, action = tool_navigate_ui(args)
                agent_steps.append({"kind": "tool", "name": name, "summary": summ})
                if action:
                    client_actions.append(action)
            else:
                result = {"ok": False, "error": f"unknown_tool:{name}"}
                agent_steps.append({
                    "kind": "tool",
                    "name": name,
                    "summary": f"Blocked disallowed tool: {name}",
                })
            oai_messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )

    return (
        AIMessage(content="I've done what I could in this turn. Want to continue?"),
        total_saved,
        agent_steps,
        client_actions,
    )


def _get_llm():
    """Plain chat completion via OpenRouter (no tools) for fallback when the tool loop errors."""
    client, model, _fallback = _openrouter_chat_client_models()

    class _OpenRouterChatWrapper:
        def __init__(self, client, model: str):
            self._client = client
            self._model = model

        def invoke(self, messages: list[BaseMessage]):
            oai_messages = _messages_to_openai_dicts(messages)
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=oai_messages,
                max_tokens=4096,
            )
            text = (resp.choices[0].message.content or "").strip()
            return AIMessage(content=text)

    return _OpenRouterChatWrapper(client, model)


def _last_user_text(messages: list) -> str:
    """Get the most recent user message content for retrieval query."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            content = getattr(m, "content", str(m))
            if isinstance(content, list):
                content = " ".join(c.get("text", str(c)) for c in content if isinstance(c, dict))
            return (content or "").strip()
    return ""


def interviewer_node(state: JournalState) -> JournalState:
    """
    Respond with empathy. When personalization > 0, retrieve relevant context from
    the vector store (gist_facts + episodic_log) and inject it so the model can personalize.
    """
    llm = _get_llm()
    personalization = max(0.0, min(1.0, state.get("personalization", 1.0)))
    personalization_percent = int(personalization * 100)
    intrusiveness = max(0.0, min(1.0, state.get("intrusiveness", 0.5)))
    intrusiveness_percent = int(intrusiveness * 100)

    now = datetime.now(timezone.utc)
    date_context = now.strftime("%A, %B %d, %Y")  # e.g. Wednesday, March 12, 2025
    _client_tc = (state.get("client_time_context") or "").strip()
    if _client_tc:
        date_context = f"{date_context} (server UTC calendar day). {_client_tc}"
    mode_raw = (state.get("mode") or "").strip().lower()
    retrieval_log: str | None = None

    if mode_raw == "autobiography":
        _local_time = _client_tc if _client_tc else date_context
        system_parts = [
            _ASSISTED_JOURNAL_PROMPT.format(local_time=_local_time),
            "User-facing formatting (mandatory): The chat surface is plain text. Do not use Markdown in your reply: "
            "no asterisks for bold, no hash headings, no fenced code blocks. Use normal sentences, commas, and em dashes. "
            "Prefer emphasis through wording, not formatting.",
            "Use **navigate_ui** when they explicitly ask to open another main screen (Brain, Chat, Home). "
            "Never claim you changed data unless the corresponding tool returned ok. "
            "Do not offer **navigate_ui** for destructive operations. After tool success, reply briefly and warmly.\n\n"
            "Semantic Memory / Library tools: use **add_library_items** when they finished a podcast, article, or research "
            "paper and want it logged (not for books). For books they read or finished, always use **extract_books_read**. "
            "Use **update_library_item** to change notes or fix metadata on something already saved. "
            "Only claim saves when the tool result shows items were added.",
        ]
    else:
        _length_hint = "Keep replies concise (2-4 sentences). "
        _memory_personalization = (
            "At 0%, do not use memory; keep questions general and present-focused only. "
            "At higher levels, use the memory context only as loose inspiration. Ask BROAD, open-ended questions. "
            "Do NOT invent, assume, or assert details that are not clearly and explicitly stated in the memory context. "
            "If the context is vague or summary-like (e.g. 'reflected on stress', 'mentioned work'), do not fill in specifics—ask generally, e.g. 'How have things been with work?' or 'What's felt different lately?' "
            "Only reference a specific person, place, or event if it is clearly named in the context; otherwise keep questions broad (e.g. 'How are you feeling about that?' 'What's on your mind today?'). "
            "Do NOT repeat back emotions or themes as facts ('You felt anxious about X') unless the user has just said so; prefer open invitations ('Want to say more about that?' 'What would be helpful to explore?'). "
        )
        _menu_explore_hint = (
            "When the user asks 'what should we talk about?' or 'what should we explore?', you may offer 1–2 broad areas "
            "if the memory context clearly suggests them; otherwise keep it open: 'Whatever feels most present—we can go wherever you'd like.' "
        )
        system_parts = [
            f"Today's date (use for time perspective): {date_context}. "
            + "You are a warm, empathetic journaling companion. Listen actively and respond with care. "
            + _length_hint
            + f"Personalization level: {personalization_percent}%. "
            + _memory_personalization,
            f"Questioning style (intrusiveness): {intrusiveness_percent}%. "
            "At 0%, be very gentle and non-intrusive; ask only soft, open-ended questions and let the user lead. "
            "At low levels, ask sparingly and avoid probing. "
            "At high levels, you may ask more direct questions when it feels supportive, while still respecting boundaries. ",
            _menu_explore_hint,
            "Use **navigate_ui** when they explicitly ask to open another main screen (Brain, Chat, Home). "
            "Never claim you changed data unless the corresponding tool returned ok. Do not offer **navigate_ui** for destructive operations. "
            "After tool success, reply briefly and warmly.\n\n"
            "Semantic Memory / Library: **add_library_items** for podcasts, articles, and research they finished; "
            "**extract_books_read** for books (never add books via add_library_items). **update_library_item** to edit saved entries. "
            "Only claim saves when tools return success with new items.",
        ]

    if _GRAPH_RETRIEVAL_ENABLED and personalization > 0:
        query = _last_user_text(state["messages"])
        if not query and state["messages"]:
            query = str(state["messages"][-1])[:500]
        instance_id = state.get("instance_id") or ""
        try:
            processed, raw = get_relevant_context_dual(
                query,
                top_k_gist=10,
                top_k_episodic=6,
                instance_id=instance_id,
                session_id=state.get("session_id"),
                log=True,
            )
            ctx_sections: list[str] = [
                "## Who This Person Is (your understanding)\n" + (processed or "(no profile signals yet)"),
                "## Relevant Context (from their actual entries)\n" + raw,
            ]
            ctx = "\n\n".join(ctx_sections)
            ctx_tail = "\n\n(use only as broad inspiration; synthesize faithfully—do not assert or invent details not explicitly stated)"
            system_parts.append("\n\n" + ctx + ctx_tail)
            retrieval_log = ctx
        except Exception:
            system_parts.append(
                "\n\n(No prior journal context available. Use the reflective fallback opening style — "
                "warm, time-aware, psychologically evocative. Do NOT mention missing context to the user.)"
            )

    system = SystemMessage(content="\n".join(system_parts))

    lc_messages = [system] + state["messages"]
    oai_messages = _messages_to_openai_dicts(lc_messages)
    instance_id = state.get("instance_id") or ""
    agent_steps: list[dict] = []
    client_actions: list[dict] = []
    response: AIMessage
    try:
        client, model, _fallback = _openrouter_chat_client_models()
        extra_body: dict | None = None
        picked = _user_pick_openrouter_model(mode_raw, state.get("openrouter_model"))
        if picked:
            model = picked
            extra_body = _reasoning_extra_body_for_model(model)
        elif mode_raw == "conversation":
            model = (os.getenv("OPENROUTER_CONVERSATION_MODEL") or DEFAULT_OPENROUTER_CONVERSATION_MODEL).strip()
            extra_body = {"reasoning": {"enabled": True}}
        elif mode_raw == "autobiography":
            model = (os.getenv("OPENROUTER_ASSISTED_JOURNAL_MODEL") or DEFAULT_ASSISTED_JOURNAL_MODEL).strip()
            extra_body = _reasoning_extra_body_for_model(model)
        assisted_temp = 0.525 if mode_raw == "autobiography" else None
        response, lib_saved, tool_steps, nav_actions = _interviewer_run_with_tools(
            client,
            model,
            oai_messages,
            instance_id,
            extra_body=extra_body,
            temperature=assisted_temp,
        )
        agent_steps = list(tool_steps)
        client_actions = list(nav_actions)
        library_items_added = int(lib_saved) if lib_saved > 0 else 0
    except Exception as e:
        print("[backend] interviewer tool path failed; plain OpenRouter chat fallback:", e)
        response = llm.invoke(lc_messages)
        agent_steps = []
        client_actions = []
        library_items_added = 0

    if retrieval_log is not None and personalization > 0:
        agent_steps.insert(0, {"kind": "retrieval", "summary": "Retrieved relevant context from your journals"})

    out: dict = {"messages": [response], "agent_steps": agent_steps}
    if retrieval_log is not None:
        out["retrieval_log"] = retrieval_log
    if client_actions:
        out["client_actions"] = client_actions
    if library_items_added > 0:
        out["library_items_added"] = library_items_added
    return out


def librarian_node(state: JournalState) -> JournalState:
    """Background worker: extract, embed, save to SQLite+sqlite-vec (primary memory pipeline)."""
    session_id = state.get("session_id", "default")
    instance_id = state.get("instance_id") or ""
    transcript = _messages_to_transcript(state["messages"])
    extracted = ingest_journal_entry(
        session_id, transcript, instance_id=instance_id, entry_source="assisted"
    )
    return {
        "messages": [],
        "last_transcript": transcript,
        "last_summary": extracted.get("summary", ""),
        "last_facts": extracted.get("facts", []),
    }


def _messages_to_transcript(messages: list[BaseMessage]) -> str:
    """Convert message list to plain transcript."""
    parts = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = getattr(m, "content", str(m))
        if isinstance(content, list):
            content = " ".join(
                c.get("text", str(c)) for c in content if isinstance(c, dict)
            )
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def build_graph():
    """Wire Interviewer and Librarian. No middle agent."""
    graph = StateGraph(JournalState)

    graph.add_node("interviewer", interviewer_node)
    graph.add_node("librarian", librarian_node)

    # Chat flow: user -> interviewer -> END
    graph.set_entry_point("interviewer")
    graph.add_edge("interviewer", END)

    return graph.compile()


# Separate graph for end-session (Librarian only)
def build_librarian_graph():
    """Graph that runs only the Librarian node."""
    graph = StateGraph(JournalState)
    graph.add_node("librarian", librarian_node)
    graph.set_entry_point("librarian")
    graph.add_edge("librarian", END)
    return graph.compile()
