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

from learning import get_learning_system_prompt
from library import (
    apply_library_tool_items,
    get_assisted_journal_continuity_block,
    get_relevant_context_dual,
    ingest_journal_entry,
    resolve_books_via_openlibrary,
    save_resolved_books,
)
from agent_site_tools import (
    log_tool_invocation,
    tool_mark_recommendation_consumed,
    tool_navigate_ui,
    tool_request_focused_recommendation,
    tool_submit_content_feedback,
    tool_update_content_preferences,
    tool_update_library_item,
)

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
    library_items_added: NotRequired[int]  # chat agent saved N items to Library (journal mode)
    agent_steps: NotRequired[list]  # UI: retrieval + tool summaries for this turn
    client_actions: NotRequired[list]  # UI: allowlisted navigate actions from navigate_ui tool
    # Optional: browser-built local time + daypart for Assisted Journal secondary check-ins
    client_time_context: NotRequired[str]


DEFAULT_OPENROUTER_CHAT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_OPENROUTER_CHAT_FALLBACK_MODEL = "openai/gpt-5.4"
DEFAULT_OPENROUTER_CONVERSATION_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_ASSISTED_JOURNAL_MODEL = "openai/gpt-5-mini"

# UI allowlist for Conversation + Assisted Journal (must match frontend chatCompletionModels.ts)
USER_SELECTABLE_CHAT_MODELS: frozenset[str] = frozenset(
    {
        "openai/gpt-5.4",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5-nano",
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


LIBRARY_ITEMS_TOOL = {
    "type": "function",
    "function": {
        "name": "add_library_items",
        "description": (
            "Save podcasts, articles, or research papers the user has finished (or wants logged) into their "
            "Library for recommendations. For BOOKS, always use extract_books_read instead — it normalizes "
            "title/author via the same LLM stack as chat. Use this tool when they paste a list, enumerate several titles, or ask to "
            "add/track what they listened to or read (non-book). Use correct type per item. "
            "Leave url empty unless they gave a real link. "
            "Do NOT use for hypothetical picks, things they might read later, or casual single mentions unless "
            "they clearly want them saved."
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
                                "description": "Author, show name, venue, or first author for papers; omit or empty if unknown",
                            },
                            "url": {
                                "type": "string",
                                "description": "Canonical URL if the user provided one; otherwise empty string",
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
            "Update an existing Library item's metadata (note, completion date, title, author, url). Use when the user asks to change a note, "
            "fix a mistaken title/author, rename an item, or set completion date for something already in their library "
            "(books, podcasts, articles, research already consumed). Provide item_id if known; otherwise use "
            "title_query to find the best match by title."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "Exact library item id from the app if the user pasted it or it is in context",
                },
                "title_query": {
                    "type": "string",
                    "description": "Substring or full title to search when item_id is unknown",
                },
                "note": {
                    "type": "string",
                    "description": "New note text; include only if updating the note",
                },
                "date_completed": {
                    "type": "string",
                    "description": "YYYY-MM-DD; include only if updating completion date",
                },
                "new_title": {
                    "type": "string",
                    "description": "Corrected title to store for this item; include only if changing the title",
                },
                "new_author": {
                    "type": "string",
                    "description": "Corrected author/creator; include only if changing the author",
                },
                "new_url": {
                    "type": "string",
                    "description": "Corrected canonical URL; include only if changing the URL",
                },
            },
        },
    },
}

MARK_RECOMMENDATION_CONSUMED_TOOL = {
    "type": "function",
    "function": {
        "name": "mark_recommendation_consumed",
        "description": (
            "Record that the user finished or consumed a recommended book, podcast, article, or research item "
            "(same as marking read/listened in Recommendations). Use when they say they finished, read, or listened to something."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["book", "podcast", "article", "research"],
                    "description": "Media category",
                },
                "title": {"type": "string", "description": "Title of the work"},
                "author": {"type": "string", "description": "Author or show name if known; omit if unknown"},
                "url": {"type": "string", "description": "URL if the user provided one"},
                "liked": {
                    "type": "boolean",
                    "description": "True if they enjoyed; false if they disliked",
                },
            },
            "required": ["type", "title"],
        },
    },
}

NAVIGATE_UI_TOOL = {
    "type": "function",
    "function": {
        "name": "navigate_ui",
        "description": (
            "Switch the user's main app screen when they ask to open a section (e.g. open Recommendations, "
            "go to Brain, open Chat). Does not change server data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": ["home", "chat", "brain", "recommendations"],
                    "description": "home=journal home; chat=full chat; brain=knowledge/calendar hub; recommendations",
                },
                "brain_section": {
                    "type": "string",
                    "enum": ["knowledgeBase", "calendar"],
                    "description": "When view is brain, optional subsection to open",
                },
            },
            "required": ["view"],
        },
    },
}

UPDATE_CONTENT_PREFERENCES_TOOL = {
    "type": "function",
    "function": {
        "name": "update_content_preferences",
        "description": (
            "Update subscriptions, paywall policy, and preferred/avoided content types when the user "
            "mentions sources they subscribe to, paywalls, or what formats they want more/less of."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "add_subscriptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Domains e.g. nytimes.com",
                },
                "remove_subscriptions": {"type": "array", "items": {"type": "string"}},
                "paywall_policy": {
                    "type": "string",
                    "enum": ["only_subscribed", "allow_all", "no_paywalled"],
                },
                "preferred_types": {"type": "array", "items": {"type": "string"}},
                "avoid_types": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

SUBMIT_CONTENT_FEEDBACK_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_content_feedback",
        "description": (
            "Record feedback on content they consumed (articles, books, podcasts). Capture why in user_notes when they explain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content_title": {"type": "string"},
                "content_type": {
                    "type": "string",
                    "enum": ["article", "book", "podcast", "video", "paper"],
                },
                "feedback": {
                    "type": "string",
                    "enum": ["liked", "disliked", "loved", "not_relevant"],
                },
                "user_notes": {"type": "string"},
                "content_url": {"type": "string"},
            },
            "required": ["content_title", "feedback"],
        },
    },
}

REQUEST_FOCUSED_REC_TOOL = {
    "type": "function",
    "function": {
        "name": "request_focused_recommendation",
        "description": (
            "Search for real links on a topic the user asks about. Returns URLs from search APIs only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "content_type": {
                    "type": "string",
                    "enum": ["article", "podcast", "book", "research", "any"],
                    "default": "any",
                },
            },
            "required": ["topic"],
        },
    },
}

EXTRACT_BOOKS_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_books_read",
        "description": (
            "Extract books the user mentions having read, finished, or consumed during conversation. "
            "Use this when the user says they read a book, finished a book, loved a book, etc. — even if "
            "mentioned casually or indirectly (e.g. 'I just finished Dune' or 'that Brené Brown book was great'). "
            "Extract the raw title and author as the user said them; the backend will normalize title/author with the LLM. "
            "Include any opinion or short comment the user expressed as a note (e.g. 'it was good', 'didn't love the ending'). "
            "Do NOT use add_library_items for books — always use this tool instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "books": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "raw_title": {
                                "type": "string",
                                "description": "Book title as the user said it (may have typos or be informal)",
                            },
                            "raw_author": {
                                "type": "string",
                                "description": "Author name as the user said it; null or empty if not mentioned",
                            },
                            "liked": {
                                "type": "boolean",
                                "description": "True if the user enjoyed or was neutral; false if they disliked",
                            },
                            "note": {
                                "type": "string",
                                "description": "Any short opinion or comment the user expressed about the book",
                            },
                        },
                        "required": ["raw_title"],
                    },
                },
            },
            "required": ["books"],
        },
    },
}

_CHAT_TOOLS = [
    LIBRARY_ITEMS_TOOL,
    UPDATE_LIBRARY_ITEM_TOOL,
    MARK_RECOMMENDATION_CONSUMED_TOOL,
    EXTRACT_BOOKS_READ_TOOL,
    NAVIGATE_UI_TOOL,
    UPDATE_CONTENT_PREFERENCES_TOOL,
    SUBMIT_CONTENT_FEEDBACK_TOOL,
    REQUEST_FOCUSED_REC_TOOL,
]


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
) -> tuple[AIMessage, int, list[dict], list[dict]]:
    """
    Multi-turn chat completion (OpenRouter) with allowlisted site tools.
    Returns (assistant message, total library items saved from add_library_items, agent_steps, client_actions).
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
            elif name == "mark_recommendation_consumed":
                result, summ = tool_mark_recommendation_consumed(args, instance_id)
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
            elif name == "update_content_preferences":
                result, summ = tool_update_content_preferences(args, instance_id)
                agent_steps.append({"kind": "tool", "name": name, "summary": summ})
            elif name == "submit_content_feedback":
                result, summ = tool_submit_content_feedback(args, instance_id)
                agent_steps.append({"kind": "tool", "name": name, "summary": summ})
            elif name == "request_focused_recommendation":
                result, summ = tool_request_focused_recommendation(args, instance_id)
                agent_steps.append({"kind": "tool", "name": name, "summary": summ})
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


def _assisted_journal_inspiration_intent(text: str) -> bool:
    """User wants the model to scan journals and suggest concrete memory threads."""
    t = (text or "").lower()
    needles = (
        "inspiration",
        "inspire me",
        "look at my journal",
        "look through my journal",
        "check my journal",
        "my journals",
        "from my journal",
        "scan my",
        "pull from my",
        "you pick",
        "you choose",
        "surprise me",
        "pick a memory",
        "suggest something",
        "ideas from",
        "what from my past",
        "what stands out",
        "see what's in",
        "for some inspiration",
        "journal inspiration",
        "ideas from my life",
        "past memories",
        "my memories",
        "our memories",
        "you can look",
        "go ahead and look",
        "look them up",
        "use my entries",
        "knowledge base",
        "knowledgebase",
        "my brain",
        "the brain",
        "in my brain",
        "brain hub",
        "saved in the app",
        "what i saved",
        "what i've saved",
        "look at everything",
        "everything i wrote",
        "selfmeridian",
    )
    return any(n in t for n in needles)


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
    # Only journal mode uses this interviewer; recommendations uses library_interview in main.py
    _kb_intro = ""
    if mode_raw == "autobiography":
        _kb_intro = (
            "Assisted Journal — your view of their knowledge base: Anything below titled Continuity anchor, Who This Person Is, or "
            "Relevant Context is loaded from this user's SelfMeridian knowledge base (saved journal text and Brain memory excerpts). "
            "If they say knowledge base, Brain, journals, or what I saved, they mean that injected content, not an external system you cannot read. "
            "When those sections list real excerpts, bullets, or dated lines (not only the bare word \"None.\"), you already have that access. "
            "Do not say you lack access to their knowledge base in that case.\n\n"
        )
    _length_hint = (
        "Keep replies concise (2-4 sentences). "
        if mode_raw != "autobiography"
        else "Keep replies concise (1-2 sentences for openings, 2-4 for follow-ups) except when synthesizing journal inspiration from memory—then follow the Assisted Journal length and prose rules. "
    )
    if mode_raw == "autobiography":
        _memory_personalization = (
            "At 0%, do not use memory; keep questions general and present-focused only. "
            "At higher levels, treat Continuity anchor (if present) and Relevant Context as ground truth for prior journal text. "
            "Memory is intentionally balanced: solo written journals and AI-assisted sessions are equally weighted—do not anchor only on chat fragments. "
            "You may blend a thread that shows up in both (e.g. sleep in their written journal and again in a recent chat) without naming the source. "
            "\n\nRECENCY-FIRST OPENING RULE (critical for first replies / check-in openers like 'hey', 'how's it going', 'what should we talk about'):\n"
            "Priority order — always follow this:\n"
            "1. LATEST MANUAL JOURNAL ENTRY (highest priority) — reference 1 specific concrete thing from it.\n"
            "2. Latest assisted journal session — only if more recent than the latest manual entry.\n"
            "3. Recurring theme from last 2-3 entries — only as a brief secondary offer.\n"
            "4. Older high-salience memory — ONLY if repeated recently. Do NOT lead with older emotionally dramatic material.\n\n"
            "OPENING BREVITY (this is spoken aloud — optimize for natural speech cadence):\n"
            "- 1-2 short sentences MAX, ideally under 25 words total.\n"
            "- ONE direct question, no menus or multiple-choice lists.\n"
            "- Use time-of-day awareness naturally (morning/afternoon/evening) when the client local time is available.\n"
            "- Prefer emotional phrasing over analytical framing: 'what's still sitting with you?' not 'the thing that's been taking up the most space in your head.'\n"
            "- Remove setup phrases. Get to the question fast.\n"
            "- Sound like a calm therapist speaking, not a generated prompt.\n\n"
            "Good openers (study these):\n"
            "- 'How's the evening going — still thinking about [concrete detail], or has the day shifted things?'\n"
            "- 'You wrote about [detail] earlier. Still feel the same way tonight?'\n"
            "- 'Good morning. How'd you sleep after [thing from latest journal]?'\n"
            "- 'Last time you mentioned [detail]. What's that been like since?'\n\n"
            "Bad openers (avoid):\n"
            "- 'Since it's evening, do you want to start with a recap of the day, or with the thing that's been taking up the most space in your head?' (too long, menu-like)\n"
            "- 'Hey, in your last journal you talked about X and Y. Want to start there, or would it feel better to zoom out and talk about the bigger theme?' (over-constructed)\n\n"
            "Do NOT invent people, events, or outcomes not clearly supported by Continuity or Relevant Context. "
            "Banned first-reply openers (always): "
            "A couple threads seem alive; There are a few themes; What feels most present?; What should we talk about?; How are you?; What's on your mind? "
            "If Continuity anchor and Relevant Context are both empty or bare None, use the reflective fallback style from the Assisted Journal section — "
            "warm, time-aware, psychologically evocative, 1-2 sentences. Never mention missing context.\n"
            "When the user explicitly requests inspiration, a journal scan, surprise me, you pick, etc., follow the Inspiration Scan rules in the Assisted Journal section for that turn instead. "
        )
    else:
        _memory_personalization = (
            "At 0%, do not use memory; keep questions general and present-focused only. "
            "At higher levels, use the memory context only as loose inspiration. Ask BROAD, open-ended questions. "
            "Do NOT invent, assume, or assert details that are not clearly and explicitly stated in the memory context. "
            "If the context is vague or summary-like (e.g. 'reflected on stress', 'mentioned work'), do not fill in specifics—ask generally, e.g. 'How have things been with work?' or 'What's felt different lately?' "
            "Only reference a specific person, place, or event if it is clearly named in the context; otherwise keep questions broad (e.g. 'How are you feeling about that?' 'What's on your mind today?'). "
            "Do NOT repeat back emotions or themes as facts ('You felt anxious about X') unless the user has just said so; prefer open invitations ('Want to say more about that?' 'What would be helpful to explore?'). "
        )
    _menu_explore_hint = (
        "When the user asks what to talk about or what to explore, answer with continuity: reference 1 concrete detail from the LATEST manual journal entry, then ONE short question (under 25 words total). "
        "No menus. Only if there is truly no journal text, give a brief warm invitation without the banned vague openers. "
        if mode_raw == "autobiography"
        else "When the user asks 'what should we talk about?' or 'what should we explore?', you may offer 1–2 broad areas if the memory context clearly suggests them; otherwise keep it open: 'Whatever feels most present—we can go wherever you'd like.' "
    )
    system_parts = [
            f"Today's date (use for time perspective): {date_context}. "
            + _kb_intro
            + "You are a warm, empathetic journaling companion. Listen actively and respond with care. "
            + _length_hint
            + f"Personalization level: {personalization_percent}%. "
            + _memory_personalization,
            f"Questioning style (intrusiveness): {intrusiveness_percent}%. "
            "At 0%, be very gentle and non-intrusive; ask only soft, open-ended questions and let the user lead. "
            "At low levels, ask sparingly and avoid probing. "
            "At high levels, you may ask more direct questions when it feels supportive, while still respecting boundaries. ",
            _menu_explore_hint,
            "When the user mentions a **book** they read, finished, or enjoyed (even casually), call **extract_books_read** — "
            "the backend will normalize title/author with the configured OpenRouter model and save to the library. "
            "Use **add_library_items** only for non-book media (podcasts, articles, research). "
            "Use **update_library_item** to change an existing library item's **note** or **date_completed** when they ask to edit, rename phrasing in a note, or fix metadata (prefer **title_query** if you don't have an id). "
            "Use **mark_recommendation_consumed** when they say they finished or consumed a recommendation. "
            "Use **navigate_ui** when they explicitly ask to open another main screen (Recommendations, Brain, Chat, Home). "
            "Use **update_content_preferences** when they mention subscriptions, outlets, paywalls, or types of media they want more/less of. "
            "Use **submit_content_feedback** when they share how they felt about something they read, watched, or listened to — capture why in user_notes. "
            "Use **request_focused_recommendation** when they want links on a specific topic; it returns verified URLs from search. "
            "Never claim you changed data unless the corresponding tool returned ok. Do not offer **navigate_ui** for destructive operations. "
            "After tool success, reply briefly and warmly. ",
    ]

    retrieval_log: str | None = None
    inspiration_journal_scan = False
    if personalization > 0:
        query = _last_user_text(state["messages"])
        if not query and state["messages"]:
            query = str(state["messages"][-1])[:500]
        if mode_raw == "autobiography" and _assisted_journal_inspiration_intent(query):
            inspiration_journal_scan = True
            query = (
                "Meaningful personal memories, relationships, trips, milestones, achievements, "
                "warm or proud moments, formative experiences, people and places named in journal entries"
            )
        instance_id = state.get("instance_id") or ""
        continuity_block = ""
        if mode_raw == "autobiography":
            continuity_block = get_assisted_journal_continuity_block(instance_id)
        try:
            tg, te = (12, 12) if inspiration_journal_scan else (10, 6)
            processed, raw = get_relevant_context_dual(
                query,
                top_k_gist=tg,
                top_k_episodic=te,
                instance_id=instance_id,
                session_id=state.get("session_id"),
                log=True,
            )
            raw_s = (raw or "").strip()
            if (
                mode_raw == "autobiography"
                and personalization > 0
                and raw_s in ("None.", "")
            ):
                processed2, raw2 = get_relevant_context_dual(
                    "Journal entries life story memories people events relationships saved knowledge base",
                    top_k_gist=14,
                    top_k_episodic=10,
                    instance_id=instance_id,
                    session_id=state.get("session_id"),
                    log=True,
                    balance_journal_sources=_balance_sources,
                )
                if (raw2 or "").strip() not in ("None.", ""):
                    processed, raw = processed2, raw2
            ctx_sections: list[str] = []
            if continuity_block.strip():
                ctx_sections.append(
                    "## Continuity anchor (RECENCY-FIRST: latest manual journal is the primary opener anchor)\n"
                    + continuity_block
                )
            ctx_sections.append(
                "## Who This Person Is (your understanding)\n" + (processed or "(no profile signals yet)")
            )
            ctx_sections.append("## Relevant Context (from their actual entries)\n" + raw)
            ctx = "\n\n".join(ctx_sections)
            hint = ""
            if inspiration_journal_scan:
                hint = (
                    "\n\n(This block was retrieved for a broad journal scan: synthesize distinct threads for the user in natural prose—"
                    "not a stack of bold labels. Do not invent events not supported by excerpts.)"
                )
            if inspiration_journal_scan:
                ctx_tail = (
                    hint
                    + "\n\n(use only as broad inspiration; synthesize faithfully—do not assert or invent details not explicitly stated)"
                )
            elif mode_raw == "autobiography":
                ctx_tail = "\n\n(Continuity-first when appropriate; do not invent details not in the excerpts above.)"
            else:
                ctx_tail = "\n\n(use only as broad inspiration; synthesize faithfully—do not assert or invent details not explicitly stated)"
            system_parts.append("\n\n" + ctx + ctx_tail)
            retrieval_log = ctx
        except Exception:
            if mode_raw == "autobiography" and continuity_block.strip():
                system_parts.append(
                    "\n\n## Continuity anchor (most recent saved journals — broader retrieval failed)\n"
                    + continuity_block
                    + "\n\n(Anchor openings in the LATEST manual journal entry above. Recency beats salience.)"
                )
            system_parts.append("\n\n(No prior journal context available. Use the reflective fallback opening style — warm, time-aware, psychologically evocative. Do NOT mention missing context to the user.)")

    if mode_raw == "autobiography":
        system_parts.append(
            "\n\nAssisted Journal — you always use this mode's dedicated model\n"
            "Steer from the user's latest message. Default to warm, concise replies (1-2 sentences for openings, 2-4 for follow-ups) unless they explicitly asked for a broad "
            "journal inspiration scan (section C).\n\n"
            "User-facing formatting (mandatory): The chat surface is plain text. Do not use Markdown in your reply: no asterisks for bold, "
            "no hash headings, no fenced code blocks. Use normal sentences, commas, and em dashes. Prefer emphasis through wording, not formatting.\n\n"
            "Opening and continuity (when they did NOT just trigger an inspiration scan):\n"
            "STRICT RECENCY-FIRST: Ground in the LATEST MANUAL JOURNAL ENTRY. Reference ONE concrete detail. "
            "Do NOT jump to older memories unless the latest entry is empty, the older memory is repeated recently, or the user references it.\n"
            "BREVITY: 1-2 sentences, under 25 words, ONE question. This is spoken aloud — write for natural speech rhythm. "
            "Blend time-of-day naturally. No menus, no multiple-choice, no setup phrases. Sound like a calm therapist, not a generated menu.\n\n"
            "Banned first-sentence patterns (always banned): "
            "A couple threads seem alive; There are a few themes; What feels most present?; What should we talk about?; How are you?; What's on your mind? "
            "When context is empty, use the reflective fallback openers defined in section E below.\n\n"
            "C) Inspiration from their journals — When they explicitly choose inspiration (look at journals, surprise me, you pick, scan, etc.), "
            "ignore the single-hook opening rule for that turn. Treat Relevant Context as the scan result (already balanced between solo-written and assisted-chat excerpts). "
            "Write about 100–185 words total (stay compact). "
            "Digest and synthesize; weave several threads into flowing prose (short paragraphs or soft line breaks), not staccato labels. "
            "Modest warmth and connective tissue; every concrete name, place, event, and factual beat anchored in excerpts. Clean up phrasing; never paste "
            "fragmentary chunks. End by inviting them toward one thread. Do not invent a happy arc the text does not support.\n\n"
            "A) Journaling about today — If they want today or a daily recap, still prefer a continuity hook when recent entries exist; then ask about their day.\n\n"
            "B) Life-story or autobiographical work — If they ask for that without asking for a scan, continuity-first opening; you may offer a journal scan "
            "as one option, not a dump of guessed themes.\n\n"
            "D) Continuing after they pick — Stay on their thread; reflective questions; no fabricated details.\n\n"
            "E) Generic let's journal — If recent journal text exists, open with continuity (no theme synthesis, no vague menu opener). If there is truly "
            "nothing to ground on, use a REFLECTIVE FALLBACK opener (see below) — never a menu or apology.\n\n"
            "REFLECTIVE FALLBACK OPENERS (when no journal context exists):\n"
            "Having no recent journal context is NORMAL — it is NOT an error state, NOT degraded intelligence. "
            "NEVER mention missing context, missing journals, or suggest the user save an entry first. "
            "NEVER say: 'I don't have context', 'since the slate is clean', 'I'm not seeing your journals.' "
            "Instead, pivot naturally into a warm, time-aware, psychologically stimulating opener that meets the present moment.\n\n"
            "These must create gentle reflective momentum — not small talk. Use time-of-day awareness and one emotionally evocative question "
            "that helps the user begin thinking. 1-2 conversational sentences, warm therapist tone, easy to answer aloud.\n\n"
            "Good fallback openers by time of day:\n"
            "Evening:\n"
            "- 'How's your evening been so far? What's your mind still coming back to?'\n"
            "- 'As the day winds down, what still feels like it's lingering tonight?'\n"
            "- 'What from today feels finished, and what still has some emotional weight?'\n"
            "Morning:\n"
            "- 'How's your morning starting off? What kind of energy are you carrying into today?'\n"
            "- 'As the day starts, what already feels most present?'\n"
            "Afternoon:\n"
            "- 'How's your afternoon been unfolding? What's been taking up the most mental space?'\n"
            "- 'What part of today still feels unfinished in your head?'\n"
            "General:\n"
            "- 'What's your mind been circling around today?'\n"
            "- 'How are you really doing right now — not the quick answer, the real one?'\n\n"
            "These should feel like a warm check-in that naturally turns into reflection within seconds. "
            "Gently pull on: emotional residue, unresolved moments, energy shifts, mental loops, contrast between outer functioning and inner feeling. "
            "Never just 'How are you?' or 'What's on your mind?' — those invite flat answers.\n\n"
            "Critical (access): Continuity anchor, Relevant Context, and Who This Person Is are their in-app knowledge base. If substantive text exists "
            "(not bare None everywhere), use it and do not claim you cannot see their journals. If all sections "
            "are empty or bare None, use the reflective fallback openers above — seamlessly, as if this is how you always start."
        )

    if mode_raw == "learning":
        import vec_store as _vs
        _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _article = _vs.daily_article_get(instance_id, _today) or {}
        learning_sys = get_learning_system_prompt(instance_id, _article)
        system = SystemMessage(content=learning_sys)
    else:
        system = SystemMessage(content="\n".join(system_parts))

    lc_messages = [system] + state["messages"]
    oai_messages = _messages_to_openai_dicts(lc_messages)
    instance_id = state.get("instance_id") or ""
    library_added = 0
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
        elif mode_raw == "learning":
            model = (os.getenv("OPENROUTER_LEARNING_MODEL") or "anthropic/claude-opus-4.6").strip()
            extra_body = _reasoning_extra_body_for_model(model)
        response, library_added, tool_steps, nav_actions = _interviewer_run_with_tools(
            client, model, oai_messages, instance_id, extra_body=extra_body
        )
        agent_steps = list(tool_steps)
        client_actions = list(nav_actions)
    except Exception as e:
        print("[backend] interviewer tool path failed; plain OpenRouter chat fallback:", e)
        response = llm.invoke(lc_messages)
        library_added = 0
        agent_steps = []
        client_actions = []

    if retrieval_log is not None and personalization > 0:
        _retrieval_summary = (
            "Broad scan of recent journals for memory inspiration"
            if inspiration_journal_scan
            else (
                "Retrieved balanced solo + assisted journal context"
                if mode_raw == "autobiography"
                else "Retrieved relevant context from your journals"
            )
        )
        agent_steps.insert(0, {"kind": "retrieval", "summary": _retrieval_summary})

    out: dict = {"messages": [response], "agent_steps": agent_steps}
    if retrieval_log is not None:
        out["retrieval_log"] = retrieval_log
    if library_added > 0:
        out["library_items_added"] = library_added
    if client_actions:
        out["client_actions"] = client_actions
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
