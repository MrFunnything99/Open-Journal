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

from library import apply_library_tool_items, get_relevant_context, save_session_data
from agent_site_tools import (
    log_tool_invocation,
    tool_mark_recommendation_consumed,
    tool_navigate_ui,
    tool_update_library_item,
)

# State: list of messages + session_id for Librarian + personalization + intrusiveness + mode + optional retrieval log + instance_id
class JournalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    personalization: float
    intrusiveness: NotRequired[float]
    mode: NotRequired[str]  # "journal" (recommendations handled in main)
    retrieval_log: NotRequired[str]
    last_transcript: NotRequired[str]
    last_summary: NotRequired[str]
    last_facts: NotRequired[list]
    instance_id: NotRequired[str]  # X-Instance-ID for per-device data isolation
    library_items_added: NotRequired[int]  # chat agent saved N items to Library (journal mode)
    agent_steps: NotRequired[list]  # UI: retrieval + tool summaries for this turn
    client_actions: NotRequired[list]  # UI: allowlisted navigate actions from navigate_ui tool


DEFAULT_OPENROUTER_CHAT_MODEL = "openai/gpt-5.4"


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


LIBRARY_ITEMS_TOOL = {
    "type": "function",
    "function": {
        "name": "add_library_items",
        "description": (
            "Save books, podcasts, articles, or research papers the user has finished (or wants logged) into their "
            "Library for recommendations. Use when they paste a list, enumerate several titles, or ask to add/track "
            "what they read or listened to. Use correct type per item. Normalize book titles and author names when "
            "obvious (e.g. 'dune' → Dune, Frank Herbert). Leave url empty unless they gave a real link. "
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
            "Update an existing Library item's note and/or completion date. Use when the user asks to change a note, "
            "rename wording stored as a note, or set completion date for something already in their library "
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

_CHAT_TOOLS = [
    LIBRARY_ITEMS_TOOL,
    UPDATE_LIBRARY_ITEM_TOOL,
    MARK_RECOMMENDATION_CONSUMED_TOOL,
    NAVIGATE_UI_TOOL,
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
) -> tuple[AIMessage, int, list[dict], list[dict]]:
    """
    Multi-turn chat completion (OpenRouter) with allowlisted site tools.
    Returns (assistant message, total library items saved from add_library_items, agent_steps, client_actions).
    """
    total_saved = 0
    agent_steps: list[dict] = []
    client_actions: list[dict] = []
    rounds = 0
    while rounds < max_tool_rounds:
        rounds += 1
        resp = client.chat.completions.create(
            model=model,
            messages=oai_messages,
            tools=_CHAT_TOOLS,
            tool_choice="auto",
            max_tokens=4096,
        )
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
    client, model = _openrouter_chat_client_and_model()

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
    mode_raw = (state.get("mode") or "").strip().lower()
    # Only journal mode uses this interviewer; recommendations uses library_interview in main.py
    system_parts = [
            f"Today's date (use for time perspective): {date_context}. "
            "You are a warm, empathetic journaling companion. Listen actively and respond with care. Keep replies concise (2-4 sentences). "
            f"Personalization level: {personalization_percent}%. "
            "At 0%, do not use memory; keep questions general and present-focused only. "
            "At higher levels, use the memory context only as loose inspiration. Ask BROAD, open-ended questions. "
            "Do NOT invent, assume, or assert details that are not clearly and explicitly stated in the memory context. "
            "If the context is vague or summary-like (e.g. 'reflected on stress', 'mentioned work'), do not fill in specifics—ask generally, e.g. 'How have things been with work?' or 'What's felt different lately?' "
            "Only reference a specific person, place, or event if it is clearly named in the context; otherwise keep questions broad (e.g. 'How are you feeling about that?' 'What's on your mind today?'). "
            "Do NOT repeat back emotions or themes as facts ('You felt anxious about X') unless the user has just said so; prefer open invitations ('Want to say more about that?' 'What would be helpful to explore?'). ",
            f"Questioning style (intrusiveness): {intrusiveness_percent}%. "
            "At 0%, be very gentle and non-intrusive; ask only soft, open-ended questions and let the user lead. "
            "At low levels, ask sparingly and avoid probing. "
            "At high levels, you may ask more direct questions when it feels supportive, while still respecting boundaries. ",
            "When the user asks 'what should we talk about?' or 'what should we explore?', you may offer 1–2 broad areas if the memory context clearly suggests them; otherwise keep it open: 'Whatever feels most present—we can go wherever you'd like.' ",
            "You can save things to the user's **Library** by calling **add_library_items** when they paste a list or clearly want new items recorded. "
            "Use **update_library_item** to change an existing library item's **note** or **date_completed** when they ask to edit, rename phrasing in a note, or fix metadata (prefer **title_query** if you don't have an id). "
            "Use **mark_recommendation_consumed** when they say they finished or consumed a recommendation. "
            "Use **navigate_ui** when they explicitly ask to open another main screen (Recommendations, Brain, Chat, Home). "
            "Never claim you changed data unless the corresponding tool returned ok. Do not offer **navigate_ui** for destructive operations. "
            "After tool success, reply briefly and warmly. ",
    ]

    retrieval_log: str | None = None
    if personalization > 0:
        query = _last_user_text(state["messages"])
        if not query and state["messages"]:
            query = str(state["messages"][-1])[:500]
        instance_id = state.get("instance_id") or ""
        try:
            context = get_relevant_context(query, top_k_gist=10, top_k_episodic=6, instance_id=instance_id)
            system_parts.append("\n\nRelevant context from the user's journals (use only as broad inspiration; do not assert or invent details not explicitly stated):\n" + context)
            retrieval_log = context
        except Exception:
            system_parts.append("\n\n(Memory retrieval unavailable; respond without prior context.)")

    system = SystemMessage(content="\n".join(system_parts))
    lc_messages = [system] + state["messages"]
    oai_messages = _messages_to_openai_dicts(lc_messages)
    instance_id = state.get("instance_id") or ""
    library_added = 0
    agent_steps: list[dict] = []
    client_actions: list[dict] = []
    response: AIMessage
    try:
        client, model = _openrouter_chat_client_and_model()
        response, library_added, tool_steps, nav_actions = _interviewer_run_with_tools(
            client, model, oai_messages, instance_id
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
        agent_steps.insert(
            0,
            {"kind": "retrieval", "summary": "Retrieved relevant context from your journals"},
        )

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
    extracted = save_session_data(session_id, transcript, instance_id=instance_id)
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
