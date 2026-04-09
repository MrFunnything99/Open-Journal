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
    get_relevant_context_dual,
    ingest_journal_entry,
)
from agent_site_tools import (
    log_tool_invocation,
    tool_navigate_ui,
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

_CHAT_TOOLS = [
    NAVIGATE_UI_TOOL,
]

_ASSISTED_JOURNAL_PROMPT = (
    "You are a journaling companion in an app called Open Journal. "
    "The user is here to think out loud. Your job is to help them reflect — "
    "ask questions that fit the moment, follow their energy, and help them go deeper than they would alone.\n\n"
    "The current time is {local_time}.\n\n"
    "Three kinds of questions (no script, no stock phrasing — invent fresh wording every turn and every session):\n"
    "- CONTEXT BUILDING: establish background and continuity — including what they actually want to explore — before you probe harder.\n"
    "- DYNAMIC: stay with the lived beat — what happened in the doing, not only the headline.\n"
    "- EMOTIONAL: invite feeling-tone and inner texture once the scene is clear enough to hold it.\n\n"
    "Opening of the session: unless they have already named what they want to explore, your first reply must explicitly ask "
    "what they want to talk about — plain and direct, in your own words (no clever dodge). A short warm greeting first is fine. "
    "If they only say hello or stay vague, ask again next turn until there is a direction. "
    "If they ask you what to talk about instead of naming a topic, help them choose in plain language so they still land on "
    "something concrete — not a poetic sidestep. "
    "Once they name something, use one or two context-building follow-ups before dynamic or emotional depth.\n\n"
    "After that, prioritize CONTEXT BUILDING whenever grounding is still thin; move to dynamic and emotional layers as the "
    "picture firms up, unless they are already deep in feeling — then meet them there.\n\n"
    "When they ask what to talk about or for a topic, do not default to a familiar opener shape. "
    "Use a different angle and rhythm each time; avoid generic mental-bandwidth or headspace formulas.\n\n"
    "Light mirroring: start many turns with a very short echo of their words or gist — one phrase or half a sentence, "
    "not a long paraphrase — then your question. The mirror is not the question.\n\n"
    "Keep it short. Keep it natural. One question at a time. "
    "No advice, no clinical language. "
    "Just good questions and the occasional honest observation."
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
) -> tuple[AIMessage, list[dict], list[dict]]:
    """
    Multi-turn chat completion (OpenRouter) with allowlisted site tools.
    Returns (assistant message, agent_steps, client_actions).
    """
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
            return AIMessage(content=text), agent_steps, client_actions

        oai_messages.append(_assistant_message_to_dict(msg))
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            log_tool_invocation(name, instance_id, json.dumps(args)[:400])

            if name == "navigate_ui":
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
            "Do not offer **navigate_ui** for destructive operations. After tool success, reply briefly and warmly.",
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
            "After tool success, reply briefly and warmly. ",
        ]

    if personalization > 0 and mode_raw != "autobiography":
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
        response, tool_steps, nav_actions = _interviewer_run_with_tools(
            client, model, oai_messages, instance_id, extra_body=extra_body
        )
        agent_steps = list(tool_steps)
        client_actions = list(nav_actions)
    except Exception as e:
        print("[backend] interviewer tool path failed; plain OpenRouter chat fallback:", e)
        response = llm.invoke(lc_messages)
        agent_steps = []
        client_actions = []

    if retrieval_log is not None and personalization > 0:
        agent_steps.insert(0, {"kind": "retrieval", "summary": "Retrieved relevant context from your journals"})

    out: dict = {"messages": [response], "agent_steps": agent_steps}
    if retrieval_log is not None:
        out["retrieval_log"] = retrieval_log
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
