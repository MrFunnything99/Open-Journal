"""
LangGraph for Open-Journal: Interviewer (chat) and Librarian (end-session).
"""
from __future__ import annotations

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

from library import get_relevant_context, save_session_data

# State: list of messages + session_id for Librarian + personalization + intrusiveness + mode + optional retrieval log + instance_id
class JournalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    personalization: float
    intrusiveness: NotRequired[float]
    mode: NotRequired[str]  # "journal" | "extreme" (recommendations handled in main)
    retrieval_log: NotRequired[str]
    last_transcript: NotRequired[str]
    last_summary: NotRequired[str]
    last_facts: NotRequired[list]
    instance_id: NotRequired[str]  # X-Instance-ID for per-device data isolation


def _get_llm():
    """Return a lightweight wrapper around xAI Grok 4.20 reasoning (direct API) with an .invoke(messages) API."""
    from openai import OpenAI

    key = os.getenv("XAI_API_KEY")
    if not key:
        raise ValueError("XAI_API_KEY is required for Grok interviewer")
    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
    model = "grok-4.20-beta-0309-reasoning"

    class _GrokWrapper:
        def __init__(self, client: OpenAI, model: str):
            self._client = client
            self._model = model

        def invoke(self, messages: list[BaseMessage]):
            oai_messages = []
            for m in messages:
                content = getattr(m, "content", str(m))
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", str(c)) for c in content if isinstance(c, dict)
                    )
                content = (content or "").strip()
                if isinstance(m, SystemMessage):
                    oai_messages.append({"role": "system", "content": content})
                elif isinstance(m, HumanMessage):
                    oai_messages.append({"role": "user", "content": content})
                elif isinstance(m, AIMessage):
                    oai_messages.append({"role": "assistant", "content": content})
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=oai_messages,
                max_tokens=4096,
            )
            text = (resp.choices[0].message.content or "").strip()
            return AIMessage(content=text)

    return _GrokWrapper(client, model)


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
    is_extreme = mode_raw == "extreme"
    is_therapy = mode_raw == "therapy"

    if is_therapy:
        system_parts = [
            f"Today's date: {date_context}. ",
            "THERAPY MODE. You are a warm, grounded therapeutic presence. Your goal is to support the user through reflection and careful listening—not to give advice or diagnose. ",
            "Use the space well: reflect back what you hear ('It sounds like...', 'What I'm hearing is...'), ask about feelings and meaning ('How does that sit with you?', 'What does that bring up for you?'), and invite depth when it fits ('Tell me more.', 'Where do you feel that?'). ",
            "Validate without minimizing. Ask one or two thoughtful questions per reply. Keep responses concise (2–4 sentences). Do not lecture or interpret; stay curious and present. ",
            "You may gently explore patterns, earlier experiences, or relationships when the user opens the door—but follow their lead and do not push. ",
            "Use memory context only as loose inspiration; do not assert or invent details. ",
        ]
    elif is_extreme:
        system_parts = [
            f"Today's date: {date_context}. ",
            "EXTREME MODE. Your only goal is to help the user feel better by making them reflect—not by giving advice. ",
            "Do NOT give advice, suggestions, or solutions. Do NOT summarize or lecture. ",
            "Build context first. Before asking hard or intrusive questions, take 1–3 exchanges to understand the situation: what's going on, who's involved, how long it's been, what they've noticed or tried. Ask gentle, open questions (e.g. 'What's going on?' 'How long has this been on your mind?' 'What does that look like for you?') until you have enough of the picture. ",
            "Only once you have some context, move into more direct, intrusive, or private questions that push reflection. Do not jump straight to the hardest question—earn it by showing you're listening. ",
            "Validate briefly (e.g. 'That makes sense,' 'I hear you') then ask your question. ",
            "NSFW and adult or intimate topics are okay in this mode; the user has opted in. You may ask about sex, desire, body image, relationships, or other sensitive topics when it serves reflection. Stay warm and non-judgmental. ",
            "Go deep when the moment is right: relationships, fears, hopes, shame, joy, what they avoid, what they want to change, how they really feel. ",
            "One or two questions per reply. Keep replies short (2–4 sentences). Be warm but unafraid to ask the hard thing once context is there. ",
            "Use memory context only as inspiration for what to ask about; do not assert details. ",
        ]
    else:
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
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    out: dict = {"messages": [response]}
    if retrieval_log is not None:
        out["retrieval_log"] = retrieval_log
    return out


def librarian_node(state: JournalState) -> JournalState:
    """Background worker: extract, embed, save to SQLite+sqlite-vec (and optionally LightRAG)."""
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
