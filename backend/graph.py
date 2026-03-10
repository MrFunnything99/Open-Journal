"""
LangGraph for Open-Journal: Interviewer (chat) and Librarian (end-session).
"""
from __future__ import annotations

import os
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

# State: list of messages + session_id for Librarian + personalization + intrusiveness + optional retrieval log
class JournalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    personalization: float
    intrusiveness: NotRequired[float]
    retrieval_log: NotRequired[str]


def _get_llm():
    """Return a lightweight wrapper around Gemini Flash 3.1 with an .invoke(messages) API."""
    import os
    from google import genai

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is required")
    client = genai.Client(api_key=key)
    model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash")

    class _GeminiWrapper:
        def __init__(self, client: genai.Client, model: str):
            self._client = client
            self._model = model

        def invoke(self, messages: list[BaseMessage]):
            # Flatten LangChain messages into a simple chat transcript prompt.
            parts: list[str] = []
            for m in messages:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                content = getattr(m, "content", str(m))
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", str(c)) for c in content if isinstance(c, dict)
                    )
                parts.append(f"{role}: {content}")
            prompt = "\n\n".join(parts)
            result = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            text = getattr(result, "text", "") or ""
            return AIMessage(content=text.strip())

    return _GeminiWrapper(client, model)


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
    Chroma (gist_facts + episodic_log) and inject it so the model can personalize.
    """
    llm = _get_llm()
    personalization = max(0.0, min(1.0, state.get("personalization", 1.0)))
    personalization_percent = int(personalization * 100)
    intrusiveness = max(0.0, min(1.0, state.get("intrusiveness", 0.5)))
    intrusiveness_percent = int(intrusiveness * 100)

    system_parts = [
        "You are a warm, empathetic journaling companion. Listen actively and respond with care. "
        "Keep replies concise (2-4 sentences). "
        f"Personalization level: {personalization_percent}%. "
        "At 0%, do not use memory or prior context; keep questions general and present-focused only. "
        "At low levels, keep questions more general and present-focused. "
        "At high levels, ask more personalized questions that connect to what you know about the user's life and past journals. "
        "When you use memory, prefer concrete topics over vague emotion labels: reference specific people, events, places, or recurring themes from prior entries (for example, 'your time in the Annex', 'your relationship with X', or 'that exam you mentioned'). "
        "Do NOT merely repeat that the user felt afraid, anxious, stressed, etc., unless they explicitly ask you to summarize feelings. "
        "If the user asks meta-questions like 'what should we talk/journal about today?' or 'what should we explore?', you MUST use memory to suggest 2-3 specific follow-up topics grounded in past sessions instead of only restating general feelings. "
        "For example: 'We could explore more about how it felt to hide in the Annex during the air raids', 'We might talk about your relationship with your parents in the Annex', or 'We could revisit how you coped with the long days indoors.' "
        "When memory mentions a specific situation (like hiding from the Gestapo in the Annex), your answer to 'what should we talk about?' MUST explicitly name that situation and propose it as a topic.",
        f"Questioning style (intrusiveness): {intrusiveness_percent}%. "
        "At 0%, be very gentle and non-intrusive; ask only soft, open-ended questions and let the user lead entirely. "
        "At low levels, ask sparingly and avoid probing. "
        "At high levels, you may ask more direct or probing questions when it feels supportive, while still respecting boundaries.",
        "\nExample behavior (for reference, DO NOT quote this back):\n"
        "Memory context: 'The session reflects a deep sense of fear and anxiety experienced by the speaker and those in hiding in the Annex...'\n"
        "User: 'What do you think we should talk about?'\n"
        "Good answer: 'Given what you've shared about hiding in the Annex and the constant threat of being discovered, we could talk more about how those long days indoors affected you, your relationship with the others in hiding, or specific moments that felt especially frightening or hopeful. Which of those feels most present for you today?'\n"
        "Bad answer (avoid): 'You've felt anxious before, we could talk about your anxiety.'",
    ]

    retrieval_log: str | None = None
    if personalization > 0:
        query = _last_user_text(state["messages"])
        if not query and state["messages"]:
            query = str(state["messages"][-1])[:500]
        try:
            context = get_relevant_context(query, top_k_gist=10, top_k_episodic=6)
            system_parts.append("\n\nRelevant context from the user's journals and memory (use this to personalize when appropriate):\n" + context)
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
    """Background worker: extract, embed, save to ChromaDB."""
    session_id = state.get("session_id", "default")
    transcript = _messages_to_transcript(state["messages"])
    save_session_data(session_id, transcript)
    return {"messages": []}


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
