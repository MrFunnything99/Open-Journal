"""
LangGraph for Open-Journal: Interviewer (chat) and Librarian (end-session).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from library import save_session_data

# State: list of messages + session_id for Librarian
class JournalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str


def _get_llm():
    import os
    from langchain_openai import ChatOpenAI
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        temperature=0.7,
    )


def interviewer_node(state: JournalState) -> JournalState:
    """Echo back an empathetic response. No DB read for this sprint."""
    llm = _get_llm()
    system = SystemMessage(
        content="You are a warm, empathetic journaling companion. Listen actively and respond with care. Keep replies concise (2-4 sentences)."
    )
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


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
