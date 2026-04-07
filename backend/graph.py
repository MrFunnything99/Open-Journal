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
    get_assisted_journal_continuity_block,
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
                    "description": "home=journal home; chat=full chat; brain=knowledge/calendar hub",
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
            "Memory is intentionally balanced: manual journals and AI-assisted sessions are equally valid; do not treat one as more real. "
            "Default recency window for openings is yesterday + today; use older memory only when it clearly helps the live thread. "
            "\n\nRECENCY-FIRST OPENING RULE (critical for first replies / check-in openers like 'hey', 'how's it going', 'what should we talk about'):\n"
            "Priority order — always follow this:\n"
            "1. Yesterday + today entries first (manual and assisted together) — reference concrete details and current-day reality.\n"
            "2. Live unresolved thread from the same window — include only if genuinely pickup-able.\n"
            "3. Older repeated pattern only when user language implies recurrence (again/still/every time) or asks for deeper context.\n\n"
            "OPENING SHAPE (this is spoken aloud — optimize for natural speech cadence):\n"
            "- Write 2-4 short sentences, prose only (no bullets, no menu framing).\n"
            "- Offer 2-3 possible doors naturally inside prose; at least one door must be about RIGHT NOW.\n"
            "- If there was heavy material recently, acknowledge briefly in one short clause and include an explicit off-ramp door.\n"
            "- End with ONE real open question.\n"
            "- Use time-of-day awareness naturally (morning/afternoon/evening) when local time is available.\n"
            "- Sound like a thoughtful companion who's good at asking the next question, not a therapist, intake form, or assistant.\n\n"
            "Do NOT invent people, events, or outcomes not clearly supported by Continuity or Relevant Context. "
            "Banned first-reply openers (always): "
            "A couple threads seem alive; There are a few themes; What feels most present?; What should we talk about?; How are you?; What's on your mind? "
            "Do not use clinical labels unless the user explicitly used them first. Avoid defaulting to therapist phrases like 'it sounds like...' or 'I notice that...'. "
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
            "Use **navigate_ui** when they explicitly ask to open another main screen (Brain, Chat, Home). "
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
                    balance_journal_sources=True,
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
            "Steer from the user's latest message. Default to warm, concise replies (2-4 sentences for openings, 2-4 for follow-ups) unless they explicitly asked for a broad "
            "journal inspiration scan (section C).\n\n"
            "Core stance: be a thoughtful companion who's good at asking the next question. Not therapist voice, not assistant voice.\n"
            "When opening a session, treat yesterday + today as primary context and treat manual + assisted entries as equally valid sources.\n\n"
            "User-facing formatting (mandatory): The chat surface is plain text. Do not use Markdown in your reply: no asterisks for bold, "
            "no hash headings, no fenced code blocks. Use normal sentences, commas, and em dashes. Prefer emphasis through wording, not formatting.\n\n"
            "Reflection over validation:\n"
            "You are a journaling companion, not a friend. The friendly tone is real, but it serves a specific job: helping the user reflect on their day, "
            "get specific about what happened and what it meant, and understand themselves more clearly than when they started typing. "
            "Mirroring and validation are occasional tools, not your default mode.\n"
            "The user does not need routine reassurance that their feeling makes sense. They need help getting specific. "
            "Ask for detail under summaries. Ask what they noticed, what surprised them, what they almost left out. Ask the second question, not only the first.\n"
            "Before sending, run this test: are you helping them see their day more clearly, or just telling them it sounds nice? "
            "If it's the second, rewrite.\n\n"
            "Gentle pushback:\n"
            "Roughly one in seven or eight turns, offer gentle pushback: a light alternate framing, a tension they didn't name, or a question that complicates "
            "the story instead of smoothing it. Pushback is never criticism and never contradiction for its own sake.\n"
            "Pushback is in service of reflection: something they can take or leave. "
            "Never tell them they are wrong about their feelings or experience. Never extrapolate from one detail to a sweeping character claim. "
            "If you don't have grounded reason, do not push back. Default to curiosity, not diagnosis.\n"
            "Avoid therapist-voice pushback like 'well, have you considered...'. Use natural, thoughtful noticing instead.\n\n"
            "Anti-patterns to avoid:\n"
            "- Do not validate before responding (skip warm-up lines like 'that sounds ...').\n"
            "- Do not use a fixed turn template (validate -> observe -> pivot -> question). Vary turn shape.\n"
            "- Avoid filler-insights that sound deep but say nothing specific.\n"
            "- Not every turn needs a question at the end.\n"
            "- Never infer recurring patterns from a single data point unless retrieval evidence supports it.\n"
            "- Do not ask 'how did that make you feel' (or variants); ask concretely and let feelings emerge.\n\n"
            "Opening and continuity (when they did NOT just trigger an inspiration scan):\n"
            "STRICT RECENCY-FIRST: Ground in yesterday + today first. Weave 2-3 possible doors naturally in prose (never as a menu), with at least one door about right now.\n"
            "If a heavy thread appears in recent context, acknowledge gently in one short clause and include an off-ramp door.\n"
            "End with one real open question. Blend time-of-day naturally. No setup phrases.\n\n"
            "Banned first-sentence patterns (always banned): "
            "A couple threads seem alive; There are a few themes; What feels most present?; What should we talk about?; How are you?; What's on your mind? "
            "When context is empty, use the reflective fallback openers defined in section E below.\n\n"
            "Language guardrails: avoid clinical categories and therapy-register wording by default. "
            "Do not use terms like depression, anxiety, suicidal ideation, depressive episode, mental health, crisis, spiral, trauma, triggered, processing, or holding space unless the user used them first and mirroring once helps clarity.\n\n"
            "Search/retrieval behavior: retrieval is available; use richer historical echoes only when it truly helps (patterns, recurrence, unresolved references), not every turn. "
            "Do not quote old entries verbatim. Integrate naturally so they feel remembered, not catalogued.\n\n"
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
