"""
ChromaDB library for Open-Journal: gist_facts (semantic) and episodic_log (episodic) memory.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import chromadb

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from langchain_openai import ChatOpenAI
from voyageai import Client as VoyageClient

# Paths
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_data"
COLLECTION_GIST = "gist_facts"
COLLECTION_EPISODIC = "episodic_log"
COLLECTION_CONSUMED = "consumed_content"

# Clients (lazy init)
_client: chromadb.PersistentClient | None = None
_voyage: VoyageClient | None = None
_llm: ChatOpenAI | None = None
_recommendations_llm: ChatOpenAI | None = None


def _get_chroma() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def _get_voyage() -> VoyageClient:
    global _voyage
    if _voyage is None:
        key = os.getenv("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY is required. Get one at https://dash.voyageai.com/")
        _voyage = VoyageClient(api_key=key)
    return _voyage


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required. Get one at https://openrouter.ai/keys")
        _llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.3,
        )
    return _llm


def _get_recommendations_llm() -> ChatOpenAI:
    """Dedicated LLM for recommendations: uses Gemini for accurate article URLs (non-negotiable)."""
    global _recommendations_llm
    if _recommendations_llm is None:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required for recommendations")
        _recommendations_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=os.getenv("RECOMMENDATIONS_MODEL", "google/gemini-3.1-pro-preview"),
            temperature=0.2,
        )
    return _recommendations_llm


def _ensure_collections():
    """Create gist_facts, episodic_log, and consumed_content collections with cosine similarity."""
    client = _get_chroma()
    for name in [COLLECTION_GIST, COLLECTION_EPISODIC, COLLECTION_CONSUMED]:
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(name, metadata={"hnsw:space": "cosine"})


def wipe_memory() -> None:
    """Delete Chroma collections and recreate them empty (gist, episodic only; consumed_content kept)."""
    client = _get_chroma()
    for name in [COLLECTION_GIST, COLLECTION_EPISODIC]:
        try:
            client.delete_collection(name=name)
        except Exception:
            pass
    _ensure_collections()


def add_consumed(
    content_type: str,
    title: str,
    *,
    author: str | None = None,
    url: str | None = None,
    liked: bool = True,
) -> None:
    """
    Record that the user has read/listened to a recommendation (book, podcast, article).
    Stored in Chroma for the recommendation agent to avoid re-suggesting and to learn preferences.
    """
    _ensure_collections()
    client = _get_chroma()
    voyage = _get_voyage()
    ts = datetime.utcnow().isoformat() + "Z"
    ts_safe = ts.replace(":", "-").replace(".", "-")
    doc = f"User consumed {content_type}: {title}"
    if author:
        doc += f" by {author}"
    doc += f". Liked: {'yes' if liked else 'no'}."
    emb = voyage.embed(
        [doc],
        model=os.getenv("VOYAGE_MODEL", "voyage-3"),
        input_type="document",
    ).embeddings[0]
    col = client.get_collection(COLLECTION_CONSUMED)
    uid = f"consumed_{ts_safe}_{hash(title) % 10**8}"
    col.add(
        ids=[uid],
        embeddings=[emb],
        documents=[doc],
        metadatas=[{
            "type": content_type,
            "title": title[:500],
            "author": (author or "")[:300],
            "url": (url or "")[:500],
            "liked": liked,
            "timestamp": ts,
        }],
    )


def get_consumed_context(max_items: int = 80) -> str:
    """
    Return a single string describing what the user has consumed and liked,
    for injection into the recommendation agent prompt.
    """
    _ensure_collections()
    client = _get_chroma()
    try:
        col = client.get_collection(COLLECTION_CONSUMED)
        if col.count() == 0:
            return "The user has not marked any books, podcasts, or articles as read/listened yet."
        res = col.get(
            include=["metadatas"],
            limit=min(max_items, col.count()),
        )
        metadatas = res.get("metadatas") or []
        lines = []
        for m in metadatas:
            if not m:
                continue
            t = m.get("type", "item")
            title = m.get("title", "?")
            author = m.get("author", "")
            liked = m.get("liked", True)
            line = f"- {t}: {title}"
            if author:
                line += f" ({author})"
            line += " — liked" if liked else " — read/listened"
            lines.append(line)
        return "What the user has already read or listened to (do not recommend these again; use their tastes to suggest similar things):\n" + "\n".join(lines)
    except Exception:
        return "The user has not marked any recommendations as consumed yet."


def _extract_session_data(transcript: str) -> dict:
    """Use OpenRouter LLM to extract summary and facts from transcript."""
    llm = _get_llm()
    prompt = f"""You are a journal analyst. Extract structured data from this journal session transcript.

Transcript:
---
{transcript}
---

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "summary": "A 3-sentence summary of the session: what was discussed, key themes, and emotions felt.",
  "facts": ["Fact 1 about the user", "Fact 2 about the user", ...]
}}

Rules:
- summary: exactly 3 sentences, capture emotions and themes
- facts: list of hard, verifiable facts about the user (job, relationships, preferences, life events). Empty list if none.
"""
    response = llm.invoke(prompt)
    text = response.content.strip()
    # Strip markdown code blocks if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def save_session_data(session_id: str, transcript: str) -> None:
    """
    Extract summary + facts from transcript via OpenRouter, embed via Voyage, save to ChromaDB.
    """
    _ensure_collections()
    client = _get_chroma()
    voyage = _get_voyage()

    data = _extract_session_data(transcript)
    summary = data.get("summary", "")
    facts = data.get("facts", [])

    ts = datetime.utcnow().isoformat() + "Z"

    # Episodic: add summary with metadata
    if summary:
        ep_col = client.get_collection(COLLECTION_EPISODIC)
        summary_emb = voyage.embed(
            [summary],
            model=os.getenv("VOYAGE_MODEL", "voyage-3"),
            input_type="document",
        ).embeddings[0]
        ep_id = f"{session_id}_ep_{ts.replace(':', '-').replace('.', '-')}"
        ep_col.add(
            ids=[ep_id],
            embeddings=[summary_emb],
            documents=[summary],
            metadatas=[{"session_id": session_id, "timestamp": ts}],
        )

    # Gist: add each fact
    if facts:
        gist_col = client.get_collection(COLLECTION_GIST)
        fact_embs = voyage.embed(
            facts,
            model=os.getenv("VOYAGE_MODEL", "voyage-3"),
            input_type="document",
        ).embeddings
        ts_safe = ts.replace(":", "-").replace(".", "-")
        gist_col.add(
            ids=[f"{session_id}_gist_{i}_{ts_safe}" for i in range(len(facts))],
            embeddings=fact_embs,
            documents=facts,
            metadatas=[{"session_id": session_id, "timestamp": ts} for _ in facts],
        )


def get_relevant_context(query: str, top_k_gist: int = 8, top_k_episodic: int = 5) -> str:
    """
    Embed the query, retrieve relevant gist facts and episodic summaries from Chroma,
    and return a single string for injection into the interviewer's context.
    """
    if not query or not query.strip():
        return "None."
    _ensure_collections()
    client = _get_chroma()
    voyage = _get_voyage()
    model = os.getenv("VOYAGE_MODEL", "voyage-3")
    query_emb = voyage.embed([query.strip()], model=model, input_type="document").embeddings[0]

    parts = []
    try:
        gist_col = client.get_collection(COLLECTION_GIST)
        if gist_col.count() > 0:
            gist_res = gist_col.query(
                query_embeddings=[query_emb],
                n_results=min(top_k_gist, gist_col.count()),
                include=["documents"],
            )
            docs = (gist_res.get("documents") or [[]])[0] or []
            if docs:
                parts.append("Facts and details from the user's life and journals:\n" + "\n".join(f"- {d}" for d in docs))
    except Exception:
        pass

    try:
        ep_col = client.get_collection(COLLECTION_EPISODIC)
        if ep_col.count() > 0:
            ep_res = ep_col.query(
                query_embeddings=[query_emb],
                n_results=min(top_k_episodic, ep_col.count()),
                include=["documents"],
            )
            docs = (ep_res.get("documents") or [[]])[0] or []
            if docs:
                parts.append("Relevant journal summaries:\n" + "\n".join(f"- {d}" for d in docs))
    except Exception:
        pass

    if not parts:
        return "None."
    return "\n\n".join(parts)


def get_memory_for_visualization() -> tuple[list[str], list[str]]:
    """
    Return (gist_facts, episodic_summaries) as lists of document strings for diagram generation.
    """
    _ensure_collections()
    client = _get_chroma()
    gist_docs: list[str] = []
    episodic_docs: list[str] = []

    try:
        gist_col = client.get_collection(COLLECTION_GIST)
        if gist_col.count() > 0:
            res = gist_col.get(include=["documents"])
            gist_docs = list(res.get("documents") or [])
    except Exception:
        pass

    try:
        ep_col = client.get_collection(COLLECTION_EPISODIC)
        if ep_col.count() > 0:
            res = ep_col.get(include=["documents"])
            episodic_docs = list(res.get("documents") or [])
    except Exception:
        pass

    return (gist_docs, episodic_docs)


def generate_memory_mermaid(gist_facts: list[str], episodic_summaries: list[str]) -> str:
    """
    Use LLM to generate a Mermaid diagram (mindmap or flowchart) from vector DB content.
    Returns the raw Mermaid code string.
    """
    if not gist_facts and not episodic_summaries:
        return """mindmap
  root((Your memory))
    Empty
    Start journaling to see facts and themes here"""

    llm = _get_llm()
    facts_blob = "\n".join(f"- {f}" for f in (gist_facts or [])[:50])
    summaries_blob = "\n".join(f"- {s}" for s in (episodic_summaries or [])[:30])

    prompt = f"""You are a visualization expert. Create a single Mermaid diagram that represents this person's journaled memory in a satisfying, visual way.

FACTS ABOUT THE USER (from their journals):
{facts_blob or "(none)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none)"}

Instructions:
- Output ONLY valid Mermaid code. No markdown code fence, no explanation.
- Use a mindmap diagram with root "My journal" or "Memory" and organize facts and themes into clear branches (e.g. Work, Relationships, Health, Goals, Emotions). Keep node labels SHORT (a few words) so the diagram stays readable.
- If you have both facts and summaries, group related items under thematic branches. Make it feel personal and reflective of what they wrote.
- Use simple labels; avoid long sentences. Use parentheses for the root: root((My journal)).
- Maximum 30–40 nodes total to keep the diagram clean."""

    response = llm.invoke(prompt)
    code = response.content.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code or """mindmap
  root((My journal))
    Add entries to see your memory here"""


def _parse_recommendation_json(text: str, default: list) -> list:
    """Parse LLM JSON output into list of dicts with title, author, reason, url."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return default
    if not isinstance(raw, list):
        return default
    return [
        {
            "title": isinstance(x, dict) and x.get("title") or str(x),
            "author": isinstance(x, dict) and x.get("author") or "",
            "reason": isinstance(x, dict) and x.get("reason") or "",
            "url": isinstance(x, dict) and x.get("url") or "",
        }
        for x in raw
    ]


def _books_agent(facts_blob: str, summaries_blob: str, consumed: str) -> list:
    """Dedicated agent for book recommendations only."""
    llm = _get_recommendations_llm()
    prompt = f"""You are a book curator. Based on this person's journal-derived memory and what they have already read, suggest 3–5 books they might find helpful or comforting.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

Rules: Do NOT suggest books they have already consumed. For each book give title, author, and a short "reason" (one sentence) tied to their life or journal themes. No URLs.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "..."}}, ...]"""
    response = llm.invoke(prompt)
    return _parse_recommendation_json(response.content, [])


LISTEN_NOTES_BASE = "https://listen-api.listennotes.com/api/v2"


def _listen_notes_search_episodes(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    """Call Listen Notes API search (type=episode). Returns list of {title, author, url, reason}."""
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode({"q": query, "type": "episode", "only_one_episode_per_podcast": 1})
    url = f"{LISTEN_NOTES_BASE}/search?{qs}"
    req = urllib.request.Request(url, headers={"X-ListenAPI-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print("[backend] Listen Notes API error:", e)
        return []
    results = data.get("results") or []
    out = []
    for r in results:
        title = (r.get("title_original") or r.get("title_highlighted") or "").strip()
        podcast = r.get("podcast") or {}
        author = (podcast.get("title_original") or podcast.get("title_highlighted") or "").strip()
        ln_url = (r.get("listennotes_url") or "").strip()
        if title and ln_url:
            out.append({
                "title": title[:500],
                "author": author[:300],
                "reason": f"Suggested based on your interests: {query[:80]}.",
                "url": ln_url,
            })
        if len(out) >= max_results:
            break
    return out


def _podcasts_agent(facts_blob: str, summaries_blob: str, consumed: str) -> list:
    """Dedicated podcast agent: uses Listen Notes API for real episode links when key is set."""
    api_key = (os.getenv("LISTENNOTES_API_KEY") or "").strip()
    if api_key:
        llm = _get_recommendations_llm()
        prompt = f"""Based on this person's journal-derived memory and what they have already listened to, suggest 2–3 short search queries (topics or themes) to find relevant podcast episodes. Examples: "mindfulness sleep", "anxiety therapy", "Huberman Lab sleep".

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

Return ONLY a JSON array of 2–3 short search query strings, no markdown. Example: ["mindfulness and sleep", "therapy for anxiety"]"""
        try:
            response = llm.invoke(prompt)
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            queries = json.loads(text)
            if not isinstance(queries, list):
                queries = []
        except Exception as e:
            print("[backend] Podcast agent query parsing:", e)
            queries = []
        consumed_lower = consumed.lower()
        seen_urls = set()
        out = []
        per_query = 2
        for q in queries[:3]:
            if not isinstance(q, str) or not q.strip():
                continue
            for item in _listen_notes_search_episodes(q.strip(), api_key, max_results=per_query):
                if item["url"] in seen_urls:
                    continue
                if item["title"].lower() in consumed_lower or (item["author"] and item["author"].lower() in consumed_lower):
                    continue
                seen_urls.add(item["url"])
                out.append(item)
                if len(out) >= 5:
                    break
            if len(out) >= 5:
                break
        if out:
            return out
    llm = _get_recommendations_llm()
    prompt = f"""You are a podcast curator. Based on this person's journal-derived memory and what they have already listened to, suggest 3–5 specific podcast episodes (show + episode).

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

Rules: Use "author" for the show name and "title" for the episode. Provide direct "url" (Spotify or Apple Podcasts episode link) when you know it; otherwise leave "url" empty. Do NOT suggest items they have already consumed. For each give a short "reason" (one sentence).
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    response = llm.invoke(prompt)
    return _parse_recommendation_json(response.content, [])


def _articles_agent(facts_blob: str, summaries_blob: str, consumed: str) -> list:
    """Dedicated agent for article recommendations only. Article URLs must work (non-negotiable)."""
    llm = _get_recommendations_llm()
    prompt = f"""You are an article curator. Based on this person's journal-derived memory and what they have already read, suggest 3–5 news articles or long-reads.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

CRITICAL: Every article MUST have a "url" that is a real, working link to the actual article page—no 404s. Use only URLs you are certain exist (exact paths from nytimes.com, theatlantic.com, healthline.com, bbc.com, nature.com, apa.org, etc.). Do NOT guess or construct URLs. If you cannot provide a verified working URL for an article, do NOT include it. For each give title, author/source, reason, and url.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    response = llm.invoke(prompt)
    return _parse_recommendation_json(response.content, [])


def _research_agent(facts_blob: str, summaries_blob: str, consumed: str) -> list:
    """Dedicated agent for research paper recommendations only."""
    llm = _get_recommendations_llm()
    prompt = f"""You are a research curator. Based on this person's journal-derived memory and what they have already read, suggest 3–5 academic or scientific research papers (peer-reviewed articles, studies, or review papers) they might find relevant or helpful.

FACTS AND THEMES FROM THEIR JOURNALS:
{facts_blob or "(none yet)"}

JOURNAL SESSION SUMMARIES:
{summaries_blob or "(none yet)"}

{consumed}

Rules: Do NOT suggest papers they have already consumed. For each paper give: "title" (paper title), "author" (lead author or author list, or journal name and year), "reason" (one sentence on why it fits their interests), and "url" (working link to the paper—DOI link like https://doi.org/10.1234/... or publisher link to the abstract/full text). Only include papers where you can provide a real, working url (doi.org, PubMed, PMC, journal websites). Do not guess URLs.
Return ONLY a JSON array, no markdown: [{{"title": "...", "author": "...", "reason": "...", "url": "..."}}, ...]"""
    response = llm.invoke(prompt)
    return _parse_recommendation_json(response.content, [])


def generate_recommendations() -> dict:
    """
    Run four dedicated agents (books, podcasts, articles, research) in parallel, each with
    its own prompt and specialization. Combines results into one response.
    """
    gist_docs, episodic_docs = get_memory_for_visualization()
    consumed = get_consumed_context()
    facts_blob = "\n".join(f"- {f}" for f in (gist_docs or [])[:60])
    summaries_blob = "\n".join(f"- {s}" for s in (episodic_docs or [])[:40])

    from concurrent.futures import ThreadPoolExecutor
    books_list: list = []
    podcasts_list: list = []
    articles_list: list = []
    research_list: list = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_books = executor.submit(_books_agent, facts_blob, summaries_blob, consumed)
        future_podcasts = executor.submit(_podcasts_agent, facts_blob, summaries_blob, consumed)
        future_articles = executor.submit(_articles_agent, facts_blob, summaries_blob, consumed)
        future_research = executor.submit(_research_agent, facts_blob, summaries_blob, consumed)
        try:
            books_list = future_books.result()
            podcasts_list = future_podcasts.result()
            articles_list = future_articles.result()
            research_list = future_research.result()
        except Exception as e:
            print("[backend] recommendations agent error:", e)

    return {
        "books": books_list if isinstance(books_list, list) else [],
        "podcasts": podcasts_list if isinstance(podcasts_list, list) else [],
        "articles": articles_list if isinstance(articles_list, list) else [],
        "research": research_list if isinstance(research_list, list) else [],
    }
