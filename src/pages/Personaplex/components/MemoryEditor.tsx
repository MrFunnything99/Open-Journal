import { useCallback, useMemo, useState } from "react";
import { BrainPeopleView } from "./BrainPeopleView";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export type MemoryFact = {
  id: number;
  document: string;
  session_id?: string;
  timestamp?: string;
};

export type MemorySummary = {
  id: number;
  document: string;
  session_id?: string;
  timestamp?: string;
  metadata_json?: string | null;
};

export type MemoryStatsState = {
  gist_facts_count: number;
  episodic_log_count: number;
  episodic_metadata_count?: number;
} | null;

export type MemoryViewTab = "time" | "person" | "topic" | "place" | "activity" | "emotion";

type Metadata = {
  people?: string[];
  topics?: string[];
  activities?: string[];
  events?: string[];
  emotions?: string[];
};

function parseMetadata(metadata_json: string | null | undefined): Metadata {
  if (!metadata_json || !metadata_json.trim()) return {};
  try {
    const o = JSON.parse(metadata_json) as Record<string, unknown>;
    return {
      people: Array.isArray(o.people) ? o.people.map(String) : [],
      topics: Array.isArray(o.topics) ? o.topics.map(String) : [],
      activities: Array.isArray(o.activities) ? o.activities.map(String) : [],
      events: Array.isArray(o.events) ? o.events.map(String) : [],
      emotions: Array.isArray(o.emotions) ? o.emotions.map(String) : [],
    };
  } catch {
    return {};
  }
}

function formatDate(iso: string | undefined): string {
  if (!iso) return "Unknown date";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function dateKey(iso: string | undefined): string {
  if (!iso) return "unknown";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "unknown";
  return d.toISOString().slice(0, 10);
}

export type MemoryEditorProps = {
  facts: MemoryFact[];
  summaries: MemorySummary[];
  stats: MemoryStatsState;
  loading: boolean;
  onRefresh: () => void;
  onRefreshStats: () => void;
  onToast: (msg: string) => void;
  onWipeMemory?: () => void;
  isWipingMemory?: boolean;
};

export function MemoryEditor({
  facts,
  summaries,
  stats,
  loading,
  onRefresh,
  onRefreshStats,
  onToast,
  onWipeMemory,
  isWipingMemory = false,
}: MemoryEditorProps) {
  const [viewTab, setViewTab] = useState<MemoryViewTab>("person");
  const [editingFactId, setEditingFactId] = useState<number | null>(null);
  const [editingSummaryId, setEditingSummaryId] = useState<number | null>(null);
  const [editText, setEditText] = useState("");
  const [editMetadata, setEditMetadata] = useState<Metadata>({});
  const [saving, setSaving] = useState(false);
  const [addFactText, setAddFactText] = useState("");
  const [addSummaryText, setAddSummaryText] = useState("");
  const [addSaving, setAddSaving] = useState(false);

  const summariesByTime = useMemo(() => {
    const map = new Map<string, MemorySummary[]>();
    for (const s of summaries) {
      const key = dateKey(s.timestamp);
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(s);
    }
    const keys = Array.from(map.keys()).sort().reverse();
    return { keys, map };
  }, [summaries]);

  const summariesByPerson = useMemo(() => {
    const map = new Map<string, MemorySummary[]>();
    for (const s of summaries) {
      const meta = parseMetadata(s.metadata_json);
      const people = meta.people?.length ? meta.people : ["(No person)"];
      for (const p of people) {
        if (!map.has(p)) map.set(p, []);
        map.get(p)!.push(s);
      }
    }
    const keys = Array.from(map.keys()).sort();
    return { keys, map };
  }, [summaries]);

  const summariesByActivity = useMemo(() => {
    const map = new Map<string, MemorySummary[]>();
    for (const s of summaries) {
      const meta = parseMetadata(s.metadata_json);
      const activities = meta.activities?.length ? meta.activities : ["(No activity)"];
      for (const a of activities) {
        if (!map.has(a)) map.set(a, []);
        map.get(a)!.push(s);
      }
    }
    const keys = Array.from(map.keys()).sort();
    return { keys, map };
  }, [summaries]);

  const summariesByEmotion = useMemo(() => {
    const map = new Map<string, MemorySummary[]>();
    for (const s of summaries) {
      const meta = parseMetadata(s.metadata_json);
      const emotions = meta.emotions?.length ? meta.emotions : ["(No emotion)"];
      for (const e of emotions) {
        if (!map.has(e)) map.set(e, []);
        map.get(e)!.push(s);
      }
    }
    const keys = Array.from(map.keys()).sort();
    return { keys, map };
  }, [summaries]);

  const factsByDate = useMemo(() => {
    const map = new Map<string, MemoryFact[]>();
    for (const f of facts) {
      const key = dateKey(f.timestamp);
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(f);
    }
    return map;
  }, [facts]);

  const summariesByTopic = useMemo(() => {
    const map = new Map<string, MemorySummary[]>();
    for (const s of summaries) {
      const meta = parseMetadata(s.metadata_json);
      const topics = meta.topics?.length ? meta.topics : ["(No topic)"];
      for (const t of topics) {
        if (!map.has(t)) map.set(t, []);
        map.get(t)!.push(s);
      }
    }
    const keys = Array.from(map.keys()).sort();
    return { keys, map };
  }, [summaries]);

  const saveFact = useCallback(
    (id: number) => {
      if (!editText.trim()) return;
      setSaving(true);
      fetch(`${BACKEND_URL}/memory/facts/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document: editText.trim() }),
      })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
        .then(() => {
          onRefresh();
          onRefreshStats();
          setEditingFactId(null);
          setEditText("");
          onToast("Fact saved.");
        })
        .catch(() => onToast("Failed to save fact."))
        .finally(() => setSaving(false));
    },
    [editText, onRefresh, onRefreshStats, onToast]
  );

  const saveSummary = useCallback(
    (id: number) => {
      if (!editText.trim()) return;
      setSaving(true);
      fetch(`${BACKEND_URL}/memory/summaries/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document: editText.trim(), metadata: editMetadata }),
      })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
        .then(() => {
          onRefresh();
          onRefreshStats();
          setEditingSummaryId(null);
          setEditText("");
          setEditMetadata({});
          onToast("Summary saved.");
        })
        .catch(() => onToast("Failed to save summary."))
        .finally(() => setSaving(false));
    },
    [editText, editMetadata, onRefresh, onRefreshStats, onToast]
  );

  const deleteFact = useCallback(
    (id: number) => {
      if (!confirm("Delete this fact?")) return;
      fetch(`${BACKEND_URL}/memory/facts/${id}`, { method: "DELETE" })
        .then((r) => r.ok && onRefresh())
        .then(() => {
          onRefreshStats();
          onToast("Fact deleted.");
        })
        .catch(() => onToast("Failed to delete fact."));
    },
    [onRefresh, onRefreshStats, onToast]
  );

  const deleteSummary = useCallback(
    (id: number) => {
      if (!confirm("Delete this summary?")) return;
      fetch(`${BACKEND_URL}/memory/summaries/${id}`, { method: "DELETE" })
        .then((r) => r.ok && onRefresh())
        .then(() => {
          onRefreshStats();
          onToast("Summary deleted.");
        })
        .catch(() => onToast("Failed to delete summary."));
    },
    [onRefresh, onRefreshStats, onToast]
  );

  const addFact = useCallback(() => {
    if (!addFactText.trim()) return;
    setAddSaving(true);
    fetch(`${BACKEND_URL}/memory/facts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document: addFactText.trim() }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
      .then(() => {
        setAddFactText("");
        onRefresh();
        onRefreshStats();
        onToast("Fact added.");
      })
      .catch(() => onToast("Failed to add fact."))
      .finally(() => setAddSaving(false));
  }, [addFactText, onRefresh, onRefreshStats, onToast]);

  const addSummary = useCallback(() => {
    if (!addSummaryText.trim()) return;
    setAddSaving(true);
    fetch(`${BACKEND_URL}/memory/summaries`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document: addSummaryText.trim() }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
      .then(() => {
        setAddSummaryText("");
        onRefresh();
        onRefreshStats();
        onToast("Summary added.");
      })
      .catch(() => onToast("Failed to add summary."))
      .finally(() => setAddSaving(false));
  }, [addSummaryText, onRefresh, onRefreshStats, onToast]);

  const openEditSummary = (s: MemorySummary) => {
    setEditingSummaryId(s.id);
    setEditText(s.document);
    setEditMetadata(parseMetadata(s.metadata_json));
  };

  const openEditFact = (f: MemoryFact) => {
    setEditingFactId(f.id);
    setEditText(f.document);
  };

  const renderSummaryBlock = (s: MemorySummary) => (
    <div
      key={s.id}
      className="rounded-lg bg-slate-800/50 border border-slate-700/50 p-3 flex flex-col gap-2"
    >
      <div className="flex items-start justify-between gap-2">
        <button
          type="button"
          className="text-left text-sm text-slate-200 flex-1 min-w-0"
          onClick={() => openEditSummary(s)}
        >
          <span className="text-xs uppercase tracking-wide text-slate-400">
            Events & tags
          </span>
        </button>
        <button
          type="button"
          onClick={() => deleteSummary(s.id)}
          className="text-red-400 hover:text-red-300 text-xs font-medium shrink-0"
          aria-label="Delete summary"
        >
          Delete
        </button>
      </div>
      {(() => {
        const meta = parseMetadata(s.metadata_json);
        const hasMeta =
          (meta.events?.length ?? 0) > 0 ||
          (meta.people?.length ?? 0) > 0 ||
          (meta.topics?.length ?? 0) > 0 ||
          (meta.activities?.length ?? 0) > 0 ||
          (meta.emotions?.length ?? 0) > 0;
        if (!hasMeta) return null;
        return (
          <div className="space-y-1.5 text-xs text-slate-200">
            {meta.events && meta.events.length > 0 && (
              <div>
                <span className="font-semibold text-slate-300">Events:</span>
                <ul className="list-disc list-inside mt-0.5 space-y-0.5">
                  {meta.events.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              </div>
            )}
            <div className="flex flex-wrap gap-1.5">
              {meta.people?.map((p) => (
                <span key={p} className="px-1.5 py-0.5 rounded bg-violet-900/40 text-violet-200">
                  {p}
                </span>
              ))}
              {meta.activities?.map((a) => (
                <span key={a} className="px-1.5 py-0.5 rounded bg-sky-900/40 text-sky-200">
                  {a}
                </span>
              ))}
              {meta.topics?.map((t) => (
                <span key={t} className="px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-300">
                  {t}
                </span>
              ))}
              {meta.emotions?.map((em) => (
                <span key={em} className="px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-200">
                  {em}
                </span>
              ))}
            </div>
          </div>
        );
      })()}
    </div>
  );

  const renderFactBlock = (f: MemoryFact) => (
    <div
      key={f.id}
      className="rounded-lg bg-slate-800/50 border border-slate-700/50 p-2 flex items-start justify-between gap-2"
    >
      <button
        type="button"
        className="text-left text-sm text-slate-200 flex-1 min-w-0"
        onClick={() => openEditFact(f)}
      >
        <span className="line-clamp-2">{f.document}</span>
      </button>
      <button
        type="button"
        onClick={() => deleteFact(f.id)}
        className="text-red-400 hover:text-red-300 text-xs font-medium shrink-0"
        aria-label="Delete fact"
      >
        Delete
      </button>
    </div>
  );

  return (
    <div className="flex-1 flex flex-col min-h-0 p-4 md:p-6 overflow-auto">
      <h2 className="text-lg font-medium text-slate-300 uppercase tracking-wider mb-4 flex-shrink-0">
        Brain
      </h2>
      <p className="text-sm text-slate-500 mb-4 flex-shrink-0">
        Inspect and correct what the system knows about you. Edits persist to the database.
      </p>

      {/* Stats */}
      <div className="flex flex-wrap items-center gap-4 mb-4 flex-shrink-0">
        <div className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4">
          <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
            Brain stats
          </h3>
          {stats !== null ? (
            <div className="space-y-1.5 text-sm text-slate-300">
              <p>
                Facts: <span className="font-medium text-violet-300">{stats.gist_facts_count}</span>
              </p>
              <p>
                Episodic summaries:{" "}
                <span className="font-medium text-violet-300">{stats.episodic_log_count}</span>
              </p>
              <p>
                With metadata:{" "}
                <span className="font-medium text-violet-300">
                  {stats.episodic_metadata_count ?? 0}
                </span>
              </p>
              <div className="flex flex-wrap gap-2 mt-2">
                <button
                  type="button"
                  onClick={onRefreshStats}
                  className="px-3 py-1.5 rounded-lg bg-slate-700/50 text-slate-400 text-xs font-medium hover:bg-slate-600/50"
                >
                  Refresh memory
                </button>
                {onWipeMemory && (
                  <button
                    type="button"
                    onClick={onWipeMemory}
                    disabled={isWipingMemory}
                    className="px-3 py-1.5 rounded-lg bg-red-900/40 text-red-300 text-xs font-medium hover:bg-red-800/50 disabled:opacity-50"
                  >
                    {isWipingMemory ? "Wiping…" : "Wipe memory"}
                  </button>
                )}
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500">Load stats from backend.</p>
          )}
        </div>
      </div>

      {/* Brain navigation cards */}
      <div className="mb-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {(
          [
            ["time", "Timeline", "Coming soon", true],
            ["person", "People", "See who appears in your memories.", false],
            ["topic", "Topics", "Coming soon", true],
            ["place", "Places", "Coming soon", true],
            ["facts", "Facts", "Coming soon", true],
            ["activity", "Activities", "Coming soon", true],
            ["emotion", "Emotions", "Coming soon", true],
          ] as const
        ).map(([key, title, desc, comingSoon]) => (
          <button
            key={key}
            type="button"
            onClick={() => {
              if (comingSoon) {
                onToast("Coming soon");
                return;
              }
              setViewTab(key === "facts" ? "time" : (key as Exclude<MemoryViewTab, "facts">));
            }}
            className={`text-left rounded-xl bg-slate-900/60 border px-4 py-3 transition-colors ${
              !comingSoon && (key === "facts" ? viewTab === "time" : viewTab === key)
                ? "border-violet-500/80 ring-1 ring-violet-500/60"
                : "border-slate-700/70 hover:border-slate-500/80"
            } ${comingSoon ? "opacity-80" : ""}`}
          >
            <div className="text-sm font-semibold text-slate-200 mb-1">{title}</div>
            <div className="text-xs text-slate-500">{desc}</div>
          </button>
        ))}
      </div>

      {loading && facts.length === 0 && summaries.length === 0 ? (
        <p className="text-slate-500 text-sm py-4">Loading memory…</p>
      ) : (
        <div className="space-y-6">
          {viewTab === "time" && (
            <div className="space-y-4">
              <p className="text-xs text-slate-500">Summaries grouped by date.</p>
              {summariesByTime.keys.length === 0 ? (
                <p className="text-slate-500 text-sm">No summaries yet.</p>
              ) : (
                summariesByTime.keys.map((key) => (
                  <div key={key}>
                    <h4 className="text-sm font-medium text-slate-400 mb-2">
                      {formatDate(key === "unknown" ? undefined : `${key}T12:00:00Z`)}
                    </h4>
                    {(() => {
                      const daySummaries = summariesByTime.map.get(key)!;
                      const agg: Metadata = { events: [], people: [], activities: [], topics: [], emotions: [] };
                      for (const s of daySummaries) {
                        const m = parseMetadata(s.metadata_json);
                        agg.events = [...new Set([...(agg.events ?? []), ...(m.events ?? [])])];
                        agg.people = [...new Set([...(agg.people ?? []), ...(m.people ?? [])])];
                        agg.activities = [...new Set([...(agg.activities ?? []), ...(m.activities ?? [])])];
                        agg.topics = [...new Set([...(agg.topics ?? []), ...(m.topics ?? [])])];
                        agg.emotions = [...new Set([...(agg.emotions ?? []), ...(m.emotions ?? [])])];
                      }
                      const dayFacts = factsByDate.get(key) ?? [];
                      const hasAnything =
                        (agg.events?.length ?? 0) > 0 ||
                        (agg.people?.length ?? 0) > 0 ||
                        (agg.activities?.length ?? 0) > 0 ||
                        (agg.topics?.length ?? 0) > 0 ||
                        (agg.emotions?.length ?? 0) > 0 ||
                        dayFacts.length > 0;
                      if (!hasAnything) return null;
                      return (
                        <div className="mb-2 rounded-lg bg-slate-900/60 border border-slate-700/60 p-3 text-xs text-slate-200 space-y-1.5">
                          {agg.events && agg.events.length > 0 && (
                            <div>
                              <span className="font-semibold text-slate-300">Events:</span>
                              <ul className="list-disc list-inside mt-0.5 space-y-0.5">
                                {agg.events.map((e) => (
                                  <li key={e}>{e}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {agg.people && agg.people.length > 0 && (
                            <div>
                              <span className="font-semibold text-slate-300">People:</span>{" "}
                              {agg.people.join(", ")}
                            </div>
                          )}
                          {agg.activities && agg.activities.length > 0 && (
                            <div>
                              <span className="font-semibold text-slate-300">Activities:</span>{" "}
                              {agg.activities.join(", ")}
                            </div>
                          )}
                          {agg.topics && agg.topics.length > 0 && (
                            <div>
                              <span className="font-semibold text-slate-300">Topics:</span>{" "}
                              {agg.topics.join(", ")}
                            </div>
                          )}
                          {agg.emotions && agg.emotions.length > 0 && (
                            <div>
                              <span className="font-semibold text-slate-300">Emotions:</span>{" "}
                              {agg.emotions.join(", ")}
                            </div>
                          )}
                          {dayFacts.length > 0 && (
                            <div>
                              <span className="font-semibold text-slate-300">Facts:</span>
                              <ul className="list-disc list-inside mt-0.5 space-y-0.5">
                                {dayFacts.map((f) => (
                                  <li key={f.id}>{f.document}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      );
                    })()}
                    <div className="space-y-2">
                      {summariesByTime.map.get(key)!.map(renderSummaryBlock)}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {viewTab === "person" && (
            <BrainPeopleView onToast={onToast} />
          )}

          {viewTab === "topic" && (
            <div className="space-y-4">
              <p className="text-xs text-slate-500">Summaries grouped by topic.</p>
              {summariesByTopic.keys.length === 0 ? (
                <p className="text-slate-500 text-sm">No summaries with topics yet.</p>
              ) : (
                summariesByTopic.keys.map((topic) => (
                  <div key={topic}>
                    <h4 className="text-sm font-medium text-slate-400 mb-2">{topic}</h4>
                    <div className="space-y-2">
                      {summariesByTopic.map.get(topic)!.map(renderSummaryBlock)}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {viewTab === "place" && (
            <p className="text-slate-500 text-sm py-4">
              By-place view is not implemented yet. You can add location metadata support later.
            </p>
          )}

          {viewTab === "activity" && (
            <div className="space-y-4">
              <p className="text-xs text-slate-500">Summaries grouped by activities.</p>
              {summariesByActivity.keys.length === 0 ? (
                <p className="text-slate-500 text-sm">No activities detected yet.</p>
              ) : (
                summariesByActivity.keys.map((activity) => (
                  <div key={activity}>
                    <h4 className="text-sm font-medium text-sky-300 mb-2">{activity}</h4>
                    <div className="space-y-2">
                      {summariesByActivity.map.get(activity)!.map(renderSummaryBlock)}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {viewTab === "emotion" && (
            <div className="space-y-4">
              <p className="text-xs text-slate-500">
                Summaries grouped by emotions. Edit individual entries to adjust events and tags.
              </p>
              {summariesByEmotion.keys.length === 0 ? (
                <p className="text-slate-500 text-sm">No emotions stored yet.</p>
              ) : (
                summariesByEmotion.keys.map((emotionKey) => (
                  <div key={emotionKey}>
                    <h4 className="text-sm font-medium text-amber-300 mb-2">{emotionKey}</h4>
                    <div className="space-y-2">
                      {summariesByEmotion.map.get(emotionKey)!.map(renderSummaryBlock)}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Facts section (all tabs) */}
          <section className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4">
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
              Facts (long-term)
            </h4>
            <div className="space-y-2 max-h-48 overflow-y-auto mb-3">
              {facts.length === 0 ? (
                <p className="text-slate-500 text-xs">No facts yet. Add one below.</p>
              ) : (
                facts.map(renderFactBlock)
              )}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={addFactText}
                onChange={(e) => setAddFactText(e.target.value)}
                placeholder="Add a fact…"
                className="flex-1 min-w-0 px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
              />
              <button
                type="button"
                disabled={addSaving || !addFactText.trim()}
                onClick={addFact}
                className="px-3 py-1.5 rounded-lg bg-emerald-600/90 text-white text-sm font-medium hover:bg-emerald-500 disabled:opacity-50"
              >
                {addSaving ? "Adding…" : "Add fact"}
              </button>
            </div>
          </section>

          {/* Add summary */}
          <section className="rounded-xl bg-slate-900/50 border border-slate-700/50 p-4">
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
              Add session summary
            </h4>
            <div className="flex gap-2">
              <input
                type="text"
                value={addSummaryText}
                onChange={(e) => setAddSummaryText(e.target.value)}
                placeholder="Add a summary…"
                className="flex-1 min-w-0 px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
              />
              <button
                type="button"
                disabled={addSaving || !addSummaryText.trim()}
                onClick={addSummary}
                className="px-3 py-1.5 rounded-lg bg-emerald-600/90 text-white text-sm font-medium hover:bg-emerald-500 disabled:opacity-50"
              >
                {addSaving ? "Adding…" : "Add summary"}
              </button>
            </div>
          </section>
        </div>
      )}

      {/* Edit fact modal */}
      {editingFactId !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => {
              setEditingFactId(null);
              setEditText("");
            }}
          />
          <div className="relative bg-slate-900 border border-slate-700 rounded-xl shadow-xl max-w-lg w-full p-4">
            <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Edit fact</h4>
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              rows={3}
              className="w-full px-3 py-2 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50 resize-y mb-4"
            />
            <div className="flex gap-2">
              <button
                type="button"
                disabled={saving || !editText.trim()}
                onClick={() => saveFact(editingFactId)}
                className="px-3 py-1.5 rounded-lg bg-violet-600 text-white text-sm font-medium hover:bg-violet-500 disabled:opacity-50"
              >
                {saving ? "Saving…" : "Save"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setEditingFactId(null);
                  setEditText("");
                }}
                className="px-3 py-1.5 rounded-lg bg-slate-700/60 text-slate-400 text-sm font-medium"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit summary modal (with metadata) */}
      {editingSummaryId !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => {
              setEditingSummaryId(null);
              setEditText("");
              setEditMetadata({});
            }}
          />
          <div className="relative bg-slate-900 border border-slate-700 rounded-xl shadow-xl max-w-lg w-full max-h-[90vh] overflow-y-auto p-4">
            <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">
              Edit summary & metadata
            </h4>
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              rows={4}
              className="w-full px-3 py-2 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50 resize-y mb-4"
              placeholder="Summary text"
            />
            <div className="space-y-3 mb-4">
              <label className="block text-xs font-medium text-slate-500">People (comma-separated)</label>
              <input
                type="text"
                value={(editMetadata.people ?? []).join(", ")}
                onChange={(e) =>
                  setEditMetadata((m) => ({
                    ...m,
                    people: e.target.value.split(",").map((x) => x.trim()).filter(Boolean),
                  }))
                }
                className="w-full px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm"
                placeholder="Sarah, Mom"
              />
              <label className="block text-xs font-medium text-slate-500">Topics (comma-separated)</label>
              <input
                type="text"
                value={(editMetadata.topics ?? []).join(", ")}
                onChange={(e) =>
                  setEditMetadata((m) => ({
                    ...m,
                    topics: e.target.value.split(",").map((x) => x.trim()).filter(Boolean),
                  }))
                }
                className="w-full px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm"
                placeholder="Career, Health"
              />
              <div className="flex gap-4 flex-wrap">
                <div>
                  <label className="block text-xs font-medium text-slate-500">Mood (-5 to +5)</label>
                  <input
                    type="number"
                    min={-5}
                    max={5}
                    value={editMetadata.mood ?? ""}
                    onChange={(e) => {
                      const v = e.target.value === "" ? null : parseInt(e.target.value, 10);
                      setEditMetadata((m) => ({ ...m, mood: Number.isNaN(v) ? null : v }));
                    }}
                    className="w-20 px-2 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-500">Energy (1 to 5)</label>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    value={editMetadata.energy ?? ""}
                    onChange={(e) => {
                      const v = e.target.value === "" ? null : parseInt(e.target.value, 10);
                      setEditMetadata((m) => ({ ...m, energy: Number.isNaN(v) ? null : v }));
                    }}
                    className="w-20 px-2 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm"
                  />
                </div>
              </div>
              <label className="block text-xs font-medium text-slate-500">Activities (comma-separated)</label>
              <input
                type="text"
                value={(editMetadata.activities ?? []).join(", ")}
                onChange={(e) =>
                  setEditMetadata((m) => ({
                    ...m,
                    activities: e.target.value.split(",").map((x) => x.trim()).filter(Boolean),
                  }))
                }
                className="w-full px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-sm"
                placeholder="Workout, Dinner"
              />
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={saving || !editText.trim()}
                onClick={() => saveSummary(editingSummaryId)}
                className="px-3 py-1.5 rounded-lg bg-violet-600 text-white text-sm font-medium hover:bg-violet-500 disabled:opacity-50"
              >
                {saving ? "Saving…" : "Save"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setEditingSummaryId(null);
                  setEditText("");
                  setEditMetadata({});
                }}
                className="px-3 py-1.5 rounded-lg bg-slate-700/60 text-slate-400 text-sm font-medium"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
