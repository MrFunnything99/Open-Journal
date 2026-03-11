import { useCallback, useEffect, useMemo, useState } from "react";
import ForceGraph2D, { NodeObject, LinkObject } from "react-force-graph-2d";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

type GraphNode = {
  id: string;
  label: string;
  type: "person" | "group";
};

type GraphLink = {
  source: string;
  target: string;
};

type PersonEvent = {
  summary_id: number;
  timestamp: string;
  events: string[];
};

type PersonThought = {
  id: number;
  date: string;
  thought_text: string;
};

type PersonFact = {
  id: number;
  fact_text: string;
  confidence?: number | null;
  source_journal_id?: string;
  created_at?: string;
};

type PersonDetail = {
  id: number;
  name: string;
  ai_relationship_summary: string;
  groups: string[];
  events: PersonEvent[];
  facts: PersonFact[];
  thoughts: PersonThought[];
};

type BrainPeopleViewProps = {
  onToast: (msg: string) => void;
};

export function BrainPeopleView({ onToast }: BrainPeopleViewProps) {
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [graphLinks, setGraphLinks] = useState<GraphLink[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null);
  const [detail, setDetail] = useState<PersonDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<{ name: string; members: string[] } | null>(
    null
  );

  const [newThoughtText, setNewThoughtText] = useState("");
  const [newThoughtDate, setNewThoughtDate] = useState("");
  const [savingThought, setSavingThought] = useState(false);
  const [autoGrouping, setAutoGrouping] = useState(false);

  const loadGraph = useCallback(() => {
    setLoading(true);
    fetch(`${BACKEND_URL}/brain/people-graph`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
      .then((data: { nodes: GraphNode[]; links: GraphLink[] }) => {
        setGraphNodes(data.nodes);
        setGraphLinks(data.links);
      })
      .catch(() => {
        setGraphNodes([]);
        setGraphLinks([]);
      })
      .finally(() => setLoading(false));
  }, []);

  const loadDetail = useCallback(
    (personId: number) => {
      setDetailLoading(true);
      fetch(`${BACKEND_URL}/brain/people/${personId}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
        .then((d: PersonDetail) => {
          setDetail(d);
          setSelectedPersonId(d.id);
        })
        .catch(() => {
          setDetail(null);
        })
        .finally(() => setDetailLoading(false));
    },
    []
  );

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  const graphData = useMemo(
    () => ({
      nodes: graphNodes,
      links: graphLinks,
    }),
    [graphNodes, graphLinks]
  );

  const handleNodeClick = useCallback(
    (node: NodeObject) => {
      const n = node as GraphNode;
      if (n.type === "person") {
        setSelectedGroup(null);
        const idPart = String(n.id).replace("person:", "");
        const pid = Number.parseInt(idPart, 10);
        if (!Number.isNaN(pid)) {
          loadDetail(pid);
        }
      } else if (n.type === "group") {
        // Highlight / list members of this group in the side panel
        setDetail(null);
        setSelectedPersonId(null);
        const groupId = String(n.id);
        const memberIds = graphLinks
          .filter((l) => l.source === groupId)
          .map((l) => String(l.target));
        const memberNames = graphNodes
          .filter((gn) => gn.type === "person" && memberIds.includes(String(gn.id)))
          .map((gn) => gn.label);
        setSelectedGroup({ name: n.label, members: memberNames });
      }
    },
    [graphLinks, graphNodes, loadDetail]
  );

  const handleAddThought = useCallback(() => {
    if (!detail || selectedPersonId === null) return;
    if (!newThoughtText.trim()) return;
    setSavingThought(true);
    fetch(`${BACKEND_URL}/brain/people/${selectedPersonId}/thoughts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        date: newThoughtDate || null,
        thought_text: newThoughtText.trim(),
      }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
      .then(() => {
        setNewThoughtText("");
        setNewThoughtDate("");
        loadDetail(selectedPersonId);
        onToast("Thought added.");
      })
      .catch(() => onToast("Failed to add thought."))
      .finally(() => setSavingThought(false));
  }, [detail, selectedPersonId, newThoughtDate, newThoughtText, onToast, loadDetail]);

  const handleDeleteThought = useCallback(
    (thoughtId: number) => {
      if (selectedPersonId === null) return;
      if (!window.confirm("Delete this thought?")) return;
      fetch(`${BACKEND_URL}/brain/people/${selectedPersonId}/thoughts/${thoughtId}`, {
        method: "DELETE",
      })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
        .then(() => {
          loadDetail(selectedPersonId);
          onToast("Thought deleted.");
        })
        .catch(() => onToast("Failed to delete thought."));
    },
    [selectedPersonId, onToast, loadDetail]
  );

  const handleAutoGroup = useCallback(() => {
    setAutoGrouping(true);
    fetch(`${BACKEND_URL}/brain/people/auto-groups`, {
      method: "POST",
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
      .then(() => {
        onToast("People auto-organized into groups.");
        loadGraph();
      })
      .catch(() => onToast("Failed to auto-organize people into groups."))
      .finally(() => setAutoGrouping(false));
  }, [loadGraph, onToast]);

  return (
    <div className="flex flex-col lg:flex-row gap-4 flex-1 min-h-0">
      <div className="flex-1 min-h-[260px] rounded-xl bg-slate-900/50 border border-slate-700/50 overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700/70">
          <div>
            <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              People graph
            </h3>
            <p className="text-[11px] text-slate-500">
              Explore your social world. Click a person to inspect details.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={loading}
              onClick={loadGraph}
              className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-xs font-medium hover:bg-slate-700 disabled:opacity-50"
            >
              {loading ? "Refreshing…" : "Refresh graph"}
            </button>
            <button
              type="button"
              disabled={autoGrouping}
              onClick={handleAutoGroup}
              className="px-3 py-1.5 rounded-lg bg-violet-700 text-slate-50 text-xs font-medium hover:bg-violet-600 disabled:opacity-50"
            >
              {autoGrouping ? "Organizing…" : "Auto-organize groups"}
            </button>
          </div>
        </div>
        <div className="w-full h-[260px] sm:h-[320px] lg:h-full">
          <ForceGraph2D
            graphData={graphData}
            nodeLabel={(node) => (node as GraphNode).label}
            nodeAutoColorBy={(node) => (node as GraphNode).type}
            onNodeClick={handleNodeClick}
          />
        </div>
      </div>
      <div className="w-full lg:w-96 xl:w-[420px] flex-shrink-0 rounded-xl bg-slate-900/50 border border-slate-700/50 p-4 flex flex-col min-h-0">
        <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
          {selectedGroup ? "Group details" : "Person details"}
        </h3>
        {detailLoading && !selectedGroup && <p className="text-xs text-slate-500">Loading…</p>}
        {!detailLoading && !detail && !selectedGroup && (
          <p className="text-xs text-slate-500">
            Click a person node in the graph to inspect their details, or a group to see its members.
          </p>
        )}
        {selectedGroup && !detail && (
          <div className="text-xs text-slate-200 space-y-2">
            <div className="font-semibold text-slate-300">{selectedGroup.name}</div>
            {selectedGroup.members.length === 0 ? (
              <p className="text-slate-500">
                No people are assigned to this group yet. Edit a person and add this group to their
                groups list.
              </p>
            ) : (
              <div>
                <p className="text-slate-400 mb-1">Members:</p>
                <ul className="list-disc list-inside space-y-0.5">
                  {selectedGroup.members.map((m) => (
                    <li key={m}>{m}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
        {detail && (
          <div className="flex flex-col gap-3 overflow-y-auto text-sm text-slate-200">
            {/* Panel 2: AI relationship summary */}
            <div className="border border-slate-700/70 rounded-lg p-2.5">
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="text-xs font-semibold text-slate-300">
                  1. AI relationship summary
                </span>
                <span className="text-[10px] text-slate-500">
                  Generated from journal memories about this person.
                </span>
              </div>
              {detail.ai_relationship_summary ? (
                <p className="text-[11px] text-slate-200">
                  {detail.ai_relationship_summary}
                </p>
              ) : (
                <p className="text-[11px] text-slate-500">
                  No AI summary yet. As you journal more about this person, the system will infer a brief
                  description of how you tend to feel about them.
                </p>
              )}
            </div>

            {/* Panel 3: Memories */}
            <div className="border border-slate-700/70 rounded-lg p-2.5">
              <h4 className="text-xs font-semibold text-slate-300 mb-1">
                2. Memories with this person
              </h4>
              {detail.events.length === 0 ? (
                <p className="text-[11px] text-slate-500">
                  No events detected yet. As you journal, the system will connect sessions that mention this
                  person.
                </p>
              ) : (
                <ul className="space-y-1.5 text-[11px] text-slate-300 max-h-40 overflow-y-auto">
                  {detail.events.map((ev) => (
                    <li key={`${ev.summary_id}-${ev.timestamp}`}>
                      <span className="font-semibold text-slate-400">
                        {ev.timestamp || "Unknown date"}:
                      </span>{" "}
                      {ev.events.join("; ")}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* Panel 4: Facts */}
            <div className="border border-slate-700/70 rounded-lg p-2.5">
              <h4 className="text-xs font-semibold text-slate-300 mb-1">
                3. Facts about this person
              </h4>
              {detail.facts.length === 0 ? (
                <p className="text-[11px] text-slate-500">
                  As you journal more, the system will collect stable factual details (school, work, hobbies,
                  projects) about this person.
                </p>
              ) : (
                <ul className="space-y-1 text-[11px] text-slate-200 max-h-32 overflow-y-auto">
                  {detail.facts.map((f) => (
                    <li key={f.id} className="flex items-start gap-2">
                      <span>•</span>
                      <div>
                        <span>{f.fact_text}</span>
                        {typeof f.confidence === "number" && (
                          <span className="ml-1 text-[10px] text-slate-500">
                            ({Math.round(f.confidence * 100)}%)
                          </span>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* Thoughts (user reflections) */}
            <div className="mt-1">
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">
                Thoughts about this person
              </h4>
              {detail.thoughts.length === 0 ? (
                <p className="text-[11px] text-slate-500 mb-2">
                  Capture reflections, worries, or insights you have about this person.
                </p>
              ) : (
                <ul className="space-y-2 mb-2 text-[11px] text-slate-200">
                  {detail.thoughts.map((t) => (
                    <li
                      key={t.id}
                      className="border border-slate-700/70 rounded-md px-2 py-1.5 flex items-start gap-2"
                    >
                      <div className="flex-1">
                        {t.date && (
                          <div className="text-[10px] text-slate-500 mb-0.5">
                            {t.date}
                          </div>
                        )}
                        <div>{t.thought_text}</div>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleDeleteThought(t.id)}
                        className="text-[10px] text-red-400 hover:text-red-300 font-medium"
                      >
                        Delete
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <div className="space-y-1.5">
                <input
                  type="date"
                  value={newThoughtDate}
                  onChange={(e) => setNewThoughtDate(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-[11px]"
                />
                <textarea
                  value={newThoughtText}
                  onChange={(e) => setNewThoughtText(e.target.value)}
                  rows={2}
                  className="w-full px-2 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/70 text-slate-200 text-[11px] resize-y"
                  placeholder="Add a new thought or reflection…"
                />
                <button
                  type="button"
                  disabled={savingThought || !newThoughtText.trim()}
                  onClick={handleAddThought}
                  className="px-3 py-1.5 rounded-lg bg-emerald-600/90 text-white text-xs font-medium hover:bg-emerald-500 disabled:opacity-50 self-start"
                >
                  {savingThought ? "Adding…" : "Add thought"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

