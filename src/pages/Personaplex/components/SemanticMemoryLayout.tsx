import { FC, useCallback, useEffect, useMemo, useState } from "react";
import {
  apiSemanticConsumedDelete,
  apiSemanticConsumedList,
  type SemanticConsumedItem,
  type SemanticMemoryCategory,
} from "../../../backendApi";

type Props = {
  onToast?: (message: string) => void;
};

const kbAsideClass =
  "w-full lg:w-[min(100%,340px)] lg:flex-shrink-0 flex flex-col min-h-0 border-b lg:border-b-0 lg:border-r border-[rgba(120,180,200,0.12)] bg-[rgba(15,28,40,0.6)] backdrop-blur-md";
const kbExplorerHeaderClass =
  "border-b border-[rgba(120,180,200,0.1)] bg-[rgba(10,18,28,0.35)] px-4 py-3 backdrop-blur-sm";
const kbTabRailClass =
  "flex w-full gap-0.5 rounded-xl border border-[rgba(120,180,200,0.12)] bg-[rgba(20,37,52,0.8)] p-1 backdrop-blur-sm";
const kbTabSelectedClass = "bg-white text-gray-900 shadow-sm dark:bg-white/90";
const kbTabIdleClass = "text-[#9BB1BE] hover:bg-white/[0.06] hover:text-[#E8F1F5]";
const kbDetailShellClass =
  "flex-1 flex flex-col min-h-0 m-3 md:m-4 rounded-2xl border border-[rgba(120,180,200,0.14)] bg-[rgba(15,28,40,0.55)] backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.2)] overflow-hidden";
const kbMetaRowClass =
  "flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-[rgba(120,180,200,0.12)] bg-[rgba(12,22,32,0.35)] backdrop-blur-sm";
const kbTreeBorder = "border-[rgba(120,180,200,0.12)]";
const kbHoverRow = "hover:bg-white/[0.06]";
const kbEmptyListClass =
  "text-xs text-[#5F7585] py-1 pl-2 pr-1 border-l border-[rgba(45,212,191,0.25)]";

const categoryLabels: Record<SemanticMemoryCategory, string> = {
  book: "Books",
  podcast: "Podcasts",
  research_article: "Research Articles",
};

export const SemanticMemoryLayout: FC<Props> = ({ onToast }) => {
  const [bucketTab, setBucketTab] = useState<"media" | "relationships">("media");
  const [category, setCategory] = useState<SemanticMemoryCategory>("book");
  const [items, setItems] = useState<SemanticConsumedItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<SemanticMemoryCategory>>(
    () => new Set(["book", "podcast", "research_article"])
  );

  const selectedItem = useMemo(
    () => (selectedId == null ? null : items.find((i) => i.id === selectedId) ?? null),
    [items, selectedId]
  );

  const loadItems = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await apiSemanticConsumedList();
      setItems(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load semantic memory.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadItems();
  }, [loadItems]);

  const deleteItem = useCallback(
    async (id: number) => {
      if (!confirm("Delete this consumed item?")) return;
      const ok = await apiSemanticConsumedDelete(id);
      if (!ok) {
        onToast?.("Could not delete item.");
        return;
      }
      if (selectedId === id) setSelectedId(null);
      await loadItems();
      onToast?.("Deleted.");
    },
    [onToast, selectedId, loadItems]
  );

  return (
    <div className="flex flex-1 min-h-0 flex-col lg:flex-row gap-0 overflow-hidden">
      <aside className={kbAsideClass}>
        <div className={kbExplorerHeaderClass}>
          <p className="mb-2.5 text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-[#5F7585]">Explorer</p>
          <div className={kbTabRailClass}>
            {(
              [
                { key: "media", label: "Media" },
                { key: "relationships", label: "Relationships" },
              ] as const
            ).map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => setBucketTab(tab.key)}
                className={`flex-1 rounded-lg px-2 py-1.5 text-[0.7rem] font-medium leading-tight transition-colors ${
                  bucketTab === tab.key ? kbTabSelectedClass : kbTabIdleClass
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex-1 overflow-y-auto scrollbar p-2">
          {loading ? <p className="text-xs text-[#5F7585]">Loading…</p> : null}
          {bucketTab === "media" ? (
            <div className={`ml-1 border-l ${kbTreeBorder} pl-2`}>
              {(Object.keys(categoryLabels) as SemanticMemoryCategory[]).map((cat) => {
                const catItems = items.filter((i) => i.category === cat);
                const open = expandedCategories.has(cat);
                return (
                  <div key={cat} className="mb-1">
                    <button
                      type="button"
                      onClick={() => {
                        setCategory(cat);
                        setExpandedCategories((prev) => {
                          const next = new Set(prev);
                          if (next.has(cat)) next.delete(cat);
                          else next.add(cat);
                          return next;
                        });
                      }}
                      className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs font-semibold text-[#C5D4DE] ${kbHoverRow}`}
                    >
                      <svg
                        className={`h-3 w-3 shrink-0 text-[#5F7585] transition-transform ${open ? "rotate-90" : ""}`}
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                      <span className="flex-1">Consumed {categoryLabels[cat]}</span>
                      <span className="text-[0.65rem] font-normal text-[#5F7585]">{catItems.length}</span>
                    </button>
                    {open ? (
                      <div className={`ml-3 border-l ${kbTreeBorder} pl-2`}>
                        {catItems.length === 0 ? (
                          <p className={kbEmptyListClass}>No consumed items yet.</p>
                        ) : (
                          <ul className="space-y-0.5 pb-1">
                            {catItems.map((item) => (
                              <li key={item.id}>
                                <button
                                  type="button"
                                  onClick={() => {
                                    setCategory(cat);
                                    setSelectedId(item.id);
                                  }}
                                  className={`w-full rounded-lg px-2 py-1.5 text-left text-xs transition-colors ${
                                    selectedId === item.id
                                      ? "bg-white text-gray-900 shadow-sm dark:bg-white dark:text-gray-900"
                                      : "text-[#C5D4DE] hover:bg-white/[0.06] hover:text-[#E8F1F5]"
                                  }`}
                                >
                                  {item.title}
                                </button>
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className={`ml-1 border-l ${kbTreeBorder} pl-2`}>
              <p className={kbEmptyListClass}>No relationship memory items yet.</p>
            </div>
          )}
        </div>
      </aside>

      <section className="flex-1 flex flex-col min-h-0 min-w-0 bg-transparent">
        <div className={kbDetailShellClass}>
          <div className={kbMetaRowClass}>
            <span className="text-xs font-semibold uppercase tracking-widest text-[#5F7585]">Semantic Memory</span>
            <span className="text-xs text-[#9BB1BE]">
              {bucketTab === "media" ? `${categoryLabels[category]}` : "Relationships"}
            </span>
          </div>
          <div className="flex-1 overflow-y-auto scrollbar px-6 py-8 md:px-10 md:py-10">
            <div className="mx-auto max-w-3xl space-y-4">
              {bucketTab === "relationships" ? (
                <div className={`rounded-xl border ${kbTreeBorder} bg-[rgba(10,18,28,0.5)] p-4`}>
                  <p className="text-sm text-[#9BB1BE]">
                    Relationships bucket is ready. We can wire relationship memories here next.
                  </p>
                </div>
              ) : null}

              {error ? <p className="text-sm text-[#F43F5E]">{error}</p> : null}

              {bucketTab === "media" ? (
                selectedItem ? (
                  <div className={`rounded-xl border ${kbTreeBorder} bg-[rgba(10,18,28,0.5)] p-4`}>
                    <p className="text-sm font-semibold text-[#E8F1F5]">{selectedItem.title}</p>
                    <p className="mt-1 text-xs text-[#9BB1BE]">
                      {selectedItem.creator_or_source || "No creator/source"} · {selectedItem.consumed_on || "No consumed date"}
                    </p>
                    {selectedItem.notes ? <p className="mt-3 whitespace-pre-wrap text-sm text-[#C5D4DE]">{selectedItem.notes}</p> : null}
                    <div className="mt-3 flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => void deleteItem(selectedItem.id)}
                        className="rounded-full border border-[rgba(244,63,94,0.4)] bg-[rgba(244,63,94,0.12)] px-3 py-1.5 text-xs font-semibold text-[#F43F5E]"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-[#9BB1BE]">Select an item from the explorer.</p>
                )
              ) : (
                <p className="text-sm text-[#9BB1BE]">Select an item from the explorer.</p>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
