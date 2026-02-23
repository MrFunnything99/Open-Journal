import { FC, useCallback, useState } from "react";
import type { JournalEntry } from "../hooks/useJournalHistory";

type JournalGalleryProps = {
  entries: JournalEntry[];
  onDeleteEntry: (id: string) => void;
  getFormattedDate: (entry: JournalEntry) => string;
  onToast?: (message: string) => void;
};

async function fetchReformattedEntry(messages: { role: "user" | "ai"; text: string }[]): Promise<string> {
  const res = await fetch("/api/reformat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Failed to reformat");
  return data.text;
}

export const JournalGallery: FC<JournalGalleryProps> = ({
  entries,
  onDeleteEntry,
  getFormattedDate,
  onToast,
}) => {
  const [selectedEntry, setSelectedEntry] = useState<JournalEntry | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [reformattedModal, setReformattedModal] = useState<{ text: string; entry: JournalEntry } | null>(null);

  const openModal = useCallback((entry: JournalEntry) => {
    setSelectedEntry(entry);
    setReformattedModal(null);
  }, []);

  const closeModal = useCallback(() => {
    setSelectedEntry(null);
    setReformattedModal(null);
  }, []);

  const handleAiReformat = useCallback(
    async (entry: JournalEntry) => {
      if (entry.fullTranscript.length === 0) return;
      setIsGenerating(true);
      try {
        const text = await fetchReformattedEntry(entry.fullTranscript);
        setReformattedModal({ text, entry });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "AI reformatting failed";
        onToast?.(msg);
      } finally {
        setIsGenerating(false);
      }
    },
    [onToast]
  );

  const copyReformattedToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      onToast?.("Copied to clipboard");
    } catch {
      onToast?.("Failed to copy");
    }
  }, [onToast]);

  const downloadReformatted = useCallback(
    (text: string, entry: JournalEntry) => {
      const dateStr = new Date(entry.date).toISOString().slice(0, 10);
      const filename = `Journal_Entry_Reformatted_${dateStr}.txt`;
      const blob = new Blob([text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    },
    []
  );

  const exportToFile = useCallback((entry: JournalEntry) => {
    const dateStr = new Date(entry.date).toISOString().slice(0, 10);
    const filename = `Journal_Entry_${dateStr}.txt`;
    const lines = [
      getFormattedDate(entry),
      "",
      ...entry.fullTranscript.map((msg) =>
        msg.role === "user" ? `You: ${msg.text}` : `AI: ${msg.text}`
      ),
    ];
    const content = lines.join("\n");
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, [getFormattedDate]);

  if (entries.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <p className="text-slate-500 text-center">
          No journal entries yet.
          <br />
          <span className="text-sm">Connect and have a conversation to save entries.</span>
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5 p-6 overflow-y-auto">
        {entries.map((entry) => (
          <div
            key={entry.id}
            className="group relative flex flex-col rounded-xl bg-slate-900/60 border border-slate-700/50 overflow-hidden hover:border-slate-600/60 transition-all cursor-pointer min-h-[220px] max-h-[280px]"
            onClick={() => openModal(entry)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && openModal(entry)}
          >
            {/* Delete button - top-right corner */}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteEntry(entry.id);
              }}
              className="absolute top-3 right-3 z-10 p-2 rounded-lg bg-slate-800/90 text-slate-400 hover:text-red-400 hover:bg-red-500/20 transition-colors focus:outline-none focus:ring-2 focus:ring-red-500/50"
              aria-label="Delete entry"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>

            {/* Date header */}
            <div className="px-5 py-3 border-b border-slate-700/50 flex-shrink-0">
              <p className="text-sm font-medium text-slate-300 tracking-wide">
                {getFormattedDate(entry)}
              </p>
            </div>

            {/* Preview with fade-out */}
            <div className="relative flex-1 min-h-0 px-5 py-4 overflow-hidden">
              <p className="text-sm text-slate-300 leading-relaxed max-h-[140px] overflow-hidden">
                {entry.preview || "No preview"}
              </p>
              <div
                className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-slate-900/60 to-transparent pointer-events-none"
                aria-hidden
              />
            </div>

            {/* Read hint */}
            <div className="px-5 py-3 flex-shrink-0 border-t border-slate-700/50">
              <span className="text-xs text-violet-400/80 font-medium">
                Click to read full transcript
              </span>
            </div>
          </div>
        ))}
      </div>

      {selectedEntry && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md"
          onClick={closeModal}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Escape" && closeModal()}
          aria-label="Close modal"
        >
          <div
            className="bg-slate-900/95 bg-gradient-to-b from-slate-900 to-slate-900/90 border border-slate-700/80 rounded-2xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-6 py-4 border-b border-slate-700/80 flex justify-between items-center bg-slate-800/30">
              <h3 className="text-lg font-medium text-slate-200 tracking-wide">
                {getFormattedDate(selectedEntry)}
              </h3>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => exportToFile(selectedEntry)}
                  className="px-3 py-2 rounded-lg bg-slate-700/50 text-slate-300 text-sm font-medium hover:bg-slate-600/50 transition-colors flex items-center gap-2"
                  title="Download as .txt"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    aria-hidden
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                  </svg>
                  Download
                </button>
                <button
                  type="button"
                  onClick={() => handleAiReformat(selectedEntry)}
                  disabled={selectedEntry.fullTranscript.length === 0 || isGenerating}
                  className="px-3 py-2 rounded-lg border border-slate-600 text-slate-300 text-sm font-medium hover:bg-slate-600/50 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  title="AI Reformatted Download"
                >
                  {isGenerating ? (
                    <>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-4 w-4 animate-spin"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        aria-hidden
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                        />
                      </svg>
                      Generating...
                    </>
                  ) : (
                    <>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-4 w-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        aria-hidden
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"
                        />
                      </svg>
                      AI Reformatted Download
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={closeModal}
                  className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 transition-colors"
                  aria-label="Close"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-8 scrollbar">
              <div className="mx-auto max-w-2xl font-serif text-base leading-[1.7] space-y-5">
                {selectedEntry.fullTranscript.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${
                      msg.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[85%] ${
                        msg.role === "user"
                          ? "text-violet-200 text-right"
                          : "text-slate-200 text-left"
                      }`}
                    >
                      <span className="text-xs font-sans font-medium uppercase tracking-wider text-slate-500 block mb-1.5">
                        {msg.role === "user" ? "You" : "AI"}
                      </span>
                      <p className="whitespace-pre-wrap break-words">{msg.text}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {reformattedModal && (
        <div
          className="fixed inset-0 z-[110] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md"
          onClick={() => setReformattedModal(null)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Escape" && setReformattedModal(null)}
          aria-label="Close reformatted modal"
        >
          <div
            className="bg-slate-900/95 border border-slate-700/80 rounded-2xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-6 py-4 border-b border-slate-700/80 flex justify-between items-center bg-slate-800/30">
              <h3 className="text-lg font-medium text-slate-200 tracking-wide">
                AI Reformatted Journal Entry
              </h3>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => copyReformattedToClipboard(reformattedModal.text)}
                  className="px-3 py-2 rounded-lg bg-slate-700/50 text-slate-300 text-sm font-medium hover:bg-slate-600/50 transition-colors flex items-center gap-2"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                  Copy to Clipboard
                </button>
                <button
                  type="button"
                  onClick={() => downloadReformatted(reformattedModal.text, reformattedModal.entry)}
                  className="px-3 py-2 rounded-lg bg-slate-700/50 text-slate-300 text-sm font-medium hover:bg-slate-600/50 transition-colors flex items-center gap-2"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                  </svg>
                  Download
                </button>
                <button
                  type="button"
                  onClick={() => setReformattedModal(null)}
                  className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 transition-colors"
                  aria-label="Close"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-8 scrollbar">
              <p className="font-serif text-base leading-[1.7] whitespace-pre-wrap text-slate-200">
                {reformattedModal.text}
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
