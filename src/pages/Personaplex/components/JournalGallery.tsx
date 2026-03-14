import { FC, useCallback, useState } from "react";
import type { JournalEntry } from "../hooks/useJournalHistory";

type JournalGalleryProps = {
  entries: JournalEntry[];
  onDeleteEntry: (id: string) => void;
  getFormattedDate: (entry: JournalEntry) => string;
  onToast?: (message: string) => void;
};

export const JournalGallery: FC<JournalGalleryProps> = ({
  entries,
  onDeleteEntry,
  getFormattedDate,
  onToast,
}) => {
  const [selectedEntry, setSelectedEntry] = useState<JournalEntry | null>(null);
  const [expandedLogIndex, setExpandedLogIndex] = useState<number | null>(null);
  const [reformattedModal, setReformattedModal] = useState<{ text: string; entry: JournalEntry } | null>(null);
  const [thinkingLogsModal, setThinkingLogsModal] = useState<string | null>(null);

  const openModal = useCallback((entry: JournalEntry) => {
    setSelectedEntry(entry);
    setExpandedLogIndex(null);
    setReformattedModal(null);
    setThinkingLogsModal(null);
  }, []);

  const closeModal = useCallback(() => {
    setSelectedEntry(null);
    setReformattedModal(null);
    setThinkingLogsModal(null);
  }, []);

  const buildThinkingLogsContent = useCallback((entry: JournalEntry): string => {
    const lines: string[] = [
      `AI Thinking Logs — ${getFormattedDate(entry)}`,
      "=".repeat(50),
      "",
    ];
    let aiIndex = 0;
    entry.fullTranscript.forEach((msg) => {
      if (msg.role === "ai") {
        aiIndex += 1;
        lines.push(`--- AI Response ${aiIndex} ---`);
        lines.push("");
        lines.push("Reply:");
        lines.push(msg.text);
        lines.push("");
        if (msg.retrievalLog) {
          lines.push("Memory context from vector DB:");
          lines.push(msg.retrievalLog);
        } else {
          lines.push("(No memory context retrieved for this response.)");
        }
        lines.push("");
      }
    });
    if (aiIndex === 0) {
      lines.push("No AI responses in this entry.");
    }
    return lines.join("\n");
  }, [getFormattedDate]);

  const handleAiThinkingLogs = useCallback(
    (entry: JournalEntry) => {
      const content = buildThinkingLogsContent(entry);
      const dateStr = new Date(entry.date).toISOString().slice(0, 10);
      const filename = `AI_Thinking_Logs_${dateStr}.txt`;
      const blob = new Blob([content], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      setThinkingLogsModal(content);
      onToast?.("Downloaded. Viewing logs.");
    },
    [buildThinkingLogsContent, onToast]
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

  const entriesByDate = [...entries].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );

  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5 px-6 pt-6 pb-2 overflow-y-auto">
        {entriesByDate.map((entry) => (
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
                  title="Download transcript as .txt"
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
                  onClick={() => handleAiThinkingLogs(selectedEntry)}
                  className="px-3 py-2 rounded-lg bg-slate-700/50 text-slate-300 text-sm font-medium hover:bg-slate-600/50 transition-colors flex items-center gap-2"
                  title="Download and view AI thinking logs (memory context)"
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
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  AI thinking logs
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
                      {msg.role === "ai" && msg.retrievalLog && (
                        <div className="mt-2 text-left">
                          <button
                            type="button"
                            onClick={() => setExpandedLogIndex((prev) => (prev === i ? null : i))}
                            className="text-xs text-violet-400/90 hover:text-violet-300 font-medium"
                          >
                            {expandedLogIndex === i ? "Hide" : "Show"} memory context (vector DB)
                          </button>
                          {expandedLogIndex === i && (
                            <pre className="mt-1.5 p-2 rounded bg-slate-800/80 text-slate-400 text-xs whitespace-pre-wrap break-words border border-slate-700/50 max-h-48 overflow-y-auto font-sans">
                              {msg.retrievalLog}
                            </pre>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {thinkingLogsModal !== null && (
        <div
          className="fixed inset-0 z-[110] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md"
          onClick={() => setThinkingLogsModal(null)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Escape" && setThinkingLogsModal(null)}
          aria-label="Close AI thinking logs"
        >
          <div
            className="bg-slate-900/95 border border-slate-700/80 rounded-2xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-6 py-4 border-b border-slate-700/80 flex justify-between items-center bg-slate-800/30">
              <h3 className="text-lg font-medium text-slate-200 tracking-wide">
                AI thinking logs
              </h3>
              <button
                type="button"
                onClick={() => setThinkingLogsModal(null)}
                className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 transition-colors"
                aria-label="Close"
              >
                ✕
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 scrollbar">
              <pre className="font-sans text-sm text-slate-300 whitespace-pre-wrap break-words">
                {thinkingLogsModal}
              </pre>
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
