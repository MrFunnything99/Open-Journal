import type { TranscriptEntry } from "../hooks/usePersonaplexSession";

export function TranscriptBubble({
  entry,
  isLogExpanded,
  onToggleLog,
}: {
  entry: TranscriptEntry;
  isLogExpanded: boolean;
  onToggleLog: () => void;
}) {
  const isUser = entry.role === "user";
  const hasLog = !isUser && entry.retrievalLog;
  return (
    <div className={`text-sm flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] break-words ${
          isUser ? "text-gray-900 text-right dark:text-gray-100" : "text-gray-700 text-left dark:text-gray-300"
        }`}
      >
        <span className="font-medium opacity-80 block mb-0.5">{isUser ? "You" : "AI"}</span>
        {entry.text}
        {hasLog && (
          <div className="mt-2 text-left">
            <button
              type="button"
              onClick={onToggleLog}
              className="text-xs text-[#10a37f] hover:underline font-medium dark:text-emerald-400"
            >
              {isLogExpanded ? "Hide" : "Show"} memory context (vector DB)
            </button>
            {isLogExpanded && (
              <pre className="mt-1.5 p-2 rounded bg-gray-100 text-gray-600 text-xs whitespace-pre-wrap break-words border border-gray-200 max-h-48 overflow-y-auto dark:bg-[#343541] dark:text-gray-400 dark:border-gray-600">
                {entry.retrievalLog}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
