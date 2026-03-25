import { FC } from "react";
import { backendFetch } from "../../../backendApi";
import type { JournalEntry } from "../hooks/useJournalHistory";

export type BrainCalendarPanelProps = {
  entries: JournalEntry[];
  calendarMonth: { year: number; month: number };
  setCalendarMonth: React.Dispatch<React.SetStateAction<{ year: number; month: number }>>;
  calendarSelectedDate: string | null;
  setCalendarSelectedDate: (v: string | null) => void;
  calendarDaySummary: string | null;
  setCalendarDaySummary: (v: string | null) => void;
  calendarDaySummaryLoading: boolean;
  setCalendarDaySummaryLoading: (v: boolean) => void;
};

export const BrainCalendarPanel: FC<BrainCalendarPanelProps> = ({
  entries,
  calendarMonth,
  setCalendarMonth,
  calendarSelectedDate,
  setCalendarSelectedDate,
  calendarDaySummary,
  setCalendarDaySummary,
  calendarDaySummaryLoading,
  setCalendarDaySummaryLoading,
}) => {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-4 md:p-6">
      <div className="flex min-h-0 flex-1 flex-col gap-6 sm:flex-row">
        <div className="shrink-0">
          <div className="mb-3 flex items-center justify-between gap-4">
            <button
              type="button"
              onClick={() =>
                setCalendarMonth((prev) => {
                  const d = new Date(prev.year, prev.month - 1, 1);
                  return { year: d.getFullYear(), month: d.getMonth() };
                })
              }
              className="rounded-lg bg-gray-100 p-2 text-gray-700 hover:bg-gray-200 dark:bg-[#404040] dark:text-gray-300 dark:hover:bg-[#505050]"
              aria-label="Previous month"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {new Date(calendarMonth.year, calendarMonth.month, 1).toLocaleString("default", { month: "long", year: "numeric" })}
            </span>
            <button
              type="button"
              onClick={() =>
                setCalendarMonth((prev) => {
                  const d = new Date(prev.year, prev.month + 1, 1);
                  return { year: d.getFullYear(), month: d.getMonth() };
                })
              }
              className="rounded-lg bg-gray-100 p-2 text-gray-700 hover:bg-gray-200 dark:bg-[#404040] dark:text-gray-300 dark:hover:bg-[#505050]"
              aria-label="Next month"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
          <div className="grid grid-cols-7 gap-1 text-center">
            {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((day) => (
              <div key={day} className="py-1 text-xs font-medium text-gray-500 dark:text-gray-400">
                {day}
              </div>
            ))}
            {(() => {
              const first = new Date(calendarMonth.year, calendarMonth.month, 1);
              const last = new Date(calendarMonth.year, calendarMonth.month + 1, 0);
              const startPad = first.getDay();
              const daysInMonth = last.getDate();
              const cells: (number | null)[] = [];
              for (let i = 0; i < startPad; i++) cells.push(null);
              for (let d = 1; d <= daysInMonth; d++) cells.push(d);
              const dateStr = (d: number) =>
                `${calendarMonth.year}-${String(calendarMonth.month + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
              const hasEntry = (d: number) => entries.some((e) => (e.date || "").startsWith(dateStr(d)));
              return cells.map((d, i) => (
                <div key={i}>
                  {d === null ? (
                    <div className="h-9 w-9" />
                  ) : (
                    <button
                      type="button"
                      onClick={() => {
                        const date = dateStr(d);
                        setCalendarSelectedDate(date);
                        setCalendarDaySummary(null);
                        setCalendarDaySummaryLoading(true);
                        const dayEntries = entries.filter((e) => (e.date || "").startsWith(date));
                        const rawTranscript = dayEntries.length
                          ? dayEntries.map((e) => e.fullTranscript.map((m) => `${m.role}: ${m.text}`).join("\n")).join("\n\n")
                          : "";
                        backendFetch("/calendar-day-summary", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ date, raw_transcript: rawTranscript }),
                        })
                          .then((r) => (r.ok ? r.json() : Promise.reject(new Error("Failed"))))
                          .then((data: { summary?: string }) => {
                            setCalendarDaySummary(data.summary ?? "");
                          })
                          .catch(() => setCalendarDaySummary("Could not load summary."))
                          .finally(() => setCalendarDaySummaryLoading(false));
                      }}
                      className={`h-9 w-9 rounded-lg text-sm font-medium transition-colors ${
                        calendarSelectedDate === dateStr(d)
                          ? "bg-[#10a37f] text-white"
                          : hasEntry(d)
                            ? "bg-gray-200 text-gray-900 hover:bg-gray-300 dark:bg-[#404040] dark:text-gray-100 dark:hover:bg-[#505050]"
                            : "text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-[#343541]"
                      }`}
                    >
                      {d}
                    </button>
                  )}
                </div>
              ));
            })()}
          </div>
        </div>
        <div className="flex min-h-0 min-w-0 flex-1 flex-col rounded-2xl border border-gray-100 bg-white shadow-sm dark:rounded-xl dark:border-gray-700 dark:bg-[#2f2f2f]">
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-4">
            {calendarSelectedDate ? (
              <>
                <h3 className="mb-2 font-medium text-gray-800 dark:text-gray-200">
                  {new Date(calendarSelectedDate + "T12:00:00").toLocaleDateString("default", {
                    weekday: "long",
                    month: "long",
                    day: "numeric",
                    year: "numeric",
                  })}
                </h3>
                {calendarDaySummaryLoading ? (
                  <p className="text-sm text-gray-500 dark:text-gray-400">Analyzing journal and memory…</p>
                ) : calendarDaySummary ? (
                  <p className="flex-1 overflow-y-auto whitespace-pre-wrap text-sm text-gray-900 dark:text-gray-100">{calendarDaySummary}</p>
                ) : null}
              </>
            ) : (
              <p className="text-sm text-gray-500 dark:text-gray-400">Click a date to see the day summary.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
