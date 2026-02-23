import { useCallback, useEffect, useState } from "react";

export type ChatMessage = { role: "user" | "ai"; text: string };

export type JournalEntry = {
  id: string;
  date: string;
  preview: string;
  fullTranscript: ChatMessage[];
};

const STORAGE_KEY = "openjournal-history";

function loadFromStorage(): JournalEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveToStorage(entries: JournalEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch (e) {
    console.warn("[Personaplex] Failed to save journal history:", e);
  }
}

function generateId(): string {
  return `entry-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function formatDate(isoDate: string): string {
  try {
    const d = new Date(isoDate);
    const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
    const day = d.getDate();
    const year = d.getFullYear();
    return `${month} ${day}, ${year}`;
  } catch {
    return isoDate;
  }
}

function transcriptToPreview(messages: ChatMessage[], maxLen = 100): string {
  const text = messages.map((m) => m.text).join(" ");
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trim() + "…";
}

export const useJournalHistory = () => {
  const [entries, setEntries] = useState<JournalEntry[]>([]);

  useEffect(() => {
    setEntries(loadFromStorage());
  }, []);

  useEffect(() => {
    saveToStorage(entries);
  }, [entries]);

  const saveEntry = useCallback((transcript: ChatMessage[]) => {
    if (transcript.length === 0) return;
    const id = generateId();
    const date = new Date().toISOString();
    const preview = transcriptToPreview(transcript);
    const entry: JournalEntry = {
      id,
      date,
      preview,
      fullTranscript: transcript,
    };
    setEntries((prev) => [entry, ...prev]);
    return id;
  }, []);

  const clearHistory = useCallback(() => {
    setEntries([]);
  }, []);

  const deleteEntry = useCallback((id: string) => {
    setEntries((prev) => prev.filter((e) => e.id !== id));
  }, []);

  const getFormattedDate = useCallback((entry: JournalEntry) => {
    return formatDate(entry.date);
  }, []);

  return {
    entries,
    saveEntry,
    clearHistory,
    deleteEntry,
    getFormattedDate,
  };
};
