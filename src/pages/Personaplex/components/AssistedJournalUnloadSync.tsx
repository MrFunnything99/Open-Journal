import { useEffect, useRef } from "react";
import { appendJournalEntrySync } from "../hooks/useJournalHistory";
import { personaplexChatToJournalTranscript, usePersonaplexChat } from "../PersonaplexChatContext";

/**
 * On tab/window close, persist in-flight Home /chat reflection (AI-Assisted Journal Mode, or residue after
 * switching to Manual Journal Mode).
 */
export function AssistedJournalUnloadSync() {
  const { messages } = usePersonaplexChat();
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  useEffect(() => {
    const onPageHide = () => {
      const msgs = messagesRef.current;
      if (msgs.length === 0) return;
      appendJournalEntrySync(personaplexChatToJournalTranscript(msgs), "conversation");
    };
    window.addEventListener("pagehide", onPageHide);
    return () => window.removeEventListener("pagehide", onPageHide);
  }, []);

  return null;
}
