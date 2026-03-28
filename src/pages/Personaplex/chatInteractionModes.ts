export type ChatInteractionMode = "conversation" | "journal" | "autobiography" | "learning";

export const CHAT_INTERACTION_MODES: ChatInteractionMode[] = ["conversation", "journal", "autobiography"];

export const CHAT_INTERACTION_MODE_META: Record<
  ChatInteractionMode,
  { label: string; sublabel: string; description: string; /** Pill next to + in composer (ChatGPT-style tag). */ composerChipLabel: string }
> = {
  conversation: {
    label: "Conversation",
    sublabel: "Chat freely",
    composerChipLabel: "Conversation",
    description:
      "Your default AI assistant. Chat freely, ask questions, and explore ideas—enhanced with your personal context.",
  },
  journal: {
    label: "Journal",
    sublabel: "Reflect",
    composerChipLabel: "Journaling",
    description:
      "A space for free-flow reflection. Write naturally, process your thoughts, and receive optional AI feedback or gentle structure.",
  },
  autobiography: {
    label: "Autobiography",
    sublabel: "Track your life",
    composerChipLabel: "Autobiography",
    description:
      "A structured way to track and understand your life. Reflect on your day, habits, goals, and past experiences through guided conversation.",
  },
  learning: {
    label: "Learning",
    sublabel: "Reflect on today's article",
    composerChipLabel: "Learning",
    description:
      "Reflect on today's article and connect it to your own thinking through guided questions.",
  },
};
