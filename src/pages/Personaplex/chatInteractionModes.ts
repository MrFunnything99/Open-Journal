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
    label: "Assisted Journal",
    sublabel: "Guided prompts from chat",
    composerChipLabel: "Assisted Journal",
    description:
      "Chat-driven journaling: ask about today and get probing questions; ask for autobiographical reflection and explore a past moment already in your journals; or say you want to journal and get a short menu of directions to choose from.",
  },
  learning: {
    label: "Learning",
    sublabel: "Reflect on today's article",
    composerChipLabel: "Learning",
    description:
      "Reflect on today's article and connect it to your own thinking through guided questions.",
  },
};
