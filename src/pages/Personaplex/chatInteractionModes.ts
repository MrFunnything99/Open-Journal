/** Modes the backend /chat accepts; home composer offers Manual Journal + AI-Assisted Journal (plus Learning on its tab). */
export type ChatInteractionMode = "journal" | "autobiography" | "learning";

/** Options in the + menu where it appears (e.g. Learning tab). */
export const CHAT_INTERACTION_MODES: ChatInteractionMode[] = ["journal", "autobiography"];

export const CHAT_INTERACTION_MODE_META: Record<
  ChatInteractionMode,
  { label: string; sublabel: string; description: string; /** Pill next to + in composer (ChatGPT-style tag). */ composerChipLabel: string }
> = {
  journal: {
    label: "Manual Journal Mode",
    sublabel: "Reflect",
    composerChipLabel: "Manual Journal Mode",
    description:
      "Your private writing space: type or dictate, optionally run AI Journal Cleanup, get feedback, and save. Switch to AI-Assisted Journal Mode when you want a reflective chat with the model.",
  },
  autobiography: {
    label: "AI-Assisted Journal Mode",
    sublabel: "Reflect with AI",
    composerChipLabel: "AI-Assisted Journal Mode",
    description:
      "Chat with the AI to reflect—on today, on memories from your manual journals, or on where to steer your writing. When you are ready, use Save to Journal to move the thread into Manual Journal Mode for editing and saving.",
  },
  learning: {
    label: "Learning",
    sublabel: "Reflect on today's article",
    composerChipLabel: "Learning",
    description:
      "Reflect on today's article and connect it to your own thinking through guided questions.",
  },
};
