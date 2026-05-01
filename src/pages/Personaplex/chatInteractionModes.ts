/** Modes the backend /chat accepts; home composer offers Manual Journal + AI-Assisted Journal. */
export type ChatInteractionMode = "journal" | "autobiography";

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
      "Your private writing space: type or dictate, optionally run AI Spelling Correction/Reformatting, and save. Switch to AI-Assisted Journal Mode when you want a reflective chat with the model.",
  },
  autobiography: {
    label: "AI-Assisted Journal Mode",
    sublabel: "Reflect with AI",
    composerChipLabel: "AI-Assisted Journal Mode",
    description:
      "Reflective chat via Tinfoil — pick the model above the composer. Keep private details inside Selfmeridian at your discretion.",
  },
};
