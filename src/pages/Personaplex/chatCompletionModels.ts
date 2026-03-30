/** OpenRouter ids allowlisted for Conversation + Assisted Journal (client + server). Journal mode ignores this. */
export const CHAT_COMPLETION_MODEL_OPTIONS = [
  { id: "openai/gpt-5.4", label: "GPT‑5.4" },
  { id: "anthropic/claude-sonnet-4.6", label: "Sonnet 4.6" },
  { id: "openai/gpt-5-nano", label: "GPT‑5 Nano" },
] as const;

export type UserSelectableChatModelId = (typeof CHAT_COMPLETION_MODEL_OPTIONS)[number]["id"];

export const DEFAULT_USER_CHAT_MODEL: UserSelectableChatModelId = "openai/gpt-5.4";

export const USER_CHAT_MODEL_STORAGE_KEY = "personaplex-user-chat-model-v1";

export function isUserSelectableChatModelId(v: string): v is UserSelectableChatModelId {
  return CHAT_COMPLETION_MODEL_OPTIONS.some((o) => o.id === v);
}

export function readStoredUserChatModel(): UserSelectableChatModelId {
  try {
    const v = localStorage.getItem(USER_CHAT_MODEL_STORAGE_KEY);
    if (v && isUserSelectableChatModelId(v)) return v;
  } catch {
    /* ignore */
  }
  return DEFAULT_USER_CHAT_MODEL;
}
