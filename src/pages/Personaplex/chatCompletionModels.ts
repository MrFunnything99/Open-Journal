/** Tinfoil model ids for AI-Assisted Journal /chat (must match backend graph.USER_SELECTABLE_CHAT_MODELS). */
export const CHAT_COMPLETION_MODEL_OPTIONS = [
  { id: "kimi-k2-6", label: "Kimi K2.6" },
  { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
] as const;

export type UserSelectableChatModelId = (typeof CHAT_COMPLETION_MODEL_OPTIONS)[number]["id"];

export const DEFAULT_TINFOIL_PRIMARY_MODEL: UserSelectableChatModelId = "kimi-k2-6";

export const DEFAULT_USER_CHAT_MODEL = DEFAULT_TINFOIL_PRIMARY_MODEL;

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

/** Tinfoil model ids for Manual Journal: AI feedback + spelling/cleanup (must match backend journal LLM allowlist). */
export const JOURNAL_MANUAL_AI_MODEL_OPTIONS = [
  { id: "kimi-k2-6", label: "Kimi K2.6" },
  { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
] as const;

export type JournalManualAiModelId = (typeof JOURNAL_MANUAL_AI_MODEL_OPTIONS)[number]["id"];

export const DEFAULT_JOURNAL_MANUAL_AI_MODEL: JournalManualAiModelId = "kimi-k2-6";

export const JOURNAL_MANUAL_AI_MODEL_STORAGE_KEY = "selfmeridian:journal-manual-ai-model-v1";

export function isJournalManualAiModelId(v: string): v is JournalManualAiModelId {
  return JOURNAL_MANUAL_AI_MODEL_OPTIONS.some((o) => o.id === v);
}

export function readStoredJournalManualAiModel(): JournalManualAiModelId {
  try {
    const v = localStorage.getItem(JOURNAL_MANUAL_AI_MODEL_STORAGE_KEY);
    if (v && isJournalManualAiModelId(v)) return v;
  } catch {
    /* ignore */
  }
  return DEFAULT_JOURNAL_MANUAL_AI_MODEL;
}
