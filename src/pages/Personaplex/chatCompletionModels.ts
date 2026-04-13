/** Single OpenRouter id for AI-Assisted Journal /chat and server allowlist (see backend graph.USER_SELECTABLE_CHAT_MODELS). */
export const DEFAULT_OPENROUTER_PRIMARY_MODEL = "anthropic/claude-opus-4.6" as const;

export type UserSelectableChatModelId = typeof DEFAULT_OPENROUTER_PRIMARY_MODEL;

export const DEFAULT_USER_CHAT_MODEL = DEFAULT_OPENROUTER_PRIMARY_MODEL;

export function isUserSelectableChatModelId(v: string): v is UserSelectableChatModelId {
  return v === DEFAULT_OPENROUTER_PRIMARY_MODEL;
}

export function readStoredUserChatModel(): UserSelectableChatModelId {
  return DEFAULT_USER_CHAT_MODEL;
}
