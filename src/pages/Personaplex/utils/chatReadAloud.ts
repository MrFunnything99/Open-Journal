let unavailableToastShown = false;

export type PlayChatReadAloudOptions = {
  onLoading?: (loading: boolean) => void;
};

export async function playChatReadAloud(
  text: string,
  onError: (message: string) => void,
  opts?: PlayChatReadAloudOptions,
): Promise<void> {
  opts?.onLoading?.(false);
  if (!text.trim()) {
    onError("Nothing to read.");
    return;
  }
  if (!unavailableToastShown) {
    unavailableToastShown = true;
    onError("Read aloud is unavailable in the Tinfoil-only build.");
  }
}

export function stopReadAloudPlayback() {
  /* no-op: TTS disabled */
}

export function isReadAloudPlaying(): boolean {
  return false;
}

export async function stopReadAloudAndCooldown(): Promise<void> {
  /* no-op: TTS disabled */
}
