/**
 * Fallback when backend TTS (Mistral) is down or misconfigured.
 * Uses the browser's built-in speech synthesis — quality varies by OS/browser.
 */
function normalizeSpeechText(text: string): string {
  return text
    .replace(/[*_`#>|]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 12_000);
}

export type BrowserSpeechHandlers = {
  onEnd: () => void;
  onError: () => void;
};

/**
 * Start speaking; returns cancel to stop (e.g. skip / end session).
 * If synthesis is unavailable, calls onError synchronously.
 */
export function speakWithSpeechSynthesis(text: string, handlers: BrowserSpeechHandlers): { cancel: () => void } {
  const raw = normalizeSpeechText(text);
  if (typeof window === "undefined" || !window.speechSynthesis) {
    handlers.onError();
    return { cancel: () => {} };
  }
  if (!raw) {
    handlers.onEnd();
    return { cancel: () => {} };
  }

  try {
    window.speechSynthesis.cancel();
  } catch {
    /* ignore */
  }
  try {
    window.speechSynthesis.resume();
  } catch {
    /* ignore — some browsers throw if not paused */
  }

  const u = new SpeechSynthesisUtterance(raw);
  u.lang =
    typeof navigator !== "undefined" && navigator.language ? navigator.language : "en-US";
  u.rate = 1;

  let done = false;
  const finish = (fn: () => void) => {
    if (done) return;
    done = true;
    fn();
  };

  u.onend = () => finish(handlers.onEnd);
  u.onerror = () => finish(handlers.onError);

  window.speechSynthesis.speak(u);

  return {
    cancel: () => {
      try {
        window.speechSynthesis.cancel();
      } catch {
        /* ignore */
      }
      finish(handlers.onEnd);
    },
  };
}

export function browserSpeechSynthesisAvailable(): boolean {
  return typeof window !== "undefined" && typeof window.speechSynthesis !== "undefined";
}
