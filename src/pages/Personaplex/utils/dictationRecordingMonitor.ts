/** Min recorded duration before we consider sending audio to transcription. */
export const MIN_DICTATION_DURATION_MS = 1000;

/**
 * Peak RMS during the session must exceed this to count as speech (Web Audio time-domain, 0–1 scale).
 * Slightly below barge-in threshold so quiet dictation still registers.
 */
export const DICTATION_SPEECH_RMS_THRESHOLD = 0.014;

function rmsFromTimeDomain(data: Uint8Array): number {
  if (data.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i]! - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / data.length);
}

export type DictationLevelMonitor = {
  /** Stop sampling and release audio nodes (call once). */
  stop: () => void;
  /** Peak RMS observed since start. */
  getMaxRms: () => number;
};

/**
 * Prefer echo cancellation / noise suppression for dictation (same as voice session).
 */
export const DICTATION_MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

/**
 * Track input level while recording so we can discard near-silence without calling the API.
 * Uses a **cloned** audio track so Web Audio does not share the same MediaStreamTrack as
 * MediaRecorder — on some browsers tapping the track for analysis caused silent or corrupt
 * recordings and STT that echoed on-screen assistant text (e.g. from speaker bleed).
 */
export function attachDictationLevelMonitor(stream: MediaStream): DictationLevelMonitor {
  let maxRms = 0;
  let raf = 0;
  let closed = false;

  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioCtx) {
    return {
      stop: () => {
        closed = true;
      },
      getMaxRms: () => 1,
    };
  }

  const track = stream.getAudioTracks()[0];
  const monitorStream = track ? new MediaStream([track.clone()]) : stream;

  const ctx = new AudioCtx();
  void ctx.resume().catch(() => {});
  const src = ctx.createMediaStreamSource(monitorStream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.3;
  src.connect(analyser);
  const buf = new Uint8Array(analyser.frequencyBinCount);

  const tick = () => {
    if (closed) return;
    analyser.getByteTimeDomainData(buf);
    const rms = rmsFromTimeDomain(buf);
    if (rms > maxRms) maxRms = rms;
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  return {
    stop() {
      if (closed) return;
      closed = true;
      cancelAnimationFrame(raf);
      try {
        src.disconnect();
        analyser.disconnect();
      } catch {
        /* ignore */
      }
      monitorStream.getTracks().forEach((t) => {
        try {
          t.stop();
        } catch {
          /* ignore */
        }
      });
      void ctx.close().catch(() => {});
    },
    getMaxRms() {
      return maxRms;
    },
  };
}

export function shouldDiscardDictationRecording(
  elapsedMs: number,
  peakRms: number,
): boolean {
  if (elapsedMs < MIN_DICTATION_DURATION_MS) return true;
  if (peakRms < DICTATION_SPEECH_RMS_THRESHOLD) return true;
  return false;
}
