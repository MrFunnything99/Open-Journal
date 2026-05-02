/**
 * Converts a MediaRecorder blob (audio/webm or audio/mp4) to WAV format
 * for Tinfoil Whisper, which accepts mp3 and wav.
 * Only use for reasonably short recordings/files that the browser can decode.
 */
const TRANSCRIPTION_WAV_SAMPLE_RATE = 16_000;

export async function blobToWavBase64(blob: Blob): Promise<string> {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioContextCtor =
    window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const audioContext = new AudioContextCtor();

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const transcriptionBuffer = await audioBufferToTranscriptionBuffer(audioContext, audioBuffer);
    const wavBuffer = audioBufferToWav(transcriptionBuffer);
    return arrayBufferToBase64(wavBuffer);
  } finally {
    void audioContext.close();
  }
}

/** Base64-encode a blob without format conversion (keeps mp3/m4a/etc. small). */
export async function blobToBase64(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  return arrayBufferToBase64(buf);
}

/**
 * Prepare a MediaRecorder blob for /voice-memo transcription.
 * Tinfoil Whisper accepts mp3 and wav, so other browser-decodable audio is
 * converted to WAV before upload.
 */
export async function micBlobToTranscriptionPayload(blob: Blob): Promise<{
  b64: string;
  filename: string;
  mimeType: string;
}> {
  const rawMime = (blob.type || "").trim().toLowerCase() || "audio/webm";

  const needsWavConversion = !(rawMime.includes("wav") || rawMime.includes("mpeg") || rawMime.includes("mp3"));

  if (needsWavConversion) {
    const b64 = await blobToWavBase64(blob);
    return { b64, filename: "dictation.wav", mimeType: "audio/wav" };
  }

  const b64 = await blobToBase64(blob);
  let filename = "dictation.wav";
  if (rawMime.includes("wav")) {
    filename = "dictation.wav";
  } else if (rawMime.includes("mpeg") || rawMime.includes("mp3")) {
    filename = "dictation.mp3";
  }
  return { b64, filename, mimeType: rawMime };
}

function audioBufferToWav(audioBuffer: AudioBuffer): ArrayBuffer {
  const numChannels = 1; // Mono for transcription
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;

  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = audioBuffer.length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  let offset = 0;

  function writeString(str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset++, str.charCodeAt(i));
    }
  }

  writeString("RIFF");
  view.setUint32(offset, 36 + dataSize, true);
  offset += 4;
  writeString("WAVE");
  writeString("fmt ");
  view.setUint32(offset, 16, true);
  offset += 4;
  view.setUint16(offset, format, true);
  offset += 2;
  view.setUint16(offset, numChannels, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, byteRate, true);
  offset += 4;
  view.setUint16(offset, blockAlign, true);
  offset += 2;
  view.setUint16(offset, bitDepth, true);
  offset += 2;
  writeString("data");
  view.setUint32(offset, dataSize, true);
  offset += 4;

  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : null;

  for (let i = 0; i < audioBuffer.length; i++) {
    const mixed = right ? (left[i]! + right[i]!) / 2 : left[i]!;
    const s = Math.max(-1, Math.min(1, mixed));
    const pcm = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, pcm, true);
    offset += 2;
  }

  return buffer;
}

async function audioBufferToTranscriptionBuffer(
  audioContext: BaseAudioContext,
  audioBuffer: AudioBuffer,
): Promise<AudioBuffer> {
  const monoBuffer = downmixToMono(audioContext, audioBuffer);
  if (monoBuffer.sampleRate === TRANSCRIPTION_WAV_SAMPLE_RATE) {
    return monoBuffer;
  }

  try {
    const length = Math.max(1, Math.ceil(monoBuffer.duration * TRANSCRIPTION_WAV_SAMPLE_RATE));
    const offlineContext = new OfflineAudioContext(1, length, TRANSCRIPTION_WAV_SAMPLE_RATE);
    const source = offlineContext.createBufferSource();
    source.buffer = monoBuffer;
    source.connect(offlineContext.destination);
    source.start(0);
    return await offlineContext.startRendering();
  } catch {
    // If browser resampling is unavailable, still return a valid mono WAV.
    return monoBuffer;
  }
}

function downmixToMono(audioContext: BaseAudioContext, audioBuffer: AudioBuffer): AudioBuffer {
  if (audioBuffer.numberOfChannels === 1) {
    return audioBuffer;
  }

  const monoBuffer = audioContext.createBuffer(1, audioBuffer.length, audioBuffer.sampleRate);
  const monoData = monoBuffer.getChannelData(0);
  for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
    const input = audioBuffer.getChannelData(ch);
    for (let i = 0; i < input.length; i++) {
      monoData[i] = (monoData[i] ?? 0) + input[i]! / audioBuffer.numberOfChannels;
    }
  }
  return monoBuffer;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}
