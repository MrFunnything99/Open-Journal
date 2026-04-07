/**
 * Converts a MediaRecorder blob (audio/webm or audio/mp4) to WAV format
 * for sending to Voxtral/OpenRouter which expects base64 WAV.
 * Only use for short mic recordings — uploaded files should use blobToBase64.
 */
export async function blobToWavBase64(blob: Blob): Promise<string> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();

  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  const wavBuffer = audioBufferToWav(audioBuffer);
  const base64 = arrayBufferToBase64(wavBuffer);

  audioContext.close();
  return base64;
}

/** Base64-encode a blob without format conversion (keeps mp3/m4a/etc. small). */
export async function blobToBase64(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  return arrayBufferToBase64(buf);
}

/**
 * Use MediaRecorder output as-is for /voice-memo (OpenRouter accepts webm/m4a/etc.).
 * Avoids decode→re-encode to WAV, which could degrade or mis-handle some captures.
 */
export async function micBlobToTranscriptionPayload(blob: Blob): Promise<{
  b64: string;
  filename: string;
  mimeType: string;
}> {
  const mimeType = (blob.type || "").trim() || "audio/webm";
  const b64 = await blobToBase64(blob);
  const mt = mimeType.toLowerCase();
  let filename = "dictation.webm";
  if (mt.includes("mp4") || mt.includes("m4a") || mt.includes("aac") || mt === "audio/mp4") {
    filename = "dictation.m4a";
  } else if (mt.includes("webm")) {
    filename = "dictation.webm";
  } else if (mt.includes("wav")) {
    filename = "dictation.wav";
  } else if (mt.includes("mpeg") || mt.includes("mp3")) {
    filename = "dictation.mp3";
  } else if (mt.includes("ogg") || mt.includes("opus")) {
    filename = "dictation.ogg";
  } else if (mt.includes("flac")) {
    filename = "dictation.flac";
  }
  return { b64, filename, mimeType };
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

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}
