import { decodeWavMono, vocosMelFromMono } from '../vendor/model-stack-bitnet/audio_mel_runtime.js';
import { F5TTSQ4DiTRuntime } from '../vendor/model-stack-bitnet/f5tts_q4_dit_runtime.js';
import { Q4TensorBundleWASM } from '../vendor/model-stack-bitnet/q4_wasm_runtime.js';
import { SAMPLE_RATE, VocosMel24khzRuntime } from '../vendor/model-stack-bitnet/vocos_fp16_runtime.js';

let runtimePromise = null;

const RUNTIME_VERSION = '20260521-peyton-q4-fullq4-surface-v2';
const SPEAK_PRESET = 'custom-wasm-f5-fullq4-surface-v2-vocos-q4-dynamic-ref-duration-cfg2-step8-speed115-limited';
const REFERENCE_TEXT = "Hi, I'm recording this sample to create a ";
const FULL_REFERENCE_TEXT = "Hi, I'm recording this sample to create a digital copy of my voice. I want it to sound natural and conversational, just like how I normally speak.";
const REFERENCE_MEL_FRAMES = 256;
const FULL_REFERENCE_MEL_FRAMES = 938;
const MAX_DURATION_FRAMES = 1536;
const SHORT_TEXT_SPEED = 0.3;
const SPEECH_SPEED = 1.15;
const CROSS_FADE_SECONDS = 0.15;
const OUTPUT_PEAK = 0.82;

const DEFAULTS = {
  f5Manifest: '../models/f5tts_peyton_q4_v0/manifest.json',
  vocosManifest: '../models/vocos_mel_24khz_q4_v0/manifest.json',
  refWav: '../voice/peyton/sample_0.wav',
  vocab: '../voice/peyton/F5TTS_Base_vocab.txt',
};

self.addEventListener('message', (event) => {
  const message = event.data || {};
  if (message.type === 'load') {
    loadRuntime().then((runtime) => {
      postMessage({ type: 'ready', detail: runtime.detail, runtimeVersion: RUNTIME_VERSION });
    }).catch((error) => {
      postMessage({ type: 'error', error: error?.message || String(error) });
    });
  }
  if (message.type === 'speak') {
    speak(message).catch((error) => {
      postMessage({ type: 'error', error: error?.message || String(error) });
    });
  }
});

async function loadRuntime() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      postProgress({ phase: 'runtime', detail: 'Loading Peyton Q4 F5TTS weights', percent: 4 });
      const f5Bundle = await loadStage('Loading Peyton Q4 F5TTS', () => (
        Q4TensorBundleWASM.fromManifestUrl(versionedUrl(DEFAULTS.f5Manifest))
      ));
      postProgress({ phase: 'runtime', detail: 'Loading Vocos Q4 weights', percent: 9 });
      const vocosBundle = await loadStage('Loading Vocos Q4', () => (
        Q4TensorBundleWASM.fromManifestUrl(versionedUrl(DEFAULTS.vocosManifest))
      ));
      postProgress({ phase: 'runtime', detail: 'Loading Peyton reference audio and vocab', percent: 13 });
      const [refAudioBuffer, vocabText] = await loadStage('Loading Peyton reference assets', () => Promise.all([
        fetchArrayBuffer(versionedUrl(DEFAULTS.refWav)),
        fetchText(versionedUrl(DEFAULTS.vocab)),
      ]));
      const wav = decodeWavMono(refAudioBuffer);
      const vocabMap = buildVocabMap(vocabText);
      const f5Id = f5Bundle.manifest?.model_id || 'f5tts-peyton-q4';
      const vocosId = vocosBundle.manifest?.model_id || 'vocos-q4';
      const f5 = new F5TTSQ4DiTRuntime(f5Bundle);
      postMessage({ type: 'status', detail: 'Preparing F5 WASM session' });
      postProgress({ phase: 'runtime', detail: 'Preparing F5 WASM session', percent: 17 });
      const sessionStartedAt = performance.now();
      f5.prepareSession();
      postMessage({ type: 'status', detail: `F5 WASM session ready (${Math.round(performance.now() - sessionStartedAt)} ms)` });
      postProgress({ phase: 'runtime', detail: 'Peyton voice runtime ready', percent: 20 });
      return {
        f5,
        vocos: new VocosMel24khzRuntime(vocosBundle),
        vocosBundle,
        refSamples: wav.samples,
        vocabMap,
        detail: `${RUNTIME_VERSION} | ${f5Id} | ${vocosId} | ${SPEAK_PRESET}`,
      };
    })();
  }
  return runtimePromise;
}

async function loadStage(label, fn) {
  postMessage({ type: 'status', detail: label });
  try {
    return await fn();
  } catch (error) {
    throw new Error(`${label} failed: ${error?.message || String(error)}`);
  }
}

function postProgress(progress) {
  const percent = Number.isFinite(Number(progress.percent))
    ? Math.max(0, Math.min(100, Math.round(Number(progress.percent))))
    : 0;
  postMessage({
    type: 'progress',
    phase: progress.phase || 'generate',
    detail: progress.detail || 'Peyton voice working',
    percent,
    chunk: progress.chunk || 0,
    chunks: progress.chunks || 0,
    frames: progress.frames || 0,
    steps: progress.steps || 0,
    elapsedMs: Math.max(0, Math.round(progress.elapsedMs || 0)),
    etaMs: Number.isFinite(Number(progress.etaMs)) ? Math.max(0, Math.round(Number(progress.etaMs))) : 0,
  });
}

async function speak(message) {
  const startedAt = performance.now();
  postProgress({ phase: 'runtime', detail: 'Starting Peyton voice request', percent: 1 });
  const runtime = await loadRuntime();
  const loadedAt = performance.now();
  const text = String(message.text || 'This is Peyton speaking from Agent Kernel Lite.').trim();
  const condSeqLen = clampInt(message.condSeqLen, REFERENCE_MEL_FRAMES, 2, MAX_DURATION_FRAMES - 1);
  const steps = clampInt(message.steps, 8, 1, 32);
  const cfgStrength = clampNumber(message.cfgStrength, 2.0, 0, 4);
  const speechSpeed = clampNumber(message.speed, SPEECH_SPEED, 0.5, 2.0);
  const reference = referenceProfile(condSeqLen);
  const maxGenFrames = MAX_DURATION_FRAMES - condSeqLen;
  const chunks = splitTextForSpeech(text, reference.framesPerTextByte, maxGenFrames);
  const explicitGenFrames = Number.isFinite(Number(message.genFrames)) ? Number(message.genFrames) : null;
  const preset = `custom-wasm-cond${condSeqLen}-ref${reference.textBytes}-${explicitGenFrames ? `gen${explicitGenFrames}` : 'duration'}-cfg${formatPresetNumber(cfgStrength)}-step${steps}-speed${formatPresetNumber(speechSpeed)}`;

  postProgress({
    phase: 'condition',
    detail: `Extracting Peyton reference mel (${condSeqLen} frames)`,
    percent: 22,
    chunks: chunks.length,
    steps,
    elapsedMs: performance.now() - startedAt,
  });
  postMessage({ type: 'status', detail: `Extracting Peyton reference mel (${preset})` });
  const { mel: condMel } = vocosMelFromMono(runtime.refSamples, runtime.vocosBundle, { maxFrames: condSeqLen });
  const conditionedAt = performance.now();
  const audioParts = [];
  let totalSamples = 0;
  let generationMs = 0;
  let decodeMs = 0;
  const generationPlan = chunks.map((chunk) => clampInt(explicitGenFrames, estimateTargetFrames(chunk, reference.framesPerTextByte, speechSpeed), 96, maxGenFrames));
  const totalPlannedFrames = generationPlan.reduce((sum, frames) => sum + frames, 0);
  let completedFrames = 0;

  for (let index = 0; index < chunks.length; index += 1) {
    const chunk = chunks[index];
    const genFrames = generationPlan[index];
    const duration = condSeqLen + genFrames;
    const textIds = tokenize(`${reference.text}${chunk}`, runtime.vocabMap, duration);
    const chunkStartPercent = 28 + Math.round((completedFrames / Math.max(1, totalPlannedFrames)) * 58);
    const chunkGeneratePercent = Math.min(88, chunkStartPercent + Math.max(1, Math.round((genFrames / Math.max(1, totalPlannedFrames)) * 46)));
    const estimatedChunkMs = generationMs > 0 && completedFrames > 0
      ? (generationMs / completedFrames) * genFrames
      : 0;

    postMessage({ type: 'status', detail: `Preparing WASM F5 session ${index + 1}/${chunks.length}` });
    postProgress({
      phase: 'prepare',
      detail: `Preparing chunk ${index + 1}/${chunks.length} (${genFrames} frames)`,
      percent: chunkStartPercent,
      chunk: index + 1,
      chunks: chunks.length,
      frames: genFrames,
      steps,
      elapsedMs: performance.now() - startedAt,
      etaMs: estimatedChunkMs,
    });
    postMessage({ type: 'status', detail: `Generating Q4 F5TTS mel ${index + 1}/${chunks.length} (${genFrames} target frames)` });
    postProgress({
      phase: 'generate',
      detail: `Diffusion chunk ${index + 1}/${chunks.length}: ${steps} steps, ${genFrames} frames`,
      percent: chunkStartPercent,
      chunk: index + 1,
      chunks: chunks.length,
      frames: genFrames,
      steps,
      elapsedMs: performance.now() - startedAt,
      etaMs: estimatedChunkMs,
    });
    const generateStartedAt = performance.now();
    const mel = runtime.f5.sampleMel({
      condMel,
      condSeqLen,
      textIds,
      duration,
      steps,
      cfgStrength,
      onProgress: (step, total) => {
        const stepNumber = Math.max(0, Number(step) || 0);
        const totalSteps = Math.max(1, Number(total) || steps);
        const stepPercent = chunkStartPercent + Math.round((stepNumber / totalSteps) * Math.max(1, chunkGeneratePercent - chunkStartPercent));
        const stepElapsedMs = performance.now() - generateStartedAt;
        const etaMs = stepNumber > 0 ? (stepElapsedMs / stepNumber) * (totalSteps - stepNumber) : estimatedChunkMs;
        postProgress({
          phase: 'generate',
          detail: `Diffusion chunk ${index + 1}/${chunks.length}: step ${stepNumber}/${totalSteps}`,
          percent: Math.min(chunkGeneratePercent, stepPercent),
          chunk: index + 1,
          chunks: chunks.length,
          frames: genFrames,
          steps: totalSteps,
          elapsedMs: performance.now() - startedAt,
          etaMs,
        });
      },
    });
    generationMs += performance.now() - generateStartedAt;
    completedFrames += genFrames;
    postProgress({
      phase: 'generate',
      detail: `Generated mel chunk ${index + 1}/${chunks.length}`,
      percent: chunkGeneratePercent,
      chunk: index + 1,
      chunks: chunks.length,
      frames: completedFrames,
      steps,
      elapsedMs: performance.now() - startedAt,
    });

    postMessage({ type: 'status', detail: `Decoding waveform ${index + 1}/${chunks.length}` });
    postProgress({
      phase: 'decode',
      detail: `Decoding waveform chunk ${index + 1}/${chunks.length}`,
      percent: Math.min(94, chunkGeneratePercent + 2),
      chunk: index + 1,
      chunks: chunks.length,
      frames: genFrames,
      steps,
      elapsedMs: performance.now() - startedAt,
    });
    const targetMel = mel.subarray(condSeqLen * runtime.f5.melDim);
    const decodeStartedAt = performance.now();
    const audio = runtime.vocos.decode(targetMel);
    decodeMs += performance.now() - decodeStartedAt;
    audioParts.push(audio);
    totalSamples += audio.length;
  }

  postProgress({
    phase: 'render',
    detail: `Assembling ${chunks.length} audio chunk${chunks.length === 1 ? '' : 's'}`,
    percent: 96,
    chunks: chunks.length,
    steps,
    elapsedMs: performance.now() - startedAt,
  });
  const audio = normalizePeak(concatAudioParts(audioParts, totalSamples, SAMPLE_RATE, CROSS_FADE_SECONDS), OUTPUT_PEAK);
  const wav = encodeWav(audio, SAMPLE_RATE);
  postProgress({
    phase: 'render',
    detail: `Peyton voice ready (${formatDurationMs(performance.now() - startedAt)})`,
    percent: 100,
    chunks: chunks.length,
    steps,
    elapsedMs: performance.now() - startedAt,
  });
  postMessage({
    type: 'audio',
    text,
    sampleRate: SAMPLE_RATE,
    samples: audio.length,
    bytes: wav.byteLength,
    preset,
    runtimeVersion: message.runtimeVersion || RUNTIME_VERSION,
    chunks: chunks.length,
    timing: {
      loadMs: Math.round(loadedAt - startedAt),
      conditioningMs: Math.round(conditionedAt - loadedAt),
      generationMs: Math.round(generationMs),
      decodeMs: Math.round(decodeMs),
      totalMs: Math.round(performance.now() - startedAt),
    },
    wav,
  }, [wav]);
}

function formatDurationMs(ms) {
  const seconds = Math.max(0, Math.round(Number(ms || 0) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`;
}

function versionedUrl(path) {
  const url = new URL(path, import.meta.url);
  if (url.protocol === 'http:' || url.protocol === 'https:') {
    url.searchParams.set('v', RUNTIME_VERSION);
  }
  return url.href;
}

async function fetchArrayBuffer(url) {
  try {
    const response = await fetch(url, { cache: 'no-store' });
    if (response.ok || response.status === 0) return response.arrayBuffer();
    throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    return xhrArrayBuffer(url).catch(() => {
      throw new Error(`failed to fetch ${url}: ${error.message || String(error)}`);
    });
  }
}

async function fetchText(url) {
  try {
    const response = await fetch(url, { cache: 'no-store' });
    if (response.ok || response.status === 0) return response.text();
    throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    return xhrText(url).catch(() => {
      throw new Error(`failed to fetch ${url}: ${error.message || String(error)}`);
    });
  }
}

function xhrArrayBuffer(url) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = () => {
      if ((request.status >= 200 && request.status < 300) || request.status === 0) {
        resolve(request.response);
      } else {
        reject(new Error(`XHR ${request.status}`));
      }
    };
    request.onerror = () => reject(new Error('XHR network error'));
    request.send();
  });
}

function xhrText(url) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'text';
    request.onload = () => {
      if ((request.status >= 200 && request.status < 300) || request.status === 0) {
        resolve(request.responseText);
      } else {
        reject(new Error(`XHR ${request.status}`));
      }
    };
    request.onerror = () => reject(new Error('XHR network error'));
    request.send();
  });
}

function buildVocabMap(vocabText) {
  const map = new Map();
  vocabText.split(/\r?\n/).filter(Boolean).forEach((token, index) => map.set(token, index));
  return map;
}

function tokenize(text, vocabMap, maxLen) {
  const ids = new Int32Array(maxLen);
  ids.fill(-1);
  const chars = Array.from(text);
  for (let i = 0; i < Math.min(chars.length, maxLen); i += 1) {
    ids[i] = vocabMap.has(chars[i]) ? vocabMap.get(chars[i]) : -1;
  }
  return ids;
}

function referenceProfile(condSeqLen) {
  const text = referenceTextForFrames(condSeqLen);
  const textBytes = utf8Length(text);
  return {
    text,
    textBytes,
    framesPerTextByte: condSeqLen / Math.max(1, textBytes),
  };
}

function referenceTextForFrames(condSeqLen) {
  if (condSeqLen <= REFERENCE_MEL_FRAMES) return REFERENCE_TEXT;
  if (condSeqLen >= FULL_REFERENCE_MEL_FRAMES) return FULL_REFERENCE_TEXT;
  const targetBytes = Math.max(1, Math.round(utf8Length(FULL_REFERENCE_TEXT) * condSeqLen / FULL_REFERENCE_MEL_FRAMES));
  let out = '';
  for (const char of Array.from(FULL_REFERENCE_TEXT)) {
    if (utf8Length(out + char) > targetBytes) break;
    out += char;
  }
  return out.endsWith(' ') ? out : `${out} `;
}

function splitTextForSpeech(text, framesPerTextByte, maxGenFrames) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return ['This is Peyton speaking from Agent Kernel Lite.'];
  const sentences = normalized.match(/[^.!?]+[.!?]*/g) || [normalized];
  const chunks = [];
  let current = '';
  const maxChars = maxChunkChars(framesPerTextByte, maxGenFrames);
  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;
    if ((current.length + trimmed.length + 1) <= maxChars) {
      current = current ? `${current} ${trimmed}` : trimmed;
      continue;
    }
    if (current) chunks.push(current);
    if (trimmed.length <= maxChars) {
      current = trimmed;
    } else {
      for (let start = 0; start < trimmed.length; start += maxChars) chunks.push(trimmed.slice(start, start + maxChars).trim());
      current = '';
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

function estimateTargetFrames(text, framesPerTextByte, speechSpeed = 1) {
  const bytes = utf8Length(String(text || '').trim());
  const speed = bytes < 10 ? SHORT_TEXT_SPEED : 1;
  return Math.ceil(framesPerTextByte * bytes / speed / Math.max(0.5, speechSpeed));
}

function maxChunkChars(framesPerTextByte, maxGenFrames) {
  return Math.max(24, Math.floor(maxGenFrames / Math.max(1e-6, framesPerTextByte)));
}

function utf8Length(text) {
  return new TextEncoder().encode(String(text || '')).length;
}

function concatAudioParts(parts, totalLength, sampleRate, crossFadeSeconds) {
  if (parts.length <= 1 || crossFadeSeconds <= 0) return concatFloat32(parts, totalLength);
  const fadeSamples = Math.max(0, Math.floor(sampleRate * crossFadeSeconds));
  let out = parts[0].slice();
  for (let i = 1; i < parts.length; i += 1) {
    const next = parts[i];
    const overlap = Math.min(fadeSamples, out.length, next.length);
    if (overlap <= 0) {
      out = concatFloat32([out, next], out.length + next.length);
      continue;
    }
    const merged = new Float32Array(out.length + next.length - overlap);
    merged.set(out.subarray(0, out.length - overlap));
    const overlapStart = out.length - overlap;
    for (let j = 0; j < overlap; j += 1) {
      const fadeIn = j / Math.max(1, overlap - 1);
      const fadeOut = 1 - fadeIn;
      merged[overlapStart + j] = out[overlapStart + j] * fadeOut + next[j] * fadeIn;
    }
    merged.set(next.subarray(overlap), out.length);
    out = merged;
  }
  return out;
}

function concatFloat32(parts, totalLength) {
  const out = new Float32Array(totalLength);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

function normalizePeak(samples, targetPeak = OUTPUT_PEAK) {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const abs = Math.abs(samples[i]);
    if (abs > peak) peak = abs;
  }
  if (!(peak > targetPeak) || !Number.isFinite(peak)) return samples;
  const gain = targetPeak / peak;
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) out[i] = samples[i] * gain;
  return out;
}

function clampInt(value, fallback, min, max) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function clampNumber(value, fallback, min, max) {
  const parsed = Number.parseFloat(String(value ?? ''));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function formatPresetNumber(value) {
  return Number.isInteger(value) ? String(value) : String(value).replace(/0+$/, '').replace(/\.$/, '');
}

function encodeWav(samples, sampleRate) {
  const dataBytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataBytes, true);
  for (let i = 0; i < samples.length; i += 1) {
    const clipped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(clipped * 32767), true);
  }
  return buffer;
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
}
