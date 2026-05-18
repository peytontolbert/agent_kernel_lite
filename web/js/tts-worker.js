import { decodeWavMono, vocosMelFromMono } from '../vendor/model-stack-bitnet/audio_mel_runtime.js';
import { F5TTSQ4DiTRuntime } from '../vendor/model-stack-bitnet/f5tts_q4_dit_runtime.js';
import { Q4TensorBundleWASM } from '../vendor/model-stack-bitnet/q4_wasm_runtime.js';
import { SAMPLE_RATE, VocosMel24khzRuntime } from '../vendor/model-stack-bitnet/vocos_fp16_runtime.js';

let runtimePromise = null;

const RUNTIME_VERSION = '20260517-peyton-q4-wasm-v20';
const SPEAK_PRESET = 'custom-wasm-cond64-duration-cfg2-step8';
const REFERENCE_TEXT = "Hi, I'm recording this sample to create a digital copy of my voice. I want it to sound natural and conversational, just like how I normally speak.";
const REFERENCE_MEL_FRAMES = 938;
const REFERENCE_TEXT_BYTES = new TextEncoder().encode(REFERENCE_TEXT).length;
const MAX_DURATION_FRAMES = 384;

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
      postMessage({ type: 'status', detail: 'Loading Peyton Q4 F5TTS' });
      const f5Bundle = await Q4TensorBundleWASM.fromManifestUrl(versionedUrl(DEFAULTS.f5Manifest));
      postMessage({ type: 'status', detail: 'Loading Vocos Q4' });
      const vocosBundle = await Q4TensorBundleWASM.fromManifestUrl(versionedUrl(DEFAULTS.vocosManifest));
      const [refAudioBuffer, vocabText] = await Promise.all([
        fetchArrayBuffer(versionedUrl(DEFAULTS.refWav)),
        fetchText(versionedUrl(DEFAULTS.vocab)),
      ]);
      const wav = decodeWavMono(refAudioBuffer);
      const vocabMap = buildVocabMap(vocabText);
      const f5Id = f5Bundle.manifest?.model_id || 'f5tts-peyton-q4';
      const vocosId = vocosBundle.manifest?.model_id || 'vocos-q4';
      const f5 = new F5TTSQ4DiTRuntime(f5Bundle);
      postMessage({ type: 'status', detail: 'Preparing F5 WASM session' });
      const sessionStartedAt = performance.now();
      f5.prepareSession();
      postMessage({ type: 'status', detail: `F5 WASM session ready (${Math.round(performance.now() - sessionStartedAt)} ms)` });
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

async function speak(message) {
  const startedAt = performance.now();
  const runtime = await loadRuntime();
  const loadedAt = performance.now();
  const text = String(message.text || 'This is Peyton speaking from Agent Kernel Lite.').trim();
  const condSeqLen = clampInt(message.condSeqLen, 64, 2, 96);
  const steps = clampInt(message.steps, 8, 1, 32);
  const cfgStrength = clampNumber(message.cfgStrength, 2.0, 0, 4);
  const chunks = splitTextForSpeech(text);
  const explicitGenFrames = Number.isFinite(Number(message.genFrames)) ? Number(message.genFrames) : null;
  const preset = `custom-wasm-cond${condSeqLen}-${explicitGenFrames ? `gen${explicitGenFrames}` : 'duration'}-cfg${formatPresetNumber(cfgStrength)}-step${steps}`;

  postMessage({ type: 'status', detail: `Extracting Peyton reference mel (${preset})` });
  const { mel: condMel } = vocosMelFromMono(runtime.refSamples, runtime.vocosBundle, { maxFrames: condSeqLen });
  const conditionedAt = performance.now();
  const audioParts = [];
  let totalSamples = 0;
  let generationMs = 0;
  let decodeMs = 0;

  for (let index = 0; index < chunks.length; index += 1) {
    const chunk = chunks[index];
    const genFrames = clampInt(explicitGenFrames, estimateTargetFrames(chunk), 96, MAX_DURATION_FRAMES - condSeqLen);
    const duration = condSeqLen + genFrames;
    const textIds = tokenize(`${REFERENCE_TEXT} ${chunk}`, runtime.vocabMap, duration);

    postMessage({ type: 'status', detail: `Preparing WASM F5 session ${index + 1}/${chunks.length}` });
    postMessage({ type: 'status', detail: `Generating Q4 F5TTS mel ${index + 1}/${chunks.length} (${genFrames} target frames)` });
    const generateStartedAt = performance.now();
    const mel = runtime.f5.sampleMel({
      condMel,
      condSeqLen,
      textIds,
      duration,
      steps,
      cfgStrength,
    });
    generationMs += performance.now() - generateStartedAt;

    postMessage({ type: 'status', detail: `Decoding waveform ${index + 1}/${chunks.length}` });
    const targetMel = mel.subarray(condSeqLen * runtime.f5.melDim);
    const decodeStartedAt = performance.now();
    const audio = runtime.vocos.decode(targetMel);
    decodeMs += performance.now() - decodeStartedAt;
    audioParts.push(audio);
    totalSamples += audio.length;
  }

  const audio = concatFloat32(audioParts, totalSamples);
  const wav = encodeWav(audio, SAMPLE_RATE);
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

function versionedUrl(path) {
  const url = new URL(path, import.meta.url);
  url.searchParams.set('v', RUNTIME_VERSION);
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
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok && response.status !== 0) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return response.text();
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

function splitTextForSpeech(text) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return ['This is Peyton speaking from Agent Kernel Lite.'];
  const sentences = normalized.match(/[^.!?]+[.!?]*/g) || [normalized];
  const chunks = [];
  let current = '';
  const maxChars = maxChunkChars();
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

function estimateTargetFrames(text) {
  const bytes = new TextEncoder().encode(String(text || '').trim()).length;
  return Math.ceil((REFERENCE_MEL_FRAMES / REFERENCE_TEXT_BYTES) * bytes);
}

function maxChunkChars() {
  const maxGenFrames = MAX_DURATION_FRAMES - 64;
  return Math.max(24, Math.floor(maxGenFrames / (REFERENCE_MEL_FRAMES / REFERENCE_TEXT_BYTES)));
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
