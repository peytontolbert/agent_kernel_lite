import { decodeWavMono, vocosMelFromMono } from '../vendor/model-stack-bitnet/audio_mel_runtime.js';
import { Q4TensorBundleWASM } from '../vendor/model-stack-bitnet/q4_wasm_runtime.js';
import { SAMPLE_RATE, VocosMel24khzRuntime } from '../vendor/model-stack-bitnet/vocos_fp16_runtime.js';

const RUNTIME_VERSION = '20260517-peyton-native-coreml-v0';
const SEQ_LEN = 64;
const MEL_DIM = 100;
const COND_SEQ_LEN = 12;
const STEPS = 2;

export class NativePeytonTTSRuntime {
  constructor(plugin) {
    this.plugin = plugin;
    this.ready = false;
  }

  async load(onStatus = () => {}) {
    if (this.ready) return this;
    onStatus('Loading native Peyton Core ML int4 DiT');
    const status = await this.plugin.status();
    if (!status?.available) throw new Error('native Peyton Core ML model is not bundled');
    onStatus('Loading Q4 Vocos');
    const vocosBundle = await Q4TensorBundleWASM.fromManifestUrl(new URL('../models/vocos_mel_24khz_q4_v0/manifest.json', import.meta.url).href);
    const [refAudioBuffer, vocabText] = await Promise.all([
      fetchArrayBuffer(new URL('../voice/peyton/sample_0.wav', import.meta.url).href),
      fetchText(new URL('../voice/peyton/F5TTS_Base_vocab.txt', import.meta.url).href),
    ]);
    const wav = decodeWavMono(refAudioBuffer);
    const { mel: condMel } = vocosMelFromMono(wav.samples, vocosBundle, { maxFrames: COND_SEQ_LEN });
    this.vocos = new VocosMel24khzRuntime(vocosBundle);
    this.condMel = condMel;
    this.vocabMap = buildVocabMap(vocabText);
    this.ready = true;
    return this;
  }

  async speak(text, onStatus = () => {}) {
    await this.load(onStatus);
    const chunks = splitTextForSpeech(text);
    const parts = [];
    let totalSamples = 0;
    for (let i = 0; i < chunks.length; i += 1) {
      const chunk = chunks[i];
      onStatus(`Native F5TTS chunk ${i + 1}/${chunks.length}`);
      const mel = await this.sampleChunk(chunk, i + 1);
      onStatus(`Decoding native chunk ${i + 1}/${chunks.length}`);
      const audio = this.vocos.decode(mel);
      parts.push(audio);
      totalSamples += audio.length;
    }
    const audio = concatFloat32(parts, totalSamples);
    const wav = encodeWav(audio, SAMPLE_RATE);
    return {
      text,
      sampleRate: SAMPLE_RATE,
      samples: audio.length,
      bytes: wav.byteLength,
      preset: `native-coreml-seq${SEQ_LEN}-cond${COND_SEQ_LEN}-step${STEPS}`,
      runtimeVersion: RUNTIME_VERSION,
      chunks: chunks.length,
      wav,
    };
  }

  async sampleChunk(text, seed) {
    const cond = new Float32Array(SEQ_LEN * MEL_DIM);
    cond.set(this.condMel, 0);
    const textIds = tokenize(text, this.vocabMap, SEQ_LEN);
    let y = gaussianArray(SEQ_LEN * MEL_DIM, 1337 + seed);
    y.set(cond.subarray(0, COND_SEQ_LEN * MEL_DIM), 0);
    for (let step = 0; step < STEPS; step += 1) {
      const time = step / STEPS;
      const next = (step + 1) / STEPS;
      const response = await this.plugin.forward({
        x: Array.from(y),
        cond: Array.from(cond),
        text: Array.from(textIds),
        time,
      });
      const pred = response?.pred || [];
      if (pred.length !== y.length) throw new Error(`native DiT returned ${pred.length} floats, expected ${y.length}`);
      const dt = next - time;
      for (let i = 0; i < y.length; i += 1) y[i] += dt * Number(pred[i] || 0);
      y.set(cond.subarray(0, COND_SEQ_LEN * MEL_DIM), 0);
    }
    return y.subarray(COND_SEQ_LEN * MEL_DIM);
  }
}

export function nativePeytonPlugin() {
  return globalThis.Capacitor?.Plugins?.PeytonTTS || null;
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok && response.status !== 0) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return response.arrayBuffer();
}

async function fetchText(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok && response.status !== 0) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return response.text();
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
  const words = String(text || 'This is Peyton speaking from Agent Kernel Lite.').replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
  const chunks = [];
  let current = [];
  for (const word of words) {
    const candidate = [...current, word].join(' ');
    if (candidate.length <= 42) {
      current.push(word);
      continue;
    }
    if (current.length) chunks.push(current.join(' '));
    current = [word];
  }
  if (current.length) chunks.push(current.join(' '));
  return chunks.length ? chunks : ['This is Peyton speaking from Agent Kernel Lite.'];
}

function gaussianArray(length, seed) {
  const out = new Float32Array(length);
  let state = seed >>> 0;
  const random = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  for (let i = 0; i < length; i += 2) {
    const u1 = Math.max(random(), 1e-7);
    const u2 = random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    out[i] = radius * Math.cos(theta);
    if (i + 1 < length) out[i + 1] = radius * Math.sin(theta);
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
