import { decodeWavMono, vocosMelFromMono } from '../vendor/model-stack-bitnet/audio_mel_runtime.js';
import { F5TTSQ4DiTRuntime } from '../vendor/model-stack-bitnet/f5tts_q4_dit_runtime.js';
import { Q4TensorBundleWASM } from '../vendor/model-stack-bitnet/q4_wasm_runtime.js';
import { SAMPLE_RATE, VocosMel24khzRuntime } from '../vendor/model-stack-bitnet/vocos_fp16_runtime.js';

let runtimePromise = null;

const DEFAULTS = {
  f5Manifest: '../models/f5tts_peyton_q4_v0/manifest.json',
  vocosManifest: '../models/vocos_mel_24khz_q4_v0/manifest.json',
  refWav: '../voice/peyton/sample_0.wav',
  vocab: '../voice/peyton/F5TTS_Base_vocab.txt',
};

self.addEventListener('message', (event) => {
  const message = event.data || {};
  if (message.type === 'load') {
    loadRuntime().then(() => {
      postMessage({ type: 'ready' });
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
      const f5Bundle = await Q4TensorBundleWASM.fromManifestUrl(new URL(DEFAULTS.f5Manifest, import.meta.url).href);
      postMessage({ type: 'status', detail: 'Loading Vocos' });
      const vocosBundle = await Q4TensorBundleWASM.fromManifestUrl(new URL(DEFAULTS.vocosManifest, import.meta.url).href);
      const [refAudioBuffer, vocabText] = await Promise.all([
        fetchArrayBuffer(new URL(DEFAULTS.refWav, import.meta.url).href),
        fetchText(new URL(DEFAULTS.vocab, import.meta.url).href),
      ]);
      const wav = decodeWavMono(refAudioBuffer);
      const vocabMap = buildVocabMap(vocabText);
      return {
        f5: new F5TTSQ4DiTRuntime(f5Bundle),
        vocos: new VocosMel24khzRuntime(vocosBundle),
        vocosBundle,
        refSamples: wav.samples,
        vocabMap,
      };
    })();
  }
  return runtimePromise;
}

async function speak(message) {
  const runtime = await loadRuntime();
  const text = String(message.text || 'This is Peyton speaking from Agent Kernel Lite.').trim();
  const condSeqLen = clampInt(message.condSeqLen, 4, 2, 24);
  const genFrames = clampInt(message.genFrames, 8, 2, 40);
  const steps = clampInt(message.steps, 1, 1, 4);
  const duration = condSeqLen + genFrames;

  postMessage({ type: 'status', detail: 'Extracting Peyton reference mel' });
  const { mel: condMel } = vocosMelFromMono(runtime.refSamples, runtime.vocosBundle, { maxFrames: condSeqLen });
  const textIds = tokenize(text, runtime.vocabMap, duration);

  postMessage({ type: 'status', detail: 'Generating Q4 F5TTS mel' });
  const mel = runtime.f5.sampleMel({
    condMel,
    condSeqLen,
    textIds,
    duration,
    steps,
    cfgStrength: 0.0,
  });

  postMessage({ type: 'status', detail: 'Decoding waveform' });
  const audio = runtime.vocos.decode(mel);
  const wav = encodeWav(audio, SAMPLE_RATE);
  postMessage({
    type: 'audio',
    text,
    sampleRate: SAMPLE_RATE,
    samples: audio.length,
    bytes: wav.byteLength,
    wav,
  }, [wav]);
}

async function fetchArrayBuffer(url) {
  try {
    const response = await fetch(url);
    if (response.ok || response.status === 0) return response.arrayBuffer();
    throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    return xhrArrayBuffer(url).catch(() => {
      throw new Error(`failed to fetch ${url}: ${error.message || String(error)}`);
    });
  }
}

async function fetchText(url) {
  const response = await fetch(url);
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

function clampInt(value, fallback, min, max) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
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
