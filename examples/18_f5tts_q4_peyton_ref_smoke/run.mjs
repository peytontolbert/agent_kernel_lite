import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { decodeWavMono, vocosMelFromMono } from "../../model-stack/browser/bitnet/audio_mel_runtime.js";
import { F5TTSQ4DiTRuntime } from "../../model-stack/browser/bitnet/f5tts_q4_dit_runtime.js";
import { FP16TensorBundle, SAMPLE_RATE, VocosMel24khzRuntime } from "../../model-stack/browser/bitnet/vocos_fp16_runtime.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const wasmPkg = path.join(repoRoot, "model-stack/browser/bitnet/pkg");
const defaultF5Bundle = "/data/resumebot/checkpoints/f5tts_peyton_q4_v0";
const defaultVocosBundle = "/data/resumebot/checkpoints/vocos_mel_24khz_q4_v0";
const defaultRefWav = "/data/resumebot/voice_profiles/Peyton/sample_0.wav";
const defaultVocab = "/data/resumebot/checkpoints/F5TTS_Base_vocab.txt";
const outDir = path.join(here, "out");

const { default: initWasm, q4_symmetric_linear_f32 } = await import(
  pathToFileURL(path.join(wasmPkg, "model_stack_bitnet_wasm.js")).href
);

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function alignedSlice(buffer, offset, nbytes, TypedArray) {
  const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset + offset, nbytes);
  const copy = new Uint8Array(nbytes);
  copy.set(bytes);
  return new TypedArray(copy.buffer);
}

function f16ToF32(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x03ff;
  if (exp === 0) return sign * (frac ? 2 ** -14 * (frac / 1024) : 0);
  if (exp === 0x1f) return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

class NodeQ4Bundle {
  constructor(bundleDir) {
    this.manifest = readJson(path.join(bundleDir, "manifest.json"));
    this.q4Index = readJson(path.join(bundleDir, this.manifest.files.q4_index));
    this.denseIndex = readJson(path.join(bundleDir, this.manifest.files.dense_index));
    this.q4Buffer = fs.readFileSync(path.join(bundleDir, this.manifest.files.q4));
    this.denseBuffer = fs.readFileSync(path.join(bundleDir, this.manifest.files.dense));
    this.denseCache = new Map();
  }

  q4Tensor(name) {
    const entry = this.q4Index[name];
    if (!entry) throw new Error(`Q4 tensor not found: ${name}`);
    return {
      entry,
      packedWeight: alignedSlice(this.q4Buffer, entry.offset, entry.nbytes, Uint8Array),
      rowScalesF16: alignedSlice(this.q4Buffer, entry.scale_offset, entry.scale_nbytes, Uint16Array),
    };
  }

  denseF32Tensor(name) {
    if (this.denseCache.has(name)) return this.denseCache.get(name);
    const entry = this.denseIndex[name];
    if (!entry) throw new Error(`dense tensor not found: ${name}`);
    if (entry.dtype !== "float16") throw new Error(`unsupported dense dtype for ${name}: ${entry.dtype}`);
    const raw = alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Uint16Array);
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) out[i] = f16ToF32(raw[i]);
    this.denseCache.set(name, out);
    return out;
  }

  runQ4Linear(name, input, rows = 1, biasName = "") {
    const { entry, packedWeight, rowScalesF16 } = this.q4Tensor(name);
    const shape = entry.shape.map(Number);
    const outDim = shape[0];
    const inDim = shape.slice(1).reduce((acc, value) => acc * value, 1);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return q4_symmetric_linear_f32(input, packedWeight, rowScalesF16, bias, rows, inDim, outDim);
  }
}

function loadFP16Bundle(bundleDir) {
  const manifest = readJson(path.join(bundleDir, "manifest.json"));
  const index = readJson(path.join(bundleDir, manifest.files.index));
  const buffer = fs.readFileSync(path.join(bundleDir, manifest.files.tensors));
  return new FP16TensorBundle({ manifest, index, buffer });
}

function loadVocosBundle(bundleDir) {
  const manifest = readJson(path.join(bundleDir, "manifest.json"));
  return manifest.files?.q4 ? new NodeQ4Bundle(bundleDir) : loadFP16Bundle(bundleDir);
}

function tokenize(text, vocabPath, maxLen) {
  const vocab = fs.readFileSync(vocabPath, "utf8").split(/\r?\n/).filter(Boolean);
  const map = new Map();
  vocab.forEach((token, index) => map.set(token, index));
  const ids = new Int32Array(maxLen);
  ids.fill(-1);
  const chars = Array.from(text);
  for (let i = 0; i < Math.min(chars.length, maxLen); i += 1) {
    ids[i] = map.has(chars[i]) ? map.get(chars[i]) : -1;
  }
  return ids;
}

function writeWav(filePath, samples, sampleRate) {
  const dataBytes = samples.length * 2;
  const buffer = Buffer.alloc(44 + dataBytes);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataBytes, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataBytes, 40);
  for (let i = 0; i < samples.length; i += 1) {
    const clipped = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.round(clipped * 32767), 44 + i * 2);
  }
  fs.writeFileSync(filePath, buffer);
}

const f5BundleDir = process.argv[2] || defaultF5Bundle;
const vocosBundleDir = process.argv[3] || defaultVocosBundle;
const refWav = process.argv[4] || defaultRefWav;
const text = process.argv[5] || "This is Peyton speaking from the local int4 browser stack.";
const condSeqLen = Number(process.argv[6] || 4);
const genFrames = Number(process.argv[7] || 4);
const duration = condSeqLen + genFrames;

await initWasm({ module_or_path: fs.readFileSync(path.join(wasmPkg, "model_stack_bitnet_wasm_bg.wasm")) });
const vocosBundle = loadVocosBundle(vocosBundleDir);
const wavBytes = fs.readFileSync(refWav);
const wav = decodeWavMono(wavBytes.buffer.slice(wavBytes.byteOffset, wavBytes.byteOffset + wavBytes.byteLength));
const { mel: refMel, frames: refFrames } = vocosMelFromMono(wav.samples, vocosBundle, { maxFrames: condSeqLen });
if (refFrames < condSeqLen) {
  throw new Error(`reference wav only yielded ${refFrames} mel frames`);
}

const f5 = new F5TTSQ4DiTRuntime(new NodeQ4Bundle(f5BundleDir));
const vocos = new VocosMel24khzRuntime(vocosBundle);
const textIds = tokenize(text, defaultVocab, duration);

const started = Date.now();
const mel = f5.sampleMel({ condMel: refMel, condSeqLen, textIds, duration, steps: 1, cfgStrength: 0.0 });
const audio = vocos.decode(mel);
const elapsedMs = Date.now() - started;

let finite = true;
let peak = 0;
let checksum = 0;
for (const value of audio) {
  if (!Number.isFinite(value)) finite = false;
  peak = Math.max(peak, Math.abs(value));
  checksum += value;
}

fs.mkdirSync(outDir, { recursive: true });
const wavPath = path.join(outDir, "peyton_ref_q4_vocos_smoke.wav");
writeWav(wavPath, audio, SAMPLE_RATE);

console.log(JSON.stringify({
  refWav,
  text,
  condSeqLen,
  genFrames,
  duration,
  audioSamples: audio.length,
  wavPath,
  finite,
  peak: Number(peak.toFixed(6)),
  checksum: Number(checksum.toFixed(6)),
  elapsedMs,
}, null, 2));

if (!finite || audio.length <= 0) process.exitCode = 1;
