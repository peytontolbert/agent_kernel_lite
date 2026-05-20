import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { decodeWavMono, vocosMelFromMono } from "../../model-stack/browser/bitnet/audio_mel_runtime.js";
import { F5TTSQ4DiTRuntime } from "../../model-stack/browser/bitnet/f5tts_q4_dit_runtime.js";
import { FP16TensorBundle, SAMPLE_RATE, VocosMel24khzRuntime } from "../../model-stack/browser/bitnet/vocos_fp16_runtime.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const wasmPkg = path.join(repoRoot, "web/vendor/model-stack-bitnet");
const defaultF5Bundle = "/data/resumebot/checkpoints/f5tts_peyton_q4_v0";
const defaultVocosBundle = "/data/resumebot/checkpoints/vocos_mel_24khz_fp16_v0";
const defaultRefWav = path.join(repoRoot, "apps/mobile/www/app/voice/peyton/sample_0.wav");
const defaultVocab = path.join(repoRoot, "apps/mobile/www/app/voice/peyton/F5TTS_Base_vocab.txt");
const outDir = path.join(here, "out");
const defaultReferenceText = "Hi, I'm recording this sample to create a ";
const referenceAudioStartSec = 0.0;
const referenceMelFrames = 256;
const fullReferenceTextBytes = 42;
const referenceFramesPerTextByte = referenceMelFrames / fullReferenceTextBytes;

const { default: initWasm, q4_symmetric_linear_f32, q4_conv1d_f32, q4_depthwise_conv1d_f32, vocos_istft_head_f32, F5Q4DiTSession, Q4LinearHandle } = await import(
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
    this.q4LinearHandleCache = new Map();
    this.f5SessionCache = null;
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
    let out;
    if (entry.dtype === "float32") {
      out = alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Float32Array);
    } else if (entry.dtype === "float16") {
      const raw = alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Uint16Array);
      out = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i += 1) out[i] = f16ToF32(raw[i]);
    } else {
      throw new Error(`unsupported dense dtype for ${name}: ${entry.dtype}`);
    }
    this.denseCache.set(name, out);
    return out;
  }

  runQ4Linear(name, input, rows = 1, biasName = "") {
    if (Q4LinearHandle) {
      return this.q4LinearHandle(name, biasName).forward(input instanceof Float32Array ? input : new Float32Array(input), rows);
    }
    const { entry, packedWeight, rowScalesF16 } = this.q4Tensor(name);
    const shape = entry.shape.map(Number);
    const outDim = shape[0];
    const inDim = shape.slice(1).reduce((acc, value) => acc * value, 1);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return q4_symmetric_linear_f32(input, packedWeight, rowScalesF16, bias, rows, inDim, outDim);
  }

  runQ4Conv1d(name, biasName, input, seqLen, inChannels, outChannels, kernel, padding) {
    const { packedWeight, rowScalesF16 } = this.q4Tensor(name);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return q4_conv1d_f32(
      input instanceof Float32Array ? input : new Float32Array(input),
      packedWeight,
      rowScalesF16,
      bias,
      seqLen,
      inChannels,
      outChannels,
      kernel,
      padding,
    );
  }

  runQ4DepthwiseConv1d(name, biasName, input, seqLen, channels, kernel, padding) {
    const { packedWeight, rowScalesF16 } = this.q4Tensor(name);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return q4_depthwise_conv1d_f32(
      input instanceof Float32Array ? input : new Float32Array(input),
      packedWeight,
      rowScalesF16,
      bias,
      seqLen,
      channels,
      kernel,
      padding,
    );
  }

  runVocosIstftHead(stftRows, frames) {
    return vocos_istft_head_f32(stftRows instanceof Float32Array ? stftRows : new Float32Array(stftRows), frames);
  }

  q4LinearHandle(name, biasName = "") {
    const key = `${name}:${biasName || ""}`;
    const cached = this.q4LinearHandleCache.get(key);
    if (cached) return cached;
    const { entry, packedWeight, rowScalesF16 } = this.q4Tensor(name);
    const shape = entry.shape.map(Number);
    const outDim = shape[0];
    const inDim = shape.slice(1).reduce((acc, value) => acc * value, 1);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    const handle = new Q4LinearHandle(packedWeight, rowScalesF16, bias, inDim, outDim);
    this.q4LinearHandleCache.set(key, handle);
    return handle;
  }

  f5Session() {
    if (this.f5SessionCache) return this.f5SessionCache;
    const session = new F5Q4DiTSession();
    for (const name of Object.keys(this.q4Index)) {
      const { entry, packedWeight, rowScalesF16 } = this.q4Tensor(name);
      const shape = entry.shape.map(Number);
      const outDim = shape[0];
      const inDim = shape.slice(1).reduce((acc, value) => acc * value, 1);
      const biasName = name.endsWith(".weight") ? `${name.slice(0, -".weight".length)}.bias` : "";
      const bias = biasName && this.denseIndex[biasName] ? this.denseF32Tensor(biasName) : new Float32Array(0);
      session.add_q4_tensor(name, packedWeight, rowScalesF16, bias, inDim, outDim);
    }
    for (const [name, entry] of Object.entries(this.denseIndex)) {
      if (entry.dtype === "float16" || entry.dtype === "float32") {
        session.add_dense_f32(name, this.denseF32Tensor(name));
      }
    }
    this.f5SessionCache = session;
    return session;
  }

  prepareF5Session() {
    return this.f5Session();
  }

  runF5SampleMel({ condMel, condSeqLen, textIds, duration, steps, cfgStrength, swaySamplingCoef = -1.0, seed = 1337 }) {
    return this.f5Session().sample_mel(
      condMel instanceof Float32Array ? condMel : new Float32Array(condMel),
      condSeqLen,
      textIds instanceof Int32Array ? textIds : new Int32Array(textIds),
      duration,
      steps,
      cfgStrength,
      swaySamplingCoef,
      seed,
    );
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
const text = process.argv[5] || "This is Peyton speaking from Agent Kernel Lite.";
const condSeqLen = Number(process.argv[6] || referenceMelFrames);
const genFrames = Number(process.argv[7] || Math.ceil(new TextEncoder().encode(text.trim()).length * referenceFramesPerTextByte));
const steps = Number(process.argv[8] || 12);
const cfgStrength = Number(process.argv[9] || 2.0);
const referenceText = process.argv[10] || defaultReferenceText;
const outputStartFrame = Number(process.argv[11] || condSeqLen);
const duration = condSeqLen + genFrames;

await initWasm({ module_or_path: fs.readFileSync(path.join(wasmPkg, "model_stack_bitnet_wasm_bg.wasm")) });
const vocosBundle = loadVocosBundle(vocosBundleDir);
const wavBytes = fs.readFileSync(refWav);
const wav = decodeWavMono(wavBytes.buffer.slice(wavBytes.byteOffset, wavBytes.byteOffset + wavBytes.byteLength));
const refStartSample = Math.min(wav.samples.length, Math.max(0, Math.round(referenceAudioStartSec * wav.sampleRate)));
const { mel: refMel, frames: refFrames } = vocosMelFromMono(wav.samples.subarray(refStartSample), vocosBundle, { maxFrames: condSeqLen });
if (refFrames < condSeqLen) {
  throw new Error(`reference wav only yielded ${refFrames} mel frames`);
}

const f5 = new F5TTSQ4DiTRuntime(new NodeQ4Bundle(f5BundleDir));
const sessionStarted = Date.now();
f5.prepareSession();
const sessionMs = Date.now() - sessionStarted;
const vocos = new VocosMel24khzRuntime(vocosBundle);
const textIds = tokenize(`${referenceText}${text}`, defaultVocab, duration);

const started = Date.now();
const mel = f5.sampleMel({ condMel: refMel, condSeqLen, textIds, duration, steps, cfgStrength });
const generationMs = Date.now() - started;
const decodeStarted = Date.now();
const generatedMel = mel.subarray(outputStartFrame * 100);
const audio = vocos.decode(generatedMel);
const decodeMs = Date.now() - decodeStarted;

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
const melPath = path.join(outDir, "peyton_ref_q4_generated_mel.f32");
writeWav(wavPath, audio, SAMPLE_RATE);
fs.writeFileSync(melPath, Buffer.from(generatedMel.buffer, generatedMel.byteOffset, generatedMel.byteLength));

console.log(JSON.stringify({
  refWav,
  text,
  referenceText,
  refStartSample,
  condSeqLen,
  outputStartFrame,
  genFrames,
  duration,
  steps,
  cfgStrength,
  audioSamples: audio.length,
  wavPath,
  melPath,
  melFrames: generatedMel.length / 100,
  finite,
  peak: Number(peak.toFixed(6)),
  checksum: Number(checksum.toFixed(6)),
  sessionMs,
  generationMs,
  decodeMs,
}, null, 2));

if (!finite || audio.length <= 0) process.exitCode = 1;
