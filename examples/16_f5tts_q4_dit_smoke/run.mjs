import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { F5TTSQ4DiTRuntime } from "../../model-stack/browser/bitnet/f5tts_q4_dit_runtime.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const wasmPkg = path.join(repoRoot, "model-stack/browser/bitnet/pkg");
const defaultBundle = "/data/resumebot/checkpoints/f5tts_peyton_q4_v0";

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
    this.bundleDir = bundleDir;
    this.manifest = readJson(path.join(bundleDir, "manifest.json"));
    this.q4Index = readJson(path.join(bundleDir, this.manifest.files.q4_index));
    this.denseIndex = readJson(path.join(bundleDir, this.manifest.files.dense_index));
    this.q4Buffer = fs.readFileSync(path.join(bundleDir, this.manifest.files.q4));
    this.denseBuffer = fs.readFileSync(path.join(bundleDir, this.manifest.files.dense));
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
    const entry = this.denseIndex[name];
    if (!entry) throw new Error(`dense tensor not found: ${name}`);
    if (entry.dtype === "float32") {
      return alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Float32Array);
    }
    if (entry.dtype !== "float16") {
      throw new Error(`dense tensor is not float16/float32: ${name}`);
    }
    const raw = alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Uint16Array);
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) out[i] = f16ToF32(raw[i]);
    return out;
  }

  runQ4Linear(name, input, rows = 1, biasName = "") {
    const { entry, packedWeight, rowScalesF16 } = this.q4Tensor(name);
    const shape = entry.shape.map(Number);
    const outDim = shape[0];
    const inDim = shape.slice(1).reduce((acc, value) => acc * value, 1);
    if (input.length !== rows * inDim) {
      throw new Error(`${name} input length mismatch: got ${input.length}, expected ${rows * inDim}`);
    }
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return q4_symmetric_linear_f32(input, packedWeight, rowScalesF16, bias, rows, inDim, outDim);
  }
}

function makeMel(seqLen, melDim) {
  const values = new Float32Array(seqLen * melDim);
  for (let i = 0; i < values.length; i += 1) {
    values[i] = 0.01 * Math.sin(i * 0.17) + 0.005 * Math.cos(i * 0.031);
  }
  return values;
}

const bundleDir = process.argv[2] || defaultBundle;
const seqLen = Number(process.argv[3] || 4);
const wasmBytes = fs.readFileSync(path.join(wasmPkg, "model_stack_bitnet_wasm_bg.wasm"));
await initWasm({ module_or_path: wasmBytes });

const bundle = new NodeQ4Bundle(bundleDir);
const runtime = new F5TTSQ4DiTRuntime(bundle);
const x = makeMel(seqLen, 100);
const cond = new Float32Array(x.length);
const textIds = new Int32Array(seqLen);
textIds.fill(-1);

const started = Date.now();
const output = runtime.forward({ x, cond, textIds, time: 0.25 });
const generatedMel = runtime.sampleMel({
  condMel: x.subarray(0, Math.max(1, Math.floor(seqLen / 2)) * 100),
  condSeqLen: Math.max(1, Math.floor(seqLen / 2)),
  textIds,
  duration: seqLen,
  steps: 1,
  cfgStrength: 0.0,
});
const elapsedMs = Date.now() - started;

let finite = true;
let checksum = 0;
let maxAbs = 0;
for (const value of output) {
  if (!Number.isFinite(value)) finite = false;
  checksum += value;
  maxAbs = Math.max(maxAbs, Math.abs(value));
}
let sampleFinite = true;
let sampleChecksum = 0;
for (const value of generatedMel) {
  if (!Number.isFinite(value)) sampleFinite = false;
  sampleChecksum += value;
}

console.log(JSON.stringify({
  bundleDir,
  seqLen,
  outputShape: [seqLen, 100],
  outputLength: output.length,
  finite,
  checksum: Number(checksum.toFixed(6)),
  maxAbs: Number(maxAbs.toFixed(6)),
  sampleMelShape: [seqLen, 100],
  sampleMelFinite: sampleFinite,
  sampleMelChecksum: Number(sampleChecksum.toFixed(6)),
  elapsedMs,
}, null, 2));

if (!finite || !sampleFinite || output.length !== seqLen * 100 || generatedMel.length !== seqLen * 100) {
  process.exitCode = 1;
}
