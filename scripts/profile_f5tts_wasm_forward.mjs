import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

import initWasm, { F5Q4DiTSession } from '../model-stack/browser/bitnet/pkg/model_stack_bitnet_wasm.js';

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function alignedSlice(buffer, offset, nbytes, TypedArray) {
  const byteOffset = buffer.byteOffset + offset;
  if (byteOffset % TypedArray.BYTES_PER_ELEMENT === 0) {
    return new TypedArray(buffer.buffer, byteOffset, nbytes / TypedArray.BYTES_PER_ELEMENT);
  }
  const copy = Buffer.from(buffer.subarray(offset, offset + nbytes));
  return new TypedArray(copy.buffer, copy.byteOffset, nbytes / TypedArray.BYTES_PER_ELEMENT);
}

function f16ToF32(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

class NodeQ4Bundle {
  constructor(bundleDir) {
    this.bundleDir = bundleDir;
    this.manifest = readJson(path.join(bundleDir, 'manifest.json'));
    this.q4Index = readJson(path.join(bundleDir, this.manifest.files.q4_index));
    this.denseIndex = readJson(path.join(bundleDir, this.manifest.files.dense_index));
    this.q4Buffer = fs.readFileSync(path.join(bundleDir, this.manifest.files.q4));
    this.denseBuffer = fs.readFileSync(path.join(bundleDir, this.manifest.files.dense));
    this.denseCache = new Map();
    this.session = null;
  }

  denseF32Tensor(name) {
    if (this.denseCache.has(name)) return this.denseCache.get(name);
    const entry = this.denseIndex[name];
    if (!entry) throw new Error(`dense tensor not found: ${name}`);
    let out;
    if (entry.dtype === 'float32') {
      out = alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Float32Array);
    } else if (entry.dtype === 'float16') {
      const raw = alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Uint16Array);
      out = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i += 1) out[i] = f16ToF32(raw[i]);
    } else {
      throw new Error(`unsupported dense dtype ${entry.dtype} for ${name}`);
    }
    this.denseCache.set(name, out);
    return out;
  }

  f5Session() {
    if (this.session) return this.session;
    const session = new F5Q4DiTSession();
    for (const [name, entry] of Object.entries(this.q4Index)) {
      const shape = entry.shape.map(Number);
      const outDim = shape[0];
      const inDim = shape.slice(1).reduce((acc, value) => acc * value, 1);
      const biasName = name.replace(/\.weight$/, '.bias');
      const bias = this.denseIndex[biasName] ? this.denseF32Tensor(biasName) : new Float32Array(0);
      session.add_q4_tensor(
        name,
        alignedSlice(this.q4Buffer, entry.offset, entry.nbytes, Uint8Array),
        alignedSlice(this.q4Buffer, entry.scale_offset, entry.scale_nbytes, Uint16Array),
        bias,
        inDim,
        outDim,
      );
    }
    for (const name of Object.keys(this.denseIndex)) {
      const entry = this.denseIndex[name];
      if (entry.dtype === 'float16' || entry.dtype === 'float32') session.add_dense_f32(name, this.denseF32Tensor(name));
    }
    this.session = session;
    return session;
  }
}

function timed(name, fn, reports) {
  const started = performance.now();
  const value = fn();
  reports.push({ name, ms: Math.round(performance.now() - started) });
  return value;
}

function lcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

const bundleDir = process.argv[2] || '/data/resumebot/checkpoints/f5tts_peyton_q4_v0';
const seqLen = Number(process.argv[3] || 347);
const genText = process.argv[4] || "Hi, I'm recording this sample to create a This is Peyton.";
const detailBlockArg = process.argv.find((arg) => arg.startsWith('--detail-block='));
const detailBlock = detailBlockArg ? Number(detailBlockArg.split('=')[1]) : null;
const detailInput = process.argv.includes('--detail-input');

await initWasm({ module_or_path: fs.readFileSync('/data/agent_kernel_lite/model-stack/browser/bitnet/pkg/model_stack_bitnet_wasm_bg.wasm') });
const session = new NodeQ4Bundle(bundleDir).f5Session();
const next = lcg(1337);
const x = new Float32Array(seqLen * 100);
const cond = new Float32Array(seqLen * 100);
for (let i = 0; i < x.length; i += 1) {
  x[i] = (next() * 2 - 1) * 0.25;
  cond[i] = (next() * 2 - 1) * 0.25;
}
const textIds = new Int32Array(seqLen);
textIds.fill(-1);
for (let i = 0; i < Math.min(seqLen, genText.length); i += 1) {
  textIds[i] = genText.charCodeAt(i) % 255;
}

const reports = [];
const totalStarted = performance.now();
const t = timed('time_embedding', () => session.debug_time_embedding(0.5), reports);
for (let i = 0; i < t.length; i += 1) t[i] = t[i] / (1 + Math.exp(-t[i]));
const text = timed('text_embedding', () => session.debug_text_embedding(textIds, seqLen, false), reports);
const inputDetail = detailInput ? JSON.parse(session.debug_input_embedding_profile_json(x, cond, text, seqLen, false)) : null;
let hidden = timed('input_embedding', () => session.debug_input_embedding(x, cond, text, seqLen, false), reports);
let blockDetail = null;
for (let block = 0; block < 22; block += 1) {
  if (block === detailBlock) {
    blockDetail = JSON.parse(session.debug_dit_block_profile_json(block, hidden, t, seqLen));
  }
  hidden = timed(`block_${block}`, () => session.debug_dit_block(block, hidden, t, seqLen), reports);
}
timed('final_norm', () => session.debug_final_ada_norm(hidden, t, seqLen), reports);
console.log(JSON.stringify({ bundleDir, seqLen, totalMs: Math.round(performance.now() - totalStarted), reports, inputDetail, blockDetail }, null, 2));
