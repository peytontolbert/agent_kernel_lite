import fs from 'node:fs';
import path from 'node:path';
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

function readF32(file) {
  const buffer = fs.readFileSync(file);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
}

function readI32(file) {
  const buffer = fs.readFileSync(file);
  return new Int32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
}

function compareArray(name, actual, expected) {
  let maxAbs = 0;
  let mae = 0;
  let maxIdx = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const diff = Math.abs(actual[i] - expected[i]);
    mae += diff;
    if (diff > maxAbs) {
      maxAbs = diff;
      maxIdx = i;
    }
  }
  mae /= expected.length;
  return {
    name,
    len: expected.length,
    actualChecksum: Array.from(actual).reduce((a, b) => a + b, 0),
    expectedChecksum: Array.from(expected).reduce((a, b) => a + b, 0),
    mae,
    maxAbs,
    maxIdx,
    actualAtMax: actual[maxIdx],
    expectedAtMax: expected[maxIdx],
  };
}

const bundleDir = process.argv[2] || '/data/resumebot/checkpoints/f5tts_peyton_q4_v0';
const fixtureDir = process.argv[3] || '/data/agent_kernel_lite/tmp/f5tts_q4_forward_fixture';
const seqLen = Number(process.argv[4] || 32);

await initWasm({ module_or_path: fs.readFileSync('/data/agent_kernel_lite/model-stack/browser/bitnet/pkg/model_stack_bitnet_wasm_bg.wasm') });
const bundle = new NodeQ4Bundle(bundleDir);
const x = readF32(path.join(fixtureDir, 'x.f32'));
const cond = readF32(path.join(fixtureDir, 'cond.f32'));
const textIds = readI32(path.join(fixtureDir, 'text_ids.i32'));
const expected = readF32(path.join(fixtureDir, 'output.f32'));
const time = readF32(path.join(fixtureDir, 'time.f32'))[0];

const session = bundle.f5Session();
const timeEmbedding = session.debug_time_embedding(time);
const textEmbedding = session.debug_text_embedding(textIds, seqLen, false);
const inputEmbedding = session.debug_input_embedding(x, cond, textEmbedding, seqLen, false);
let hidden = inputEmbedding;
const reports = [
  compareArray('time_embedding', timeEmbedding, readF32(path.join(fixtureDir, 'time_embedding.f32'))),
  compareArray('text_embedding', textEmbedding, readF32(path.join(fixtureDir, 'text_embedding.f32'))),
  compareArray('input_embedding', inputEmbedding, readF32(path.join(fixtureDir, 'input_embedding.f32'))),
];
const timeSilu = readF32(path.join(fixtureDir, 'time_embedding_silu.f32'));
for (let block = 0; block < 22; block += 1) {
  hidden = session.debug_dit_block(block, hidden, timeSilu, seqLen);
  if (block === 0 || block === 1 || block === 21) {
    reports.push(compareArray(`block_${block}`, hidden, readF32(path.join(fixtureDir, `block_${block}.f32`))));
  }
}
const finalNorm = session.debug_final_ada_norm(hidden, timeSilu, seqLen);
reports.push(compareArray('final_norm', finalNorm, readF32(path.join(fixtureDir, 'final_norm.f32'))));
const actual = session.forward(x, cond, textIds, time, false, false);
reports.push(compareArray('output', actual, expected));
console.log(JSON.stringify({ bundleDir, fixtureDir, seqLen, reports }, null, 2));
