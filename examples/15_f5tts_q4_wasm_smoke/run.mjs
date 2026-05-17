import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const wasmPkg = path.join(repoRoot, "model-stack/browser/bitnet/pkg");
const defaultBundle = "/data/resumebot/checkpoints/f5tts_peyton_q4_v0";

const { default: initWasm, q4_symmetric_linear_f32 } = await import(
  pathToFileUrl(path.join(wasmPkg, "model_stack_bitnet_wasm.js"))
);

function pathToFileUrl(value) {
  return new URL(`file://${value}`).href;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function alignedSlice(buffer, offset, nbytes, TypedArray) {
  const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset + offset, nbytes);
  const copy = new Uint8Array(nbytes);
  copy.set(bytes);
  return new TypedArray(copy.buffer);
}

function makeInput(length) {
  const input = new Float32Array(length);
  for (let i = 0; i < input.length; i += 1) {
    input[i] = Math.sin(i * 0.013) * 0.5 + Math.cos(i * 0.031) * 0.25;
  }
  return input;
}

const bundleDir = process.argv[2] || defaultBundle;
const tensorName = process.argv[3] || "";
const manifest = readJson(path.join(bundleDir, "manifest.json"));
const q4Index = readJson(path.join(bundleDir, manifest.files.q4_index));
const q4Buffer = fs.readFileSync(path.join(bundleDir, manifest.files.q4));
const wasmBytes = fs.readFileSync(path.join(wasmPkg, "model_stack_bitnet_wasm_bg.wasm"));

await initWasm({ module_or_path: wasmBytes });

const selectedName = tensorName || Object.keys(q4Index)[0];
const entry = q4Index[selectedName];
if (!entry) {
  throw new Error(`Q4 tensor not found: ${selectedName}`);
}

const shape = entry.shape.map(Number);
const outDim = shape[0];
const inDim = shape.slice(1).reduce((acc, value) => acc * value, 1);
const input = makeInput(inDim);
const packedWeight = alignedSlice(q4Buffer, entry.offset, entry.nbytes, Uint8Array);
const rowScalesF16 = alignedSlice(q4Buffer, entry.scale_offset, entry.scale_nbytes, Uint16Array);
const output = q4_symmetric_linear_f32(input, packedWeight, rowScalesF16, new Float32Array(0), 1, inDim, outDim);

let finite = true;
let checksum = 0;
for (const value of output) {
  if (!Number.isFinite(value)) finite = false;
  checksum += value;
}

console.log(JSON.stringify({
  bundleDir,
  tensorName: selectedName,
  shape,
  inputLength: input.length,
  outputLength: output.length,
  finite,
  checksum: Number(checksum.toFixed(6)),
}, null, 2));

if (!finite || output.length !== outDim) {
  process.exitCode = 1;
}
