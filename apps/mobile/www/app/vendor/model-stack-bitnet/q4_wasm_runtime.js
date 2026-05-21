let wasmModulePromise = null;
const WASM_RUNTIME_VERSION = "20260521-f5-webgpu-session-v2";

function resolveUrl(path, baseUrl) {
  return new URL(path, baseUrl).toString();
}

function tensorView(buffer, offset, nbytes, TypedArray) {
  const elementBytes = TypedArray.BYTES_PER_ELEMENT || 1;
  if (offset % elementBytes === 0) {
    return new TypedArray(buffer, offset, Math.floor(nbytes / elementBytes));
  }
  const bytes = new Uint8Array(buffer, offset, nbytes);
  const copy = new Uint8Array(nbytes);
  copy.set(bytes);
  return new TypedArray(copy.buffer);
}

async function ensureQ4Wasm() {
  if (!wasmModulePromise) {
    wasmModulePromise = (async () => {
      let module;
      let moduleUrl;
      try {
        moduleUrl = new URL("model_stack_bitnet_wasm.js", import.meta.url);
        moduleUrl.searchParams.set("v", WASM_RUNTIME_VERSION);
        module = await import(moduleUrl.href);
      } catch (error) {
        moduleUrl = new URL("pkg/model_stack_bitnet_wasm.js", import.meta.url);
        moduleUrl.searchParams.set("v", WASM_RUNTIME_VERSION);
        module = await import(moduleUrl.href);
      }
      const wasmUrl = new URL("model_stack_bitnet_wasm_bg.wasm", moduleUrl);
      wasmUrl.searchParams.set("v", WASM_RUNTIME_VERSION);
      const wasmBytes = await fetchBuffer(wasmUrl.href, "Model Stack WASM runtime");
      await module.default(wasmBytes);
      return module;
    })();
  }
  return wasmModulePromise;
}

async function fetchJson(url, label = "JSON asset") {
  try {
    const response = await fetch(url, { cache: "force-cache" });
    if (!response.ok && response.status !== 0) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  } catch (error) {
    const text = await xhrText(url).catch((xhrError) => {
      throw new Error(`${label} load failed: ${url}: ${error.message || String(error)}; XHR fallback: ${xhrError.message || String(xhrError)}`);
    });
    return JSON.parse(text);
  }
}

async function fetchBuffer(url, label = "binary asset") {
  try {
    const response = await fetch(url, { cache: "force-cache" });
    if (!response.ok && response.status !== 0) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.arrayBuffer();
  } catch (error) {
    return xhrArrayBuffer(url).catch((xhrError) => {
      throw new Error(`${label} load failed: ${url}: ${error.message || String(error)}; XHR fallback: ${xhrError.message || String(xhrError)}`);
    });
  }
}

async function fetchChunkedBuffer(baseUrl, chunks, label = "chunked binary asset", expectedBytes = null) {
  const chunkList = chunks.map((chunk) => (typeof chunk === "string" ? { path: chunk } : chunk));
  const totalBytes = Number.isFinite(Number(expectedBytes)) ? Number(expectedBytes) : null;
  if (totalBytes !== null) {
    const output = new Uint8Array(totalBytes);
    let offset = 0;
    for (let i = 0; i < chunkList.length; i += 1) {
      const chunk = chunkList[i];
      const path = chunk.path || chunk.file || chunk.url;
      if (!path) {
        throw new Error(`${label} has chunk ${i} without a path`);
      }
      const bytes = new Uint8Array(await fetchBuffer(resolveUrl(path, baseUrl), `${label} chunk ${i + 1}/${chunkList.length}`));
      if (offset + bytes.byteLength > output.byteLength) {
        throw new Error(`${label} exceeded expected byte length ${totalBytes}`);
      }
      output.set(bytes, offset);
      offset += bytes.byteLength;
    }
    if (offset !== output.byteLength) {
      throw new Error(`${label} expected ${output.byteLength} bytes but loaded ${offset}`);
    }
    return output.buffer;
  }

  const loaded = [];
  let total = 0;
  for (let i = 0; i < chunkList.length; i += 1) {
    const chunk = chunkList[i];
    const path = chunk.path || chunk.file || chunk.url;
    if (!path) {
      throw new Error(`${label} has chunk ${i} without a path`);
    }
    const bytes = new Uint8Array(await fetchBuffer(resolveUrl(path, baseUrl), `${label} chunk ${i + 1}/${chunkList.length}`));
    loaded.push(bytes);
    total += bytes.byteLength;
  }
  const output = new Uint8Array(total);
  let offset = 0;
  for (const bytes of loaded) {
    output.set(bytes, offset);
    offset += bytes.byteLength;
  }
  return output.buffer;
}

function xhrArrayBuffer(url) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.responseType = "arraybuffer";
    request.onload = () => {
      if ((request.status >= 200 && request.status < 300) || request.status === 0) {
        resolve(request.response);
      } else {
        reject(new Error(`XHR ${request.status}`));
      }
    };
    request.onerror = () => reject(new Error("XHR network error"));
    request.send();
  });
}

function xhrText(url) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.responseType = "text";
    request.onload = () => {
      if ((request.status >= 200 && request.status < 300) || request.status === 0) {
        resolve(request.responseText);
      } else {
        reject(new Error(`XHR ${request.status}`));
      }
    };
    request.onerror = () => reject(new Error("XHR network error"));
    request.send();
  });
}



function align4(value) {
  return (value + 3) & ~3;
}

function gpuBufferUsage() {
  const usage = globalThis.GPUBufferUsage;
  if (!usage) throw new Error('WebGPU buffer constants are not available in this worker');
  return usage;
}

function gpuMapMode() {
  const mode = globalThis.GPUMapMode;
  if (!mode) throw new Error('WebGPU map constants are not available in this worker');
  return mode;
}

function createGpuStorageBuffer(device, data, usage = 0) {
  const bufferUsage = gpuBufferUsage();
  const source = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
  const buffer = device.createBuffer({
    size: align4(source.byteLength),
    usage: (usage || bufferUsage.STORAGE) | bufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, source.buffer, source.byteOffset, source.byteLength);
  return buffer;
}

function q4ShaderSource() {
  return `
struct Q4LinearParams {
  rows: u32,
  in_features: u32,
  out_features: u32,
  has_bias: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weight: array<u32>;
@group(0) @binding(2) var<storage, read> row_scales: array<f32>;
@group(0) @binding(3) var<storage, read> bias_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Q4LinearParams;

fn decode_q4(byte_value: u32, linear: u32) -> f32 {
  let code = select(byte_value >> 4u, byte_value & 15u, (linear & 1u) == 0u) & 15u;
  var signed_code = i32(code);
  if (code >= 8u) {
    signed_code = signed_code - 16;
  }
  return f32(signed_code);
}

fn load_q4(row: u32, col: u32) -> f32 {
  let linear = row * params.in_features + col;
  let byte_offset = linear / 2u;
  let word = packed_weight[byte_offset / 4u];
  let lane = byte_offset & 3u;
  let byte_value = (word >> (lane * 8u)) & 255u;
  return decode_q4(byte_value, linear) * row_scales[row];
}

@compute @workgroup_size(8, 8, 1)
fn q4_linear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_idx = gid.x;
  let batch_row = gid.y;
  if (out_idx >= params.out_features || batch_row >= params.rows) {
    return;
  }
  var acc = 0.0;
  var col = 0u;
  loop {
    if (col >= params.in_features) { break; }
    acc = acc + input[batch_row * params.in_features + col] * load_q4(out_idx, col);
    col = col + 1u;
  }
  if (params.has_bias != 0u) {
    acc = acc + bias_values[out_idx];
  }
  output[batch_row * params.out_features + out_idx] = acc;
}
`;
}

function f32ScalesFromF16(rowScalesF16) {
  const out = new Float32Array(rowScalesF16.length);
  for (let i = 0; i < rowScalesF16.length; i += 1) out[i] = f16ToF32(rowScalesF16[i]);
  return out;
}


class WebGPUTensorF32 {
  constructor(device, buffer, length, rows = 1, cols = length) {
    this.device = device;
    this.buffer = buffer;
    this.length = length;
    this.rows = rows;
    this.cols = cols;
  }

  async readback() {
    const bufferUsage = gpuBufferUsage();
    const bytes = this.length * Float32Array.BYTES_PER_ELEMENT;
    const readbackBuffer = this.device.createBuffer({ size: align4(bytes), usage: bufferUsage.MAP_READ | bufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, readbackBuffer, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await readbackBuffer.mapAsync(gpuMapMode().READ);
    const result = new Float32Array(readbackBuffer.getMappedRange().slice(0, bytes));
    readbackBuffer.unmap();
    return result;
  }
}





function q4Conv1dShaderSource() {
  return `
struct Q4Conv1dParams {
  seq_len: u32,
  channels: u32,
  kernel: u32,
  padding: u32,
  groups: u32,
  group_in: u32,
  row_size: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weight: array<u32>;
@group(0) @binding(2) var<storage, read> row_scales: array<f32>;
@group(0) @binding(3) var<storage, read> bias_values: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(5) var<uniform> params: Q4Conv1dParams;

fn decode_q4(byte_value: u32, linear: u32) -> f32 {
  let code = select(byte_value >> 4u, byte_value & 15u, (linear & 1u) == 0u) & 15u;
  var signed_code = i32(code);
  if (code >= 8u) {
    signed_code = signed_code - 16;
  }
  return f32(signed_code);
}

fn load_q4(row: u32, col: u32) -> f32 {
  let linear = row * params.row_size + col;
  let byte_offset = linear / 2u;
  let word = packed_weight[byte_offset / 4u];
  let lane = byte_offset & 3u;
  let byte_value = (word >> (lane * 8u)) & 255u;
  return decode_q4(byte_value, linear) * row_scales[row];
}

@compute @workgroup_size(8, 8, 1)
fn q4_conv1d_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_ch = gid.x;
  let pos = gid.y;
  if (out_ch >= params.channels || pos >= params.seq_len) { return; }
  let group = out_ch / params.group_in;
  let in_start = group * params.group_in;
  var sum = bias_values[out_ch];
  var local_in = 0u;
  loop {
    if (local_in >= params.group_in) { break; }
    var k = 0u;
    loop {
      if (k >= params.kernel) { break; }
      let src_signed = i32(pos) + i32(k) - i32(params.padding);
      if (src_signed >= 0 && src_signed < i32(params.seq_len)) {
        let src_pos = u32(src_signed);
        let input_index = src_pos * params.channels + in_start + local_in;
        let weight_col = local_in * params.kernel + k;
        sum = sum + input_values[input_index] * load_q4(out_ch, weight_col);
      }
      k = k + 1u;
    }
    local_in = local_in + 1u;
  }
  output_values[pos * params.channels + out_ch] = sum;
}
`;
}

function gatedAddShaderSource() {
  return `
struct GatedAddParams {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> src_values: array<f32>;
@group(0) @binding(2) var<storage, read> gate_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(4) var<uniform> params: GatedAddParams;

@compute @workgroup_size(256, 1, 1)
fn gated_add_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.rows * params.cols;
  if (idx >= total) { return; }
  let col = idx % params.cols;
  output_values[idx] = input_values[idx] + gate_values[col] * src_values[idx];
}
`;
}

function rotaryAttentionShaderSource() {
  return `
struct AttentionParams {
  seq_len: u32,
  heads: u32,
  head_dim: u32,
  dim: u32,
};

@group(0) @binding(0) var<storage, read> q_values: array<f32>;
@group(0) @binding(1) var<storage, read> k_values: array<f32>;
@group(0) @binding(2) var<storage, read> v_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: AttentionParams;

fn rotary_q(pos: u32, head: u32, d: u32) -> f32 {
  let half = params.head_dim / 2u;
  let local = d % half;
  let base = pos * params.dim + head * params.head_dim;
  let a = q_values[base + local];
  let b = q_values[base + local + half];
  let angle = f32(pos) / pow(10000.0, f32(2u * local) / f32(params.head_dim));
  let c = cos(angle);
  let s = sin(angle);
  if (d < half) {
    return a * c - b * s;
  }
  return b * c + a * s;
}

fn rotary_k(pos: u32, head: u32, d: u32) -> f32 {
  let half = params.head_dim / 2u;
  let local = d % half;
  let base = pos * params.dim + head * params.head_dim;
  let a = k_values[base + local];
  let b = k_values[base + local + half];
  let angle = f32(pos) / pow(10000.0, f32(2u * local) / f32(params.head_dim));
  let c = cos(angle);
  let s = sin(angle);
  if (d < half) {
    return a * c - b * s;
  }
  return b * c + a * s;
}

@compute @workgroup_size(8, 8, 1)
fn rotary_attention_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let query = gid.x;
  let head = gid.y;
  if (query >= params.seq_len || head >= params.heads) { return; }
  let scale = inverseSqrt(f32(params.head_dim));
  var max_score = -3.402823e38;
  var key = 0u;
  loop {
    if (key >= params.seq_len) { break; }
    var score = 0.0;
    var d = 0u;
    loop {
      if (d >= params.head_dim) { break; }
      score = score + rotary_q(query, head, d) * rotary_k(key, head, d);
      d = d + 1u;
    }
    score = score * scale;
    max_score = max(max_score, score);
    key = key + 1u;
  }
  var denom = 0.0;
  key = 0u;
  loop {
    if (key >= params.seq_len) { break; }
    var score = 0.0;
    var d = 0u;
    loop {
      if (d >= params.head_dim) { break; }
      score = score + rotary_q(query, head, d) * rotary_k(key, head, d);
      d = d + 1u;
    }
    denom = denom + exp(score * scale - max_score);
    key = key + 1u;
  }
  var d = 0u;
  loop {
    if (d >= params.head_dim) { break; }
    var acc = 0.0;
    key = 0u;
    loop {
      if (key >= params.seq_len) { break; }
      var score = 0.0;
      var inner = 0u;
      loop {
        if (inner >= params.head_dim) { break; }
        score = score + rotary_q(query, head, inner) * rotary_k(key, head, inner);
        inner = inner + 1u;
      }
      let weight = exp(score * scale - max_score) / denom;
      let value_index = key * params.dim + head * params.head_dim + d;
      acc = acc + weight * v_values[value_index];
      key = key + 1u;
    }
    output[query * params.dim + head * params.head_dim + d] = acc;
    d = d + 1u;
  }
}
`;
}

function layerNormAffineShaderSource() {
  return `
struct LayerNormParams {
  rows: u32,
  cols: u32,
  eps: f32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> shift: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: LayerNormParams;

@compute @workgroup_size(1, 1, 1)
fn layernorm_affine_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) { return; }
  let base = row * params.cols;
  var mean = 0.0;
  var col = 0u;
  loop {
    if (col >= params.cols) { break; }
    mean = mean + input[base + col];
    col = col + 1u;
  }
  mean = mean / f32(params.cols);
  var variance = 0.0;
  col = 0u;
  loop {
    if (col >= params.cols) { break; }
    let delta = input[base + col] - mean;
    variance = variance + delta * delta;
    col = col + 1u;
  }
  let inv = inverseSqrt(variance / f32(params.cols) + params.eps);
  col = 0u;
  loop {
    if (col >= params.cols) { break; }
    output[base + col] = (input[base + col] - mean) * inv * (1.0 + scale[col]) + shift[col];
    col = col + 1u;
  }
}
`;
}

function activationShaderSource() {
  return `
struct ActivationParams {
  length: u32,
  activation: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> params: ActivationParams;

fn gelu_tanh(x: f32) -> f32 {
  let coeff = 0.7978845608028654;
  return 0.5 * x * (1.0 + tanh(coeff * (x + 0.044715 * x * x * x)));
}

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

fn mish(x: f32) -> f32 {
  return x * tanh(log(1.0 + exp(x)));
}

@compute @workgroup_size(256, 1, 1)
fn activation_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.length) { return; }
  let x = values[idx];
  if (params.activation == 1u) {
    values[idx] = gelu_tanh(x);
  } else if (params.activation == 2u) {
    values[idx] = silu(x);
  } else if (params.activation == 3u) {
    values[idx] = mish(x);
  }
}
`;
}

class Q4LinearWebGPUHandle {
  constructor(device, pipeline, tensor, bias, inDim, outDim) {
    this.device = device;
    this.pipeline = pipeline;
    this.inDim = inDim;
    this.outDim = outDim;
    this.hasBias = bias && bias.length > 0;
    this.packedWeightBuffer = createGpuStorageBuffer(device, tensor.packedWeight);
    this.scaleBuffer = createGpuStorageBuffer(device, f32ScalesFromF16(tensor.rowScalesF16));
    this.biasBuffer = createGpuStorageBuffer(device, this.hasBias ? bias : new Float32Array([0]));
    const bufferUsage = gpuBufferUsage();
    this.paramsBuffer = device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
    this.runCache = new Map();
  }

  createOutputBuffer(rows) {
    const bufferUsage = gpuBufferUsage();
    return this.device.createBuffer({
      size: align4(rows * this.outDim * Float32Array.BYTES_PER_ELEMENT),
      usage: bufferUsage.STORAGE | bufferUsage.COPY_SRC,
    });
  }

  encodeForward(inputBuffer, outputBuffer, rows = 1) {
    this.device.queue.writeBuffer(this.paramsBuffer, 0, new Uint32Array([rows, this.inDim, this.outDim, this.hasBias ? 1 : 0]));
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: this.packedWeightBuffer } },
        { binding: 2, resource: { buffer: this.scaleBuffer } },
        { binding: 3, resource: { buffer: this.biasBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: this.paramsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.outDim / 8), Math.ceil(rows / 8), 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async forwardGpu(input, rows = 1) {
    const bufferUsage = gpuBufferUsage();
    let inputBuffer;
    if (input instanceof WebGPUTensorF32) {
      if (input.length !== rows * this.inDim) {
        throw new Error(`Q4 WebGPU tensor input length mismatch: got ${input.length}, expected ${rows * this.inDim}`);
      }
      inputBuffer = input.buffer;
    } else {
      const x = input instanceof Float32Array ? input : new Float32Array(input);
      if (x.length !== rows * this.inDim) {
        throw new Error(`Q4 WebGPU input length mismatch: got ${x.length}, expected ${rows * this.inDim}`);
      }
      inputBuffer = this.device.createBuffer({ size: align4(x.byteLength), usage: bufferUsage.STORAGE | bufferUsage.COPY_DST });
      this.device.queue.writeBuffer(inputBuffer, 0, x.buffer, x.byteOffset, x.byteLength);
    }
    const outputBuffer = this.createOutputBuffer(rows);
    this.encodeForward(inputBuffer, outputBuffer, rows);
    return new WebGPUTensorF32(this.device, outputBuffer, rows * this.outDim, rows, this.outDim);
  }

  async forward(input, rows = 1) {
    const x = input instanceof Float32Array ? input : new Float32Array(input);
    if (x.length !== rows * this.inDim) {
      throw new Error(`Q4 WebGPU input length mismatch: got ${x.length}, expected ${rows * this.inDim}`);
    }
    const outputLength = rows * this.outDim;
    const inputBytes = x.byteLength;
    const outputBytes = outputLength * Float32Array.BYTES_PER_ELEMENT;
    const cacheKey = `${rows}:${this.inDim}:${this.outDim}`;
    let cache = this.runCache.get(cacheKey);
    if (!cache) {
      const bufferUsage = gpuBufferUsage();
      const inputBuffer = this.device.createBuffer({ size: align4(inputBytes), usage: bufferUsage.STORAGE | bufferUsage.COPY_DST });
      const outputBuffer = this.device.createBuffer({ size: align4(outputBytes), usage: bufferUsage.STORAGE | bufferUsage.COPY_SRC });
      const readbackBuffer = this.device.createBuffer({ size: align4(outputBytes), usage: bufferUsage.MAP_READ | bufferUsage.COPY_DST });
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: this.packedWeightBuffer } },
          { binding: 2, resource: { buffer: this.scaleBuffer } },
          { binding: 3, resource: { buffer: this.biasBuffer } },
          { binding: 4, resource: { buffer: outputBuffer } },
          { binding: 5, resource: { buffer: this.paramsBuffer } },
        ],
      });
      cache = { inputBuffer, outputBuffer, readbackBuffer, bindGroup };
      this.runCache.set(cacheKey, cache);
    }
    this.device.queue.writeBuffer(cache.inputBuffer, 0, x.buffer, x.byteOffset, x.byteLength);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, new Uint32Array([rows, this.inDim, this.outDim, this.hasBias ? 1 : 0]));
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, cache.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.outDim / 8), Math.ceil(rows / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(cache.outputBuffer, 0, cache.readbackBuffer, 0, outputBytes);
    this.device.queue.submit([encoder.finish()]);
    await cache.readbackBuffer.mapAsync(gpuMapMode().READ);
    const result = new Float32Array(cache.readbackBuffer.getMappedRange().slice(0, outputBytes));
    cache.readbackBuffer.unmap();
    return result;
  }
}

export class Q4TensorBundleWebGPU {
  constructor(baseBundle, device, pipeline) {
    this.base = baseBundle;
    this.device = device;
    this.pipeline = pipeline;
    this.activationPipeline = null;
    this.layerNormAffinePipeline = null;
    this.rotaryAttentionPipeline = null;
    this.gatedAddPipeline = null;
    this.q4Conv1dPipeline = null;
    this.manifest = baseBundle.manifest;
    this.q4Index = baseBundle.q4Index;
    this.denseIndex = baseBundle.denseIndex;
    this.q4LinearHandleCache = new Map();
    this.f5GpuSession = null;
    this.backend = 'webgpu';
  }

  activationComputePipeline() {
    if (!this.activationPipeline) {
      const shaderModule = this.device.createShaderModule({ code: activationShaderSource() });
      const descriptor = { layout: 'auto', compute: { module: shaderModule, entryPoint: 'activation_main' } };
      this.activationPipeline = this.device.createComputePipeline(descriptor);
    }
    return this.activationPipeline;
  }

  layerNormAffineComputePipeline() {
    if (!this.layerNormAffinePipeline) {
      const shaderModule = this.device.createShaderModule({ code: layerNormAffineShaderSource() });
      const descriptor = { layout: 'auto', compute: { module: shaderModule, entryPoint: 'layernorm_affine_main' } };
      this.layerNormAffinePipeline = this.device.createComputePipeline(descriptor);
    }
    return this.layerNormAffinePipeline;
  }

  rotaryAttentionComputePipeline() {
    if (!this.rotaryAttentionPipeline) {
      const shaderModule = this.device.createShaderModule({ code: rotaryAttentionShaderSource() });
      const descriptor = { layout: 'auto', compute: { module: shaderModule, entryPoint: 'rotary_attention_main' } };
      this.rotaryAttentionPipeline = this.device.createComputePipeline(descriptor);
    }
    return this.rotaryAttentionPipeline;
  }

  gatedAddComputePipeline() {
    if (!this.gatedAddPipeline) {
      const shaderModule = this.device.createShaderModule({ code: gatedAddShaderSource() });
      const descriptor = { layout: 'auto', compute: { module: shaderModule, entryPoint: 'gated_add_main' } };
      this.gatedAddPipeline = this.device.createComputePipeline(descriptor);
    }
    return this.gatedAddPipeline;
  }

  q4Conv1dComputePipeline() {
    if (!this.q4Conv1dPipeline) {
      const shaderModule = this.device.createShaderModule({ code: q4Conv1dShaderSource() });
      const descriptor = { layout: 'auto', compute: { module: shaderModule, entryPoint: 'q4_conv1d_main' } };
      this.q4Conv1dPipeline = this.device.createComputePipeline(descriptor);
    }
    return this.q4Conv1dPipeline;
  }

  runActivationInPlace(tensor, activation = 'gelu') {
    const activationId = activation === 'gelu' ? 1 : activation === 'silu' ? 2 : activation === 'mish' ? 3 : 0;
    if (!activationId) return tensor;
    const bufferUsage = gpuBufferUsage();
    const paramsBuffer = this.device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([tensor.length, activationId, 0, 0]));
    const pipeline = this.activationComputePipeline();
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(tensor.length / 256), 1, 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return tensor;
  }


  runLayerNormAffineGpu(input, shift, scale, rows, cols, eps = 1e-6) {
    const source = this.uploadF32Tensor(input, rows, cols);
    const bufferUsage = gpuBufferUsage();
    const outputBuffer = this.device.createBuffer({ size: align4(source.length * Float32Array.BYTES_PER_ELEMENT), usage: bufferUsage.STORAGE | bufferUsage.COPY_SRC });
    const shiftBuffer = createGpuStorageBuffer(this.device, shift instanceof Float32Array ? shift : new Float32Array(shift));
    const scaleBuffer = createGpuStorageBuffer(this.device, scale instanceof Float32Array ? scale : new Float32Array(scale));
    const paramsBuffer = this.device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
    const params = new ArrayBuffer(16);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = rows;
    paramsU32[1] = cols;
    paramsF32[2] = eps;
    this.device.queue.writeBuffer(paramsBuffer, 0, params);
    const pipeline = this.layerNormAffineComputePipeline();
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source.buffer } },
        { binding: 1, resource: { buffer: shiftBuffer } },
        { binding: 2, resource: { buffer: scaleBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(rows, 1, 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return new WebGPUTensorF32(this.device, outputBuffer, source.length, rows, cols);
  }

  async runLayerNormAffineAsync(input, shift, scale, rows, cols, eps = 1e-6) {
    return this.runLayerNormAffineGpu(input, shift, scale, rows, cols, eps);
  }

  async runQ4GroupedConv1dAsync(weightName, biasName, input, seqLen, channels, kernel, padding, groups) {
    const tensor = this.q4Tensor(weightName);
    const rowSize = tensor.entry.shape.slice(1).reduce((acc, value) => acc * Number(value), 1);
    const groupIn = channels / groups;
    const source = this.uploadF32Tensor(input, seqLen, channels);
    const bufferUsage = gpuBufferUsage();
    const outputBytes = seqLen * channels * Float32Array.BYTES_PER_ELEMENT;
    const outputBuffer = this.device.createBuffer({ size: align4(outputBytes), usage: bufferUsage.STORAGE | bufferUsage.COPY_SRC });
    const weightBuffer = createGpuStorageBuffer(this.device, tensor.packedWeight);
    const scaleBuffer = createGpuStorageBuffer(this.device, f32ScalesFromF16(tensor.rowScalesF16));
    const biasBuffer = createGpuStorageBuffer(this.device, this.denseF32Tensor(biasName));
    const paramsBuffer = this.device.createBuffer({ size: 32, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([seqLen, channels, kernel, padding, groups, groupIn, rowSize, 0]));
    const pipeline = this.q4Conv1dComputePipeline();
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source.buffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: scaleBuffer } },
        { binding: 3, resource: { buffer: biasBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(channels / 8), Math.ceil(seqLen / 8), 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return new WebGPUTensorF32(this.device, outputBuffer, seqLen * channels, seqLen, channels).readback();
  }

  async runQ4DepthwiseConv1dAsync(weightName, biasName, input, seqLen, channels, kernel, padding) {
    return this.runQ4GroupedConv1dAsync(weightName, biasName, input, seqLen, channels, kernel, padding, channels);
  }

  async runGatedAddRowsAsync(input, src, gate, rows, cols) {
    const inputTensor = this.uploadF32Tensor(input, rows, cols);
    const srcTensor = this.uploadF32Tensor(src, rows, cols);
    if (inputTensor.length !== rows * cols || srcTensor.length !== rows * cols) {
      throw new Error(`WebGPU gated add length mismatch: expected ${rows * cols}, got ${inputTensor.length}/${srcTensor.length}`);
    }
    const bufferUsage = gpuBufferUsage();
    const outputBuffer = this.device.createBuffer({ size: align4(inputTensor.length * Float32Array.BYTES_PER_ELEMENT), usage: bufferUsage.STORAGE | bufferUsage.COPY_SRC });
    const gateBuffer = createGpuStorageBuffer(this.device, gate instanceof Float32Array ? gate : new Float32Array(gate));
    const paramsBuffer = this.device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rows, cols, 0, 0]));
    const pipeline = this.gatedAddComputePipeline();
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputTensor.buffer } },
        { binding: 1, resource: { buffer: srcTensor.buffer } },
        { binding: 2, resource: { buffer: gateBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((rows * cols) / 256), 1, 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return new WebGPUTensorF32(this.device, outputBuffer, rows * cols, rows, cols);
  }

  async runRotaryAttentionAsync(q, k, v, seqLen, heads, headDim) {
    const dim = heads * headDim;
    const expected = seqLen * dim;
    if (q.length !== expected || k.length !== expected || v.length !== expected) {
      throw new Error(`WebGPU attention length mismatch: expected ${expected}, got ${q.length}/${k.length}/${v.length}`);
    }
    const bufferUsage = gpuBufferUsage();
    const qBuffer = createGpuStorageBuffer(this.device, q instanceof Float32Array ? q : new Float32Array(q));
    const kBuffer = createGpuStorageBuffer(this.device, k instanceof Float32Array ? k : new Float32Array(k));
    const vBuffer = createGpuStorageBuffer(this.device, v instanceof Float32Array ? v : new Float32Array(v));
    const outputBytes = expected * Float32Array.BYTES_PER_ELEMENT;
    const outputBuffer = this.device.createBuffer({ size: align4(outputBytes), usage: bufferUsage.STORAGE | bufferUsage.COPY_SRC });
    const paramsBuffer = this.device.createBuffer({ size: 16, usage: bufferUsage.UNIFORM | bufferUsage.COPY_DST });
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([seqLen, heads, headDim, dim]));
    const pipeline = this.rotaryAttentionComputePipeline();
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: kBuffer } },
        { binding: 2, resource: { buffer: vBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(heads / 8), 1);
    pass.end();
    const out = new WebGPUTensorF32(this.device, outputBuffer, expected, seqLen, dim);
    this.device.queue.submit([encoder.finish()]);
    return out.readback();
  }

  static async fromManifestUrl(manifestUrl, options = {}) {
    if (!globalThis.navigator?.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter request failed');
    const device = options.device || await adapter.requestDevice();
    const shaderModule = device.createShaderModule({ code: q4ShaderSource() });
    const descriptor = { layout: 'auto', compute: { module: shaderModule, entryPoint: 'q4_linear_main' } };
    const pipeline = typeof device.createComputePipelineAsync === 'function'
      ? await device.createComputePipelineAsync(descriptor)
      : device.createComputePipeline(descriptor);
    const baseBundle = await Q4TensorBundleWASM.fromManifestUrl(manifestUrl);
    return new Q4TensorBundleWebGPU(baseBundle, device, pipeline);
  }

  q4Tensor(name) { return this.base.q4Tensor(name); }
  denseTensor(name) { return this.base.denseTensor(name); }
  denseF32Tensor(name) { return this.base.denseF32Tensor(name); }
  runAttention(...args) { return this.base.runAttention(...args); }
  runLayerNormAffine(...args) { return this.base.runLayerNormAffine(...args); }
  runGatedAddRows(...args) { return this.base.runGatedAddRows(...args); }
  runQ4DepthwiseConv1d(...args) { return this.base.runQ4DepthwiseConv1d(...args); }
  runQ4Conv1d(...args) { return this.base.runQ4Conv1d(...args); }
  runQ4GroupedConv1d(...args) { return this.base.runQ4GroupedConv1d(...args); }
  runVocosIstftHead(...args) { return this.base.runVocosIstftHead(...args); }
  prepareF5Session() {
    if (!this.f5GpuSession) {
      this.f5GpuSession = {
        backend: 'webgpu',
        modelId: this.manifest?.model_id || 'f5tts-q4',
        linearKernel: 'rowwise-q4-f32',
        sharedQkvInputUpload: true,
        fusedLinearTriples: false,
        residentWeights: true,
        gpuResidentMlp: true,
        fusedMlpActivation: 'webgpu-gelu',
        gpuLayerNormAffine: true,
        gpuRotaryAttention: true,
        gpuGatedResidual: true,
        gpuQ4Conv1d: true,
        remainingCpuOps: ['text-embedding', 'sampler-update'],
      };
      this.prewarmF5LinearHandles();
    }
    return this.f5GpuSession;
  }

  prewarmF5LinearHandles() {
    for (const name of Object.keys(this.q4Index || {})) {
      if (!name.startsWith('transformer.')) continue;
      if (!name.endsWith('.weight')) continue;
      const biasName = name.slice(0, -'.weight'.length) + '.bias';
      this.q4LinearHandle(name, this.denseIndex?.[biasName] ? biasName : '');
    }
  }

  uploadF32Tensor(input, rows = 1, cols = 0) {
    if (input instanceof WebGPUTensorF32) return input;
    const x = input instanceof Float32Array ? input : new Float32Array(input);
    const bufferUsage = gpuBufferUsage();
    const buffer = this.device.createBuffer({ size: align4(x.byteLength), usage: bufferUsage.STORAGE | bufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, x.buffer, x.byteOffset, x.byteLength);
    return new WebGPUTensorF32(this.device, buffer, x.length, rows, cols || Math.floor(x.length / Math.max(1, rows)));
  }

  q4LinearHandle(name, biasName = '') {
    const key = `${name}:${biasName || ''}`;
    let handle = this.q4LinearHandleCache.get(key);
    if (handle) return handle;
    const tensor = this.q4Tensor(name);
    const shape = tensor.entry.shape.map(Number);
    const outDim = shape[0];
    const inDim = shape.slice(1).reduce((acc, value) => acc * value, 1);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    handle = new Q4LinearWebGPUHandle(this.device, this.pipeline, tensor, bias, inDim, outDim);
    this.q4LinearHandleCache.set(key, handle);
    return handle;
  }

  async runQ4LinearGpu(name, input, rows = 1, biasName = '') {
    return this.q4LinearHandle(name, biasName).forwardGpu(input, rows);
  }

  async runQ4LinearAsync(name, input, rows = 1, biasName = '') {
    if (input instanceof WebGPUTensorF32) return (await this.runQ4LinearGpu(name, input, rows, biasName)).readback();
    return this.q4LinearHandle(name, biasName).forward(input instanceof Float32Array ? input : new Float32Array(input), rows);
  }

  async runQ4Linear3Async(first, second, third, input, rows = 1) {
    const sharedInput = this.uploadF32Tensor(input, rows);
    const [aGpu, bGpu, cGpu] = await Promise.all([
      this.runQ4LinearGpu(first.weightName, sharedInput, rows, first.biasName || ''),
      this.runQ4LinearGpu(second.weightName, sharedInput, rows, second.biasName || ''),
      this.runQ4LinearGpu(third.weightName, sharedInput, rows, third.biasName || ''),
    ]);
    const [a, b, c] = await Promise.all([aGpu.readback(), bGpu.readback(), cGpu.readback()]);
    const out = new Float32Array(a.length + b.length + c.length);
    out.set(a, 0);
    out.set(b, a.length);
    out.set(c, a.length + b.length);
    return out;
  }

  async runQ4MlpAsync(first, second, input, rows = 1, activation = 'gelu') {
    const hidden = await this.runQ4LinearGpu(first.weightName, input, rows, first.biasName || '');
    this.runActivationInPlace(hidden, activation);
    return (await this.runQ4LinearGpu(second.weightName, hidden, rows, second.biasName || '')).readback();
  }
}

export class Q4TensorBundleWASM {
  constructor(bundle) {
    this.manifest = bundle.manifest;
    this.q4Index = bundle.q4Index;
    this.denseIndex = bundle.denseIndex;
    this.q4Buffer = bundle.q4Buffer;
    this.denseBuffer = bundle.denseBuffer;
    this.wasm = bundle.wasm;
    this.q4TensorCache = new Map();
    this.q4LinearHandleCache = new Map();
    this.denseTensorCache = new Map();
    this.denseF32TensorCache = new Map();
    this.f5SessionCache = null;
  }

  static async fromManifestUrl(manifestUrl) {
    const resolvedManifestUrl = new URL(manifestUrl, globalThis.location?.href || import.meta.url).href;
    const manifest = await fetchJson(resolvedManifestUrl, "model manifest");
    const baseUrl = new URL(".", resolvedManifestUrl).toString();
    if (manifest.files?.index && manifest.files?.tensors && !manifest.files?.q4) {
      const [wasm, denseIndex, denseBuffer] = await Promise.all([
        ensureQ4Wasm(),
        fetchJson(resolveUrl(manifest.files.index, baseUrl), "dense tensor index"),
        fetchBuffer(resolveUrl(manifest.files.tensors, baseUrl), "dense tensor buffer"),
      ]);
      return new Q4TensorBundleWASM({
        manifest,
        q4Index: {},
        denseIndex,
        q4Buffer: new ArrayBuffer(0),
        denseBuffer,
        wasm,
      });
    }
    const [wasm, q4Index, denseIndex, q4Buffer, denseBuffer] = await Promise.all([
      ensureQ4Wasm(),
      fetchJson(resolveUrl(manifest.files.q4_index, baseUrl), "Q4 tensor index"),
      fetchJson(resolveUrl(manifest.files.dense_index, baseUrl), "dense tensor index"),
      Array.isArray(manifest.files.q4_chunks)
        ? fetchChunkedBuffer(baseUrl, manifest.files.q4_chunks, "Q4 tensor buffer", manifest.files.q4_nbytes)
        : fetchBuffer(resolveUrl(manifest.files.q4, baseUrl), "Q4 tensor buffer"),
      fetchBuffer(resolveUrl(manifest.files.dense, baseUrl), "dense tensor buffer"),
    ]);
    return new Q4TensorBundleWASM({ manifest, q4Index, denseIndex, q4Buffer, denseBuffer, wasm });
  }

  q4Tensor(name) {
    const cached = this.q4TensorCache.get(name);
    if (cached) return cached;
    const entry = this.q4Index[name];
    if (!entry) {
      throw new Error(`Q4 tensor not found: ${name}`);
    }
    const tensor = {
      entry,
      packedWeight: tensorView(this.q4Buffer, entry.offset, entry.nbytes, Uint8Array),
      rowScalesF16: tensorView(this.q4Buffer, entry.scale_offset, entry.scale_nbytes, Uint16Array),
    };
    this.q4TensorCache.set(name, tensor);
    return tensor;
  }

  denseTensor(name) {
    const cached = this.denseTensorCache.get(name);
    if (cached) return cached;
    const entry = this.denseIndex[name];
    if (!entry) {
      throw new Error(`dense tensor not found: ${name}`);
    }
    let tensor;
    if (entry.dtype === "float16") {
      tensor = tensorView(this.denseBuffer, entry.offset, entry.nbytes, Uint16Array);
    } else if (entry.dtype === "float32") {
      tensor = tensorView(this.denseBuffer, entry.offset, entry.nbytes, Float32Array);
    } else if (entry.dtype === "bool_u8" || entry.dtype === "uint8") {
      tensor = tensorView(this.denseBuffer, entry.offset, entry.nbytes, Uint8Array);
    } else if (entry.dtype === "int64") {
      tensor = tensorView(this.denseBuffer, entry.offset, entry.nbytes, BigInt64Array);
    } else {
      throw new Error(`unsupported dense dtype for ${name}: ${entry.dtype}`);
    }
    this.denseTensorCache.set(name, tensor);
    return tensor;
  }

  denseF32Tensor(name) {
    const cached = this.denseF32TensorCache.get(name);
    if (cached) return cached;
    const entry = this.denseIndex[name];
    const raw = this.denseTensor(name);
    if (entry.dtype === "float32") {
      this.denseF32TensorCache.set(name, raw);
      return raw;
    }
    if (entry.dtype !== "float16") {
      throw new Error(`dense tensor is not floating point: ${name}`);
    }
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) {
      out[i] = f16ToF32(raw[i]);
    }
    this.denseF32TensorCache.set(name, out);
    return out;
  }

  q4LinearHandle(name, biasName = "") {
    if (!this.wasm?.Q4LinearHandle) {
      return null;
    }
    const key = `${name}:${biasName || ""}`;
    let handle = this.q4LinearHandleCache.get(key);
    if (handle) {
      return handle;
    }
    const { entry, packedWeight, rowScalesF16 } = this.q4Tensor(name);
    const shape = entry.shape;
    const outDim = Number(shape[0]);
    const inDim = Number(shape.slice(1).reduce((acc, value) => acc * Number(value), 1));
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    handle = new this.wasm.Q4LinearHandle(packedWeight, rowScalesF16, bias, inDim, outDim);
    this.q4LinearHandleCache.set(key, handle);
    return handle;
  }

  runQ4Linear(name, input, rows = 1, biasName = "") {
    if (this.wasm?.Q4LinearHandle) {
      const handle = this.q4LinearHandle(name, biasName);
      return handle.forward(input instanceof Float32Array ? input : new Float32Array(input), rows);
    }
    const { entry, packedWeight, rowScalesF16 } = this.q4Tensor(name);
    const shape = entry.shape;
    const outDim = Number(shape[0]);
    const inDim = Number(shape.slice(1).reduce((acc, value) => acc * Number(value), 1));
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return this.wasm.q4_symmetric_linear_f32(
      input instanceof Float32Array ? input : new Float32Array(input),
      packedWeight,
      rowScalesF16,
      bias,
      rows,
      inDim,
      outDim,
    );
  }

  runQ4Linear3(first, second, third, input, rows = 1) {
    if (!this.wasm?.q4_linear3_f32 || !this.wasm?.Q4LinearHandle) {
      throw new Error("q4_linear3_f32 is not available in the WASM runtime");
    }
    return this.wasm.q4_linear3_f32(
      this.q4LinearHandle(first.weightName, first.biasName || ""),
      this.q4LinearHandle(second.weightName, second.biasName || ""),
      this.q4LinearHandle(third.weightName, third.biasName || ""),
      input instanceof Float32Array ? input : new Float32Array(input),
      rows,
    );
  }

  runQ4Mlp(first, second, input, rows = 1, activation = "gelu") {
    if (!this.wasm?.q4_mlp_f32 || !this.wasm?.Q4LinearHandle) {
      throw new Error("q4_mlp_f32 is not available in the WASM runtime");
    }
    return this.wasm.q4_mlp_f32(
      this.q4LinearHandle(first.weightName, first.biasName || ""),
      this.q4LinearHandle(second.weightName, second.biasName || ""),
      input instanceof Float32Array ? input : new Float32Array(input),
      rows,
      activation,
    );
  }

  runF5DiTBlock(block, input, timeEmbedding, seqLen, dim, heads, headDim, eps = 1e-6) {
    if (!this.wasm?.f5_dit_block_f32 || !this.wasm?.Q4LinearHandle) {
      throw new Error("f5_dit_block_f32 is not available in the WASM runtime");
    }
    const prefix = `transformer.transformer_blocks.${block}`;
    return this.wasm.f5_dit_block_f32(
      this.q4LinearHandle(`${prefix}.attn_norm.linear.weight`, `${prefix}.attn_norm.linear.bias`),
      this.q4LinearHandle(`${prefix}.attn.to_q.weight`, `${prefix}.attn.to_q.bias`),
      this.q4LinearHandle(`${prefix}.attn.to_k.weight`, `${prefix}.attn.to_k.bias`),
      this.q4LinearHandle(`${prefix}.attn.to_v.weight`, `${prefix}.attn.to_v.bias`),
      this.q4LinearHandle(`${prefix}.attn.to_out.0.weight`, `${prefix}.attn.to_out.0.bias`),
      this.q4LinearHandle(`${prefix}.ff.ff.0.0.weight`, `${prefix}.ff.ff.0.0.bias`),
      this.q4LinearHandle(`${prefix}.ff.ff.2.weight`, `${prefix}.ff.ff.2.bias`),
      input instanceof Float32Array ? input : new Float32Array(input),
      timeEmbedding instanceof Float32Array ? timeEmbedding : new Float32Array(timeEmbedding),
      seqLen,
      dim,
      heads,
      headDim,
      eps,
    );
  }

  f5Session() {
    if (this.f5SessionCache) {
      return this.f5SessionCache;
    }
    if (!this.wasm?.F5Q4DiTSession) {
      return null;
    }
    const session = new this.wasm.F5Q4DiTSession();
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

  runF5Forward({ x, cond, textIds, time = 0.5, dropAudioCond = false, dropText = false }) {
    const session = this.f5Session();
    if (!session) {
      throw new Error("F5Q4DiTSession is not available in the WASM runtime");
    }
    return session.forward(
      x instanceof Float32Array ? x : new Float32Array(x),
      cond instanceof Float32Array ? cond : new Float32Array(cond),
      textIds instanceof Int32Array ? textIds : new Int32Array(textIds),
      time,
      Boolean(dropAudioCond),
      Boolean(dropText),
    );
  }

  runF5SampleMel({ condMel, condSeqLen, textIds, duration, steps, cfgStrength, swaySamplingCoef = -1.0, seed = 1337, onProgress = null }) {
    const session = this.f5Session();
    if (!session) {
      throw new Error("F5Q4DiTSession is not available in the WASM runtime");
    }
    const args = [
      condMel instanceof Float32Array ? condMel : new Float32Array(condMel),
      condSeqLen,
      textIds instanceof Int32Array ? textIds : new Int32Array(textIds),
      duration,
      steps,
      cfgStrength,
      swaySamplingCoef,
      seed,
    ];
    if (typeof onProgress === "function" && typeof session.sample_mel_with_progress === "function") {
      return session.sample_mel_with_progress(...args, onProgress);
    }
    return session.sample_mel(...args);
  }

  runAttention(q, k, v, qLen, kvLen, heads, headDim, causal = false, pastLen = 0) {
    if (!this.wasm?.attention_f32) {
      throw new Error("attention_f32 is not available in the WASM runtime");
    }
    return this.wasm.attention_f32(
      q instanceof Float32Array ? q : new Float32Array(q),
      k instanceof Float32Array ? k : new Float32Array(k),
      v instanceof Float32Array ? v : new Float32Array(v),
      qLen,
      kvLen,
      heads,
      headDim,
      Boolean(causal),
      pastLen,
    );
  }

  runLayerNormAffine(input, shift, scale, rows, cols, eps = 1e-6) {
    if (!this.wasm?.layer_norm_affine_f32) {
      throw new Error("layer_norm_affine_f32 is not available in the WASM runtime");
    }
    return this.wasm.layer_norm_affine_f32(
      input instanceof Float32Array ? input : new Float32Array(input),
      shift instanceof Float32Array ? shift : new Float32Array(shift),
      scale instanceof Float32Array ? scale : new Float32Array(scale),
      rows,
      cols,
      eps,
    );
  }

  runGatedAddRows(input, src, gate, rows, cols) {
    if (!this.wasm?.gated_add_rows_f32) {
      throw new Error("gated_add_rows_f32 is not available in the WASM runtime");
    }
    return this.wasm.gated_add_rows_f32(
      input instanceof Float32Array ? input : new Float32Array(input),
      src instanceof Float32Array ? src : new Float32Array(src),
      gate instanceof Float32Array ? gate : new Float32Array(gate),
      rows,
      cols,
    );
  }

  runQ4DepthwiseConv1d(weightName, biasName, input, seqLen, channels, kernel, padding) {
    if (!this.wasm?.q4_depthwise_conv1d_f32) {
      throw new Error("q4_depthwise_conv1d_f32 is not available in the WASM runtime");
    }
    const { packedWeight, rowScalesF16 } = this.q4Tensor(weightName);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return this.wasm.q4_depthwise_conv1d_f32(
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

  runQ4Conv1d(weightName, biasName, input, seqLen, inChannels, outChannels, kernel, padding) {
    if (!this.wasm?.q4_conv1d_f32) {
      throw new Error("q4_conv1d_f32 is not available in the WASM runtime");
    }
    const { packedWeight, rowScalesF16 } = this.q4Tensor(weightName);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return this.wasm.q4_conv1d_f32(
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

  runQ4GroupedConv1d(weightName, biasName, input, seqLen, channels, kernel, padding, groups) {
    if (!this.wasm?.q4_grouped_conv1d_f32) {
      throw new Error("q4_grouped_conv1d_f32 is not available in the WASM runtime");
    }
    const { packedWeight, rowScalesF16 } = this.q4Tensor(weightName);
    const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
    return this.wasm.q4_grouped_conv1d_f32(
      input instanceof Float32Array ? input : new Float32Array(input),
      packedWeight,
      rowScalesF16,
      bias,
      seqLen,
      channels,
      kernel,
      padding,
      groups,
    );
  }

  runVocosIstftHead(stftRows, frames) {
    if (!this.wasm?.vocos_istft_head_f32) {
      throw new Error("vocos_istft_head_f32 is not available in the WASM runtime");
    }
    return this.wasm.vocos_istft_head_f32(
      stftRows instanceof Float32Array ? stftRows : new Float32Array(stftRows),
      frames,
    );
  }
}

function f16ToF32(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x03ff;
  if (exp === 0) {
    return sign * (frac ? 2 ** -14 * (frac / 1024) : 0);
  }
  if (exp === 0x1f) {
    return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}
