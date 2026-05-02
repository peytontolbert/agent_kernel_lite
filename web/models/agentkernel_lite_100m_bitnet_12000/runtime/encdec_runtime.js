import { BitNetLinearWebGPU } from "./bitnet_webgpu.js";
import { BitNetLinearWASM } from "./bitnet_wasm_runtime.js";

function resolveUrl(path, baseUrl) {
  return new URL(path, baseUrl).toString();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(url, options = {}) {
  const attempts = Math.max(1, Number(options.attempts || 5));
  let lastError = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
      if (response.status < 500 && response.status !== 408 && response.status !== 429) {
        throw new Error(`failed to fetch ${url}: ${response.status}`);
      }
      lastError = new Error(`failed to fetch ${url}: ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    if (attempt < attempts - 1) {
      await sleep(Math.min(2000, 150 * 2 ** attempt));
    }
  }
  throw lastError || new Error(`failed to fetch ${url}`);
}

async function fetchJson(url) {
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchFloatTensor(entry, baseUrl) {
  const url = resolveUrl(entry.path, baseUrl);
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${entry.path}: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  const dtype = String(entry.dtype || "float32").toLowerCase();
  if (dtype === "float16" || dtype === "fp16" || dtype === "f16") {
    return { data: float16ArrayToFloat32(new Uint16Array(buffer)), shape: entry.shape };
  }
  return { data: new Float32Array(buffer), shape: entry.shape };
}

function float16ArrayToFloat32(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    out[i] = float16ToFloat32(values[i]);
  }
  return out;
}

function float16ToFloat32(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    return fraction === 0 ? sign * 0 : sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function zeros(length) {
  return new Float32Array(length);
}

function addInPlace(dst, src) {
  for (let i = 0; i < dst.length; i += 1) {
    dst[i] += src[i];
  }
  return dst;
}

function appendRows(existing, next) {
  if (!existing || existing.length === 0) return next.slice();
  const out = new Float32Array(existing.length + next.length);
  out.set(existing, 0);
  out.set(next, existing.length);
  return out;
}

function layerNorm(x, rows, cols, weight, bias, eps = 1e-5) {
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r += 1) {
    let mean = 0;
    for (let c = 0; c < cols; c += 1) mean += x[r * cols + c];
    mean /= cols;
    let variance = 0;
    for (let c = 0; c < cols; c += 1) {
      const d = x[r * cols + c] - mean;
      variance += d * d;
    }
    const inv = 1 / Math.sqrt(variance / cols + eps);
    for (let c = 0; c < cols; c += 1) {
      out[r * cols + c] = (x[r * cols + c] - mean) * inv * weight[c] + (bias ? bias[c] : 0);
    }
  }
  return out;
}

function rmsNorm(x, rows, cols, weight, eps = 1e-6) {
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r += 1) {
    let meanSq = 0;
    const rowOffset = r * cols;
    for (let c = 0; c < cols; c += 1) {
      const value = x[rowOffset + c];
      meanSq += value * value;
    }
    const inv = 1 / Math.sqrt(meanSq / cols + eps);
    for (let c = 0; c < cols; c += 1) {
      out[rowOffset + c] = x[rowOffset + c] * inv * weight[c];
    }
  }
  return out;
}

function silu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    out[i] = x[i] / (1 + Math.exp(-x[i]));
  }
  return out;
}

function gelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
  }
  return out;
}

function activate(x, name) {
  const normalized = String(name || "silu").toLowerCase();
  if (normalized === "gelu") return gelu(x);
  return silu(x);
}

function gatedActivation(x, rows, cols, name) {
  const out = new Float32Array(rows * cols);
  const gateName = String(name || "swiglu").toLowerCase();
  for (let row = 0; row < rows; row += 1) {
    const inputOffset = row * cols * 2;
    const outputOffset = row * cols;
    for (let i = 0; i < cols; i += 1) {
      const a = x[inputOffset + i];
      const b = x[inputOffset + cols + i];
      const activated = gateName === "geglu"
        ? 0.5 * a * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (a + 0.044715 * a * a * a)))
        : a / (1 + Math.exp(-a));
      out[outputOffset + i] = activated * b;
    }
  }
  return out;
}

function embed(tokens, embedding, dModel) {
  const out = new Float32Array(tokens.length * dModel);
  for (let t = 0; t < tokens.length; t += 1) {
    const token = tokens[t];
    out.set(embedding.subarray(token * dModel, token * dModel + dModel), t * dModel);
  }
  return out;
}

function addPositionEmbeddingInPlace(x, positionEmbedding, dModel) {
  if (!positionEmbedding) return x;
  for (let t = 0; t < x.length / dModel; t += 1) {
    const src = t * dModel;
    for (let c = 0; c < dModel; c += 1) {
      x[src + c] += positionEmbedding[src + c];
    }
  }
  return x;
}

function traceTensor(name, tensor, shape) {
  let maxAbs = 0;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < tensor.length; i += 1) {
    const value = Number(tensor[i]);
    const abs = Math.abs(value);
    if (abs > maxAbs) maxAbs = abs;
    sum += value;
    sumSq += value * value;
  }
  return {
    name,
    shape,
    len: tensor.length,
    maxAbs,
    mean: tensor.length ? sum / tensor.length : 0,
    rms: tensor.length ? Math.sqrt(sumSq / tensor.length) : 0,
    values: Array.from(tensor),
  };
}

class DenseLinear {
  constructor(name, weightTensor, biasTensor = null) {
    if (!weightTensor?.data || !Array.isArray(weightTensor.shape) || weightTensor.shape.length !== 2) {
      throw new Error(`dense linear ${name} is missing a rank-2 weight tensor`);
    }
    this.name = name;
    this.weight = weightTensor.data;
    this.bias = biasTensor?.data || null;
    this.outFeatures = Number(weightTensor.shape[0]);
    this.inFeatures = Number(weightTensor.shape[1]);
    this.layout = {
      logicalIn: this.inFeatures,
      logicalOut: this.outFeatures,
    };
  }

  async run(input, rows) {
    const rowCount = Number(rows || 0);
    if (rowCount <= 0) return new Float32Array(0);
    if (input.length < rowCount * this.inFeatures) {
      throw new Error(
        `dense linear ${this.name} input too small: got ${input.length}, expected ${rowCount * this.inFeatures}`,
      );
    }
    const out = new Float32Array(rowCount * this.outFeatures);
    for (let r = 0; r < rowCount; r += 1) {
      const inputOffset = r * this.inFeatures;
      const outputOffset = r * this.outFeatures;
      for (let o = 0; o < this.outFeatures; o += 1) {
        let sum = this.bias ? this.bias[o] : 0;
        const weightOffset = o * this.inFeatures;
        for (let i = 0; i < this.inFeatures; i += 1) {
          sum += input[inputOffset + i] * this.weight[weightOffset + i];
        }
        out[outputOffset + o] = sum;
      }
    }
    return out;
  }
}

function splitHeads(x, seqLen, nHeads, headDim) {
  const out = [];
  for (let h = 0; h < nHeads; h += 1) {
    const head = new Float32Array(seqLen * headDim);
    for (let t = 0; t < seqLen; t += 1) {
      const src = t * nHeads * headDim + h * headDim;
      head.set(x.subarray(src, src + headDim), t * headDim);
    }
    out.push(head);
  }
  return out;
}

function mergeHeads(heads, seqLen, nHeads, headDim) {
  const out = new Float32Array(seqLen * nHeads * headDim);
  for (let h = 0; h < nHeads; h += 1) {
    for (let t = 0; t < seqLen; t += 1) {
      out.set(heads[h].subarray(t * headDim, t * headDim + headDim), t * nHeads * headDim + h * headDim);
    }
  }
  return out;
}

function attention(q, k, v, qLen, kvLen, nHeads, headDim, causal) {
  const qh = splitHeads(q, qLen, nHeads, headDim);
  const kh = splitHeads(k, kvLen, nHeads, headDim);
  const vh = splitHeads(v, kvLen, nHeads, headDim);
  const outHeads = [];
  const scale = 1 / Math.sqrt(headDim);
  for (let h = 0; h < nHeads; h += 1) {
    const out = new Float32Array(qLen * headDim);
    for (let i = 0; i < qLen; i += 1) {
      const scores = new Float32Array(kvLen);
      let maxScore = -Infinity;
      for (let j = 0; j < kvLen; j += 1) {
        let score = causal && j > i ? -1e30 : 0;
        if (score > -1e20) {
          for (let d = 0; d < headDim; d += 1) {
            score += qh[h][i * headDim + d] * kh[h][j * headDim + d] * scale;
          }
        }
        scores[j] = score;
        maxScore = Math.max(maxScore, score);
      }
      let denom = 0;
      for (let j = 0; j < kvLen; j += 1) {
        scores[j] = Math.exp(scores[j] - maxScore);
        denom += scores[j];
      }
      for (let d = 0; d < headDim; d += 1) {
        let sum = 0;
        for (let j = 0; j < kvLen; j += 1) {
          sum += (scores[j] / denom) * vh[h][j * headDim + d];
        }
        out[i * headDim + d] = sum;
      }
    }
    outHeads.push(out);
  }
  return mergeHeads(outHeads, qLen, nHeads, headDim);
}

function decoderUsesRotary(manifest, graph) {
  const positional = String(manifest?.model?.positional || graph?.positional || "").toLowerCase();
  return positional === "apply_rotary" || positional === "rotary" || positional === "rope";
}

function rotaryBase(manifest, graph) {
  const value = Number(manifest?.model?.rope_theta || graph?.rope_theta || 1000000);
  return Number.isFinite(value) && value > 0 ? value : 1000000;
}

function applyRotaryMergedInPlace(q, k, seqLen, nHeads, headDim, baseTheta, startPosition = 0) {
  if (headDim % 2 !== 0) throw new Error("RoPE head_dim must be even");
  const half = headDim / 2;
  const invFreq = new Float32Array(half);
  for (let i = 0; i < half; i += 1) {
    invFreq[i] = 1 / (baseTheta ** ((2 * i) / headDim));
  }
  for (let t = 0; t < seqLen; t += 1) {
    const position = startPosition + t;
    for (let h = 0; h < nHeads; h += 1) {
      const baseOffset = t * nHeads * headDim + h * headDim;
      for (let i = 0; i < half; i += 1) {
        const angle = position * invFreq[i];
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const left = baseOffset + i;
        const right = baseOffset + i + half;
        const q1 = q[left];
        const q2 = q[right];
        const k1 = k[left];
        const k2 = k[right];
        q[left] = q1 * cos - q2 * sin;
        q[right] = q2 * cos + q1 * sin;
        k[left] = k1 * cos - k2 * sin;
        k[right] = k2 * cos + k1 * sin;
      }
    }
  }
  return [q, k];
}

export class BitNetEncoderDecoderWebGPU {
  constructor(device, manifest, manifestUrl, denseTensors, linears) {
    if (manifest.graph?.architecture !== "encoder_decoder") {
      throw new Error("manifest is not an encoder_decoder browser BitNet bundle");
    }
    this.device = device;
    this.manifest = manifest;
    this.manifestUrl = manifestUrl;
    this.dense = denseTensors;
    this.linears = linears;
    this.denseLinears = {};
    this.graph = manifest.graph;
    this.decoderRotary = decoderUsesRotary(manifest, this.graph);
    this.decoderRotaryBase = rotaryBase(manifest, this.graph);
  }

  static async fromManifestUrl(device, manifestUrl, options = {}) {
    const progress = typeof options.progress === "function" ? options.progress : () => {};
    progress({ phase: "manifest", message: "Loading model manifest" });
    const manifest = options.manifest || await fetchJson(manifestUrl);
    const baseUrl = new URL(".", manifestUrl).toString();
    const dense = {};
    const denseEntries = Object.entries(manifest.dense_tensors || {});
    for (const [index, [name, entry]] of denseEntries.entries()) {
      progress({
        phase: "dense",
        index: index + 1,
        total: denseEntries.length,
        name,
        message: `Loading dense tensor ${index + 1}/${denseEntries.length}: ${name}`,
      });
      dense[name] = await fetchFloatTensor(entry, baseUrl);
    }
    progress({
      phase: "dense_ready",
      index: denseEntries.length,
      total: denseEntries.length,
      message: "Dense tensors ready",
    });
    const linears = {};
    const layers = manifest.layers || [];
    const layerConcurrency = Math.max(1, Math.min(Number(options.layerConcurrency || 4), layers.length || 1));
    progress({
      phase: "prepare_layers",
      index: 0,
      total: layers.length,
      message: `Preparing ${layers.length} BitNet layers (${layerConcurrency} parallel)`,
    });
    let nextLayer = 0;
    let completedLayers = 0;
    async function loadLayerWorker() {
      while (nextLayer < layers.length) {
        const index = nextLayer;
        nextLayer += 1;
        const layer = layers[index];
        progress({
          phase: "layer",
          index: index + 1,
          total: layers.length,
          name: layer.name,
          message: `Loading BitNet layer ${index + 1}/${layers.length}: ${layer.name}`,
        });
        linears[layer.name] = await BitNetLinearWebGPU.fromManifestLayer(device, manifest, layer, manifestUrl, {
          progress,
          index: index + 1,
          total: layers.length,
          name: layer.name,
        });
        completedLayers += 1;
        progress({
          phase: "layer_ready",
          index: completedLayers,
          total: layers.length,
          name: layer.name,
          message: `BitNet layer ${completedLayers}/${layers.length} ready: ${layer.name}`,
        });
      }
    }
    await Promise.all(Array.from({ length: Math.min(layerConcurrency, layers.length) }, () => loadLayerWorker()));
    progress({ phase: "ready", message: "Model runtime ready" });
    return new BitNetEncoderDecoderWebGPU(device, manifest, manifestUrl, dense, linears);
  }

  linear(name) {
    const layer = this.linears[name];
    if (layer) return layer;
    if (this.denseLinears[name]) return this.denseLinears[name];
    const weight = this.dense[`${name}.weight`];
    if (!weight) throw new Error(`missing linear layer: ${name}`);
    const bias = this.dense[`${name}.bias`] || null;
    const denseLayer = new DenseLinear(name, weight, bias);
    this.denseLinears[name] = denseLayer;
    return denseLayer;
  }

  tensor(name) {
    const tensor = this.dense[name];
    if (!tensor) throw new Error(`missing dense tensor: ${name}`);
    return tensor.data;
  }

  norm(prefix, x, rows) {
    const weight = this.tensor(`${prefix}.weight`);
    const bias = this.dense[`${prefix}.bias`]?.data || null;
    if (bias) {
      return layerNorm(x, rows, this.graph.d_model, weight, bias);
    }
    return rmsNorm(
      x,
      rows,
      this.graph.d_model,
      weight,
      Number(this.manifest?.model?.rms_norm_eps || this.graph?.rms_norm_eps || 1e-6),
    );
  }

  async attentionBlock(prefix, x, seqLen, kv, kvLen, causal) {
    const dModel = this.graph.d_model;
    const nHeads = this.graph.n_heads;
    const headDim = this.graph.head_dim;
    const q = await this.linear(`${prefix}.w_q`).run(x, seqLen);
    const kInput = kv || x;
    const kRows = kvLen || seqLen;
    const k = await this.linear(`${prefix}.w_k`).run(kInput, kRows);
    const v = await this.linear(`${prefix}.w_v`).run(kInput, kRows);
    if (causal && this.decoderRotary) {
      applyRotaryMergedInPlace(q, k, seqLen, nHeads, headDim, this.decoderRotaryBase, 0);
    }
    const merged = attention(q, k, v, seqLen, kRows, nHeads, headDim, causal);
    return this.linear(`${prefix}.w_o`).run(merged, seqLen);
  }

  async selfAttentionIncremental(prefix, x, layerCache) {
    const nHeads = this.graph.n_heads;
    const headDim = this.graph.head_dim;
    const q = await this.linear(`${prefix}.w_q`).run(x, 1);
    const kNew = await this.linear(`${prefix}.w_k`).run(x, 1);
    const vNew = await this.linear(`${prefix}.w_v`).run(x, 1);
    const position = Number(layerCache.selfLen || 0);
    if (this.decoderRotary) {
      applyRotaryMergedInPlace(q, kNew, 1, nHeads, headDim, this.decoderRotaryBase, position);
    }
    layerCache.selfK = appendRows(layerCache.selfK, kNew);
    layerCache.selfV = appendRows(layerCache.selfV, vNew);
    layerCache.selfLen = Number(layerCache.selfLen || 0) + 1;
    const merged = attention(q, layerCache.selfK, layerCache.selfV, 1, layerCache.selfLen, nHeads, headDim, false);
    return this.linear(`${prefix}.w_o`).run(merged, 1);
  }

  async crossAttentionCached(prefix, x, memory, memoryLen, layerCache) {
    const nHeads = this.graph.n_heads;
    const headDim = this.graph.head_dim;
    const q = await this.linear(`${prefix}.w_q`).run(x, 1);
    if (!layerCache.crossK || !layerCache.crossV) {
      layerCache.crossK = await this.linear(`${prefix}.w_k`).run(memory, memoryLen);
      layerCache.crossV = await this.linear(`${prefix}.w_v`).run(memory, memoryLen);
    }
    const merged = attention(q, layerCache.crossK, layerCache.crossV, 1, memoryLen, nHeads, headDim, false);
    return this.linear(`${prefix}.w_o`).run(merged, 1);
  }

  async mlp(prefix, x, seqLen) {
    const wIn = this.linear(`${prefix}.w_in`);
    const wOut = this.linear(`${prefix}.w_out`);
    const hidden = await wIn.run(x, seqLen);
    const activation = String(this.graph.activation || "silu").toLowerCase();
    const isGated =
      wIn.layout.logicalOut === wOut.layout.logicalIn * 2 ||
      hidden.length === seqLen * wOut.layout.logicalIn * 2;
    const activated = isGated || ["swiglu", "gated-silu", "geglu", "reglu"].includes(activation)
      ? gatedActivation(hidden, seqLen, wOut.layout.logicalIn, activation)
      : activate(hidden, activation);
    return wOut.run(activated, seqLen);
  }

  async encoderLayer(index, x, seqLen) {
    const n1 = this.norm(`encoder.${index}.n1`, x, seqLen);
    const attnOut = await this.attentionBlock(`encoder.${index}.attn`, n1, seqLen, null, null, false);
    x = addInPlace(x.slice(), attnOut);
    const n2 = this.norm(`encoder.${index}.n2`, x, seqLen);
    return addInPlace(x, await this.mlp(`encoder.${index}.mlp`, n2, seqLen));
  }

  async decoderLayer(index, x, seqLen, memory, memoryLen) {
    let n = this.norm(`decoder.${index}.self_attn_block.n1`, x, seqLen);
    x = addInPlace(x.slice(), await this.attentionBlock(`decoder.${index}.self_attn_block.attn`, n, seqLen, null, null, true));
    n = this.norm(`decoder.${index}.self_attn_block.n2`, x, seqLen);
    x = addInPlace(x, await this.mlp(`decoder.${index}.self_attn_block.mlp`, n, seqLen));
    n = this.norm(`decoder.${index}.cross_block.n1`, x, seqLen);
    x = addInPlace(x.slice(), await this.attentionBlock(`decoder.${index}.cross_block.cross`, n, seqLen, memory, memoryLen, false));
    n = this.norm(`decoder.${index}.cross_block.n2`, x, seqLen);
    return addInPlace(x, await this.mlp(`decoder.${index}.cross_block.mlp`, n, seqLen));
  }

  async decoderLayerIncremental(index, x, memory, memoryLen, layerCache) {
    let n = this.norm(`decoder.${index}.self_attn_block.n1`, x, 1);
    x = addInPlace(
      x.slice(),
      await this.selfAttentionIncremental(`decoder.${index}.self_attn_block.attn`, n, layerCache),
    );
    n = this.norm(`decoder.${index}.self_attn_block.n2`, x, 1);
    x = addInPlace(x, await this.mlp(`decoder.${index}.self_attn_block.mlp`, n, 1));
    n = this.norm(`decoder.${index}.cross_block.n1`, x, 1);
    x = addInPlace(
      x.slice(),
      await this.crossAttentionCached(`decoder.${index}.cross_block.cross`, n, memory, memoryLen, layerCache),
    );
    n = this.norm(`decoder.${index}.cross_block.n2`, x, 1);
    return addInPlace(x, await this.mlp(`decoder.${index}.cross_block.mlp`, n, 1));
  }

  async encode(encInputIds) {
    let x = embed(encInputIds, this.tensor("enc_embed.weight"), this.graph.d_model);
    if (this.graph.encoder_position_embeddings) {
      x = addPositionEmbeddingInPlace(x, this.tensor("enc_pos_embed.weight"), this.graph.d_model);
    }
    for (let i = 0; i < this.graph.n_layers; i += 1) {
      x = await this.encoderLayer(i, x, encInputIds.length);
    }
    return layerNorm(
      x,
      encInputIds.length,
      this.graph.d_model,
      this.tensor("enc_norm.weight"),
      this.dense["enc_norm.bias"]?.data,
    );
  }

  async decode(decInputIds, memory, memoryLen) {
    let x = embed(decInputIds, this.tensor("dec_embed.weight"), this.graph.d_model);
    for (let i = 0; i < this.graph.n_layers; i += 1) {
      x = await this.decoderLayer(i, x, decInputIds.length, memory, memoryLen);
    }
    return layerNorm(
      x,
      decInputIds.length,
      this.graph.d_model,
      this.tensor("dec_norm.weight"),
      this.dense["dec_norm.bias"]?.data,
    );
  }

  async forward(encInputIds, decInputIds) {
    const memory = await this.encode(encInputIds);
    const hidden = await this.decode(decInputIds, memory, encInputIds.length);
    return this.linear("lm_head").run(hidden, decInputIds.length);
  }

  async debugTrace(encInputIds, decInputIds) {
    const traces = [];
    let x = embed(encInputIds, this.tensor("enc_embed.weight"), this.graph.d_model);
    if (this.graph.encoder_position_embeddings) {
      x = addPositionEmbeddingInPlace(x, this.tensor("enc_pos_embed.weight"), this.graph.d_model);
    }
    traces.push(traceTensor("enc_embed", x, [encInputIds.length, this.graph.d_model]));
    for (let i = 0; i < this.graph.n_layers; i += 1) {
      const n1 = this.norm(`encoder.${i}.n1`, x, encInputIds.length);
      traces.push(traceTensor(`encoder.${i}.n1`, n1, [encInputIds.length, this.graph.d_model]));
      const attnOut = await this.attentionBlock(`encoder.${i}.attn`, n1, encInputIds.length, null, null, false);
      traces.push(traceTensor(`encoder.${i}.attn`, attnOut, [encInputIds.length, this.graph.d_model]));
      x = addInPlace(x.slice(), attnOut);
      traces.push(traceTensor(`encoder.${i}.attn_resid`, x, [encInputIds.length, this.graph.d_model]));
      const n2 = this.norm(`encoder.${i}.n2`, x, encInputIds.length);
      traces.push(traceTensor(`encoder.${i}.n2`, n2, [encInputIds.length, this.graph.d_model]));
      const mlpOut = await this.mlp(`encoder.${i}.mlp`, n2, encInputIds.length);
      traces.push(traceTensor(`encoder.${i}.mlp`, mlpOut, [encInputIds.length, this.graph.d_model]));
      x = addInPlace(x, mlpOut);
      traces.push(traceTensor(`encoder.${i}`, x, [encInputIds.length, this.graph.d_model]));
    }
    const memory = layerNorm(
      x,
      encInputIds.length,
      this.graph.d_model,
      this.tensor("enc_norm.weight"),
      this.dense["enc_norm.bias"]?.data,
    );
    traces.push(traceTensor("enc_norm", memory, [encInputIds.length, this.graph.d_model]));

    let hidden = embed(decInputIds, this.tensor("dec_embed.weight"), this.graph.d_model);
    traces.push(traceTensor("dec_embed", hidden, [decInputIds.length, this.graph.d_model]));
    for (let i = 0; i < this.graph.n_layers; i += 1) {
      hidden = await this.decoderLayer(i, hidden, decInputIds.length, memory, encInputIds.length);
      traces.push(traceTensor(`decoder.${i}`, hidden, [decInputIds.length, this.graph.d_model]));
    }
    hidden = layerNorm(
      hidden,
      decInputIds.length,
      this.graph.d_model,
      this.tensor("dec_norm.weight"),
      this.dense["dec_norm.bias"]?.data,
    );
    traces.push(traceTensor("dec_norm", hidden, [decInputIds.length, this.graph.d_model]));
    const logits = await this.linear("lm_head").run(hidden, decInputIds.length);
    traces.push(traceTensor("logits", logits, [decInputIds.length, this.graph.vocab_size]));
    return { traces };
  }

  createGenerationSession(encInputIds) {
    return new BitNetEncoderDecoderGenerationSession(this, encInputIds);
  }
}

export class BitNetEncoderDecoderGenerationSession {
  constructor(runtime, encInputIds) {
    this.runtime = runtime;
    this.encInputIds = Array.from(encInputIds || [], Number);
    this.memory = null;
    this.memoryLen = this.encInputIds.length;
    this.layerCaches = Array.from({ length: runtime.graph.n_layers }, () => ({}));
  }

  async prepare() {
    if (!this.memory) {
      this.memory = await this.runtime.encode(this.encInputIds);
    }
    return this;
  }

  async next(tokenId) {
    await this.prepare();
    let x = embed([Number(tokenId)], this.runtime.tensor("dec_embed.weight"), this.runtime.graph.d_model);
    for (let i = 0; i < this.runtime.graph.n_layers; i += 1) {
      x = await this.runtime.decoderLayerIncremental(i, x, this.memory, this.memoryLen, this.layerCaches[i]);
    }
    const hidden = layerNorm(
      x,
      1,
      this.runtime.graph.d_model,
      this.runtime.tensor("dec_norm.weight"),
      this.runtime.dense["dec_norm.bias"]?.data,
    );
    return this.runtime.linear("lm_head").run(hidden, 1);
  }
}

export class BitNetEncoderDecoderWASM extends BitNetEncoderDecoderWebGPU {
  constructor(manifest, manifestUrl, denseTensors, linears) {
    super(null, manifest, manifestUrl, denseTensors, linears);
  }

  static async fromManifestUrl(manifestUrl, options = {}) {
    const progress = typeof options.progress === "function" ? options.progress : () => {};
    progress({ phase: "manifest", message: "Loading model manifest" });
    const manifest = options.manifest || await fetchJson(manifestUrl);
    const baseUrl = new URL(".", manifestUrl).toString();
    const dense = {};
    const denseEntries = Object.entries(manifest.dense_tensors || {});
    for (const [index, [name, entry]] of denseEntries.entries()) {
      progress({
        phase: "dense",
        index: index + 1,
        total: denseEntries.length,
        name,
        message: `Loading dense tensor ${index + 1}/${denseEntries.length}: ${name}`,
      });
      dense[name] = await fetchFloatTensor(entry, baseUrl);
    }
    progress({
      phase: "dense_ready",
      index: denseEntries.length,
      total: denseEntries.length,
      message: "Dense tensors ready",
    });

    const linears = {};
    const layers = manifest.layers || [];
    const layerConcurrency = Math.max(1, Math.min(Number(options.layerConcurrency || 4), layers.length || 1));
    progress({
      phase: "prepare_layers",
      index: 0,
      total: layers.length,
      message: `Preparing ${layers.length} BitNet WASM layers (${layerConcurrency} parallel)`,
    });
    let nextLayer = 0;
    let completedLayers = 0;
    async function loadLayerWorker() {
      while (nextLayer < layers.length) {
        const index = nextLayer;
        nextLayer += 1;
        const layer = layers[index];
        progress({
          phase: "layer",
          index: index + 1,
          total: layers.length,
          name: layer.name,
          message: `Loading BitNet WASM layer ${index + 1}/${layers.length}: ${layer.name}`,
        });
        linears[layer.name] = await BitNetLinearWASM.fromManifestLayer(manifest, layer, manifestUrl, {
          progress,
          index: index + 1,
          total: layers.length,
          name: layer.name,
        });
        completedLayers += 1;
        progress({
          phase: "layer_ready",
          index: completedLayers,
          total: layers.length,
          name: layer.name,
          message: `BitNet WASM layer ${completedLayers}/${layers.length} ready: ${layer.name}`,
        });
      }
    }
    await Promise.all(Array.from({ length: Math.min(layerConcurrency, layers.length) }, () => loadLayerWorker()));
    progress({ phase: "wasm_ready", message: "BitNet WASM runtime ready" });
    return new BitNetEncoderDecoderWASM(manifest, manifestUrl, dense, linears);
  }
}
