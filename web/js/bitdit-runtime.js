const TWO_PI = Math.PI * 2;
let imageWasmRuntimePromise = null;

async function loadImageWasmRuntime() {
  if (!imageWasmRuntimePromise) {
    imageWasmRuntimePromise = import('../wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core.js')
      .then(async (module) => {
        const wasmUrl = new URL('../wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core_bg.wasm', import.meta.url);
        await module.default({ module_or_path: wasmUrl });
        return module;
      })
      .catch((error) => {
        console.warn('BitDiT WASM runtime unavailable; falling back to JS kernels.', error);
        return null;
      });
  }
  return imageWasmRuntimePromise;
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

function softmax(values) {
  let max = -Infinity;
  for (const value of values) if (value > max) max = value;
  let sum = 0;
  const out = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.exp(values[index] - max);
    out[index] = value;
    sum += value;
  }
  const inv = 1 / Math.max(sum, 1e-12);
  for (let index = 0; index < out.length; index += 1) out[index] *= inv;
  return out;
}

function hashText(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function normalPair(random) {
  const u1 = Math.max(random(), 1e-7);
  const u2 = random();
  const mag = Math.sqrt(-2 * Math.log(u1));
  return [mag * Math.cos(TWO_PI * u2), mag * Math.sin(TWO_PI * u2)];
}

function normalArray(length, seed) {
  const random = mulberry32(seed);
  const out = new Float32Array(length);
  for (let index = 0; index < length; index += 2) {
    const [a, b] = normalPair(random);
    out[index] = a;
    if (index + 1 < length) out[index + 1] = b;
  }
  return out;
}

function tensorView(buffer, entry) {
  return new Float32Array(buffer, Number(entry.offset || 0), Number(entry.nbytes || 0) / 4);
}

function ternaryTensorView(buffer, entry) {
  const dtype = String(entry.dtype || '');
  if (dtype.includes('packed_2bit')) {
    return {
      values: new Uint8Array(buffer, Number(entry.offset || 0), Number(entry.nbytes || 0)),
      scales: new Float32Array(buffer, Number(entry.scale_offset || 0), Number(entry.scale_nbytes || 0) / 4),
      density: entry.density_offset == null ? null : new Float32Array(buffer, Number(entry.density_offset || 0), Number(entry.density_nbytes || 0) / 4),
      shape: entry.shape || [],
      packed2Bit: true,
    };
  }
  return {
    values: new Int8Array(buffer, Number(entry.offset || 0), Number(entry.nbytes || 0)),
    scales: new Float32Array(buffer, Number(entry.scale_offset || 0), Number(entry.scale_nbytes || 0) / 4),
    shape: entry.shape || [],
    packed2Bit: false,
  };
}

function linear(input, weight, bias, inDim, outDim) {
  const rows = input.length / inDim;
  const out = new Float32Array(rows * outDim);
  for (let row = 0; row < rows; row += 1) {
    const inputOffset = row * inDim;
    const outOffset = row * outDim;
    for (let col = 0; col < outDim; col += 1) {
      let sum = bias ? bias[col] : 0;
      const weightOffset = col * inDim;
      for (let inner = 0; inner < inDim; inner += 1) {
        sum += input[inputOffset + inner] * weight[weightOffset + inner];
      }
      out[outOffset + col] = sum;
    }
  }
  return out;
}

function linearTernary(input, tensor, bias, inDim, outDim) {
  const rows = input.length / inDim;
  const out = new Float32Array(rows * outDim);
  const weight = tensor.values;
  const scales = tensor.scales;
  for (let row = 0; row < rows; row += 1) {
    const inputOffset = row * inDim;
    const outOffset = row * outDim;
    for (let col = 0; col < outDim; col += 1) {
      let sum = bias ? bias[col] : 0;
      const weightOffset = col * inDim;
      let dot = 0;
      for (let inner = 0; inner < inDim; inner += 1) {
        const q = weight[weightOffset + inner];
        if (q) dot += input[inputOffset + inner] * q;
      }
      out[outOffset + col] = sum + dot * scales[col];
    }
  }
  return out;
}

function unpack2BitCode(weight, index) {
  const code = (weight[index >> 2] >> ((index & 3) * 2)) & 3;
  if (code === 0) return -1;
  if (code === 2) return 1;
  return 0;
}

function linearTernaryPacked2Bit(input, tensor, bias, inDim, outDim) {
  const rows = input.length / inDim;
  const out = new Float32Array(rows * outDim);
  const weight = tensor.values;
  const scales = tensor.scales;
  for (let row = 0; row < rows; row += 1) {
    const inputOffset = row * inDim;
    const outOffset = row * outDim;
    for (let col = 0; col < outDim; col += 1) {
      let sum = bias ? bias[col] : 0;
      const weightOffset = col * inDim;
      let dot = 0;
      for (let inner = 0; inner < inDim; inner += 1) {
        const q = unpack2BitCode(weight, weightOffset + inner);
        if (q) dot += input[inputOffset + inner] * q;
      }
      out[outOffset + col] = sum + dot * scales[col];
    }
  }
  return out;
}

function layerNorm(input, weight, bias, rows, dim) {
  const out = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * dim;
    let mean = 0;
    for (let index = 0; index < dim; index += 1) mean += input[offset + index];
    mean /= dim;
    let variance = 0;
    for (let index = 0; index < dim; index += 1) {
      const delta = input[offset + index] - mean;
      variance += delta * delta;
    }
    const inv = 1 / Math.sqrt(variance / dim + 1e-5);
    for (let index = 0; index < dim; index += 1) {
      out[offset + index] = (input[offset + index] - mean) * inv * weight[index] + bias[index];
    }
  }
  return out;
}

function addInPlace(target, source) {
  for (let index = 0; index < target.length; index += 1) target[index] += source[index];
}

function timestepEmbedding(timestep, dim) {
  const half = Math.floor(dim / 2);
  const out = new Float32Array(dim);
  for (let index = 0; index < half; index += 1) {
    const freq = Math.exp(-Math.log(10000) * index / Math.max(half - 1, 1));
    const value = timestep * freq;
    out[index] = Math.sin(value);
    out[index + half] = Math.cos(value);
  }
  return out;
}

function patchify(image, imageSize, patchSize, channels) {
  const side = imageSize / patchSize;
  const patchDim = channels * patchSize * patchSize;
  const patches = new Float32Array(side * side * patchDim);
  let out = 0;
  for (let py = 0; py < side; py += 1) {
    for (let px = 0; px < side; px += 1) {
      for (let c = 0; c < channels; c += 1) {
        for (let y = 0; y < patchSize; y += 1) {
          for (let x = 0; x < patchSize; x += 1) {
            patches[out] = image[((c * imageSize + py * patchSize + y) * imageSize) + px * patchSize + x];
            out += 1;
          }
        }
      }
    }
  }
  return patches;
}

function unpatchify(patches, imageSize, patchSize, channels) {
  const side = imageSize / patchSize;
  const image = new Float32Array(channels * imageSize * imageSize);
  let input = 0;
  for (let py = 0; py < side; py += 1) {
    for (let px = 0; px < side; px += 1) {
      for (let c = 0; c < channels; c += 1) {
        for (let y = 0; y < patchSize; y += 1) {
          for (let x = 0; x < patchSize; x += 1) {
            image[((c * imageSize + py * patchSize + y) * imageSize) + px * patchSize + x] = patches[input];
            input += 1;
          }
        }
      }
    }
  }
  return image;
}

function classFromPrompt(prompt, classes) {
  const text = String(prompt || '').toLowerCase();
  for (let index = 0; index < classes.length; index += 1) {
    const label = String(classes[index]).toLowerCase();
    if (text.includes(label)) return index;
  }
  if (/\bcar\b|vehicle|auto/.test(text)) return classes.indexOf('automobile');
  if (/plane|jet|aircraft/.test(text)) return classes.indexOf('airplane');
  if (/boat|ocean|sea/.test(text)) return classes.indexOf('ship');
  return hashText(text) % classes.length;
}

export class BitDiTRuntime {
  constructor(manifest, tensors, ternaryTensors = {}, wasm = null) {
    this.manifest = manifest;
    this.config = manifest.config;
    this.classes = manifest.classes || [];
    this.t = tensors;
    this.qt = ternaryTensors;
    this.wasm = wasm;
    this.backend = wasm?.image_ternary_packed_2bit_linear_f32 || wasm?.image_ternary_linear_f32 ? 'bitdit-wasm' : 'bitdit-js';
    this.imageSize = this.config.image_size;
    this.patchSize = this.config.patch_size;
    this.channels = this.config.channels;
    this.dim = this.config.dim;
    this.tokens = (this.imageSize / this.patchSize) ** 2;
    this.patchDim = this.channels * this.patchSize * this.patchSize;
    this.timesteps = this.config.timesteps;
    this.betas = new Float32Array(this.timesteps);
    this.alphas = new Float32Array(this.timesteps);
    this.alphaBars = new Float32Array(this.timesteps);
    let product = 1;
    for (let index = 0; index < this.timesteps; index += 1) {
      const beta = 1e-4 + (0.02 - 1e-4) * index / Math.max(this.timesteps - 1, 1);
      this.betas[index] = beta;
      this.alphas[index] = 1 - beta;
      product *= 1 - beta;
      this.alphaBars[index] = product;
    }
  }

  static async fromManifestUrl(manifestUrl) {
    const manifestResponse = await fetch(manifestUrl);
    if (!manifestResponse.ok) throw new Error(`Image manifest failed: ${manifestResponse.status}`);
    const manifest = await manifestResponse.json();
    return BitDiTRuntime.fromManifest(manifest, manifestResponse.url);
  }

  static async fromManifest(manifest, manifestUrl) {
    const base = new URL('.', manifestUrl);
    const ternaryEnabled = Boolean(manifest.ternary_tensors?.enabled);
    const [indexResponse, tensorResponse, ternaryIndexResponse, ternaryTensorResponse] = await Promise.all([
      fetch(new URL(manifest.tensors.index, base)),
      fetch(new URL(manifest.tensors.data, base)),
      ternaryEnabled ? fetch(new URL(manifest.ternary_tensors.index, base)) : Promise.resolve(null),
      ternaryEnabled ? fetch(new URL(manifest.ternary_tensors.data, base)) : Promise.resolve(null),
    ]);
    if (!indexResponse.ok) throw new Error(`Image tensor index failed: ${indexResponse.status}`);
    if (!tensorResponse.ok) throw new Error(`Image tensor data failed: ${tensorResponse.status}`);
    const index = await indexResponse.json();
    const buffer = await tensorResponse.arrayBuffer();
    const tensors = {};
    for (const [name, entry] of Object.entries(index)) tensors[name] = tensorView(buffer, entry);
    const ternaryTensors = {};
    if (ternaryEnabled) {
      if (!ternaryIndexResponse?.ok) throw new Error(`Image ternary index failed: ${ternaryIndexResponse?.status}`);
      if (!ternaryTensorResponse?.ok) throw new Error(`Image ternary data failed: ${ternaryTensorResponse?.status}`);
      const ternaryIndex = await ternaryIndexResponse.json();
      const ternaryBuffer = await ternaryTensorResponse.arrayBuffer();
      for (const [name, entry] of Object.entries(ternaryIndex)) ternaryTensors[name] = ternaryTensorView(ternaryBuffer, entry);
    }
    const wasm = await loadImageWasmRuntime();
    return new BitDiTRuntime(manifest, tensors, ternaryTensors, wasm);
  }

  tensor(name) {
    const tensor = this.t[name];
    if (!tensor) throw new Error(`Missing BitDiT tensor: ${name}`);
    return tensor;
  }

  linearPrefix(prefix, input, inDim, outDim) {
    const weightName = `${prefix}.weight`;
    const bias = this.t[`${prefix}.bias`];
    if (this.qt[weightName]) {
      const rows = input.length / inDim;
      const tensor = this.qt[weightName];
      if (tensor.packed2Bit) {
        if (this.wasm?.image_ternary_packed_2bit_linear_f32) {
          const result = this.wasm.image_ternary_packed_2bit_linear_f32(
            input,
            tensor.values,
            tensor.scales,
            bias || new Float32Array(0),
            rows,
            inDim,
            outDim,
          );
          return result instanceof Float32Array ? result : new Float32Array(result);
        }
        return linearTernaryPacked2Bit(input, tensor, bias, inDim, outDim);
      }
      if (this.wasm?.image_ternary_linear_f32) {
        const result = this.wasm.image_ternary_linear_f32(
          input,
          tensor.values,
          tensor.scales,
          bias || new Float32Array(0),
          rows,
          inDim,
          outDim,
        );
        return result instanceof Float32Array ? result : new Float32Array(result);
      }
      return linearTernary(input, tensor, bias, inDim, outDim);
    }
    return linear(input, this.tensor(weightName), bias, inDim, outDim);
  }

  predictNoise(image, timestep, label) {
    let tokens = this.linearPrefix('patch_in', patchify(image, this.imageSize, this.patchSize, this.channels), this.patchDim, this.dim);
    addInPlace(tokens, this.tensor('pos'));
    let cond = this.linearPrefix('time_mlp.0', timestepEmbedding(timestep, this.dim), this.dim, this.dim);
    for (let index = 0; index < cond.length; index += 1) cond[index] = cond[index] / (1 + Math.exp(-cond[index]));
    cond = this.linearPrefix('time_mlp.2', cond, this.dim, this.dim);
    const classEmbed = this.tensor('class_embed.weight').subarray(label * this.dim, (label + 1) * this.dim);
    for (let index = 0; index < this.dim; index += 1) cond[index] += classEmbed[index];
    for (let block = 0; block < this.config.depth; block += 1) {
      tokens = this.block(tokens, cond, block);
    }
    tokens = layerNorm(tokens, this.tensor('norm.weight'), this.tensor('norm.bias'), this.tokens, this.dim);
    const patches = this.linearPrefix('patch_out', tokens, this.dim, this.patchDim);
    return unpatchify(patches, this.imageSize, this.patchSize, this.channels);
  }

  block(tokens, cond, block) {
    const condInput = new Float32Array(cond);
    for (let index = 0; index < cond.length; index += 1) {
      const value = cond[index];
      condInput[index] = value / (1 + Math.exp(-value));
    }
    const condOut = this.linearPrefix(`blocks.${block}.cond.1`, condInput, this.dim, this.dim * 4);
    const shift1 = condOut.subarray(0, this.dim);
    const scale1 = condOut.subarray(this.dim, this.dim * 2);
    const shift2 = condOut.subarray(this.dim * 2, this.dim * 3);
    const scale2 = condOut.subarray(this.dim * 3, this.dim * 4);
    let y = layerNorm(tokens, this.tensor(`blocks.${block}.norm1.weight`), this.tensor(`blocks.${block}.norm1.bias`), this.tokens, this.dim);
    for (let token = 0; token < this.tokens; token += 1) {
      const offset = token * this.dim;
      for (let index = 0; index < this.dim; index += 1) y[offset + index] = y[offset + index] * (1 + scale1[index]) + shift1[index];
    }
    addInPlace(tokens, this.attention(y, block));
    y = layerNorm(tokens, this.tensor(`blocks.${block}.norm2.weight`), this.tensor(`blocks.${block}.norm2.bias`), this.tokens, this.dim);
    for (let token = 0; token < this.tokens; token += 1) {
      const offset = token * this.dim;
      for (let index = 0; index < this.dim; index += 1) y[offset + index] = y[offset + index] * (1 + scale2[index]) + shift2[index];
    }
    let mlp = this.linearPrefix(`blocks.${block}.mlp.0`, y, this.dim, this.dim * this.config.mlp_ratio);
    for (let index = 0; index < mlp.length; index += 1) mlp[index] = gelu(mlp[index]);
    mlp = this.linearPrefix(`blocks.${block}.mlp.2`, mlp, this.dim * this.config.mlp_ratio, this.dim);
    addInPlace(tokens, mlp);
    return tokens;
  }

  attention(input, block) {
    const heads = this.config.heads;
    const headDim = this.dim / heads;
    const qkv = this.linearPrefix(`blocks.${block}.attn.qkv`, input, this.dim, this.dim * 3);
    const out = new Float32Array(input.length);
    const scores = new Float32Array(this.tokens);
    for (let head = 0; head < heads; head += 1) {
      for (let query = 0; query < this.tokens; query += 1) {
        for (let key = 0; key < this.tokens; key += 1) {
          let score = 0;
          for (let inner = 0; inner < headDim; inner += 1) {
            const qIndex = query * this.dim * 3 + head * headDim + inner;
            const kIndex = key * this.dim * 3 + this.dim + head * headDim + inner;
            score += qkv[qIndex] * qkv[kIndex];
          }
          scores[key] = score / Math.sqrt(headDim);
        }
        const probs = softmax(scores);
        for (let inner = 0; inner < headDim; inner += 1) {
          let value = 0;
          for (let key = 0; key < this.tokens; key += 1) {
            const vIndex = key * this.dim * 3 + this.dim * 2 + head * headDim + inner;
            value += probs[key] * qkv[vIndex];
          }
          out[query * this.dim + head * headDim + inner] = value;
        }
      }
    }
    return this.linearPrefix(`blocks.${block}.attn.out`, out, this.dim, this.dim);
  }

  async generate({ prompt, seed, steps = 50, onProgress }) {
    const label = classFromPrompt(prompt, this.classes);
    const total = this.channels * this.imageSize * this.imageSize;
    let image = normalArray(total, Number(seed || hashText(prompt)));
    const stride = Math.max(1, Math.floor(this.timesteps / steps));
    const schedule = [];
    for (let t = this.timesteps - 1; t >= 0; t -= stride) schedule.push(t);
    if (schedule[schedule.length - 1] !== 0) schedule.push(0);
    for (let step = 0; step < schedule.length; step += 1) {
      const t = schedule[step];
      const eps = this.predictNoise(image, t, label);
      const alphaBar = this.alphaBars[t];
      const nextT = schedule[Math.min(step + 1, schedule.length - 1)];
      const nextAlphaBar = t === 0 ? 1 : this.alphaBars[nextT];
      const predX0 = new Float32Array(total);
      for (let index = 0; index < total; index += 1) {
        predX0[index] = Math.max(-1, Math.min(1, (image[index] - Math.sqrt(1 - alphaBar) * eps[index]) / Math.sqrt(alphaBar)));
        image[index] = Math.sqrt(nextAlphaBar) * predX0[index] + Math.sqrt(Math.max(0, 1 - nextAlphaBar)) * eps[index];
      }
      onProgress?.({ step: step + 1, total: schedule.length, label: this.classes[label] || String(label) });
      if (step % 2 === 1) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return { image, label, labelName: this.classes[label] || String(label) };
  }
}

export class SanaSnapshotRuntime {
  constructor(manifest, samples, baseUrl) {
    this.manifest = manifest;
    this.samples = samples;
    this.baseUrl = baseUrl;
    this.config = {
      image_size: Number(manifest.config?.image_size || 512),
      quality_tier: manifest.quality_tier || manifest.config?.quality_tier || 'current-best-student-dev',
      checkpoint: manifest.source_checkpoint || '',
      training_step: Number(manifest.training_step || 0),
    };
    this.backend = 'sana-student-dev-snapshot';
  }

  static async fromManifest(manifest, manifestUrl) {
    const baseUrl = new URL('.', manifestUrl);
    const sampleUrl = new URL(manifest.samples?.index || 'samples.json', baseUrl);
    const response = await fetch(sampleUrl, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Sana student sample index failed: ${response.status}`);
    const payload = await response.json();
    const samples = Array.isArray(payload.samples) ? payload.samples : [];
    if (!samples.length) throw new Error('Sana student dev bundle has no samples');
    return new SanaSnapshotRuntime(manifest, samples, baseUrl);
  }

  pickSample(prompt, seed) {
    const text = String(prompt || '').toLowerCase();
    let best = null;
    let bestScore = -1;
    for (const sample of this.samples) {
      const samplePrompt = String(sample.prompt || '').toLowerCase();
      const words = samplePrompt.split(/[^a-z0-9]+/).filter((word) => word.length > 3);
      let score = 0;
      for (const word of words) if (text.includes(word)) score += 1;
      if (score > bestScore) {
        best = sample;
        bestScore = score;
      }
    }
    if (bestScore > 0 && best) return best;
    return this.samples[((hashText(prompt) ^ Number(seed || 0)) >>> 0) % this.samples.length];
  }

  async generate({ prompt, seed, steps = 8, onProgress }) {
    const total = Math.max(2, Number(steps || 8));
    for (let step = 1; step <= total; step += 1) {
      onProgress?.({ step, total, label: 'Sana student snapshot' });
      if (step % 2 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const sample = this.pickSample(prompt, seed);
    const response = await fetch(new URL(sample.file, this.baseUrl), { cache: 'no-store' });
    if (!response.ok) throw new Error(`Sana student sample failed: ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    return {
      image: new Float32Array(0),
      imageBase64: bytesToBase64(bytes),
      labelName: sample.title || 'Sana student',
      checkpoint: this.config.checkpoint,
      trainingStep: this.config.training_step,
    };
  }
}

export class ImageDevServerRuntime {
  constructor(manifest) {
    this.manifest = manifest;
    this.config = {
      image_size: Number(manifest.config?.image_size || 512),
      quality_tier: manifest.quality_tier || 'local-dev-live',
      checkpoint: manifest.source_checkpoint || '',
      training_step: Number(manifest.training_step || 0),
    };
    this.endpoint = String(manifest.runtime?.endpoint || 'http://127.0.0.1:8798/generate');
    this.backend = String(manifest.runtime?.backend || manifest.backend || 'image-student-local-dev');
  }

  static async fromManifest(manifest) {
    const healthUrl = new URL('/health', thisEndpointOrigin(manifest.runtime?.endpoint || 'http://127.0.0.1:8798/generate')).href;
    const response = await fetch(healthUrl, { cache: 'no-store' });
    if (!response.ok) throw new Error(`local image runtime health failed: ${response.status}`);
    return new ImageDevServerRuntime(manifest);
  }

  async generate({ prompt, seed, steps = 24, onProgress }) {
    onProgress?.({ step: 1, total: 3, label: 'Connecting to local image student' });
    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        seed,
        steps,
        width: this.config.image_size,
        height: this.config.image_size,
        guidance: 0,
      }),
    });
    onProgress?.({ step: 2, total: 3, label: 'Sampling image student' });
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(`local image generation failed: ${response.status}${text ? ` ${text}` : ''}`);
    }
    const result = await response.json();
    onProgress?.({ step: 3, total: 3, label: 'Rendering PNG' });
    return {
      image: new Float32Array(0),
      imageBase64: String(result.image_base64 || ''),
      labelName: result.model || 'Image student live',
      checkpoint: result.checkpoint || this.config.checkpoint,
      trainingStep: result.training_step || this.config.training_step,
    };
  }
}

export class SanaBrowserBundleRuntime {
  constructor(manifest, tensorIndex, ternaryIndex) {
    this.manifest = manifest;
    this.tensorIndex = tensorIndex;
    this.ternaryIndex = ternaryIndex;
    this.config = {
      image_size: Number(manifest.config?.resolution || manifest.config?.image_size || 512),
      quality_tier: manifest.quality_tier || 'sana-browser-export',
      checkpoint: manifest.source_checkpoint || '',
      training_step: Number(manifest.training_step || 0),
    };
    this.backend = 'sana-browser-bundle';
  }

  static async fromManifest(manifest, manifestUrl) {
    const base = new URL('.', manifestUrl);
    const [indexResponse, ternaryIndexResponse] = await Promise.all([
      fetch(new URL(manifest.tensors?.index || 'tensor_index.json', base), { cache: 'no-store' }),
      manifest.ternary_tensors?.enabled
        ? fetch(new URL(manifest.ternary_tensors?.index || 'tensor_ternary_index.json', base), { cache: 'no-store' })
        : Promise.resolve(null),
    ]);
    if (!indexResponse.ok) throw new Error(`Sana tensor index failed: ${indexResponse.status}`);
    if (ternaryIndexResponse && !ternaryIndexResponse.ok) throw new Error(`Sana ternary index failed: ${ternaryIndexResponse.status}`);
    const tensorIndex = await indexResponse.json();
    const ternaryIndex = ternaryIndexResponse ? await ternaryIndexResponse.json() : {};
    return new SanaBrowserBundleRuntime(manifest, tensorIndex, ternaryIndex);
  }

  async generate() {
    throw new Error('Sana browser bundle is exported, but live SANA inference still needs the WASM text-encoder, scheduler, transformer, and VAE runtime. This path no longer returns restored sample artifacts.');
  }
}

function thisEndpointOrigin(endpoint) {
  const url = new URL(String(endpoint || 'http://127.0.0.1:8798/generate'));
  return `${url.protocol}//${url.host}`;
}

export function imageToSvg(image, { width = 32, height = 32, scale = 12, prompt = '', labelName = '' } = {}) {
  const outWidth = width * scale;
  const outHeight = height * scale;
  const rects = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const r = Math.max(0, Math.min(255, Math.round((image[y * width + x] + 1) * 127.5)));
      const g = Math.max(0, Math.min(255, Math.round((image[width * height + y * width + x] + 1) * 127.5)));
      const b = Math.max(0, Math.min(255, Math.round((image[width * height * 2 + y * width + x] + 1) * 127.5)));
      rects.push(`<rect x="${x * scale}" y="${y * scale}" width="${scale}" height="${scale}" fill="rgb(${r},${g},${b})"/>`);
    }
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${outWidth}" height="${outHeight}" viewBox="0 0 ${outWidth} ${outHeight}" role="img" aria-label="${prompt.replace(/"/g, '&quot;')}"><rect width="${outWidth}" height="${outHeight}" fill="#111"/>${rects.join('')}<text x="8" y="${outHeight - 10}" fill="rgba(255,255,255,0.82)" font-size="12" font-family="system-ui">${labelName}</text></svg>`;
}

function bytesToBase64(bytes) {
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function writeImageData(image, width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  const plane = width * height;
  for (let index = 0; index < plane; index += 1) {
    data[index * 4] = Math.max(0, Math.min(255, Math.round((image[index] + 1) * 127.5)));
    data[index * 4 + 1] = Math.max(0, Math.min(255, Math.round((image[plane + index] + 1) * 127.5)));
    data[index * 4 + 2] = Math.max(0, Math.min(255, Math.round((image[plane * 2 + index] + 1) * 127.5)));
    data[index * 4 + 3] = 255;
  }
  return new ImageData(data, width, height);
}

export async function imageToPngBase64(image, { width = 32, height = 32, outputSize = 512 } = {}) {
  if (typeof OffscreenCanvas === 'undefined') return '';
  const source = new OffscreenCanvas(width, height);
  const sourceContext = source.getContext('2d', { willReadFrequently: true });
  if (!sourceContext) return '';
  sourceContext.putImageData(writeImageData(image, width, height), 0, 0);

  const target = new OffscreenCanvas(outputSize, outputSize);
  const targetContext = target.getContext('2d');
  if (!targetContext) return '';
  targetContext.imageSmoothingEnabled = true;
  targetContext.imageSmoothingQuality = 'high';
  targetContext.fillStyle = '#080b0f';
  targetContext.fillRect(0, 0, outputSize, outputSize);
  targetContext.drawImage(source, 0, 0, outputSize, outputSize);

  const blob = await target.convertToBlob({ type: 'image/png' });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  return bytesToBase64(bytes);
}
