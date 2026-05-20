const SAMPLE_RATE = 24000;
const N_FFT = 1024;
const HOP = 256;
const CHANNELS = 100;
const DIM = 512;
const INNER = 1536;
const LAYERS = 8;
const EPS = 1e-6;
const HANN_WINDOW = makeHannWindow();
const IFFT_BIT_REVERSE = makeBitReverseTable(N_FFT);

export class VocosMel24khzRuntime {
  constructor(bundle) {
    this.bundle = bundle;
  }

  decode(melFrames) {
    const frames = melFrames.length / CHANNELS;
    if (!Number.isInteger(frames) || frames <= 0) {
      throw new Error("melFrames must be flat [frames, 100]");
    }
    let x = conv1d(this.bundle, "backbone.embed.weight", "backbone.embed.bias", melFrames, frames, CHANNELS, DIM, 7, 3);
    x = layerNormRows(x, this.bundle.denseF32Tensor("backbone.norm.weight"), this.bundle.denseF32Tensor("backbone.norm.bias"), frames, DIM);
    for (let layer = 0; layer < LAYERS; layer += 1) {
      x = this.convnextBlock(layer, x, frames);
    }
    x = layerNormRows(
      x,
      this.bundle.denseF32Tensor("backbone.final_layer_norm.weight"),
      this.bundle.denseF32Tensor("backbone.final_layer_norm.bias"),
      frames,
      DIM,
    );
    const stft = linear(this.bundle, "head.out.weight", "head.out.bias", x, frames, DIM, N_FFT + 2);
    return istftHead(this.bundle, stft, frames);
  }

  convnextBlock(layer, input, frames) {
    const prefix = `backbone.convnext.${layer}`;
    let x = depthwiseConv1d(this.bundle, `${prefix}.dwconv.weight`, `${prefix}.dwconv.bias`, input, frames, DIM, 7, 3);
    x = layerNormRows(x, this.bundle.denseF32Tensor(`${prefix}.norm.weight`), this.bundle.denseF32Tensor(`${prefix}.norm.bias`), frames, DIM);
    x = linear(this.bundle, `${prefix}.pwconv1.weight`, `${prefix}.pwconv1.bias`, x, frames, DIM, INNER);
    geluInPlace(x);
    x = linear(this.bundle, `${prefix}.pwconv2.weight`, `${prefix}.pwconv2.bias`, x, frames, INNER, DIM);
    const gamma = this.bundle.denseF32Tensor(`${prefix}.gamma`);
    for (let row = 0; row < frames; row += 1) {
      const offset = row * DIM;
      for (let col = 0; col < DIM; col += 1) {
        x[offset + col] = input[offset + col] + gamma[col] * x[offset + col];
      }
    }
    return x;
  }
}

export class FP16TensorBundle {
  constructor({ manifest, index, buffer }) {
    this.manifest = manifest;
    this.index = index;
    this.buffer = buffer;
    this.cache = new Map();
  }

  static async fromManifestUrl(manifestUrl) {
    const manifest = await fetchJson(manifestUrl);
    const baseUrl = new URL(".", manifestUrl).toString();
    const [index, buffer] = await Promise.all([
      fetchJson(new URL(manifest.files.index, baseUrl).toString()),
      fetchBuffer(new URL(manifest.files.tensors, baseUrl).toString()),
    ]);
    return new FP16TensorBundle({ manifest, index, buffer });
  }

  denseF32Tensor(name) {
    if (this.cache.has(name)) return this.cache.get(name);
    const entry = this.index[name];
    if (!entry) throw new Error(`tensor not found: ${name}`);
    if (entry.dtype !== "float16") throw new Error(`unsupported dtype for ${name}: ${entry.dtype}`);
    const raw = alignedSlice(this.buffer, entry.offset, entry.nbytes, Uint16Array);
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) out[i] = f16ToF32(raw[i]);
    this.cache.set(name, out);
    return out;
  }
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return response.json();
}

async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return response.arrayBuffer();
}

function alignedSlice(buffer, offset, nbytes, TypedArray) {
  const bytes = buffer instanceof ArrayBuffer
    ? new Uint8Array(buffer, offset, nbytes)
    : new Uint8Array(buffer.buffer, buffer.byteOffset + offset, nbytes);
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

function conv1d(bundle, weightName, biasName, input, frames, inChannels, outChannels, kernel, padding) {
  if (bundle.q4Index?.[weightName] && typeof bundle.runQ4Conv1d === 'function') {
    return bundle.runQ4Conv1d(weightName, biasName, input, frames, inChannels, outChannels, kernel, padding);
  }
  const bias = bundle.denseF32Tensor(biasName);
  const q4 = maybeQ4Tensor(bundle, weightName);
  const weight = q4 ? null : bundle.denseF32Tensor(weightName);
  const out = new Float32Array(frames * outChannels);
  for (let t = 0; t < frames; t += 1) {
    for (let oc = 0; oc < outChannels; oc += 1) {
      let sum = bias[oc] || 0;
      const row = oc * inChannels * kernel;
      for (let ic = 0; ic < inChannels; ic += 1) {
        for (let k = 0; k < kernel; k += 1) {
          const srcT = t + k - padding;
          if (srcT >= 0 && srcT < frames) {
            const w = q4 ? q4At(q4, outChannels === q4.entry.shape[0] ? inChannels * kernel : inChannels * kernel, oc, ic * kernel + k) : weight[row + ic * kernel + k];
            sum += input[srcT * inChannels + ic] * w;
          }
        }
      }
      out[t * outChannels + oc] = sum;
    }
  }
  return out;
}

function depthwiseConv1d(bundle, weightName, biasName, input, frames, channels, kernel, padding) {
  if (bundle.q4Index?.[weightName] && typeof bundle.runQ4DepthwiseConv1d === 'function') {
    return bundle.runQ4DepthwiseConv1d(weightName, biasName, input, frames, channels, kernel, padding);
  }
  const bias = bundle.denseF32Tensor(biasName);
  const q4 = maybeQ4Tensor(bundle, weightName);
  const weight = q4 ? null : bundle.denseF32Tensor(weightName);
  const out = new Float32Array(frames * channels);
  for (let t = 0; t < frames; t += 1) {
    for (let ch = 0; ch < channels; ch += 1) {
      let sum = bias[ch] || 0;
      const row = ch * kernel;
      for (let k = 0; k < kernel; k += 1) {
        const srcT = t + k - padding;
        if (srcT >= 0 && srcT < frames) {
          const w = q4 ? q4At(q4, kernel, ch, k) : weight[row + k];
          sum += input[srcT * channels + ch] * w;
        }
      }
      out[t * channels + ch] = sum;
    }
  }
  return out;
}

function linear(bundle, weightName, biasName, input, rows, inDim, outDim) {
  if (bundle.q4Index?.[weightName] && typeof bundle.runQ4Linear === 'function') {
    return bundle.runQ4Linear(weightName, input, rows, biasName);
  }
  const weight = bundle.denseF32Tensor(weightName);
  const bias = bundle.denseF32Tensor(biasName);
  const out = new Float32Array(rows * outDim);
  for (let row = 0; row < rows; row += 1) {
    const inOffset = row * inDim;
    const outOffset = row * outDim;
    for (let oc = 0; oc < outDim; oc += 1) {
      let sum = bias[oc] || 0;
      const wOffset = oc * inDim;
      for (let ic = 0; ic < inDim; ic += 1) {
        sum += input[inOffset + ic] * weight[wOffset + ic];
      }
      out[outOffset + oc] = sum;
    }
  }
  return out;
}

function maybeQ4Tensor(bundle, weightName) {
  if (!bundle.q4Index?.[weightName] || typeof bundle.q4Tensor !== 'function') return null;
  return bundle.q4Tensor(weightName);
}

function q4At(q4, rowSize, row, col) {
  const linear = row * rowSize + col;
  const byte = q4.packedWeight[linear >> 1];
  let nibble = (linear & 1) === 0 ? byte & 0x0f : byte >> 4;
  if (nibble >= 8) nibble -= 16;
  return nibble * f16ToF32(q4.rowScalesF16[row]);
}

function layerNormRows(input, weight, bias, rows, cols) {
  const out = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * cols;
    let mean = 0;
    for (let col = 0; col < cols; col += 1) mean += input[offset + col];
    mean /= cols;
    let variance = 0;
    for (let col = 0; col < cols; col += 1) {
      const delta = input[offset + col] - mean;
      variance += delta * delta;
    }
    const inv = 1 / Math.sqrt(variance / cols + EPS);
    for (let col = 0; col < cols; col += 1) {
      out[offset + col] = (input[offset + col] - mean) * inv * weight[col] + bias[col];
    }
  }
  return out;
}

function geluInPlace(x) {
  for (let i = 0; i < x.length; i += 1) {
    const value = x[i];
    x[i] = 0.5 * value * (1 + erf(value / Math.SQRT2));
  }
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}

function istftHead(bundle, stftRows, frames) {
  if (typeof bundle.runVocosIstftHead === 'function') {
    return bundle.runVocosIstftHead(stftRows, frames);
  }
  const bins = N_FFT / 2 + 1;
  const frameTime = new Float32Array(frames * N_FFT);
  const real = new Float32Array(N_FFT);
  const imag = new Float32Array(N_FFT);
  for (let t = 0; t < frames; t += 1) {
    const row = t * (N_FFT + 2);
    real.fill(0);
    imag.fill(0);
    for (let f = 0; f < bins; f += 1) {
      const mag = Math.exp(Math.min(stftRows[row + f], Math.log(100)));
      const phase = stftRows[row + bins + f];
      const re = mag * Math.cos(phase);
      const im = mag * Math.sin(phase);
      real[f] = re;
      imag[f] = im;
      if (f > 0 && f < bins - 1) {
        real[N_FFT - f] = re;
        imag[N_FFT - f] = -im;
      }
    }
    inverseFftInPlace(real, imag);
    const frameOffset = t * N_FFT;
    for (let n = 0; n < N_FFT; n += 1) {
      frameTime[frameOffset + n] = real[n] * HANN_WINDOW[n];
    }
  }

  const paddedLength = (frames - 1) * HOP + N_FFT;
  const audio = new Float32Array(paddedLength);
  const envelope = new Float32Array(paddedLength);
  for (let t = 0; t < frames; t += 1) {
    const offset = t * HOP;
    for (let n = 0; n < N_FFT; n += 1) {
      const w = HANN_WINDOW[n];
      audio[offset + n] += frameTime[t * N_FFT + n];
      envelope[offset + n] += w * w;
    }
  }
  for (let i = 0; i < audio.length; i += 1) {
    if (envelope[i] > 1e-11) audio[i] /= envelope[i];
  }
  return audio.subarray(N_FFT / 2, Math.max(N_FFT / 2, paddedLength - N_FFT / 2));
}

function hann(n) {
  return Math.sin(Math.PI * n / N_FFT) ** 2;
}

function makeHannWindow() {
  const out = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n += 1) out[n] = hann(n);
  return out;
}

function makeBitReverseTable(size) {
  const bits = Math.log2(size);
  const table = new Uint16Array(size);
  for (let i = 0; i < size; i += 1) {
    let value = i;
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    table[i] = reversed;
  }
  return table;
}

function inverseFftInPlace(real, imag) {
  for (let i = 0; i < N_FFT; i += 1) {
    const j = IFFT_BIT_REVERSE[i];
    if (j > i) {
      const re = real[i];
      const im = imag[i];
      real[i] = real[j];
      imag[i] = imag[j];
      real[j] = re;
      imag[j] = im;
    }
  }
  for (let len = 2; len <= N_FFT; len <<= 1) {
    const half = len >> 1;
    const angle = (2 * Math.PI) / len;
    const stepRe = Math.cos(angle);
    const stepIm = Math.sin(angle);
    for (let start = 0; start < N_FFT; start += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < half; j += 1) {
        const even = start + j;
        const odd = even + half;
        const oddRe = real[odd] * wRe - imag[odd] * wIm;
        const oddIm = real[odd] * wIm + imag[odd] * wRe;
        real[odd] = real[even] - oddRe;
        imag[odd] = imag[even] - oddIm;
        real[even] += oddRe;
        imag[even] += oddIm;
        const nextRe = wRe * stepRe - wIm * stepIm;
        wIm = wRe * stepIm + wIm * stepRe;
        wRe = nextRe;
      }
    }
  }
  for (let i = 0; i < N_FFT; i += 1) {
    real[i] /= N_FFT;
    imag[i] /= N_FFT;
  }
}

export { SAMPLE_RATE };
