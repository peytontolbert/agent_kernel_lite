let wasmModulePromise = null;

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
      try {
        module = await import(new URL("model_stack_bitnet_wasm.js", import.meta.url).href);
      } catch (error) {
        module = await import(new URL("pkg/model_stack_bitnet_wasm.js", import.meta.url).href);
      }
      await module.default();
      return module;
    })();
  }
  return wasmModulePromise;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok && response.status !== 0) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchBuffer(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok && response.status !== 0) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.arrayBuffer();
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
  }

  static async fromManifestUrl(manifestUrl) {
    const manifest = await fetchJson(manifestUrl);
    const baseUrl = new URL(".", manifestUrl).toString();
    const [wasm, q4Index, denseIndex, q4Buffer, denseBuffer] = await Promise.all([
      ensureQ4Wasm(),
      fetchJson(resolveUrl(manifest.files.q4_index, baseUrl)),
      fetchJson(resolveUrl(manifest.files.dense_index, baseUrl)),
      fetchBuffer(resolveUrl(manifest.files.q4, baseUrl)),
      fetchBuffer(resolveUrl(manifest.files.dense, baseUrl)),
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

  runQ4Linear(name, input, rows = 1, biasName = "") {
    if (this.wasm?.Q4LinearHandle) {
      let handle = this.q4LinearHandleCache.get(`${name}:${biasName || ""}`);
      if (!handle) {
        const { entry, packedWeight, rowScalesF16 } = this.q4Tensor(name);
        const shape = entry.shape;
        const outDim = Number(shape[0]);
        const inDim = Number(shape.slice(1).reduce((acc, value) => acc * Number(value), 1));
        const bias = biasName ? this.denseF32Tensor(biasName) : new Float32Array(0);
        handle = new this.wasm.Q4LinearHandle(packedWeight, rowScalesF16, bias, inDim, outDim);
        this.q4LinearHandleCache.set(`${name}:${biasName || ""}`, handle);
      }
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
