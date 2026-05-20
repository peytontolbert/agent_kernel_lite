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
      let moduleUrl;
      try {
        moduleUrl = new URL("model_stack_bitnet_wasm.js", import.meta.url);
        module = await import(moduleUrl.href);
      } catch (error) {
        moduleUrl = new URL("pkg/model_stack_bitnet_wasm.js", import.meta.url);
        module = await import(moduleUrl.href);
      }
      const wasmUrl = new URL("model_stack_bitnet_wasm_bg.wasm", moduleUrl).href;
      const wasmBytes = await fetchBuffer(wasmUrl, "Model Stack WASM runtime");
      await module.default(wasmBytes);
      return module;
    })();
  }
  return wasmModulePromise;
}

async function fetchJson(url, label = "JSON asset") {
  try {
    const response = await fetch(url, { cache: "no-store" });
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
    const response = await fetch(url, { cache: "no-store" });
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

  runF5SampleMel({ condMel, condSeqLen, textIds, duration, steps, cfgStrength, swaySamplingCoef = -1.0, seed = 1337 }) {
    const session = this.f5Session();
    if (!session) {
      throw new Error("F5Q4DiTSession is not available in the WASM runtime");
    }
    return session.sample_mel(
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
