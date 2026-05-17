let wasmModulePromise = null;

function resolveUrl(path, baseUrl) {
  return new URL(path, baseUrl).toString();
}

function alignedSlice(buffer, offset, nbytes, TypedArray) {
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
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) {
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
    const entry = this.q4Index[name];
    if (!entry) {
      throw new Error(`Q4 tensor not found: ${name}`);
    }
    return {
      entry,
      packedWeight: alignedSlice(this.q4Buffer, entry.offset, entry.nbytes, Uint8Array),
      rowScalesF16: alignedSlice(this.q4Buffer, entry.scale_offset, entry.scale_nbytes, Uint16Array),
    };
  }

  denseTensor(name) {
    const entry = this.denseIndex[name];
    if (!entry) {
      throw new Error(`dense tensor not found: ${name}`);
    }
    if (entry.dtype === "float16") {
      return alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Uint16Array);
    }
    if (entry.dtype === "float32") {
      return alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Float32Array);
    }
    if (entry.dtype === "bool_u8" || entry.dtype === "uint8") {
      return alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, Uint8Array);
    }
    if (entry.dtype === "int64") {
      return alignedSlice(this.denseBuffer, entry.offset, entry.nbytes, BigInt64Array);
    }
    throw new Error(`unsupported dense dtype for ${name}: ${entry.dtype}`);
  }

  denseF32Tensor(name) {
    const entry = this.denseIndex[name];
    const raw = this.denseTensor(name);
    if (entry.dtype === "float32") {
      return raw;
    }
    if (entry.dtype !== "float16") {
      throw new Error(`dense tensor is not floating point: ${name}`);
    }
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) {
      out[i] = f16ToF32(raw[i]);
    }
    return out;
  }

  runQ4Linear(name, input, rows = 1, biasName = "") {
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
