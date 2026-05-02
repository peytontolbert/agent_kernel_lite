let wasmModulePromise = null;

function normalizeLayout(layoutHeader) {
  if (!layoutHeader || layoutHeader.length < 13) {
    throw new Error("BitNet layout_header must contain at least 13 entries");
  }
  const header = Int32Array.from(Array.from(layoutHeader, Number));
  if (header[0] !== 1 || header[1] !== 16 || header[2] !== 32 || header[9] !== 1) {
    throw new Error("Unsupported BitNet WASM layout; expected v1 16x32 interleave mode 1");
  }
  return {
    header,
    logicalOut: header[3],
    logicalIn: header[4],
    paddedOut: header[5],
    paddedIn: header[6],
    scaleGranularity: header[7],
    scaleGroupSize: header[8],
    segmentCount: header[11],
  };
}

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

async function fetchTensor(entry, baseUrl, TypedArray) {
  const url = resolveUrl(entry.path, baseUrl);
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${entry.path}: ${response.status}`);
  }
  return new TypedArray(await response.arrayBuffer());
}

function tensorType(entry) {
  if (entry.dtype === "uint8") return Uint8Array;
  if (entry.dtype === "int32") return Int32Array;
  if (entry.dtype === "float32") return Float32Array;
  throw new Error(`unsupported tensor dtype: ${entry.dtype}`);
}

async function ensureBitNetWasm() {
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

export class BitNetLinearWASM {
  constructor(bundle) {
    this.layout = normalizeLayout(bundle.layoutHeader);
    this.packedWeight = bundle.packedWeight instanceof Uint8Array
      ? bundle.packedWeight
      : new Uint8Array(bundle.packedWeight);
    this.scaleValues = bundle.scaleValues instanceof Float32Array
      ? bundle.scaleValues
      : new Float32Array(bundle.scaleValues);
    this.segmentOffsets = bundle.segmentOffsets instanceof Int32Array
      ? bundle.segmentOffsets
      : Int32Array.from(bundle.segmentOffsets || []);
    this.bias = bundle.bias
      ? (bundle.bias instanceof Float32Array ? bundle.bias : new Float32Array(bundle.bias))
      : new Float32Array(0);
    this.inputScales = bundle.inputScales
      ? (bundle.inputScales instanceof Float32Array ? bundle.inputScales : new Float32Array(bundle.inputScales))
      : new Float32Array([1]);
    this.inputQuantMode = bundle.inputQuantMode ?? 0;
    this.inputQuantBits = bundle.inputQuantBits ?? 8;
    this.inputScaleRows = bundle.inputScaleRows ?? 1;
  }

  static async fromManifestLayer(manifest, layer, manifestUrl, options = {}) {
    const progress = typeof options.progress === "function" ? options.progress : () => {};
    const index = Number(options.index || 0);
    const total = Number(options.total || 0);
    const name = String(options.name || layer.name || "layer");
    const label = total ? `${index}/${total}: ${name}` : name;
    const baseUrl = new URL(".", manifestUrl).toString();
    const tensors = layer.tensors;
    const layersBaseUrl = resolveUrl("layers/", baseUrl);
    progress({ phase: "layer_tensors", index, total, name, message: `Loading BitNet WASM tensors ${label}` });
    const [packedWeight, scaleValues, segmentOffsets, bias, inputScales] = await Promise.all([
      fetchTensor(tensors.packed_weight, layersBaseUrl, Uint8Array),
      fetchTensor(tensors.scale_values, layersBaseUrl, Float32Array),
      fetchTensor(tensors.segment_offsets, layersBaseUrl, Int32Array),
      tensors.bias ? fetchTensor(tensors.bias, layersBaseUrl, Float32Array) : Promise.resolve(null),
      fetchTensor(tensors.act_scale, layersBaseUrl, tensorType(tensors.act_scale)),
    ]);
    progress({ phase: "layer_ready", index, total, name, message: `BitNet WASM layer ${label} ready` });
    return new BitNetLinearWASM({
      layoutHeader: layer.layout_header,
      packedWeight,
      scaleValues,
      segmentOffsets,
      bias,
      inputScales,
      inputQuantMode: layer.act_quant_mode === "none" ? 0 : 1,
      inputQuantBits: layer.act_quant_bits,
      inputScaleRows: layer.act_quant_mode === "static_int8" ? 1 : 1,
    });
  }

  async run(input, rows = 1) {
    const x = input instanceof Float32Array ? input : new Float32Array(input);
    if (x.length !== rows * this.layout.logicalIn) {
      throw new Error(`BitNet input length mismatch: got ${x.length}, expected ${rows * this.layout.logicalIn}`);
    }
    const wasm = await ensureBitNetWasm();
    return wasm.bitnet_linear_f32(
      x,
      this.packedWeight,
      this.scaleValues,
      this.segmentOffsets,
      this.bias,
      this.layout.header,
      this.inputScales,
      rows,
      this.inputQuantMode,
      this.inputQuantBits,
      this.inputScaleRows,
    );
  }
}
