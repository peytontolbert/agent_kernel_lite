const PARAM_U32_COUNT = 12;
const PARAM_BUFFER_BYTES = PARAM_U32_COUNT * 4;
const shaderTextCache = new Map();
const pipelineCache = new WeakMap();

function align4(value) {
  return (value + 3) & ~3;
}

function packedWeightToWords(packedWeight) {
  const bytes = packedWeight instanceof Uint8Array ? packedWeight : new Uint8Array(packedWeight);
  const padded = new Uint8Array(align4(bytes.byteLength));
  padded.set(bytes);
  return new Uint32Array(padded.buffer);
}

function createStorageBuffer(device, data, usage = GPUBufferUsage.STORAGE) {
  const source = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
  const buffer = device.createBuffer({
    size: align4(source.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, source.buffer, source.byteOffset, source.byteLength);
  return buffer;
}

function createOutputBuffer(device, byteLength) {
  return device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

function createReadbackBuffer(device, byteLength) {
  return device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
}

function normalizeLayout(layoutHeader) {
  if (!layoutHeader || layoutHeader.length < 13) {
    throw new Error("BitNet layout_header must contain at least 13 entries");
  }
  const header = Array.from(layoutHeader, Number);
  if (header[0] !== 1 || header[1] !== 16 || header[2] !== 32 || header[9] !== 1) {
    throw new Error("Unsupported BitNet browser layout; expected v1 16x32 interleave mode 1");
  }
  return {
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

async function fetchJson(url) {
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchText(url) {
  const response = await fetchWithRetry(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  return response.text();
}

async function fetchTextCached(url) {
  if (!shaderTextCache.has(url)) {
    shaderTextCache.set(url, fetchText(url));
  }
  return shaderTextCache.get(url);
}

async function getBitNetPipeline(device, shaderCode, cacheKey) {
  let deviceCache = pipelineCache.get(device);
  if (!deviceCache) {
    deviceCache = new Map();
    pipelineCache.set(device, deviceCache);
  }
  if (!deviceCache.has(cacheKey)) {
    deviceCache.set(cacheKey, (async () => {
      const module = device.createShaderModule({ code: shaderCode });
      const descriptor = {
        layout: "auto",
        compute: { module, entryPoint: "bitnet_linear_main" },
      };
      const pipeline = typeof device.createComputePipelineAsync === "function"
        ? await device.createComputePipelineAsync(descriptor)
        : device.createComputePipeline(descriptor);
      return { module, pipeline };
    })());
  }
  return deviceCache.get(cacheKey);
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
  if (entry.dtype === "uint8") {
    return Uint8Array;
  }
  if (entry.dtype === "int32") {
    return Int32Array;
  }
  if (entry.dtype === "float32") {
    return Float32Array;
  }
  throw new Error(`unsupported tensor dtype: ${entry.dtype}`);
}

export async function createBitNetWebGPUDevice() {
  if (!globalThis.navigator?.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter request failed");
  }
  const device = await adapter.requestDevice();
  return { adapter, device };
}

export class BitNetLinearWebGPU {
  constructor(device, bundle) {
    this.device = device;
    this.layout = normalizeLayout(bundle.layoutHeader);
    this.hasBias = bundle.bias != null;
    this.inputQuantMode = bundle.inputQuantMode ?? 0;
    this.inputQuantBits = bundle.inputQuantBits ?? 8;
    this.inputScaleRows = bundle.inputScaleRows ?? 1;

    if (!bundle.shaderCode && !bundle.pipeline) {
      throw new Error("BitNetLinearWebGPU requires shaderCode or pipeline; use fromManifestLayer() or fromManifestUrl()");
    }
    if (bundle.pipeline) {
      this.module = bundle.module || null;
      this.pipeline = bundle.pipeline;
    } else {
      this.module = device.createShaderModule({ code: bundle.shaderCode });
      this.pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: this.module, entryPoint: "bitnet_linear_main" },
      });
    }

    this.packedWeightBuffer = createStorageBuffer(device, packedWeightToWords(bundle.packedWeight));
    this.scaleBuffer = createStorageBuffer(device, new Float32Array(bundle.scaleValues));
    this.segmentOffsetBuffer = createStorageBuffer(device, new Uint32Array(bundle.segmentOffsets));
    this.biasBuffer = createStorageBuffer(
      device,
      this.hasBias ? new Float32Array(bundle.bias) : new Float32Array([0]),
    );
    this.inputScaleBuffer = createStorageBuffer(
      device,
      bundle.inputScales ? new Float32Array(bundle.inputScales) : new Float32Array([1]),
    );
    this.paramsBuffer = device.createBuffer({
      size: PARAM_BUFFER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  static async fromManifestLayer(device, manifest, layer, manifestUrl, options = {}) {
    const progress = typeof options.progress === "function" ? options.progress : () => {};
    const index = Number(options.index || 0);
    const total = Number(options.total || 0);
    const name = String(options.name || layer.name || "layer");
    const label = total ? `${index}/${total}: ${name}` : name;
    const baseUrl = new URL(".", manifestUrl).toString();
    const shaderUrl = resolveUrl(manifest.runtime.files.wgsl, baseUrl);
    const runtimeBaseUrl = resolveUrl(".", shaderUrl);
    progress({ phase: "layer_shader", index, total, name, message: `Loading shader for BitNet layer ${label}` });
    const shaderCode = options.shaderCode || await fetchTextCached(shaderUrl);
    progress({ phase: "layer_pipeline", index, total, name, message: `Preparing WebGPU pipeline for BitNet layer ${label}` });
    const pipelineBundle = options.pipeline
      ? { module: options.module || null, pipeline: options.pipeline }
      : await getBitNetPipeline(device, shaderCode, shaderUrl);
    const tensors = layer.tensors;
    const layersBaseUrl = resolveUrl("layers/", baseUrl);
    progress({ phase: "layer_tensors", index, total, name, message: `Loading tensors for BitNet layer ${label}` });
    const [packedWeight, scaleValues, segmentOffsets, bias, inputScales] = await Promise.all([
      fetchTensor(tensors.packed_weight, layersBaseUrl, Uint8Array),
      fetchTensor(tensors.scale_values, layersBaseUrl, Float32Array),
      fetchTensor(tensors.segment_offsets, layersBaseUrl, Int32Array),
      tensors.bias ? fetchTensor(tensors.bias, layersBaseUrl, Float32Array) : Promise.resolve(null),
      fetchTensor(tensors.act_scale, layersBaseUrl, tensorType(tensors.act_scale)),
    ]);
    progress({ phase: "layer_upload", index, total, name, message: `Uploading BitNet layer ${label}` });
    return new BitNetLinearWebGPU(device, {
      shaderCode,
      module: pipelineBundle.module,
      pipeline: pipelineBundle.pipeline,
      layoutHeader: layer.layout_header,
      packedWeight,
      scaleValues,
      segmentOffsets,
      bias,
      inputScales,
      inputQuantMode: layer.act_quant_mode === "none" ? 0 : 1,
      inputQuantBits: layer.act_quant_bits,
      inputScaleRows: layer.act_quant_mode === "static_int8" ? 1 : 1,
      runtimeBaseUrl,
    });
  }

  static async fromManifestUrl(device, manifestUrl, layerName) {
    const manifest = await fetchJson(manifestUrl);
    const layer = manifest.layers.find((candidate) => candidate.name === layerName);
    if (!layer) {
      throw new Error(`BitNet layer not found in manifest: ${layerName}`);
    }
    return BitNetLinearWebGPU.fromManifestLayer(device, manifest, layer, manifestUrl);
  }

  async run(input, rows = 1) {
    const x = input instanceof Float32Array ? input : new Float32Array(input);
    if (x.length !== rows * this.layout.logicalIn) {
      throw new Error(`BitNet input length mismatch: got ${x.length}, expected ${rows * this.layout.logicalIn}`);
    }

    const outputLength = rows * this.layout.logicalOut;
    const inputBuffer = createStorageBuffer(this.device, x);
    const outputBuffer = createOutputBuffer(this.device, outputLength * Float32Array.BYTES_PER_ELEMENT);
    const readbackBuffer = createReadbackBuffer(this.device, outputLength * Float32Array.BYTES_PER_ELEMENT);

    const params = new Uint32Array([
      rows,
      this.layout.logicalIn,
      this.layout.logicalOut,
      this.layout.paddedIn,
      this.layout.scaleGranularity,
      this.layout.scaleGroupSize,
      this.layout.segmentCount,
      this.hasBias ? 1 : 0,
      this.inputQuantMode,
      this.inputQuantBits,
      this.inputScaleRows,
      0,
    ]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: this.packedWeightBuffer } },
        { binding: 2, resource: { buffer: this.scaleBuffer } },
        { binding: 3, resource: { buffer: this.segmentOffsetBuffer } },
        { binding: 4, resource: { buffer: this.biasBuffer } },
        { binding: 5, resource: { buffer: this.inputScaleBuffer } },
        { binding: 6, resource: { buffer: outputBuffer } },
        { binding: 7, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.layout.logicalOut / 8), Math.ceil(rows / 8), 1);
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * Float32Array.BYTES_PER_ELEMENT);
    this.device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const mapped = readbackBuffer.getMappedRange();
    const result = new Float32Array(mapped.slice(0));
    readbackBuffer.unmap();
    return result;
  }
}
