const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const MODEL_STACK_BITNET_RUNTIME_URL = new URL('../vendor/model-stack-bitnet/encdec_runtime.js?v=20260514-intent-head', import.meta.url).href;
const MODEL_STACK_BITNET_WGSL_URL = new URL('../vendor/model-stack-bitnet/bitnet_linear.wgsl', import.meta.url).href;
const AGENT_INTENT_LABELS = [
  'plan',
  'action_items',
  'rewrite',
  'translation',
  'web_search',
  'casual',
  'source_echo',
  'saved_data',
  'ask_user',
  'summary',
  'title',
  'checklist',
  'risks',
  'json',
  'ranking',
  'extraction',
  'subject',
  'brainstorm',
];

let hfPipeline = null;

function configureRuntime(env) {
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  env.useBrowserCache = true;

  // ORT often emits non-fatal WebGPU placement warnings for shape/control ops.
  // Keep real errors visible while avoiding noisy expected warnings in the demo.
  env.backends.onnx.logLevel = 'error';
  env.backends.onnx.wasm ??= {};
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;
}

async function ensureHfPipeline() {
  if (hfPipeline) return hfPipeline;
  const module = await import(TRANSFORMERS_CDN);
  configureRuntime(module.env);
  hfPipeline = module.pipeline;
  return hfPipeline;
}

let generator = null;
let loadedModelId = '';
let modelStackRuntime = null;
let modelStackManifestUrl = '';
let modelStackTokenizer = null;
let modelStackDevice = '';
let modelStackRuntimeDtype = '';
let vllmEndpoint = '';
let vllmModel = '';
let activeGenerationId = 0;
const cancelledGenerationIds = new Set();

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function normalizeGeneratedText(result, prompt) {
  const first = Array.isArray(result) ? result[0] : result;
  let text = String(first?.generated_text ?? first?.text ?? result ?? '');
  if (text.startsWith(prompt)) text = text.slice(prompt.length);
  const marker = 'Assistant:';
  const index = text.lastIndexOf(marker);
  if (index >= 0) text = text.slice(index + marker.length);
  return text.trim();
}

async function loadModel({ modelId, device }) {
  if (generator && loadedModelId === modelId) {
    post('loaded', { modelId });
    return;
  }
  if (String(modelId || '').startsWith('vllm:')) {
    await loadVllm({ modelId, endpoint: device?.vllmEndpoint });
    return;
  }
  if (String(modelId || '').startsWith('modelstack:')) {
    await loadModelStack({ modelId, device });
    return;
  }
  generator = null;
  modelStackRuntime = null;
  modelStackManifestUrl = '';
  modelStackTokenizer = null;
  modelStackDevice = '';
  modelStackRuntimeDtype = '';
  vllmEndpoint = '';
  vllmModel = '';
  loadedModelId = '';
  post('status', { message: `loading ${modelId}` });
  const selectedDevice = device === 'webgpu' ? 'webgpu' : 'wasm';
  const dtype = selectedDevice === 'webgpu' ? 'q4f16' : 'q8';
  const pipeline = await ensureHfPipeline();
  generator = await pipeline('text-generation', modelId, {
    device: selectedDevice,
    dtype,
    progress_callback: (progress) => {
      const file = progress.file || progress.name || '';
      const status = progress.status || '';
      const loaded = Number(progress.loaded || 0);
      const total = Number(progress.total || 0);
      post('progress', { file, status, loaded, total });
    },
  });
  loadedModelId = modelId;
  post('loaded', { modelId, device: selectedDevice, dtype });
}

async function loadVllm({ modelId, endpoint }) {
  const model = String(modelId || '').replace(/^vllm:/, '').trim();
  vllmEndpoint = String(endpoint || '').trim();
  if (!vllmEndpoint) throw new Error('vLLM backend requires ?vllmEndpoint=<chat-completions-url>.');
  vllmModel = model || 'Qwen/Qwen3.5-9B';
  generator = null;
  modelStackRuntime = null;
  modelStackManifestUrl = '';
  modelStackTokenizer = null;
  modelStackDevice = '';
  modelStackRuntimeDtype = '';
  loadedModelId = `vllm:${vllmModel}`;
  post('status', { message: `using local vLLM ${vllmModel}` });
  post('loaded', { modelId: loadedModelId, device: 'vllm', dtype: 'server' });
}

function modelStackUrl(modelId) {
  return String(modelId || '').replace(/^modelstack:/, '').trim();
}

function localRuntimeManifest(manifest) {
  const cloned = JSON.parse(JSON.stringify(manifest || {}));
  cloned.runtime ??= {};
  cloned.runtime.files ??= {};
  cloned.runtime.files.wgsl = MODEL_STACK_BITNET_WGSL_URL;
  return cloned;
}

function isGenerationCancelled(generationId) {
  return generationId && cancelledGenerationIds.has(Number(generationId));
}

function unloadRuntime() {
  try {
    modelStackRuntime?.device?.destroy?.();
  } catch (_error) {
    // WebGPU device destruction is best effort; terminating the worker also releases it.
  }
  generator = null;
  loadedModelId = '';
  modelStackRuntime = null;
  modelStackManifestUrl = '';
  modelStackTokenizer = null;
  modelStackDevice = '';
  modelStackRuntimeDtype = '';
  vllmEndpoint = '';
  vllmModel = '';
  activeGenerationId = 0;
  cancelledGenerationIds.clear();
  post('unloaded');
}

function byteEncode(text, maxLength = 1024) {
  const bytes = new TextEncoder().encode(String(text || ''));
  const ids = [1];
  for (const byte of bytes) {
    if (ids.length >= maxLength - 1) break;
    ids.push(Number(byte) + 4);
  }
  ids.push(2);
  return ids;
}

function byteDecode(ids) {
  const bytes = [];
  for (const id of ids) {
    const value = Number(id);
    if (value === 2) break;
    if (value >= 4 && value <= 259) bytes.push(value - 4);
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

function byteToUnicodeMap() {
  const bs = [];
  for (let i = 33; i <= 126; i += 1) bs.push(i);
  for (let i = 161; i <= 172; i += 1) bs.push(i);
  for (let i = 174; i <= 255; i += 1) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const byteEncoder = new Map();
  const byteDecoder = new Map();
  for (let i = 0; i < bs.length; i += 1) {
    const ch = String.fromCodePoint(cs[i]);
    byteEncoder.set(bs[i], ch);
    byteDecoder.set(ch, bs[i]);
  }
  return { byteEncoder, byteDecoder };
}

const BYTE_UNICODE = byteToUnicodeMap();
const BPE_PAIR_SEP = '\u0001';
const GPT2_PRETOKEN_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function createByteTokenizer() {
  return {
    bosTokenId: 1,
    eosTokenId: 2,
    encode: byteEncode,
    decode: byteDecode,
  };
}

function createBpeTokenizer(tokenizerJson) {
  const model = tokenizerJson?.model || {};
  const vocab = model.vocab || {};
  const idToToken = [];
  for (const [token, id] of Object.entries(vocab)) idToToken[Number(id)] = token;
  const ranks = new Map();
  for (const [rank, merge] of (model.merges || []).entries()) {
    const pair = Array.isArray(merge) ? merge : String(merge).split(/\s+/);
    if (pair.length >= 2) ranks.set(`${pair[0]}${BPE_PAIR_SEP}${pair[1]}`, rank);
  }
  const specialIds = new Set(
    (tokenizerJson?.added_tokens || [])
      .filter((item) => item?.special)
      .map((item) => Number(item.id)),
  );
  const specialTokenEntries = (tokenizerJson?.added_tokens || [])
    .filter((item) => item?.special && typeof item.content === 'string' && item.content.length > 0)
    .map((item) => ({ token: item.content, id: Number(item.id) }))
    .filter((item) => Number.isFinite(item.id))
    .sort((a, b) => b.token.length - a.token.length);
  const padTokenId = Number(vocab['<pad>'] ?? 0);
  const bosTokenId = Number(vocab['<s>'] ?? 1);
  const eosTokenId = Number(vocab['</s>'] ?? 2);
  const unkTokenId = Number(vocab['<unk>'] ?? 3);

  function bpeToken(byteLevelToken) {
    let word = Array.from(byteLevelToken);
    while (word.length > 1) {
      let bestIndex = -1;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i += 1) {
        const rank = ranks.get(`${word[i]}${BPE_PAIR_SEP}${word[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;
      const merged = `${word[bestIndex]}${word[bestIndex + 1]}`;
      const next = [];
      for (let i = 0; i < word.length; i += 1) {
        if (i === bestIndex) {
          next.push(merged);
          i += 1;
        } else {
          next.push(word[i]);
        }
      }
      word = next;
    }
    return word;
  }

  function splitSpecialSegments(text) {
    const source = String(text || '');
    if (!specialTokenEntries.length) return [{ text: source, specialId: null }];
    const segments = [];
    let cursor = 0;
    while (cursor < source.length) {
      let match = null;
      let matchIndex = -1;
      for (const entry of specialTokenEntries) {
        const index = source.indexOf(entry.token, cursor);
        if (index >= 0 && (matchIndex < 0 || index < matchIndex)) {
          match = entry;
          matchIndex = index;
        }
      }
      if (!match) {
        segments.push({ text: source.slice(cursor), specialId: null });
        break;
      }
      if (matchIndex > cursor) {
        segments.push({ text: source.slice(cursor, matchIndex), specialId: null });
      }
      segments.push({ text: match.token, specialId: match.id });
      cursor = matchIndex + match.token.length;
    }
    return segments;
  }

  function encodePlainSegment(text, ids, maxLength) {
    const pieces = String(text || '').match(GPT2_PRETOKEN_PATTERN) || [];
    for (const piece of pieces) {
      const bytes = new TextEncoder().encode(piece);
      let byteLevel = '';
      for (const byte of bytes) byteLevel += BYTE_UNICODE.byteEncoder.get(byte);
      for (const token of bpeToken(byteLevel)) {
        if (ids.length >= maxLength - 1) break;
        ids.push(Number(vocab[token] ?? unkTokenId));
      }
      if (ids.length >= maxLength - 1) break;
    }
  }

  function encode(text, maxLength = 1024) {
    const ids = [bosTokenId];
    for (const segment of splitSpecialSegments(text)) {
      if (ids.length >= maxLength - 1) break;
      if (segment.specialId !== null) {
        ids.push(segment.specialId);
      } else {
        encodePlainSegment(segment.text, ids, maxLength);
      }
    }
    ids.push(eosTokenId);
    return ids;
  }

  function decode(ids) {
    let byteLevel = '';
    for (const rawId of ids) {
      const id = Number(rawId);
      if (id === eosTokenId) break;
      if (id === padTokenId || specialIds.has(id)) continue;
      byteLevel += idToToken[id] || '';
    }
    const bytes = [];
    for (const ch of Array.from(byteLevel)) {
      const byte = BYTE_UNICODE.byteDecoder.get(ch);
      if (byte !== undefined) bytes.push(byte);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  return {
    bosTokenId,
    eosTokenId,
    padTokenId,
    unkTokenId,
    specialIds,
    encode,
    decode,
  };
}

async function tokenizerFromManifest(manifest, manifestUrl) {
  const tokenizer = manifest?.tokenizer || {};
  const tokenizerKind = String(tokenizer.kind || '').toLowerCase();
  if ((tokenizerKind === 'agentkernel-bpe' || tokenizerKind === 'agentkernel_bytelevel_bpe_v1') && tokenizer.path) {
    const tokenizerUrl = new URL(tokenizer.path, new URL('.', manifestUrl)).href;
    const tokenizerJson = await fetch(tokenizerUrl, { mode: 'cors' }).then((response) => {
      if (!response.ok) throw new Error(`model-stack tokenizer failed: ${response.status}`);
      return response.json();
    });
    return createBpeTokenizer(tokenizerJson);
  }
  return createByteTokenizer();
}

function repeatsTail(generatedIds, tokenId, maxTail = 3) {
  let count = 0;
  for (let i = generatedIds.length - 1; i >= 0; i -= 1) {
    if (generatedIds[i] !== tokenId) break;
    count += 1;
    if (count >= maxTail) return true;
  }
  return false;
}

function wouldRepeatNgram(generatedIds, tokenId, ngramSize = 4) {
  if (ngramSize <= 1 || generatedIds.length < ngramSize - 1) return false;
  const prefix = generatedIds.slice(generatedIds.length - ngramSize + 1);
  for (let i = 0; i <= generatedIds.length - ngramSize; i += 1) {
    let matches = true;
    for (let j = 0; j < prefix.length; j += 1) {
      if (generatedIds[i + j] !== prefix[j]) {
        matches = false;
        break;
      }
    }
    if (matches && generatedIds[i + prefix.length] === tokenId) return true;
  }
  return false;
}

function repetitionAdjustedLogit(value, tokenId, generatedIds, penalty = 1.16) {
  if (!generatedIds.includes(tokenId)) return value;
  return value >= 0 ? value / penalty : value * penalty;
}

function sampleFromCandidates(candidates, temperature, topP) {
  if (!candidates.length) return 0;
  if (temperature <= 0) return candidates[0].id;
  const scaled = candidates.map((candidate) => ({
    id: candidate.id,
    value: candidate.value / Math.max(temperature, 1e-4),
  }));
  const maxValue = Math.max(...scaled.map((candidate) => candidate.value));
  let total = 0;
  const probs = scaled.map((candidate) => {
    const weight = Math.exp(candidate.value - maxValue);
    total += weight;
    return { id: candidate.id, weight };
  });
  probs.sort((a, b) => b.weight - a.weight);
  const targetP = Math.max(0.01, Math.min(1, topP));
  let cumulative = 0;
  const kept = [];
  for (const candidate of probs) {
    kept.push(candidate);
    cumulative += candidate.weight / Math.max(total, 1e-12);
    if (cumulative >= targetP) break;
  }
  const keptTotal = kept.reduce((sum, candidate) => sum + candidate.weight, 0);
  let sample = Math.random() * Math.max(keptTotal, 1e-12);
  for (const candidate of kept) {
    sample -= candidate.weight;
    if (sample <= 0) return candidate.id;
  }
  return kept[0]?.id ?? candidates[0].id;
}

function adjustedTokenCandidates(logits, decLength, vocabSize, generatedIds, tokenizer, options = {}) {
  const offset = (decLength - 1) * vocabSize;
  const repetitionPenalty = Math.max(1, Math.min(2, Number(options.repetitionPenalty ?? 1.16)));
  const eosTokenId = Number(tokenizer.eosTokenId || 2);
  const blocked = new Set([
    Number(tokenizer.padTokenId ?? 0),
    Number(tokenizer.bosTokenId ?? 1),
    Number(tokenizer.unkTokenId ?? 3),
  ]);
  for (const id of tokenizer.specialIds || []) {
    if (Number(id) !== eosTokenId) blocked.add(Number(id));
  }
  const candidates = [];
  for (let i = 0; i < vocabSize; i += 1) {
    if (blocked.has(i)) continue;
    if (repeatsTail(generatedIds, i, 3)) continue;
    if (wouldRepeatNgram(generatedIds, i, 4)) continue;
    const raw = Number(logits[offset + i]);
    if (!Number.isFinite(raw)) continue;
    const value = repetitionAdjustedLogit(raw, i, generatedIds, repetitionPenalty);
    candidates.push({ id: i, value });
  }
  candidates.sort((a, b) => b.value - a.value);
  return candidates;
}

function blockedTokenIds(tokenizer) {
  const eosTokenId = Number(tokenizer.eosTokenId || 2);
  const blocked = new Set([
    Number(tokenizer.padTokenId ?? 0),
    Number(tokenizer.bosTokenId ?? 1),
    Number(tokenizer.unkTokenId ?? 3),
  ]);
  for (const id of tokenizer.specialIds || []) {
    if (Number(id) !== eosTokenId) blocked.add(Number(id));
  }
  return Array.from(blocked);
}

function selectNextToken(logits, decLength, vocabSize, generatedIds, tokenizer, options = {}) {
  const temperature = Math.max(0, Math.min(1.2, Number(options.temperature ?? 0.35)));
  const topP = Math.max(0.01, Math.min(1, Number(options.topP ?? 0.9)));
  const candidates = adjustedTokenCandidates(logits, decLength, vocabSize, generatedIds, tokenizer, options);
  const eosTokenId = Number(tokenizer.eosTokenId || 2);
  return candidates.length ? sampleFromCandidates(candidates, temperature, topP) : eosTokenId;
}

function tokenProbability(logits, decLength, vocabSize, generatedIds, tokenizer, tokenId, options = {}) {
  const temperature = Math.max(0.001, Math.min(1.2, Number(options.temperature ?? 0.35)));
  const topP = Math.max(0.01, Math.min(1, Number(options.topP ?? 0.9)));
  const candidates = adjustedTokenCandidates(logits, decLength, vocabSize, generatedIds, tokenizer, options);
  if (!candidates.length) return { probability: 0, rank: Infinity, topProbability: 0 };
  const scaled = candidates.map((candidate) => ({
    id: candidate.id,
    value: candidate.value / temperature,
  }));
  const maxValue = scaled[0].value;
  let total = 0;
  const probs = scaled.map((candidate, index) => {
    const weight = Math.exp(candidate.value - maxValue);
    total += weight;
    return { id: candidate.id, weight, index };
  });
  let cumulative = 0;
  const kept = [];
  for (const item of probs) {
    kept.push(item);
    cumulative += item.weight / Math.max(total, 1e-12);
    if (cumulative >= topP) break;
  }
  const keptTotal = kept.reduce((sum, item) => sum + item.weight, 0);
  const found = kept.find((item) => item.id === Number(tokenId));
  return {
    probability: found ? found.weight / Math.max(keptTotal, 1e-12) : 0,
    rank: found ? found.index + 1 : Infinity,
    topProbability: kept[0] ? kept[0].weight / Math.max(keptTotal, 1e-12) : 0,
  };
}

function finalLogitRow(logits, rows, vocabSize) {
  const rowCount = Math.max(1, Number(rows || 1));
  const start = (rowCount - 1) * vocabSize;
  return logits.slice(start, start + vocabSize);
}

function proposeNgramDraft(generatedIds, maxDraft = 4) {
  const ids = Array.from(generatedIds || [], Number);
  if (ids.length < 6) return [];
  for (let n = Math.min(5, Math.floor(ids.length / 2)); n >= 2; n -= 1) {
    const suffix = ids.slice(ids.length - n);
    for (let i = ids.length - n - 1; i >= 0; i -= 1) {
      let matched = true;
      for (let j = 0; j < n; j += 1) {
        if (ids[i + j] !== suffix[j]) {
          matched = false;
          break;
        }
      }
      if (!matched) continue;
      const draft = ids.slice(i + n, i + n + maxDraft);
      if (draft.length) return draft;
    }
  }
  return [];
}

function createSpeculationStats() {
  return {
    attempts: 0,
    acceptedSpans: 0,
    draftedTokens: 0,
    acceptedTokens: 0,
    cooldownUntilStep: 0,
    lastReportAccepted: 0,
  };
}

function shouldAttemptSpeculation(stats, step, generatedCount, isControlDecode) {
  if (step < Number(stats.cooldownUntilStep || 0)) return false;
  if (!isControlDecode && generatedCount < 24) return false;
  if (stats.attempts < 6) return true;
  const tokenRate = stats.draftedTokens ? stats.acceptedTokens / stats.draftedTokens : 0;
  return tokenRate >= 0.18;
}

function recordSpeculation(stats, draftLength, acceptedLength, step) {
  stats.attempts += 1;
  stats.draftedTokens += Math.max(0, Number(draftLength || 0));
  stats.acceptedTokens += Math.max(0, Number(acceptedLength || 0));
  if (acceptedLength > 0) stats.acceptedSpans += 1;
  if (stats.attempts >= 6 && stats.draftedTokens > 0 && stats.acceptedTokens / stats.draftedTokens < 0.12) {
    stats.cooldownUntilStep = step + 24;
  }
}

async function loadModelStack({ modelId, device }) {
  const manifestUrl = modelStackUrl(modelId);
  if (!manifestUrl) throw new Error('modelstack model id must be modelstack:<manifest-url>');
  if (modelStackRuntime && modelStackManifestUrl === manifestUrl) {
    post('loaded', { modelId, device: modelStackDevice || 'runtime', dtype: modelStackRuntimeDtype || 'runtime' });
    return;
  }
  generator = null;
  modelStackRuntime = null;
  modelStackManifestUrl = '';
  modelStackTokenizer = null;
  modelStackDevice = '';
  modelStackRuntimeDtype = '';
  post('status', { message: 'loading model manifest' });
  const manifest = await fetch(manifestUrl, { mode: 'cors' }).then((response) => {
    if (!response.ok) throw new Error(`model-stack manifest failed: ${response.status}`);
    return response.json();
  });
  const bitnetLayerCount = Array.isArray(manifest?.layers) ? manifest.layers.length : 0;
  const denseTensorCount = manifest?.dense_tensors ? Object.keys(manifest.dense_tensors).length : 0;
  const runtimeDtype = bitnetLayerCount > 0 ? 'bitnet-hybrid' : 'dense-f32';
  const runtimeManifest = localRuntimeManifest(manifest);
  post('status', { message: 'loading app-hosted BitNet runtime module' });
  const layerConcurrency = Math.max(2, Math.min(8, Number(navigator.hardwareConcurrency || 4)));
  const runtimeModule = await import(MODEL_STACK_BITNET_RUNTIME_URL);
  const progress = (progress) => {
    post('status', {
      message: progress?.message || 'loading model runtime',
      phase: progress?.phase || '',
      index: progress?.index || 0,
      total: progress?.total || 0,
      name: progress?.name || '',
    });
  };
  const requestedDevice = typeof device === 'string' ? device : '';
  const wasmAvailable = Boolean(runtimeModule.BitNetEncoderDecoderWASM && runtimeManifest?.runtime?.fallback === 'wasm');
  if (requestedDevice !== 'wasm' && navigator.gpu && runtimeModule.BitNetEncoderDecoderWebGPU) {
    try {
      post('status', { message: 'starting BitNet WebGPU runtime' });
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable.');
      const gpuDevice = await adapter.requestDevice();
      post('status', { message: `preparing BitNet WebGPU layer loader (${layerConcurrency} parallel)` });
      modelStackRuntime = await runtimeModule.BitNetEncoderDecoderWebGPU.fromManifestUrl(gpuDevice, manifestUrl, {
        manifest: runtimeManifest,
        layerConcurrency,
        progress,
      });
      modelStackDevice = 'webgpu';
    } catch (error) {
      if (!wasmAvailable) throw error;
      post('status', { message: `WebGPU unavailable; falling back to BitNet WASM (${error.message || String(error)})` });
    }
  }
  if (!modelStackRuntime && wasmAvailable) {
    post('status', { message: `starting BitNet WASM runtime (${layerConcurrency} parallel layer load)` });
    modelStackRuntime = await runtimeModule.BitNetEncoderDecoderWASM.fromManifestUrl(manifestUrl, {
      manifest: runtimeManifest,
      layerConcurrency,
      progress,
    });
    modelStackDevice = 'wasm';
  }
  if (!modelStackRuntime) {
    throw new Error('BitNet WebGPU is unavailable and this bundle does not include a BitNet WASM fallback.');
  }
  post('status', { message: 'loading tokenizer' });
  modelStackTokenizer = await tokenizerFromManifest(manifest, manifestUrl);
  post('status', { message: 'model runtime ready' });
  modelStackManifestUrl = manifestUrl;
  loadedModelId = modelId;
  modelStackRuntimeDtype = runtimeDtype;
  post('loaded', {
    modelId,
    device: modelStackDevice,
    dtype: runtimeDtype,
    bitnetLayerCount,
    denseTensorCount,
  });
}

async function generate({ prompt, options, generationId }) {
  const id = Number(generationId || Date.now());
  activeGenerationId = id;
  cancelledGenerationIds.delete(id);
  try {
    if (vllmEndpoint && vllmModel) {
      await generateVllm({ prompt, options, generationId: id });
      return;
    }
    if (modelStackRuntime) {
      await generateModelStack({ prompt, options, generationId: id });
      return;
    }
    if (!generator) throw new Error('Model is not loaded.');
    const maxNewTokens = Math.max(48, Math.min(1024, Number(options?.maxNewTokens || 560)));
    const temperature = Math.max(0, Math.min(1.2, Number(options?.temperature ?? 0.35)));
    post('status', { message: 'generating answer' });
    const result = await generator(prompt, {
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: 0.9,
      do_sample: temperature > 0,
      repetition_penalty: 1.08,
    });
    if (isGenerationCancelled(id)) {
      post('cancelled', { generationId: id });
      return;
    }
    post('generated', { text: normalizeGeneratedText(result, prompt), generationId: id });
  } finally {
    if (activeGenerationId === id) activeGenerationId = 0;
    cancelledGenerationIds.delete(id);
  }
}

async function generateVllm({ prompt, options, generationId }) {
  const maxTokens = Math.max(48, Math.min(1536, Number(options?.maxNewTokens || 560)));
  const temperature = Math.max(0, Math.min(1.2, Number(options?.temperature ?? 0.35)));
  post('status', { message: 'generating with local vLLM' });
  const response = await fetch(vllmEndpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: vllmModel,
      messages: [{ role: 'user', content: String(prompt || '') }],
      max_tokens: maxTokens,
      temperature,
      top_p: 0.9,
      chat_template_kwargs: { enable_thinking: false },
    }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error?.message || data?.message || `vLLM request failed: ${response.status}`);
  }
  if (isGenerationCancelled(generationId)) {
    post('cancelled', { generationId });
    return;
  }
  const message = data?.choices?.[0]?.message || {};
  const text = String(message.content || message.reasoning || '').trim();
  post('generated', { text: text || 'No answer generated.', generationId });
}

async function generateModelStack({ prompt, options, generationId }) {
  if (!modelStackRuntime) throw new Error('Model-stack runtime is not loaded.');
  const maxNewTokens = Math.max(16, Math.min(768, Number(options?.maxNewTokens || 320)));
  const maxEncoderTokens = Math.max(128, Math.min(4096, Number(options?.maxEncoderTokens || 1024)));
  const temperature = Math.max(0, Math.min(1.2, Number(options?.temperature ?? 0.35)));
  const topP = Math.max(0.01, Math.min(1, Number(options?.topP ?? 0.9)));
  const tokenizer = modelStackTokenizer || createByteTokenizer();
  const encIds = tokenizer.encode(prompt, maxEncoderTokens);
  const decoderPrefix = String(options?.decoderPrefix || '');
  const prefixIds = decoderPrefix
    ? tokenizer.encode(decoderPrefix, Math.min(128, Math.max(16, decoderPrefix.length + 8))).slice(1, -1)
    : [];
  const generatedIds = [...prefixIds];
  const generatedIdsForSampler = new Uint32Array(maxNewTokens + prefixIds.length + 8);
  let generatedIdsForSamplerLength = 0;
  for (const id of generatedIds) {
    generatedIdsForSampler[generatedIdsForSamplerLength] = Number(id);
    generatedIdsForSamplerLength += 1;
  }
  const appendGeneratedId = (id) => {
    const tokenId = Number(id);
    generatedIds.push(tokenId);
    if (generatedIdsForSamplerLength < generatedIdsForSampler.length) {
      generatedIdsForSampler[generatedIdsForSamplerLength] = tokenId;
      generatedIdsForSamplerLength += 1;
    }
  };
  const appendGeneratedIds = (ids) => {
    for (const id of ids) appendGeneratedId(id);
  };
  let nextInputId = Number(tokenizer.bosTokenId || 1);
  let pendingLogits = null;
  const speculationStats = createSpeculationStats();
  const isControlDecode = Boolean(options?.stopOnDecision || decoderPrefix);
  const session = typeof modelStackRuntime.createGenerationSession === 'function'
    ? modelStackRuntime.createGenerationSession(encIds)
    : null;
  if (session?.prepare) {
    post('status', { message: 'encoding prompt once for cached decode' });
    await session.prepare();
    if (prefixIds.length && typeof session.nextMany === 'function') {
      const prefixInputIds = [nextInputId, ...prefixIds];
      const prefixLogits = await session.nextMany(prefixInputIds);
      pendingLogits = finalLogitRow(prefixLogits, prefixInputIds.length, Number(modelStackRuntime.graph?.vocab_size || modelStackRuntime.manifest?.model?.vocab_size || 260));
      nextInputId = Number(prefixIds[prefixIds.length - 1]);
    } else {
      for (const tokenId of prefixIds) {
        await session.next(nextInputId);
        nextInputId = Number(tokenId);
      }
    }
    post('status', { message: `generating with cached BitNet ${modelStackDevice || 'runtime'} decoder` });
  } else {
    post('status', { message: 'generating with model-stack encoder-decoder' });
  }
  const fastSamplerBlockedIds = Uint32Array.from(blockedTokenIds(tokenizer));
  for (let step = 0; step < maxNewTokens; step += 1) {
    if (isGenerationCancelled(generationId)) {
      post('cancelled', { generationId });
      return;
    }
    if (session && step % 4 === 0) {
      post('status', { message: `cached BitNet decoder working on token ${step + 1}` });
    }
    const canUseFastSampler = Boolean(
      session
      && !pendingLogits
      && !isControlDecode
      && typeof session.sampleNext === 'function'
    );
    const fastSample = canUseFastSampler
      ? await session.sampleNext(nextInputId, generatedIdsForSampler.subarray(0, generatedIdsForSamplerLength), {
          blockedIds: fastSamplerBlockedIds,
          temperature,
          topP,
          repetitionPenalty: 1.16,
          randomValue: Math.random(),
        })
      : null;
    const logits = fastSample ? null : (pendingLogits || (session
      ? await session.next(nextInputId)
      : await modelStackRuntime.forward(encIds, [Number(tokenizer.bosTokenId || 1), ...generatedIds])));
    pendingLogits = null;
    if (isGenerationCancelled(generationId)) {
      post('cancelled', { generationId });
      return;
    }
    const vocabSize = Number(modelStackRuntime.graph?.vocab_size || modelStackRuntime.manifest?.model?.vocab_size || 260);
    const nextId = fastSample
      ? Number(fastSample.tokenId)
      : selectNextToken(logits, session ? 1 : generatedIds.length + 1, vocabSize, generatedIds, tokenizer, {
          temperature,
          topP,
          repetitionPenalty: 1.16,
        });
    if (nextId === Number(tokenizer.eosTokenId || 2)) break;
    appendGeneratedId(nextId);
    nextInputId = nextId;
    const canSpeculate = Boolean(
      session
      && typeof session.nextMany === 'function'
      && typeof session.cloneState === 'function'
      && typeof session.restoreState === 'function'
      && generatedIds.length - prefixIds.length < maxNewTokens
      && shouldAttemptSpeculation(speculationStats, step, generatedIds.length - prefixIds.length, isControlDecode)
    );
    if (canSpeculate) {
      const remaining = maxNewTokens - (generatedIds.length - prefixIds.length);
      const draft = proposeNgramDraft(generatedIds, Math.min(4, Math.max(0, remaining)));
      if (draft.length >= (isControlDecode ? 1 : 2)) {
        const snapshot = session.cloneState();
        const verifyInput = [nextId, ...draft];
        const verifyLogits = await session.nextMany(verifyInput);
        let accepted = true;
        let acceptanceMode = 'strict';
        let acceptedDraftTokens = 0;
        const speculativeIds = [...generatedIds];
        for (let draftIndex = 0; draftIndex < draft.length; draftIndex += 1) {
          const row = verifyLogits.subarray(draftIndex * vocabSize, (draftIndex + 1) * vocabSize);
          if (temperature <= 0.001) {
            const expected = selectNextToken(row, 1, vocabSize, speculativeIds, tokenizer, {
              temperature: 0,
              topP: 1,
              repetitionPenalty: 1.16,
            });
            if (expected !== draft[draftIndex]) {
              accepted = false;
              break;
            }
          } else {
            acceptanceMode = 'probabilistic';
            const stats = tokenProbability(row, 1, vocabSize, speculativeIds, tokenizer, draft[draftIndex], {
              temperature,
              topP,
              repetitionPenalty: 1.16,
            });
            const highConfidence = stats.rank <= 3 && stats.probability >= Math.max(0.08, stats.topProbability * 0.34);
            if (!highConfidence || Math.random() > stats.probability) {
              accepted = false;
              break;
            }
          }
          speculativeIds.push(draft[draftIndex]);
          acceptedDraftTokens += 1;
        }
        if (accepted) {
          appendGeneratedIds(draft);
          nextInputId = generatedIds[generatedIds.length - 1];
          pendingLogits = finalLogitRow(verifyLogits, verifyInput.length, vocabSize);
          step += draft.length;
          recordSpeculation(speculationStats, draft.length, draft.length, step);
          post('status', { message: `${acceptanceMode} n-gram speculation accepted ${draft.length} tokens` });
        } else {
          session.restoreState(snapshot);
          recordSpeculation(speculationStats, draft.length, acceptedDraftTokens, step);
        }
      }
    }
    if (options?.stopOnDecision && step >= 1) {
      const partial = tokenizer.decode(generatedIds);
      if (/selected_candidate_id\s*=\s*(?:P?\d{1,2})(?:\s*,\s*P?\d{1,2}){0,2}(?:\s|$)/i.test(partial)) {
        post('status', { message: 'selector decision decoded' });
        break;
      }
    }
    if (step % 16 === 0) post('status', { message: `cached BitNet decoder generated ${step + 1} tokens` });
  }
  if (speculationStats.draftedTokens > 0) {
    const rate = Math.round((speculationStats.acceptedTokens / Math.max(1, speculationStats.draftedTokens)) * 100);
    post('status', {
      message: `speculation accepted ${speculationStats.acceptedTokens}/${speculationStats.draftedTokens} draft tokens (${rate}%)`,
    });
  }
  post('generated', { text: tokenizer.decode(generatedIds), generationId });
}

async function embedModelStack({ text, requestId, maxEncoderTokens }) {
  if (!modelStackRuntime) throw new Error('Model-stack runtime is not loaded.');
  if (typeof modelStackRuntime.retrievalQueryEmbedding !== 'function') {
    throw new Error('Loaded model does not expose neural retrieval embeddings.');
  }
  const tokenizer = modelStackTokenizer || createByteTokenizer();
  const encIds = tokenizer.encode(String(text || ''), Math.max(64, Math.min(1024, Number(maxEncoderTokens || 256))));
  post('status', { message: 'embedding query with AgentKernel BitNet encoder' });
  const embedding = await modelStackRuntime.retrievalQueryEmbedding(encIds);
  post('embedded', { requestId, embedding: Array.from(embedding) });
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const weights = values.map((value) => Math.exp(Number(value) - maxValue));
  const total = weights.reduce((sum, value) => sum + value, 0) || 1;
  return weights.map((value) => value / total);
}

async function classifyAgentIntentModelStack({ text, requestId, maxEncoderTokens }) {
  if (!modelStackRuntime) throw new Error('Model-stack runtime is not loaded.');
  if (typeof modelStackRuntime.agentIntentLogits !== 'function') {
    throw new Error('Loaded model does not expose agent intent logits.');
  }
  const tokenizer = modelStackTokenizer || createByteTokenizer();
  const encIds = tokenizer.encode(String(text || ''), Math.max(64, Math.min(1024, Number(maxEncoderTokens || 768))));
  post('status', { message: 'classifying agent intent with AgentKernel BitNet encoder' });
  const logits = Array.from(await modelStackRuntime.agentIntentLogits(encIds));
  const probabilities = softmax(logits);
  const ranked = probabilities
    .map((probability, index) => ({
      id: AGENT_INTENT_LABELS[index] || `intent_${index}`,
      index,
      probability,
    }))
    .sort((a, b) => b.probability - a.probability);
  post('intent', {
    requestId,
    intent: ranked[0]?.id || '',
    confidence: ranked[0]?.probability || 0,
    ranked: ranked.slice(0, 5),
  });
}

self.addEventListener('message', (event) => {
  const data = event.data || {};
  (async () => {
    if (data.type === 'load') {
      await loadModel(data);
    } else if (data.type === 'generate') {
      await generate(data);
    } else if (data.type === 'embed') {
      await embedModelStack(data);
    } else if (data.type === 'intent') {
      await classifyAgentIntentModelStack(data);
    } else if (data.type === 'cancel') {
      const generationId = Number(data.generationId || activeGenerationId || 0);
      if (generationId) cancelledGenerationIds.add(generationId);
    } else if (data.type === 'unload') {
      unloadRuntime();
    }
  })().catch((error) => {
    post('error', { message: error?.message || String(error) });
  });
});
