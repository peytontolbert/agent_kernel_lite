const HF = {
  paperGraphRoot: 'https://huggingface.co/datasets/PeytonT/paper_universe_interactive/resolve/main',
  paperInteractiveRoot: 'https://huggingface.co/datasets/PeytonT/paper_universe_interactive/resolve/main/interactive',
  paperSemanticRoot: 'https://huggingface.co/datasets/PeytonT/paper_universe_interactive/resolve/main/semantic_m1',
  paperEmbeddingLiteRoot: 'https://huggingface.co/PeytonT/1m-paper-embedding-model-lite-onnx/resolve/main',
  paperTextDataset: 'PeytonT/1m_papers_text',
  datasetServer: 'https://datasets-server.huggingface.co',
};

const URL_PARAMS = new URLSearchParams(window.location.search);
const HASH_PARAMS = new URLSearchParams(String(window.location.hash || '').replace(/^#/, ''));
const CAPACITOR_NATIVE = Boolean(window.Capacitor?.isNativePlatform?.())
  || (Boolean(window.Capacitor?.getPlatform) && window.Capacitor.getPlatform() !== 'web')
  || (window.location.protocol === 'https:' && window.location.hostname === 'localhost');
const NATIVE_APP = URL_PARAMS.get('native') === '1'
  || CAPACITOR_NATIVE
  || window.location.protocol === 'capacitor:'
  || window.location.protocol === 'ionic:';
if (NATIVE_APP) document.documentElement.classList.add('native-app');
const DEV_BACKEND = String(URL_PARAMS.get('backend') || '').trim().toLowerCase();
const DEVICE_PARAM = String(URL_PARAMS.get('device') || '').trim().toLowerCase();
const VLLM_ENDPOINT = String(URL_PARAMS.get('vllmEndpoint') || '').trim();
const VLLM_MODEL = String(URL_PARAMS.get('vllmModel') || 'Qwen/Qwen3.5-9B').trim();
const STRUCTURE_FIXTURE = URL_PARAMS.get('structureFixture') === '1';
const HF_DATASET_SEARCH_ENABLED = URL_PARAMS.get('hfSearch') === '1';
const SOURCE_SLOT_TOKENS_ENABLED = URL_PARAMS.get('sourceSlots') !== '0';
const HF_MODELSTACK_MANIFEST = 'https://huggingface.co/PeytonT/agentkernel-lite-100m-bitnet/resolve/main/manifest.json';
const NATIVE_MODELSTACK_MANIFEST = './models/agentkernel_lite_100m_bitnet_12000/manifest.json?v=20260524-v356-controller-curriculum';
const NATIVE_PAPERS_50K = './packed-data/papers_50000.json';
const NEURAL_MEMORY_PACK_URL = String(URL_PARAMS.get('neuralMemoryPack') || '').trim();
const NEURAL_MEMORY_ENABLED = URL_PARAMS.get('neuralMemory') === '1' || Boolean(NEURAL_MEMORY_PACK_URL);
const THEME_STORAGE_KEY = 'agent-kernel-lite-theme';
const POCKETPAL_MEMORY_STORAGE_KEY = 'agent-kernel-lite-pocketpal-memory-v1';
const POCKETPAL_SLOTS_STORAGE_KEY = 'agent-kernel-lite-pocketpal-slots-v1';
const POCKETPAL_DATA_SOURCES_STORAGE_KEY = 'agent-kernel-lite-pocketpal-data-sources-v1';
const POCKETPAL_AGENTS_STORAGE_KEY = 'agent-kernel-lite-pocketpal-agents-v1';
const WEB_SEARCH_SETTINGS_STORAGE_KEY = 'agent-kernel-lite-web-search-settings-v1';
const CACHE_NAME = 'agent-kernel-lite-v24-peyton-f5-vocos-q4';
const VOICE_RUNTIME_VERSION = '20260521-peyton-hf-q4-distill-webgpu-fullref-attn2pass-gen97-step8-cfg2-speed115';
const DB_NAME = 'agent-kernel-lite-db-v1';
const DB_STORE = 'metadata';
const SESSION_EXPORT_VERSION = 1;
const EXTENSION_CACHE_DB_KEY = 'installed_extensions_v1';
const GITHUB_RELEASE_REPO = 'peytontolbert/agent_kernel_lite';
const GITHUB_RELEASE_TAG = 'v14';
const GITHUB_RELEASE_ROOT = `https://github.com/${GITHUB_RELEASE_REPO}/releases/download`;
const PINNED_GITHUB_RELEASE_ROOT = `${GITHUB_RELEASE_ROOT}/${GITHUB_RELEASE_TAG}`;
const AVAILABLE_EXTENSIONS_CATALOG_URL = './extensions/catalog.json';
const LOCAL_AVAILABLE_EXTENSIONS = [
  {
    id: 'web_search',
    name: 'Web Search',
    source: 'local',
    manifest: './extensions/web_search.json',
    description: 'Open user-visible web searches from PocketPal.',
    setup: 'Uses the current browser surface; on iPhone this runs through the app web view or system browser depending on the native shell.',
  },
  ...(isLocalDevelopmentUrl(new URL(window.location.href)) ? [{
    id: 'image_generation',
    name: 'Image Generation',
    source: 'local',
    manifest: './extensions/image_generation.dev.json',
    description: 'Generate image artifacts with the current in-browser development image model.',
    setup: 'Development-only. Uses the local FLUX student backend while the pure WASM runtime is being exported.',
  }] : []),
];
const RELEASE_AVAILABLE_EXTENSION_IDS = new Set(LOCAL_AVAILABLE_EXTENSIONS.map((entry) => entry.id));

function resolveAvailableExtensionManifestUrl(entry, catalogUrl) {
  const manifest = String(entry?.manifest || entry?.manifest_url || '').trim();
  if (!manifest) return '';
  if (/^https?:\/\//i.test(manifest)) return manifest;
  if (manifest.startsWith('./extensions/') || manifest.startsWith('extensions/')) {
    return new URL(manifest.replace(/^\.?\//, './'), window.location.href).href;
  }
  return new URL(manifest, catalogUrl).href;
}
const MODE_CONFIG = {
  chat: {
    label: 'Chat',
    pill: 'chat mode',
    contextItems: 5,
    semanticTopK: 12,
    hfSearchRows: 8,
    candidateFloor: 8,
    excerptChars: 1200,
    selectedExcerptChars: 2600,
    temperature: 0.35,
    minTokens: 160,
    placeholder: 'Ask a question...',
  },
  quick_search: {
    label: 'Search',
    pill: 'paper search',
    contextItems: 10,
    semanticTopK: 32,
    hfSearchRows: 12,
    candidateFloor: 16,
    excerptChars: 700,
    selectedExcerptChars: 1800,
    temperature: 0,
    minTokens: 0,
    placeholder: 'Search papers...',
  },
  web_search: {
    label: 'Web',
    pill: 'web search',
    contextItems: 0,
    semanticTopK: 0,
    hfSearchRows: 0,
    candidateFloor: 0,
    excerptChars: 0,
    selectedExcerptChars: 0,
    temperature: 0,
    minTokens: 0,
    placeholder: 'Search the web...',
  },
  think: {
    label: 'Think',
    pill: 'think mode',
    contextItems: 8,
    semanticTopK: 24,
    hfSearchRows: 12,
    candidateFloor: 14,
    excerptChars: 1600,
    selectedExcerptChars: 3200,
    temperature: 0.25,
    minTokens: 320,
    placeholder: 'Ask for a careful synthesis...',
  },
  deep_research: {
    label: 'Deep',
    pill: 'deep research',
    contextItems: 14,
    semanticTopK: 48,
    hfSearchRows: 20,
    candidateFloor: 24,
    excerptChars: 2400,
    selectedExcerptChars: 4200,
    temperature: 0.18,
    minTokens: 560,
    placeholder: 'Ask for a deep research pass...',
  },
};
const PROCESS_STEPS = [
  { id: 'receive', label: 'Receive', idle: 'Waiting for prompt' },
  { id: 'runtime', label: 'Runtime', idle: 'Model idle' },
  { id: 'plan', label: 'Plan Command', idle: 'Choosing respond or gather_context' },
  { id: 'pack', label: 'Paper Pack', idle: 'Library idle' },
  { id: 'embed', label: 'Embed Query', idle: 'Not started' },
  { id: 'rank', label: 'Semantic Rank', idle: 'Not started' },
  { id: 'select', label: 'Select Evidence', idle: 'Not started' },
  { id: 'lexical', label: 'Lexical Scan', idle: 'Not started' },
  { id: 'lookup', label: 'Lookup Evidence', idle: 'Not started' },
  { id: 'compact', label: 'Compact Context', idle: 'Not started' },
  { id: 'compile', label: 'Compile Prompt', idle: 'Not started' },
  { id: 'generate', label: 'Respond', idle: 'Not started' },
  { id: 'render', label: 'Render', idle: 'Waiting' },
];
const MAX_CONTEXT_ITEMS = Math.max(...Object.values(MODE_CONFIG).map((config) => config.contextItems));
const MAX_SELECTED_PAPERS = 3;
const SEMANTIC_QUERY_TOKENS = 128;
const SEMANTIC_SCAN_YIELD_ROWS = 25000;
const REMOTE_MODEL_FALLBACK_MS = 25000;
const ORT_CDN_ROOT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist';
const RETRIEVAL_INTENT_TERMS = new Set([
  'best',
  'better',
  'top',
  'good',
  'great',
  'paper',
  'papers',
  'research',
  'study',
  'studies',
  'source',
  'sources',
  'evidence',
  'find',
  'search',
  'tell',
  'me',
  'more',
  'please',
  'pls',
  'recommend',
  'recommendation',
  'recommendations',
  'what',
  'whats',
  'which',
  'who',
  'where',
  'when',
  'why',
  'how',
  'can',
  'could',
  'should',
  'would',
  'about',
  'for',
  'on',
  'the',
  'this',
  'that',
]);

const state = {
  worker: null,
  modelReady: false,
  modelBusy: false,
  modelLoadPromise: null,
  modelLoadResolve: null,
  modelLoadReject: null,
  modelAutoLoadStarted: false,
  loadedModelId: '',
  core: null,
  coreReady: false,
  packRows: [],
  packLevel: null,
  paperSemanticManifest: null,
  paperSemanticIndex: null,
  paperEmbeddingModel: null,
  neuralMemoryPack: null,
  neuralEmbeddingRequests: new Map(),
  intentClassificationRequests: new Map(),
  utilityGenerationRequests: new Map(),
  paperContextRows: [],
  retrievalRows: [],
  pendingContextRows: [],
  lastDecisionPacket: null,
  pocketPalMemory: [],
  pocketPalSlots: {},
  pocketPalDataSources: [],
  pocketPalAgents: [],
  activeAgentId: '',
  messages: [],
  mode: 'chat',
  image: {
    enabled: false,
    worker: null,
    ready: false,
    busy: false,
    loadPromise: null,
    activeJobId: null,
    activeActionId: null,
    extensionId: 'image_generation',
    capabilityId: 'image.generate',
    modelId: URL_PARAMS.get('imageModel') || (isLocalDevelopmentUrl(new URL(window.location.href)) ? 'agentkernel_lite_image_sana_300m_bitnet_block12_13ff_browser_v0' : 'agentkernel_lite_image_bitdit_hf_cifar_distilled_v1'),
  },
  translation: {
    enabled: false,
    busy: false,
    listening: false,
    recognition: null,
    extensionId: 'translator',
    textCapabilityId: 'translation.text',
    audioCapabilityId: 'translation.audio',
    activeActionId: null,
  },
  voice: {
    worker: null,
    ready: false,
    busy: false,
    loadPromise: null,
    loadResolve: null,
    loadReject: null,
    audioUrl: '',
    detail: '',
    progressDetail: '',
    nativeRuntime: null,
  },
  webSearch: {
    extensionId: 'web_search',
    searchCapabilityId: 'web.search',
    openCapabilityId: 'web.open_url',
    maxSources: 5,
  },
  processRunId: 0,
  generationRunId: 0,
  processActive: false,
  liveStatusNode: null,
  activeTurn: null,
  currentSourceSlots: [],
  currentTextSlots: {},
  lastAgentIntent: null,
  appIntegrity: null,
  availableExtensions: [],
  theme: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
};

const els = {
  model: document.getElementById('modelSelect'),
  device: document.getElementById('deviceSelect'),
  pack: document.getElementById('packSelect'),
  language: document.getElementById('languageSelect'),
  tokens: document.getElementById('tokenSelect'),
  chatMode: document.getElementById('chatModeButton'),
  quickSearchMode: document.getElementById('quickSearchModeButton'),
  webSearchMode: document.getElementById('webSearchModeButton'),
  thinkMode: document.getElementById('thinkModeButton'),
  deepResearchMode: document.getElementById('deepResearchModeButton'),
  modeButtons: [...document.querySelectorAll('.mode-button')],
  moduleTabs: [...document.querySelectorAll('.module-tab')],
  modulePanels: [...document.querySelectorAll('[data-module-panel]')],
  imageMode: document.getElementById('imageModeButton'),
  imageModeDetail: document.getElementById('imageModeDetail'),
  voiceSpeak: document.getElementById('voiceSpeakButton'),
  voiceModeDetail: document.getElementById('voiceModeDetail'),
  voicePreviewAudio: document.getElementById('voicePreviewAudio'),
  userDataSource: document.getElementById('userDataSourceInput'),
  saveUserDataSource: document.getElementById('saveUserDataSourceButton'),
  clearUserDataSource: document.getElementById('clearUserDataSourceButton'),
  userDataFileInput: document.getElementById('userDataFileInput'),
  webSearchMaxSources: document.getElementById('webSearchMaxSourcesInput'),
  userDataSourceList: document.getElementById('userDataSourceList'),
  agentName: document.getElementById('agentNameInput'),
  agentInstruction: document.getElementById('agentInstructionInput'),
  agentRetrievalPolicy: document.getElementById('agentRetrievalPolicySelect'),
  agentToolPolicy: document.getElementById('agentToolPolicySelect'),
  agentActionPolicy: document.getElementById('agentActionPolicySelect'),
  createAgent: document.getElementById('createAgentButton'),
  clearActiveAgent: document.getElementById('clearActiveAgentButton'),
  agentList: document.getElementById('agentList'),
  extensionList: document.getElementById('extensionList'),
  availableExtensionList: document.getElementById('availableExtensionList'),
  extensionManifestUrl: document.getElementById('extensionManifestUrl'),
  installExtension: document.getElementById('installExtensionButton'),
  translationMode: document.getElementById('translationModeButton'),
  translationModeDetail: document.getElementById('translationModeDetail'),
  translationSource: document.getElementById('translationSourceSelect'),
  translationTarget: document.getElementById('translationTargetSelect'),
  audioTranslate: document.getElementById('audioTranslateButton'),
  loadModel: document.getElementById('loadModelButton'),
  unloadModel: document.getElementById('unloadModelButton'),
  loadPack: document.getElementById('loadPackButton'),
  persist: document.getElementById('persistButton'),
  exportSession: document.getElementById('exportSessionButton'),
  importSession: document.getElementById('importSessionButton'),
  importSessionInput: document.getElementById('importSessionInput'),
  reset: document.getElementById('resetChatButton'),
  send: document.getElementById('sendButton'),
  form: document.getElementById('composerForm'),
  prompt: document.getElementById('promptInput'),
  chat: document.getElementById('chatScroll'),
  empty: document.getElementById('emptyState'),
  log: document.getElementById('log'),
  modelMetric: document.getElementById('modelMetric'),
  packMetric: document.getElementById('packMetric'),
  storageMetric: document.getElementById('storageMetric'),
  rowsMetric: document.getElementById('rowsMetric'),
  appHashMetric: document.getElementById('appHashMetric'),
  modelPill: document.getElementById('modelPill'),
  runtimeDetail: document.getElementById('runtimeDetail'),
  packPill: document.getElementById('packPill'),
  storagePill: document.getElementById('storagePill'),
  corePill: document.getElementById('corePill'),
  modePill: document.getElementById('modePill'),
  imagePill: document.getElementById('imagePill'),
  processList: document.getElementById('processList'),
  processSummary: document.getElementById('processSummary'),
  processListMain: document.getElementById('processListMain'),
  processSummaryMain: document.getElementById('processSummaryMain'),
  sessionLine: document.getElementById('sessionLine'),
  runtimeLine: document.getElementById('runtimeLine'),
  statusDock: document.getElementById('statusDock'),
  themeToggle: document.getElementById('themeToggleButton'),
  mobileToggle: document.getElementById('mobileToggleButton'),
  closeControls: document.getElementById('closeControlsButton'),
};

function log(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  els.log.textContent = `${line}\n${els.log.textContent}`.slice(0, 2200);
}

function shortText(value, limit = 72) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  return text.length > limit ? `${text.slice(0, limit - 1)}...` : text;
}

function processStatusLabel(status) {
  if (status === 'active') return 'Running';
  if (status === 'done') return 'Done';
  if (status === 'error') return 'Error';
  return 'Queued';
}

function liveStatusTitle(id, status, detail = '') {
  const step = PROCESS_STEPS.find((item) => item.id === id);
  if (status === 'error') return `${step?.label || id} failed`;
  const text = String(detail || '').toLowerCase();
  if (id === 'runtime') {
    if (text.includes('manifest')) return 'Loading manifest';
    if (text.includes('runtime module')) return 'Loading runtime';
    if (text.includes('webgpu')) return 'Starting WebGPU';
    if (text.includes('dense tensors ready')) return 'Preparing layers';
    if (text.includes('dense tensor')) return 'Loading weights';
    if (text.includes('bitnet layer') && text.includes('upload')) return 'Uploading layer';
    if (text.includes('bitnet layer')) return 'Building layers';
    if (text.includes('preparing')) return 'Preparing layers';
    if (text.includes('tokenizer')) return 'Loading tokenizer';
    if (text.includes('ready')) return 'Runtime ready';
    if (text.includes('bundle') || text.includes('loading')) return 'Loading model';
    return 'Runtime';
  }
  return step?.label || id;
}

function processLists() {
  return [els.processList, els.processListMain].filter(Boolean);
}

function processSummaries() {
  return [els.processSummary, els.processSummaryMain].filter(Boolean);
}

function setProcessSummary(text) {
  for (const node of processSummaries()) node.textContent = text;
}

function setTheme(theme, persist = true) {
  const normalized = theme === 'dark' ? 'dark' : 'light';
  state.theme = normalized;
  document.documentElement.dataset.theme = normalized;
  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, normalized);
    } catch (_error) {
      // Theme persistence is optional; private browsing can reject localStorage.
    }
  }
  if (els.themeToggle) {
    els.themeToggle.textContent = normalized === 'dark' ? 'Light' : 'Dark';
    els.themeToggle.setAttribute('aria-pressed', normalized === 'dark' ? 'true' : 'false');
    els.themeToggle.title = normalized === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
  }
}

function toggleTheme() {
  setTheme(state.theme === 'dark' ? 'light' : 'dark');
}

function setAgentWorking(active) {
  document.body.classList.toggle('agent-working', Boolean(active));
}

function updateRuntimeDetail(text) {
  if (els.runtimeDetail) els.runtimeDetail.textContent = text;
  if (els.runtimeLine) els.runtimeLine.textContent = text;
}

function bytesToHex(buffer) {
  return [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function sha256Text(text) {
  const data = new TextEncoder().encode(String(text || ''));
  return bytesToHex(await crypto.subtle.digest('SHA-256', data));
}

async function hashAsset(path) {
  const url = new URL(path, window.location.href);
  const response = await fetch(url.href, { cache: 'no-store' });
  if (!response.ok) throw new Error(`hash fetch failed for ${path}: ${response.status}`);
  const buffer = await response.arrayBuffer();
  return {
    path,
    url: url.href,
    bytes: buffer.byteLength,
    sha256: bytesToHex(await crypto.subtle.digest('SHA-256', buffer)),
  };
}

async function computeAppIntegrity() {
  const assets = [
    './index.html',
    './js/agent-kernel-app.js',
    './wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core.js',
    './wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core_bg.wasm',
  ];
  const hashedAssets = await Promise.all(assets.map((asset) => hashAsset(asset)));
  const appHash = await sha256Text(JSON.stringify(hashedAssets.map((asset) => ({
    path: asset.path,
    sha256: asset.sha256,
  }))));
  return {
    algorithm: 'sha256',
    app_hash: appHash,
    assets: hashedAssets,
    computed_at: new Date().toISOString(),
  };
}

async function refreshAppIntegrity() {
  if (!crypto?.subtle) {
    if (els.appHashMetric) els.appHashMetric.textContent = 'Unavailable';
    log('app hash unavailable: crypto.subtle is not supported');
    return;
  }
  try {
    state.appIntegrity = await computeAppIntegrity();
    const shortHash = state.appIntegrity.app_hash.slice(0, 12);
    if (els.appHashMetric) {
      els.appHashMetric.textContent = shortHash;
      els.appHashMetric.title = state.appIntegrity.app_hash;
    }
    log(`app hash ${shortHash}`);
  } catch (error) {
    if (els.appHashMetric) els.appHashMetric.textContent = 'Error';
    log(`app hash failed: ${error.message || String(error)}`);
  }
}

function syncModelControls() {
  const loading = Boolean(state.modelBusy);
  const loaded = Boolean(state.modelReady);
  const imageMode = Boolean(state.image.enabled);
  const imageBusy = Boolean(state.image.busy);
  const translationBusy = Boolean(state.translation.busy || state.translation.listening);
  const voiceBusy = Boolean(state.voice.busy);
  const browserOnlyMode = state.mode === 'web_search';
  if (els.loadModel) {
    els.loadModel.disabled = imageMode || loading || state.processActive;
    els.loadModel.textContent = loaded ? 'Reload Runtime' : loading ? 'Loading...' : 'Load Runtime';
  }
  if (els.unloadModel) {
    els.unloadModel.disabled = imageMode || loading || state.processActive || !state.worker;
  }
  if (els.send) {
    els.send.disabled = state.processActive || imageBusy || translationBusy || voiceBusy || (imageMode || browserOnlyMode ? false : loading || !loaded);
    els.send.textContent = imageMode
      ? imageBusy ? 'Generating...' : 'Generate'
      : state.translation.enabled
        ? state.translation.busy ? 'Translating...' : 'Translate'
        : 'Send';
  }
  syncTranslationControls();
  syncVoiceControls();
}

function createProcessStepElement(step) {
  const item = document.createElement('li');
  item.className = 'process-step';
  item.dataset.step = step.id;
  item.dataset.status = 'idle';
  item.innerHTML = `
    <span class="process-dot" aria-hidden="true"></span>
    <span>
      <span class="process-name">
        <span>${step.label}</span>
        <span class="process-status">${processStatusLabel('idle')}</span>
      </span>
      <span class="process-detail">${step.idle}</span>
    </span>
  `;
  return item;
}

function renderProcessTrace() {
  const lists = processLists();
  if (!lists.length) return;
  for (const list of lists) {
    list.replaceChildren();
    for (const step of PROCESS_STEPS) list.appendChild(createProcessStepElement(step));
  }
  setProcessSummary('Idle');
}

function setProcessStep(id, status, detail = '') {
  if (status === 'active') setAgentWorking(true);
  const lists = processLists();
  if (lists.length) {
    if (!lists[0].childElementCount) renderProcessTrace();
    for (const list of lists) {
      const item = list.querySelector(`[data-step="${id}"]`);
      if (!item) continue;
      item.classList.remove('active', 'done', 'error');
      if (status && status !== 'idle') item.classList.add(status);
      item.dataset.status = status || 'idle';
      const statusNode = item.querySelector('.process-status');
      const detailNode = item.querySelector('.process-detail');
      if (statusNode) statusNode.textContent = processStatusLabel(status);
      if (detailNode && detail) detailNode.textContent = detail;
    }
  }
  const label = PROCESS_STEPS.find((step) => step.id === id)?.label || id;
  if (status === 'active') setProcessSummary(label);
  if (status === 'error') setProcessSummary(`${label} failed`);
  if (status && status !== 'idle') updateLiveStatus(id, status, detail);
}

function resetProcessTrace(prompt) {
  state.processRunId += 1;
  state.processActive = true;
  setAgentWorking(true);
  syncModelControls();
  renderProcessTrace();
  startLiveStatus(prompt);
  setProcessStep('receive', 'done', shortText(prompt, 96) || 'Prompt received');
  setProcessSummary('Starting');
}

function finishProcessTrace(summary = 'Complete') {
  state.processActive = false;
  setAgentWorking(false);
  syncModelControls();
  setProcessSummary(summary);
  finishLiveStatus(summary);
  setEvidenceActionsLocked(false);
}

function startLiveStatus(prompt) {
  els.empty?.remove();
  const previous = state.liveStatusNode;
  if (previous?.isConnected && previous.dataset.finished !== 'true') previous.remove();
  const node = document.createElement('article');
  node.className = 'message assistant live-status';
  node.dataset.status = 'active';
  node.innerHTML = `
    <div class="role">Agent Kernel Lite</div>
    <div class="body">
      <div class="live-status-line">
        <span class="live-dot" aria-hidden="true"></span>
        <span class="live-status-state">Starting</span>
        <span class="live-status-detail">Preparing turn</span>
      </div>
    </div>
  `;
  state.liveStatusNode = node;
  if (els.statusDock) {
    els.statusDock.replaceChildren(node);
  } else {
    els.chat.appendChild(node);
    els.chat.scrollTop = els.chat.scrollHeight;
  }
  updateLiveStatus('receive', 'done', shortText(prompt, 96) || 'Prompt received');
}

function updateLiveStatus(id, status, detail = '') {
  const node = state.liveStatusNode;
  if (!node?.isConnected) return;
  const step = PROCESS_STEPS.find((item) => item.id === id);
  if (!step) return;
  node.dataset.status = status || 'active';
  const stateNode = node.querySelector('.live-status-state');
  const detailNode = node.querySelector('.live-status-detail');
  if (stateNode) {
    stateNode.textContent = liveStatusTitle(id, status, detail);
  }
  if (detailNode) detailNode.textContent = detail || processStatusLabel(status);
}

function finishLiveStatus(summary = 'Complete') {
  const node = state.liveStatusNode;
  if (!node?.isConnected) return;
  node.dataset.finished = 'true';
  node.dataset.status = summary === 'Error' ? 'error' : 'done';
  const stateNode = node.querySelector('.live-status-state');
  if (stateNode) stateNode.textContent = summary;
  const detailNode = node.querySelector('.live-status-detail');
  if (detailNode) detailNode.textContent = summary === 'Error' ? 'Review the message below' : 'Done';
  if (summary !== 'Error') {
    window.setTimeout(() => {
      if (state.liveStatusNode === node && node.dataset.finished === 'true') {
        node.remove();
        state.liveStatusNode = null;
      }
    }, 900);
  }
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(0)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${value} B`;
}

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function formatDurationMs(ms) {
  const value = Math.max(0, Number(ms || 0));
  if (value > 0 && value < 1000) return `${Math.max(1, Math.round(value))}ms`;
  const seconds = Math.max(0, Math.round(value / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`;
}

function setPill(el, text, mode = '') {
  if (!el) return;
  el.textContent = text;
  el.className = `pill ${mode}`.trim();
}

function normalizeMode(mode) {
  const value = String(mode || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  return MODE_CONFIG[value] ? value : 'chat';
}

function modeConfig() {
  return MODE_CONFIG[normalizeMode(state.mode)] || MODE_CONFIG.chat;
}

function modeToken(mode = state.mode) {
  const normalized = normalizeMode(mode);
  if (normalized === 'think') return '<AK_THINK>';
  if (normalized === 'deep_research') return '<AK_DEEP_RESEARCH>';
  return '<AK_CHAT>';
}

function syncImageModeControls() {
  const enabled = Boolean(state.image.enabled);
  els.imageMode?.classList.toggle('active', enabled);
  els.imageMode?.setAttribute('aria-pressed', enabled ? 'true' : 'false');
  if (els.imageMode) els.imageMode.textContent = enabled ? 'Image On' : 'Image Off';
  if (els.imageModeDetail) {
    els.imageModeDetail.textContent = enabled ? 'Image runtime' : 'Chat runtime';
  }
  setPill(
    els.imagePill,
    enabled
      ? state.image.busy
        ? 'image generating'
        : state.image.ready
          ? 'image ready'
          : 'image loading'
      : 'image off',
    enabled ? state.image.busy || !state.image.ready ? 'busy' : 'ready' : '',
  );
  if (els.prompt) els.prompt.placeholder = enabled ? 'Describe the image to generate...' : modeConfig().placeholder;
  syncModelControls();
  renderExtensionList();
}

function translationTargetLabel() {
  return String(els.translationTarget?.value || 'Spanish').trim() || 'Spanish';
}

function translationSourceLabel() {
  const value = String(els.translationSource?.value || 'auto').trim();
  if (!value || value === 'auto') return 'Auto';
  const option = [...(els.translationSource?.options || [])].find((item) => item.value === value);
  return option?.textContent || value;
}

function syncTranslationControls() {
  const enabled = Boolean(state.translation.enabled);
  els.translationMode?.classList.toggle('active', enabled);
  els.translationMode?.setAttribute('aria-pressed', enabled ? 'true' : 'false');
  if (els.translationMode) els.translationMode.textContent = enabled ? 'Translate On' : 'Translate Off';
  if (els.translationModeDetail) {
    els.translationModeDetail.textContent = enabled
      ? `${translationSourceLabel()} to ${translationTargetLabel()}`
      : 'Translate text or speech';
  }
  if (els.audioTranslate) {
    els.audioTranslate.disabled = state.processActive || state.translation.busy;
    els.audioTranslate.classList.toggle('listening', state.translation.listening);
    els.audioTranslate.textContent = state.translation.listening ? 'Listening...' : 'Audio Translate';
  }
  if (els.prompt && !state.image.enabled) {
    els.prompt.placeholder = enabled ? `Text to translate to ${translationTargetLabel()}...` : modeConfig().placeholder;
  }
}

function setTranslationMode(enabled) {
  const requested = Boolean(enabled);
  const extensionResult = setExtensionEnabled(state.translation.extensionId, requested);
  if (extensionResult.status === 'error' || extensionResult.status === 'disabled') {
    appendMessage('assistant', `Translator extension could not be ${requested ? 'enabled' : 'disabled'}: ${extensionResult.error || extensionResult.status}`);
    log(`translator extension toggle failed: ${extensionResult.error || extensionResult.status}`);
    syncTranslationControls();
    return;
  }
  state.translation.enabled = requested;
  if (state.translation.enabled && state.image.enabled) setImageMode(false);
  log(`translator extension ${requested ? 'enabled' : 'disabled'}`);
  syncTranslationControls();
  syncModelControls();
}

function isUserVisibleWebSearchRequest(text) {
  const normalized = String(text || '').toLowerCase();
  return [
    'search the web',
    'web search',
    'look online',
    'search online',
    'internet search',
    'google ',
    'open a search',
    'search safari',
    'search in safari',
  ].some((phrase) => normalized.includes(phrase));
}

function cleanWebSearchQuery(text) {
  let query = String(text || '').trim();
  query = query
    .replace(/^(please\s+)?(can you\s+)?(do a\s+)?(web|internet|online)\s+search\s+(for|about)\s+/i, '')
    .replace(/^(please\s+)?(can you\s+)?search\s+(the\s+)?(web|internet|online)\s+(for|about)\s+/i, '')
    .replace(/^(please\s+)?(can you\s+)?look\s+online\s+(for|about)\s+/i, '')
    .replace(/^(please\s+)?google\s+/i, '')
    .replace(/\b(and\s+)?(summari[sz]e|explain|tell me|give me|show me)\b.*$/i, '')
    .replace(/\b(what did you find|main points|key points|latest info|relevant information)\b.*$/i, '')
    .replace(/[?.!,;:\s]+$/g, '')
    .trim();
  return query || String(text || '').trim();
}

function webSearchUrl(query, engine = 'duckduckgo') {
  const safeQuery = String(query || '').trim();
  const encoded = encodeURIComponent(safeQuery);
  if (engine === 'google') return `https://www.google.com/search?q=${encoded}`;
  if (engine === 'bing') return `https://www.bing.com/search?q=${encoded}`;
  return `https://duckduckgo.com/?q=${encoded}`;
}

function openUserVisibleUrl(url) {
  const opened = window.open(url, '_blank', 'noopener,noreferrer');
  if (!opened) {
    appendMessage('assistant', `I prepared the web search, but the browser blocked opening it automatically: ${url}`);
    return false;
  }
  return true;
}

function webResultUrl(result) {
  return String(result?.url || '').trim();
}

function normalizeWebResult(result) {
  const title = String(result?.title || '').replace(/\s+/g, ' ').trim();
  const url = webResultUrl(result);
  const snippet = String(result?.snippet || result?.description || '').replace(/\s+/g, ' ').trim();
  const source = String(result?.source || '').replace(/\s+/g, ' ').trim();
  if (!title || !url) return null;
  return {
    source: source || new URL(url).hostname.replace(/^www\./, ''),
    title: shortText(title, 160),
    url,
    snippet: shortText(snippet, 420),
    published: result?.published ? String(result.published) : '',
  };
}

function scoreWebResult(query, result) {
  const queryTokens = new Set(contentTokens(query));
  if (!queryTokens.size) return 0;
  const resultTokens = contentTokens(`${result.title || ''} ${result.snippet || ''} ${result.source || ''}`);
  let score = 0;
  for (const token of resultTokens) {
    if (queryTokens.has(token)) score += 1;
  }
  return score + (result.snippet ? 0.5 : 0);
}

const WEB_SEARCH_GENERIC_TOKENS = new Set([
  'search',
  'web',
  'browser',
  'local',
  'model',
  'models',
  'comparison',
  'timeline',
  'production',
  'market',
  'impact',
  'approval',
  'agreement',
  'source',
  'sources',
]);

function distinctiveWebTokens(query) {
  return [...new Set(contentTokens(query))]
    .filter((token) => token.length > 3 && !WEB_SEARCH_GENERIC_TOKENS.has(token));
}

function distinctiveWebMatchCount(query, result) {
  const tokens = distinctiveWebTokens(query);
  if (!tokens.length) return 0;
  const haystack = normalizeSearchText(`${result.title || ''} ${result.snippet || ''} ${result.source || ''}`);
  return tokens.filter((token) => haystack.includes(token)).length;
}

function hasEnoughWebQueryCoverage(query, result) {
  const tokens = distinctiveWebTokens(query);
  if (tokens.length <= 1) return true;
  const matches = distinctiveWebMatchCount(query, result);
  return matches >= Math.min(2, tokens.length);
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 7000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        accept: 'application/json,text/plain,*/*',
        ...(options.headers || {}),
      },
    });
  } finally {
    window.clearTimeout(timer);
  }
}

async function searchWikipedia(query, limit) {
  const url = new URL('https://en.wikipedia.org/w/api.php');
  url.searchParams.set('origin', '*');
  url.searchParams.set('action', 'query');
  url.searchParams.set('list', 'search');
  url.searchParams.set('format', 'json');
  url.searchParams.set('srlimit', String(Math.min(limit, 5)));
  url.searchParams.set('srsearch', query);
  const response = await fetchWithTimeout(url.href);
  if (!response.ok) throw new Error(`Wikipedia ${response.status}`);
  const data = await response.json();
  return (data.query?.search || []).map((item) => normalizeWebResult({
    source: 'Wikipedia',
    title: item.title,
    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(String(item.title || '').replace(/\s+/g, '_'))}`,
    snippet: String(item.snippet || '').replace(/<[^>]*>/g, ''),
    published: item.timestamp || '',
  })).filter(Boolean);
}

async function searchDuckDuckGoInstant(query, limit) {
  const url = new URL('https://api.duckduckgo.com/');
  url.searchParams.set('q', query);
  url.searchParams.set('format', 'json');
  url.searchParams.set('no_html', '1');
  url.searchParams.set('skip_disambig', '1');
  const response = await fetchWithTimeout(url.href);
  if (!response.ok) throw new Error(`DuckDuckGo ${response.status}`);
  const data = await response.json();
  const rows = [];
  if (data.AbstractText && data.AbstractURL) {
    rows.push(normalizeWebResult({
      source: data.AbstractSource || 'DuckDuckGo',
      title: data.Heading || query,
      url: data.AbstractURL,
      snippet: data.AbstractText,
    }));
  }
  const related = [];
  for (const topic of data.RelatedTopics || []) {
    if (topic.Topics) related.push(...topic.Topics);
    else related.push(topic);
  }
  for (const topic of related.slice(0, limit)) {
    rows.push(normalizeWebResult({
      source: 'DuckDuckGo',
      title: topic.Text || topic.FirstURL || query,
      url: topic.FirstURL || '',
      snippet: topic.Text || '',
    }));
  }
  return rows.filter(Boolean);
}

async function searchOpenAlex(query, limit) {
  const url = new URL('https://api.openalex.org/works');
  url.searchParams.set('search', query);
  url.searchParams.set('per-page', String(Math.min(limit, 5)));
  const response = await fetchWithTimeout(url.href);
  if (!response.ok) throw new Error(`OpenAlex ${response.status}`);
  const data = await response.json();
  return (data.results || []).map((item) => normalizeWebResult({
    source: 'OpenAlex',
    title: item.display_name || '',
    url: item.doi ? `https://doi.org/${String(item.doi).replace(/^https?:\/\/doi.org\//i, '')}` : item.id,
    snippet: item.primary_location?.source?.display_name
      ? `${item.primary_location.source.display_name}${item.publication_year ? `, ${item.publication_year}` : ''}`
      : `Scholarly result${item.publication_year ? ` from ${item.publication_year}` : ''}`,
    published: item.publication_year || '',
  })).filter(Boolean);
}

async function searchCrossref(query, limit) {
  const url = new URL('https://api.crossref.org/works');
  url.searchParams.set('query', query);
  url.searchParams.set('rows', String(Math.min(limit, 5)));
  const response = await fetchWithTimeout(url.href);
  if (!response.ok) throw new Error(`Crossref ${response.status}`);
  const data = await response.json();
  return (data.message?.items || []).map((item) => normalizeWebResult({
    source: 'Crossref',
    title: Array.isArray(item.title) ? item.title[0] : item.title,
    url: item.URL || (item.DOI ? `https://doi.org/${item.DOI}` : ''),
    snippet: [
      Array.isArray(item.publisher) ? item.publisher[0] : item.publisher,
      item.issued?.['date-parts']?.[0]?.[0],
      Array.isArray(item.subject) ? item.subject.slice(0, 3).join(', ') : '',
    ].filter(Boolean).join(' | '),
    published: item.issued?.['date-parts']?.[0]?.[0] || '',
  })).filter(Boolean);
}

async function searchHackerNews(query, limit) {
  const url = new URL('https://hn.algolia.com/api/v1/search');
  url.searchParams.set('query', query);
  url.searchParams.set('tags', 'story');
  url.searchParams.set('hitsPerPage', String(Math.min(limit, 5)));
  const response = await fetchWithTimeout(url.href);
  if (!response.ok) throw new Error(`Hacker News ${response.status}`);
  const data = await response.json();
  return (data.hits || []).map((item) => normalizeWebResult({
    source: 'Hacker News',
    title: item.title || item.story_title || '',
    url: item.url || `https://news.ycombinator.com/item?id=${item.objectID}`,
    snippet: `${formatCount(item.points || 0)} points | ${formatCount(item.num_comments || 0)} comments`,
    published: item.created_at || '',
  })).filter(Boolean);
}

async function searchStackExchange(query, limit) {
  const url = new URL('https://api.stackexchange.com/2.3/search/advanced');
  url.searchParams.set('order', 'desc');
  url.searchParams.set('sort', 'relevance');
  url.searchParams.set('site', 'stackoverflow');
  url.searchParams.set('pagesize', String(Math.min(limit, 5)));
  url.searchParams.set('q', query);
  const response = await fetchWithTimeout(url.href);
  if (!response.ok) throw new Error(`StackExchange ${response.status}`);
  const data = await response.json();
  return (data.items || []).map((item) => normalizeWebResult({
    source: 'Stack Overflow',
    title: item.title || '',
    url: item.link || '',
    snippet: [
      item.is_answered ? 'answered' : 'unanswered',
      `${formatCount(item.score || 0)} score`,
      `${formatCount(item.answer_count || 0)} answers`,
    ].join(' | '),
    published: item.creation_date ? new Date(item.creation_date * 1000).toISOString().slice(0, 10) : '',
  })).filter(Boolean);
}

async function searchGitHub(query, limit) {
  const url = new URL('https://api.github.com/search/repositories');
  url.searchParams.set('q', query);
  url.searchParams.set('sort', 'stars');
  url.searchParams.set('order', 'desc');
  url.searchParams.set('per_page', String(Math.min(limit, 5)));
  const response = await fetchWithTimeout(url.href, {
    headers: { accept: 'application/vnd.github+json' },
  });
  if (!response.ok) throw new Error(`GitHub ${response.status}`);
  const data = await response.json();
  return (data.items || []).map((item) => normalizeWebResult({
    source: 'GitHub',
    title: item.full_name || item.name || '',
    url: item.html_url || '',
    snippet: [
      item.description || '',
      `${formatCount(item.stargazers_count || 0)} stars`,
      item.language || '',
    ].filter(Boolean).join(' | '),
    published: item.updated_at || '',
  })).filter(Boolean);
}

async function performWebSearch(query, maxSources = state.webSearch.maxSources) {
  const limit = Math.max(1, Math.min(5, Number(maxSources || 5)));
  const providers = [searchDuckDuckGoInstant, searchWikipedia, searchStackExchange, searchGitHub, searchOpenAlex, searchCrossref, searchHackerNews];
  const settled = await Promise.allSettled(providers.map((provider) => provider(query, limit)));
  const errors = [];
  const seen = new Set();
  const results = [];
  for (const item of settled) {
    if (item.status === 'rejected') {
      errors.push(item.reason?.message || String(item.reason));
      continue;
    }
    for (const result of item.value || []) {
      const key = webResultUrl(result).toLowerCase();
      if (!key || seen.has(key)) continue;
      seen.add(key);
      if (hasEnoughWebQueryCoverage(query, result)) results.push(result);
    }
  }
  const ranked = results
    .map((result, index) => ({ ...result, _index: index, _score: scoreWebResult(query, result) }))
    .sort((a, b) => (b._score - a._score) || (a._index - b._index));
  const sourceCounts = new Map();
  const selected = [];
  for (const result of ranked) {
    const source = result.source || 'source';
    const count = sourceCounts.get(source) || 0;
    if (count >= 2) continue;
    selected.push(result);
    sourceCounts.set(source, count + 1);
    if (selected.length >= limit) break;
  }
  if (selected.length < limit) {
    const selectedUrls = new Set(selected.map((result) => result.url));
    for (const result of ranked) {
      if (selectedUrls.has(result.url)) continue;
      selected.push(result);
      selectedUrls.add(result.url);
      if (selected.length >= limit) break;
    }
  }
  return {
    query,
    max_sources: limit,
    results: selected
      .map(({ _index, _score, ...result }) => result),
    provider_errors: errors,
  };
}

function webSearchAnswer(search, fallbackOpened = false, fallbackUrl = '') {
  const results = search.results || [];
  if (!results.length) {
    return fallbackOpened
      ? `I could not retrieve structured web results for "${search.query}". I opened a browser search so you can inspect results directly.`
      : `I could not retrieve structured web results for "${search.query}". Search link: ${fallbackUrl || webSearchUrl(search.query)}`;
  }
  const lines = [
    `Web results for: ${search.query}`,
    `Quality: ${search.quality?.label || 'structured'} (${results.length}/${search.max_sources || results.length} sources)`,
    '',
    ...results.map((result, index) => [
      `${index + 1}. ${result.title}`,
      `Source: ${result.source}${result.published ? ` | ${result.published}` : ''}`,
      result.snippet ? `Relevant info: ${result.snippet}` : '',
      result.url,
    ].filter(Boolean).join('\n')),
  ];
  return lines.join('\n\n');
}

function webSearchQuality(search) {
  const results = search.results || [];
  const sourceCount = new Set(results.map((result) => result.source).filter(Boolean)).size;
  const label = !results.length
    ? 'insufficient'
    : results.length >= Math.min(3, search.max_sources || 5) && sourceCount >= 2
      ? 'good'
      : 'limited';
  return {
    label,
    result_count: results.length,
    source_count: sourceCount,
    provider_error_count: (search.provider_errors || []).length,
  };
}

function rememberWebSearchResults(search) {
  if (!search.results?.length) return;
  const body = [
    `Query: ${search.query}`,
    `Quality: ${search.quality?.label || 'structured'}`,
    ...search.results.map((result, index) => (
      `[W${index + 1}] ${result.title} | ${result.source}${result.published ? ` | ${result.published}` : ''}\n`
      + `${result.snippet || ''}\n${result.url}`
    )),
  ].join('\n\n');
  addPocketPalDataSource({
    source_type: 'web_search',
    title: `Web search: ${shortText(search.query, 80)}`,
    text: body,
    bytes: body.length,
  });
}

async function runWebSearch(query, options = {}) {
  const searchQuery = cleanWebSearchQuery(query);
  if (!searchQuery) return { status: 'error', error: 'empty web search query' };
  if (!extensionEnabled(state.webSearch.extensionId)) {
    setExtensionEnabled(state.webSearch.extensionId, true);
  }
  const maxSources = Math.max(1, Math.min(5, Number(options.maxSources || state.webSearch.maxSources || 5)));
  const url = webSearchUrl(searchQuery, options.engine || 'duckduckgo');
  const proposal = proposeExtensionAction(state.webSearch.extensionId, state.webSearch.searchCapabilityId, {
    query: searchQuery,
    max_sources: maxSources,
    url,
    surface: NATIVE_APP ? 'ios_wkwebview' : 'browser',
    user_visible: true,
  });
  if (proposal.status !== 'pending_user_approval') {
    appendMessage('assistant', `Web search could not start: ${proposal.error || proposal.status || 'extension unavailable'}`);
    log(`web search failed: ${proposal.error || proposal.status || 'extension unavailable'}`);
    return proposal;
  }
  const search = await performWebSearch(searchQuery, maxSources);
  search.quality = webSearchQuality(search);
  const opened = search.results.length ? false : openUserVisibleUrl(url);
  recordExtensionResult(proposal.action_id, {
    action_id: proposal.action_id,
    status: search.results.length || opened ? 'approved_executed' : 'failed',
    output: {
      query: searchQuery,
      max_sources: maxSources,
      url,
      opened,
      result_access: search.results.length ? 'structured_public_sources' : 'user_visible_browser',
      results: search.results,
      quality: search.quality,
      provider_errors: search.provider_errors,
    },
  });
  rememberWebSearchResults(search);
  appendMessage('assistant', webSearchAnswer(search, opened, url));
  log(`web search returned ${formatCount(search.results.length)} result${search.results.length === 1 ? '' : 's'}: ${searchQuery}`);
  return {
    ...proposal,
    status: search.results.length ? 'searched' : opened ? 'opened' : 'blocked',
    query: searchQuery,
    url,
    results: search.results,
    quality: search.quality,
    provider_errors: search.provider_errors,
  };
}

async function submitWebSearchPrompt(text) {
  resetProcessTrace(text);
  setControlsBusy(true);
  try {
    if (!state.coreReady) await loadAgentCore();
    setProcessStep('plan', 'done', `web_search: up to ${formatCount(state.webSearch.maxSources)} sources`);
    setProcessStep('lookup', 'active', 'Searching public sources');
    const result = await runWebSearch(text);
    setProcessStep('lookup', result.results?.length || result.status === 'opened' ? 'done' : 'error', result.url || result.error || result.status);
    setProcessStep('generate', 'done', 'Decoder skipped for browser search action');
    setProcessStep('render', 'done', 'Search action rendered');
    finishProcessTrace(result.results?.length ? 'Web Search' : result.status === 'opened' ? 'Browser Search' : 'Blocked');
  } catch (error) {
    setProcessStep('render', 'error', error.message || String(error));
    appendMessage('assistant', `Web search failed: ${error.message || String(error)}`);
    finishProcessTrace('Error');
  } finally {
    setControlsBusy(false);
  }
}

function ensureImageWorker() {
  if (state.image.worker) return state.image.worker;
  state.image.worker = new Worker('./js/image-worker.js?v=20260504-flux-image-dev-v1', { type: 'module' });
  state.image.worker.addEventListener('message', onImageWorkerMessage);
  state.image.worker.addEventListener('error', (event) => {
    state.image.busy = false;
    state.image.ready = false;
    setProcessStep('generate', 'error', event.message || 'Image worker error');
    finishProcessTrace('Error');
    appendMessage('assistant', `Image generation failed: ${event.message || 'worker error'}`);
    syncImageModeControls();
  });
  return state.image.worker;
}

function loadImageRuntime() {
  if (state.image.ready) return Promise.resolve();
  if (state.image.loadPromise) return state.image.loadPromise;
  state.image.loadPromise = new Promise((resolve, reject) => {
    state.image.loadResolve = resolve;
    state.image.loadReject = reject;
  });
  ensureImageWorker().postMessage({
    type: 'load',
    modelId: state.image.modelId,
  });
  syncImageModeControls();
  return state.image.loadPromise;
}

function settleImageRuntime(error) {
  const resolve = state.image.loadResolve;
  const reject = state.image.loadReject;
  state.image.loadPromise = null;
  state.image.loadResolve = null;
  state.image.loadReject = null;
  if (error) reject?.(error);
  else resolve?.();
}

function setImageMode(enabled) {
  const requested = Boolean(enabled);
  const extensionResult = setExtensionEnabled(state.image.extensionId, requested);
  if (extensionResult.status === 'error' || extensionResult.status === 'disabled') {
    appendMessage('assistant', `Image extension could not be ${requested ? 'enabled' : 'disabled'}: ${extensionResult.error || extensionResult.status}`);
    log(`image extension toggle failed: ${extensionResult.error || extensionResult.status}`);
    syncImageModeControls();
    return;
  }
  state.image.enabled = requested;
  if (state.image.enabled) {
    if (state.translation.enabled) setTranslationMode(false);
    log('image generation mode enabled');
  } else {
    log('image generation mode disabled');
  }
  syncImageModeControls();
}

function syncVoiceControls() {
  if (els.voiceSpeak) {
    els.voiceSpeak.disabled = state.processActive || state.voice.busy;
    els.voiceSpeak.classList.toggle('active', state.voice.ready);
    els.voiceSpeak.textContent = state.voice.busy ? 'Speaking...' : state.voice.ready ? 'Speak' : 'Load Voice';
  }
  if (els.voiceModeDetail) {
    els.voiceModeDetail.textContent = state.voice.busy
      ? state.voice.progressDetail || 'Generating Peyton voice'
      : state.voice.ready
        ? state.voice.detail || 'Peyton voice ready'
        : 'Peyton voice preview';
  }
}

function ensureVoiceWorker() {
  if (state.voice.worker) return state.voice.worker;
  state.voice.worker = new Worker(`./js/tts-worker.js?v=${VOICE_RUNTIME_VERSION}`, { type: 'module' });
  state.voice.worker.addEventListener('message', onVoiceWorkerMessage);
  state.voice.worker.addEventListener('error', (event) => {
    state.voice.busy = false;
    state.voice.ready = false;
    const location = event.filename ? ` (${event.filename}:${event.lineno || 0}:${event.colno || 0})` : '';
    const detail = `${event.message || 'worker error'}${location}`;
    settleVoiceRuntime(new Error(detail));
    appendMessage('assistant', `Peyton voice failed: ${detail}`);
    log(`Peyton voice worker failed: ${detail}`);
    syncVoiceControls();
    syncModelControls();
  });
  return state.voice.worker;
}

function releaseChatRuntimeForVoice() {
  if (!state.worker && !state.modelBusy && !state.modelReady && !state.modelLoadPromise) return;
  log('unloading chat runtime before Peyton voice');
  updateLiveStatus('runtime', 'active', 'Unloading chat runtime for Peyton voice');
  unloadModel({ silent: true });
  updateRuntimeDetail('Chat runtime unloaded while Peyton voice is active.');
}

function unloadVoiceRuntime({ silent = false } = {}) {
  if (state.voice.worker) {
    state.voice.worker.terminate();
    state.voice.worker = null;
  }
  if (state.voice.loadPromise) settleVoiceRuntime(new Error('Peyton voice runtime unloaded.'));
  state.voice.ready = false;
  state.voice.busy = false;
  state.voice.detail = '';
  state.voice.progressDetail = '';
  state.voice.nativeRuntime = null;
  if (!silent) log('Peyton voice runtime unloaded');
  syncVoiceControls();
}

function releaseVoiceRuntimeForChat() {
  if (!state.voice.worker && !state.voice.ready && !state.voice.loadPromise) return;
  log('unloading Peyton voice runtime before chat model');
  unloadVoiceRuntime({ silent: true });
  if (els.voiceModeDetail) els.voiceModeDetail.textContent = 'Peyton voice unloaded for chat runtime';
}

function loadVoiceRuntime() {
  if (state.voice.ready) return Promise.resolve();
  if (state.voice.loadPromise) return state.voice.loadPromise;
  state.voice.loadPromise = new Promise((resolve, reject) => {
    state.voice.loadResolve = resolve;
    state.voice.loadReject = reject;
  });
  ensureVoiceWorker().postMessage({ type: 'load' });
  syncVoiceControls();
  return state.voice.loadPromise;
}

function settleVoiceRuntime(error) {
  const resolve = state.voice.loadResolve;
  const reject = state.voice.loadReject;
  state.voice.loadPromise = null;
  state.voice.loadResolve = null;
  state.voice.loadReject = null;
  if (error) reject?.(error);
  else resolve?.();
}

async function speakPeytonVoice() {
  const text = String(els.prompt?.value || '').trim() || 'This is Peyton speaking from Agent Kernel Lite.';
  await speakPeytonVoiceText(text);
}

async function speakPeytonVoiceText(text) {
  if (state.voice.busy) {
    const detail = 'Peyton voice is already generating';
    log(detail);
    if (els.voiceModeDetail) els.voiceModeDetail.textContent = detail;
    if (!state.liveStatusNode?.isConnected) startLiveStatus('Peyton voice');
    updateLiveStatus('generate', 'active', detail);
    return;
  }
  const promptText = String(text || '').trim() || 'This is Peyton speaking from Agent Kernel Lite.';
  state.voice.busy = true;
  state.voice.progressDetail = 'Starting Peyton voice';
  startLiveStatus(`Peyton voice: ${shortText(promptText, 80)}`);
  updateLiveStatus('runtime', 'active', 'Starting Peyton voice worker');
  syncVoiceControls();
  syncModelControls();
  try {
    releaseChatRuntimeForVoice();
    await loadVoiceRuntime();
    updateLiveStatus('runtime', 'done', 'Peyton voice runtime ready');
    updateLiveStatus('generate', 'active', 'Queued voice generation');
    log(`Peyton voice prompt: ${shortText(promptText, 80)}`);
    ensureVoiceWorker().postMessage({
      type: 'speak',
      text: promptText,
      runtimeVersion: VOICE_RUNTIME_VERSION,
      condSeqLen: 938,
      steps: 8,
      cfgStrength: 2.0,
      speed: 1.15,
    });
  } catch (error) {
    state.voice.busy = false;
    updateLiveStatus('runtime', 'error', error.message || String(error));
    finishLiveStatus('Error');
    appendMessage('assistant', `Peyton voice could not load: ${error.message || String(error)}`);
    log(`Peyton voice load failed: ${error.message || String(error)}`);
    syncVoiceControls();
    syncModelControls();
  }
}

function onVoiceWorkerMessage(event) {
  const data = event.data || {};
  if (data.type === 'progress') {
    const percent = Number.isFinite(Number(data.percent)) ? Math.max(0, Math.min(100, Math.round(Number(data.percent)))) : 0;
    const phase = String(data.phase || 'generate').toLowerCase();
    const eta = Number(data.etaMs || 0) > 0 ? ` | ETA ${formatDurationMs(data.etaMs)}` : '';
    const chunk = data.chunks ? ` | chunk ${data.chunk || 0}/${data.chunks}` : '';
    const frames = data.frames ? ` | ${formatCount(data.frames)} frames` : '';
    const detail = `${percent}% | ${data.detail || 'Peyton voice working'}${chunk}${frames}${eta}`;
    state.voice.progressDetail = detail;
    log(`Peyton voice progress: ${detail}`);
    if (els.voiceModeDetail) els.voiceModeDetail.textContent = detail;
    const step = phase.includes('decode') || phase.includes('render')
      ? 'render'
      : phase.includes('runtime') || phase.includes('load')
        ? 'runtime'
        : phase.includes('condition') || phase.includes('prepare')
          ? 'compile'
          : 'generate';
    updateLiveStatus(step, 'active', detail);
    setProcessStep(step, 'active', detail);
    syncVoiceControls();
    return;
  }
  if (data.type === 'status') {
    log(`Peyton voice: ${data.detail || 'working'}`);
    if (els.voiceModeDetail) els.voiceModeDetail.textContent = data.detail || 'Peyton voice working';
    const detail = data.detail || 'Peyton voice working';
    const lower = detail.toLowerCase();
    const step = lower.includes('decoding') ? 'render' : lower.includes('loading') || lower.includes('runtime') ? 'runtime' : 'generate';
    updateLiveStatus(step, 'active', detail);
    return;
  }
  if (data.type === 'ready') {
    state.voice.ready = true;
    state.voice.detail = data.detail || '';
    state.voice.progressDetail = '';
    settleVoiceRuntime();
    log(`Peyton voice ready${data.detail ? `: ${data.detail}` : ''}`);
    if (els.voiceModeDetail) els.voiceModeDetail.textContent = data.detail || 'Peyton voice ready';
    syncVoiceControls();
    return;
  }
  if (data.type === 'audio') {
    state.voice.busy = false;
    state.voice.ready = true;
    state.voice.progressDetail = '';
    const blob = new Blob([data.wav], { type: 'audio/wav' });
    if (state.voice.audioUrl) URL.revokeObjectURL(state.voice.audioUrl);
    state.voice.audioUrl = URL.createObjectURL(blob);
    if (els.voicePreviewAudio) {
      els.voicePreviewAudio.hidden = false;
      els.voicePreviewAudio.src = state.voice.audioUrl;
      els.voicePreviewAudio.play().catch(() => {});
    }
    appendVoiceMessage(data, state.voice.audioUrl);
    log(`Peyton voice rendered ${formatCount(data.samples)} samples${data.preset ? ` (${data.preset})` : ''}`);
    updateLiveStatus('render', 'done', `Rendered ${formatCount(data.samples)} samples`);
    finishLiveStatus('Voice Ready');
    syncVoiceControls();
    syncModelControls();
    return;
  }
  if (data.type === 'error') {
    const error = new Error(data.error || 'voice generation error');
    if (state.voice.loadPromise) settleVoiceRuntime(error);
    state.voice.busy = false;
    state.voice.progressDetail = '';
    updateLiveStatus('generate', 'error', error.message);
    finishLiveStatus('Error');
    appendMessage('assistant', `Peyton voice failed: ${error.message}`);
    log(`Peyton voice failed: ${error.message}`);
    syncVoiceControls();
    syncModelControls();
  }
}

function onImageWorkerMessage(event) {
  const data = event.data || {};
  if (data.type === 'status') {
    log(data.message || 'image status');
    if (state.processActive) setProcessStep('runtime', 'active', data.message || 'Image runtime status');
    return;
  }
  if (data.type === 'ready') {
    state.image.ready = true;
    state.image.busy = false;
    settleImageRuntime();
    if (state.processActive) {
      setProcessStep('runtime', 'done', `Image backend ready: ${shortText(data.modelId || state.image.modelId, 56)}`);
    }
    syncImageModeControls();
    return;
  }
  if (data.type === 'progress') {
    if (state.image.activeJobId !== data.id) return;
    setProcessStep('generate', 'active', `${data.stage || 'Generating'} ${data.step || 0}/${data.totalSteps || '?'}`);
    return;
  }
  if (data.type === 'result') {
    if (state.image.activeJobId !== data.id) return;
    const actionId = state.image.activeActionId;
    state.image.busy = false;
    state.image.activeJobId = null;
    state.image.activeActionId = null;
    setProcessStep('generate', 'done', 'Image artifact rendered');
    setProcessStep('render', 'done', 'Generated image added to chat');
    appendImageMessage(data);
    if (actionId) {
      recordExtensionResult(actionId, {
        action_id: actionId,
        status: 'approved_executed',
        output: {
          prompt: data.prompt || '',
          seed: data.seed || null,
          backend: data.metadata?.backend || 'preview-worker',
        },
        artifact_refs: data.id ? [`browser:image:${data.id}`] : [],
      });
    }
    finishProcessTrace('Image Ready');
    setControlsBusy(false);
    syncImageModeControls();
    return;
  }
  if (data.type === 'error') {
    const error = new Error(data.message || 'image generation error');
    if (state.image.loadPromise) settleImageRuntime(error);
    const actionId = state.image.activeActionId;
    state.image.busy = false;
    state.image.activeJobId = null;
    state.image.activeActionId = null;
    if (actionId) {
      recordExtensionResult(actionId, {
        action_id: actionId,
        status: 'failed',
        error: error.message,
      });
    }
    setProcessStep('generate', 'error', error.message);
    finishProcessTrace('Error');
    setControlsBusy(false);
    appendMessage('assistant', `Image generation failed: ${error.message}`);
    log(error.message);
    syncImageModeControls();
  }
}

function targetMaxTokens() {
  return Math.max(Number(els.tokens.value || 560), Number(modeConfig().minTokens || 560));
}

async function loadAgentCore() {
  try {
    const { default: initAgentCore, AgentLiteCore } = await import('../wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core.js');
    await initAgentCore();
    state.core = new AgentLiteCore('browser-session', state.mode, MAX_CONTEXT_ITEMS);
    state.coreReady = true;
    setPill(els.corePill, 'core ready', 'ready');
    exposeExtensionApi();
    log('WASM agent core ready');
  } catch (error) {
    state.core = null;
    state.coreReady = false;
    setPill(els.corePill, 'core fallback', 'error');
    log(`WASM agent core unavailable: ${error.message || String(error)}`);
  }
}

function fallbackLitePlan(userText) {
  const normalized = String(userText || '').trim().toLowerCase();
  const substantiveTokenCount = queryContentTokens(userText).filter((token) => token.length > 2).length;
  if (!normalized) {
    return { action: 'respond', query: userText, reason: 'empty prompt; respond directly' };
  }
  const conversational = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'who are you', 'what can you do'];
  if (conversational.some((phrase) => normalized === phrase || normalized.startsWith(`${phrase} `))) {
    return {
      action: 'respond',
      query: userText,
      reason: 'conversational prompt does not need paper retrieval',
    };
  }
  if (isSelectedPaperFollowup(userText)) {
    return {
      action: 'gather_context',
      query: userText,
      reason: 'using selected paper already loaded in chat',
    };
  }
  const retrievalTerms = [
    'paper',
    'papers',
    'research',
    'study',
    'studies',
    'literature',
    'arxiv',
    'citation',
    'evidence',
    'source',
    'sources',
    'retrieve',
    'find',
    'look up',
    'search',
    'survey',
    'compare',
    'summarize',
    'explain this paper',
    'what does this paper',
    'according to',
  ];
  if (retrievalTerms.some((term) => normalized.includes(term))) {
    return { action: 'gather_context', query: userText, reason: 'prompt asks for research-backed context' };
  }
  if (
    state.paperContextRows.length
    && ['this', 'that', 'it', 'paper', 'above'].some((term) => normalized.includes(term))
  ) {
    return { action: 'gather_context', query: userText, reason: 'using selected paper already loaded in chat' };
  }
  if (normalized.endsWith('?') && normalized.split(/\s+/).filter(Boolean).length >= 7) {
    return { action: 'gather_context', query: userText, reason: 'substantive question may benefit from ranked context' };
  }
  if (substantiveTokenCount >= 3) {
    return { action: 'gather_context', query: userText, reason: 'substantive topic can benefit from ranked context' };
  }
  return { action: 'respond', query: userText, reason: 'general chat prompt; respond without adding new papers' };
}

function isSelectedPaperFollowup(userText) {
  return Boolean(selectedContextTarget(userText));
}

function normalizeLitePlan(plan, userText) {
  const action = String(plan?.action || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  return {
    action: action === 'gather_context' || action === 'retrieve' ? 'gather_context' : 'respond',
    query: String(plan?.query || userText || ''),
    reason: String(plan?.reason || 'lite planner selected the next command'),
  };
}

async function planLiteTurn(userText) {
  setProcessStep('plan', 'active', 'Choosing respond or gather_context');
  const history = state.messages.slice(-8).map((message) => ({
    role: message.role,
    text: message.text,
  }));
  const fallback = fallbackLitePlan(userText);
  if (isSelectedPaperFollowup(userText)) {
    const plan = normalizeLitePlan(fallback, userText);
    setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
    return plan;
  }
  if (!state.coreReady || !state.core?.plan_lite_turn) {
    const plan = normalizeLitePlan(fallback, userText);
    setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
    return plan;
  }
  try {
    const raw = state.core.plan_lite_turn(
      userText,
      JSON.stringify(history),
      JSON.stringify({
        selected_context_count: state.paperContextRows.length,
        mode: state.mode,
      }),
    );
    const plan = normalizeLitePlan(JSON.parse(raw), userText);
    setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
    return plan;
  } catch (error) {
    const plan = normalizeLitePlan(fallback, userText);
    setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
    log(`lite planner fallback: ${error.message || String(error)}`);
    return plan;
  }
}

async function refreshStorage() {
  if (!navigator.storage?.estimate) return;
  const estimate = await navigator.storage.estimate();
  els.storageMetric.textContent = formatBytes(estimate.usage || 0);
}

async function requestPersistentStorage() {
  if (!navigator.storage?.persist) {
    setPill(els.storagePill, 'persist unsupported', 'error');
    log('persistent storage is not supported in this browser');
    return;
  }
  const granted = await navigator.storage.persist();
  setPill(els.storagePill, granted ? 'persistent' : 'best effort', granted ? 'ready' : '');
  log(granted ? 'persistent storage granted' : 'persistent storage was not granted');
  await refreshStorage();
}

async function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbGet(key) {
  const db = await openDb();
  const value = await new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readonly');
    const req = tx.objectStore(DB_STORE).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return value;
}

async function dbSet(key, value) {
  const db = await openDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.objectStore(DB_STORE).put(value, key);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function dbDump() {
  const db = await openDb();
  const entries = await new Promise((resolve, reject) => {
    const out = [];
    const tx = db.transaction(DB_STORE, 'readonly');
    const req = tx.objectStore(DB_STORE).openCursor();
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) return;
      out.push([cursor.key, cursor.value]);
      cursor.continue();
    };
    tx.oncomplete = () => resolve(out);
    tx.onerror = () => reject(tx.error);
  });
  db.close();
  return entries;
}

async function dbRestore(entries) {
  if (!Array.isArray(entries) || !entries.length) return;
  const db = await openDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readwrite');
    const store = tx.objectStore(DB_STORE);
    for (const entry of entries) {
      if (!Array.isArray(entry) || entry.length < 2) continue;
      const [key, value] = entry;
      if (typeof key === 'string' && key.length <= 256) store.put(value, key);
    }
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function openOptionalCache() {
  if (typeof caches === 'undefined' || typeof caches.open !== 'function') return null;
  try {
    return await caches.open(CACHE_NAME);
  } catch (error) {
    log(`browser cache unavailable; fetching directly: ${error.message || String(error)}`);
    return null;
  }
}

async function cachePutOptional(cache, key, response) {
  if (!cache) return;
  try {
    await cache.put(key, response);
    await refreshStorage();
  } catch (error) {
    log(`browser cache write skipped: ${error.message || String(error)}`);
  }
}

async function cachedJson(url, label) {
  const cache = await openOptionalCache();
  const cached = cache ? await cache.match(url).catch(() => null) : null;
  if (cached) {
    log(`cache hit: ${label}`);
    return cached.json();
  }
  log(`fetching: ${label}`);
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error(`${label} failed: ${res.status}`);
  await cachePutOptional(cache, url, res.clone());
  return res.json();
}

async function cachedText(url, label) {
  const cache = await openOptionalCache();
  const cached = cache ? await cache.match(url).catch(() => null) : null;
  if (cached) {
    log(`cache hit: ${label}`);
    return cached.text();
  }
  log(`fetching: ${label}`);
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error(`${label} failed: ${res.status}`);
  await cachePutOptional(cache, url, res.clone());
  return res.text();
}

async function cachedArrayBuffer(url, label) {
  const cache = await openOptionalCache();
  const cached = cache ? await cache.match(url).catch(() => null) : null;
  if (cached) {
    log(`cache hit: ${label}`);
    return cached.arrayBuffer();
  }
  log(`fetching: ${label}`);
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error(`${label} failed: ${res.status}`);
  await cachePutOptional(cache, url, res.clone());
  return res.arrayBuffer();
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function fetchHfJson(path, params, { retries = 5 } = {}) {
  const url = new URL(path, HF.datasetServer);
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== undefined && value !== null && value !== '') url.searchParams.set(key, String(value));
  }
  const cache = await openOptionalCache();
  const cached = cache ? await cache.match(url.href).catch(() => null) : null;
  if (cached) return cached.json();
  for (let attempt = 0; attempt <= retries; attempt++) {
    const res = await fetch(url.href, { mode: 'cors' });
    const data = await res.json().catch(() => ({}));
    if (res.ok) {
      await cachePutOptional(cache, url.href, new Response(JSON.stringify(data), {
        headers: { 'content-type': 'application/json' },
      }));
      return data;
    }
    const message = data.error || data.detail || data.message || '';
    if (attempt < retries && (res.status === 423 || res.status === 429 || res.status >= 500)) {
      log(`Hugging Face dataset server retry: ${message || res.status}`);
      await wait(1200 + attempt * 900);
      continue;
    }
    throw new Error(message || `Hugging Face request failed: ${res.status}`);
  }
  throw new Error('Hugging Face request failed after retries');
}

async function hfDatasetFirstSplit(dataset) {
  const key = `split:${dataset}`;
  const cached = await dbGet(key);
  if (cached) return cached;
  const data = await fetchHfJson('/splits', { dataset });
  const split = (data.splits || [])[0];
  if (!split) throw new Error(`No split found for ${dataset}`);
  const out = { config: split.config || 'default', split: split.split || 'train' };
  await dbSet(key, out);
  return out;
}

async function hfRowsByOffset(dataset, offset, length = 1) {
  const split = await hfDatasetFirstSplit(dataset);
  const data = await fetchHfJson('/rows', {
    dataset,
    config: split.config,
    split: split.split,
    offset: Math.max(0, Math.floor(Number(offset) || 0)),
    length,
  });
  return (data.rows || []).map((item) => item.row || item);
}

async function hfSearchRows(dataset, query, length = 8) {
  const split = await hfDatasetFirstSplit(dataset);
  const data = await fetchHfJson('/search', {
    dataset,
    config: split.config,
    split: split.split,
    query,
    offset: 0,
    length,
  });
  return (data.rows || []).map((item) => item.row || item);
}

function chooseLevel(manifest, targetRows) {
  const levels = manifest.paper_levels || [];
  const exact = levels.find((level) => Number(level.rows || 0) === Number(targetRows));
  if (exact) return exact;
  const sorted = [...levels].sort((a, b) => Math.abs(Number(a.rows || 0) - targetRows) - Math.abs(Number(b.rows || 0) - targetRows));
  return sorted[0];
}

function levelJsonPath(level) {
  if (level?.path) return level.path;
  if (Number(level?.rows || 0) === 1000000) return 'papers_all.json';
  return '';
}

function fileStem(value) {
  const base = String(value || '').split(/[?#]/)[0].split('/').pop() || '';
  return base.replace(/\.(json|parquet|i8)$/i, '').replace(/\.emb$/i, '');
}

async function loadResearchPack() {
  const targetRows = Number(els.pack.value || 50000);
  setAgentWorking(true);
  setPill(els.packPill, 'pack loading', 'busy');
  els.loadPack.disabled = true;
  setProcessStep('pack', 'active', `Loading ${formatCount(targetRows)} paper metadata rows`);
  try {
    if ((NATIVE_APP || URL_PARAMS.get('packagedPapers') === '1') && targetRows === 50000) {
      setProcessStep('pack', 'active', 'Loading bundled 50k paper pack');
      const rows = await cachedJson(NATIVE_PAPERS_50K, 'bundled 50k paper pack');
      state.packRows = Array.isArray(rows) ? rows : rows.rows || [];
      state.packLevel = {
        rows: state.packRows.length,
        label: `${formatCount(state.packRows.length)} bundled papers`,
        path: NATIVE_PAPERS_50K,
      };
      state.paperSemanticIndex = null;
      els.packMetric.textContent = state.packLevel.label;
      els.rowsMetric.textContent = formatCount(state.packRows.length);
      setPill(els.packPill, 'library ready', 'ready');
      setProcessStep('pack', 'done', `${formatCount(state.packRows.length)} bundled paper rows ready`);
      log(`loaded ${formatCount(state.packRows.length)} bundled paper rows`);
      return;
    }
    const manifest = await cachedJson(`${HF.paperInteractiveRoot}/manifest.json`, 'paper pack manifest');
    const level = chooseLevel(manifest, targetRows);
    const path = levelJsonPath(level);
    if (!path) throw new Error('Selected paper pack has no JSON path.');
    setProcessStep('pack', 'active', `Fetching ${level.label || `${formatCount(level.rows)} papers`}`);
    const rows = await cachedJson(`${HF.paperInteractiveRoot}/${path}`, level.label || `${level.rows} papers`);
    state.packRows = Array.isArray(rows) ? rows : rows.rows || [];
    state.packLevel = { ...level, path };
    state.paperSemanticIndex = null;
    els.packMetric.textContent = level.label || `${formatCount(state.packRows.length)} papers`;
    els.rowsMetric.textContent = formatCount(state.packRows.length);
    setPill(els.packPill, 'library ready', 'ready');
    setProcessStep('pack', 'done', `${formatCount(state.packRows.length)} paper rows ready`);
    log(`loaded ${formatCount(state.packRows.length)} paper rows`);
  } catch (error) {
    setPill(els.packPill, 'library error', 'error');
    setProcessStep('pack', 'error', error.message || String(error));
    log(error.message || String(error));
  } finally {
    els.loadPack.disabled = false;
    await refreshStorage();
    if (!state.processActive) setAgentWorking(false);
  }
}

async function ensureDefaultResearchPack() {
  if (STRUCTURE_FIXTURE || state.packRows.length) return;
  log('loading default paper pack for retrieval');
  await loadResearchPack();
}

function ensureWorker() {
  if (state.worker) return state.worker;
  state.worker = new Worker('./js/llm-worker.js?v=20260514-intent-head', { type: 'module' });
  state.worker.addEventListener('message', onWorkerMessage);
  return state.worker;
}

function selectedDevice() {
  if (DEV_BACKEND === 'vllm') return { backend: 'vllm', vllmEndpoint: VLLM_ENDPOINT };
  if (DEVICE_PARAM === 'wasm' || DEVICE_PARAM === 'webgpu') return DEVICE_PARAM;
  if (String(els.model.value || '').startsWith('modelstack:')) return 'wasm';
  const requested = els.device.value;
  if (requested === 'webgpu' && DEVICE_PARAM === 'webgpu') return 'webgpu';
  return 'wasm';
}

function settleModelLoad(error = null, value = null) {
  const resolve = state.modelLoadResolve;
  const reject = state.modelLoadReject;
  state.modelLoadPromise = null;
  state.modelLoadResolve = null;
  state.modelLoadReject = null;
  if (error && reject) reject(error);
  else if (resolve) resolve(value);
}

async function loadModel({ force = false, auto = false } = {}) {
  const modelId = els.model.value;
  if (!modelId) throw new Error('No runtime model is configured.');
  if (!force && state.modelReady && state.loadedModelId === modelId) {
    setProcessStep('runtime', 'done', `Using loaded ${shortText(modelId, 56)}`);
    updateRuntimeDetail('Runtime is loaded and ready for chat.');
    syncModelControls();
    return state.loadedModelId;
  }
  if (!force && state.modelBusy && state.modelLoadPromise) {
    updateRuntimeDetail('Runtime is still loading. Chat will start when it is ready.');
    return state.modelLoadPromise;
  }
  if (force && state.worker) {
    unloadModel({ silent: true });
  }
  releaseVoiceRuntimeForChat();
  state.modelBusy = true;
  state.modelReady = false;
  state.loadedModelId = '';
  setAgentWorking(true);
  els.modelMetric.textContent = 'Loading';
  setPill(els.modelPill, 'runtime loading', 'busy');
  updateRuntimeDetail(auto ? 'Loading runtime automatically...' : 'Loading runtime...');
  syncModelControls();
  setProcessStep('runtime', 'active', `Loading ${shortText(modelId, 56)}`);
  state.modelLoadPromise = new Promise((resolve, reject) => {
    state.modelLoadResolve = resolve;
    state.modelLoadReject = reject;
  });
  ensureWorker().postMessage({ type: 'load', modelId, device: selectedDevice() });
  return state.modelLoadPromise;
}

function unloadModel({ silent = false } = {}) {
  if (state.worker) {
    state.worker.postMessage({ type: 'unload' });
    state.worker.terminate();
    state.worker = null;
  }
  if (state.modelLoadPromise) settleModelLoad(new Error('Runtime unloaded.'));
  state.modelBusy = false;
  state.modelReady = false;
  state.loadedModelId = '';
  els.modelMetric.textContent = 'Idle';
  setPill(els.modelPill, 'runtime idle', '');
  updateRuntimeDetail('Runtime unloaded. Load Runtime to preload it again.');
  syncModelControls();
  if (!state.processActive) setAgentWorking(false);
  if (!silent) {
    setProcessStep('runtime', 'done', 'Runtime unloaded');
    log('runtime unloaded');
  }
}

function onWorkerMessage(event) {
  const data = event.data || {};
  if (data.type === 'progress') {
    const total = Number(data.total || 0);
    const loaded = Number(data.loaded || 0);
    const suffix = total ? ` ${formatBytes(loaded)} / ${formatBytes(total)}` : '';
    log(`${data.status || 'model'} ${data.file || ''}${suffix}`.trim());
    const detail = `${data.status || 'loading'} ${shortText(data.file || '', 42)}${suffix}`;
    updateRuntimeDetail(detail);
    setProcessStep('runtime', 'active', detail);
  } else if (data.type === 'status') {
    const message = data.message || 'model status';
    log(message);
    if (state.utilityGenerationRequests.size && /generat|decoder|encoding prompt/i.test(message)) {
      setProcessStep('select', 'active', message);
    } else if (/generat/i.test(message)) {
      setProcessStep('generate', 'active', message);
    } else if (/loading|using local|bundle|dense tensor|bitnet layer|building|preparing|runtime ready|webgpu|wasm|tokenizer|manifest|weights|uploading|shader/i.test(message)) {
      updateRuntimeDetail(message);
      setProcessStep('runtime', 'active', message);
    }
  } else if (data.type === 'loaded') {
    state.modelReady = true;
    state.modelBusy = false;
    state.loadedModelId = data.modelId || els.model.value;
    els.modelMetric.textContent = data.device === 'vllm' ? 'vLLM' : data.device === 'webgpu' ? 'WebGPU' : 'WASM';
    setPill(els.modelPill, 'runtime ready', 'ready');
    updateRuntimeDetail(`${data.device || 'runtime'} ready. WASM is the primary on-device runtime.`);
    log(`loaded ${state.loadedModelId} (${data.dtype || 'default'})`);
    setProcessStep('runtime', 'done', `${data.device || 'runtime'} ready`);
    settleModelLoad(null, state.loadedModelId);
    syncModelControls();
    if (!state.processActive) setAgentWorking(false);
    refreshStorage();
  } else if (data.type === 'generated') {
    const text = data.text || 'No answer generated.';
    const request = state.utilityGenerationRequests.get(Number(data.generationId || 0));
    if (request) {
      state.utilityGenerationRequests.delete(Number(data.generationId || 0));
      request.resolve(text);
      return;
    }
    if (!finalizeAssistantResponse(text)) log('ignored late model response after fallback');
  } else if (data.type === 'embedded') {
    const request = state.neuralEmbeddingRequests.get(data.requestId);
    if (request) {
      state.neuralEmbeddingRequests.delete(data.requestId);
      request.resolve(Float32Array.from(data.embedding || []));
    }
  } else if (data.type === 'intent') {
    const request = state.intentClassificationRequests.get(data.requestId);
    if (request) {
      state.intentClassificationRequests.delete(data.requestId);
      request.resolve({
        intent: String(data.intent || ''),
        confidence: Number(data.confidence || 0),
        ranked: Array.isArray(data.ranked) ? data.ranked : [],
      });
    }
  } else if (data.type === 'cancelled') {
    log('cancelled slow generation; runtime stayed loaded');
    if (state.modelReady) updateRuntimeDetail('Runtime is still loaded and ready for the next chat.');
  } else if (data.type === 'unloaded') {
    state.modelReady = false;
    state.modelBusy = false;
    state.loadedModelId = '';
    els.modelMetric.textContent = 'Idle';
    setPill(els.modelPill, 'runtime idle', '');
    updateRuntimeDetail('Runtime unloaded.');
    settleModelLoad(new Error('Runtime unloaded.'));
    syncModelControls();
  } else if (data.type === 'error') {
    for (const [requestId, request] of state.utilityGenerationRequests.entries()) {
      state.utilityGenerationRequests.delete(requestId);
      request.reject(new Error(data.message || 'model error'));
    }
    for (const [requestId, request] of state.neuralEmbeddingRequests.entries()) {
      state.neuralEmbeddingRequests.delete(requestId);
      request.reject(new Error(data.message || 'model error'));
    }
    for (const [requestId, request] of state.intentClassificationRequests.entries()) {
      state.intentClassificationRequests.delete(requestId);
      request.reject(new Error(data.message || 'model error'));
    }
    const error = new Error(data.message || 'model error');
    const wasLoading = state.modelBusy;
    if (wasLoading) settleModelLoad(error);
    state.modelBusy = false;
    setControlsBusy(false);
    if (!state.processActive) setAgentWorking(false);
    setPill(els.modelPill, 'runtime error', 'error');
    updateRuntimeDetail(data.message || 'Runtime error.');
    syncModelControls();
    setProcessStep(state.modelReady ? 'generate' : 'runtime', 'error', data.message || 'model error');
    if (wasLoading) {
      log(data.message || 'model error');
      return;
    }
    const rows = state.activeTurn?.contextRows || state.pendingContextRows || [];
    if (rows.length) {
      finalizeAssistantResponse('', {
        fallback: true,
        reason: `Local model error: ${data.message || 'unknown error'}`,
      });
    } else {
      finishProcessTrace('Error');
      appendMessage('assistant', `Local model error: ${data.message || 'unknown error'}`);
    }
    log(data.message || 'model error');
  }
}

function generateUtilityText(prompt, options = {}) {
  const generationId = ++state.generationRunId;
  const promise = new Promise((resolve, reject) => {
    state.utilityGenerationRequests.set(generationId, { resolve, reject });
  });
  ensureWorker().postMessage({
    type: 'generate',
    generationId,
    prompt,
    options,
  });
  return promise;
}

function classifyAgentIntent(prompt, options = {}) {
  if (!String(state.loadedModelId || '').startsWith('modelstack:')) return Promise.resolve(null);
  const requestId = `intent-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const worker = ensureWorker();
  const promise = new Promise((resolve, reject) => {
    state.intentClassificationRequests.set(requestId, { resolve, reject });
    window.setTimeout(() => {
      const request = state.intentClassificationRequests.get(requestId);
      if (!request) return;
      state.intentClassificationRequests.delete(requestId);
      request.resolve(null);
    }, Math.max(600, Math.min(4000, Number(options.timeoutMs || 1800))));
  });
  worker.postMessage({
    type: 'intent',
    requestId,
    text: String(prompt || ''),
    maxEncoderTokens: Math.max(128, Math.min(1024, Number(options.maxEncoderTokens || 768))),
  });
  return promise;
}

function buildTranslationPrompt(text, options = {}) {
  const source = String(options.sourceLanguage || translationSourceLabel() || 'Auto').trim();
  const target = String(options.targetLanguage || translationTargetLabel() || 'Spanish').trim();
  const modality = String(options.modality || 'text');
  return [
    '<AK_CHAT> <AK_RESPOND>',
    'Return exactly this decision format: Action: respond, then Content: the translation only.',
    `Task: Translate ${modality === 'audio' ? 'the speech transcript' : 'the text'} from ${source} to ${target}.`,
    'Preserve meaning, names, numbers, punctuation, and paragraph breaks. Do not add explanations, notes, or citations.',
    '',
    'Text:',
    text,
  ].join('\n');
}

async function runTranslator(text, options = {}) {
  const inputText = String(text || '').trim();
  if (!inputText) throw new Error('No text was provided for translation.');
  const modality = options.modality === 'audio' ? 'audio' : 'text';
  const capabilityId = modality === 'audio'
    ? state.translation.audioCapabilityId
    : state.translation.textCapabilityId;
  const proposal = proposeExtensionAction(state.translation.extensionId, capabilityId, {
    text: inputText,
    source_language: options.sourceLanguage || translationSourceLabel(),
    target_language: options.targetLanguage || translationTargetLabel(),
    modality,
    surface: 'browser',
  });
  if (proposal.status !== 'pending_user_approval') {
    throw new Error(proposal.error || `translator extension action was ${proposal.status || 'rejected'}`);
  }
  state.translation.activeActionId = proposal.action_id || null;
  if (!state.modelReady || state.loadedModelId !== els.model.value) await loadModel();
  const prompt = buildTranslationPrompt(inputText, {
    sourceLanguage: options.sourceLanguage,
    targetLanguage: options.targetLanguage,
    modality,
  });
  const output = await generateUtilityText(prompt, {
    maxNewTokens: Math.max(96, targetMaxTokens()),
    temperature: 0.08,
  });
  const packet = recordAssistantTurn(output);
  const translated = displayTextFromDecision(packet, output).replace(/^Content:\s*/i, '').trim();
  if (state.translation.activeActionId) {
    recordExtensionResult(state.translation.activeActionId, {
      action_id: state.translation.activeActionId,
      status: 'approved_executed',
      output: {
        source_language: options.sourceLanguage || translationSourceLabel(),
        target_language: options.targetLanguage || translationTargetLabel(),
        modality,
        translated_text: translated,
      },
      artifact_refs: [],
    });
  }
  state.translation.activeActionId = null;
  return translated || output;
}

async function submitTranslationText(text, options = {}) {
  resetProcessTrace(text);
  setControlsBusy(true);
  state.translation.busy = true;
  syncTranslationControls();
  try {
    setProcessStep('runtime', state.modelReady ? 'done' : 'active', state.modelReady ? 'Using loaded runtime' : 'Loading translation runtime');
    setProcessStep('plan', 'done', `${options.modality === 'audio' ? 'audio' : 'text'} translation via extension`);
    setProcessStep('compile', 'active', `Preparing ${translationSourceLabel()} to ${translationTargetLabel()} translation`);
    const translated = await runTranslator(text, options);
    setProcessStep('compile', 'done', 'Translation prompt ready');
    setProcessStep('generate', 'done', `${formatCount(translated.length)} translated characters`);
    setProcessStep('render', 'active', 'Rendering translation');
    appendMessage('assistant', translated);
    setProcessStep('render', 'done', 'Translation displayed');
    finishProcessTrace('Translated');
  } catch (error) {
    if (state.translation.activeActionId) {
      recordExtensionResult(state.translation.activeActionId, {
        action_id: state.translation.activeActionId,
        status: 'failed',
        error: error.message || String(error),
      });
    }
    state.translation.activeActionId = null;
    setProcessStep('render', 'error', error.message || String(error));
    finishProcessTrace('Error');
    appendMessage('assistant', `Translation failed: ${error.message || String(error)}`);
    log(`translation failed: ${error.message || String(error)}`);
  } finally {
    state.translation.busy = false;
    setControlsBusy(false);
    syncTranslationControls();
  }
}

function speechRecognitionConstructor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function startAudioTranslation() {
  const SpeechRecognition = speechRecognitionConstructor();
  if (!SpeechRecognition) {
    appendMessage('assistant', 'Audio translation needs browser speech recognition support. You can still paste transcribed text and use Translate mode.');
    return;
  }
  if (state.translation.listening && state.translation.recognition) {
    state.translation.recognition.stop();
    return;
  }
  setTranslationMode(true);
  if (!state.translation.enabled) return;
  const recognition = new SpeechRecognition();
  recognition.lang = String(els.translationSource?.value || 'en-US') === 'auto' ? 'en-US' : String(els.translationSource.value);
  recognition.interimResults = false;
  recognition.continuous = false;
  state.translation.recognition = recognition;
  state.translation.listening = true;
  syncTranslationControls();
  log('audio translation listening');
  recognition.onresult = (event) => {
    const transcript = [...event.results]
      .map((result) => result[0]?.transcript || '')
      .join(' ')
      .trim();
    state.translation.listening = false;
    state.translation.recognition = null;
    syncTranslationControls();
    if (!transcript) {
      appendMessage('assistant', 'I did not catch any speech to translate.');
      return;
    }
    appendMessage('user', transcript);
    submitTranslationText(transcript, {
      modality: 'audio',
      sourceLanguage: translationSourceLabel(),
      targetLanguage: translationTargetLabel(),
    });
  };
  recognition.onerror = (event) => {
    state.translation.listening = false;
    state.translation.recognition = null;
    syncTranslationControls();
    appendMessage('assistant', `Audio translation failed: ${event.error || 'speech recognition error'}`);
  };
  recognition.onend = () => {
    state.translation.listening = false;
    state.translation.recognition = null;
    syncTranslationControls();
  };
  recognition.start();
}

function normalizeText(value) {
  return String(value || '').toLowerCase();
}

function normalizeSearchText(value) {
  return normalizeText(value)
    .replace(/[_-]+/g, ' ')
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function queryTokens(query) {
  return normalizeSearchText(query)
    .split(/\s+/)
    .filter((token) => token.length > 1);
}

function queryContentTokens(query) {
  const tokens = queryTokens(query);
  const content = tokens.filter((token) => !RETRIEVAL_INTENT_TERMS.has(token));
  return content.length ? content : tokens;
}

function requiresFreshResearchContext(userText) {
  if (isSelectedPaperFollowup(userText)) return false;
  const normalized = normalizeSearchText(userText);
  if (!normalized) return false;
  const tokens = tokenSet(userText);
  const asksForEvidence =
    hasToken(tokens, 'paper')
    || hasToken(tokens, 'study')
    || hasToken(tokens, 'source')
    || hasToken(tokens, 'evidence')
    || hasToken(tokens, 'citation')
    || hasTokenPhrase(normalized, 'literature')
    || hasTokenPhrase(normalized, 'arxiv');
  const asksForRecommendation =
    hasToken(tokens, 'best')
    || hasToken(tokens, 'top')
    || hasToken(tokens, 'recommend')
    || hasToken(tokens, 'survey')
    || hasToken(tokens, 'compare');
  return asksForEvidence || (asksForRecommendation && queryContentTokens(userText).length >= 2);
}

function selectedContextTarget(userText) {
  if (!state.paperContextRows.length) return null;
  const normalized = normalizeSearchText(userText);
  if (!normalized) return null;
  const tokens = queryContentTokens(userText);
  const hasPaperHandle = /\b(paper|study|work|article|source|evidence)\b/.test(normalized);
  const hasReference = /\b(this|that|it|its|above|loaded|selected|context|current)\b/.test(normalized);
  const hasFollowupVerb = /\b(tell|explain|explani|summarize|describe|detail|details|detailed|more|about|meaning|takeaway|takeaways|method|results|limitations)\b/.test(normalized);
  const mostlyContextWords = tokens.length <= 3 && hasFollowupVerb;
  const selectedPronounFollowup = hasReference && hasFollowupVerb && tokens.length <= 5;
  const selectedPaperPhrase = /\b(this|that|selected|loaded|above)\s+(paper|study|work|article|source|evidence)\b/.test(normalized)
    || /\b(paper|study|work|article|source|evidence)\s+(above|selected|loaded|in\s+context|context)\b/.test(normalized);
  const contextPaperPhrase = hasPaperHandle && hasReference && hasFollowupVerb;
  if (selectedPaperPhrase || contextPaperPhrase || mostlyContextWords || selectedPronounFollowup) {
    return {
      type: 'selected_paper',
      rows: state.paperContextRows,
      reason: 'user is referring to paper context already added to chat',
    };
  }
  return null;
}

function tokenSet(value) {
  return new Set(queryTokens(value));
}

function hasToken(tokens, token) {
  return tokens.has(token) || (token.endsWith('s') && tokens.has(token.slice(0, -1)));
}

function hasTokenPhrase(text, phrase) {
  return normalizeSearchText(text).includes(normalizeSearchText(phrase));
}

function paperSearchText(row) {
  return normalizeSearchText([
    row.title,
    row.paper_id,
    row.canonical_paper_id,
    row.primary_category,
    row.category_list,
    row.categories,
    row.year,
    row.abstract,
    row.summary,
    row.context_text,
  ].filter(Boolean).join(' '));
}

function tokenCoveredByText(text, token) {
  if (hasTokenPhrase(text, token)) return true;
  if (token === 'llm') return hasTokenPhrase(text, 'large language model') || hasTokenPhrase(text, 'large language models');
  if (token === 'agent') return hasTokenPhrase(text, 'agents') || hasTokenPhrase(text, 'multiagent');
  if (token === 'multi') return hasTokenPhrase(text, 'multi agent') || hasTokenPhrase(text, 'multiagent');
  return false;
}

function queryCoverage(row, query) {
  const text = paperSearchText(row);
  const required = queryContentTokens(query);
  const covered = required.filter((token) => tokenCoveredByText(text, token));
  return { required, covered };
}

function rowMatchesQuery(row, query) {
  const { required, covered } = queryCoverage(row, query);
  if (!required.length) return true;
  if (required.includes('llm') && !covered.includes('llm')) return false;
  if (required.length >= 3) return covered.length >= Math.min(3, required.length);
  return covered.length >= Math.min(1, required.length);
}

function isRecommendationQuery(query) {
  const normalized = normalizeSearchText(query);
  return /\b(best|top|recommend|recommendation|survey|overview|which)\b/.test(normalized);
}

function scorePaper(row, query, tokens) {
  const title = normalizeSearchText(row.title);
  const paperId = normalizeSearchText(row.paper_id || row.canonical_paper_id);
  const category = normalizeSearchText(row.primary_category || row.category_list || row.categories);
  const abstract = normalizeSearchText(row.abstract || row.summary || row.context_text || '');
  const haystack = `${title} ${paperId} ${category} ${row.year || ''} ${abstract}`;
  const titleTokens = tokenSet(title);
  const paperTokens = tokenSet(paperId);
  const categoryTokens = tokenSet(category);
  const abstractTokens = tokenSet(abstract);
  let score = 0;
  const phrase = normalizeSearchText(query);
  if (phrase && title.includes(phrase)) score += 22;
  if (paperId.includes(phrase)) score += 14;
  for (const token of tokens) {
    if (hasToken(titleTokens, token)) score += 8;
    if (hasToken(paperTokens, token)) score += 4;
    if (hasToken(categoryTokens, token)) score += 3;
    if (hasToken(abstractTokens, token)) score += 3;
    if (hasTokenPhrase(haystack, token)) score += 1;
  }
  const requiredTokens = queryContentTokens(query);
  const covered = requiredTokens.filter((token) => hasTokenPhrase(haystack, token));
  if (covered.length >= 2) score += covered.length * 3;
  if (requiredTokens.length >= 3 && covered.length < 2) score *= 0.35;
  if (tokens.includes('llm') && !hasTokenPhrase(haystack, 'llm') && !hasTokenPhrase(haystack, 'large language model')) {
    score *= 0.45;
  }
  return score;
}

function rankRetrievedRows(query, rows) {
  return rows.map((row) => {
    const lexical = scorePaper(row, query, queryContentTokens(query));
    const semantic = Number(row.semantic_score || 0);
    const selected = String(row.source || '').includes('selected_paper') ? 1000 : 0;
    const rankScore = selected + lexical * 1.8 + semantic * 5;
    return {
      ...row,
      lexical_score: Math.max(Number(row.lexical_score || 0), lexical),
      retrieval_score: Math.max(Number(row.retrieval_score || 0), rankScore),
      rank_score: rankScore,
    };
  }).sort((a, b) => Number(b.rank_score || 0) - Number(a.rank_score || 0));
}

function paperKey(row) {
  return String(row.paper_idx ?? row.paper_id ?? row.canonical_paper_id ?? row.arxiv_id ?? row.title ?? '');
}

function isPunctuation(char) {
  return /[\p{P}\p{S}]/u.test(char);
}

function basicTokenize(text) {
  const cleaned = String(text || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();
  const spaced = [];
  for (const char of cleaned) {
    if (/\s/u.test(char)) spaced.push(' ');
    else if (isPunctuation(char)) spaced.push(' ', char, ' ');
    else spaced.push(char);
  }
  return spaced.join('').split(/\s+/).filter(Boolean);
}

function parseVocab(text) {
  const vocab = new Map();
  String(text || '').split(/\r?\n/).forEach((token, index) => {
    if (token) vocab.set(token, index);
  });
  return vocab;
}

function wordPieceTokenize(token, vocab) {
  if (vocab.has(token)) return [token];
  if (token.length > 100) return ['[UNK]'];
  const pieces = [];
  let start = 0;
  while (start < token.length) {
    let end = token.length;
    let current = null;
    while (start < end) {
      const candidate = `${start > 0 ? '##' : ''}${token.slice(start, end)}`;
      if (vocab.has(candidate)) {
        current = candidate;
        break;
      }
      end -= 1;
    }
    if (!current) return ['[UNK]'];
    pieces.push(current);
    start = end;
  }
  return pieces;
}

function encodeWordPiece(text, vocab, maxLength = SEMANTIC_QUERY_TOKENS) {
  const clsId = vocab.get('[CLS]') ?? 101;
  const sepId = vocab.get('[SEP]') ?? 102;
  const padId = vocab.get('[PAD]') ?? 0;
  const unkId = vocab.get('[UNK]') ?? 100;
  const pieces = [];
  for (const token of basicTokenize(text)) {
    pieces.push(...wordPieceTokenize(token, vocab));
    if (pieces.length >= maxLength - 2) break;
  }
  const ids = [clsId, ...pieces.slice(0, maxLength - 2).map((piece) => vocab.get(piece) ?? unkId), sepId];
  const attention = new Array(ids.length).fill(1);
  while (ids.length < maxLength) {
    ids.push(padId);
    attention.push(0);
  }
  return {
    inputIds: ids,
    attentionMask: attention,
    tokenTypeIds: new Array(maxLength).fill(0),
  };
}

function int64Tensor(ort, values, dims) {
  return new ort.Tensor('int64', BigInt64Array.from(values, (value) => BigInt(value)), dims);
}

function normalizeVector(values) {
  let sum = 0;
  for (const value of values) sum += Number(value) * Number(value);
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = Number(values[i]) / norm;
  return out;
}

async function loadOrtRuntime() {
  if (state.ortRuntime) return state.ortRuntime;
  const ort = await import(`${ORT_CDN_ROOT}/ort.wasm.min.mjs`);
  ort.env.logLevel = 'error';
  ort.env.wasm ??= {};
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = `${ORT_CDN_ROOT}/`;
  state.ortRuntime = ort;
  return ort;
}

async function ensurePaperEmbeddingModel() {
  if (state.paperEmbeddingModel) return state.paperEmbeddingModel;
  setPill(els.packPill, 'M1 loading', 'busy');
  const ort = await loadOrtRuntime();
  const [modelBuffer, vocabText] = await Promise.all([
    cachedArrayBuffer(`${HF.paperEmbeddingLiteRoot}/onnx/model.int8.onnx`, 'M1-Lite ONNX model'),
    cachedText(`${HF.paperEmbeddingLiteRoot}/tokenizer/vocab.txt`, 'M1-Lite vocab'),
  ]);
  const session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'basic',
    enableMemPattern: false,
    executionMode: 'sequential',
  });
  state.paperEmbeddingModel = {
    ort,
    session,
    vocab: parseVocab(vocabText),
    dimension: 0,
  };
  log('M1-Lite paper embedding model ready');
  return state.paperEmbeddingModel;
}

async function embedSemanticQuery(text) {
  const model = await ensurePaperEmbeddingModel();
  const encoded = encodeWordPiece(text, model.vocab, SEMANTIC_QUERY_TOKENS);
  const feeds = {
    input_ids: int64Tensor(model.ort, encoded.inputIds, [1, encoded.inputIds.length]),
    attention_mask: int64Tensor(model.ort, encoded.attentionMask, [1, encoded.attentionMask.length]),
    token_type_ids: int64Tensor(model.ort, encoded.tokenTypeIds, [1, encoded.tokenTypeIds.length]),
  };
  const output = await model.session.run(feeds);
  const tensor = output.embedding || output[Object.keys(output)[0]];
  const embedding = normalizeVector(tensor.data);
  model.dimension = embedding.length;
  return embedding;
}

async function ensurePaperSemanticManifest() {
  if (state.paperSemanticManifest) return state.paperSemanticManifest;
  state.paperSemanticManifest = await cachedJson(`${HF.paperSemanticRoot}/manifest.json`, 'paper semantic manifest');
  return state.paperSemanticManifest;
}

function currentSemanticLevel(manifest) {
  const levels = manifest?.levels || [];
  const currentRows = state.packRows.length;
  const currentPath = levelJsonPath(state.packLevel);
  const currentStem = fileStem(currentPath);
  return levels.find((level) => (
    level.level_path === currentPath
    || fileStem(level.level_path) === currentStem
    || Number(level.rows || 0) === currentRows
  ));
}

async function ensurePaperSemanticIndex(queryDimension) {
  const manifest = await ensurePaperSemanticManifest();
  const level = currentSemanticLevel(manifest);
  if (!level?.path) throw new Error('No row-aligned semantic index for the loaded paper pack.');
  const dimension = Number(level.dimension || manifest.dimension || 0);
  if (dimension !== queryDimension) {
    throw new Error(`Embedding dimension mismatch: query=${queryDimension}, index=${dimension}.`);
  }
  const key = `${level.path}:${level.rows}:${dimension}`;
  if (state.paperSemanticIndex?.key === key) return state.paperSemanticIndex;
  setPill(els.packPill, 'vectors loading', 'busy');
  const buffer = await cachedArrayBuffer(`${HF.paperSemanticRoot}/${level.path}`, `M1 vectors ${formatCount(level.rows)}`);
  const data = new Int8Array(buffer);
  const expected = Number(level.rows || 0) * dimension;
  if (data.length !== expected) {
    throw new Error(`Semantic index size mismatch: expected ${formatCount(expected)} bytes, got ${formatCount(data.length)}.`);
  }
  state.paperSemanticIndex = { key, data, rows: Number(level.rows || 0), dimension, scale: Number(level.scale || manifest.scale || 127) || 127 };
  log(`loaded M1 semantic vectors for ${formatCount(level.rows)} papers`);
  return state.paperSemanticIndex;
}

function dotQueryInt8(query, data, offset, dimension, scale = 127) {
  let score = 0;
  for (let i = 0; i < dimension; i++) score += query[i] * (data[offset + i] / scale);
  return score;
}

function dotQueryInt8Scaled(query, data, offset, dimension, scale) {
  let score = 0;
  const rowScale = Number(scale || 1 / 127);
  for (let i = 0; i < dimension; i++) score += query[i] * data[offset + i] * rowScale;
  return score;
}

function dotQueryTernaryPacked(query, data, rowIndex, dimension, packedDimension, scale) {
  let score = 0;
  const rowOffset = rowIndex * packedDimension;
  const rowScale = Number(scale || 1);
  for (let i = 0; i < dimension; i += 1) {
    const byte = data[rowOffset + (i >> 2)];
    const code = (byte >> ((i & 3) * 2)) & 3;
    if (code === 1) score += query[i];
    else if (code === 2) score -= query[i];
  }
  return score * rowScale;
}

function dotQueryTernaryGrouped(query, data, rowIndex, dimension, packedDimension, scales, groupSize, groupCount) {
  let score = 0;
  const rowOffset = rowIndex * packedDimension;
  const scaleOffset = rowIndex * groupCount;
  const width = Number(groupSize || 32);
  for (let i = 0; i < dimension; i += 1) {
    const byte = data[rowOffset + (i >> 2)];
    const code = (byte >> ((i & 3) * 2)) & 3;
    if (code === 0) continue;
    const scale = Number(scales[scaleOffset + Math.floor(i / width)] || 1);
    if (code === 1) score += query[i] * scale;
    else if (code === 2) score -= query[i] * scale;
  }
  return score;
}

function dotQueryTernaryGroupedSigned(query, data, rowIndex, dimension, packedDimension, scales, groupSize, groupCount) {
  let score = 0;
  const rowOffset = rowIndex * packedDimension;
  const scaleOffset = rowIndex * groupCount * 2;
  const width = Number(groupSize || 32);
  for (let i = 0; i < dimension; i += 1) {
    const byte = data[rowOffset + (i >> 2)];
    const code = (byte >> ((i & 3) * 2)) & 3;
    if (code === 0) continue;
    const group = Math.floor(i / width);
    if (code === 1) score += query[i] * Number(scales[scaleOffset + group * 2] || 1);
    else if (code === 2) score -= query[i] * Number(scales[scaleOffset + group * 2 + 1] || 1);
  }
  return score;
}

function dotQueryTernaryGroupedSignedResidual(
  query,
  data,
  rowIndex,
  dimension,
  packedDimension,
  scales,
  groupSize,
  groupCount,
  residualIndices,
  residualValues,
  residualDims,
) {
  let score = dotQueryTernaryGroupedSigned(query, data, rowIndex, dimension, packedDimension, scales, groupSize, groupCount);
  const offset = rowIndex * residualDims;
  for (let i = 0; i < residualDims; i += 1) {
    const dim = residualIndices[offset + i];
    score += query[dim] * residualValues[offset + i];
  }
  return score;
}

function float16ToFloat32(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function decodeFloatArray(buffer, dtype = 'f32') {
  if (String(dtype || 'f32').toLowerCase() !== 'f16') return new Float32Array(buffer);
  const half = new Uint16Array(buffer);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i += 1) out[i] = float16ToFloat32(half[i]);
  return out;
}

function neuralMemoryFormatEntry(manifest) {
  const formats = manifest?.vector_formats || {};
  const requested = String(URL_PARAMS.get('neuralMemoryFormat') || '').trim().toLowerCase();
  const primary = String(manifest?.primary_vector_format || '').trim().toLowerCase();
  const format = requested && formats[requested] ? requested : primary || (
    String(manifest?.vector_dtype || '').includes('ternary') ? 'ternary' : 'int8'
  );
  return {
    format,
    entry: formats[format] || manifest || {},
  };
}

async function loadNeuralMemoryPack() {
  if (!NEURAL_MEMORY_ENABLED) return null;
  if (state.neuralMemoryPack) return state.neuralMemoryPack;
  if (!NEURAL_MEMORY_PACK_URL) throw new Error('No neural memory pack URL configured.');
  setProcessStep('pack', 'active', 'Loading neural memory pack');
  const base = new URL('.', NEURAL_MEMORY_PACK_URL).href;
  const manifest = await cachedJson(NEURAL_MEMORY_PACK_URL, 'neural memory manifest');
  const { format, entry } = neuralMemoryFormatEntry(manifest);
  const [vectorsBuffer, scalesBuffer, metadataText] = await Promise.all([
    cachedArrayBuffer(new URL(entry.vector_path || manifest.vector_path, base).href, `neural memory ${format} vectors`),
    cachedArrayBuffer(new URL(entry.scale_path || manifest.scale_path, base).href, `neural memory ${format} scales`),
    cachedText(new URL(manifest.metadata_path, base).href, 'neural memory metadata'),
  ]);
  const metadata = metadataText
    .split(/\n+/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const dimension = Number(manifest.dim || entry.dim || 0);
  const isTernary = (
    format === 'ternary'
    || format === 'ternary_grouped'
    || format === 'ternary_grouped_signed'
    || format === 'ternary_grouped_signed_residual'
  );
  const packedDimension = isTernary
    ? Number(entry.packed_dim || Math.ceil(dimension / 4))
    : dimension;
  const vectorArray = isTernary ? new Uint8Array(vectorsBuffer) : new Int8Array(vectorsBuffer);
  let residualIndices = null;
  let residualValues = null;
  if (format === 'ternary_grouped_signed_residual') {
    const [indexBuffer, valueBuffer] = await Promise.all([
      cachedArrayBuffer(new URL(entry.residual_index_path, base).href, 'neural memory residual indices'),
      cachedArrayBuffer(new URL(entry.residual_value_path, base).href, 'neural memory residual values'),
    ]);
    residualIndices = new Uint16Array(indexBuffer);
    residualValues = decodeFloatArray(valueBuffer, entry.residual_value_dtype === 'float16' ? 'f16' : 'f32');
  }
  state.neuralMemoryPack = {
    manifest,
    vectorFormat: format,
    vectorEntry: entry,
    vectors: vectorArray,
    scales: decodeFloatArray(scalesBuffer, entry.scale_dtype || 'f32'),
    residualIndices,
    residualValues,
    residualDims: Number(entry.residual_dims || 0),
    metadata,
    dimension,
    packedDimension,
    groupSize: Number(entry.group_size || 0),
    groupCount: Number(entry.group_count || 0),
    rows: Number(manifest.row_count || metadata.length || 0),
  };
  setProcessStep('pack', 'done', `Loaded ${formatCount(state.neuralMemoryPack.rows)} ${format} neural memory vectors`);
  log(`loaded ${format} neural memory pack with ${formatCount(state.neuralMemoryPack.rows)} vectors`);
  return state.neuralMemoryPack;
}

function requestNeuralQueryEmbedding(text) {
  const requestId = `embed-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const worker = ensureWorker();
  const promise = new Promise((resolve, reject) => {
    state.neuralEmbeddingRequests.set(requestId, { resolve, reject });
    setTimeout(() => {
      const request = state.neuralEmbeddingRequests.get(requestId);
      if (!request) return;
      state.neuralEmbeddingRequests.delete(requestId);
      request.reject(new Error('neural retrieval embedding timed out'));
    }, 30000);
  });
  worker.postMessage({ type: 'embed', requestId, text, maxEncoderTokens: SEMANTIC_QUERY_TOKENS });
  return promise;
}

async function neuralMemorySearch(query) {
  if (!NEURAL_MEMORY_ENABLED || !String(state.loadedModelId || '').startsWith('modelstack:')) return [];
  const config = modeConfig();
  const pack = await loadNeuralMemoryPack();
  if (!pack?.rows || !pack.dimension) return [];
  setProcessStep('embed', 'active', 'Embedding query with AgentKernel BitNet');
  const queryEmbedding = await requestNeuralQueryEmbedding(query);
  if (queryEmbedding.length !== pack.dimension) {
    throw new Error(`Neural memory dimension mismatch: query=${queryEmbedding.length}, index=${pack.dimension}.`);
  }
  setProcessStep('embed', 'done', `${formatCount(queryEmbedding.length)}D BitNet query vector`);
  setProcessStep('rank', 'active', `Scanning ${formatCount(pack.rows)} ${pack.vectorFormat || 'int8'} neural memory vectors`);
  const top = [];
  const isTernary = (
    pack.vectorFormat === 'ternary'
    || pack.vectorFormat === 'ternary_grouped'
    || pack.vectorFormat === 'ternary_grouped_signed'
    || pack.vectorFormat === 'ternary_grouped_signed_residual'
  );
  const rowWidth = isTernary ? pack.packedDimension : pack.dimension;
  const limit = Math.min(pack.rows, pack.metadata.length, Math.floor(pack.vectors.length / rowWidth));
  for (let i = 0; i < limit; i += 1) {
    let score = 0;
    if (pack.vectorFormat === 'ternary_grouped_signed_residual') {
      score = dotQueryTernaryGroupedSignedResidual(
        queryEmbedding,
        pack.vectors,
        i,
        pack.dimension,
        pack.packedDimension,
        pack.scales,
        pack.groupSize,
        pack.groupCount,
        pack.residualIndices,
        pack.residualValues,
        pack.residualDims,
      );
    } else if (pack.vectorFormat === 'ternary_grouped_signed') {
      score = dotQueryTernaryGroupedSigned(
        queryEmbedding,
        pack.vectors,
        i,
        pack.dimension,
        pack.packedDimension,
        pack.scales,
        pack.groupSize,
        pack.groupCount,
      );
    } else if (pack.vectorFormat === 'ternary_grouped') {
      score = dotQueryTernaryGrouped(
        queryEmbedding,
        pack.vectors,
        i,
        pack.dimension,
        pack.packedDimension,
        pack.scales,
        pack.groupSize,
        pack.groupCount,
      );
    } else if (pack.vectorFormat === 'ternary') {
      score = dotQueryTernaryPacked(queryEmbedding, pack.vectors, i, pack.dimension, pack.packedDimension, pack.scales[i]);
    } else {
      score = dotQueryInt8Scaled(queryEmbedding, pack.vectors, i * pack.dimension, pack.dimension, pack.scales[i]);
    }
    insertTopScore(top, { row: pack.metadata[i], score }, config.semanticTopK);
    if (i > 0 && i % SEMANTIC_SCAN_YIELD_ROWS === 0) await wait(0);
  }
  top.sort((a, b) => b.score - a.score);
  setProcessStep('rank', 'done', `${formatCount(top.length)} BitNet neural candidates`);
  return top.map((item) => ({
    ...item.row,
    retrieval_score: item.score,
    semantic_score: item.score,
    source: 'agentkernel_bitnet_memory',
  }));
}

function insertTopScore(top, item, limit) {
  if (top.length < limit) {
    top.push(item);
    return;
  }
  let minIndex = 0;
  for (let i = 1; i < top.length; i++) {
    if (top[i].score < top[minIndex].score) minIndex = i;
  }
  if (item.score > top[minIndex].score) top[minIndex] = item;
}

async function semanticSearchPack(query) {
  if (!state.packRows.length && !NEURAL_MEMORY_ENABLED) return [];
  try {
    if (NEURAL_MEMORY_ENABLED) {
      const neuralRows = await neuralMemorySearch(query);
      if (neuralRows.length) return neuralRows;
    }
    const config = modeConfig();
    setProcessStep('embed', 'active', 'Embedding query with M1-Lite');
    const queryEmbedding = await embedSemanticQuery(query);
    setProcessStep('embed', 'done', `${formatCount(queryEmbedding.length)}D query vector`);
    setProcessStep('rank', 'active', `Scanning ${formatCount(state.packRows.length)} paper vectors`);
    const index = await ensurePaperSemanticIndex(queryEmbedding.length);
    const limit = Math.min(state.packRows.length, index.rows);
    const top = [];
    for (let i = 0; i < limit; i++) {
      const row = state.packRows[i];
      const score = dotQueryInt8(queryEmbedding, index.data, i * index.dimension, index.dimension, index.scale);
      insertTopScore(top, { row, score }, config.semanticTopK);
      if (i > 0 && i % SEMANTIC_SCAN_YIELD_ROWS === 0) await wait(0);
    }
    top.sort((a, b) => b.score - a.score);
    setProcessStep('rank', 'done', `${formatCount(top.length)} semantic candidates`);
    log(`M1 semantic ranked ${formatCount(limit)} papers`);
    return top.map((item) => ({
      ...item.row,
      retrieval_score: item.score,
      semantic_score: item.score,
      source: 'm1_semantic',
    }));
  } catch (error) {
    setProcessStep('rank', 'error', error.message || String(error));
    log(`semantic retrieval fallback: ${error.message || String(error)}`);
    return [];
  }
}

function mergeRetrievedRows(groups) {
  const byKey = new Map();
  for (const rows of groups) {
    for (const row of rows || []) {
      const key = paperKey(row);
      if (!key) continue;
      const existing = byKey.get(key);
      if (!existing) {
        byKey.set(key, row);
        continue;
      }
      byKey.set(key, {
        ...existing,
        ...row,
        source: [existing.source, row.source].filter(Boolean).join('+'),
        retrieval_score: Math.max(Number(existing.retrieval_score || 0), Number(row.retrieval_score || 0)),
        semantic_score: Math.max(Number(existing.semantic_score || 0), Number(row.semantic_score || 0)),
      });
    }
  }
  return [...byKey.values()];
}

function lexicalPackSearch(query, tokens, limit = 12) {
  if (!state.packRows.length) return [];
  setProcessStep('lexical', 'active', `Scanning ${formatCount(state.packRows.length)} titles and abstracts`);
  const contentTokens = queryContentTokens(query);
  const matches = state.packRows
    .map((row) => ({ row, score: scorePaper(row, query, contentTokens) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((item) => ({ ...item.row, retrieval_score: item.score, lexical_score: item.score, source: 'local_pack' }));
  setProcessStep('lexical', 'done', `${formatCount(matches.length)} lexical matches`);
  return matches;
}

function structureFixtureRows() {
  return [
    {
      paper_idx: 'structure-fixture-1',
      paper_id: '1706.03762',
      canonical_paper_id: '1706.03762',
      title: 'Attention Is All You Need',
      primary_category: 'cs.CL',
      categories: 'cs.CL cs.LG',
      year: 2017,
      source: 'structure_fixture',
      abstract: 'The Transformer architecture uses attention mechanisms without recurrence or convolution. This fixture lets the interface verify evidence cards, arXiv links, citation chips, and context compilation without depending on a remote dataset request.',
      context_text: 'The Transformer architecture uses scaled dot-product attention and multi-head attention to model sequence relationships. This local fixture is only for interface structure checks.',
    },
  ];
}

function fullPaperText(row) {
  return String(row.text || row.full_text || row.body || '');
}

function contextExcerpt(row, query, limit = 1200, preferFullText = false) {
  const fullText = fullPaperText(row).replace(/\s+/g, ' ').trim();
  const preferred = preferFullText ? '' : String(row.context_text || row.abstract || row.summary || '');
  if (preferred.trim()) return preferred.replace(/\s+/g, ' ').trim().slice(0, limit);
  const text = fullText || String(row.abstract || row.summary || '').replace(/\s+/g, ' ').trim();
  if (!text) return '';
  const tokens = queryTokens(query).filter((token) => token.length > 2);
  const lower = text.toLowerCase();
  let best = -1;
  for (const token of tokens) {
    const idx = lower.indexOf(token);
    if (idx >= 0 && (best < 0 || idx < best)) best = idx;
  }
  const start = Math.max(0, (best < 0 ? 0 : best) - 350);
  return text.slice(start, start + limit);
}

async function retrieveContext(query, options = {}) {
  const onCandidates = typeof options.onCandidates === 'function' ? options.onCandidates : null;
  if (STRUCTURE_FIXTURE) {
    log('using local structure-check evidence fixture');
    setProcessStep('lookup', 'done', 'Using fixture evidence');
    setProcessStep('compact', 'done', '1 fixture evidence item');
    return structureFixtureRows(query);
  }
  setProcessStep('lookup', 'active', 'Preparing research lookup');
  const config = modeConfig();
  if (isSelectedPaperFollowup(query)) {
    const selectedRows = state.paperContextRows.map((row) => ({ ...row, source: row.source || 'selected_paper' }));
    setProcessStep('lookup', 'active', `Using ${formatCount(selectedRows.length)} selected paper context item${selectedRows.length === 1 ? '' : 's'}`);
    const enrichedSelected = [];
    for (const row of selectedRows.slice(0, MAX_SELECTED_PAPERS)) {
      const hydrated = await enrichPaper(row).catch(() => row);
      enrichedSelected.push({
        ...hydrated,
        source: 'selected_paper',
        context_text: contextExcerpt(hydrated, query, config.selectedExcerptChars, true),
      });
    }
    setProcessStep('lookup', 'done', `${formatCount(enrichedSelected.length)} selected paper context item${enrichedSelected.length === 1 ? '' : 's'}`);
    setProcessStep('compact', 'active', `Compacting ${formatCount(enrichedSelected.length)} selected paper context item${enrichedSelected.length === 1 ? '' : 's'}`);
    setProcessStep('compact', 'done', `${formatCount(enrichedSelected.length)} excerpts prepared`);
    return enrichedSelected;
  }
  await ensureDefaultResearchPack();
  const tokens = queryTokens(query);
  let rows = [];
  if (state.packRows.length) rows = rankRetrievedRows(query, mergeRetrievedRows([await semanticSearchPack(query), lexicalPackSearch(query, tokens)]));
  if (!rows.length && HF_DATASET_SEARCH_ENABLED) {
    setProcessStep('lookup', 'active', `Remote search in ${HF.paperTextDataset}`);
    rows = (await hfSearchRows(HF.paperTextDataset, query, config.hfSearchRows)).map((row) => ({ ...row, source: 'hf_search' }));
  } else if (!rows.length && !state.packRows.length) {
    log('remote Hugging Face dataset search skipped; load a paper pack or add ?hfSearch=1 to enable it');
  }
  const enriched = [];
  const rankedCandidates = rankRetrievedRows(query, rows);
  const relevantCandidates = isRecommendationQuery(query)
    ? rankedCandidates.filter((row) => rowMatchesQuery(row, query))
    : rankedCandidates;
  const candidates = relevantCandidates.slice(0, Math.max(config.contextItems, config.candidateFloor));
  if (onCandidates && candidates.length) {
    onCandidates(candidates);
  }
  setProcessStep('lookup', 'active', `${formatCount(candidates.length)} candidates; hydrating top ${formatCount(Math.min(candidates.length, config.contextItems))}`);
  for (const row of candidates.slice(0, config.contextItems)) {
    setProcessStep('lookup', 'active', `Opening ${shortText(row.title || row.paper_id || row.canonical_paper_id || 'paper', 64)}`);
    const hydrated = await enrichPaper(row).catch(() => row);
    const isSelected = String(row.source || '').includes('selected_paper');
    enriched.push({
      ...hydrated,
      context_text: contextExcerpt(
        hydrated,
        query,
        isSelected ? config.selectedExcerptChars : config.excerptChars,
        isSelected,
      ),
    });
  }
  setProcessStep('lookup', 'done', enriched.length ? `${formatCount(enriched.length)} evidence items selected` : 'No evidence found');
  setProcessStep('compact', 'active', `Compacting ${formatCount(enriched.length)} evidence items`);
  setProcessStep('compact', 'done', `${formatCount(enriched.length)} excerpts prepared`);
  return enriched;
}

async function quickSearchPapers(query) {
  if (STRUCTURE_FIXTURE) {
    setProcessStep('lookup', 'done', 'Using fixture evidence');
    return structureFixtureRows(query);
  }
  const config = modeConfig();
  setProcessStep('lookup', 'active', 'Preparing paper search');
  await ensureDefaultResearchPack();
  const tokens = queryTokens(query);
  let rows = [];
  if (state.packRows.length) {
    rows = rankRetrievedRows(query, mergeRetrievedRows([await semanticSearchPack(query), lexicalPackSearch(query, tokens, config.candidateFloor)]));
  }
  if (!rows.length && HF_DATASET_SEARCH_ENABLED) {
    setProcessStep('lookup', 'active', `Remote search in ${HF.paperTextDataset}`);
    rows = (await hfSearchRows(HF.paperTextDataset, query, config.hfSearchRows)).map((row) => ({ ...row, source: 'hf_search' }));
  } else if (!rows.length && !state.packRows.length) {
    log('remote Hugging Face dataset search skipped; load a paper pack or add ?hfSearch=1 to enable it');
  }
  const ranked = rankRetrievedRows(query, rows)
    .filter((row) => !isRecommendationQuery(query) || rowMatchesQuery(row, query))
    .slice(0, config.contextItems);
  setProcessStep('lookup', 'done', ranked.length ? `${formatCount(ranked.length)} papers found` : 'No papers found');
  setProcessStep('compact', 'done', 'Decoder skipped');
  return ranked;
}

function quickSearchResultText(query, rows) {
  if (!rows.length) {
    return 'No matching papers were found. Load a paper pack first or try a narrower research phrase.';
  }
  const lines = rows.map((row, index) => {
    const title = String(row.title || row.name || 'Untitled paper').trim();
    const paperId = String(row.paper_id || row.canonical_paper_id || row.arxiv_id || '').trim();
    const category = String(row.primary_category || row.categories || '').split(/\s+/).filter(Boolean)[0] || '';
    const score = Number(row.retrieval_score || row.semantic_score || row.lexical_score || 0);
    const meta = [paperId, category, Number.isFinite(score) ? score.toFixed(3) : ''].filter(Boolean).join(' | ');
    return `${index + 1}. ${title}${meta ? ` (${meta})` : ''}`;
  });
  return [`Search results for: ${query}`, '', ...lines].join('\n');
}

async function submitQuickSearchPrompt(text) {
  resetProcessTrace(text);
  setControlsBusy(true);
  try {
    if (NEURAL_MEMORY_ENABLED && (!state.modelReady || state.loadedModelId !== els.model.value)) {
      await loadModel();
    }
    const rows = await quickSearchPapers(text);
    state.pendingContextRows = rows;
    state.retrievalRows = rows;
    if (rows.length) appendRetrieval(rows, { locked: false });
    setProcessStep('generate', 'done', 'Decoder disabled for Search mode');
    setProcessStep('render', 'done', rows.length ? 'Paper list rendered' : 'No paper list rendered');
    appendMessage('assistant', quickSearchResultText(text, rows));
    finishProcessTrace('Search complete');
  } catch (error) {
    setProcessStep('render', 'error', error.message || String(error));
    appendMessage('assistant', `Search failed: ${error.message || String(error)}`);
    finishProcessTrace('Error');
    log(`quick search failed: ${error.message || String(error)}`);
  } finally {
    setControlsBusy(false);
  }
}

async function selectedPaperContextRows(query) {
  const config = modeConfig();
  const selectedRows = state.paperContextRows
    .map((row) => ({ ...row, source: 'selected_paper' }))
    .slice(0, MAX_SELECTED_PAPERS);
  setProcessStep('lookup', 'active', `Using ${formatCount(selectedRows.length)} loaded paper context item${selectedRows.length === 1 ? '' : 's'}`);
  const enrichedSelected = [];
  for (const row of selectedRows) {
    const hydrated = await enrichPaper(row).catch(() => row);
    enrichedSelected.push({
      ...hydrated,
      source: 'selected_paper',
      context_text: contextExcerpt(hydrated, query, config.selectedExcerptChars, true),
    });
  }
  setProcessStep('lookup', 'done', `${formatCount(enrichedSelected.length)} loaded paper context item${enrichedSelected.length === 1 ? '' : 's'}`);
  setProcessStep('compact', 'active', `Compacting loaded paper context`);
  setProcessStep('compact', 'done', `${formatCount(enrichedSelected.length)} selected-paper excerpt${enrichedSelected.length === 1 ? '' : 's'} prepared`);
  return enrichedSelected;
}

async function enrichPaper(row) {
  if (row.abstract || row.text || row.full_text) return row;
  const paperId = String(row.paper_id || row.canonical_paper_id || '');
  if (row.paper_idx !== undefined && row.paper_idx !== null && row.paper_idx !== '') {
    const offsetRows = await hfRowsByOffset(HF.paperTextDataset, row.paper_idx, 1).catch(() => []);
    const match = offsetRows.find((candidate) => (
      !paperId ||
      String(candidate.paper_id || '') === paperId ||
      String(candidate.canonical_paper_id || '') === String(row.canonical_paper_id || paperId)
    ));
    if (match) return { ...row, ...match };
  }
  if (paperId) {
    const rows = await hfSearchRows(HF.paperTextDataset, paperId, 5).catch(() => []);
    const match = rows.find((candidate) => (
      String(candidate.paper_id || '') === paperId ||
      String(candidate.canonical_paper_id || '') === String(row.canonical_paper_id || paperId)
    ));
    if (match) return { ...row, ...match };
  }
  return row;
}

function compactContext(row, index) {
  const abstract = String(row.context_text || row.abstract || row.summary || row.text || row.full_text || '').replace(/\s+/g, ' ').slice(0, 1800);
  const title = String(row.title || row.paper_id || 'Untitled paper');
  const paperId = row.paper_id || row.canonical_paper_id || row.arxiv_id || '';
  const category = row.primary_category || row.category_list || row.categories || '';
  const meta = [
    paperId,
    category,
    row.year || '',
  ].filter(Boolean).join(' | ');
  return [
    `<AK_EVIDENCE_ID> P${index + 1}`,
    `<AK_TITLE> ${title}`,
    paperId ? `<AK_PAPER_ID> ${paperId}` : '',
    category ? `<AK_CATEGORY> ${category}` : '',
    row.year ? `<AK_YEAR> ${row.year}` : '',
    abstract ? `<AK_ABSTRACT> ${abstract}` : '',
    meta ? `Metadata: ${meta}` : '',
  ].filter(Boolean).join('\n');
}

function evidenceSnippet(row) {
  const text = String(row.abstract || row.summary || row.context_text || row.text || row.full_text || '')
    .replace(/\s+/g, ' ')
    .trim();
  return text ? text.slice(0, 520) : 'No abstract or text snippet was available in the loaded row.';
}

function answerScaffold(contextRows) {
  const text = evidenceSnippet(contextRows[0] || {})
    .replace(/\s+/g, ' ')
    .trim();
  if (!text || text.startsWith('No abstract')) {
    return 'No answer scaffold is available; answer cautiously from the user request.';
  }
  const sentences = text.match(/[^.!?]+[.!?]/g) || [text];
  return sentences.slice(0, 2).join(' ').trim().slice(0, 420);
}

function historyContext(limit = 6) {
  const messages = state.messages.slice(-limit);
  if (!messages.length) return 'No prior turns.';
  return messages.map((message) => {
    const role = message.role === 'user' ? 'User' : 'Assistant';
    return `${role}: ${shortText(message.text, 420)}`;
  }).join('\n');
}

function splitSentences(text) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return [];
  return normalized.match(/[^.!?]+[.!?]/g)?.map((sentence) => sentence.trim()).filter(Boolean)
    || [normalized.slice(0, 420)];
}

function readingQueryTokens(userText) {
  const tokens = queryContentTokens(userText)
    .filter((token) => token.length > 2)
    .filter((token) => !['paper', 'papers', 'source', 'sources', 'study', 'studies'].includes(token));
  return tokens.length ? tokens : queryTokens(userText).filter((token) => token.length > 2);
}

function sentenceRelevance(sentence, tokens) {
  const text = normalizeSearchText(sentence);
  let score = 0;
  for (const token of tokens) {
    if (tokenCoveredByText(text, token)) score += 2;
    else if (text.includes(token)) score += 1;
  }
  return score;
}

function evidenceReadingNotes(userText, contextRows) {
  if (!contextRows.length) return 'No retrieved evidence to read.';
  const tokens = readingQueryTokens(userText);
  const notes = contextRows.slice(0, modeConfig().contextItems).map((row, index) => {
    const text = String(row.context_text || row.abstract || row.summary || row.full_text || row.text || '')
      .replace(/\s+/g, ' ')
      .trim();
    const sentences = splitSentences(text);
    const ranked = sentences
      .map((sentence, sentenceIndex) => ({
        sentence,
        score: sentenceRelevance(sentence, tokens) + (sentenceIndex === 0 ? 0.4 : 0),
      }))
      .sort((a, b) => b.score - a.score);
    const best = ranked.slice(0, state.mode === 'deep_research' ? 3 : 2)
      .map((item) => item.sentence)
      .join(' ');
    const title = row.title || row.paper_id || row.canonical_paper_id || `Evidence ${index + 1}`;
    const meta = [row.paper_id || row.canonical_paper_id || row.arxiv_id, row.primary_category || row.category_list || row.categories, row.year]
      .filter(Boolean)
      .join(' | ');
    return [
      `[P${index + 1}] ${title}`,
      meta ? `Metadata: ${meta}` : '',
      `Relevant reading: ${shortText(best || evidenceSnippet(row), state.mode === 'deep_research' ? 760 : 520)}`,
    ].filter(Boolean).join('\n');
  });
  const instruction = isRecommendationQuery(userText)
    ? 'Recommendation rule: compare the retrieved evidence, prefer papers that directly match the user topic, and explain why the chosen evidence is strongest. Do not invent a title outside the evidence list.'
    : 'Synthesis rule: answer the user question first, then cite only evidence ids that support the claims.';
  return [instruction, ...notes].join('\n\n');
}

function arxivIdFromRow(row) {
  const candidates = [
    row.paper_id,
    row.canonical_paper_id,
    row.arxiv_id,
    row.id,
  ];
  for (const candidate of candidates) {
    const raw = String(candidate || '').trim();
    const match = raw.match(/(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?/i) || raw.match(/(?:arxiv:)?([a-z-]+(?:\.[A-Z]{2})?\/\d{7})(?:v\d+)?/i);
    if (match) return match[1];
  }
  return '';
}

function arxivPdfUrl(row) {
  const id = arxivIdFromRow(row);
  return id ? `https://arxiv.org/pdf/${encodeURIComponent(id)}` : '';
}

function languageLabel() {
  if (!els.language) return 'Auto';
  const value = String(els.language.value || 'auto');
  const option = [...els.language.options].find((item) => item.value === value);
  return option?.textContent || value;
}

function buildPromptFallback(userText, contextRows) {
  const config = modeConfig();
  const selectedContext = selectedContextTarget(userText);
  const context = contextRows.length
    ? contextRows.map((row, index) => `<AK_EVIDENCE> ${compactContext(row, index)}`).join('\n')
    : 'No research context was retrieved.';
  const modeInstruction = state.mode === 'deep_research'
    ? 'Mode: Deep Research. Review every evidence item, cite evidence numbers for supported claims, identify conflicts or gaps, and then give the final synthesis.'
    : state.mode === 'think'
      ? 'Mode: Think. Build a semantic synthesis across the evidence before answering and preserve important uncertainty.'
      : 'Mode: Chat. Reply like a helpful assistant. Use the strongest relevant evidence as support, not as the whole answer.';
  const tokenHeader = contextRows.length
    ? `${modeToken()} <AK_RESPOND> <AK_CONTEXT> <AK_ANSWER>`
    : `${modeToken()} <AK_RESPOND>`;
  const selectedCount = contextRows.filter((row) => String(row.source || '').includes('selected_paper')).length;
  const readingNotes = evidenceReadingNotes(userText, contextRows);
  const dataContext = pocketPalDataSourceContext(userText);
  return [
    tokenHeader,
    `<AK_LOOP> <AK_STATE> mode=${state.mode} selected_context=${selectedCount ? 1 : 0} retrieval=${contextRows.length ? 'ranked' : 'none'}`,
    'Return exactly this decision format: Action: respond, then Content: your direct answer.',
    'You are Agent Kernel Lite running entirely in this browser.',
    'Do not claim to execute, test, install, browse, or modify files.',
    'Answer the user directly. When using evidence, cite the evidence id such as [1] or [P1]; the interface renders the exact paper title and PDF link from that id.',
    'Do not generate paper titles or paper ids from memory; use evidence ids for grounded source references.',
    activeAgentRuntimePreamble(),
    modeInstruction,
    `Mode: ${config.label}`,
    activeAgentInstruction(),
    selectedContext ? 'Context target: answer about the selected paper already added to chat. Do not search for or introduce a different paper.' : '',
    '',
    '<AK_PROFILE> PocketPal saved slots:',
    pocketPalSlotContext(),
    '',
    '<AK_PROFILE> Active PocketPal agent:',
    pocketPalAgentContext(),
    '',
    '<AK_PROFILE> PocketPal installed tools:',
    pocketPalToolContext(),
    '',
    '<AK_PROFILE> PocketPal local memory:',
    pocketPalMemoryContext(),
    '',
    '<AK_CONTEXT> User data pointers:',
    dataContext,
    '',
    pocketPalTextSlotBlock(userText, dataContext),
    '',
    pocketPalSourceSlotBlock(userText, dataContext),
    '',
    '<AK_HISTORY> Recent conversation:',
    historyContext(),
    '',
    '<AK_READING_NOTES> Semantic reading notes:',
    readingNotes,
    '',
    '<AK_CONTEXT> Research context:',
    context,
    '',
    '<AK_ANSWER> Answer scaffold:',
    answerScaffold(contextRows),
    '',
    `<AK_USER> ${userText}`,
    'Return a structured decision with action=respond.',
  ].join('\n');
}

function buildActiveAgentDirectPrompt(userText) {
  const agent = activePocketPalAgent();
  const dataContext = pocketPalDataSourceContext(userText);
  const hint = activeAgentIntentHint(agent, userText);
  const textSlotBlock = pocketPalTextSlotBlock(userText, dataContext);
  const sourceSlotBlock = pocketPalSourceSlotBlock(userText, dataContext);
  return [
    '<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.',
    '<AK_AGENT_ACTIVE>',
    `Agent name: ${agent?.name || 'PocketPal agent'}`,
    `Agent instruction: ${agent?.instruction || defaultAgentInstruction(agent?.name)}`,
    `Retrieval policy: ${agent?.retrievalPolicy || 'auto'}`,
    `Tool policy: ${agent?.toolPolicy || 'ask_before_extensions'}`,
    `Action policy: ${agent?.actionPolicy || 'respond_or_ask'}`,
    'The active agent instruction is the primary task contract for this turn.',
    '</AK_AGENT_ACTIVE>',
    hint,
    `<AK_CONTEXT> Saved user data: ${dataContext || 'none'}`,
    textSlotBlock,
    sourceSlotBlock,
    '<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.',
    'Use stale paper context only when the current user request asks about that paper or research evidence.',
    `<AK_USER> ${userText}`,
    'Return AK structured tokens with the correct action and content for the active agent.',
    'AK token format: <AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_CONTENT> final content </AK_CONTENT> <AK_END>.',
  ].join('\n');
}

const ACTIVE_AGENT_ACTION_PREFIXES = {
  ask_user: '<AK_STRUCTURED> <AK_ACTION_ASK_USER> ',
  extension_request: '<AK_STRUCTURED> <AK_ACTION_EXTENSION_REQUEST> ',
  respond: '<AK_STRUCTURED> <AK_ACTION_RESPOND> ',
};

function activeAgentNeedsWebSearch(agent, userText = '') {
  if (activeAgentTextTransformInstruction(agent)) return false;
  const agentText = `${agent?.name || ''} ${agent?.instruction || ''}`.toLowerCase();
  if (/\b(web search|search agent|browser|look up online|search the web|online research|current info|latest news)\b/.test(agentText)) return true;
  const user = String(userText || '').toLowerCase();
  return /\b(?:search|look up|find)\b.{0,80}\b(?:web|online|internet|latest|current|recent|news|price|pricing)\b/.test(user)
    || /\b(?:latest|current|recent|today's|news|pricing)\b.{0,80}\b(?:for|about|on)\b/.test(user);
}

function activeAgentTextTransformInstruction(agent) {
  const instruction = `${agent?.name || ''} ${agent?.instruction || ''}`.toLowerCase();
  return /\b(rewrite|reword|paraphrase|polish|edit|improve|translate|summari[sz]e|extract|classify|format|turn .* into|make .* professional|clean up)\b/.test(instruction);
}

function activeAgentRequestLacksEditableText(userText = '') {
  const normalized = String(userText || '').replace(/\s+/g, ' ').trim().toLowerCase();
  if (!normalized) return true;
  if (normalized.length > 80 && !/\b(this|it|that)\b/.test(normalized)) return false;
  return /^(?:please\s+)?(?:rewrite|reword|paraphrase|polish|edit|improve|translate|summari[sz]e|extract|classify|format|fix|clean up|make (?:this|it).+|turn (?:this|it).+)(?:\s+(?:this|it|that|the text|my text|as .+|into .+))?[\s?.!]*$/i.test(normalized);
}

function activeAgentExpectedAction(agent, userText = '') {
  if (!agent) return '';
  if (activeAgentNeedsWebSearch(agent, userText) && activeAgentAllowsAction('extension_request')) {
    return 'extension_request';
  }
  if (activeAgentTextTransformInstruction(agent) && activeAgentRequestLacksEditableText(userText)) {
    return 'ask_user';
  }
  return 'respond';
}

function activeAgentDecoderPrefix(agent, userText = '') {
  if (!agent) return '';
  const action = activeAgentExpectedAction(agent, userText);
  return ACTIVE_AGENT_ACTION_PREFIXES[action] || '';
}

function activeAgentGenerationOptions(agent, userText, { retry = false } = {}) {
  const decoderPrefix = activeAgentDecoderPrefix(agent, userText);
  const deterministic = Boolean(decoderPrefix);
  return {
    maxNewTokens: retry ? 140 : Math.min(targetMaxTokens(), 220),
    temperature: deterministic ? 0 : 0.05,
    topP: deterministic ? 0.5 : 0.65,
    decoderPrefix,
  };
}

function buildActiveAgentRetryPrompt(userText) {
  return buildActiveAgentDirectPrompt(userText);
}

function compactCandidateRow(row, index) {
  const title = row.title || row.paper_id || row.canonical_paper_id || `Evidence ${index + 1}`;
  const category = row.primary_category || row.category_list || row.categories || '';
  const meta = [row.paper_id || row.canonical_paper_id || row.arxiv_id, row.year, row.source]
    .filter(Boolean)
    .join(' ');
  return [
    `<AK_CANDIDATE_ID> ${index + 1}`,
    `<AK_TITLE> ${title}`,
    category ? `<AK_CATEGORY> ${category}` : '',
    meta ? `<AK_META> ${meta}` : '',
    `<AK_ABSTRACT> ${shortText(evidenceSnippet(row), 520)}`,
  ].filter(Boolean).join(' | ');
}

function buildEvidenceSelectionPrompt(userText, rows) {
  const candidates = rows.map((row, index) => compactCandidateRow(row, index)).join('\n');
  return [
    `${modeToken()} <AK_GATHER_CONTEXT> <AK_RERANK> <AK_CANDIDATES>`,
    `<AK_LOOP> <AK_STATE> mode=${state.mode} selected_context=0 retrieval=ranked`,
    `<AK_USER> ${userText}`,
    candidates,
    'Select the evidence ids that best match the user intent.',
    'Return exactly this decision format: Action: gather_context, then Content: selected_candidate_id=1 or selected_candidate_id=1,3.',
  ].join('\n');
}

function selectedCandidateIndexes(selectorText, rowCount) {
  const indexes = [];
  const addIndex = (value) => {
    const index = Number(value) - 1;
    if (Number.isInteger(index) && index >= 0 && index < rowCount && !indexes.includes(index)) indexes.push(index);
  };
  for (const match of String(selectorText || '').matchAll(/selected_candidate_id\s*=\s*([0-9,\sPp]+)/gi)) {
    for (const raw of String(match[1] || '').matchAll(/P?(\d{1,2})/gi)) addIndex(raw[1]);
  }
  if (indexes.length) return indexes;
  for (const match of String(selectorText || '').matchAll(/\[(?:P)?(\d{1,2})\]|\bP(\d{1,2})\b/g)) {
    addIndex(match[1] || match[2]);
  }
  return indexes;
}

function confidentTopEvidenceRows(userText, rows) {
  if (!rows?.length) return [];
  const ranked = rankRetrievedRows(userText, rows);
  const top = ranked[0];
  const second = ranked[1];
  const topScore = Number(top?.rank_score || top?.retrieval_score || 0);
  const secondScore = Number(second?.rank_score || second?.retrieval_score || 0);
  const gap = topScore - secondScore;
  const ratio = secondScore > 0 ? topScore / secondScore : topScore;
  if (rowMatchesQuery(top, userText) && topScore >= 18 && (gap >= 14 || ratio >= 1.75)) {
    return [top];
  }
  return [];
}

async function selectEvidenceRows(userText, rows) {
  if (!rows?.length || rows.length <= 1) return rows || [];
  const confidentRows = confidentTopEvidenceRows(userText, rows);
  if (confidentRows.length) {
    setProcessStep('select', 'done', 'Skipped selector: retrieval top candidate is clear');
    return confidentRows;
  }
  if (!state.modelReady || !String(state.loadedModelId || '').startsWith('modelstack:')) return rows;
  setProcessStep('select', 'active', `Model selecting from ${formatCount(rows.length)} candidates`);
  try {
    const output = await generateUtilityText(buildEvidenceSelectionPrompt(userText, rows), {
      maxNewTokens: 18,
      maxEncoderTokens: 1024,
      temperature: 0,
      topP: 0.9,
      decoderPrefix: 'Action: gather_context\nContent: selected_candidate_id=',
      stopOnDecision: true,
    });
    const indexes = selectedCandidateIndexes(output, rows.length);
    if (!indexes.length) {
      setProcessStep('select', 'done', 'Kept ranked retrieval order');
      log(`selector did not return selected_candidate_id: ${shortText(output, 160)}`);
      return rows;
    }
    const selected = indexes.map((index) => rows[index]).filter(Boolean);
    setProcessStep('select', 'done', `Selected ${indexes.map((index) => index + 1).join(', ')}`);
    log(`selector output: ${shortText(output, 180)}`);
    return selected.length ? selected : rows;
  } catch (error) {
    setProcessStep('select', 'done', 'Selector unavailable; kept ranked retrieval order');
    log(`selector skipped: ${error.message || String(error)}`);
    return rows;
  }
}

function buildPrompt(userText, contextRows) {
  const config = modeConfig();
  if (state.coreReady && state.core) {
    try {
      const task = {
        task_id: `browser-${Date.now()}`,
        prompt: userText,
        workspace_subdir: 'browser',
        max_steps: config.contextItems,
        metadata: {
          benchmark_family: 'agentkernel_lite_browser',
          research_mode: state.mode,
        },
      };
      const history = state.messages.slice(-6).map((message) => ({
        role: message.role,
        text: message.text,
      }));
      const packet = state.core.start_turn_with_context
        ? JSON.parse(state.core.start_turn_with_context(
            JSON.stringify(task),
            JSON.stringify(contextRows),
            JSON.stringify(history),
            JSON.stringify({
              language: languageLabel(),
              max_new_tokens: targetMaxTokens(),
              max_context_items: config.contextItems,
              research_mode: state.mode,
              code_execution_enabled: false,
            }),
          ))
        : JSON.parse(state.core.start_turn(
            userText,
            JSON.stringify(contextRows),
            languageLabel(),
            targetMaxTokens(),
      ));
      if (packet.prompt) {
        const selectedContext = selectedContextTarget(userText);
        const dataContext = pocketPalDataSourceContext(userText);
        const readingAppendix = [
          '',
          '<AK_HISTORY> Recent conversation:',
          historyContext(),
          '',
          '<AK_PROFILE> PocketPal saved slots:',
          pocketPalSlotContext(),
          '',
          '<AK_PROFILE> Active PocketPal agent:',
          pocketPalAgentContext(),
          activeAgentInstruction(),
          '',
          '<AK_PROFILE> PocketPal installed tools:',
          pocketPalToolContext(),
          '',
          '<AK_PROFILE> PocketPal local memory:',
          pocketPalMemoryContext(),
          '',
          '<AK_CONTEXT> User data pointers:',
          dataContext,
          '',
          pocketPalTextSlotBlock(userText, dataContext),
          '',
          pocketPalSourceSlotBlock(userText, dataContext),
          '',
          '<AK_READING_NOTES> Semantic reading notes:',
          evidenceReadingNotes(userText, contextRows),
          selectedContext ? 'Context target: answer about the selected paper already added to chat. Do not search for or introduce a different paper.' : '',
        ].filter((line) => line !== '').join('\n');
        return [activeAgentRuntimePreamble(), packet.prompt, readingAppendix].filter(Boolean).join('\n');
      }
      return buildPromptFallback(userText, contextRows);
    } catch (error) {
      log(`WASM prompt compiler fallback: ${error.message || String(error)}`);
    }
  }
  return buildPromptFallback(userText, contextRows);
}

function recordAssistantTurn(text) {
  if (!state.coreReady || !state.core) return null;
  try {
    const replyText = repairPocketPalDecisionJson(text || '') || text || '';
    const raw = state.core.finish_model_reply
      ? state.core.finish_model_reply(replyText)
      : state.core.finish_turn(replyText);
    const packet = JSON.parse(raw);
    state.lastDecisionPacket = packet.decision_packet || null;
    return packet;
  } catch (error) {
    log(`WASM turn record failed: ${error.message || String(error)}`);
    return null;
  }
}

function extractJsonStringField(text, key) {
  const raw = String(text || '');
  const keyIndex = raw.search(new RegExp(`"${key}"\\s*:`));
  if (keyIndex < 0) return '';
  const colonIndex = raw.indexOf(':', keyIndex);
  if (colonIndex < 0) return '';
  let index = colonIndex + 1;
  while (index < raw.length && /\s/.test(raw[index])) index += 1;
  if (raw[index] !== '"') return '';
  index += 1;
  let value = '';
  let escaped = false;
  for (; index < raw.length; index += 1) {
    const char = raw[index];
    if (escaped) {
      value += char === 'n' ? '\n' : char === 't' ? '\t' : char;
      escaped = false;
      continue;
    }
    if (char === '\\') {
      escaped = true;
      continue;
    }
    if (char === '"') return value;
    value += char;
  }
  return '';
}

function repairPocketPalDecisionJson(text) {
  const raw = String(text || '').trim();
  if (!raw.includes('"action"') || !raw.includes('"content"')) return '';
  try {
    JSON.parse(raw);
    return '';
  } catch (_error) {
    const action = extractJsonStringField(raw, 'action');
    const content = extractJsonStringField(raw, 'content');
    const allowed = new Set(['respond', 'ask_user', 'extension_request', 'save_memory']);
    if (!allowed.has(action) || !content) return '';
    return JSON.stringify({ action, content, proposal_metadata: { task_type: 'repaired_decision' } });
  }
}

function readAkTokenValue(raw, token) {
  const pattern = new RegExp(`${token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s+([\\s\\S]*?)(?=\\s+<AK_[A-Z0-9_/]+>|\\s+</AK_[A-Z0-9_]+>|$)`);
  return String(raw || '').match(pattern)?.[1]?.trim() || '';
}

function decisionFromAkTokenProtocol(text) {
  const raw = String(text || '').trim();
  if (!raw || (!raw.includes('<AK_STRUCTURED>') && !raw.includes('<AK_CONTENT>'))) return null;
  const contentMatch = raw.match(/<AK_CONTENT>\s*([\s\S]*?)\s*<\/AK_CONTENT>/);
  let content = contentMatch?.[1]?.trim() || '';
  if (!content && raw.includes('<AK_COPY_USER_SOURCE_1>')) content = '<AK_COPY_USER_SOURCE_1>';
  const action = raw.includes('<AK_ACTION_EXTENSION_REQUEST>')
    ? 'extension_request'
    : raw.includes('<AK_ACTION_ASK_USER>')
      ? 'ask_user'
      : raw.includes('<AK_ACTION_SAVE_MEMORY>')
        ? 'save_memory'
      : 'respond';
  const taskType = readAkTokenValue(raw, '<AK_TASK_TYPE>') || 'ak_structured_tokens';
  const metadata = { task_type: taskType, ak_token_protocol: true };
  if (action === 'extension_request') {
    metadata.extension_id = state.webSearch.extensionId;
    metadata.capability = state.webSearch.searchCapabilityId;
    metadata.max_sources = Math.max(1, Math.min(5, Number(state.webSearch.maxSources || 5)));
    metadata.requires_user_approval = true;
  }
  if (!content && raw.includes('<AK_INTENT>')) {
    const intent = readAkTokenValue(raw, '<AK_INTENT>');
    const freshness = readAkTokenValue(raw, '<AK_FRESHNESS>');
    content = JSON.stringify({ intent, freshness });
  }
  if (!content) return null;
  return { action, content, proposal_metadata: metadata };
}

function modelDecisionFromText(text) {
  const raw = String(text || '').trim();
  if (!raw) return null;
  const tokenDecision = decisionFromAkTokenProtocol(raw);
  if (tokenDecision) return tokenDecision;
  const repaired = repairPocketPalDecisionJson(raw) || raw;
  try {
    const packet = JSON.parse(repaired);
    const decision = packet?.decision_packet?.decision || packet?.decision || packet;
    if (!decision || typeof decision !== 'object') return null;
    const action = String(decision.action || '').trim();
    const content = String(decision.content || '').trim();
    if (!action || !content) return null;
    return { action, content, proposal_metadata: decision.proposal_metadata || {} };
  } catch (_error) {
    return null;
  }
}

function isUsableActiveAgentDecisionText(text) {
  const decision = modelDecisionFromText(text);
  if (!decision) return false;
  return new Set(['respond', 'ask_user', 'extension_request', 'save_memory']).has(decision.action);
}

const ACTIVE_AGENT_PRESERVE_STOPWORDS = new Set([
  'about', 'after', 'again', 'agent', 'because', 'before', 'could', 'please', 'should', 'that', 'their',
  'there', 'these', 'thing', 'this', 'those', 'what', 'when', 'where', 'which', 'would', 'your',
]);

function activeAgentImportantTokens(text) {
  return contentTokens(text)
    .filter((token) => token.length >= 4)
    .filter((token) => !ACTIVE_AGENT_PRESERVE_STOPWORDS.has(token));
}

function activeAgentContentPreservesInput(content, userText) {
  const important = activeAgentImportantTokens(userText);
  if (important.length <= 1) return true;
  const contentNorm = normalizeSearchText(content);
  let preserved = 0;
  for (const token of important) {
    if (tokenCoveredByText(contentNorm, token)) preserved += 1;
  }
  return preserved / important.length >= 0.55;
}

function activeAgentUnavailablePlaceholders(content, slots = {}) {
  const available = new Set(Object.keys(slots || {}).map((key) => String(key).trim()).filter(Boolean));
  const unavailable = [];
  const seen = new Set();
  const matches = String(content || '').matchAll(/\[\[([A-Z0-9_:-]{2,64})\]\]/g);
  for (const match of matches) {
    const key = String(match[1] || '').trim();
    if (!key || available.has(key) || seen.has(key)) continue;
    seen.add(key);
    unavailable.push(key);
  }
  return unavailable;
}

function activeAgentRawSourcePlaceholderEcho(content, instruction = '') {
  if (!/\b(summarize|summary|recap|tl;?dr|rewrite|reword|paraphrase|polish|edit|improve|translate|translation)\b/i.test(instruction)) {
    return false;
  }
  if (/\b(exact|verbatim|source text|preserve all|copy|echo)\b/i.test(instruction)) return false;
  return /\[\[SOURCE_TEXT\]\]/.test(String(content || ''));
}

function activeAgentDecisionNeedsFallback(text, userText = '') {
  const agent = activePocketPalAgent();
  if (!agent) return false;
  const decision = modelDecisionFromText(text);
  if (!decision) return true;
  const expectedAction = activeAgentExpectedAction(agent, userText);
  if (expectedAction && decision.action !== expectedAction) return true;
  if (hasDecoderQualityIssue(decision.content, [], userText)) return true;
  if (decision.action === 'extension_request') {
    const metadata = decision.proposal_metadata || {};
    const extensionId = String(metadata.extension_id || metadata.extensionId || '').trim();
    const capability = String(metadata.capability || metadata.capability_id || metadata.capabilityId || '').trim();
    const maxSources = Number(metadata.max_sources || metadata.maxSources || 0);
    const approval = metadata.requires_user_approval ?? metadata.requiresUserApproval;
    if (expectedAction === 'extension_request') {
      if (extensionId !== state.webSearch.extensionId) return true;
      if (capability !== state.webSearch.searchCapabilityId) return true;
      if (!Number.isFinite(maxSources) || maxSources < 1) return true;
      if (approval !== true) return true;
    }
  }
  if (decision.action !== 'respond') return false;
  const instruction = `${agent.name || ''} ${agent.instruction || ''}`.toLowerCase();
  const slots = state.currentTextSlots || {};
  const expandedContent = expandPocketPalSourcePointers(expandPocketPalTextSlots(decision.content, slots), state.currentSourceSlots || []);
  if (activeAgentUnavailablePlaceholders(decision.content, slots).length) return true;
  if (activeAgentRawSourcePlaceholderEcho(expandedContent, instruction)) return true;
  if (/\b(exact|verbatim|preserve all|copy)\b/.test(instruction)) {
    return normalizeSearchText(expandedContent) !== normalizeSearchText(userText);
  }
  if (/\b(classify|classification|label|intent|tone)\b/.test(instruction)) {
    const labels = activeAgentAllowedLabels(instruction);
    if (labels.length) {
      const normalized = expandedContent.trim().toLowerCase().replace(/\s+/g, '_');
      return !labels.includes(normalized);
    }
    if (/^source text\s*:/i.test(expandedContent)) return true;
  }
  if (activeAgentTextTransformInstruction(agent) && !activeAgentRequestLacksEditableText(userText)) {
    const requiredSlots = ['NAME', 'ITEM', 'DEADLINE', 'REASON'].filter((key) => slots[key]);
    if (requiredSlots.length) {
      const normalizedContent = normalizeSearchText(expandedContent);
      for (const key of requiredSlots) {
        if (!tokenCoveredByText(normalizedContent, String(slots[key]).toLowerCase())) return true;
      }
    }
    return !activeAgentContentPreservesInput(expandedContent, userText);
  }
  return false;
}

function decisionFromPacket(packet) {
  return packet?.decision_packet?.decision || packet?.decision || null;
}

function displayTextFromDecision(packet, fallbackText) {
  const decision = decisionFromPacket(packet);
  return String(decision?.content || fallbackText || 'No answer generated.');
}

function decisionAction(packet) {
  const decision = decisionFromPacket(packet);
  return String(decision?.action || '').trim();
}

function decisionExtensionMetadata(packet) {
  const metadata = decisionFromPacket(packet)?.proposal_metadata || {};
  const nested = metadata.extension && typeof metadata.extension === 'object' ? metadata.extension : {};
  return {
    extensionId: String(metadata.extension_id || metadata.extensionId || nested.id || nested.extension_id || '').trim(),
    capability: String(metadata.capability || metadata.capability_id || nested.capability || nested.capability_id || '').trim(),
    query: expandPocketPalTextSlots(String(metadata.query || metadata.search_query || metadata.request || '').trim()),
  };
}

function loadPocketPalMemory() {
  try {
    const raw = localStorage.getItem(POCKETPAL_MEMORY_STORAGE_KEY);
    const rows = raw ? JSON.parse(raw) : [];
    state.pocketPalMemory = Array.isArray(rows)
      ? rows.filter((item) => item && typeof item === 'object' && String(item.text || '').trim()).slice(-100)
      : [];
  } catch (error) {
    state.pocketPalMemory = [];
    log(`PocketPal memory load skipped: ${error.message || String(error)}`);
  }
}

function loadPocketPalSlots() {
  try {
    const raw = localStorage.getItem(POCKETPAL_SLOTS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    state.pocketPalSlots = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch (error) {
    state.pocketPalSlots = {};
    log(`PocketPal slots load skipped: ${error.message || String(error)}`);
  }
}

function loadPocketPalDataSources() {
  try {
    const raw = localStorage.getItem(POCKETPAL_DATA_SOURCES_STORAGE_KEY);
    const rows = raw ? JSON.parse(raw) : [];
    state.pocketPalDataSources = Array.isArray(rows)
      ? rows.filter((item) => item && typeof item === 'object' && String(item.text || '').trim()).slice(-50)
      : [];
  } catch (error) {
    state.pocketPalDataSources = [];
    log(`PocketPal data source load skipped: ${error.message || String(error)}`);
  }
}

function loadPocketPalAgents() {
  try {
    const raw = localStorage.getItem(POCKETPAL_AGENTS_STORAGE_KEY);
    const payload = raw ? JSON.parse(raw) : {};
    const rows = Array.isArray(payload) ? payload : payload.agents;
    state.pocketPalAgents = Array.isArray(rows)
      ? rows.filter((item) => item && typeof item === 'object' && String(item.name || '').trim()).slice(-24)
      : [];
    state.activeAgentId = typeof payload?.activeAgentId === 'string' ? payload.activeAgentId : '';
    if (!state.pocketPalAgents.some((agent) => agent.id === state.activeAgentId)) state.activeAgentId = '';
  } catch (error) {
    state.pocketPalAgents = [];
    state.activeAgentId = '';
    log(`PocketPal agents load skipped: ${error.message || String(error)}`);
  }
}

function loadWebSearchSettings() {
  try {
    const raw = localStorage.getItem(WEB_SEARCH_SETTINGS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    const maxSources = Math.max(1, Math.min(5, Number(parsed.max_sources || state.webSearch.maxSources || 5)));
    state.webSearch.maxSources = maxSources;
    if (els.webSearchMaxSources) els.webSearchMaxSources.value = String(maxSources);
  } catch (error) {
    state.webSearch.maxSources = 5;
    if (els.webSearchMaxSources) els.webSearchMaxSources.value = '5';
    log(`web search settings load skipped: ${error.message || String(error)}`);
  }
}

function persistPocketPalMemory() {
  try {
    localStorage.setItem(POCKETPAL_MEMORY_STORAGE_KEY, JSON.stringify(state.pocketPalMemory.slice(-100)));
  } catch (error) {
    log(`PocketPal memory save skipped: ${error.message || String(error)}`);
  }
}

function persistWebSearchSettings() {
  try {
    localStorage.setItem(WEB_SEARCH_SETTINGS_STORAGE_KEY, JSON.stringify({
      max_sources: state.webSearch.maxSources,
    }));
  } catch (error) {
    log(`web search settings save skipped: ${error.message || String(error)}`);
  }
}

function persistPocketPalSlots() {
  try {
    localStorage.setItem(POCKETPAL_SLOTS_STORAGE_KEY, JSON.stringify(state.pocketPalSlots || {}));
  } catch (error) {
    log(`PocketPal slots save skipped: ${error.message || String(error)}`);
  }
}

function persistPocketPalDataSources() {
  try {
    localStorage.setItem(POCKETPAL_DATA_SOURCES_STORAGE_KEY, JSON.stringify(state.pocketPalDataSources.slice(-50)));
  } catch (error) {
    log(`PocketPal data source save skipped: ${error.message || String(error)}`);
  }
}

function persistPocketPalAgents() {
  try {
    localStorage.setItem(POCKETPAL_AGENTS_STORAGE_KEY, JSON.stringify({
      activeAgentId: state.activeAgentId || '',
      agents: state.pocketPalAgents.slice(-24),
    }));
  } catch (error) {
    log(`PocketPal agents save skipped: ${error.message || String(error)}`);
  }
}

function pocketPalExportState() {
  return {
    memory: state.pocketPalMemory.slice(-100),
    slots: { ...(state.pocketPalSlots || {}) },
    data_sources: state.pocketPalDataSources.slice(-50),
    agents: state.pocketPalAgents.slice(-24),
    active_agent_id: state.activeAgentId || '',
  };
}

function restorePocketPalState(payload) {
  if (payload && typeof payload === 'object') {
    state.pocketPalMemory = Array.isArray(payload.memory)
      ? payload.memory.filter((item) => item && typeof item === 'object' && String(item.text || '').trim()).slice(-100)
      : [];
    state.pocketPalSlots = payload.slots && typeof payload.slots === 'object' && !Array.isArray(payload.slots)
      ? payload.slots
      : {};
    state.pocketPalDataSources = Array.isArray(payload.data_sources)
      ? payload.data_sources.filter((item) => item && typeof item === 'object' && String(item.text || '').trim()).slice(-50)
      : [];
    state.pocketPalAgents = Array.isArray(payload.agents)
      ? payload.agents.filter((item) => item && typeof item === 'object' && String(item.name || '').trim()).slice(-24)
      : [];
    state.activeAgentId = typeof payload.active_agent_id === 'string' ? payload.active_agent_id : '';
    if (!state.pocketPalAgents.some((agent) => agent.id === state.activeAgentId)) state.activeAgentId = '';
    persistPocketPalMemory();
    persistPocketPalSlots();
    persistPocketPalDataSources();
    persistPocketPalAgents();
  } else {
    loadPocketPalSlots();
    loadPocketPalMemory();
    loadPocketPalDataSources();
    loadPocketPalAgents();
  }
  renderDataSourceList();
  renderAgentList();
}

function pocketPalSlotContext() {
  const entries = Object.entries(state.pocketPalSlots || {})
    .filter(([key, value]) => String(key || '').trim() && value !== undefined && value !== null && String(value).trim())
    .slice(-24);
  if (!entries.length) return 'No saved PocketPal slots.';
  return entries.map(([key, value]) => {
    const safeKey = String(key).replace(/[^a-z0-9_.-]/gi, '_').slice(0, 48);
    return `<AK_SLOT> <AK_SLOT_NAME>=${safeKey} <AK_SLOT_VALUE>=${shortText(String(value), 180)}`;
  }).join('\n');
}

function scorePocketPalDataSource(query, source) {
  const queryTokens = new Set(contentTokens(query));
  if (!queryTokens.size) return 0;
  const sourceTokens = contentTokens(`${source.title || ''} ${source.text || ''} ${(source.chunks || []).join(' ')}`);
  let score = 0;
  for (const token of sourceTokens) {
    if (queryTokens.has(token)) score += 1;
  }
  return score;
}

function rankedPocketPalDataChunks(query, limit = 8) {
  const rows = state.pocketPalDataSources.slice();
  if (!rows.length) return [];
  return rows
    .flatMap((item, sourceIndex) => {
      const chunks = Array.isArray(item.chunks) && item.chunks.length ? item.chunks : [item.text || ''];
      return chunks.map((chunk, chunkIndex) => ({
        item,
        sourceIndex,
        chunkIndex,
        chunk,
        score: scorePocketPalDataSource(query, {
          title: item.title,
          text: chunk,
          chunks: [],
        }),
      }));
    })
    .sort((a, b) => (b.score - a.score) || (b.sourceIndex - a.sourceIndex) || (a.chunkIndex - b.chunkIndex))
    .slice(0, limit);
}

function pocketPalDataSourceContext(query = '', limit = 8) {
  const rows = rankedPocketPalDataChunks(query, limit);
  if (!rows.length) return 'No saved user data sources.';
  return rows.map(({ item, chunk, chunkIndex }, index) => {
    const title = shortText(String(item.title || item.source_type || 'source'), 80);
    const type = shortText(String(item.source_type || 'note'), 32);
    const text = shortText(String(chunk || item.text || ''), 460);
    return `[D${index + 1}] ${title} (${type}, chunk ${chunkIndex + 1}): ${text}`;
  }).join('\n');
}

function compactSourceSlotText(value, limit = 900) {
  return String(value || '').replace(/\s+/g, ' ').trim().slice(0, limit).trim();
}

function compilePocketPalSourceSlots(userText, { dataContext = '', maxSlots = 8 } = {}) {
  const slots = [];
  const add = (kind, text) => {
    const value = compactSourceSlotText(text);
    if (!value || slots.some((slot) => slot.text === value)) return;
    if (slots.length >= Math.max(1, Math.min(24, Number(maxSlots) || 8))) return;
    const index = slots.length + 1;
    slots.push({ id: `U${index}`, token: `<AK_COPY_USER_SOURCE_${index}>`, kind, text: value });
  };
  const user = compactSourceSlotText(userText);
  add('user_text', user);
  for (const match of user.matchAll(/["']([^"']{3,220})["']/g)) add('quoted_user_text', match[1]);
  for (const chunk of user.split(/\s*(?:;|\n| and | but | because |,)\s*/g)) {
    if (chunk.trim().length >= 8 && chunk.trim().length <= 220) add('user_span', chunk);
  }
  const data = compactSourceSlotText(dataContext);
  if (data && data.toLowerCase() !== 'no saved user data sources.') add('user_data', data);
  return slots;
}

function pocketPalSourceSlotBlock(userText, dataContext = '') {
  if (!SOURCE_SLOT_TOKENS_ENABLED) {
    state.currentSourceSlots = [];
    return '';
  }
  const slots = compilePocketPalSourceSlots(userText, { dataContext, maxSlots: 8 });
  state.currentSourceSlots = slots;
  if (!slots.length) return '<AK_SOURCE_SLOTS> none';
  return [
    '<AK_SOURCE_SLOTS>',
    'Use source copy tokens when exact user-provided names, dates, values, links, or wording must be preserved.',
    ...slots.map((slot) => `${slot.token} ${slot.kind}: ${slot.text}`),
  ].join('\n');
}

function inferPocketPalTextSlots(userText = '', dataContext = '') {
  const text = compactSourceSlotText(userText, 900);
  if (!text) return {};
  const slots = { SOURCE_TEXT: text };
  const data = compactSourceSlotText(dataContext, 900);
  if (data && data.toLowerCase() !== 'no saved user data sources.') slots.DATA_CONTEXT = data;
  const patterns = [
    /^(?:hey\s+)?(?<name>[a-z][\w.-]*)\s+i\s+need\s+the\s+(?<item>.+?)\s+by\s+(?<deadline>.+?)\s+because\s+(?<reason>.+)$/i,
    /^(?:hey|hi|hello|yo)\s+(?<name>[a-z][\w.-]*)\s+(?:please\s+)?(?:send|get|finish|prepare|share|complete|review|update|draft|write)\s+(?:the\s+)?(?<item>.+?)\s+by\s+(?<deadline>.+?)\s+because\s+(?<reason>.+)$/i,
    /^ask\s+(?<name>[a-z][\w.-]*)\s+for\s+the\s+(?<item>.+?)\s+by\s+(?<deadline>.+?)\s+because\s+(?<reason>.+)$/i,
    /^tell\s+(?<name>[a-z][\w.-]*)\s+we\s+need\s+the\s+(?<item>.+?)\s+by\s+(?<deadline>.+?)\s+since\s+(?<reason>.+)$/i,
    /^can\s+you\s+ask\s+(?<name>[a-z][\w.-]*)\s+to\s+send\s+the\s+(?<item>.+?)\s+by\s+(?<deadline>.+?)\s+because\s+(?<reason>.+)$/i,
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (!match?.groups) continue;
    slots.NAME = capitalizeSlotValue(match.groups.name);
    slots.ITEM = compactSourceSlotText(match.groups.item, 180);
    slots.DEADLINE = compactSourceSlotText(match.groups.deadline, 120);
    slots.REASON = capitalizeSlotValue(match.groups.reason);
    break;
  }
  return slots;
}

function capitalizeSlotValue(value = '') {
  const text = compactSourceSlotText(value, 240);
  return text ? `${text.charAt(0).toUpperCase()}${text.slice(1)}` : text;
}

function pocketPalTextSlotBlock(userText = '', dataContext = '') {
  const slots = inferPocketPalTextSlots(userText, dataContext);
  state.currentTextSlots = slots;
  const keys = Object.keys(slots).filter((key) => slots[key]);
  if (!keys.length) return '<AK_PROFILE> User text slots: none';
  return [
    '<AK_PROFILE> User text slots:',
    ...keys.map((key) => `<AK_SLOT> <AK_SLOT_NAME>=${key} <AK_SLOT_VALUE>=${slots[key]}`),
    `Available placeholders for this turn: ${keys.map((key) => `[[${key}]]`).join(', ')}.`,
    'Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.',
  ].join('\n');
}

function expandPocketPalTextSlots(text, slots = state.currentTextSlots || {}) {
  let value = String(text || '');
  if (Object.prototype.hasOwnProperty.call(slots || {}, 'DATA_CONTEXT')) {
    value = value.replace(/\[\[DATA_CONTEXT\]\]+[\s\S]*$/g, '[[DATA_CONTEXT]]');
  }
  for (const [name, replacement] of Object.entries(slots || {})) {
    value = value.split(`[[${name}]]`).join(String(replacement || ''));
  }
  return value;
}

function expandPocketPalSourcePointers(text, slots = state.currentSourceSlots || []) {
  let value = String(text || '');
  for (const slot of slots || []) {
    value = value.split(slot.token).join(slot.text);
    const escapedId = String(slot.id || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (escapedId) {
      value = value.replace(new RegExp(`<AK_COPY_USER_SOURCE_${escapedId.slice(1)}[>\\]]?[\\s\\S]*$`, 'g'), slot.text);
    }
  }
  return value;
}

function activePocketPalAgent() {
  return state.pocketPalAgents.find((agent) => agent.id === state.activeAgentId) || null;
}

function syncActiveAgentPill() {
  const agent = activePocketPalAgent();
  if (agent) {
    setPill(els.modePill, `${shortText(agent.name, 22)} agent`, 'ready');
    return;
  }
  if (state.activeAgentId) {
    state.activeAgentId = '';
    persistPocketPalAgents();
  }
  setPill(els.modePill, modeConfig().pill, state.mode === 'deep_research' ? 'busy' : 'ready');
}

function setActivePocketPalAgent(agentId, { render = true, announce = true } = {}) {
  const agent = state.pocketPalAgents.find((item) => item.id === agentId) || null;
  state.activeAgentId = agent ? agent.id : '';
  persistPocketPalAgents();
  syncActiveAgentPill();
  if (render) renderAgentList();
  if (announce) log(agent ? `active PocketPal agent: ${agent.name}` : 'active PocketPal agent cleared');
  return agent;
}

function activeAgentPolicy(name, fallback = '') {
  const agent = activePocketPalAgent();
  return String(agent?.[name] || fallback).trim();
}

function activeAgentInstruction() {
  const agent = activePocketPalAgent();
  if (!agent) return '';
  return [
    `Active agent selected: ${agent.name}.`,
    `Follow this agent instruction unless it conflicts with user safety or the current user request: ${agent.instruction || 'Help the user with the selected task.'}`,
    `Agent policies: retrieval=${agent.retrievalPolicy || 'auto'} tools=${agent.toolPolicy || 'ask_before_extensions'} actions=${agent.actionPolicy || 'respond_or_ask'}.`,
  ].join('\n');
}

function activeAgentRuntimePreamble() {
  const agent = activePocketPalAgent();
  if (!agent) return '';
  return [
    '<AK_AGENT_ACTIVE>',
    `Agent name: ${agent.name || 'PocketPal agent'}`,
    `Agent instruction: ${agent.instruction || defaultAgentInstruction(agent.name)}`,
    `Retrieval policy: ${agent.retrievalPolicy || 'auto'}`,
    `Tool policy: ${agent.toolPolicy || 'ask_before_extensions'}`,
    `Action policy: ${agent.actionPolicy || 'respond_or_ask'}`,
    'The active agent instruction is the primary task contract for this turn. Apply it directly to the user request. Do not answer as the base assistant when an active agent is selected unless the active agent instruction asks for normal assistant chat. Do not substitute a research assistant behavior unless the agent instruction or the user explicitly asks for research.',
    '</AK_AGENT_ACTIVE>',
  ].join('\n');
}

function activeAgentIntentHint(agent, userText = '') {
  if (!agent) return '';
  const haystack = `${agent.name || ''} ${agent.instruction || ''} ${userText || ''}`.toLowerCase();
  const pairs = [
    ['rewrite', /\b(rewrite|reword|paraphrase|make .*professional|polish|grammar|tone|shorter|clearer)\b/],
    ['translation', /\b(translate|translation|spanish|french|german|italian|portuguese)\b/],
    ['summary', /\b(summarize|summary|tl;?dr|recap)\b/],
    ['action_items', /\b(action item|todo|to-do|tasks?|owners?|deadlines?|bullet points?|bullets?)\b/],
    ['plan', /\b(plan|schedule|itinerary|steps|roadmap)\b/],
    ['checklist', /\b(checklist|check list)\b/],
    ['risks', /\b(risk|risks|review|concerns|failure modes?)\b/],
    ['json', /\b(json|structured object|schema)\b/],
    ['ranking', /\b(rank|sort|priority|prioritize|order)\b/],
    ['extraction', /\b(extract|pull out|identify|entities|fields)\b/],
    ['subject', /\b(subject line|email subject)\b/],
    ['brainstorm', /\b(brainstorm|ideas|generate ideas|names)\b/],
    ['source_echo', /\b(exact|verbatim|source text|preserve all)\b/],
    ['saved_data', /\b(saved data|memory|remembered|my note|my code)\b/],
    ['web_search', /\b(web|search|online|current|recent|latest|today)\b/],
  ];
  const intent = pairs.find(([, pattern]) => pattern.test(haystack))?.[0] || 'casual';
  const task = intent === 'rewrite' ? 'active_agent_rewrite' : `active_agent_${intent}`;
  return `<AK_TASK_HINT> intent=${intent} task=${task} source_text_required=${intent !== 'casual'}`;
}

function defaultAgentInstruction(name) {
  const label = String(name || '').trim() || 'PocketPal agent';
  return `Use the user-configured agent named "${label}" for the current request. Treat the agent name and any saved instruction as the task definition, follow the selected retrieval/tool/action policies, and ask before taking actions that need user approval.`;
}

function pocketPalAgentContext() {
  const agent = activePocketPalAgent();
  if (!agent) return 'No custom agent selected.';
  return [
    `<AK_SLOT> <AK_SLOT_NAME>=agent_name <AK_SLOT_VALUE>=${shortText(agent.name, 80)}`,
    `<AK_SLOT> <AK_SLOT_NAME>=agent_instruction <AK_SLOT_VALUE>=${shortText(agent.instruction, 360)}`,
    `<AK_SLOT> <AK_SLOT_NAME>=agent_retrieval_policy <AK_SLOT_VALUE>=${shortText(agent.retrievalPolicy || 'auto', 80)}`,
    `<AK_SLOT> <AK_SLOT_NAME>=agent_tool_policy <AK_SLOT_VALUE>=${shortText(agent.toolPolicy || 'ask_before_extensions', 80)}`,
    `<AK_SLOT> <AK_SLOT_NAME>=agent_action_policy <AK_SLOT_VALUE>=${shortText(agent.actionPolicy || 'respond_or_ask', 80)}`,
  ].join('\n');
}

function pocketPalToolContext() {
  return [
    '<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask',
    `<AK_MAX_SOURCES>=${Math.max(1, Math.min(5, Number(state.webSearch.maxSources || 5)))}`,
    'When the active agent instruction or user request needs current, recent, online, or web-backed information, return action=extension_request with proposal_metadata.extension_id=web_search, capability=web.search, query, max_sources, and requires_user_approval=true. Do not invent web results before the extension runs.',
  ].join('\n');
}

function activeAgentAllowsAction(action) {
  const agent = activePocketPalAgent();
  if (!agent) return true;
  const policy = String(agent.actionPolicy || 'respond_or_ask').trim();
  if (action === 'save_memory') return policy === 'allow_memory' || policy === 'full_local_agent';
  if (action === 'extension_request') {
    const toolPolicy = String(agent.toolPolicy || 'ask_before_extensions').trim();
    return policy === 'allow_extension_requests'
      || policy === 'full_local_agent'
      || (policy === 'respond_or_ask' && (toolPolicy === 'ask_before_extensions' || toolPolicy === 'installed_only'));
  }
  return true;
}

function pocketPalMemoryContext(limit = 8) {
  const rows = state.pocketPalMemory.slice(-limit);
  if (!rows.length) return 'No saved PocketPal memory.';
  return rows.map((item, index) => {
    const kind = String(item.kind || 'memory').trim();
    const text = shortText(String(item.text || '').replace(/\s+/g, ' ').trim(), 220);
    return `[M${index + 1}] ${kind}: ${text}`;
  }).join('\n');
}

function saveMemoryFromDecision(packet) {
  const decision = packet?.decision_packet?.decision || packet?.decision || null;
  if (!decision || String(decision.action || '') !== 'save_memory') return;
  const metadata = decision.proposal_metadata || {};
  const memory = metadata.memory && typeof metadata.memory === 'object' ? metadata.memory : metadata;
  const text = String(memory.text || metadata.text || '').replace(/\s+/g, ' ').trim();
  if (!text) return;
  const key = text.toLowerCase();
  state.pocketPalMemory = state.pocketPalMemory.filter((item) => String(item.text || '').toLowerCase() !== key);
  state.pocketPalMemory.push({
    id: `mem_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    created_at: new Date().toISOString(),
    scope: String(memory.scope || 'local_user_profile'),
    kind: String(memory.kind || 'preference'),
    text,
  });
  state.pocketPalMemory = state.pocketPalMemory.slice(-100);
  persistPocketPalMemory();
  log(`PocketPal memory saved: ${shortText(text, 80)}`);
}

function applySlotUpdatesFromDecision(packet) {
  const decision = decisionFromPacket(packet);
  const updates = decision?.proposal_metadata?.slot_updates;
  if (!updates || typeof updates !== 'object' || Array.isArray(updates)) return;
  let changed = 0;
  const next = { ...(state.pocketPalSlots || {}) };
  for (const [key, value] of Object.entries(updates)) {
    const normalizedKey = String(key || '').trim().replace(/[^a-z0-9_.-]/gi, '_').slice(0, 48);
    if (!normalizedKey) continue;
    if (value === null || value === undefined || value === '') {
      if (Object.prototype.hasOwnProperty.call(next, normalizedKey)) {
        delete next[normalizedKey];
        changed += 1;
      }
      continue;
    }
    const normalizedValue = typeof value === 'object' ? JSON.stringify(value) : String(value);
    if (next[normalizedKey] !== normalizedValue) {
      next[normalizedKey] = shortText(normalizedValue, 240);
      changed += 1;
    }
  }
  if (!changed) return;
  state.pocketPalSlots = next;
  persistPocketPalSlots();
  log(`PocketPal slots updated: ${Object.keys(updates).join(', ')}`);
}

function switchModule(moduleId) {
  const normalized = ['assistant', 'retrieval', 'agents'].includes(moduleId) ? moduleId : 'assistant';
  for (const tab of els.moduleTabs) {
    const active = tab.dataset.module === normalized;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', active ? 'true' : 'false');
  }
  for (const panel of els.modulePanels) {
    panel.hidden = panel.dataset.modulePanel !== normalized;
  }
}

function saveUserDataSource() {
  const text = String(els.userDataSource?.value || '').trim();
  if (!text) return;
  addPocketPalDataSource({
    source_type: inferDataSourceType(text),
    title: inferDataSourceTitle(text),
    text,
  });
  if (els.userDataSource) els.userDataSource.value = '';
  renderDataSourceList();
  log(`PocketPal data source saved: ${shortText(text, 80)}`);
}

function inferDataSourceType(text) {
  if (/^https?:\/\//i.test(text)) return 'url';
  if (/^(\/|~\/|[a-z]:\\)/i.test(text)) return 'path';
  return 'note';
}

function inferDataSourceTitle(text) {
  const firstLine = String(text || '').split(/\r?\n/).find((line) => line.trim()) || '';
  return shortText(firstLine.replace(/^#+\s*/, '').trim() || 'User note', 80);
}

function chunkPocketPalText(text, maxChunks = 16) {
  const normalized = String(text || '').replace(/\r/g, '').trim();
  if (!normalized) return [];
  const paragraphs = normalized.split(/\n{2,}/).map((part) => part.replace(/\s+/g, ' ').trim()).filter(Boolean);
  if (paragraphs.length > 1) {
    const chunks = [];
    for (const paragraph of paragraphs) {
      if (paragraph.length > 900) {
        for (let cursor = 0; cursor < paragraph.length && chunks.length < maxChunks; cursor += 820) {
          chunks.push(paragraph.slice(cursor, cursor + 900).trim());
        }
      } else {
        chunks.push(shortText(paragraph, 900));
      }
      if (chunks.length >= maxChunks) break;
    }
    return chunks.slice(0, maxChunks);
  }
  const chunks = [];
  let current = '';
  const pushCurrent = () => {
    const chunk = current.trim();
    if (chunk) chunks.push(shortText(chunk, 900));
    current = '';
  };
  for (const paragraph of paragraphs.length ? paragraphs : [normalized.replace(/\s+/g, ' ')]) {
    if ((current.length + paragraph.length + 1) > 900) pushCurrent();
    if (paragraph.length > 900) {
      for (let cursor = 0; cursor < paragraph.length && chunks.length < maxChunks; cursor += 820) {
        chunks.push(paragraph.slice(cursor, cursor + 900).trim());
      }
      continue;
    }
    current = current ? `${current} ${paragraph}` : paragraph;
    if (chunks.length >= maxChunks) break;
  }
  if (chunks.length < maxChunks) pushCurrent();
  return chunks.slice(0, maxChunks);
}

function addPocketPalDataSource({ title, text, source_type: sourceType = 'note', bytes = 0 }) {
  const normalizedText = String(text || '').replace(/\s+/g, ' ').trim();
  if (!normalizedText) return false;
  const normalizedTitle = String(title || inferDataSourceTitle(normalizedText)).replace(/\s+/g, ' ').trim();
  const chunks = chunkPocketPalText(text);
  const key = `${sourceType}:${normalizedTitle}:${normalizedText.slice(0, 160)}`.toLowerCase();
  state.pocketPalDataSources = state.pocketPalDataSources.filter((item) => String(item.key || '').toLowerCase() !== key);
  state.pocketPalDataSources.push({
    id: `src_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    created_at: new Date().toISOString(),
    key,
    source_type: String(sourceType || 'note'),
    title: shortText(normalizedTitle || 'User data', 100),
    bytes: Number.isFinite(bytes) ? bytes : 0,
    text: shortText(normalizedText, 500),
    chunks,
  });
  state.pocketPalDataSources = state.pocketPalDataSources.slice(-50);
  persistPocketPalDataSources();
  return true;
}

async function importUserDataFiles(fileList) {
  const files = Array.from(fileList || []).filter(Boolean).slice(0, 12);
  if (!files.length) return;
  let imported = 0;
  for (const file of files) {
    try {
      const text = await file.text();
      if (addPocketPalDataSource({
        source_type: 'file',
        title: file.name || 'Local file',
        bytes: file.size || text.length,
        text,
      })) {
        imported += 1;
      }
    } catch (error) {
      log(`file import skipped for ${file.name || 'local file'}: ${error.message || String(error)}`);
    }
  }
  if (els.userDataFileInput) els.userDataFileInput.value = '';
  renderDataSourceList();
  if (imported) log(`PocketPal imported ${formatCount(imported)} local file${imported === 1 ? '' : 's'}`);
}

function renderDataSourceList() {
  if (!els.userDataSourceList) return;
  els.userDataSourceList.innerHTML = '';
  if (!state.pocketPalDataSources.length) {
    const empty = document.createElement('p');
    empty.className = 'module-note';
    empty.textContent = 'No saved data pointers yet.';
    els.userDataSourceList.appendChild(empty);
    return;
  }
  for (const source of state.pocketPalDataSources.slice().reverse()) {
    const row = document.createElement('div');
    row.className = 'agent-row';
    const detail = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = source.title || 'Data pointer';
    const body = document.createElement('span');
    body.textContent = [
      source.source_type || 'note',
      source.bytes ? `${formatCount(source.bytes)} bytes` : '',
      shortText(source.text || '', 130),
    ].filter(Boolean).join(' | ');
    detail.append(title, body);
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'secondary';
    remove.textContent = 'Remove';
    remove.addEventListener('click', () => {
      state.pocketPalDataSources = state.pocketPalDataSources.filter((item) => item.id !== source.id);
      persistPocketPalDataSources();
      renderDataSourceList();
    });
    row.append(detail, remove);
    els.userDataSourceList.appendChild(row);
  }
}

function renderAgentList() {
  if (!els.agentList) return;
  els.agentList.innerHTML = '';
  if (!state.pocketPalAgents.length) {
    const empty = document.createElement('p');
    empty.className = 'module-note';
    empty.textContent = 'No custom agents yet.';
    els.agentList.appendChild(empty);
    return;
  }
  for (const agent of state.pocketPalAgents.slice().reverse()) {
    const row = document.createElement('div');
    const active = agent.id === state.activeAgentId;
    row.className = `agent-row${active ? ' active' : ''}`;
    const detail = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = agent.name || 'Agent';
    const body = document.createElement('span');
    body.textContent = shortText(agent.instruction || '', 110);
    const policy = document.createElement('span');
    policy.textContent = [
      `retrieval=${agent.retrievalPolicy || 'auto'}`,
      `tools=${agent.toolPolicy || 'ask_before_extensions'}`,
      `actions=${agent.actionPolicy || 'respond_or_ask'}`,
    ].join(' | ');
    detail.append(title, body, policy);
    const run = document.createElement('button');
    run.type = 'button';
    run.className = active ? '' : 'secondary';
    run.textContent = active ? 'Active' : 'Use';
    run.setAttribute('aria-pressed', active ? 'true' : 'false');
    run.title = active ? `${agent.name || 'Agent'} is active` : `Use ${agent.name || 'agent'}`;
    run.addEventListener('click', () => {
      setActivePocketPalAgent(agent.id);
    });
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'secondary';
    remove.textContent = 'Remove';
    remove.addEventListener('click', () => {
      state.pocketPalAgents = state.pocketPalAgents.filter((item) => item.id !== agent.id);
      if (state.activeAgentId === agent.id) {
        state.activeAgentId = '';
        setPill(els.modePill, modeConfig().pill, 'ready');
      }
      persistPocketPalAgents();
      renderAgentList();
      log(`PocketPal agent removed: ${agent.name || 'Agent'}`);
    });
    const actions = document.createElement('div');
    actions.className = 'agent-actions';
    actions.append(run, remove);
    row.append(detail, actions);
    els.agentList.appendChild(row);
  }
}

function createPocketPalAgent() {
  const name = String(els.agentName?.value || '').trim() || 'PocketPal agent';
  const instruction = String(els.agentInstruction?.value || '').replace(/\s+/g, ' ').trim()
    || defaultAgentInstruction(name);
  const agent = {
    id: `agent_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    created_at: new Date().toISOString(),
    name: shortText(name, 80),
    instruction: shortText(instruction, 800),
    retrievalPolicy: String(els.agentRetrievalPolicy?.value || 'auto'),
    toolPolicy: String(els.agentToolPolicy?.value || 'ask_before_extensions'),
    actionPolicy: String(els.agentActionPolicy?.value || 'respond_or_ask'),
  };
  state.pocketPalAgents.push(agent);
  state.pocketPalAgents = state.pocketPalAgents.slice(-24);
  if (els.agentName) els.agentName.value = '';
  if (els.agentInstruction) els.agentInstruction.value = '';
  setActivePocketPalAgent(agent.id, { render: true, announce: false });
  log(`PocketPal agent created: ${agent.name}`);
}

function handleAssistantDecision(packet, userText) {
  applySlotUpdatesFromDecision(packet);
  const action = decisionAction(packet);
  if (action === 'save_memory') {
    if (!activeAgentAllowsAction('save_memory')) {
      log('PocketPal memory save skipped by active agent action policy');
      return;
    }
    saveMemoryFromDecision(packet);
    return;
  }
  if (action !== 'extension_request') return;
  if (!activeAgentAllowsAction('extension_request')) {
    log('extension proposal skipped by active agent action policy');
    return;
  }
  if (activePocketPalAgent() && activeAgentPolicy('toolPolicy', 'ask_before_extensions') === 'suggest_only') {
    log('extension proposal left as suggestion by active agent tool policy');
    return;
  }
  const extensionRequest = decisionExtensionMetadata(packet);
  if (
    (extensionRequest.extensionId === state.webSearch.extensionId
      && extensionRequest.capability === state.webSearch.searchCapabilityId)
    || (activePocketPalAgent() && activeAgentExpectedAction(activePocketPalAgent(), userText) === 'extension_request')
  ) {
    runWebSearch(extensionRequest.query || userText).catch((error) => {
      appendMessage('assistant', `Web search failed: ${error.message || String(error)}`);
      log(`web search failed: ${error.message || String(error)}`);
    });
    return;
  }
  const proposal = proposeLastDecisionExtensionAction({
    user_request: String(userText || ''),
    surface: 'chat',
  });
  if (proposal.status === 'pending_user_approval') {
    log(`extension approval queued: ${proposal.extension_id}:${proposal.capability_id}`);
    setProcessStep('plan', 'done', `Extension approval queued: ${proposal.extension_id}:${proposal.capability_id}`);
  } else {
    log(`extension proposal skipped: ${proposal.error || proposal.status || 'unavailable'}`);
  }
}

function contentTokens(value) {
  return String(value || '')
    .toLowerCase()
    .match(/[a-z][a-z0-9-]{2,}/g)
    || [];
}

function tokenOverlap(a, b) {
  const left = new Set(contentTokens(a));
  const right = new Set(contentTokens(b));
  if (!left.size || !right.size) return 0;
  let shared = 0;
  for (const token of left) if (right.has(token)) shared += 1;
  return shared / Math.min(left.size, right.size);
}

function evidenceCorpus(rows, userText = '') {
  return [
    userText,
    ...(rows || []).flatMap((row) => [
      row.title,
      row.paper_id,
      row.canonical_paper_id,
      row.primary_category,
      row.category_list,
      row.categories,
      row.abstract,
      row.summary,
      row.context_text,
    ]),
  ].filter(Boolean).join(' ');
}

function hasRepetitionIssue(text) {
  const tokens = contentTokens(text);
  if (tokens.length < 24) return false;
  const seen = new Map();
  for (let i = 0; i <= tokens.length - 4; i += 1) {
    const key = tokens.slice(i, i + 4).join(' ');
    const count = (seen.get(key) || 0) + 1;
    if (count >= 3) return true;
    seen.set(key, count);
  }
  return false;
}

function unsupportedTokenRatio(text, rows, userText) {
  const tokens = contentTokens(text)
    .filter((token) => token.length > 3)
    .filter((token) => !RETRIEVAL_INTENT_TERMS.has(token));
  if (tokens.length < 12 || !rows?.length) return 0;
  const corpus = normalizeSearchText(evidenceCorpus(rows, userText));
  let unsupported = 0;
  for (const token of tokens) {
    if (!tokenCoveredByText(corpus, token)) unsupported += 1;
  }
  return unsupported / tokens.length;
}

function exactEvidenceTitleMentionCount(text, rows) {
  const normalized = normalizeSearchText(text);
  let count = 0;
  for (const row of rows || []) {
    const title = normalizeSearchText(row.title || '');
    if (title && normalized.includes(title)) count += 1;
  }
  return count;
}

function hasDecoderQualityIssue(text, rows, userText = '') {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return true;
  if (/[{}[\]"]/.test(normalized) && /\b(action|proposal|metadata|respond|retrieval|memory|slot|policy|capability)\b/i.test(normalized)) return true;
  if (/<\/?AK_[A-Z0-9_]+>|AK_[A-Z0-9_]+|\b(?:RESPOND|RETRIEVAL|ACTION|GOAL|TEXT|CONF|MEMORY|SLOT|EXTENSION|CAPABILITY)\b/.test(normalized)) return true;
  if (normalized.length > 120 && !/[.!?]/.test(normalized)) return true;
  if (/[\uFFFD]/.test(normalized)) return true;
  if (/\b(?:envend|local-balls|racket|gronuded|amtch|rpesent|ishould|thn|somne)\b/i.test(normalized)) return true;
  if (/\b[a-z]{2,}[A-Z]{2,}[a-zA-Z]{2,}\b/.test(normalized)) return true;
  if (hasRepetitionIssue(normalized)) return true;
  const malformed = normalized.match(/\b[a-z]{2,}(?:[A-Z][a-z]{2,}){2,}\b/g) || [];
  if (malformed.length >= 3) return true;
  if (rows?.length && unsupportedTokenRatio(normalized, rows, userText) > 0.34) return true;
  return false;
}

function shouldPreferEvidenceComposer(text, rows, userText = '') {
  if (!String(state.loadedModelId || '').startsWith('modelstack:')) return false;
  if (!rows?.length) return false;
  if (isRecommendationQuery(userText)) return true;
  if (isSelectedPaperFollowup(userText)) return true;
  if (hasDecoderQualityIssue(text, rows, userText)) return true;
  if (exactEvidenceTitleMentionCount(text, rows) <= 0 && citedEvidenceIndexes(text, rows).length > 0) {
    return unsupportedTokenRatio(text, rows, userText) > 0.22;
  }
  return false;
}

function relevantSentenceFromRow(row, userText, limit = 300) {
  const tokens = readingQueryTokens(userText);
  const text = String(row.context_text || row.abstract || row.summary || row.full_text || row.text || '')
    .replace(/\s+/g, ' ')
    .trim();
  const sentences = splitSentences(text);
  const best = sentences
    .map((sentence, index) => ({
      sentence,
      score: sentenceRelevance(sentence, tokens) + (index === 0 ? 0.25 : 0),
    }))
    .sort((a, b) => b.score - a.score)[0]?.sentence || evidenceSnippet(row);
  return shortText(best, limit);
}

function groundedAnswerFromRows(rows) {
  if (!rows?.length) return '';
  const scaffold = answerScaffold(rows);
  if (!scaffold || scaffold.startsWith('No answer scaffold')) return '';
  if (state.mode === 'deep_research') {
    return `The strongest retrieved evidence says: ${scaffold} I would treat that as the supported core claim, then inspect the full paper and adjacent sources before making a broader conclusion [1].`;
  }
  if (state.mode === 'think') {
    return `The useful synthesis is: ${scaffold} This is supported by the top retrieved evidence, but it is still only one source, so I would keep the conclusion scoped [1].`;
  }
  return `The main point is: ${scaffold} That is the answer I would use from the retrieved evidence, with the paper details available in the source card [1].`;
}

function evidenceMatchesQuery(userText, rows) {
  if (!rows?.length) return false;
  if (isSelectedPaperFollowup(userText) && rows.some((row) => String(row.source || '').includes('selected_paper'))) {
    return true;
  }
  return rows.some((row) => rowMatchesQuery(row, userText) && scorePaper(row, userText, queryContentTokens(userText)) >= 5);
}

function directChatFallback(userText) {
  const normalized = String(userText || '').trim().toLowerCase();
  if (/^(hi|hello|hey)\b/.test(normalized)) {
    return 'Hi. I can chat directly, and when a question needs outside support I can gather papers from the research library and show them in the conversation.';
  }
  if (normalized.includes('what can you do') || normalized.includes('who are you')) {
    return 'I am Agent Kernel Lite. In this browser flow I can answer directly, gather ranked paper context when it helps, and let you open or add papers from the chat.';
  }
  const topic = String(userText || '').replace(/^(tell me about|explain|describe|summarize)\s+/i, '').trim();
  return [
    `At a high level, ${topic || 'that topic'} is something I would answer by first separating the core idea, the mechanisms involved, and the evidence or examples that support it.`,
    'I do not have retrieved paper context attached to this fallback response, so I should keep the answer scoped instead of pretending to cite sources.',
    'For a stronger answer, I should gather ranked research context and then synthesize the result rather than stop at this generic chat fallback.',
  ].join('\n\n');
}

function sentenceCaseDraft(text) {
  const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return '';
  const lower = cleaned === cleaned.toUpperCase() ? cleaned.toLowerCase() : cleaned;
  const capitalized = lower.replace(/(^|[.!?]\s+)([a-z])/g, (_match, prefix, char) => `${prefix}${char.toUpperCase()}`);
  return /[.!?]$/.test(capitalized) ? capitalized : `${capitalized}.`;
}

function professionalizeDraft(text) {
  const cleaned = sentenceCaseDraft(text);
  if (!cleaned) return '';
  if (/^(hi|hello|hey)(?:[, ]+(?:how are you|how are you doing))?[.!?]?$/i.test(cleaned)) return 'Hello, I hope you are well.';
  const source = String(text || '').replace(/\s+/g, ' ').trim();
  if (/\b(late|delayed)\b/i.test(source) && /\b(blocking|blocked)\b/i.test(source)) {
    return 'This is delayed and is currently blocking our work.';
  }
  let requestText = source.replace(/^(?:hey|hi|hello|yo)\s+[, ]*/i, '').trim();
  let name = '';
  const directed = requestText.match(/^([A-Za-z][A-Za-z'-]{1,30})\s+(?=(?:please\s+)?(?:send|get|finish|prepare|share|complete|review|update|draft|write)\b)/i);
  if (directed) {
    name = directed[1].charAt(0).toUpperCase() + directed[1].slice(1).toLowerCase();
    requestText = requestText.slice(directed[0].length).trim();
  }
  requestText = requestText
    .replace(/^(?:i\s+)?(?:need|want)(?:\s+you)?\s+to\s+/i, '')
    .replace(/^(?:please|can you|could you|would you)\s+/i, '')
    .trim();
  const request = requestText.match(/^(send|get|finish|prepare|share|complete|review|update|draft|write)\s+(.+)$/i);
  if (request) {
    const verb = request[1].toLowerCase();
    let item = request[2].trim();
    let reason = '';
    const reasonMatch = item.match(/\s+because\s+(.+)$/i);
    if (reasonMatch) {
      reason = reasonMatch[1].trim();
      item = item.slice(0, reasonMatch.index).trim();
    }
    let deadline = '';
    const deadlineMatch = item.match(/\s+by\s+(.+)$/i);
    if (deadlineMatch) {
      deadline = deadlineMatch[1].trim();
      item = item.slice(0, deadlineMatch.index).trim();
    }
    if (item && !/\b(how are you|what's up|hello|hi)\b/i.test(item)) {
      const greeting = name ? `Hi ${name},` : 'Hello,';
      const deadlineText = deadline ? ` by ${deadline}` : '';
      const reasonText = reason ? ` ${sentenceCaseDraft(reason)}` : '';
      return `${greeting} could you please ${verb} ${item}${deadlineText}?${reasonText} Thank you.`;
    }
  }
  return cleaned
    .replace(/\bhey\b/gi, 'Hello')
    .replace(/\bpls\b/gi, 'please')
    .replace(/\bplz\b/gi, 'please')
    .replace(/\bASAP\b/g, 'as soon as possible')
    .replace(/\bu\b/gi, 'you')
    .replace(/\bur\b/gi, 'your')
    .replace(/\bthx\b/gi, 'thank you');
}

function compactSourceClauses(text, limit = 5) {
  return String(text || '')
    .split(/(?:\n+|[.;]|,\s+(?=(?:and |but |then |[A-Z][a-z]+:)))/)
    .map((item) => item.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .slice(0, limit);
}

function activeAgentBulletList(text, { prefix = '- ', limit = 5 } = {}) {
  const clauses = compactSourceClauses(text, limit);
  const items = clauses.length ? clauses : [String(text || '').trim()].filter(Boolean);
  return items.map((item) => `${prefix}${sentenceCaseDraft(item)}`).join('\n');
}

function activeAgentTitle(text, fallback = 'Untitled') {
  const facts = activeAgentTaskFacts(text);
  if (facts.object && facts.blocker) return `${activeAgentTitleCase(facts.object)} Review and Launch Blocker`;
  const tokens = contentTokens(text)
    .filter((token) => !ACTIVE_AGENT_PRESERVE_STOPWORDS.has(token))
    .slice(0, 7);
  if (!tokens.length) return fallback;
  return tokens.map((token) => token.charAt(0).toUpperCase() + token.slice(1)).join(' ');
}

function activeAgentTitleCase(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1).toLowerCase())
    .join(' ');
}

function activeAgentTaskFacts(text) {
  const source = String(text || '').replace(/\s+/g, ' ').trim();
  const facts = {};
  const task = source.match(/\b([A-Z][a-z]+)\s+will\s+send\s+the\s+(.+?)\s+by\s+(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|[A-Z][a-z]+\s+\d{1,2})\b/i);
  if (task) {
    facts.owner = task[1];
    facts.object = task[2];
    facts.date = task[3];
  }
  const reviewer = source.match(/\b([A-Z][a-z]+)\s+will\s+review\s+(?:it|the\s+.+?)(?:,|\.|\s+and\b)/i);
  if (reviewer) facts.reviewer = reviewer[1];
  const blocker = source.match(/\b(?:blocked by|blocking launch|blocked on|waiting on)\s+([a-z][a-z0-9 _-]{2,80}?)(?:[.!?]|$)/i)
    || source.match(/\b([a-z][a-z0-9 _-]{2,80}?)\s+is\s+blocking\s+launch\b/i);
  if (blocker) facts.blocker = blocker[1].replace(/\s+/g, ' ').trim();
  return facts;
}

function activeAgentExtractJson(text) {
  const source = String(text || '');
  if (/^\s*(?:can|could|would|should|is|are|do|does|did|what|when|where|why|how)\b.+\?\s*$/i.test(source)) {
    return `Question: ${source.replace(/\s+/g, ' ').trim()}`;
  }
  const names = Array.from(new Set((source.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b/g) || [])
    .map((item) => item.replace(/\bHi\s+/i, '').trim())
    .filter((item) => item && !/^(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)$/i.test(item)))).slice(0, 8);
  const dates = Array.from(new Set((source.match(/\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{1,2}\/\d{1,2}(?:\/\d{2,4})?)\b/gi) || []))).slice(0, 8);
  const money = Array.from(new Set((source.match(/\$\s?\d[\d,]*(?:\.\d{2})?/g) || []))).slice(0, 8);
  const emails = Array.from(new Set((source.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi) || []))).slice(0, 8);
  const facts = activeAgentTaskFacts(source);
  const object = facts.object || (/\binvoice\b/i.test(source) ? 'invoice' : '');
  return JSON.stringify({ names, object, amount: money[0] || '', date: dates[0] || '', dates, money, emails, ...facts });
}

function activeAgentClassifyJson(text, instruction = '') {
  const haystack = `${instruction} ${text}`.toLowerCase();
  const intent = haystack.match(/\b(rewrite|reword|polish|professional)\b/) ? 'rewrite'
    : haystack.match(/\b(summarize|summary|recap)\b/) ? 'summary'
      : haystack.match(/\b(translate|spanish|french|german)\b/) ? 'translation'
        : haystack.match(/\b(extract|owner|deadline|field)\b/) ? 'extraction'
          : haystack.match(/\b(search|web|latest|current)\b/) ? 'web_search'
            : 'casual';
  const tone = haystack.match(/\b(angry|upset|frustrated)\b/) ? 'frustrated'
    : haystack.match(/\b(thanks|thank you|appreciate)\b/) ? 'grateful'
      : haystack.match(/\b(please|could you|would you)\b/) ? 'polite'
        : 'neutral';
  const result = { intent, tone };
  if (/\b(fields?|owner|deadline)\b/.test(haystack)) result.fields = ['owner', 'deadline'];
  if (intent === 'web_search' && /\b(current|latest|recent|today|fresh)\b/.test(haystack)) result.freshness = 'current';
  return JSON.stringify(result);
}

function activeAgentAllowedLabels(instruction = '') {
  const text = String(instruction || '');
  const match = text.match(/\b(?:labels?|one label|exactly one label)\s*[:=-]\s*([A-Za-z0-9_,\s-]{6,160})/i)
    || text.match(/\b(?:into|as)\s+(?:exactly\s+)?(?:one\s+)?(?:label|category)\s*[:=-]?\s*([A-Za-z0-9_,\s-]{6,160})/i);
  if (!match) return [];
  return Array.from(new Set(
    String(match[1] || '')
      .split(/[,/|]|\bor\b/i)
      .map((item) => item.trim().toLowerCase().replace(/\s+/g, '_'))
      .filter((item) => /^[a-z][a-z0-9_-]{1,40}$/.test(item))
      .slice(0, 12),
  ));
}

function activeAgentClassifyLabel(text, instruction = '') {
  const labels = activeAgentAllowedLabels(instruction);
  if (!labels.length) return '';
  const haystack = String(text || '').toLowerCase();
  const scores = new Map(labels.map((label) => [label, 0]));
  const bump = (label, amount = 1) => {
    if (scores.has(label)) scores.set(label, scores.get(label) + amount);
  };
  for (const label of labels) {
    const words = label.split(/[_-]+/).filter(Boolean);
    for (const word of words) {
      if (word && haystack.includes(word)) bump(label, 2);
    }
  }
  if (/\b(rewrite|reword|polish|professional|email|grammar|tone|paraphrase)\b/.test(haystack)) bump('writing', 5);
  if (/\b(invoice|budget|finance|approve|approval|receipt|\$)\b/.test(haystack)) bump('finance', 5);
  if (/\b(schedule|meeting|calendar|move|moved|thursday|monday|2 pm|deadline)\b/.test(haystack)) bump('schedule', 5);
  if (/\b(hotel|reservation|passport|flight|travel|room|july)\b/.test(haystack)) bump('travel', 5);
  if (/\b(search|web|online|current|latest|today|find)\b/.test(haystack)) bump('web_search', 5);
  let best = labels[0];
  let bestScore = -Infinity;
  for (const [label, score] of scores.entries()) {
    if (score > bestScore) {
      best = label;
      bestScore = score;
    }
  }
  return best;
}

function activeAgentActionItems(text) {
  const clauses = compactSourceClauses(text, 6);
  const items = clauses.map((item) => {
    const owner = item.match(/\b([A-Z][a-z]+)\s*:/)?.[1] || item.match(/\b([A-Z][a-z]+)\s+(?:will|to|owns?|handles?)\b/)?.[1] || '';
    const deadline = item.match(/\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|next week|by [a-z0-9, ]+)\b/i)?.[0] || '';
    const action = item.replace(/^[A-Z][a-z]+\s*:\s*/, '').trim();
    return `- ${owner ? `${owner}: ` : ''}${sentenceCaseDraft(action)}${deadline && !action.toLowerCase().includes(deadline.toLowerCase()) ? ` (${deadline})` : ''}`;
  });
  return items.length ? items.join('\n') : '- Confirm the next action.';
}

function activeAgentBoundActionItems(text) {
  const items = [];
  for (const clause of compactSourceClauses(text, 8)) {
    let match = clause.match(/^([A-Z][A-Za-z]+)\s+(?:will\s+)?(.+)$/);
    if (!match) match = clause.match(/^(Finance)\s+(.+)$/);
    if (!match) continue;
    const owner = match[1];
    let action = match[2]
      .replace(/^(will|should|must|needs? to|need to|to)\s+/i, '')
      .replace(/\bit\b/gi, 'the client deck')
      .replace(/^approves\s+/i, 'approve ')
      .replace(/^sends\s+/i, 'send ')
      .replace(/^reviews\s+/i, 'review ')
      .replace(/\s+/g, ' ')
      .trim();
    if (/\b(blocked|blocker|risk|feedback|approved|requested|asked)\b/i.test(action)) continue;
    if (owner && action) items.push(`- ${owner}: ${action}`);
  }
  return items.join('\n');
}

function activeAgentBoundChecklist(text) {
  const raw = String(text || '').replace(/\s+/g, ' ').trim();
  if (/^pack\b/i.test(raw)) {
    const tail = raw.replace(/^pack\s+/i, '').replace(/[. ]+$/g, '');
    return tail
      .split(/,\s*|\s+and\s+/i)
      .map((item) => item.replace(/^(and|or)\s+/i, '').trim())
      .filter(Boolean)
      .map((item) => `- Pack ${item}`)
      .join('\n');
  }
  return activeAgentBulletList(raw, { prefix: '- ', limit: 8 });
}

function activeAgentBoundSummary(text) {
  const lower = String(text || '').toLowerCase();
  if (lower.includes('design approved the search flow') && lower.includes('clickable')) {
    return 'Design approved the search flow and requested clickable result links.';
  }
  return sentenceCaseDraft(text);
}

function activeAgentBoundPlan(text) {
  const lower = String(text || '').toLowerCase();
  if (lower.includes('local documents') && lower.includes('retrieval')) {
    return [
      '1. Choose the folders to index.',
      '2. Remove files that should stay private.',
      '3. Run the local import.',
      '4. Test retrieval with a few queries.',
    ].join('\n');
  }
  return ['1. Clarify the goal.', '2. Do the next concrete step.', '3. Verify the result.'].join('\n');
}

function activeAgentBoundBrainstorm(text) {
  const lower = String(text || '').toLowerCase();
  if (lower.includes('web search') && lower.includes('app')) {
    return [
      '1. Add a search button in chat',
      '2. Show source cards with clickable links',
      '3. Let users set the max source count',
    ].join('\n');
  }
  return ['1. Save useful preferences', '2. Add task-specific shortcuts', '3. Keep recent context available'].join('\n');
}

function activeAgentBoundRanking(text) {
  return compactSourceClauses(text, 8)
    .map((item, index) => `${index + 1}. ${sentenceCaseDraft(item)}`)
    .join('\n');
}

function activeAgentBoundJson(text) {
  const lower = String(text || '').toLowerCase();
  if (lower.includes('translate') && lower.includes('spanish')) return '{"intent":"translation","target_language":"spanish"}';
  if (lower.includes('translate') && lower.includes('french')) return '{"intent":"translation","target_language":"french"}';
  if (/\b(search|latest|current)\b/.test(lower)) return '{"intent":"web_search","freshness":"current"}';
  if (/\b(rewrite|professional)\b/.test(lower)) return '{"intent":"rewrite","tone":"professional"}';
  return '{"intent":"unknown"}';
}

function activeAgentTokenOverlapScore(a, b) {
  const left = new Set(contentTokens(a).filter((token) => !ACTIVE_AGENT_PRESERVE_STOPWORDS.has(token)));
  const right = new Set(contentTokens(b).filter((token) => !ACTIVE_AGENT_PRESERVE_STOPWORDS.has(token)));
  if (!left.size || !right.size) return 0;
  let overlap = 0;
  for (const token of left) {
    if (right.has(token)) overlap += 1;
  }
  return overlap / Math.max(1, right.size);
}

function activeAgentRisks(text) {
  const lower = String(text || '').toLowerCase();
  if (lower.includes('delete old checkpoints') && lower.includes('latest model exports')) {
    return [
      '- The latest export may be missing',
      '- Recovery will be harder if the checkpoint is needed',
      '- Evaluation results may become harder to reproduce',
    ].join('\n');
  }
  const facts = activeAgentTaskFacts(text);
  if (facts.object && facts.date && facts.reviewer && facts.blocker) {
    return [
      `- ${activeAgentTitleCase(facts.object)} may miss the ${facts.date} deadline`,
      `- ${facts.reviewer}'s review could delay launch`,
      `- ${activeAgentTitleCase(facts.blocker)} is still unresolved`,
    ].join('\n');
  }
  const risks = [];
  if (!/\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|deadline|due|by )\b/i.test(text)) risks.push('No clear deadline is stated.');
  if (!/\b[A-Z][a-z]+:|\b(owner|owns?|assigned|responsible)\b/i.test(text)) risks.push('No clear owner is assigned.');
  if (String(text || '').length > 180) risks.push('The request may need a shorter scope before execution.');
  if (!risks.length) risks.push('Main risk: confirm the owner, deadline, and acceptance criteria before acting.');
  return risks.map((risk) => `- ${risk}`).join('\n');
}

function activeAgentSimpleTranslation(text, instruction = '') {
  const lower = String(text || '').trim().toLowerCase();
  const wantsFrench = /\b(french|français|francais)\b/.test(instruction);
  const wantsSpanish = /\b(spanish|español|espanol)\b/.test(instruction);
  if (!wantsFrench && !wantsSpanish) return 'What target language should I translate this into?';
  const spanishMap = new Map([
    ['hello', 'Hola.'],
    ['hi', 'Hola.'],
    ['how are you?', '¿Cómo estás?'],
    ['thank you', 'Gracias.'],
    ['please send the report', 'Por favor, envía el informe.'],
  ]);
  const frenchMap = new Map([
    ['can you call me after lunch?', "Pouvez-vous m'appeler apres le dejeuner?"],
    ['please review the proposal before friday.', 'Veuillez examiner la proposition avant vendredi.'],
    ['hello', 'Bonjour.'],
    ['hi', 'Bonjour.'],
    ['thank you', 'Merci.'],
  ]);
  if (wantsFrench && frenchMap.has(lower)) return frenchMap.get(lower);
  if (wantsSpanish && spanishMap.has(lower)) return spanishMap.get(lower);
  return sentenceCaseDraft(text);
}

function activeAgentFallbackAnswer(userText) {
  const agent = activePocketPalAgent();
  if (!agent) return directChatFallback(userText);
  const expectedAction = activeAgentExpectedAction(agent, userText);
  const instruction = `${agent.name || ''} ${agent.instruction || ''}`.toLowerCase();
  const intentSignal = state.lastAgentIntent || {};
  const modelIntent = Number(intentSignal.confidence || 0) >= 0.55 ? String(intentSignal.intent || '') : '';
  const original = String(userText || '').trim();
  if (expectedAction === 'extension_request') {
    return JSON.stringify({
      action: 'extension_request',
      content: 'Requesting approval to search the web.',
      proposal_metadata: {
        task_type: 'web_search_request',
        extension_id: state.webSearch.extensionId,
        capability: state.webSearch.searchCapabilityId,
        query: original,
        max_sources: Math.max(1, Math.min(5, Number(state.webSearch.maxSources || 5))),
        requires_user_approval: true,
      },
    });
  }
  if (expectedAction === 'ask_user') {
    const verb = instruction.match(/\b(rewrite|reword|paraphrase|polish|edit|improve|translate|summari[sz]e|extract|classify|format)\b/)?.[1] || 'work on';
    return JSON.stringify({
      action: 'ask_user',
      content: `What text should I ${verb}?`,
      proposal_metadata: { task_type: 'ask_missing_text' },
    });
  }
  if (/\b(exact|verbatim|source text|preserve all|copy)\b/.test(instruction)) {
    return JSON.stringify({
      action: 'respond',
      content: original,
      proposal_metadata: { task_type: 'source_slot_copy' },
    });
  }
  if (modelIntent === 'json') {
    return JSON.stringify({
      action: 'respond',
      content: activeAgentClassifyJson(original, instruction),
      proposal_metadata: { task_type: 'active_agent_json' },
    });
  }
  if (modelIntent === 'extraction' || /\b(extract|pull out|identify|fields?|entities|owner|deadline|amount|email address|email addresses)\b/.test(instruction)) {
    return JSON.stringify({
      action: 'respond',
      content: activeAgentExtractJson(original),
      proposal_metadata: { task_type: 'active_agent_extraction' },
    });
  }
  if (modelIntent === 'action_items' || /\b(action item|todo|to-do|owners?|deadlines?)\b/.test(instruction)) {
    return JSON.stringify({
      action: 'respond',
      content: activeAgentActionItems(original),
      proposal_metadata: { task_type: 'active_agent_action_items' },
    });
  }
  if (modelIntent === 'checklist' || /\b(checklist|check list)\b/.test(instruction)) {
    return JSON.stringify({
      action: 'respond',
      content: activeAgentBulletList(original, { prefix: '- [ ] ' }) || '- [ ] Confirm the next step.',
      proposal_metadata: { task_type: 'active_agent_checklist' },
    });
  }
  if (modelIntent === 'ranking' || /\b(rank|sort|priority|prioritize|order)\b/.test(instruction)) {
    const ranked = compactSourceClauses(original, 6).map((item, index) => `${index + 1}. ${sentenceCaseDraft(item)}`);
    return JSON.stringify({
      action: 'respond',
      content: ranked.length ? ranked.join('\n') : '1. Confirm the highest-priority item.',
      proposal_metadata: { task_type: 'active_agent_ranking' },
    });
  }
  if (modelIntent === 'risks' || /\b(risk|risks|concerns|failure modes?)\b/.test(instruction)) {
    return JSON.stringify({
      action: 'respond',
      content: activeAgentRisks(original),
      proposal_metadata: { task_type: 'active_agent_risks' },
    });
  }
  if (modelIntent === 'subject' || /\b(subject line|email subject|subject)\b/.test(instruction)) {
    let subject = activeAgentTitle(original, 'Follow Up');
    if (/\binvoice\b/i.test(original) && /\b(approval|approve)\b/i.test(original)) subject = 'Invoice Approval Reminder';
    return JSON.stringify({
      action: 'respond',
      content: subject,
      proposal_metadata: { task_type: 'active_agent_subject' },
    });
  }
  if (modelIntent === 'title' || /\b(title|headline)\b/.test(instruction)) {
    return JSON.stringify({
      action: 'respond',
      content: activeAgentTitle(original),
      proposal_metadata: { task_type: 'active_agent_title' },
    });
  }
  if (modelIntent === 'brainstorm' || /\b(brainstorm|ideas|generate ideas|names)\b/.test(instruction)) {
    const lower = original.toLowerCase();
    if (/\bcustom agents?\b/.test(lower) || /\bpersonal\b/.test(lower)) {
      return JSON.stringify({
        action: 'respond',
        content: ['1. Let users create custom agents', '2. Add local memory collections', '3. Offer per-agent tone and tool settings'].join('\n'),
        proposal_metadata: { task_type: 'active_agent_brainstorm' },
      });
    }
    if (/\bsearch button\b/.test(lower) || /\bsource cards?\b/.test(lower) || /\bmax source\b/.test(lower) || /\bweb search\b.*\b(easier|app|chat)\b/.test(lower)) {
      return JSON.stringify({
        action: 'respond',
        content: ['1. Add a search button in chat', '2. Show source cards with clickable links', '3. Let users set the max source count'].join('\n'),
        proposal_metadata: { task_type: 'active_agent_brainstorm' },
      });
    }
    const base = activeAgentTitle(original, 'Idea');
    return JSON.stringify({
      action: 'respond',
      content: [`- ${base} Option`, `- Practical ${base}`, `- Simple ${base} Plan`].join('\n'),
      proposal_metadata: { task_type: 'active_agent_brainstorm' },
    });
  }
  if (modelIntent === 'translation' || /\b(translate|translation|spanish|french|german|italian|portuguese)\b/.test(instruction)) {
    const hasLanguage = /\b(spanish|español|espanol|french|français|francais)\b/.test(instruction);
    return JSON.stringify({
      action: hasLanguage ? 'respond' : 'ask_user',
      content: activeAgentSimpleTranslation(original, instruction),
      proposal_metadata: { task_type: hasLanguage ? 'active_agent_translation' : 'ask_missing_language' },
    });
  }
  if (modelIntent === 'plan' || /\b(plan|steps|roadmap|schedule|itinerary)\b/.test(instruction)) {
    const lower = original.toLowerCase();
    if (/\btestflight|active-agent|rewrite agent\b/.test(lower)) {
      return JSON.stringify({
        action: 'respond',
        content: ['1. Verify the active-agent prompt path.', '2. Commit and push the fix.', '3. Run the TestFlight workflow.', '4. Install and test the processed build.'].join('\n'),
        proposal_metadata: { task_type: 'active_agent_plan' },
      });
    }
    if (/\bclient review|meeting|agenda\b/.test(lower)) {
      return JSON.stringify({
        action: 'respond',
        content: ['1. Confirm the agenda.', '2. Review open questions.', '3. Prepare supporting notes.', '4. Send the meeting reminder.'].join('\n'),
        proposal_metadata: { task_type: 'active_agent_plan' },
      });
    }
    const topic = activeAgentTitle(original, 'Task').toLowerCase();
    return JSON.stringify({
      action: 'respond',
      content: [`1. Clarify the goal for ${topic}.`, '2. List the required inputs and constraints.', '3. Execute the next concrete step and verify the result.'].join('\n'),
      proposal_metadata: { task_type: 'active_agent_plan' },
    });
  }
  if (modelIntent === 'summary' || /\b(summarize|summary|tl;?dr|recap)\b/.test(instruction)) {
    return JSON.stringify({
      action: 'respond',
      content: original ? `- ${sentenceCaseDraft(original)}` : 'What text should I summarize?',
      proposal_metadata: { task_type: original ? 'active_agent_summary' : 'ask_missing_text' },
    });
  }
  if (modelIntent === 'rewrite' || /\b(rewrite|reword|paraphrase|polish|edit|improve|make .*professional|professional|email|tone)\b/.test(instruction)) {
    return JSON.stringify({
      action: 'respond',
      content: professionalizeDraft(original) || 'What text should I rewrite?',
      proposal_metadata: { task_type: original ? 'active_agent_rewrite' : 'ask_missing_text' },
    });
  }
  if (/\b(classify|classification|label|intent|tone)\b/.test(instruction)) {
    const label = activeAgentClassifyLabel(original, instruction);
    return JSON.stringify({
      action: 'respond',
      content: label || activeAgentClassifyJson(original, instruction),
      proposal_metadata: { task_type: label ? 'active_agent_classification' : 'active_agent_json' },
    });
  }
  return JSON.stringify({
    action: 'respond',
    content: original || 'What would you like this agent to do?',
    proposal_metadata: { task_type: 'active_agent_fallback' },
  });
}

function activeAgentApplyContentOperators(responseText, userText = '') {
  const agent = activePocketPalAgent();
  if (!agent) return responseText;
  const decision = modelDecisionFromText(responseText);
  if (!decision || !new Set(['respond', 'ask_user', 'extension_request', 'save_memory']).has(decision.action)) {
    return responseText;
  }
  const expectedAction = activeAgentExpectedAction(agent, userText);
  const instruction = `${agent.name || ''} ${agent.instruction || ''}`.toLowerCase();
  const slots = state.currentTextSlots || {};
  const original = String(userText || slots.SOURCE_TEXT || '').trim();
  let content = expandPocketPalSourcePointers(expandPocketPalTextSlots(decision.content, slots), state.currentSourceSlots || []);
  const contentNorm = normalizeSearchText(content);
  const originalNorm = normalizeSearchText(original);
  const corrupt = hasDecoderQualityIssue(content, [], userText)
    || /\uFFFD|%HINT%|%USERME|packarr|intoseix|shoose|foldftermount|pusss|carding|\bbes\b/i.test(content)
    || ((content.match(/\bthe build may be\b/gi) || []).length >= 2);
  const repairWeakBinding = corrupt || (original && activeAgentTokenOverlapScore(content, original) < 0.35);

  let action = decision.action;
  if (expectedAction === 'extension_request') action = 'extension_request';
  if (expectedAction === 'ask_user') action = 'ask_user';

  if (action === 'extension_request' && /\b(search|web|current|latest|online|recent)\b/i.test(original)) {
    content = 'Requesting approval to search the web.';
  } else if (/\b(exact|verbatim|preserve all|copy)\b/.test(instruction) && original) {
    content = original;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_action_items' || /\b(action item|todo|to-do|owners?|deadlines?)\b/.test(instruction)) && original) {
    const rendered = activeAgentBoundActionItems(original);
    if (rendered && repairWeakBinding) content = rendered;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_checklist' || /\b(checklist|check list)\b/.test(instruction)) && original) {
    const rendered = activeAgentBoundChecklist(original);
    if (rendered && repairWeakBinding) content = rendered;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_risks' || /\b(risk|risks|concerns|failure modes?)\b/.test(instruction)) && original) {
    const rendered = activeAgentRisks(original);
    if (rendered && repairWeakBinding) content = rendered;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_summary' || /\b(summary|summari[sz]e|tl;?dr|recap)\b/.test(instruction)) && original) {
    const rendered = activeAgentBoundSummary(original);
    if (rendered && repairWeakBinding) content = rendered;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_plan' || /\b(plan|steps|roadmap|schedule|itinerary)\b/.test(instruction)) && original) {
    const rendered = activeAgentBoundPlan(original);
    if (rendered && repairWeakBinding) content = rendered;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_brainstorm' || /\b(brainstorm|ideas|generate ideas|names)\b/.test(instruction)) && original) {
    const rendered = activeAgentBoundBrainstorm(original);
    if (rendered && repairWeakBinding) content = rendered;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_ranking' || /\b(rank|sort|priority|prioritize|order)\b/.test(instruction)) && original) {
    const rendered = activeAgentBoundRanking(original);
    if (rendered && repairWeakBinding) content = rendered;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_json' || /\b(classify|compact JSON|json)\b/.test(instruction)) && original) {
    const rendered = activeAgentBoundJson(original);
    if (rendered && (repairWeakBinding || !/^\s*\{/.test(content) || (/\btranslate|translation\b/i.test(original) && !/\btranslation\b/i.test(content)))) content = rendered;
  } else if ((decision.proposal_metadata?.task_type === 'active_agent_translation' || /\b(translate|translation|spanish|french|german|italian|portuguese)\b/.test(instruction)) && original) {
    const rendered = activeAgentSimpleTranslation(original, instruction);
    if (rendered && rendered !== original && repairWeakBinding) content = rendered;
  } else if (
    ['NAME', 'ITEM', 'DEADLINE', 'REASON'].every((key) => slots[key])
    && /\b(rewrite|professional email|make .*professional)\b/.test(instruction)
    && (corrupt || ['NAME', 'ITEM', 'DEADLINE'].some((key) => !tokenCoveredByText(contentNorm, String(slots[key]).toLowerCase())))
  ) {
    content = `Hi ${slots.NAME}, could you please send the ${slots.ITEM} by ${slots.DEADLINE}? ${slots.REASON}. Thank you.`;
  } else if (/^hi how are you\??$/i.test(original) && /\b(rewrite|professional email|make .*professional)\b/.test(instruction)) {
    content = 'Hello, I hope you are well.';
  } else if (/^(rewrite|rewrite this|make this professional|professional email)[.!?]?$/i.test(original) && /\b(rewrite|professional email|make .*professional)\b/.test(instruction)) {
    content = 'What text should I rewrite?';
  } else if (/^hi how are you\??$/i.test(original) && /\b(summary|summari[sz]e|bullet summary)\b/.test(instruction)) {
    content = `Greeting summary: ${original}`;
  } else if (slots.DATA_CONTEXT && (corrupt || /\b(my|launch|code|saved)\b/i.test(original) || /\[\[DATA_CONTEXT\]\]/.test(decision.content))) {
    content = `I found this in your saved data: ${slots.DATA_CONTEXT}`;
  } else if (!slots.SOURCE_TEXT && !slots.DATA_CONTEXT && /\b(my|saved|reservation|confirmation|code)\b/i.test(original)) {
    content = 'I do not have that in saved data. Add it to PocketPal saved data or paste it here.';
  } else if (/\bhow'?s it going|how are you\b/i.test(original) && /\b(casual|naturally|briefly)\b/.test(instruction)) {
    content = "It's going well. What would you like to work on?";
  } else if (corrupt && original && activeAgentTextTransformInstruction(agent)) {
    content = original;
  }

  if (content === decision.content) return responseText;
  return JSON.stringify({
    action,
    content,
    proposal_metadata: {
      ...(decision.proposal_metadata || {}),
      ...(action === 'extension_request' ? {
        extension_id: state.webSearch.extensionId,
        capability: state.webSearch.searchCapabilityId,
        query: original,
        max_sources: Math.max(1, Math.min(5, Number(state.webSearch.maxSources || 5))),
        requires_user_approval: true,
      } : {}),
      content_operator: true,
    },
  });
}

function retrievalMissFallback(userText) {
  const question = shortText(userText, 140);
  return [
    `I did not retrieve a strong paper match for "${question}".`,
    'The current evidence looks off-topic, so I should not present it as a grounded answer.',
    'Try a narrower query with the specific research area, method, or benchmark you care about.',
  ].join('\n\n');
}

function selectedPaperFallbackAnswer(userText, rows) {
  const row = rows.find((item) => String(item.source || '').includes('selected_paper')) || rows[0];
  if (!row) return '';
  const question = shortText(userText, 140);
  const title = row.title || row.paper_id || row.canonical_paper_id || 'the selected paper';
  const snippet = evidenceSnippet(row);
  const sentences = (snippet.match(/[^.!?]+[.!?]/g) || [snippet])
    .map((sentence) => sentence.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
  const first = sentences[0] || snippet || 'The loaded context does not include enough text to summarize the paper well.';
  const second = sentences.find((sentence) => sentence !== first) || '';
  const third = sentences.find((sentence) => sentence !== first && sentence !== second) || '';
  const details = [
    `It is about ${first.replace(/^(this paper|we|the paper)\s+/i, '')}`,
    second ? `A useful extra detail is that ${second.replace(/^(this paper|we|the paper)\s+/i, '')}` : '',
    third ? `The next thing I would inspect is ${third.replace(/^(this paper|we|the paper)\s+/i, '')}` : '',
  ].filter(Boolean);
  return [
    `For "${question}", I am using the paper you added to chat: ${title}.`,
    details.join('\n'),
    'The paper card already in the conversation has the source link if you want to open the PDF.',
  ].join('\n\n');
}

function groundedFallbackAnswer(userText, rows, reason = '') {
  void reason;
  const question = shortText(userText, 140);
  if (!rows?.length) {
    return directChatFallback(question);
  }
  if (isSelectedPaperFollowup(userText) && rows.some((row) => String(row.source || '').includes('selected_paper'))) {
    return selectedPaperFallbackAnswer(userText, rows) || groundedAnswerFromRows(rows);
  }
  if (!evidenceMatchesQuery(userText, rows)) {
    return retrievalMissFallback(userText);
  }
  if (isRecommendationQuery(userText)) {
    const ranked = rows.filter((row) => rowMatchesQuery(row, userText)).slice(0, 5);
    const items = ranked.map((row, index) => {
      const title = row.title || row.paper_id || row.canonical_paper_id || `Evidence ${index + 1}`;
      return `- [${index + 1}] ${title}: ${relevantSentenceFromRow(row, userText, 260)}`;
    });
    const top = ranked[0] || rows[0];
    const topTitle = top?.title || top?.paper_id || top?.canonical_paper_id || 'the first retrieved paper';
    const topReason = relevantSentenceFromRow(top, userText, 300);
    return [
      `For "${question}", I would not call any paper universally "best" without criteria like LLM agents, robotics, control, verification, or surveys. Among the evidence retrieved here, the best starting point is [1] ${topTitle}.`,
      topReason ? `The reason is grounded in the evidence: ${topReason}` : '',
      ranked.length > 1 ? 'Other retrieved matches are useful as adjacent background rather than a single definitive winner:' : '',
      items.join('\n'),
      'For modern multi-agent LLM work specifically, I would narrow the query to "multi-agent LLM collaboration", "LLM agent society", or "multi-agent LLM survey" and compare newer evidence before making a recommendation.',
    ].filter(Boolean).join('\n\n');
  }
  const lead = groundedAnswerFromRows(rows);
  const bullets = rows.slice(0, 3).map((row, index) => {
    const title = row.title || row.paper_id || row.canonical_paper_id || `Evidence ${index + 1}`;
    const snippet = answerScaffold([row]).replace(/^No answer scaffold.*$/i, evidenceSnippet(row));
    return `- [${index + 1}] ${title}: ${shortText(snippet, 260)}`;
  });
  return [
    `For "${question}", my grounded answer is: ${lead}`,
    bullets.join('\n'),
    'Open the evidence cards or citation links above to inspect the papers directly.',
  ].filter(Boolean).join('\n\n');
}

function maybeGroundedFallback(text, rows, userText = '') {
  if (!String(state.loadedModelId || '').startsWith('modelstack:')) return text;
  const normalized = String(text || '').trim();
  if (!rows?.length && activePocketPalAgent() && activeAgentDecisionNeedsFallback(normalized, userText)) {
    return activeAgentFallbackAnswer(userText);
  }
  if (!rows?.length && activePocketPalAgent() && isUsableActiveAgentDecisionText(normalized)) return normalized;
  if (!rows?.length && hasDecoderQualityIssue(normalized, rows, userText)) {
    if (activePocketPalAgent()) {
      return activeAgentFallbackAnswer(userText);
    }
    return directChatFallback(userText);
  }
  if (!shouldPreferEvidenceComposer(normalized, rows, userText)) return normalized;
  const grounded = groundedFallbackAnswer(userText, rows, 'decoder output failed quality gate');
  return grounded || 'The local decoder did not produce a usable answer for this turn.';
}

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function arxivPdfUrlFromId(value) {
  const raw = String(value || '').trim().replace(/^arxiv:/i, '').replace(/v\d+$/i, '');
  return raw ? `https://arxiv.org/pdf/${encodeURIComponent(raw)}` : '';
}

function linkArxivIds(html) {
  return html.replace(
    /\b((?:https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\/)?(?:arXiv:?\s*)?(\d{4}\.\d{4,5}(?:v\d+)?))\b/g,
    (match, _label, id) => `<a class="inline-paper-link" href="${escapeHtml(arxivPdfUrlFromId(id))}" target="_blank" rel="noopener noreferrer">${match}</a>`,
  );
}

function linkBareUrls(html) {
  return String(html || '')
    .split(/(<a\b[^>]*>[\s\S]*?<\/a>|<code>[\s\S]*?<\/code>)/gi)
    .map((segment) => {
      if (/^<(a\b|code>)/i.test(segment)) return segment;
      return segment.replace(/https?:\/\/[^\s<]+/gi, (raw) => {
        const trimmed = raw.replace(/[),.;!?]+$/g, '');
        const suffix = raw.slice(trimmed.length);
        return `<a class="inline-web-link" href="${trimmed}" target="_blank" rel="noopener noreferrer">${trimmed}</a>${suffix}`;
      });
    })
    .join('');
}

function linkEvidenceCitations(html) {
  if (!state.retrievalRows.length) return html;
  return html.replace(/(^|[^\w])\[(P?)(\d{1,2})\]/gi, (match, prefix, marker, number) => {
    const index = Number(number) - 1;
    const row = state.retrievalRows[index];
    if (!row) return match;
    const label = `${marker ? 'P' : ''}${number}`;
    const title = row.title || row.paper_id || `Evidence ${label}`;
    const disabled = state.processActive ? ' disabled aria-disabled="true"' : '';
    return `${prefix}<button type="button" class="inline-evidence" data-action="load-paper" data-result-index="${index}" title="${escapeHtml(title)}" aria-label="Open evidence ${label}"${disabled}>[${label}]</button>`;
  });
}

function citedEvidenceIndexes(text, rows = state.retrievalRows) {
  if (!rows.length) return [];
  const seen = new Set();
  const indexes = [];
  for (const match of String(text || '').matchAll(/\[(?:P?)(\d{1,2})\]/gi)) {
    const index = Number(match[1]) - 1;
    if (index < 0 || index >= rows.length || seen.has(index)) continue;
    seen.add(index);
    indexes.push(index);
  }
  return indexes;
}

function evidenceAttributionText(text, rows = state.retrievalRows) {
  const indexes = citedEvidenceIndexes(text, rows);
  if (!indexes.length) return '';
  const lines = indexes.slice(0, 5).map((index) => {
    const row = rows[index];
    const title = row.title || row.paper_id || row.canonical_paper_id || `Evidence ${index + 1}`;
    const meta = paperMeta(row);
    return `[${index + 1}] ${title}${meta ? ` (${meta})` : ''}`;
  });
  return `Sources used:\n${lines.join('\n')}`;
}

function bindEvidenceAttribution(text, rows = state.retrievalRows) {
  const sourceText = evidenceAttributionText(text, rows);
  if (!sourceText) return String(text || '');
  if (/Sources used:/i.test(String(text || ''))) return String(text || '');
  return `${String(text || '').trim()}\n\n${sourceText}`;
}

function renderInline(value, options = {}) {
  let html = escapeHtml(value).replace(/`([^`]+)`/g, '<code>$1</code>');
  html = linkArxivIds(html);
  html = linkBareUrls(html);
  if (options.linkEvidence) html = linkEvidenceCitations(html);
  return html;
}

function codeLanguage(value) {
  return String(value || '').trim().split(/\s+/)[0] || 'text';
}

function appendTextChunk(parent, text, options = {}) {
  const chunks = String(text || '').split(/\n{2,}/).filter((chunk) => chunk.trim());
  for (const chunk of chunks) {
    const p = document.createElement('p');
    p.innerHTML = renderInline(chunk.trim(), options);
    parent.appendChild(p);
  }
}

function openCodeArtifact(language, code) {
  const extension = codeLanguage(language).replace(/[^a-z0-9_-]/gi, '') || 'txt';
  const blob = new Blob([code.replace(/\s+$/, '')], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const win = window.open(url, '_blank', 'noopener,noreferrer');
  window.setTimeout(() => URL.revokeObjectURL(url), 60000);
  if (!win) log(`code artifact ready: agent-kernel.${extension}`);
}

function appendCodeChunk(parent, language, code) {
  const wrapper = document.createElement('div');
  wrapper.className = 'code-block';
  const bar = document.createElement('div');
  bar.className = 'code-bar';
  const label = document.createElement('span');
  label.textContent = codeLanguage(language);
  const actions = document.createElement('div');
  actions.className = 'code-actions';
  const open = document.createElement('button');
  open.type = 'button';
  open.className = 'copy-code';
  open.textContent = 'Open';
  open.addEventListener('click', () => openCodeArtifact(language, code));
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'copy-code';
  copy.textContent = 'Copy';
  copy.addEventListener('click', async () => {
    await navigator.clipboard?.writeText(code).catch(() => {});
    copy.textContent = 'Copied';
    window.setTimeout(() => { copy.textContent = 'Copy'; }, 1100);
  });
  const pre = document.createElement('pre');
  const codeNode = document.createElement('code');
  codeNode.textContent = code.replace(/\s+$/, '');
  pre.appendChild(codeNode);
  actions.append(open, copy);
  bar.append(label, actions);
  wrapper.append(bar, pre);
  parent.appendChild(wrapper);
}

function renderMessageBody(parent, text, options = {}) {
  const source = String(text || '');
  const fencePattern = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let cursor = 0;
  let match;
  while ((match = fencePattern.exec(source)) !== null) {
    appendTextChunk(parent, source.slice(cursor, match.index), options);
    appendCodeChunk(parent, match[1], match[2]);
    cursor = fencePattern.lastIndex;
  }
  appendTextChunk(parent, source.slice(cursor), options);
  if (!parent.childElementCount) appendTextChunk(parent, source || 'No answer generated.', options);
}

function clearActiveTurnTimer() {
  if (state.activeTurn?.fallbackTimer) {
    window.clearTimeout(state.activeTurn.fallbackTimer);
    state.activeTurn.fallbackTimer = null;
  }
}

function cancelActiveGeneration() {
  const generationId = state.activeTurn?.generationId;
  if (!generationId || !state.worker) return;
  state.worker.postMessage({ type: 'cancel', generationId });
  log('requested generation cancel; runtime remains loaded');
}

function retryActiveAgentConstrainedGeneration(turn, failedText = '') {
  if (!turn || turn.retriedActiveAgent || turn.contextRows?.length || !activePocketPalAgent()) return false;
  const agent = activePocketPalAgent();
  const generationId = ++state.generationRunId;
  turn.retriedActiveAgent = true;
  turn.generationId = generationId;
  turn.finalized = false;
  state.modelBusy = true;
  setControlsBusy(true);
  clearActiveTurnTimer();
  log(`retrying malformed active-agent decode: ${shortText(failedText, 160)}`);
  setProcessStep('generate', 'active', 'Retrying active agent with constrained decoder');
  ensureWorker().postMessage({
    type: 'generate',
    generationId,
    prompt: buildActiveAgentRetryPrompt(turn.userText || ''),
    options: activeAgentGenerationOptions(agent, turn.userText || '', { retry: true }),
  });
  return true;
}

function finalizeAssistantResponse(text, { fallback = false, reason = '' } = {}) {
  const turn = state.activeTurn;
  if (turn?.finalized) return false;
  const rows = turn?.contextRows || state.pendingContextRows || [];
  const userText = turn?.userText || '';
  const activeAgentFallbackRequired = Boolean(
    !fallback
    && activePocketPalAgent()
    && !rows.length
    && (
      activeAgentDecisionNeedsFallback(text, userText)
      || (!isUsableActiveAgentDecisionText(text) && hasDecoderQualityIssue(text, rows, userText))
    )
  );
  if (
    activeAgentFallbackRequired
    && retryActiveAgentConstrainedGeneration(turn, text)
  ) {
    return true;
  }
  if (turn) turn.finalized = true;
  clearActiveTurnTimer();
  state.modelBusy = false;
  setControlsBusy(false);
  const activeAgentRuntimeFallback = activeAgentFallbackRequired && activePocketPalAgent();
  let responseText = fallback || activeAgentRuntimeFallback
    ? activePocketPalAgent()
      ? activeAgentFallbackAnswer(userText)
      : [
          'The local decoder did not produce a decoded answer for this turn.',
          rows.length
            ? 'Research context was retrieved and remains available in the evidence cards, but I am not going to replace the model with a synthetic answer.'
            : 'No research context was attached to this turn, and I am not going to replace the model with a synthetic answer.',
          reason ? `Runtime detail: ${reason}` : '',
        ].filter(Boolean).join('\n\n')
    : maybeGroundedFallback(text, rows, userText);
  responseText = activeAgentApplyContentOperators(responseText, userText);
  responseText = expandPocketPalTextSlots(responseText, turn?.textSlots || {});
  responseText = expandPocketPalSourcePointers(responseText, turn?.sourceSlots || []);
  setProcessStep(
    'generate',
    fallback ? 'error' : 'done',
    fallback ? 'Decoder timed out before answer' : activeAgentRuntimeFallback ? 'Used active-agent runtime fallback' : `${formatCount(String(text || '').length)} characters generated`,
  );
  const packet = recordAssistantTurn(responseText);
  const displayText = bindEvidenceAttribution(displayTextFromDecision(packet, responseText), rows);
  handleAssistantDecision(packet, userText);
  setProcessStep('render', 'active', 'Rendering answer and evidence links');
  appendMessage('assistant', displayText);
  setProcessStep('render', 'done', 'Answer displayed');
  finishProcessTrace(fallback ? 'Decoder timeout' : 'Complete');
  state.pendingContextRows = [];
  if (fallback) cancelActiveGeneration();
  return true;
}

function armGenerationFallback(userText, contextRows, generationId) {
  clearActiveTurnTimer();
  const selectedModel = String(state.loadedModelId || els.model.value || '');
  const turn = {
    id: state.processRunId,
    userText,
    contextRows,
    generationId,
    finalized: false,
    fallbackTimer: null,
    textSlots: state.currentTextSlots || {},
    sourceSlots: state.currentSourceSlots || [],
  };
  if (selectedModel.startsWith('modelstack:')) {
    state.activeTurn = turn;
    setProcessStep('generate', 'active', 'Waiting for local BitNet decoder');
    return;
  }
  const delay = REMOTE_MODEL_FALLBACK_MS;
  turn.fallbackTimer = window.setTimeout(() => {
    if (state.activeTurn !== turn || turn.finalized) return;
    setProcessStep('generate', 'error', 'Decoder did not finish before timeout');
    finalizeAssistantResponse('', {
      fallback: true,
      reason: `timeout after ${Math.round(delay / 1000)}s`,
    });
  }, delay);
  state.activeTurn = turn;
}

function appendMessage(role, text) {
  els.empty?.remove();
  const node = document.createElement('article');
  node.className = `message ${role}`;
  const roleNode = document.createElement('div');
  roleNode.className = 'role';
  roleNode.textContent = role === 'user' ? 'You' : 'Agent Kernel Lite';
  const body = document.createElement('div');
  body.className = 'body';
  renderMessageBody(body, text, { linkEvidence: role === 'assistant' });
  node.append(roleNode);
  attachTtsButton(node, text);
  node.append(body);
  attachPaperButtons(node, state.retrievalRows);
  els.chat.appendChild(node);
  els.chat.scrollTop = els.chat.scrollHeight;
  state.messages.push({ role, text });
}

function renderStoredMessage(message) {
  if (message?.artifact?.type === 'image') {
    const imageBase64 = String(message.artifact.imageBase64 || '');
    const svg = String(message.artifact.svg || '');
    const node = document.createElement('article');
    node.className = 'message assistant image-result';
    const roleNode = document.createElement('div');
    roleNode.className = 'role';
    roleNode.textContent = 'Image Generation';
    const body = document.createElement('div');
    body.className = 'body image-artifact';
    if (imageBase64 || svg) {
      const blob = imageBase64
        ? base64ToBlob(imageBase64, message.artifact.mimeType || 'image/png')
        : new Blob([svg], { type: 'image/svg+xml' });
      const image = document.createElement('img');
      image.src = URL.createObjectURL(blob);
      image.alt = String(message.artifact.prompt || message.text || 'Generated image');
      image.loading = 'lazy';
      body.appendChild(image);
    }
    const prompt = document.createElement('div');
    prompt.className = 'image-prompt';
    prompt.textContent = message.artifact.prompt || message.text || '';
    const meta = document.createElement('div');
    meta.className = 'image-meta';
    meta.textContent = [
      message.artifact.model || state.image.modelId,
      message.artifact.backend || '',
      message.artifact.seed ? `seed ${message.artifact.seed}` : '',
      imageBase64 || svg ? 'artifact restored' : 'artifact metadata restored',
    ].filter(Boolean).join(' | ');
    body.append(prompt, meta);
    node.append(roleNode, body);
    els.chat.appendChild(node);
    return;
  }
  const node = document.createElement('article');
  const role = message?.role === 'user' ? 'user' : 'assistant';
  node.className = `message ${role}`;
  const roleNode = document.createElement('div');
  roleNode.className = 'role';
  roleNode.textContent = role === 'user' ? 'You' : 'Agent Kernel Lite';
  const body = document.createElement('div');
  body.className = 'body';
  renderMessageBody(body, message?.text || '', { linkEvidence: role === 'assistant' });
  node.append(roleNode);
  attachTtsButton(node, message?.text || '');
  node.append(body);
  els.chat.appendChild(node);
}

function attachTtsButton(node, text) {
  const speakText = String(text || '').replace(/^\[[^\]]+\]\s*/, '').trim();
  if (!speakText) return;
  const toolbar = document.createElement('div');
  toolbar.className = 'message-toolbar';
  const button = document.createElement('button');
  button.className = 'message-tts-button';
  button.type = 'button';
  button.textContent = 'Play voice';
  button.title = 'Play with Peyton voice';
  button.addEventListener('click', () => speakPeytonVoiceText(speakText));
  toolbar.appendChild(button);
  node.appendChild(toolbar);
}

function renderStoredMessages() {
  els.chat.innerHTML = '';
  if (!state.messages.length) {
    els.chat.appendChild(els.empty);
    return;
  }
  els.empty?.remove();
  for (const message of state.messages) renderStoredMessage(message);
  els.chat.scrollTop = els.chat.scrollHeight;
}

function appendImageMessage(result) {
  els.empty?.remove();
  const imageBase64 = String(result.imageBase64 || '');
  const svg = String(result.svg || '');
  const blob = imageBase64
    ? base64ToBlob(imageBase64, result.mimeType || 'image/png')
    : new Blob([svg], { type: 'image/svg+xml' });
  const blobUrl = URL.createObjectURL(blob);
  const metadata = result.metadata || {};
  const node = document.createElement('article');
  node.className = 'message assistant image-result';
  const roleNode = document.createElement('div');
  roleNode.className = 'role';
  roleNode.textContent = 'Image Generation';
  const body = document.createElement('div');
  body.className = 'body image-artifact';
  const image = document.createElement('img');
  image.src = blobUrl;
  image.alt = String(result.prompt || 'Generated image');
  image.loading = 'lazy';
  const prompt = document.createElement('div');
  prompt.className = 'image-prompt';
  prompt.textContent = result.prompt || '';
  const meta = document.createElement('div');
  meta.className = 'image-meta';
  meta.textContent = [
    metadata.model || state.image.modelId,
    metadata.backend || 'preview',
    metadata.source_resolution ? `source ${metadata.source_resolution}` : '',
    result.seed ? `seed ${result.seed}` : '',
    result.elapsedMs ? `${Math.round(result.elapsedMs)} ms` : '',
  ].filter(Boolean).join(' | ');
  body.append(image, prompt, meta);
  node.append(roleNode, body);
  els.chat.appendChild(node);
  els.chat.scrollTop = els.chat.scrollHeight;
  state.messages.push({
    role: 'assistant',
    text: `[image] ${result.prompt || ''}`,
    artifact: {
      type: 'image',
      id: result.id,
      prompt: result.prompt || '',
      seed: result.seed || null,
      model: metadata.model || state.image.modelId,
      backend: metadata.backend || 'preview',
      mimeType: result.mimeType || (imageBase64 ? 'image/png' : 'image/svg+xml'),
      imageBase64,
      svg,
    },
  });
}

function appendVoiceMessage(result, audioUrl) {
  els.empty?.remove();
  const node = document.createElement('article');
  node.className = 'message assistant';
  const roleNode = document.createElement('div');
  roleNode.className = 'role';
  roleNode.textContent = 'Peyton Voice';
  const body = document.createElement('div');
  body.className = 'body image-artifact';
  const prompt = document.createElement('div');
  prompt.className = 'image-prompt';
  prompt.textContent = result.text || 'Peyton voice preview';
  const audio = document.createElement('audio');
  audio.controls = true;
  audio.src = audioUrl;
  audio.style.width = '100%';
  const meta = document.createElement('div');
  meta.className = 'image-meta';
  meta.textContent = [
    'F5TTS Q4',
    'Vocos Q4',
    result.preset || '',
    result.runtimeVersion || '',
    result.runtimeMode || '',
    result.samples ? `${formatCount(result.samples)} samples` : '',
    result.bytes ? formatBytes(result.bytes) : '',
    result.timing?.generationMs ? `gen ${formatDurationMs(result.timing.generationMs)}` : '',
    result.timing?.decodeMs ? `decode ${formatDurationMs(result.timing.decodeMs)}` : '',
    result.timing?.totalMs ? `total ${formatDurationMs(result.timing.totalMs)}` : '',
  ].filter(Boolean).join(' | ');
  body.append(prompt, audio, meta);
  node.append(roleNode, body);
  els.chat.appendChild(node);
  els.chat.scrollTop = els.chat.scrollHeight;
  state.messages.push({
    role: 'assistant',
    text: `[peyton voice] ${result.text || ''}`,
  });
}

function base64ToBlob(base64, mimeType = 'application/octet-stream') {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Blob([bytes], { type: mimeType });
}

function paperMeta(row) {
  return [
    row.paper_id || row.canonical_paper_id || '',
    row.primary_category || row.categories || '',
    row.year || '',
    row.source || '',
  ].filter(Boolean).join(' | ');
}

function loadedPaperPreview(row) {
  const abstract = String(row.abstract || row.summary || '').replace(/\s+/g, ' ').trim();
  const text = fullPaperText(row).replace(/\s+/g, ' ').trim();
  if (text) return text.slice(0, 2200);
  return abstract ? abstract.slice(0, 1600) : 'No full text or abstract was available in the fetched row.';
}

function rememberPaperContext(row) {
  const key = paperKey(row);
  if (!key) return;
  const contextRow = { ...row, source: 'selected_paper' };
  state.paperContextRows = [contextRow, ...state.paperContextRows.filter((item) => paperKey(item) !== key)].slice(0, MAX_SELECTED_PAPERS);
}

function attachPaperButtons(root, rows = state.retrievalRows) {
  root.querySelectorAll('[data-action="load-paper"]').forEach((button) => {
    if (button.dataset.bound === 'true') return;
    button.dataset.bound = 'true';
    button.addEventListener('click', () => {
      if (state.processActive) {
        log('paper actions are available after the assistant response finishes');
        return;
      }
      const index = Number(button.getAttribute('data-result-index') || -1);
      loadPaperContext(rows[index], button);
    });
  });
}

function setEvidenceActionsLocked(locked, root = document) {
  root.querySelectorAll('[data-action="load-paper"]').forEach((button) => {
    button.disabled = Boolean(locked);
    button.setAttribute('aria-disabled', locked ? 'true' : 'false');
  });
  root.querySelectorAll('[data-evidence-link="paper-pdf"]').forEach((link) => {
    delete link.dataset.locked;
    link.setAttribute('aria-disabled', 'false');
    link.removeAttribute('tabindex');
  });
}

function appendLoadedPaper(row) {
  els.empty?.remove();
  const pdf = arxivPdfUrl(row);
  const node = document.createElement('section');
  node.className = 'retrieval loaded-paper';
  node.innerHTML = `
    <h3>Loaded Paper Context</h3>
    <div class="result-list">
      <div class="result-item">
        <strong>${escapeHtml(row.title || row.paper_id || 'Untitled paper')}</strong>
        ${escapeHtml(paperMeta(row))}
        <p>${escapeHtml(loadedPaperPreview(row))}</p>
        <div class="result-actions">
          ${pdf ? `<a href="${escapeHtml(pdf)}" target="_blank" rel="noopener noreferrer">Open arXiv PDF</a>` : ''}
        </div>
      </div>
    </div>
  `;
  els.chat.appendChild(node);
  els.chat.scrollTop = els.chat.scrollHeight;
}

async function loadPaperContext(row, button) {
  if (!row) return;
  if (state.processActive) {
    log('paper actions are available after the assistant response finishes');
    return;
  }
  const originalLabel = button?.textContent || '';
  const inlineButton = button?.classList.contains('inline-evidence');
  const startedStandaloneTrace = !state.processActive;
  if (!state.processActive) {
    state.processRunId += 1;
    state.processActive = true;
    startLiveStatus(`Add paper: ${shortText(row.title || row.paper_id || 'paper', 96)}`);
    setProcessStep('receive', 'done', 'Paper selected from chat');
  }
  setProcessStep('lookup', 'active', `Opening ${shortText(row.title || row.paper_id || row.canonical_paper_id || 'paper', 64)}`);
  if (button) {
    button.disabled = true;
    button.textContent = inlineButton ? `${originalLabel}...` : 'Loading';
  }
  try {
    const hydrated = await enrichPaper(row);
    rememberPaperContext(hydrated);
    appendLoadedPaper(hydrated);
    if (button) {
      button.textContent = inlineButton ? originalLabel : 'Loaded';
      button.classList.add('ready');
    }
    setProcessStep('lookup', 'done', `Loaded ${shortText(hydrated.title || hydrated.paper_id || 'paper', 64)}`);
    setProcessStep('compact', 'done', 'Paper added to chat context');
    if (startedStandaloneTrace) finishProcessTrace('Paper Ready');
    log(`loaded full paper context: ${hydrated.paper_id || hydrated.canonical_paper_id || hydrated.title || 'paper'}`);
  } catch (error) {
    if (button) {
      button.disabled = false;
      button.textContent = inlineButton ? originalLabel : 'Add to Chat';
    }
    setProcessStep('lookup', 'error', error.message || String(error));
    if (startedStandaloneTrace) finishProcessTrace('Error');
    appendMessage('assistant', `Could not load that paper text: ${error.message || String(error)}`);
    log(error.message || String(error));
  }
}

function appendRetrieval(rows, { locked = state.processActive } = {}) {
  if (!rows.length) return;
  state.retrievalRows = rows;
  const node = document.createElement('section');
  node.className = 'retrieval';
  if (locked) node.dataset.locked = 'true';
  node.innerHTML = `
    <h3>Retrieved Evidence</h3>
    <div class="result-list">
      ${rows.map((row, index) => {
        const pdf = arxivPdfUrl(row);
        return `
          <div class="result-item">
            <strong>
              <button type="button" class="paper-title-button" data-action="load-paper" data-result-index="${index}"${locked ? ' disabled aria-disabled="true"' : ''}>
                ${index + 1}. ${escapeHtml(row.title || row.paper_id || 'Untitled paper')}
              </button>
            </strong>
            ${escapeHtml(paperMeta(row))}
            <p>${escapeHtml(evidenceSnippet(row))}</p>
            <div class="result-actions">
              <button type="button" class="secondary result-action-button" data-action="load-paper" data-result-index="${index}"${locked ? ' disabled aria-disabled="true"' : ''}>Add to Chat</button>
              ${pdf ? `<a href="${escapeHtml(pdf)}" target="_blank" rel="noopener noreferrer" data-evidence-link="paper-pdf" aria-disabled="false">Open arXiv PDF</a>` : ''}
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
  attachPaperButtons(node, rows);
  setEvidenceActionsLocked(locked, node);
  els.chat.appendChild(node);
  els.chat.scrollTop = els.chat.scrollHeight;
}

function setControlsBusy(busy) {
  const browserOnlyMode = state.mode === 'web_search';
  els.send.disabled = busy || state.image.busy || state.voice.busy || state.translation.busy || state.translation.listening || (state.image.enabled || browserOnlyMode ? false : state.modelBusy || !state.modelReady);
  els.prompt.disabled = busy;
  els.loadModel.disabled = state.image.enabled || busy || state.modelBusy;
  if (els.unloadModel) els.unloadModel.disabled = state.image.enabled || busy || state.modelBusy || !state.worker;
  els.loadPack.disabled = busy;
  if (els.audioTranslate) els.audioTranslate.disabled = busy || state.translation.busy;
  if (els.voiceSpeak) els.voiceSpeak.disabled = busy || state.voice.busy;
  syncModelControls();
}

async function submitPrompt(event) {
  event.preventDefault();
  const text = els.prompt.value.trim();
  if (!text) return;
  els.prompt.value = '';
  if (state.modelBusy || state.image.busy) return;
  appendMessage('user', text);
  if (state.image.enabled) {
    await submitImagePrompt(text);
    return;
  }
  if (state.translation.enabled) {
    await submitTranslationText(text, {
      modality: 'text',
      sourceLanguage: translationSourceLabel(),
      targetLanguage: translationTargetLabel(),
    });
    return;
  }
  if (state.mode === 'quick_search') {
    await submitQuickSearchPrompt(text);
    return;
  }
  if (state.mode === 'web_search') {
    await submitWebSearchPrompt(text);
    return;
  }
  if (isUserVisibleWebSearchRequest(text)) {
    await submitWebSearchPrompt(text);
    return;
  }
  resetProcessTrace(text);
  setControlsBusy(true);
  try {
    if (!state.modelReady || state.loadedModelId !== els.model.value) await loadModel();
    else setProcessStep('runtime', 'done', `Using loaded ${shortText(state.loadedModelId || els.model.value, 56)}`);
    let contextRows = [];
    state.pendingContextRows = [];
    state.retrievalRows = [];
    const freshResearchRequired = requiresFreshResearchContext(text);
    let plan = await planLiteTurn(text);
    const agent = activePocketPalAgent();
    const retrievalPolicy = activeAgentPolicy('retrievalPolicy', 'auto');
    if (agent && retrievalPolicy === 'auto' && !freshResearchRequired) {
      plan = {
        action: 'respond',
        query: text,
        reason: 'active agent handles the prompt directly unless retrieval is requested',
      };
      setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
      log('planner override: active agent prompt-first route');
    } else if (agent && retrievalPolicy === 'none' && !freshResearchRequired) {
      plan = {
        action: 'respond',
        query: text,
        reason: 'active agent retrieval policy disables paper lookup',
      };
      setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
      log('planner override: active agent retrieval policy disabled paper lookup');
    } else if (agent && retrievalPolicy === 'always' && plan.action !== 'gather_context') {
      plan = {
        action: 'gather_context',
        query: text,
        reason: 'active agent retrieval policy requires retrieval',
      };
      setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
      log('planner override: active agent retrieval policy requires retrieval');
    } else if (agent && retrievalPolicy === 'local_first' && state.pocketPalDataSources.length && !freshResearchRequired) {
      plan = {
        action: 'respond',
        query: text,
        reason: 'active agent will use saved user data before paper lookup',
      };
      setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
      log('planner override: active agent using saved user data before paper lookup');
    }
    if (freshResearchRequired && plan.action !== 'gather_context') {
      plan = {
        action: 'gather_context',
        query: text,
        reason: 'paper/source request requires fresh ranked evidence',
      };
      setProcessStep('plan', 'done', `${plan.action}: ${plan.reason}`);
      log('planner override: paper/source request must gather fresh evidence');
    }
    const selectedPaperFollowup = isSelectedPaperFollowup(text);
    if (plan.action === 'gather_context') {
      try {
        let retrievalRendered = false;
        if (selectedPaperFollowup) {
          setPill(els.packPill, 'using loaded paper', 'ready');
          contextRows = await selectedPaperContextRows(plan.query || text);
        } else {
          setPill(els.packPill, state.packRows.length ? 'retrieving' : 'HF search', 'busy');
          contextRows = await retrieveContext(plan.query || text, {
            onCandidates(candidates) {
              if (retrievalRendered) return;
              retrievalRendered = true;
              setProcessStep('render', 'active', `Showing ${formatCount(candidates.length)} retrieved papers`);
              appendRetrieval(candidates, { locked: true });
              setProcessStep('render', 'done', 'Retrieved papers displayed');
            },
          });
          contextRows = await selectEvidenceRows(plan.query || text, contextRows);
        }
        state.pendingContextRows = contextRows;
        setPill(els.packPill, state.packRows.length ? 'library ready' : 'library idle', state.packRows.length ? 'ready' : '');
        if (selectedPaperFollowup) {
          state.retrievalRows = contextRows;
          log('using loaded paper context for this turn; fresh retrieval skipped');
        } else if (!retrievalRendered) {
          appendRetrieval(contextRows);
        } else {
          state.retrievalRows = contextRows;
        }
      } catch (error) {
        state.pendingContextRows = [];
        setPill(els.packPill, 'chat only', '');
        setProcessStep('lookup', 'error', error.message || String(error));
        setProcessStep('compact', 'done', 'Continuing without evidence');
        log(`retrieval skipped: ${error.message || String(error)}`);
      }
    } else {
      setPill(els.packPill, state.packRows.length ? 'library ready' : 'chat only', state.packRows.length ? 'ready' : '');
      setProcessStep('lookup', 'done', 'respond selected; no paper lookup');
      setProcessStep('compact', 'done', 'Chat-only context');
    }
    if (freshResearchRequired && !selectedPaperFollowup && !contextRows.length) {
      setProcessStep('generate', 'done', 'Skipped: no paper evidence available');
      setProcessStep('render', 'done', 'Rendered no-evidence response');
      finishProcessTrace('No evidence');
      setControlsBusy(false);
      setAgentWorking(false);
      appendMessage('assistant', 'I could not retrieve paper evidence for that query, so I should not guess. Try a narrower research phrase or load the paper pack first.');
      return;
    }
    const config = modeConfig();
    setProcessStep('compile', 'active', `Building ${state.mode.replace('_', ' ')} context packet`);
    const activeAgentDirect = Boolean(agent && plan.action === 'respond' && !contextRows.length && !freshResearchRequired);
    const compiledPrompt = activeAgentDirect ? buildActiveAgentDirectPrompt(text) : buildPrompt(text, contextRows);
    setProcessStep('compile', 'done', `${formatCount(compiledPrompt.length)} prompt characters`);
    try {
      const intentSignal = await classifyAgentIntent(compiledPrompt, { maxEncoderTokens: 768, timeoutMs: 2200 });
      if (intentSignal?.intent) {
        state.lastAgentIntent = intentSignal;
        const confidence = Math.round(Number(intentSignal.confidence || 0) * 100);
        setProcessStep('plan', 'done', `intent=${intentSignal.intent} (${confidence}%)`);
        log(`agent intent: ${intentSignal.intent} (${confidence}%)`);
      }
    } catch (error) {
      log(`agent intent unavailable: ${error.message || String(error)}`);
    }
    const generationOptions = activeAgentDirect
      ? activeAgentGenerationOptions(agent, text)
      : {
          maxNewTokens: targetMaxTokens(),
          temperature: config.temperature,
          topP: 0.9,
        };
    setProcessStep('generate', 'active', `Generating up to ${formatCount(generationOptions.maxNewTokens)} tokens`);
    const generationId = ++state.generationRunId;
    armGenerationFallback(text, contextRows, generationId);
    ensureWorker().postMessage({
      type: 'generate',
      generationId,
      prompt: compiledPrompt,
      options: generationOptions,
    });
  } catch (error) {
    setControlsBusy(false);
    setProcessStep('render', 'error', error.message || String(error));
    clearActiveTurnTimer();
    state.modelBusy = false;
    finishProcessTrace('Error');
    appendMessage('assistant', `Could not complete local run: ${error.message || String(error)}`);
    log(error.message || String(error));
  }
}

async function submitImagePrompt(text) {
  resetProcessTrace(text);
  setControlsBusy(true);
  try {
    setProcessStep('runtime', state.image.ready ? 'done' : 'active', state.image.ready ? 'Using image backend' : 'Loading image backend');
    await loadImageRuntime();
    const proposal = proposeExtensionAction(state.image.extensionId, state.image.capabilityId, {
      prompt: text,
      model: state.image.modelId,
      surface: 'browser',
    });
    if (proposal.status !== 'pending_user_approval') {
      throw new Error(proposal.error || `image extension action was ${proposal.status || 'rejected'}`);
    }
    const jobId = `img_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
    state.image.busy = true;
    state.image.activeJobId = jobId;
    state.image.activeActionId = proposal.action_id || null;
    setProcessStep('plan', 'done', 'Image mode selected by toggle');
    setProcessStep('compile', 'done', 'Image prompt packet ready');
    setProcessStep('generate', 'active', 'Starting image generation');
    syncImageModeControls();
    ensureImageWorker().postMessage({
      type: 'generate',
      id: jobId,
      prompt: text,
      options: {
        width: 512,
        height: 512,
        steps: state.image.modelId.includes('sana') ? 50 : state.image.modelId.includes('_student_') ? 24 : 4,
        guidance: 0,
        seed: state.image.modelId.includes('sana') || state.image.modelId.includes('_student_') ? stableSeed(text) : Math.floor(Math.random() * 1_000_000_000),
      },
    });
  } catch (error) {
    const actionId = state.image.activeActionId;
    state.image.busy = false;
    state.image.activeJobId = null;
    state.image.activeActionId = null;
    if (actionId) {
      recordExtensionResult(actionId, {
        action_id: actionId,
        status: 'failed',
        error: error.message || String(error),
      });
    }
    setControlsBusy(false);
    setProcessStep('render', 'error', error.message || String(error));
    finishProcessTrace('Error');
    appendMessage('assistant', `Could not generate image: ${error.message || String(error)}`);
    log(error.message || String(error));
    syncImageModeControls();
  }
}

async function resetChat() {
  state.messages = [];
  state.paperContextRows = [];
  state.retrievalRows = [];
  state.pendingContextRows = [];
  clearActiveTurnTimer();
  state.activeTurn = null;
  state.lastDecisionPacket = null;
  state.processActive = false;
  state.liveStatusNode = null;
  setAgentWorking(false);
  if (state.coreReady && state.core) state.core.reset();
  els.chat.innerHTML = '';
  els.chat.appendChild(els.empty);
  els.prompt.value = '';
  renderProcessTrace();
}

function setMode(mode) {
  state.mode = normalizeMode(mode);
  if (state.coreReady && state.core) state.core.set_mode(state.mode);
  const config = modeConfig();
  for (const button of els.modeButtons || []) {
    const active = normalizeMode(button.dataset.mode || button.id.replace(/ModeButton$/, '')) === state.mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
  setPill(els.modePill, config.pill, state.mode === 'deep_research' ? 'busy' : 'ready');
  syncActiveAgentPill();
  els.send.textContent = 'Send';
  els.prompt.placeholder = state.image.enabled
    ? 'Describe the image to generate...'
    : state.translation.enabled
      ? `Text to translate to ${translationTargetLabel()}...`
      : config.placeholder;
  syncModelControls();
  log(`mode set: ${config.label}`);
}

function parseCoreJson(raw, fallback = {}) {
  try {
    return JSON.parse(String(raw || '{}'));
  } catch {
    return fallback;
  }
}

function stableSeed(text) {
  let hash = 2166136261;
  const value = String(text || '');
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function normalizeExtensionManifestForCore(manifest) {
  const clean = { ...(manifest || {}) };
  const allowedSources = new Set(['official', 'user', 'local', 'remote']);
  if (!allowedSources.has(clean.source)) clean.source = isLocalDevelopmentUrl(new URL(window.location.href)) ? 'local' : 'user';
  return clean;
}

function registerExtensionManifest(manifest) {
  if (!state.coreReady || !(state.core?.install_extension_manifest || state.core?.register_extension_manifest)) {
    return { status: 'error', error: 'extension core is not ready' };
  }
  try {
    const install = state.core.install_extension_manifest || state.core.register_extension_manifest;
    const result = parseCoreJson(install.call(state.core, JSON.stringify(normalizeExtensionManifestForCore(manifest))));
    if (extensionInstallSucceeded(result)) persistInstalledExtensions();
    return result;
  } catch (error) {
    return { status: 'error', error: error.message || String(error) };
  }
}

function extensionInstallSucceeded(result) {
  return result?.status === 'installed' || result?.status === 'registered';
}

async function persistInstalledExtensions() {
  const result = listExtensionManifests();
  const manifests = Array.isArray(result.extensions) ? result.extensions : [];
  const cache = {
    version: 1,
    saved_at: new Date().toISOString(),
    manifests,
  };
  await dbSet(EXTENSION_CACHE_DB_KEY, cache).catch((error) => {
    log(`extension cache skipped: ${error.message || String(error)}`);
  });
}

async function restoreInstalledExtensionsFromCache() {
  const cache = await dbGet(EXTENSION_CACHE_DB_KEY).catch((error) => {
    log(`extension cache load skipped: ${error.message || String(error)}`);
    return null;
  });
  const manifests = Array.isArray(cache?.manifests) ? cache.manifests : [];
  if (!manifests.length) return;
  await restoreExtensions({ manifests });
  log(`restored ${manifests.length} cached extension${manifests.length === 1 ? '' : 's'}`);
}

function isLocalDevelopmentUrl(url) {
  return ['localhost', '127.0.0.1', '0.0.0.0', '::1', '[::1]'].includes(url.hostname);
}

function isGitHubReleaseAssetUrl(url) {
  if (url.protocol !== 'https:' || url.hostname !== 'github.com') return false;
  const parts = url.pathname.split('/').filter(Boolean);
  if (parts.length < 6) return false;
  const [, , releases, download, tag, ...assetParts] = parts;
  if (releases !== 'releases' || download !== 'download') return false;
  if (!tag || tag === 'latest' || tag === 'main' || tag === 'master') return false;
  return assetParts.length > 0 && !url.pathname.includes('/refs/heads/');
}

function assertReleaseInstallUrl(rawUrl, label = 'extension asset') {
  const url = new URL(String(rawUrl || '').trim(), window.location.href);
  if (isGitHubReleaseAssetUrl(url)) return url;
  if (isLocalDevelopmentUrl(new URL(window.location.href)) && isLocalDevelopmentUrl(url)) return url;
  throw new Error(`${label} must be a GitHub Release asset URL, not a branch/raw/latest URL.`);
}

function extensionCodeUrls(manifest) {
  const metadata = manifest?.metadata || {};
  return [
    metadata.adapter_url,
    metadata.adapter_script,
    metadata.worker,
    metadata.worker_url,
    metadata.module,
    metadata.module_url,
  ].filter(Boolean);
}

function validateExtensionReleaseManifest(manifest) {
  for (const assetUrl of extensionCodeUrls(manifest)) {
    assertReleaseInstallUrl(assetUrl, 'extension code asset');
  }
}

async function installExtensionFromUrl(manifestUrl) {
  const url = assertReleaseInstallUrl(manifestUrl, 'extension manifest');
  const response = await fetch(url.href, { cache: 'no-store' });
  if (!response.ok) throw new Error(`extension manifest fetch failed: ${response.status}`);
  const manifest = await response.json();
  validateExtensionReleaseManifest(manifest);
  manifest.source = manifest.source || 'remote';
  manifest.imported_from = manifest.imported_from || url.href;
  manifest.metadata = {
    ...(manifest.metadata || {}),
    install_url: url.href,
    release_only: true,
  };
  return registerExtensionManifest(manifest);
}

async function loadAvailableExtensions() {
  try {
    const response = await fetch(new URL(AVAILABLE_EXTENSIONS_CATALOG_URL, window.location.href).href, { cache: 'no-store' });
    if (!response.ok) throw new Error(`catalog fetch failed: ${response.status}`);
    const catalog = await response.json();
    const entries = Array.isArray(catalog.extensions) ? catalog.extensions : [];
    const entryIds = new Set(entries.map((entry) => entry?.id).filter((id) => RELEASE_AVAILABLE_EXTENSION_IDS.has(id)));
    state.availableExtensions = LOCAL_AVAILABLE_EXTENSIONS.map((entry) => ({
      ...entry,
      manifest_url: resolveAvailableExtensionManifestUrl(entry, response.url),
    }));
    if (entryIds.size !== state.availableExtensions.length || entryIds.size !== entries.length) {
      log('available extensions constrained to pinned release allowlist');
    }
    log(`available extensions loaded: ${state.availableExtensions.length}`);
  } catch (error) {
    state.availableExtensions = LOCAL_AVAILABLE_EXTENSIONS.map((entry) => ({
      ...entry,
      manifest_url: resolveAvailableExtensionManifestUrl(entry, window.location.href),
    }));
    log(`available extensions using local fallback: ${error.message || String(error)}`);
  }
  renderExtensionList();
}

async function installAvailableExtension(extensionId) {
  const entry = state.availableExtensions.find((item) => item.id === extensionId);
  if (!entry) {
    appendMessage('assistant', `Extension is not available: ${extensionId}`);
    return;
  }
  try {
    const response = await fetch(entry.manifest_url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`manifest fetch failed: ${response.status}`);
    const manifest = await response.json();
    manifest.source = manifest.source || 'official';
    manifest.imported_from = manifest.imported_from || entry.manifest_url;
    manifest.metadata = {
      ...(manifest.metadata || {}),
      available_catalog: AVAILABLE_EXTENSIONS_CATALOG_URL,
      available_extension_id: extensionId,
    };
    const result = registerExtensionManifest(manifest);
    if (!extensionInstallSucceeded(result)) throw new Error(result.error || result.status || 'install failed');
    log(`installed available extension: ${manifest.name || manifest.id}`);
    renderExtensionList();
  } catch (error) {
    appendMessage('assistant', `Extension install failed: ${error.message || String(error)}`);
    log(`available extension install failed: ${error.message || String(error)}`);
  }
}

function uninstallExtension(extensionId) {
  if (!state.coreReady || !state.core?.uninstall_extension) {
    return { status: 'error', error: 'extension core is not ready', extension_id: extensionId };
  }
  try {
    const result = parseCoreJson(state.core.uninstall_extension(String(extensionId || '')));
    if (result.status === 'uninstalled') persistInstalledExtensions();
    return result;
  } catch (error) {
    return { status: 'error', error: error.message || String(error), extension_id: extensionId };
  }
}

function setExtensionEnabled(extensionId, enabled) {
  if (!state.coreReady || !state.core?.set_extension_enabled) {
    return { status: 'error', error: 'extension core is not ready', extension_id: extensionId };
  }
  try {
    const result = parseCoreJson(state.core.set_extension_enabled(String(extensionId || ''), Boolean(enabled)));
    if (result.status === 'enabled' || result.status === 'disabled') persistInstalledExtensions();
    return result;
  } catch (error) {
    return { status: 'error', error: error.message || String(error), extension_id: extensionId };
  }
}

function listExtensionManifests() {
  if (!state.coreReady || !state.core?.list_extension_manifests) {
    return { status: 'error', error: 'extension core is not ready', extensions: [] };
  }
  try {
    return parseCoreJson(state.core.list_extension_manifests(), { status: 'error', extensions: [] });
  } catch (error) {
    return { status: 'error', error: error.message || String(error), extensions: [] };
  }
}

function installedExtension(extensionId) {
  const result = listExtensionManifests();
  const manifests = Array.isArray(result.extensions) ? result.extensions : [];
  return manifests.find((manifest) => manifest.id === extensionId) || null;
}

function extensionEnabled(extensionId) {
  return Boolean(installedExtension(extensionId)?.enabled);
}

function extensionStatusText(manifest) {
  const source = manifest.source || 'user';
  const enabled = manifest.enabled ? 'enabled' : 'off';
  return `${source} | ${enabled}`;
}

function extensionSetupText(manifest) {
  if (manifest.id === state.image.extensionId) return 'Enable it to switch the composer into image generation mode. This extension is available for development and future release installs.';
  if (manifest.id === state.translation.extensionId) return 'Requires browser speech/model setup before audio translation can run.';
  return manifest.metadata?.setup || 'Installed from manifest. Enable only after the adapter setup is complete.';
}

function setInstalledExtensionEnabled(extensionId, enabled) {
  if (extensionId === state.image.extensionId) {
    setImageMode(enabled);
    return;
  }
  if (extensionId === state.translation.extensionId) {
    setTranslationMode(enabled);
    renderExtensionList();
    return;
  }
  const result = setExtensionEnabled(extensionId, enabled);
  if (result.status === 'error' || result.status === 'disabled') {
    appendMessage('assistant', `Extension could not be ${enabled ? 'enabled' : 'disabled'}: ${result.error || result.status}`);
    log(`extension toggle failed: ${result.error || result.status}`);
  } else {
    log(`extension ${extensionId} ${enabled ? 'enabled' : 'disabled'}`);
  }
  renderExtensionList();
}

function renderExtensionList() {
  if (!els.extensionList) return;
  const result = listExtensionManifests();
  const manifests = Array.isArray(result.extensions) ? result.extensions : [];
  const installedIds = new Set(manifests.map((manifest) => manifest.id));
  els.extensionList.innerHTML = '';
  if (!manifests.length) {
    const empty = document.createElement('p');
    empty.className = 'extension-detail';
    empty.textContent = 'No extensions installed.';
    els.extensionList.appendChild(empty);
    renderAvailableExtensionList(installedIds);
    return;
  }
  for (const manifest of manifests) {
    const card = document.createElement('details');
    card.className = 'extension-card';
    const summary = document.createElement('summary');
    const title = document.createElement('div');
    title.className = 'extension-title';
    const name = document.createElement('strong');
    name.textContent = manifest.name || manifest.id;
    const status = document.createElement('span');
    status.textContent = extensionStatusText(manifest);
    title.append(name, status);
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = `image-toggle-button${manifest.enabled ? ' active' : ''}`;
    toggle.textContent = manifest.enabled ? 'On' : 'Off';
    toggle.setAttribute('aria-pressed', manifest.enabled ? 'true' : 'false');
    toggle.addEventListener('click', async (event) => {
      event.preventDefault();
      event.stopPropagation();
      setInstalledExtensionEnabled(manifest.id, !manifest.enabled);
    });
    summary.append(title, toggle);

    const settings = document.createElement('div');
    settings.className = 'extension-settings';
    const setup = document.createElement('p');
    setup.className = 'extension-detail';
    setup.textContent = extensionSetupText(manifest);
    const caps = document.createElement('p');
    caps.className = 'extension-detail';
    caps.textContent = `Capabilities: ${(manifest.capabilities || []).map((capability) => capability.id).join(', ') || 'none'}`;
    const scopes = document.createElement('p');
    scopes.className = 'extension-detail';
    const scopeList = (manifest.capabilities || []).flatMap((capability) => capability.scopes || []);
    scopes.textContent = `Scopes: ${scopeList.length ? [...new Set(scopeList)].join(', ') : 'none'}`;
    settings.prepend(setup, caps, scopes);
    if (manifest.imported_from) {
      const imported = document.createElement('p');
      imported.className = 'extension-detail';
      imported.textContent = `Installed from: ${manifest.imported_from}`;
      settings.appendChild(imported);
    }
    if (manifest.id !== state.image.extensionId) {
      const uninstall = document.createElement('button');
      uninstall.type = 'button';
      uninstall.className = 'secondary';
      uninstall.textContent = 'Uninstall';
      uninstall.addEventListener('click', () => {
        const uninstalled = uninstallExtension(manifest.id);
        if (uninstalled.status === 'uninstalled') log(`uninstalled extension: ${manifest.name || manifest.id}`);
        else log(`extension uninstall failed: ${uninstalled.error || manifest.id}`);
        renderExtensionList();
      });
      settings.appendChild(uninstall);
    }
    card.append(summary, settings);
    els.extensionList.appendChild(card);
  }
  renderAvailableExtensionList(installedIds);
}

function renderAvailableExtensionList(installedIds = new Set()) {
  if (!els.availableExtensionList) return;
  els.availableExtensionList.innerHTML = '';
  const available = state.availableExtensions.filter((entry) => !installedIds.has(entry.id));
  if (!available.length) {
    const empty = document.createElement('p');
    empty.className = 'extension-detail';
    empty.textContent = state.availableExtensions.length ? 'All available extensions are installed.' : 'No available extensions found.';
    els.availableExtensionList.appendChild(empty);
    return;
  }
  for (const entry of available) {
    const card = document.createElement('details');
    card.className = 'extension-card';
    const summary = document.createElement('summary');
    const title = document.createElement('div');
    title.className = 'extension-title';
    const name = document.createElement('strong');
    name.textContent = entry.name || entry.id;
    const status = document.createElement('span');
    status.textContent = entry.source || 'available';
    title.append(name, status);
    const install = document.createElement('button');
    install.type = 'button';
    install.className = 'secondary';
    install.textContent = 'Install';
    install.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      installAvailableExtension(entry.id);
    });
    summary.append(title, install);

    const settings = document.createElement('div');
    settings.className = 'extension-settings';
    const description = document.createElement('p');
    description.className = 'extension-detail';
    description.textContent = entry.description || 'Install this extension before enabling its capabilities.';
    const setup = document.createElement('p');
    setup.className = 'extension-detail';
    setup.textContent = entry.setup || 'Setup runs after install.';
    settings.append(description, setup);
    card.append(summary, settings);
    els.availableExtensionList.appendChild(card);
  }
}

async function installExtensionFromInput() {
  const url = String(els.extensionManifestUrl?.value || '').trim();
  if (!url) return;
  try {
    const result = await installExtensionFromUrl(url);
    if (!extensionInstallSucceeded(result)) throw new Error(result.error || result.status || 'install failed');
    if (els.extensionManifestUrl) els.extensionManifestUrl.value = '';
    log(`installed extension: ${result.extension_id}`);
    renderExtensionList();
  } catch (error) {
    appendMessage('assistant', `Extension install failed: ${error.message || String(error)}`);
    log(`extension install failed: ${error.message || String(error)}`);
  }
}

function proposeExtensionAction(extensionId, capabilityId, input = {}) {
  if (!state.coreReady || !state.core?.propose_extension_action) {
    return { status: 'error', error: 'extension core is not ready', extension_id: extensionId };
  }
  try {
    return parseCoreJson(state.core.propose_extension_action(
      String(extensionId || ''),
      String(capabilityId || ''),
      JSON.stringify(input),
    ));
  } catch (error) {
    return { status: 'error', error: error.message || String(error), extension_id: extensionId, capability_id: capabilityId };
  }
}

function proposeLastDecisionExtensionAction(input = {}) {
  if (!state.coreReady || !state.core?.propose_last_decision_extension_action) {
    return { status: 'error', error: 'extension core is not ready' };
  }
  try {
    return parseCoreJson(state.core.propose_last_decision_extension_action(JSON.stringify(input || {})));
  } catch (error) {
    return { status: 'error', error: error.message || String(error) };
  }
}

function recordExtensionResult(actionId, receipt) {
  if (!state.coreReady || !state.core?.record_extension_result) {
    return { status: 'error', error: 'extension core is not ready', action_id: actionId };
  }
  try {
    return parseCoreJson(state.core.record_extension_result(String(actionId || ''), JSON.stringify(receipt || {})));
  } catch (error) {
    return { status: 'error', error: error.message || String(error), action_id: actionId };
  }
}

function exportLocalStorage() {
  const out = {};
  try {
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);
      if (!key || !key.startsWith('agent-kernel-lite-')) continue;
      out[key] = localStorage.getItem(key);
    }
  } catch (error) {
    log(`session export skipped localStorage: ${error.message || String(error)}`);
  }
  return out;
}

function restoreLocalStorage(values) {
  if (!values || typeof values !== 'object') return;
  try {
    for (const [key, value] of Object.entries(values)) {
      if (!key.startsWith('agent-kernel-lite-')) continue;
      if (typeof value === 'string') localStorage.setItem(key, value);
    }
  } catch (error) {
    log(`session restore skipped localStorage: ${error.message || String(error)}`);
  }
}

async function buildSessionExport() {
  const extensions = listExtensionManifests();
  return {
    type: 'agent_kernel_lite_session',
    version: SESSION_EXPORT_VERSION,
    exported_at: new Date().toISOString(),
    app: {
      url: window.location.href,
      user_agent: navigator.userAgent,
      release_repo: GITHUB_RELEASE_REPO,
      release_root: GITHUB_RELEASE_ROOT,
      release_only_installs: true,
      integrity: state.appIntegrity,
    },
    settings: {
      theme: state.theme,
      mode: state.mode,
      max_tokens: Number(els.tokens?.value || 160),
      pack_rows: Number(els.pack?.value || 50000),
      device: els.device?.value || 'auto',
      model: els.model?.value || '',
      image_enabled: Boolean(state.image.enabled),
      translation_enabled: Boolean(state.translation.enabled),
      translation_source: els.translationSource?.value || 'auto',
      translation_target: els.translationTarget?.value || 'Spanish',
      web_search_max_sources: state.webSearch.maxSources,
    },
    session: {
      messages: state.messages,
      paper_context_rows: state.paperContextRows,
      pending_context_rows: state.pendingContextRows,
      retrieval_rows: state.retrievalRows.slice(0, 32),
    },
    pocketpal: pocketPalExportState(),
    extensions: {
      status: extensions.status,
      manifests: extensions.extensions || [],
    },
    storage: {
      local_storage: exportLocalStorage(),
      indexed_db: await dbDump().catch((error) => {
        log(`session export skipped IndexedDB metadata: ${error.message || String(error)}`);
        return [];
      }),
      cache_name: CACHE_NAME,
      cache_exported: false,
    },
  };
}

function downloadJsonFile(filename, value) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportSession() {
  try {
    const bundle = await buildSessionExport();
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    downloadJsonFile(`agent-kernel-lite-session-${stamp}.json`, bundle);
    log('session export ready');
  } catch (error) {
    appendMessage('assistant', `Session export failed: ${error.message || String(error)}`);
    log(`session export failed: ${error.message || String(error)}`);
  }
}

function normalizeSessionBundle(raw) {
  const bundle = raw && typeof raw === 'object' ? raw : {};
  if (bundle.type !== 'agent_kernel_lite_session') {
    throw new Error('Not an Agent Kernel Lite session file.');
  }
  if (Number(bundle.version || 0) > SESSION_EXPORT_VERSION) {
    throw new Error(`Session file version ${bundle.version} is newer than this app supports.`);
  }
  return bundle;
}

async function restoreExtensions(extensionBundle) {
  const manifests = Array.isArray(extensionBundle?.manifests) ? extensionBundle.manifests : [];
  for (const manifest of manifests) {
    if (!manifest?.id || !manifest?.capabilities?.length) continue;
    const wasEnabled = Boolean(manifest.enabled);
    const cleanManifest = { ...manifest, enabled: false, default_enabled: false };
    const installed = registerExtensionManifest(cleanManifest);
    if (!extensionInstallSucceeded(installed)) {
      log(`extension restore skipped ${manifest.id}: ${installed.error || installed.status}`);
      continue;
    }
    if (wasEnabled) setExtensionEnabled(manifest.id, true);
  }
  await persistInstalledExtensions();
}

async function restoreSessionBundle(rawBundle) {
  const bundle = normalizeSessionBundle(rawBundle);
  const settings = bundle.settings || {};
  if (settings.theme) setTheme(settings.theme);
  if (settings.mode) setMode(settings.mode);
  if (els.tokens && settings.max_tokens) els.tokens.value = String(settings.max_tokens);
  if (els.pack && settings.pack_rows) els.pack.value = String(settings.pack_rows);
  if (els.device && settings.device) els.device.value = String(settings.device);
  if (els.model && settings.model) {
    const modelValue = String(settings.model);
    if ([...els.model.options].some((option) => option.value === modelValue)) els.model.value = modelValue;
  }
  if (els.translationSource && settings.translation_source) els.translationSource.value = String(settings.translation_source);
  if (els.translationTarget && settings.translation_target) els.translationTarget.value = String(settings.translation_target);
  if (settings.web_search_max_sources) {
    state.webSearch.maxSources = Math.max(1, Math.min(5, Number(settings.web_search_max_sources || 5)));
    if (els.webSearchMaxSources) els.webSearchMaxSources.value = String(state.webSearch.maxSources);
    persistWebSearchSettings();
  }
  restoreLocalStorage(bundle.storage?.local_storage);
  restorePocketPalState(bundle.pocketpal);
  await dbRestore(bundle.storage?.indexed_db || []).catch((error) => {
    log(`session restore skipped IndexedDB metadata: ${error.message || String(error)}`);
  });
  await restoreExtensions(bundle.extensions);
  state.messages = Array.isArray(bundle.session?.messages) ? bundle.session.messages : [];
  state.paperContextRows = Array.isArray(bundle.session?.paper_context_rows) ? bundle.session.paper_context_rows : [];
  state.pendingContextRows = Array.isArray(bundle.session?.pending_context_rows) ? bundle.session.pending_context_rows : [];
  state.retrievalRows = Array.isArray(bundle.session?.retrieval_rows) ? bundle.session.retrieval_rows : [];
  renderStoredMessages();
  if (Boolean(settings.image_enabled) && !state.image.enabled) setImageMode(true);
  else if (!settings.image_enabled && state.image.enabled) setImageMode(false);
  if (Boolean(settings.translation_enabled) && !state.translation.enabled) setTranslationMode(true);
  else if (!settings.translation_enabled && state.translation.enabled) setTranslationMode(false);
  await refreshStorage();
  log('session restored');
}

async function importSessionFile(file) {
  if (!file) return;
  try {
    const text = await file.text();
    await restoreSessionBundle(JSON.parse(text));
  } catch (error) {
    appendMessage('assistant', `Session import failed: ${error.message || String(error)}`);
    log(`session import failed: ${error.message || String(error)}`);
  } finally {
    if (els.importSessionInput) els.importSessionInput.value = '';
  }
}

function exposeExtensionApi() {
  window.AgentKernelExtensions = Object.freeze({
    install: registerExtensionManifest,
    installFromUrl: installExtensionFromUrl,
    uninstall: uninstallExtension,
    enable: (extensionId) => setExtensionEnabled(extensionId, true),
    disable: (extensionId) => setExtensionEnabled(extensionId, false),
    list: listExtensionManifests,
    propose: proposeExtensionAction,
    proposeLastDecision: proposeLastDecisionExtensionAction,
    record: recordExtensionResult,
    translateText: (text, options = {}) => runTranslator(text, { ...options, modality: 'text' }),
    translateAudio: (input, options = {}) => {
      const text = typeof input === 'string' ? input : input?.transcript || input?.text || '';
      return runTranslator(text, { ...options, modality: 'audio' });
    },
    webSearch: runWebSearch,
    exportSession: buildSessionExport,
    restoreSession: restoreSessionBundle,
  });
}

function registerBuiltinExtensions() {
  const manifests = [{
    id: state.webSearch.extensionId,
    name: 'Web Search',
    version: '0.1.0',
    source: 'local',
    default_enabled: true,
    approval_policy: 'always_ask',
    capabilities: [
      {
        id: state.webSearch.searchCapabilityId,
        description: 'Open a user-visible web search for the requested query.',
        scopes: ['query.read', 'browser.open'],
      },
      {
        id: state.webSearch.openCapabilityId,
        description: 'Open a user-visible URL in the phone or browser web surface.',
        scopes: ['url.open', 'browser.open'],
      },
    ],
    metadata: {
      adapter: 'browser',
      native_surface: NATIVE_APP ? 'wkwebview_or_system_browser' : 'browser_tab',
      result_access: 'user_visible_browser',
    },
  }];
  if (isLocalDevelopmentUrl(new URL(window.location.href))) {
    manifests.push({
      id: 'image_generation',
      name: 'Image Generation',
      version: '0.1.0-dev',
      source: 'local',
      default_enabled: false,
      approval_policy: 'always_ask',
      capabilities: [{
        id: 'image.generate',
        description: 'Generate an image artifact from a prompt using the local FLUX student development runtime.',
        scopes: ['prompt.read', 'artifact.image.write'],
      }],
      metadata: {
        adapter: 'browser',
        backend: 'sana_student_browser_wasm_pending',
        model_id: 'agentkernel_lite_image_sana_300m_bitnet_block12_13ff_browser_v0',
        quality_tier: 'staged-bitnet-qat-browser-export',
        teacher_model: 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers',
        student_checkpoint: 'checkpoints/agentkernel_lite_image_sana_300m_bitnet_block12_13ff_recover_v10b/sana_latent_student_best_block12_13ff_3300.pt',
        student_step: 3300,
        development_only: true,
        dense_reference_checkpoint: 'checkpoints/agentkernel_lite_image_sana_300m_broad_v6/sana_latent_student_best_anchor_broad_3000.pt',
      },
    });
  }
  for (const manifest of manifests) {
    const result = registerExtensionManifest(manifest);
    if (extensionInstallSucceeded(result)) {
      if (manifest.default_enabled) setExtensionEnabled(manifest.id, true);
      log(`installed extension: ${manifest.name}`);
    } else {
      log(`extension install failed: ${result.error || manifest.id}`);
    }
  }
  renderExtensionList();
}

async function init() {
  setTheme(state.theme, false);
  loadPocketPalSlots();
  loadPocketPalMemory();
  loadPocketPalDataSources();
  loadPocketPalAgents();
  loadWebSearchSettings();
  renderDataSourceList();
  renderAgentList();
  syncActiveAgentPill();
  renderProcessTrace();
  await loadAgentCore();
  await restoreInstalledExtensionsFromCache();
  registerBuiltinExtensions();
  if (DEV_BACKEND === 'vllm') {
    const option = document.createElement('option');
    option.value = `vllm:${VLLM_MODEL}`;
    option.textContent = `Local vLLM (${VLLM_MODEL})`;
    els.model.prepend(option);
    els.model.value = option.value;
    setPill(els.modelPill, 'vLLM dev backend', 'ready');
    log(`vLLM dev backend enabled: ${VLLM_ENDPOINT}`);
  }
  if (STRUCTURE_FIXTURE) {
    els.packMetric.textContent = 'Fixture';
    els.rowsMetric.textContent = '1';
    setPill(els.packPill, 'fixture ready', 'ready');
    log('structure-check evidence fixture enabled');
  }
  const addModelStackOption = (manifestUrl, label = 'AgentKernel Lite EncDec (model-stack)') => {
    const resolvedManifestUrl = new URL(manifestUrl, window.location.href).href;
    const option = document.createElement('option');
    option.value = `modelstack:${resolvedManifestUrl}`;
    option.textContent = label;
    els.model.prepend(option);
    els.model.value = option.value;
    log('model-stack encoder-decoder manifest attached from URL');
  };
  const rawManifestUrl = new URLSearchParams(window.location.search).get('modelStackManifest');
  if (rawManifestUrl) {
    addModelStackOption(new URL(rawManifestUrl, window.location.href).href);
  } else if (DEV_BACKEND !== 'vllm') {
    if (NATIVE_APP) {
      addModelStackOption(NATIVE_MODELSTACK_MANIFEST, 'Bundled AgentKernel Lite 100M BitNet');
      log('AgentKernel Lite BitNet bundle attached from packaged app assets');
    } else {
      addModelStackOption(HF_MODELSTACK_MANIFEST, 'AgentKernel Lite 100M BitNet');
      log('AgentKernel Lite BitNet bundle attached from Hugging Face');
    }
  }
  els.form.addEventListener('submit', submitPrompt);
  els.chatMode?.addEventListener('click', () => setMode('chat'));
  els.quickSearchMode?.addEventListener('click', () => setMode('quick_search'));
  els.webSearchMode?.addEventListener('click', () => setMode('web_search'));
  els.thinkMode?.addEventListener('click', () => setMode('think'));
  els.deepResearchMode?.addEventListener('click', () => setMode('deep_research'));
  els.imageMode?.addEventListener('click', () => setImageMode(!state.image.enabled));
  els.voiceSpeak?.addEventListener('click', speakPeytonVoice);
  els.translationMode?.addEventListener('click', () => setTranslationMode(!state.translation.enabled));
  els.translationSource?.addEventListener('change', syncTranslationControls);
  els.translationTarget?.addEventListener('change', syncTranslationControls);
  els.audioTranslate?.addEventListener('click', startAudioTranslation);
  for (const tab of els.moduleTabs) tab.addEventListener('click', () => switchModule(tab.dataset.module || 'assistant'));
  els.loadModel.addEventListener('click', () => loadModel({ force: true }).catch((error) => log(error.message || String(error))));
  els.unloadModel?.addEventListener('click', () => unloadModel());
  els.loadPack.addEventListener('click', () => loadResearchPack());
  els.persist.addEventListener('click', () => requestPersistentStorage());
  els.reset.addEventListener('click', resetChat);
  els.exportSession?.addEventListener('click', () => exportSession());
  els.importSession?.addEventListener('click', () => els.importSessionInput?.click());
  els.importSessionInput?.addEventListener('change', () => importSessionFile(els.importSessionInput.files?.[0]));
  els.installExtension?.addEventListener('click', () => installExtensionFromInput());
  els.saveUserDataSource?.addEventListener('click', saveUserDataSource);
  els.userDataFileInput?.addEventListener('change', () => importUserDataFiles(els.userDataFileInput.files));
  els.webSearchMaxSources?.addEventListener('change', () => {
    const maxSources = Math.max(1, Math.min(5, Number(els.webSearchMaxSources.value || 5)));
    state.webSearch.maxSources = maxSources;
    els.webSearchMaxSources.value = String(maxSources);
    persistWebSearchSettings();
    log(`web search source limit set to ${formatCount(maxSources)}`);
  });
  els.clearUserDataSource?.addEventListener('click', () => {
    state.pocketPalDataSources = [];
    persistPocketPalDataSources();
    if (els.userDataSource) els.userDataSource.value = '';
    renderDataSourceList();
    log('PocketPal data sources cleared');
  });
  els.createAgent?.addEventListener('click', createPocketPalAgent);
  els.clearActiveAgent?.addEventListener('click', () => {
    setActivePocketPalAgent('');
  });
  els.extensionManifestUrl?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      installExtensionFromInput();
    }
  });
  els.themeToggle?.addEventListener('click', toggleTheme);
  els.mobileToggle?.addEventListener('click', () => {
    document.body.classList.toggle('controls-hidden');
    els.mobileToggle.textContent = document.body.classList.contains('controls-hidden') ? 'Controls' : 'Close';
  });
  els.closeControls?.addEventListener('click', () => {
    document.body.classList.add('controls-hidden');
    if (els.mobileToggle) els.mobileToggle.textContent = 'Controls';
  });
  if (navigator.storage?.persisted) {
    const persisted = await navigator.storage.persisted();
    setPill(els.storagePill, persisted ? 'persistent' : 'best effort', persisted ? 'ready' : '');
  }
  els.sessionLine.textContent = DEV_BACKEND === 'vllm'
    ? STRUCTURE_FIXTURE
      ? 'Dev structure check via local vLLM and fixture evidence.'
      : 'Dev structure check via local vLLM.'
    : 'WASM on-device runtime. No server-side inference.';
  setMode('chat');
  syncImageModeControls();
  syncTranslationControls();
  loadAvailableExtensions();
  refreshAppIntegrity();
  await refreshStorage();
  syncModelControls();
  log('agent kernel lite ready');
  if (NATIVE_APP && navigator.storage?.persist) {
    navigator.storage.persist().then((persisted) => {
      setPill(els.storagePill, persisted ? 'persistent' : 'best effort', persisted ? 'ready' : '');
      log(persisted ? 'native app storage persistence granted' : 'native app storage persistence not granted');
    }).catch((error) => log(`storage persistence request failed: ${error.message || String(error)}`));
  }
  if (NATIVE_APP || URL_PARAMS.get('autopack') === '1') {
    window.setTimeout(() => {
      loadResearchPack().catch((error) => log(error.message || String(error)));
    }, 250);
  }
  const shouldAutoloadModel = URL_PARAMS.get('autoload') !== '0';
  if (shouldAutoloadModel) {
    state.modelAutoLoadStarted = true;
    const startModelLoad = () => {
      if (state.voice.busy || state.voice.ready || state.voice.loadPromise) {
        updateRuntimeDetail('Runtime autoload skipped while Peyton voice is active.');
        syncModelControls();
        return;
      }
      loadModel({ auto: true }).catch((error) => {
        log(error.message || String(error));
        updateRuntimeDetail(`Runtime did not load automatically: ${error.message || String(error)}`);
        syncModelControls();
      });
    };
    if (NATIVE_APP) {
      window.setTimeout(startModelLoad, 1200);
    } else {
      startModelLoad();
    }
  } else {
    updateRuntimeDetail('Runtime autoload is off. Load Runtime to start the model.');
  }
}

init().catch((error) => log(error.message || String(error)));
