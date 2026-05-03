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
const DEV_BACKEND = String(URL_PARAMS.get('backend') || '').trim().toLowerCase();
const DEVICE_PARAM = String(URL_PARAMS.get('device') || '').trim().toLowerCase();
const VLLM_ENDPOINT = String(URL_PARAMS.get('vllmEndpoint') || '').trim();
const VLLM_MODEL = String(URL_PARAMS.get('vllmModel') || 'Qwen/Qwen3.5-9B').trim();
const STRUCTURE_FIXTURE = URL_PARAMS.get('structureFixture') === '1';
const HF_DATASET_SEARCH_ENABLED = URL_PARAMS.get('hfSearch') === '1';
const HF_MODELSTACK_MANIFEST = 'https://huggingface.co/PeytonT/agentkernel-lite-100m-bitnet/resolve/main/manifest.json';
const NEURAL_MEMORY_PACK_URL = String(URL_PARAMS.get('neuralMemoryPack') || '').trim();
const NEURAL_MEMORY_ENABLED = URL_PARAMS.get('neuralMemory') === '1' || Boolean(NEURAL_MEMORY_PACK_URL);
const THEME_STORAGE_KEY = 'agent-kernel-lite-theme';
const CACHE_NAME = 'agent-kernel-lite-v12';
const DB_NAME = 'agent-kernel-lite-db-v1';
const DB_STORE = 'metadata';
const SESSION_EXPORT_VERSION = 1;
const EXTENSION_CACHE_DB_KEY = 'installed_extensions_v1';
const COMPUTER_PAIRING_DB_KEY = 'computer_bridge_pairing_v1';
const COMPUTER_WORKSPACE_STORAGE_KEY = 'agent-kernel-lite-computer-workspace';
const COMPUTER_BRIDGE_URL_STORAGE_KEY = 'agent-kernel-lite-computer-bridge-url';
const COMPUTER_PROVIDER_STORAGE_KEY = 'agent-kernel-lite-computer-provider';
const COMPUTER_MODEL_STORAGE_KEY = 'agent-kernel-lite-computer-model';
const COMPUTER_DEFAULT_BRIDGE_URL = 'http://127.0.0.1:45731';
const COMPUTER_DEFAULT_RELAY_URL = `${window.location.origin}/agent_kernel/api/relay`;
const COMPUTER_ROUTE_ID_PATTERN = /^route_[A-Za-z0-9_-]{24,96}$/;
const COMPUTER_BRIDGE_PROTOCOL = 'agent-kernel-computer-bridge/v1';
const COMPUTER_SLASH_COMMANDS = [
  { name: 'diff', description: 'show git diff for this workspace' },
  { name: 'status', description: 'show current terminal configuration' },
  { name: 'model', description: 'show or set the Codex model', args: true },
  { name: 'models', description: 'show or set the Codex model', args: true },
  { name: 'permissions', description: 'show bridge workspace and sandbox permissions' },
  { name: 'approvals', description: 'show bridge workspace and sandbox permissions' },
  { name: 'clear', description: 'clear this terminal view and start fresh' },
  { name: 'new', description: 'start a new terminal for this workspace' },
  { name: 'review', description: 'review current changes and find issues', args: true },
  { name: 'init', description: 'create AGENTS.md instructions for Codex' },
  { name: 'compact', description: 'ask Codex to summarize the conversation' },
  { name: 'copy', description: 'copy the last Codex response' },
  { name: 'mention', description: 'insert @ for a file mention', localOnly: true },
  { name: 'ps', description: 'list active computer terminals' },
  { name: 'stop', description: 'cancel the running Codex turn' },
  { name: 'quit', description: 'return to the terminal list' },
  { name: 'exit', description: 'return to the terminal list' },
  { name: 'help', description: 'show available commands' },
];
const CODEX_INIT_PROMPT = `Generate a file named AGENTS.md that serves as a contributor guide for this repository.
Your goal is to produce a clear, concise, and well-structured document with descriptive headings and actionable explanations for each section.
Follow the outline below, but adapt as needed. Add sections if relevant, and omit those that do not apply to this project.

Document Requirements

- Title the document "Repository Guidelines".
- Use Markdown headings (#, ##, etc.) for structure.
- Keep the document concise. 200-400 words is optimal.
- Keep explanations short, direct, and specific to this repository.
- Provide examples where helpful, such as commands, directory paths, and naming patterns.
- Maintain a professional, instructional tone.

Recommended Sections

Project Structure & Module Organization

- Outline the project structure, including where the source code, tests, and assets are located.

Build, Test, and Development Commands

- List key commands for building, testing, and running locally.
- Briefly explain what each command does.

Coding Style & Naming Conventions

- Specify indentation rules, language-specific style preferences, and naming patterns.
- Include any formatting or linting tools used.

Testing Guidelines

- Identify testing frameworks and coverage requirements.
- State test naming conventions and how to run tests.

Commit & Pull Request Guidelines

- Summarize commit message conventions found in the project's Git history.
- Outline pull request requirements, such as descriptions, linked issues, and screenshots.

Optional: add other sections if relevant, such as Security & Configuration Tips, Architecture Overview, or Agent-Specific Instructions.`;
const CODEX_PAIRING_DB_KEY = COMPUTER_PAIRING_DB_KEY;
const CODEX_WORKSPACE_STORAGE_KEY = COMPUTER_WORKSPACE_STORAGE_KEY;
const CODEX_BRIDGE_URL_STORAGE_KEY = COMPUTER_BRIDGE_URL_STORAGE_KEY;
const CODEX_DEFAULT_BRIDGE_URL = COMPUTER_DEFAULT_BRIDGE_URL;
const CODEX_BRIDGE_PROTOCOL = COMPUTER_BRIDGE_PROTOCOL;
const GITHUB_RELEASE_REPO = 'peytontolbert/agent_kernel_lite';
const GITHUB_RELEASE_TAG = 'v12';
const GITHUB_RELEASE_ROOT = `https://github.com/${GITHUB_RELEASE_REPO}/releases/download`;
const PINNED_GITHUB_RELEASE_ROOT = `${GITHUB_RELEASE_ROOT}/${GITHUB_RELEASE_TAG}`;
const AVAILABLE_EXTENSIONS_CATALOG_URL = './extensions/catalog.json';
const LOCAL_AVAILABLE_EXTENSIONS = [
  {
    id: 'computer_use',
    name: 'Computer Use',
    source: 'official',
    manifest: './extensions/computer_use.json',
    description: 'Pair with Agent Kernel Desktop or the local computer bridge to orchestrate Codex, Claude Code, and Cursor sessions.',
    setup: 'Requires pairing the local computer bridge before enabling a provider.',
  },
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
  utilityGenerationRequests: new Map(),
  paperContextRows: [],
  retrievalRows: [],
  pendingContextRows: [],
  lastDecisionPacket: null,
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
    modelId: URL_PARAMS.get('imageModel') || 'agentkernel_lite_image_bitdit_hf_cifar_distilled_v1',
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
  codex: {
    extensionId: 'computer_use',
    startCapabilityId: 'computer.session.start',
    sendCapabilityId: 'computer.session.send',
    statusCapabilityId: 'computer.session.status',
    cancelCapabilityId: 'computer.session.cancel',
    diffCapabilityId: 'computer.diff.read',
    bridgeUrl: localStorage.getItem(COMPUTER_BRIDGE_URL_STORAGE_KEY) || COMPUTER_DEFAULT_BRIDGE_URL,
    provider: localStorage.getItem(COMPUTER_PROVIDER_STORAGE_KEY) || 'codex',
    model: localStorage.getItem(COMPUTER_MODEL_STORAGE_KEY) || '',
    providers: [],
    paired: false,
    pairing: null,
    busy: false,
    activeActionId: null,
    sessionId: '',
    activeSessionId: '',
    sessions: [],
    pollTimers: new Map(),
    eventCount: 0,
    lastStatus: null,
    seq: 0,
    broker: {
      enabled: URL_PARAMS.get('computerBroker') === '1' || HASH_PARAMS.get('computerBroker') === '1',
      token: URL_PARAMS.get('computerBrokerToken') || HASH_PARAMS.get('computerBrokerToken') || '',
      connected: false,
      origin: '',
      bridgeUrl: '',
      target: null,
      nextRequestId: 1,
      pending: new Map(),
      readyAt: 0,
    },
  },
  processRunId: 0,
  generationRunId: 0,
  processActive: false,
  liveStatusNode: null,
  activeTurn: null,
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
  thinkMode: document.getElementById('thinkModeButton'),
  deepResearchMode: document.getElementById('deepResearchModeButton'),
  modeButtons: [...document.querySelectorAll('.mode-button')],
  imageMode: document.getElementById('imageModeButton'),
  imageModeDetail: document.getElementById('imageModeDetail'),
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
  commandPalette: document.getElementById('commandPalette'),
  computerConsole: document.getElementById('computerConsole'),
  computerConsoleButton: document.getElementById('computerConsoleButton'),
  computerWorkspace: document.getElementById('computerWorkspaceInput'),
  computerProvider: document.getElementById('computerProviderSelect'),
  computerStart: document.getElementById('computerStartButton'),
  computerNewTerminalForm: document.getElementById('computerNewTerminalForm'),
  computerCreateTerminal: document.getElementById('computerCreateTerminalButton'),
  computerCancelTerminal: document.getElementById('computerCancelTerminalButton'),
  computerSessionList: document.getElementById('computerSessionList'),
  computerSessionDetail: document.getElementById('computerSessionDetail'),
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
  const computerSessionMode = document.body.classList.contains('computer-session-open');
  const codexBusy = Boolean(state.codex.busy);
  if (els.loadModel) {
    els.loadModel.disabled = imageMode || computerSessionMode || loading || state.processActive;
    els.loadModel.textContent = loaded ? 'Reload Runtime' : loading ? 'Loading...' : 'Load Runtime';
  }
  if (els.unloadModel) {
    els.unloadModel.disabled = imageMode || computerSessionMode || loading || state.processActive || !state.worker;
  }
  if (els.send) {
    els.send.disabled = state.processActive || imageBusy || translationBusy || codexBusy || (computerSessionMode ? !state.codex.paired : imageMode ? !state.image.ready : loading || !loaded);
    els.send.textContent = imageMode
      ? imageBusy ? 'Generating...' : 'Generate'
      : state.translation.enabled
        ? state.translation.busy ? 'Translating...' : 'Translate'
        : computerSessionMode
          ? codexBusy ? 'Running...' : 'Send'
          : 'Send';
  }
  syncTranslationControls();
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

function ensureImageWorker() {
  if (state.image.worker) return state.image.worker;
  state.image.worker = new Worker('./js/image-worker.js?v=20260502-image-preview', { type: 'module' });
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
    loadImageRuntime().catch((error) => {
      setExtensionEnabled(state.image.extensionId, false);
      state.image.enabled = false;
      state.image.ready = false;
      setPill(els.imagePill, 'image error', 'error');
      appendMessage('assistant', `Image mode could not start: ${error.message || String(error)}`);
      log(`image mode failed: ${error.message || String(error)}`);
      syncImageModeControls();
    });
    log('image generation mode enabled');
  } else {
    log('image generation mode disabled');
  }
  syncImageModeControls();
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

async function dbDelete(key) {
  const db = await openDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.objectStore(DB_STORE).delete(key);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

function bytesToBase64(bytes) {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function base64ToBytes(value) {
  const binary = atob(String(value || ''));
  const out = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) out[index] = binary.charCodeAt(index);
  return out;
}

function utf8Bytes(value) {
  return new TextEncoder().encode(String(value || ''));
}

function utf8Text(bytes) {
  return new TextDecoder().decode(bytes);
}

function codexBridgeUrl() {
  return String(state.codex.bridgeUrl || CODEX_DEFAULT_BRIDGE_URL).replace(/\/+$/, '');
}

function normalizeCodexBridgeInput(value) {
  const clean = String(value || '').trim().replace(/\/+$/, '');
  if (!clean) return CODEX_DEFAULT_BRIDGE_URL;
  if (COMPUTER_ROUTE_ID_PATTERN.test(clean)) {
    return `${COMPUTER_DEFAULT_RELAY_URL}/bridge/${encodeURIComponent(clean)}`;
  }
  if (/^bridge\/route_[A-Za-z0-9_-]{24,96}$/i.test(clean)) {
    return `${COMPUTER_DEFAULT_RELAY_URL}/${clean}`;
  }
  if (/^(\d{1,3}\.){3}\d{1,3}:\d+$/i.test(clean) || /^[A-Za-z0-9-]+\.local:\d+$/i.test(clean)) {
    return `http://${clean}`;
  }
  if (/^https:\/\/((10|127)\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|localhost(?::|$)|[A-Za-z0-9-]+\.local(?::|$))/i.test(clean)) {
    return clean.replace(/^https:\/\//i, 'http://');
  }
  return clean;
}

function codexWorkspace() {
  return String(localStorage.getItem(CODEX_WORKSPACE_STORAGE_KEY) || '').trim();
}

function setCodexWorkspace(value) {
  localStorage.setItem(CODEX_WORKSPACE_STORAGE_KEY, String(value || '').trim());
}

function computerProvider() {
  return String(state.codex.provider || 'codex').trim() || 'codex';
}

function setComputerProvider(value) {
  const provider = String(value || 'codex').trim() || 'codex';
  state.codex.provider = provider;
  localStorage.setItem(COMPUTER_PROVIDER_STORAGE_KEY, provider);
}

function computerModel() {
  return String(state.codex.model || '').trim();
}

function setComputerModel(value) {
  const model = String(value || '').trim();
  state.codex.model = model;
  if (model) localStorage.setItem(COMPUTER_MODEL_STORAGE_KEY, model);
  else localStorage.removeItem(COMPUTER_MODEL_STORAGE_KEY);
}

function setCodexBridgeUrl(value) {
  const clean = normalizeCodexBridgeInput(value);
  state.codex.bridgeUrl = clean;
  localStorage.setItem(CODEX_BRIDGE_URL_STORAGE_KEY, clean);
}

async function loadCodexPairing() {
  state.codex.pairing = await dbGet(CODEX_PAIRING_DB_KEY).catch(() => null);
  state.codex.paired = Boolean(state.codex.pairing?.grant_id && state.codex.pairing?.browser_private_jwk && state.codex.pairing?.bridge_public_jwk);
  state.codex.seq = Number(state.codex.pairing?.seq || 0);
  return state.codex.pairing;
}

async function saveCodexPairing(pairing) {
  state.codex.pairing = pairing;
  state.codex.paired = Boolean(pairing?.grant_id);
  state.codex.seq = Number(pairing?.seq || state.codex.seq || 0);
  await dbSet(CODEX_PAIRING_DB_KEY, pairing);
}

async function clearCodexPairing() {
  const hadPairing = Boolean(state.codex.pairing?.grant_id);
  if (hadPairing) {
    await sendCodexBridgeMessage('computer.grant.revoke').catch((error) => {
      log(`Computer bridge revoke skipped: ${error.message || String(error)}`);
    });
  }
  state.codex.pairing = null;
  state.codex.paired = false;
  state.codex.seq = 0;
  await dbDelete(CODEX_PAIRING_DB_KEY).catch(() => {});
  renderExtensionList();
  renderComputerConsole();
}

async function importEcdhPublicKey(jwk) {
  return crypto.subtle.importKey('jwk', jwk, { name: 'ECDH', namedCurve: 'P-256' }, true, []);
}

async function importEcdhPrivateKey(jwk) {
  return crypto.subtle.importKey('jwk', jwk, { name: 'ECDH', namedCurve: 'P-256' }, true, ['deriveBits']);
}

async function deriveCodexAesKey(pairing = state.codex.pairing) {
  if (!pairing?.browser_private_jwk || !pairing?.bridge_public_jwk || !pairing?.grant_id) {
    throw new Error('Computer bridge is not paired.');
  }
  const privateKey = await importEcdhPrivateKey(pairing.browser_private_jwk);
  const publicKey = await importEcdhPublicKey(pairing.bridge_public_jwk);
  const shared = await crypto.subtle.deriveBits({ name: 'ECDH', public: publicKey }, privateKey, 256);
  const hkdfKey = await crypto.subtle.importKey('raw', shared, 'HKDF', false, ['deriveKey']);
  return crypto.subtle.deriveKey(
    {
      name: 'HKDF',
      hash: 'SHA-256',
      salt: utf8Bytes(pairing.grant_id),
      info: utf8Bytes(pairing.origin || window.location.origin),
    },
    hkdfKey,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt'],
  );
}

async function encryptCodexPayload(payload) {
  const pairing = state.codex.pairing || await loadCodexPairing();
  const key = await deriveCodexAesKey(pairing);
  const seq = Number(pairing.seq || 0) + 1;
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const aad = utf8Bytes(`${pairing.grant_id}:${seq}`);
  const ciphertext = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: nonce, additionalData: aad },
    key,
    utf8Bytes(JSON.stringify(payload)),
  );
  pairing.seq = seq;
  await saveCodexPairing(pairing);
  return {
    protocol: CODEX_BRIDGE_PROTOCOL,
    grant_id: pairing.grant_id,
    seq,
    nonce: bytesToBase64(nonce),
    ciphertext: bytesToBase64(new Uint8Array(ciphertext)),
  };
}

async function decryptCodexEnvelope(envelope) {
  const pairing = state.codex.pairing || await loadCodexPairing();
  const key = await deriveCodexAesKey(pairing);
  const aad = utf8Bytes(`${pairing.grant_id}:${Number(envelope.seq || 0)}`);
  const plaintext = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: base64ToBytes(envelope.nonce), additionalData: aad },
    key,
    base64ToBytes(envelope.ciphertext),
  );
  return JSON.parse(utf8Text(new Uint8Array(plaintext)));
}

function bridgeHostIsPrivate(url) {
  const host = url.hostname.toLowerCase();
  if (host === 'localhost' || host.endsWith('.local')) return true;
  if (/^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host)) return true;
  const match = host.match(/^172\.(\d+)\./);
  return Boolean(match && Number(match[1]) >= 16 && Number(match[1]) <= 31);
}

function brokerOriginAllowed(origin) {
  try {
    const url = new URL(origin);
    return url.protocol === 'http:' && bridgeHostIsPrivate(url);
  } catch (_) {
    return false;
  }
}

function setupComputerBrokerTransport() {
  if (!window.opener) return;
  window.addEventListener('message', (event) => {
    const message = event.data || {};
    if (message.type === 'agent-kernel-computer-broker-ready') {
      if (!brokerOriginAllowed(event.origin)) return;
      if (state.codex.broker.token && message.token !== state.codex.broker.token) return;
      if (!state.codex.broker.enabled && !state.codex.broker.token) return;
      state.codex.broker.enabled = true;
      state.codex.broker.connected = true;
      state.codex.broker.origin = event.origin;
      state.codex.broker.bridgeUrl = String(message.bridgeUrl || event.origin).replace(/\/+$/, '');
      state.codex.broker.target = event.source;
      state.codex.broker.readyAt = Date.now();
      setCodexBridgeUrl(state.codex.broker.bridgeUrl);
      log(`Computer broker connected: ${state.codex.broker.bridgeUrl}`);
      renderExtensionList();
      return;
    }
    if (message.type === 'agent-kernel-computer-broker-response') {
      const requestId = String(message.requestId || '');
      const pending = state.codex.broker.pending.get(requestId);
      if (!pending || event.origin !== pending.origin) return;
      clearTimeout(pending.timeout);
      state.codex.broker.pending.delete(requestId);
      if (message.error) {
        pending.reject(new Error(message.error));
      } else {
        pending.resolve({
          ok: Boolean(message.ok),
          status: Number(message.status || 0),
          payload: message.payload || {},
        });
      }
    }
  });
  const sayHello = () => {
    if (!window.opener || state.codex.broker.connected) return;
    window.opener.postMessage({
      type: 'agent-kernel-computer-broker-hello',
      token: state.codex.broker.token,
    }, '*');
  };
  sayHello();
  window.setInterval(sayHello, 1000);
}

async function computerBridgeRequest(path, options = {}) {
  const broker = state.codex.broker;
  if (broker.enabled && broker.connected && broker.target) {
    const requestId = String(broker.nextRequestId++);
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        broker.pending.delete(requestId);
        reject(new Error('Computer broker request timed out.'));
      }, Number(options.timeout || 120000));
      broker.pending.set(requestId, { resolve, reject, timeout, origin: broker.origin });
      broker.target.postMessage({
        type: 'agent-kernel-computer-broker-request',
        requestId,
        path,
        method: options.method || 'GET',
        body: options.body || null,
      }, broker.origin);
    });
  }
  const response = await fetch(`${codexBridgeUrl()}${path}`, {
    method: options.method || 'GET',
    headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined,
    cache: 'no-store',
  });
  const payload = await response.json();
  return { ok: response.ok, status: response.status, payload };
}

function bridgeFailureHint(url, error) {
  if (window.location.protocol === 'https:' && url.protocol === 'http:' && bridgeHostIsPrivate(url)) {
    return 'Likely browser policy: this HTTPS page is trying to fetch an HTTP private/LAN bridge. Allow mixed/private-network access for this site, or test from localhost on the same computer.';
  }
  if (url.protocol === 'https:') {
    return 'Likely scheme mismatch: the desktop bridge is plain HTTP. Use http:// for localhost or LAN bridge URLs.';
  }
  if (/Failed to fetch|Load failed|NetworkError/i.test(error?.message || '')) {
    return 'The browser did not expose an HTTP response. Check that the bridge is running, the URL/IP is correct, the firewall allows the port, and the browser is not blocking the request.';
  }
  return 'The browser exposed a response-level error. Check the status and response body above.';
}

async function diagnoseCodexBridge() {
  const startedAt = new Date().toISOString();
  const rawBridgeUrl = codexBridgeUrl();
  const rows = [
    `Bridge diagnostics (${startedAt})`,
    `App origin: ${window.location.origin}`,
    `Secure context: ${window.isSecureContext ? 'yes' : 'no'}`,
    `WebCrypto: ${window.crypto?.subtle ? 'yes' : 'no'}`,
    `Bridge URL: ${rawBridgeUrl}`,
    `Broker enabled: ${state.codex.broker.enabled ? 'yes' : 'no'}`,
    `Broker connected: ${state.codex.broker.connected ? 'yes' : 'no'}`,
    `Broker origin: ${state.codex.broker.origin || '(none)'}`,
  ];
  let url;
  try {
    url = new URL(rawBridgeUrl);
    rows.push(`Bridge origin: ${url.origin}`);
    rows.push(`Bridge protocol: ${url.protocol}`);
    rows.push(`Bridge host: ${url.host}`);
    rows.push(`Private/LAN host: ${bridgeHostIsPrivate(url) ? 'yes' : 'no'}`);
  } catch (error) {
    rows.push(`URL parse: failed (${error.message || String(error)})`);
    return rows.join('\n');
  }
  if (state.codex.broker.connected) {
    rows.push('Broker GET /health');
    try {
      const response = await computerBridgeRequest('/health', { timeout: 15000 });
      rows.push(`Broker health status: ${response.status}`);
      rows.push(`Broker health body: ${JSON.stringify(response.payload || {}).slice(0, 900)}`);
    } catch (error) {
      rows.push(`Broker health failed: ${error.name || 'Error'}: ${error.message || String(error)}`);
    }
  }
  const healthUrl = `${rawBridgeUrl}/health`;
  rows.push(`GET ${healthUrl}`);
  try {
    const response = await fetch(healthUrl, { cache: 'no-store' });
    rows.push(`Health status: ${response.status} ${response.statusText || ''}`.trim());
    rows.push(`CORS visible: yes`);
    const text = await response.text();
    rows.push(`Health body: ${text.slice(0, 900) || '(empty)'}`);
  } catch (error) {
    rows.push(`Health fetch failed: ${error.name || 'Error'}: ${error.message || String(error)}`);
    rows.push(`Likely cause: ${bridgeFailureHint(url, error)}`);
  }
  return rows.join('\n');
}

async function codexBridgeHealth() {
  const response = await computerBridgeRequest('/health');
  if (!response.ok) throw new Error(`Computer bridge health failed: ${response.status}`);
  const result = response.payload;
  state.codex.providers = Array.isArray(result.providers) ? result.providers : state.codex.providers;
  state.codex.bridgeHealth = result;
  return result;
}

async function pairCodexBridge() {
  if (!window.crypto?.subtle) throw new Error('Codex pairing requires WebCrypto support.');
  const keyPair = await crypto.subtle.generateKey(
    { name: 'ECDH', namedCurve: 'P-256' },
    true,
    ['deriveBits'],
  );
  const browserPublicJwk = await crypto.subtle.exportKey('jwk', keyPair.publicKey);
  const browserPrivateJwk = await crypto.subtle.exportKey('jwk', keyPair.privateKey);
  const startResponse = await computerBridgeRequest('/pairing/start', {
    method: 'POST',
    body: {
      origin: window.location.origin,
      browser_public_jwk: browserPublicJwk,
    },
  });
  const start = startResponse.payload;
  if (!startResponse.ok || start.status !== 'pairing_code_required') {
    throw new Error(start.error || start.status || 'pairing did not start');
  }
  const code = window.prompt('Enter the computer bridge pairing code shown on the working computer. After submitting it, approve the pairing in the computer terminal or Agent Kernel Desktop.');
  if (!code) throw new Error('Pairing cancelled.');
  appendMessage('assistant', 'Pairing request sent. Approve it on the working computer to complete setup.');
  log('Computer bridge pairing waiting for computer approval');
  const confirmResponse = await computerBridgeRequest('/pairing/confirm', {
    method: 'POST',
    body: { pairing_id: start.pairing_id, code: String(code).trim() },
  });
  const confirm = confirmResponse.payload;
  if (!confirmResponse.ok || confirm.status !== 'paired') {
    throw new Error(confirm.error || confirm.status || 'pairing failed');
  }
  await saveCodexPairing({
    protocol: CODEX_BRIDGE_PROTOCOL,
    bridge_url: codexBridgeUrl(),
    grant_id: confirm.grant_id,
    origin: start.origin || window.location.origin,
    browser_private_jwk: browserPrivateJwk,
    browser_public_jwk: browserPublicJwk,
    bridge_public_jwk: start.bridge_public_jwk,
    expires_at: confirm.expires_at,
    seq: 0,
  });
  setExtensionEnabled(state.codex.extensionId, true);
  log('Computer bridge paired');
  renderExtensionList();
  renderComputerConsole();
}

async function sendCodexBridgeMessage(type, payload = {}) {
  const envelope = await encryptCodexPayload({ type, ...payload });
  const response = await computerBridgeRequest('/v1/message', {
    method: 'POST',
    body: envelope,
  });
  const raw = response.payload;
  if (!response.ok) throw new Error(raw.error || `Computer bridge message failed: ${response.status}`);
  return decryptCodexEnvelope(raw);
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
      if (cursor.key !== CODEX_PAIRING_DB_KEY) out.push([cursor.key, cursor.value]);
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
  state.worker = new Worker('./js/llm-worker.js?v=20260502-adaptive-spec', { type: 'module' });
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
  return [
    tokenHeader,
    `<AK_LOOP> <AK_STATE> mode=${state.mode} selected_context=${selectedCount ? 1 : 0} retrieval=${contextRows.length ? 'ranked' : 'none'}`,
    'Return exactly this decision format: Action: respond, then Content: your direct answer.',
    'You are Agent Kernel Lite running entirely in this browser.',
    'Do not claim to execute, test, install, browse, or modify files.',
    'Answer the user directly. When using evidence, cite the evidence id such as [1] or [P1]; the interface renders the exact paper title and PDF link from that id.',
    'Do not generate paper titles or paper ids from memory; use evidence ids for grounded source references.',
    modeInstruction,
    `Mode: ${config.label}`,
    selectedContext ? 'Context target: answer about the selected paper already added to chat. Do not search for or introduce a different paper.' : '',
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
        const readingAppendix = [
          '',
          '<AK_HISTORY> Recent conversation:',
          historyContext(),
          '',
          '<AK_READING_NOTES> Semantic reading notes:',
          evidenceReadingNotes(userText, contextRows),
          selectedContext ? 'Context target: answer about the selected paper already added to chat. Do not search for or introduce a different paper.' : '',
        ].filter((line) => line !== '').join('\n');
        return `${packet.prompt}\n${readingAppendix}`;
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
    const raw = state.core.finish_model_reply
      ? state.core.finish_model_reply(text || '')
      : state.core.finish_turn(text || '');
    const packet = JSON.parse(raw);
    state.lastDecisionPacket = packet.decision_packet || null;
    return packet;
  } catch (error) {
    log(`WASM turn record failed: ${error.message || String(error)}`);
    return null;
  }
}

function displayTextFromDecision(packet, fallbackText) {
  const decision = packet?.decision_packet?.decision || packet?.decision || null;
  return String(decision?.content || fallbackText || 'No answer generated.');
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
  if (normalized.length > 120 && !/[.!?]/.test(normalized)) return true;
  if (/[\uFFFD]/.test(normalized)) return true;
  if (/\b(?:envend|local-balls|racket|gronuded|amtch|rpesent|ishould|thn|somne)\b/i.test(normalized)) return true;
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

function finalizeAssistantResponse(text, { fallback = false, reason = '' } = {}) {
  const turn = state.activeTurn;
  if (turn?.finalized) return false;
  if (turn) turn.finalized = true;
  clearActiveTurnTimer();
  state.modelBusy = false;
  setControlsBusy(false);
  const rows = turn?.contextRows || state.pendingContextRows || [];
  const userText = turn?.userText || '';
  const responseText = fallback
    ? [
        'The local decoder did not produce a decoded answer for this turn.',
        rows.length
          ? 'Research context was retrieved and remains available in the evidence cards, but I am not going to replace the model with a synthetic answer.'
          : 'No research context was attached to this turn, and I am not going to replace the model with a synthetic answer.',
        reason ? `Runtime detail: ${reason}` : '',
      ].filter(Boolean).join('\n\n')
    : maybeGroundedFallback(text, rows, userText);
  setProcessStep('generate', fallback ? 'error' : 'done', fallback ? 'Decoder timed out before answer' : `${formatCount(String(text || '').length)} characters generated`);
  const packet = recordAssistantTurn(responseText);
  const displayText = bindEvidenceAttribution(displayTextFromDecision(packet, responseText), rows);
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
  node.append(roleNode, body);
  attachPaperButtons(node, state.retrievalRows);
  els.chat.appendChild(node);
  els.chat.scrollTop = els.chat.scrollHeight;
  state.messages.push({ role, text });
}

function renderStoredMessage(message) {
  if (message?.artifact?.type === 'image') {
    const node = document.createElement('article');
    node.className = 'message assistant image-result';
    const roleNode = document.createElement('div');
    roleNode.className = 'role';
    roleNode.textContent = 'Image Generation';
    const body = document.createElement('div');
    body.className = 'body image-artifact';
    const prompt = document.createElement('div');
    prompt.className = 'image-prompt';
    prompt.textContent = message.artifact.prompt || message.text || '';
    const meta = document.createElement('div');
    meta.className = 'image-meta';
    meta.textContent = [
      message.artifact.model || state.image.modelId,
      message.artifact.seed ? `seed ${message.artifact.seed}` : '',
      'artifact metadata restored',
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
  node.append(roleNode, body);
  els.chat.appendChild(node);
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
    },
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
  const codexMode = extensionEnabled(state.codex.extensionId);
  els.send.disabled = busy || state.image.busy || state.translation.busy || state.translation.listening || state.codex.busy || (codexMode ? !state.codex.paired : state.image.enabled ? !state.image.ready : state.modelBusy || !state.modelReady);
  els.prompt.disabled = busy;
  els.loadModel.disabled = state.image.enabled || codexMode || busy || state.modelBusy;
  if (els.unloadModel) els.unloadModel.disabled = state.image.enabled || codexMode || busy || state.modelBusy || !state.worker;
  els.loadPack.disabled = busy;
  if (els.audioTranslate) els.audioTranslate.disabled = busy || state.translation.busy;
  syncModelControls();
}

function summarizeCodexOutput(result) {
  const extractText = (value) => {
    if (!value) return '';
    if (typeof value === 'string') return value.trim();
    if (typeof value !== 'object') return String(value).trim();
    const msg = value.msg && typeof value.msg === 'object' ? value.msg : null;
    const item = value.item && typeof value.item === 'object' ? value.item : null;
    const eventItem = msg?.item && typeof msg.item === 'object' ? msg.item : null;
    const candidateItem = eventItem || item;
    if (candidateItem?.type === 'agent_message') return String(candidateItem.text || candidateItem.message || '').trim();
    if (msg?.type === 'agent_message') return String(msg.text || msg.message || '').trim();
    if (value.type === 'agent_message') return String(value.text || value.message || '').trim();
    if (value.type === 'turn.failed' || value.type === 'error') return String(value.message || value.error?.message || '').trim();
    if (msg?.type === 'turn.failed' || msg?.type === 'error') return String(msg.message || msg.error?.message || '').trim();
    if (value.last_agent_message) return String(value.last_agent_message).trim();
    if (msg?.last_agent_message) return String(msg.last_agent_message).trim();
    if (value.message && !String(value.type || '').startsWith('turn.')) return String(value.message).trim();
    if (value.content) return String(value.content).trim();
    if (value.text && !String(value.type || '').startsWith('turn.')) return String(value.text).trim();
    return '';
  };
  if (Array.isArray(result?.events)) {
    const eventText = [...result.events].reverse()
      .map((event) => {
        const parsed = event?.parsed;
        if (parsed && typeof parsed === 'object') return extractText(parsed);
        return event?.text || '';
      })
      .find((value) => String(value || '').trim());
    if (eventText) return String(eventText).trim();
  }
  if (result?.summary) return String(result.summary).trim();
  if (result?.error) return String(result.error).trim();
  const stdout = String(result?.stdout || '').trim();
  const stderr = String(result?.stderr || '').trim();
  if (!stdout && !stderr) return codexTerminalStatus(result?.status) ? `Codex finished with status ${result?.status || 'unknown'}.` : '';
  const lines = stdout.split(/\r?\n/).filter(Boolean);
  const finalLine = [...lines].reverse().find((line) => {
    try {
      const event = JSON.parse(line);
      return Boolean(extractText(event));
    } catch {
      return line.length > 0 && !line.startsWith('{');
    }
  });
  let summary = '';
  if (finalLine) {
    try {
      const event = JSON.parse(finalLine);
      summary = extractText(event);
    } catch {
      summary = finalLine;
    }
  }
  if (!summary) summary = stdout.slice(-3000);
  if (stderr && result?.status !== 'completed') summary += `\n\nstderr:\n${stderr}`;
  return summary.trim();
}

function codexTerminalStatus(status) {
  return ['completed', 'failed', 'cancelled'].includes(String(status || ''));
}

function codexStatusDetail(status) {
  const events = Array.isArray(status?.events) ? status.events : [];
  const latest = events[events.length - 1];
  if (!latest) return `Codex ${status?.status || 'running'}`;
  const parsed = latest.parsed;
  if (parsed && typeof parsed === 'object') {
    const item = parsed.item || parsed.msg?.item || null;
    if (item?.type === 'command_execution') return shortText(`Command: ${item.command || 'running'}`, 80);
    if (item?.type === 'agent_message') return 'Codex replied';
    return shortText(parsed.type || parsed.msg?.type || parsed.event || latest.text || 'Codex event', 80);
  }
  return shortText(latest.text || `Codex ${status?.status || 'running'}`, 80);
}

async function pollCodexSession(sessionId) {
  let last = state.codex.eventCount || 0;
  for (;;) {
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
    const encrypted = await sendCodexBridgeMessage('computer.session.status', {
      session_id: sessionId,
      since: last,
    });
    const result = encrypted.result || {};
    state.codex.lastStatus = result;
    state.codex.eventCount = Number(result.event_count || last);
    last = state.codex.eventCount;
    setProcessStep('generate', codexTerminalStatus(result.status) ? result.status === 'completed' ? 'done' : 'error' : 'active', codexStatusDetail(result));
    if (codexTerminalStatus(result.status)) return result;
  }
}

async function readCodexDiff(sessionId, workspace, provider = computerProvider()) {
  const bridgeSessionId = String(sessionId || '').startsWith('terminal_') ? '' : sessionId;
  const proposal = proposeExtensionAction(state.codex.extensionId, state.codex.diffCapabilityId, {
    session_id: bridgeSessionId,
    workspace,
    bridge_url: codexBridgeUrl(),
  });
  if (proposal.status !== 'pending_user_approval') return null;
  try {
    const encrypted = await sendCodexBridgeMessage('computer.diff.read', { session_id: bridgeSessionId, workspace, provider });
    recordExtensionResult(proposal.action_id, {
      action_id: proposal.action_id,
      status: encrypted.result?.status === 'ok' ? 'approved_executed' : 'failed',
      output: {
        workspace: encrypted.result?.workspace || workspace,
        diff_chars: String(encrypted.result?.diff || '').length,
        truncated: Boolean(encrypted.result?.truncated),
      },
      artifact_refs: [],
      error: encrypted.result?.stderr || '',
    });
    return encrypted.result || null;
  } catch (error) {
    recordExtensionResult(proposal.action_id, {
      action_id: proposal.action_id,
      status: 'failed',
      error: error.message || String(error),
    });
    return null;
  }
}

function computerSessionTitle(session) {
  return session?.model || computerModel() || 'Codex default';
}

function computerWorkspaceName(session) {
  return String(session?.workspace || '').split('/').filter(Boolean).pop() || 'workspace';
}

function activeComputerSession() {
  if (!state.codex.activeSessionId) return null;
  return state.codex.sessions.find((session) => session.id === state.codex.activeSessionId) || null;
}

function upsertComputerSession(patch, options = {}) {
  const id = String(patch?.id || patch?.session_id || '');
  if (!id) return null;
  const activate = options.activate !== false;
  let session = state.codex.sessions.find((item) => item.id === id);
  if (!session) {
    session = {
      id,
      provider: computerProvider(),
      workspace: codexWorkspace(),
      prompt: '',
      status: 'starting',
      eventCount: 0,
      events: [],
      output: '',
      diff: null,
      actionId: '',
      startedAt: Date.now(),
      updatedAt: Date.now(),
    };
    state.codex.sessions.unshift(session);
  }
  Object.assign(session, patch, { id, updatedAt: Date.now() });
  if (activate) {
    state.codex.activeSessionId = id;
    state.codex.sessionId = id;
  }
  if (activate || state.codex.activeSessionId === id) state.codex.lastStatus = session;
  renderComputerConsole();
  return session;
}

function removeComputerSession(sessionId) {
  state.codex.sessions = state.codex.sessions.filter((session) => session.id !== sessionId);
  window.clearInterval(state.codex.pollTimers.get(sessionId));
  state.codex.pollTimers.delete(sessionId);
  if (state.codex.activeSessionId === sessionId) {
    state.codex.activeSessionId = '';
    state.codex.sessionId = state.codex.activeSessionId;
  }
  renderComputerConsole();
}

function setComputerConsoleOpen(open) {
  const available = extensionEnabled(state.codex.extensionId) && state.codex.paired;
  const nextOpen = Boolean(open && available);
  document.body.classList.remove('computer-session-open');
  document.body.classList.toggle('computer-console-open', nextOpen);
  if (els.computerConsoleButton) {
    els.computerConsoleButton.hidden = !available;
    els.computerConsoleButton.disabled = !available;
    els.computerConsoleButton.setAttribute('aria-pressed', nextOpen ? 'true' : 'false');
    els.computerConsoleButton.textContent = nextOpen ? 'Chat' : 'Computer';
  }
  if (!nextOpen) renderStoredMessages();
  if (!nextOpen && els.prompt) els.prompt.placeholder = modeConfig().placeholder;
  renderComputerConsole();
  syncModelControls();
}

function computerOutputText(session) {
  const lines = [];
  if (session?.status === 'ready') lines.push('Codex terminal ready. Send a message to Codex through the paired bridge.');
  if (session?.output) lines.push(session.output);
  if (session?.diff?.diff) {
    lines.push('');
    lines.push('diff:');
    lines.push(String(session.diff.diff).slice(0, 12000));
    if (session.diff.truncated) lines.push('... diff truncated ...');
  }
  return lines.filter(Boolean).join('\n');
}

function computerSessionMessages(session) {
  const messages = Array.isArray(session?.messages) ? [...session.messages] : [];
  if (session?.status === 'ready' && !messages.length) {
    messages.push({ role: 'assistant', text: 'Codex terminal ready. Send a message to Codex through the paired bridge.' });
  }
  if (session?.prompt && !messages.some((message) => message.role === 'user' && message.text === session.prompt)) {
    messages.push({ role: 'user', text: session.prompt });
  }
  const output = computerOutputText(session);
  if (output && !messages.some((message) => message.role === 'assistant' && message.text === output)) {
    messages.push({ role: 'assistant', text: output });
  }
  return messages;
}

function computerSessionChatHtml(session) {
  const messages = computerSessionMessages(session);
  if (!messages.length) return '<div class="computer-empty">No messages yet.</div>';
  return messages.map((message) => `
    <article class="message ${message.role === 'user' ? 'user' : 'assistant'}">
      <div class="role">${message.role === 'user' ? 'You' : 'Codex'}</div>
      <div class="body"><p>${escapeHtml(message.text || '')}</p></div>
    </article>
  `).join('');
}

function appendComputerMessage(role, text) {
  els.empty?.remove();
  const node = document.createElement('article');
  node.className = `message ${role === 'user' ? 'user' : 'assistant'}`;
  const roleNode = document.createElement('div');
  roleNode.className = 'role';
  roleNode.textContent = role === 'user' ? 'You' : 'Codex';
  const body = document.createElement('div');
  body.className = 'body';
  renderMessageBody(body, text, { linkEvidence: false });
  node.append(roleNode, body);
  els.chat.appendChild(node);
  els.chat.scrollTop = els.chat.scrollHeight;
}

function addComputerSessionMessage(session, role, text) {
  if (!session || !text) return;
  session.messages = [...(session.messages || []), { role, text }];
  session.updatedAt = Date.now();
  appendComputerMessage(role, text);
}

function lastComputerAssistantMessage(session) {
  return [...(session?.messages || [])].reverse().find((message) => message.role === 'assistant' && String(message.text || '').trim())?.text || '';
}

function formatTokenCount(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return '0';
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(number >= 10_000_000 ? 0 : 1)}M`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(number >= 10_000 ? 0 : 1)}K`;
  return String(Math.round(number));
}

function codexUsageFromEvents(events = []) {
  const usages = [];
  for (const event of events) {
    const parsed = event?.parsed;
    if (!parsed || typeof parsed !== 'object') continue;
    const usage = parsed.usage || parsed.msg?.usage || parsed.item?.usage || parsed.msg?.item?.usage;
    if (usage && typeof usage === 'object') usages.push(usage);
  }
  const latest = usages[usages.length - 1] || null;
  const totals = usages.reduce((sum, usage) => {
    sum.input += Number(usage.input_tokens || 0);
    sum.cached += Number(usage.cached_input_tokens || 0);
    sum.output += Number(usage.output_tokens || 0);
    sum.reasoning += Number(usage.reasoning_output_tokens || 0);
    return sum;
  }, { input: 0, cached: 0, output: 0, reasoning: 0 });
  return { latest, totals, turns: usages.length };
}

function computerContextStatusText(session) {
  const usage = codexUsageFromEvents(session?.events || []);
  if (!usage.latest) {
    return [
      'used context: unavailable until Codex completes a turn with usage data',
      'weekly context: unavailable from the current bridge stream',
    ];
  }
  const latestInput = Number(usage.latest.input_tokens || 0);
  const latestCached = Number(usage.latest.cached_input_tokens || 0);
  const latestOutput = Number(usage.latest.output_tokens || 0);
  const latestReasoning = Number(usage.latest.reasoning_output_tokens || 0);
  return [
    `used context: ${formatTokenCount(latestInput)} input tokens (${formatTokenCount(latestCached)} cached), ${formatTokenCount(latestOutput)} output, ${formatTokenCount(latestReasoning)} reasoning`,
    `session tokens: ${formatTokenCount(usage.totals.input)} input, ${formatTokenCount(usage.totals.output)} output across ${usage.turns} turn${usage.turns === 1 ? '' : 's'}`,
    'weekly context: unavailable from the current bridge stream',
  ];
}

function computerCommandHelpText() {
  return COMPUTER_SLASH_COMMANDS
    .filter((command) => !command.localOnly || command.name === 'mention')
    .map((command) => `/${command.name}${command.args ? ' ...' : ''} - ${command.description}`)
    .join('\n');
}

function computerSessionStatusText(session) {
  const activeCount = state.codex.sessions.length;
  const codexSession = session.codexSessionId || session.bridgeSessionId || session.id || '';
  return [
    `provider: ${session.provider || computerProvider()}`,
    `model: ${session.model || computerModel() || 'Codex default'}`,
    `workspace: ${computerWorkspaceName(session)}`,
    `workspace: ${session.workspace || codexWorkspace()}`,
    `status: ${session.status || 'unknown'}`,
    `session: ${shortText(codexSession, 48)}`,
    `events: ${Number(session.eventCount || 0)}`,
    `active terminals: ${activeCount}`,
    ...computerContextStatusText(session),
  ].join('\n');
}

function computerPermissionsText(session) {
  const health = state.codex.bridgeHealth || {};
  const roots = Array.isArray(health.allowed_workspaces) ? health.allowed_workspaces : [];
  return [
    `workspace: ${session.workspace || codexWorkspace()}`,
    `allowed roots: ${roots.join(', ') || 'unknown'}`,
    `workspace policy: ${health.workspace_policy || 'selected workspace must be under an allowed root'}`,
    `sandbox: ${health.sandbox || 'bridge default'}`,
    `approval policy: ${health.approval_policy || 'bridge default'}`,
    `pairing: ${state.codex.paired ? 'paired and encrypted' : 'not paired'}`,
  ].join('\n');
}

function createLocalComputerTerminal(workspace, provider, options = {}) {
  const id = `terminal_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
  return upsertComputerSession({
    id,
    provider,
    model: computerModel(),
    workspace,
    prompt: '',
    status: 'ready',
    eventCount: 0,
    events: [],
    output: '',
    messages: [],
    actionId: '',
    bridgeSessionId: '',
  }, { activate: options.activate !== false });
}

async function runComputerSlashCommand(session, text) {
  const match = String(text || '').trim().match(/^\/([a-z0-9-]+)(?:\s+([\s\S]*))?$/i);
  if (!match) return false;
  const name = match[1].toLowerCase();
  const args = String(match[2] || '').trim();
  const command = COMPUTER_SLASH_COMMANDS.find((item) => item.name === name);
  if (!command) return false;

  const visibleCommand = `/${name}${args ? ` ${args}` : ''}`;
  const promptCommandNames = new Set(['init', 'compact', 'review']);
  if (name !== 'mention' && !promptCommandNames.has(name)) addComputerSessionMessage(session, 'user', visibleCommand);

  if (name === 'help') {
    addComputerSessionMessage(session, 'assistant', computerCommandHelpText());
    return true;
  }
  if (name === 'status') {
    addComputerSessionMessage(session, 'assistant', computerSessionStatusText(session));
    return true;
  }
  if (name === 'permissions' || name === 'approvals') {
    try {
      state.codex.bridgeHealth = await codexBridgeHealth();
    } catch (_error) {
      // Use the last known bridge state if the health check is temporarily unavailable.
    }
    addComputerSessionMessage(session, 'assistant', computerPermissionsText(session));
    return true;
  }
  if (name === 'model' || name === 'models') {
    if (args) {
      session.model = args;
      setComputerModel(args);
      addComputerSessionMessage(session, 'assistant', `Model set to ${args} for this terminal's next Codex turn.`);
    } else {
      addComputerSessionMessage(session, 'assistant', [
        `current: ${session.model || computerModel() || 'Codex default'}`,
        'usage: /model <model-id>',
        'example: /model gpt-5.3-codex',
      ].join('\n'));
    }
    return true;
  }
  if (name === 'diff') {
    const diff = await readCodexDiff(session.id, session.workspace, session.provider);
    const body = diff?.diff
      ? `\`\`\`diff\n${String(diff.diff).slice(0, 12000)}${diff.truncated ? '\n... diff truncated ...' : ''}\n\`\`\``
      : 'No diff found.';
    session.diff = diff;
    addComputerSessionMessage(session, 'assistant', body);
    return true;
  }
  if (name === 'copy') {
    const last = lastComputerAssistantMessage(session);
    if (!last) {
      addComputerSessionMessage(session, 'assistant', 'No Codex response to copy yet.');
      return true;
    }
    await navigator.clipboard?.writeText(last);
    addComputerSessionMessage(session, 'assistant', 'Copied the last Codex response.');
    return true;
  }
  if (name === 'clear') {
    const oldId = session.id;
    window.clearInterval(state.codex.pollTimers.get(oldId));
    state.codex.pollTimers.delete(oldId);
    const nextId = `terminal_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
    session.id = nextId;
    session.bridgeSessionId = '';
    session.codexSessionId = '';
    session.messages = [];
    session.output = '';
    session.prompt = '';
    session.diff = null;
    session.status = 'ready';
    session.eventCount = 0;
    session.events = [];
    state.codex.activeSessionId = nextId;
    state.codex.sessionId = nextId;
    renderComputerSessionPage(session);
    return true;
  }
  if (name === 'new') {
    const next = createLocalComputerTerminal(session.workspace || codexWorkspace(), session.provider || computerProvider(), { activate: true });
    renderComputerSessionPage(next);
    return true;
  }
  if (name === 'quit' || name === 'exit') {
    handleComputerAction('back', session.id);
    return true;
  }
  if (name === 'mention') {
    els.prompt.value = '@';
    els.prompt.focus();
    hideComputerCommandPalette();
    return true;
  }
  if (name === 'ps') {
    const rows = state.codex.sessions.map((item) => `${item.id === session.id ? '*' : '-'} ${computerSessionTitle(item)} | ${item.status || 'unknown'} | ${shortText(item.id, 32)}`);
    addComputerSessionMessage(session, 'assistant', rows.length ? rows.join('\n') : 'No active computer terminals.');
    return true;
  }
  if (name === 'stop') {
    if (!(session.status === 'running' || session.status === 'starting')) {
      addComputerSessionMessage(session, 'assistant', 'No Codex turn is currently running.');
      return true;
    }
    const result = await cancelCodexSession(session.id, session.provider || computerProvider());
    upsertComputerSession({ id: session.id, status: result?.status || 'cancelled', output: summarizeCodexOutput(result) });
    addComputerSessionMessage(activeComputerSession() || session, 'assistant', 'Stopped the running Codex turn.');
    return true;
  }

  const promptByCommand = {
    init: CODEX_INIT_PROMPT,
    compact: 'Summarize this conversation so far into a concise working context for continuing this task.',
    review: `Review my current changes and find issues.${args ? `\n\nFocus: ${args}` : ''}`,
  };
  const prompt = promptByCommand[name];
  if (prompt) {
    await sendComputerFollowup(session.id, prompt, visibleCommand);
    return true;
  }

  addComputerSessionMessage(session, 'assistant', `/${name} is not available in the web terminal yet.`);
  return true;
}

function renderComputerSessionPage(session) {
  if (!session) return;
  document.body.classList.remove('computer-console-open');
  document.body.classList.add('computer-session-open');
  syncModelControls();
  els.chat.innerHTML = '';
  const header = document.createElement('article');
  header.className = 'message assistant computer-session-header';
  header.innerHTML = `
    <div class="role">Codex Terminal</div>
    <div class="body">
      <div class="computer-session-heading">
        <div>
          <p>${escapeHtml(computerSessionTitle(session))}</p>
          <p class="computer-workspace">${escapeHtml(computerWorkspaceName(session))} · ${escapeHtml(session.workspace || '')}</p>
        </div>
        <button type="button" class="secondary" data-computer-action="back">Back</button>
      </div>
    </div>
  `;
  for (const button of header.querySelectorAll('[data-computer-action]')) {
    button.addEventListener('click', () => handleComputerAction(button.dataset.computerAction, session.id));
  }
  els.chat.appendChild(header);
  for (const message of computerSessionMessages(session)) appendComputerMessage(message.role, message.text);
  els.chat.scrollTop = els.chat.scrollHeight;
  if (els.prompt) els.prompt.placeholder = 'Message Codex...';
  if (els.send) els.send.textContent = 'Send';
}

function matchingComputerCommands(value) {
  const text = String(value || '');
  if (!document.body.classList.contains('computer-session-open')) return [];
  if (!text.startsWith('/') || text.startsWith('/ ')) return [];
  const name = text.slice(1).split(/\s+/)[0].toLowerCase();
  return COMPUTER_SLASH_COMMANDS
    .filter((command) => !name || command.name.startsWith(name))
    .slice(0, 8);
}

function renderComputerCommandPalette() {
  if (!els.commandPalette) return;
  const matches = matchingComputerCommands(els.prompt?.value || '');
  els.commandPalette.hidden = !matches.length;
  els.commandPalette.innerHTML = '';
  for (const command of matches) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'command-option';
    button.innerHTML = `
      <strong>/${escapeHtml(command.name)}</strong>
      <span>${escapeHtml(command.description)}</span>
    `;
    button.addEventListener('mousedown', (event) => event.preventDefault());
    button.addEventListener('click', () => {
      els.prompt.value = `/${command.name}${command.args ? ' ' : ''}`;
      hideComputerCommandPalette();
      els.prompt.focus();
    });
    els.commandPalette.appendChild(button);
  }
}

function hideComputerCommandPalette() {
  if (!els.commandPalette) return;
  els.commandPalette.hidden = true;
  els.commandPalette.innerHTML = '';
}

function completeFirstComputerCommand() {
  const first = matchingComputerCommands(els.prompt?.value || '')[0];
  if (!first) return false;
  els.prompt.value = `/${first.name}${first.args ? ' ' : ''}`;
  hideComputerCommandPalette();
  return true;
}

function renderComputerProviderSelect() {
  if (!els.computerProvider) return;
  const rows = state.codex.providers.length
    ? state.codex.providers
    : [
      { id: 'codex', name: 'Codex', available: true },
      { id: 'claude_code', name: 'Claude Code', available: false },
      { id: 'cursor', name: 'Cursor', available: false },
    ];
  const selected = computerProvider();
  els.computerProvider.innerHTML = '';
  for (const provider of rows) {
    const option = document.createElement('option');
    option.value = provider.id;
    option.textContent = `${provider.name || provider.id}${provider.available ? '' : ' (not detected)'}`;
    els.computerProvider.appendChild(option);
  }
  els.computerProvider.value = selected;
}

function renderComputerConsole() {
  if (!els.computerSessionList || !els.computerSessionDetail) return;
  if (els.computerWorkspace && !els.computerWorkspace.value) els.computerWorkspace.value = codexWorkspace();
  renderComputerProviderSelect();
  els.computerStart.disabled = !state.codex.paired;
  if (els.computerCreateTerminal) els.computerCreateTerminal.disabled = !state.codex.paired;

  els.computerSessionDetail.hidden = true;
  els.computerSessionList.hidden = false;
  els.computerSessionList.innerHTML = '';
  els.computerSessionList.classList.toggle('empty', !state.codex.sessions.length);
  if (!state.codex.sessions.length) {
    const empty = document.createElement('p');
    empty.className = 'computer-empty';
    empty.textContent = state.codex.paired
      ? 'No Codex terminals yet.'
      : 'Pair the Computer Use bridge before starting terminals.';
    els.computerSessionList.appendChild(empty);
  }
  for (const session of state.codex.sessions) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `computer-session-button${state.codex.activeSessionId === session.id ? ' active' : ''}`;
    button.innerHTML = `
      <strong>${escapeHtml(computerSessionTitle(session))}</strong>
      <span>${escapeHtml(session.status || 'unknown')} | ${escapeHtml(shortText(session.id, 28))}</span>
      <span>${escapeHtml(shortText(session.prompt || session.workspace || computerWorkspaceName(session), 96))}</span>
    `;
    button.addEventListener('click', () => {
      state.codex.activeSessionId = session.id;
      state.codex.sessionId = session.id;
      renderComputerSessionPage(session);
    });
    els.computerSessionList.appendChild(button);
  }
}

function setComputerConsoleStatus(message, mode = '') {
  if (!els.computerSessionDetail) return;
  const text = String(message || '').trim();
  if (!text) return;
  const active = activeComputerSession();
  if (active) {
    upsertComputerSession({ id: active.id, output: text });
    return;
  }
  els.computerSessionDetail.innerHTML = `
    <div class="computer-detail-title">
      <strong>Computer Use</strong>
      <span>${escapeHtml(mode || 'status')}</span>
    </div>
    <div class="computer-output">${escapeHtml(text)}</div>
    <div class="computer-actions"></div>
  `;
}

async function startComputerSession() {
  if (!state.codex.paired) await loadCodexPairing();
  if (!state.codex.paired) throw new Error('Pair the local computer bridge before starting a session.');
  const workspace = String(els.computerWorkspace?.value || codexWorkspace()).trim();
  const provider = String(els.computerProvider?.value || computerProvider()).trim() || 'codex';
  if (!workspace) throw new Error('Set a repo or workspace path.');
  setCodexWorkspace(workspace);
  setComputerProvider(provider);
  await codexBridgeHealth();
  createLocalComputerTerminal(workspace, provider, { activate: false });
}

async function launchComputerSessionFromMessage(session, prompt, displayText = '') {
  const provider = session.provider || computerProvider();
  const workspace = session.workspace || codexWorkspace();
  const model = session.model || computerModel();
  const proposal = proposeExtensionAction(state.codex.extensionId, state.codex.startCapabilityId, {
    prompt,
    workspace,
    bridge_url: codexBridgeUrl(),
    provider,
    model,
  });
  if (proposal.status !== 'pending_user_approval') {
    throw new Error(proposal.error || `Computer Use action was ${proposal.status || 'rejected'}`);
  }
  const baseMessages = session.messages || [];
  const visiblePrompt = String(displayText || prompt);
  upsertComputerSession({
    id: session.id,
    status: 'starting',
    prompt,
    output: 'Starting Codex through the paired bridge...',
    messages: [...baseMessages, { role: 'user', text: visiblePrompt }],
    actionId: proposal.action_id || '',
  });
  const encrypted = await sendCodexBridgeMessage('computer.session.start', {
    action_id: proposal.action_id,
    prompt,
    workspace,
    provider,
    model,
  });
  const result = encrypted.result || {};
  if (!result.session_id) throw new Error('Computer bridge did not return a session id.');
  const oldId = session.id;
  const output = summarizeCodexOutput(result);
  const next = {
    ...session,
    id: result.session_id,
    bridgeSessionId: result.session_id,
    provider,
    model,
    workspace: result.workspace || workspace,
    prompt,
    status: result.status || 'running',
    eventCount: Number(result.event_count || 0),
    events: result.events || [],
    output,
    messages: output ? [...baseMessages, { role: 'user', text: visiblePrompt }, { role: 'assistant', text: output }] : [...baseMessages, { role: 'user', text: visiblePrompt }],
    actionId: proposal.action_id || '',
    updatedAt: Date.now(),
  };
  state.codex.sessions = state.codex.sessions.map((item) => item.id === oldId ? next : item);
  state.codex.activeSessionId = next.id;
  state.codex.sessionId = next.id;
  state.codex.lastStatus = next;
  renderComputerConsole();
  recordExtensionResult(proposal.action_id, {
    action_id: proposal.action_id,
    status: codexTerminalStatus(result.status) ? result.status === 'completed' ? 'approved_executed' : 'failed' : 'approved_executed',
    output: {
      bridge_url: codexBridgeUrl(),
      workspace: next.workspace,
      provider,
      model,
      session_id: next.id,
      codex_session_id: result.codex_session_id || '',
    },
    artifact_refs: [],
  });
  scheduleComputerPoll(next.id);
  return next;
}

async function sendComputerFollowup(sessionId, messageText = '', displayText = '') {
  const session = state.codex.sessions.find((item) => item.id === sessionId);
  if (!session) throw new Error('No computer session selected.');
  const prompt = String(
    document.body.classList.contains('computer-session-open')
      ? messageText
      : els.computerSessionDetail?.querySelector('#computerSessionPrompt')?.value || '',
  ).trim();
  if (!prompt) throw new Error('Enter a follow-up prompt.');
  if (session.status === 'ready' || !session.bridgeSessionId && String(session.id || '').startsWith('terminal_')) {
    await launchComputerSessionFromMessage(session, prompt, displayText);
    return;
  }
  const proposal = proposeExtensionAction(state.codex.extensionId, state.codex.sendCapabilityId, {
    session_id: session.id,
    prompt,
    workspace: session.workspace,
    bridge_url: codexBridgeUrl(),
    provider: session.provider,
    model: session.model || computerModel(),
  });
  if (proposal.status !== 'pending_user_approval') {
    throw new Error(proposal.error || `Computer Use action was ${proposal.status || 'rejected'}`);
  }
  const encrypted = await sendCodexBridgeMessage('computer.session.send', {
    session_id: session.id,
    action_id: proposal.action_id,
    prompt,
    workspace: session.workspace,
    provider: session.provider,
    model: session.model || computerModel(),
  });
  const result = encrypted.result || {};
  const output = summarizeCodexOutput(result);
  const visiblePrompt = String(displayText || prompt);
  const messages = [...(session.messages || []), { role: 'user', text: visiblePrompt }];
  if (output) messages.push({ role: 'assistant', text: output });
  const oldId = session.id;
  const followupId = result.session_id || session.id;
  const followup = {
    ...session,
    id: followupId,
    bridgeSessionId: followupId,
    provider: session.provider,
    model: session.model || computerModel(),
    workspace: result.workspace || session.workspace,
    prompt,
    status: result.status || 'running',
    eventCount: Number(result.event_count || 0),
    events: result.events || [],
    output,
    messages,
    actionId: proposal.action_id || '',
    updatedAt: Date.now(),
  };
  state.codex.sessions = state.codex.sessions.map((item) => item.id === oldId ? followup : item);
  state.codex.activeSessionId = followup.id;
  state.codex.sessionId = followup.id;
  state.codex.lastStatus = followup;
  renderComputerConsole();
  recordExtensionResult(proposal.action_id, {
    action_id: proposal.action_id,
    status: 'approved_executed',
    output: {
      bridge_url: codexBridgeUrl(),
      workspace: followup.workspace,
      provider: followup.provider,
      model: followup.model || '',
      session_id: followup.id,
      parent_session_id: session.id,
    },
    artifact_refs: [],
  });
  scheduleComputerPoll(followup.id);
}

async function refreshComputerSession(sessionId) {
  const session = state.codex.sessions.find((item) => item.id === sessionId);
  if (!session) return null;
  const encrypted = await sendCodexBridgeMessage('computer.session.status', {
    session_id: session.id,
    since: Number(session.eventCount || 0),
    provider: session.provider,
  });
  const result = encrypted.result || {};
  const events = [...(session.events || []), ...(result.events || [])];
  const output = summarizeCodexOutput(result) || session.output;
  const messages = [...(session.messages || [])];
  if (output && output !== session.output && !messages.some((message) => message.role === 'assistant' && message.text === output)) {
    messages.push({ role: 'assistant', text: output });
  }
  const updated = upsertComputerSession({
    id: session.id,
    provider: session.provider,
    model: result.model || session.model || computerModel(),
    workspace: result.workspace || session.workspace,
    status: result.status || session.status,
    eventCount: Number(result.event_count || events.length || session.eventCount || 0),
    events,
    output,
    messages,
    exitCode: result.exit_code,
    elapsedMs: result.elapsed_ms,
    codexSessionId: result.codex_session_id || session.codexSessionId,
  }, { activate: state.codex.activeSessionId === session.id });
  if (codexTerminalStatus(updated.status)) {
    window.clearInterval(state.codex.pollTimers.get(session.id));
    state.codex.pollTimers.delete(session.id);
    if (updated.actionId && !updated.receiptFinalized) {
      recordExtensionResult(updated.actionId, {
        action_id: updated.actionId,
        status: updated.status === 'completed' ? 'approved_executed' : updated.status === 'cancelled' ? 'cancelled' : 'failed',
        output: {
          bridge_url: codexBridgeUrl(),
          workspace: updated.workspace,
          provider: updated.provider,
          session_id: updated.id,
          codex_session_id: updated.codexSessionId || '',
          exit_code: updated.exitCode,
          elapsed_ms: updated.elapsedMs,
        },
        artifact_refs: [],
        error: updated.status === 'failed' ? String(updated.output || 'Computer session failed') : '',
      });
      updated.receiptFinalized = true;
    }
  }
  if (document.body.classList.contains('computer-session-open') && state.codex.activeSessionId === updated.id) {
    renderComputerSessionPage(updated);
  }
  return updated;
}

function scheduleComputerPoll(sessionId) {
  if (!sessionId || state.codex.pollTimers.has(sessionId)) return;
  const timer = window.setInterval(async () => {
    try {
      await refreshComputerSession(sessionId);
    } catch (error) {
      const session = state.codex.sessions.find((item) => item.id === sessionId);
      if (session) upsertComputerSession(
        { id: sessionId, status: 'failed', output: error.message || String(error) },
        { activate: state.codex.activeSessionId === sessionId },
      );
      window.clearInterval(timer);
      state.codex.pollTimers.delete(sessionId);
    }
  }, 1600);
  state.codex.pollTimers.set(sessionId, timer);
}

async function handleComputerAction(action, sessionId = state.codex.activeSessionId) {
  try {
    if (action === 'send') await sendComputerFollowup(sessionId);
    else if (action === 'refresh') await refreshComputerSession(sessionId);
    else if (action === 'cancel') {
      const session = state.codex.sessions.find((item) => item.id === sessionId);
      const result = await cancelCodexSession(sessionId, session?.provider || computerProvider());
      upsertComputerSession({ id: sessionId, status: result?.status || 'cancelled', output: summarizeCodexOutput(result) });
    } else if (action === 'diff') {
      const session = state.codex.sessions.find((item) => item.id === sessionId);
      if (session) {
        const diff = await readCodexDiff(session.id, session.workspace, session.provider);
        upsertComputerSession({ id: session.id, diff });
      }
    } else if (action === 'new') {
      state.codex.activeSessionId = '';
      state.codex.sessionId = '';
      renderComputerConsole();
    } else if (action === 'back') {
      state.codex.activeSessionId = '';
      state.codex.sessionId = '';
      document.body.classList.remove('computer-session-open');
      document.body.classList.add('computer-console-open');
      if (els.prompt) els.prompt.placeholder = modeConfig().placeholder;
      renderComputerConsole();
      syncModelControls();
    } else if (action === 'remove') {
      removeComputerSession(sessionId);
    }
  } catch (error) {
    appendMessage('assistant', `Computer Use failed: ${error.message || String(error)}`);
    log(`Computer Use failed: ${error.message || String(error)}`);
  }
}

async function cancelCodexSession(sessionId = state.codex.sessionId, provider = computerProvider()) {
  if (!sessionId) throw new Error('No Codex session is active.');
  const proposal = proposeExtensionAction(state.codex.extensionId, state.codex.cancelCapabilityId, {
    session_id: sessionId,
    bridge_url: codexBridgeUrl(),
  });
  if (proposal.status !== 'pending_user_approval') {
    throw new Error(proposal.error || `Codex cancel was ${proposal.status || 'rejected'}`);
  }
  const encrypted = await sendCodexBridgeMessage('computer.session.cancel', { session_id: sessionId, provider });
  recordExtensionResult(proposal.action_id, {
    action_id: proposal.action_id,
    status: encrypted.result?.status === 'cancelled' ? 'approved_executed' : 'failed',
    output: {
      session_id: encrypted.result?.session_id || sessionId,
      status: encrypted.result?.status || '',
    },
    artifact_refs: [],
  });
  state.codex.lastStatus = encrypted.result || null;
  return encrypted.result;
}

async function submitCodexPrompt(text) {
  resetProcessTrace(text);
  setControlsBusy(true);
  state.codex.busy = true;
  try {
    if (!state.codex.paired) await loadCodexPairing();
    if (!state.codex.paired) throw new Error('Pair the local computer bridge before using Computer Use.');
    const workspace = codexWorkspace();
    if (!workspace) throw new Error('Set a repo or workspace path in the Computer Use extension settings.');
    setProcessStep('runtime', 'active', 'Checking paired computer bridge');
    await codexBridgeHealth();
    const followup = Boolean(state.codex.sessionId);
    const capabilityId = followup ? state.codex.sendCapabilityId : state.codex.startCapabilityId;
    const proposal = proposeExtensionAction(state.codex.extensionId, capabilityId, {
      session_id: state.codex.sessionId || '',
      prompt: text,
      workspace,
      bridge_url: codexBridgeUrl(),
      provider: computerProvider(),
    });
    if (proposal.status !== 'pending_user_approval') {
      throw new Error(proposal.error || `Computer Use action was ${proposal.status || 'rejected'}`);
    }
    state.codex.activeActionId = proposal.action_id || null;
    setProcessStep('runtime', 'done', 'Computer bridge paired');
    setProcessStep('plan', 'done', followup ? `${computerProvider()} follow-up requested through extension` : `${computerProvider()} session requested through extension`);
    setProcessStep('generate', 'active', followup ? `Sending ${computerProvider()} follow-up` : `Starting ${computerProvider()} on the working computer`);
    const encrypted = await sendCodexBridgeMessage(followup ? 'computer.session.send' : 'computer.session.start', {
      session_id: state.codex.sessionId || '',
      action_id: state.codex.activeActionId,
      prompt: text,
      workspace,
      provider: computerProvider(),
    });
    const started = encrypted.result || {};
    state.codex.sessionId = started.session_id || state.codex.sessionId;
    state.codex.eventCount = Number(started.event_count || 0);
    setProcessStep('generate', 'active', started.session_id ? `Codex session ${shortText(started.session_id, 28)} running` : 'Codex running');
    const result = codexTerminalStatus(started.status) ? started : await pollCodexSession(state.codex.sessionId);
    const status = result.status === 'completed' ? 'approved_executed' : 'failed';
    if (state.codex.activeActionId) {
      recordExtensionResult(state.codex.activeActionId, {
        action_id: state.codex.activeActionId,
        status,
        output: {
          bridge_url: codexBridgeUrl(),
          workspace: result.workspace || workspace,
          provider: computerProvider(),
          session_id: result.session_id || state.codex.sessionId,
          codex_session_id: result.codex_session_id || '',
          exit_code: result.exit_code,
          elapsed_ms: result.elapsed_ms,
        },
        artifact_refs: [],
        error: status === 'failed' ? String(result.stderr || result.stdout || 'Codex failed') : '',
      });
    }
    setProcessStep('generate', status === 'approved_executed' ? 'done' : 'error', `Codex exit ${result.exit_code ?? 'unknown'}`);
    setProcessStep('render', 'done', 'Codex result displayed');
    appendMessage('assistant', summarizeCodexOutput(result));
    finishProcessTrace(status === 'approved_executed' ? 'Codex Done' : 'Codex Failed');
  } catch (error) {
    if (state.codex.activeActionId) {
      recordExtensionResult(state.codex.activeActionId, {
        action_id: state.codex.activeActionId,
        status: 'failed',
        error: error.message || String(error),
      });
    }
    setProcessStep('render', 'error', error.message || String(error));
    finishProcessTrace('Error');
    appendMessage('assistant', `Codex failed: ${error.message || String(error)}`);
    log(`Codex failed: ${error.message || String(error)}`);
  } finally {
    state.codex.activeActionId = null;
    state.codex.busy = false;
    setControlsBusy(false);
    renderExtensionList();
  }
}

async function submitPrompt(event) {
  event.preventDefault();
  const text = els.prompt.value.trim();
  if (!text) return;
  els.prompt.value = '';
  hideComputerCommandPalette();
  if (document.body.classList.contains('computer-session-open')) {
    const session = activeComputerSession();
    if (!session) {
      document.body.classList.remove('computer-session-open');
      renderStoredMessages();
      return;
    }
    if (text.startsWith('/') && !text.startsWith('/ ')) {
      const handled = await runComputerSlashCommand(session, text).then((handled) => {
        if (handled && document.body.classList.contains('computer-session-open')) renderComputerSessionPage(activeComputerSession() || session);
        return handled;
      }).catch((error) => {
        addComputerSessionMessage(activeComputerSession() || session, 'assistant', `Computer command failed: ${error.message || String(error)}`);
        log(`Computer command failed: ${error.message || String(error)}`);
        return true;
      });
      if (handled) return;
    }
    await sendComputerFollowup(session.id, text).then(() => {
      renderComputerSessionPage(activeComputerSession());
    }).catch((error) => {
      appendComputerMessage('assistant', `Computer Use failed: ${error.message || String(error)}`);
      log(`Computer Use failed: ${error.message || String(error)}`);
    });
    return;
  }
  if (state.modelBusy || state.image.busy || state.codex.busy) return;
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
    const compiledPrompt = buildPrompt(text, contextRows);
    setProcessStep('compile', 'done', `${formatCount(compiledPrompt.length)} prompt characters`);
    setProcessStep('generate', 'active', `Generating up to ${formatCount(targetMaxTokens())} tokens`);
    const generationId = ++state.generationRunId;
    armGenerationFallback(text, contextRows, generationId);
    ensureWorker().postMessage({
      type: 'generate',
      generationId,
      prompt: compiledPrompt,
      options: {
        maxNewTokens: targetMaxTokens(),
        temperature: config.temperature,
      },
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
        steps: 4,
        guidance: 0,
        seed: Math.floor(Math.random() * 1_000_000_000),
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
  els.send.textContent = 'Send';
  els.prompt.placeholder = state.image.enabled
    ? 'Describe the image to generate...'
    : state.translation.enabled
      ? `Text to translate to ${translationTargetLabel()}...`
      : config.placeholder;
  log(`mode set: ${config.label}`);
}

function parseCoreJson(raw, fallback = {}) {
  try {
    return JSON.parse(String(raw || '{}'));
  } catch {
    return fallback;
  }
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
  if (manifest.id === state.codex.extensionId) {
    return `${source} | ${enabled} | ${state.codex.paired ? 'paired' : 'setup required'}`;
  }
  return `${source} | ${enabled}`;
}

function extensionSetupText(manifest) {
  if (manifest.id === state.image.extensionId) return 'Enable it to switch the composer into image generation mode. This extension is available for development and future release installs.';
  if (manifest.id === state.translation.extensionId) return 'Requires browser speech/model setup before audio translation can run.';
  if (manifest.id === state.codex.extensionId) return 'Pair Agent Kernel Desktop or the local computer bridge on your working computer.';
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

function appendCodexSettings(settings, manifest) {
  const bridgeLabel = document.createElement('label');
  bridgeLabel.className = 'extension-detail';
  bridgeLabel.textContent = 'Bridge URL';
  const bridgeInput = document.createElement('input');
  bridgeInput.type = 'text';
  bridgeInput.value = codexBridgeUrl();
  bridgeInput.placeholder = 'http://192.168.0.85:45731';
  bridgeInput.autocomplete = 'off';
  bridgeInput.addEventListener('change', () => {
    setCodexBridgeUrl(bridgeInput.value);
    bridgeInput.value = codexBridgeUrl();
    renderExtensionList();
  });

  const status = document.createElement('p');
  status.className = 'extension-detail';
  status.textContent = state.codex.paired
    ? 'Computer bridge paired.'
    : 'Pair the local computer bridge to enable Computer Use.';

  const row = document.createElement('div');
  row.className = 'button-row compact';
  const pair = document.createElement('button');
  pair.type = 'button';
  pair.className = 'secondary';
  pair.textContent = state.codex.paired ? 'Re-pair' : 'Pair Bridge';
  pair.addEventListener('click', async () => {
    try {
      setCodexBridgeUrl(bridgeInput.value);
      bridgeInput.value = codexBridgeUrl();
      await pairCodexBridge();
    } catch (error) {
      const diagnostics = await diagnoseCodexBridge().catch((diagnosticError) => `Bridge diagnostics failed: ${diagnosticError.message || String(diagnosticError)}`);
      appendMessage('assistant', `Codex pairing failed: ${error.message || String(error)}\n\n${diagnostics}`);
      log(`Codex pairing failed: ${error.message || String(error)}`);
    }
  });
  const health = document.createElement('button');
  health.type = 'button';
  health.className = 'secondary';
  health.textContent = 'Check';
  health.addEventListener('click', async () => {
    try {
      setCodexBridgeUrl(bridgeInput.value);
      bridgeInput.value = codexBridgeUrl();
      const result = await codexBridgeHealth();
      renderExtensionList();
      log(`Computer bridge ready: ${result.codex_available ? 'codex found' : 'codex missing'}`);
      const providerText = (result.providers || []).map((provider) => `${provider.name || provider.id}: ${provider.available ? 'available' : 'missing'}`).join(', ');
      appendMessage('assistant', `Computer bridge is reachable. Providers: ${providerText || 'none'}. Allowed roots: ${(result.allowed_workspaces || []).join(', ') || 'none'}. Policy: ${result.workspace_policy || 'selected workspace must be under an allowed root'}`);
    } catch (error) {
      const diagnostics = await diagnoseCodexBridge().catch((diagnosticError) => `Bridge diagnostics failed: ${diagnosticError.message || String(diagnosticError)}`);
      appendMessage('assistant', `Computer bridge check failed: ${error.message || String(error)}\n\n${diagnostics}`);
      log(`Computer bridge check failed: ${error.message || String(error)}`);
    }
  });
  const forget = document.createElement('button');
  forget.type = 'button';
  forget.className = 'secondary';
  forget.textContent = 'Forget Pairing';
  forget.disabled = !state.codex.paired;
  forget.addEventListener('click', () => clearCodexPairing());
  row.append(pair, health, forget);

  settings.append(bridgeLabel, bridgeInput, status, row);
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
    const codexUnpaired = manifest.id === state.codex.extensionId && !state.codex.paired;
    toggle.className = `image-toggle-button${manifest.enabled && !codexUnpaired ? ' active' : ''}`;
    toggle.textContent = codexUnpaired ? 'Pair' : manifest.enabled ? 'On' : 'Off';
    toggle.setAttribute('aria-pressed', manifest.enabled && !codexUnpaired ? 'true' : 'false');
    toggle.addEventListener('click', async (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (codexUnpaired) {
        if (!manifest.enabled) setExtensionEnabled(manifest.id, true);
        try {
          await pairCodexBridge();
        } catch (error) {
          appendMessage('assistant', `Codex pairing failed: ${error.message || String(error)}`);
          log(`Codex pairing failed: ${error.message || String(error)}`);
          renderExtensionList();
        }
        return;
      }
      setInstalledExtensionEnabled(manifest.id, !manifest.enabled);
    });
    summary.append(title, toggle);

    const settings = document.createElement('div');
    settings.className = 'extension-settings';
    if (manifest.id === state.codex.extensionId) {
      appendCodexSettings(settings, manifest);
    }
    if (manifest.id !== state.codex.extensionId || state.codex.paired) {
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
    }
    if (manifest.id !== state.image.extensionId && !(manifest.id === state.codex.extensionId && !state.codex.paired)) {
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
  setComputerConsoleOpen(document.body.classList.contains('computer-console-open'));
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
    },
    session: {
      messages: state.messages,
      paper_context_rows: state.paperContextRows,
      pending_context_rows: state.pendingContextRows,
      retrieval_rows: state.retrievalRows.slice(0, 32),
    },
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
  restoreLocalStorage(bundle.storage?.local_storage);
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
    record: recordExtensionResult,
    translateText: (text, options = {}) => runTranslator(text, { ...options, modality: 'text' }),
    translateAudio: (input, options = {}) => {
      const text = typeof input === 'string' ? input : input?.transcript || input?.text || '';
      return runTranslator(text, { ...options, modality: 'audio' });
    },
    pairCodex: pairCodexBridge,
    codexHealth: codexBridgeHealth,
    runCodex: (prompt, options = {}) => sendCodexBridgeMessage('computer.session.start', {
      prompt,
      workspace: options.workspace || codexWorkspace(),
      provider: options.provider || computerProvider(),
      ...options,
    }),
    cancelCodex: cancelCodexSession,
    codexDiff: (options = {}) => sendCodexBridgeMessage('computer.diff.read', {
      session_id: options.session_id || state.codex.sessionId,
      workspace: options.workspace || codexWorkspace(),
      provider: options.provider || computerProvider(),
    }),
    exportSession: buildSessionExport,
    restoreSession: restoreSessionBundle,
  });
}

function registerBuiltinExtensions() {
  const manifests = [];
  for (const manifest of manifests) {
    const result = registerExtensionManifest(manifest);
    if (extensionInstallSucceeded(result)) {
      log(`installed extension: ${manifest.name}`);
    } else {
      log(`extension install failed: ${result.error || manifest.id}`);
    }
  }
  renderExtensionList();
}

async function init() {
  setTheme(state.theme, false);
  setupComputerBrokerTransport();
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
    addModelStackOption(HF_MODELSTACK_MANIFEST, 'AgentKernel Lite 100M BitNet');
    log('AgentKernel Lite BitNet bundle attached from Hugging Face');
  }
  els.form.addEventListener('submit', submitPrompt);
  els.prompt?.addEventListener('input', renderComputerCommandPalette);
  els.prompt?.addEventListener('focus', renderComputerCommandPalette);
  els.prompt?.addEventListener('blur', () => window.setTimeout(hideComputerCommandPalette, 120));
  els.prompt?.addEventListener('keydown', (event) => {
    if (event.key === 'Tab' && completeFirstComputerCommand()) {
      event.preventDefault();
    } else if (event.key === 'Escape') {
      hideComputerCommandPalette();
    }
  });
  els.chatMode?.addEventListener('click', () => setMode('chat'));
  els.thinkMode?.addEventListener('click', () => setMode('think'));
  els.deepResearchMode?.addEventListener('click', () => setMode('deep_research'));
  els.imageMode?.addEventListener('click', () => setImageMode(!state.image.enabled));
  els.translationMode?.addEventListener('click', () => setTranslationMode(!state.translation.enabled));
  els.translationSource?.addEventListener('change', syncTranslationControls);
  els.translationTarget?.addEventListener('change', syncTranslationControls);
  els.audioTranslate?.addEventListener('click', startAudioTranslation);
  els.loadModel.addEventListener('click', () => loadModel({ force: true }).catch((error) => log(error.message || String(error))));
  els.unloadModel?.addEventListener('click', () => unloadModel());
  els.loadPack.addEventListener('click', () => loadResearchPack());
  els.persist.addEventListener('click', () => requestPersistentStorage());
  els.reset.addEventListener('click', resetChat);
  els.exportSession?.addEventListener('click', () => exportSession());
  els.importSession?.addEventListener('click', () => els.importSessionInput?.click());
  els.importSessionInput?.addEventListener('change', () => importSessionFile(els.importSessionInput.files?.[0]));
  els.installExtension?.addEventListener('click', () => installExtensionFromInput());
  els.extensionManifestUrl?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      installExtensionFromInput();
    }
  });
  els.computerConsoleButton?.addEventListener('click', () => setComputerConsoleOpen(!document.body.classList.contains('computer-console-open')));
  els.computerWorkspace?.addEventListener('change', () => setCodexWorkspace(els.computerWorkspace.value));
  els.computerProvider?.addEventListener('change', () => setComputerProvider(els.computerProvider.value));
  els.computerStart?.addEventListener('click', () => {
    const open = !els.computerNewTerminalForm?.classList.contains('open');
    els.computerNewTerminalForm?.classList.toggle('open', open);
    els.computerStart?.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) window.setTimeout(() => els.computerWorkspace?.focus(), 0);
  });
  els.computerCancelTerminal?.addEventListener('click', () => {
    els.computerNewTerminalForm?.classList.remove('open');
    els.computerStart?.setAttribute('aria-expanded', 'false');
  });
  els.computerCreateTerminal?.addEventListener('click', () => startComputerSession().then(() => {
    els.computerNewTerminalForm?.classList.remove('open');
    els.computerStart?.setAttribute('aria-expanded', 'false');
  }).catch((error) => {
    setComputerConsoleStatus(`Computer Use failed: ${error.message || String(error)}`, 'error');
    log(`Computer Use failed: ${error.message || String(error)}`);
  }));
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
  await loadCodexPairing();
  if (els.computerWorkspace) els.computerWorkspace.value = codexWorkspace();
  renderComputerConsole();
  syncImageModeControls();
  syncTranslationControls();
  loadAvailableExtensions();
  refreshAppIntegrity();
  await refreshStorage();
  syncModelControls();
  log('agent kernel lite ready');
  if (URL_PARAMS.get('autoload') !== '0') {
    state.modelAutoLoadStarted = true;
    loadModel({ auto: true }).catch((error) => {
      log(error.message || String(error));
      updateRuntimeDetail(`Runtime did not load automatically: ${error.message || String(error)}`);
      syncModelControls();
    });
  } else {
    updateRuntimeDetail('Runtime autoload is off. Load Runtime to start the model.');
  }
}

init().catch((error) => log(error.message || String(error)));
