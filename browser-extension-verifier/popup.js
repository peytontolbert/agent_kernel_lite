const REPO = 'peytontolbert/agent_kernel_lite';
const RELEASE_ROOT = `https://github.com/${REPO}/releases/download`;
const REQUIRED_ASSETS = [
  { name: 'index.html', path: './index.html' },
  { name: 'agent-kernel-app.js', path: './js/agent-kernel-app.js', releaseName: 'agent-kernel-app.js' },
  {
    name: 'agent_kernel_lite_core.js',
    path: './wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core.js',
    releaseName: 'agent_kernel_lite_core.js',
  },
  {
    name: 'agent_kernel_lite_core_bg.wasm',
    path: './wasm/agent_kernel_lite_core/pkg/agent_kernel_lite_core_bg.wasm',
    releaseName: 'agent_kernel_lite_core_bg.wasm',
  },
];

const els = {
  tag: document.getElementById('releaseTag'),
  verify: document.getElementById('verifyButton'),
  copy: document.getElementById('copyButton'),
  summary: document.getElementById('summary'),
  details: document.getElementById('details'),
};

let lastReport = null;

function setSummary(kind, text) {
  els.summary.className = `summary ${kind}`;
  els.summary.textContent = text;
}

function normalizeTag(value) {
  const tag = String(value || '').trim();
  if (!tag) throw new Error('Release tag is required.');
  if (tag === 'latest' || tag === 'main' || tag === 'master') {
    throw new Error('Use an immutable release tag, not latest/main/master.');
  }
  return tag;
}

function releaseAssetUrl(tag, assetName) {
  return `${RELEASE_ROOT}/${encodeURIComponent(tag)}/${assetName}`;
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url) throw new Error('No active tab URL is available.');
  return tab;
}

function appBaseUrl(tabUrl) {
  const url = new URL(tabUrl);
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('Open the Agent Kernel Lite web app in a normal browser tab first.');
  }
  url.hash = '';
  url.search = '';
  if (!url.pathname.endsWith('/')) {
    url.pathname = url.pathname.replace(/[^/]*$/, '');
  }
  return url;
}

function bytesToHex(buffer) {
  return [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function sha256(buffer) {
  return bytesToHex(await crypto.subtle.digest('SHA-256', buffer));
}

async function fetchBytes(url) {
  const response = await fetch(url, { cache: 'no-store', credentials: 'omit' });
  if (!response.ok) throw new Error(`Fetch failed ${response.status}: ${url}`);
  return response.arrayBuffer();
}

function parseSha256Sums(text) {
  const out = new Map();
  for (const line of String(text || '').split(/\r?\n/)) {
    const match = line.match(/^([a-f0-9]{64})\s+\*?(.+)$/i);
    if (match) out.set(match[2].trim(), match[1].toLowerCase());
  }
  return out;
}

async function fetchExpectedHashes(tag) {
  const url = releaseAssetUrl(tag, 'SHA256SUMS');
  const response = await fetch(url, { cache: 'no-store', credentials: 'omit' });
  if (!response.ok) throw new Error(`Could not fetch SHA256SUMS for ${tag}: ${response.status}`);
  return {
    url,
    hashes: parseSha256Sums(await response.text()),
  };
}

function renderReport(report) {
  const rows = [];
  for (const asset of report.assets) {
    const row = document.createElement('div');
    row.className = `asset ${asset.status}`;
    const title = document.createElement('strong');
    title.textContent = `${asset.status.toUpperCase()} ${asset.name}`;
    const live = document.createElement('code');
    live.textContent = `live:     ${asset.actual || 'unavailable'}`;
    const expected = document.createElement('code');
    expected.textContent = `expected: ${asset.expected || 'missing from SHA256SUMS'}`;
    const url = document.createElement('code');
    url.textContent = asset.url;
    row.append(title, live, expected, url);
    rows.push(row);
  }
  els.details.replaceChildren(...rows);
}

function reportText(report) {
  return [
    `Agent Kernel Lite verification: ${report.status}`,
    `Release: ${report.releaseTag}`,
    `Page: ${report.pageUrl}`,
    `SHA256SUMS: ${report.sha256SumsUrl}`,
    `Checked: ${report.checkedAt}`,
    '',
    ...report.assets.flatMap((asset) => [
      `${asset.status.toUpperCase()} ${asset.name}`,
      `  live:     ${asset.actual || 'unavailable'}`,
      `  expected: ${asset.expected || 'missing'}`,
      `  url:      ${asset.url}`,
    ]),
  ].join('\n');
}

async function verify() {
  const releaseTag = normalizeTag(els.tag.value);
  await chrome.storage.local.set({ releaseTag });
  els.verify.disabled = true;
  setSummary('neutral', 'Verifying live assets...');
  els.details.textContent = '';
  try {
    const tab = await activeTab();
    const base = appBaseUrl(tab.url);
    const expected = await fetchExpectedHashes(releaseTag);
    const assets = [];
    for (const asset of REQUIRED_ASSETS) {
      const url = new URL(asset.path, base).href;
      const releaseName = asset.releaseName || asset.name;
      let actual = '';
      let status = 'bad';
      try {
        actual = await sha256(await fetchBytes(url));
        const expectedHash = expected.hashes.get(releaseName);
        status = expectedHash && actual === expectedHash ? 'ok' : 'bad';
        assets.push({ ...asset, url, actual, expected: expectedHash || '', status });
      } catch (error) {
        assets.push({ ...asset, url, actual, expected: expected.hashes.get(releaseName) || '', status: 'bad', error: error.message });
      }
    }
    const failed = assets.filter((asset) => asset.status !== 'ok');
    lastReport = {
      status: failed.length ? 'failed' : 'passed',
      releaseTag,
      pageUrl: tab.url,
      sha256SumsUrl: expected.url,
      checkedAt: new Date().toISOString(),
      assets,
    };
    renderReport(lastReport);
    setSummary(failed.length ? 'bad' : 'ok', failed.length ? `${failed.length} asset hash mismatch.` : 'All checked assets match the release.');
  } catch (error) {
    lastReport = null;
    els.details.textContent = error.message || String(error);
    setSummary('bad', 'Verification failed.');
  } finally {
    els.verify.disabled = false;
  }
}

async function init() {
  const stored = await chrome.storage.local.get({ releaseTag: 'v3' });
  els.tag.value = stored.releaseTag || 'v3';
  els.verify.addEventListener('click', () => verify());
  els.copy.addEventListener('click', async () => {
    if (!lastReport) return;
    await navigator.clipboard.writeText(reportText(lastReport));
    setSummary(lastReport.status === 'passed' ? 'ok' : 'warn', 'Report copied.');
  });
}

init().catch((error) => setSummary('bad', error.message || String(error)));
