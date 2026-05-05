const DEFAULT_RELEASE_MANIFEST = 'https://github.com/peytontolbert/agent_kernel_lite/releases/download/v6/app-release-manifest.json';
const RELEASE_CACHE = 'agent-kernel-lite-release-assets-v1';

const form = document.getElementById('releaseForm');
const input = document.getElementById('releaseManifestUrl');
const checkButton = document.getElementById('checkReleaseButton');
const statusEl = document.getElementById('status');
const detailsEl = document.getElementById('releaseDetails');

function setStatus(message, mode = '') {
  statusEl.textContent = message;
  statusEl.dataset.mode = mode;
}

function githubReleaseUrl(url) {
  return url.hostname === 'github.com'
    && /\/[^/]+\/[^/]+\/releases\/download\/[^/]+\/[^/]+/.test(url.pathname);
}

function huggingFaceResolveUrl(url) {
  return (url.hostname === 'huggingface.co' || url.hostname.endsWith('.huggingface.co'))
    && url.pathname.includes('/resolve/');
}

function parseManifestUrl(value) {
  const url = new URL(String(value || DEFAULT_RELEASE_MANIFEST).trim());
  if (url.protocol !== 'https:') {
    throw new Error('Release manifest must use https.');
  }
  if (!githubReleaseUrl(url)) {
    throw new Error('Release manifest must be an immutable GitHub release asset.');
  }
  return url;
}

function assetUrlAllowed(value, executable = false) {
  const url = new URL(value);
  if (url.protocol !== 'https:') return false;
  if (githubReleaseUrl(url)) return true;
  return !executable && huggingFaceResolveUrl(url);
}

function hex(buffer) {
  return [...new Uint8Array(buffer)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function sha256(data) {
  return hex(await crypto.subtle.digest('SHA-256', data));
}

async function fetchBytes(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

function releaseAssets(manifest) {
  const rows = [];
  const entry = manifest.entry || {};
  for (const [key, value] of Object.entries(entry)) {
    if (!value) continue;
    rows.push({
      key,
      url: new URL(value, input.value).toString(),
      executable: ['html', 'script', 'wasm_js', 'worker', 'module'].some((token) => key.includes(token)),
    });
  }
  for (const pack of manifest.huggingface_assets || manifest.model_assets || []) {
    if (!pack?.url) continue;
    rows.push({
      key: pack.id || pack.name || 'huggingface_asset',
      url: pack.url,
      executable: false,
    });
  }
  return rows;
}

function expectedHash(manifest, row) {
  const assets = manifest.integrity?.assets || {};
  const path = new URL(row.url).pathname.split('/').pop();
  return assets[row.key] || assets[path] || row.sha256 || '';
}

function renderManifest(manifest, rows) {
  detailsEl.innerHTML = '';
  const title = document.createElement('strong');
  title.textContent = `${manifest.repo || 'release'} ${manifest.release_tag || manifest.version || ''}`.trim();
  detailsEl.appendChild(title);
  for (const row of rows) {
    const item = document.createElement('div');
    item.className = 'asset-row';
    const name = document.createElement('span');
    name.textContent = row.key;
    const state = document.createElement('strong');
    state.textContent = row.cached ? 'cached' : 'listed';
    item.append(name, state);
    detailsEl.appendChild(item);
  }
  detailsEl.hidden = false;
}

async function cacheRelease(value, verifyOnly = false) {
  const manifestUrl = parseManifestUrl(value);
  input.value = manifestUrl.toString();
  setStatus('Fetching release manifest...', '');
  const manifestBytes = await fetchBytes(manifestUrl);
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  const rows = releaseAssets(manifest);
  if (!rows.length) throw new Error('Release manifest did not list any app assets.');
  for (const row of rows) {
    if (!assetUrlAllowed(row.url, row.executable)) {
      throw new Error(`Rejected mutable or unsupported asset URL: ${row.url}`);
    }
  }
  if (verifyOnly) {
    renderManifest(manifest, rows);
    setStatus('Release manifest is valid.', 'ok');
    return;
  }
  const cache = await caches.open(RELEASE_CACHE);
  let completed = 0;
  for (const row of rows) {
    setStatus(`Caching ${row.key}...`, '');
    const bytes = await fetchBytes(row.url);
    const expected = expectedHash(manifest, row);
    if (expected) {
      const actual = await sha256(bytes);
      if (actual !== expected) {
        throw new Error(`${row.key} hash mismatch.`);
      }
    }
    await cache.put(row.url, new Response(bytes, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Agent-Kernel-Lite-Release': manifest.release_tag || manifest.version || '',
      },
    }));
    row.cached = true;
    completed += 1;
  }
  renderManifest(manifest, rows);
  setStatus(`Cached ${completed} verified asset${completed === 1 ? '' : 's'}.`, 'ok');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  form.querySelector('button').disabled = true;
  try {
    await cacheRelease(input.value, false);
  } catch (error) {
    setStatus(error.message || String(error), 'error');
  } finally {
    form.querySelector('button').disabled = false;
  }
});

checkButton.addEventListener('click', async () => {
  checkButton.disabled = true;
  try {
    await cacheRelease(input.value, true);
  } catch (error) {
    setStatus(error.message || String(error), 'error');
  } finally {
    checkButton.disabled = false;
  }
});
