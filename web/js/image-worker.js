import { BitDiTRuntime, ImageDevServerRuntime, SanaBrowserBundleRuntime, SanaSnapshotRuntime, imageToPngBase64, imageToSvg } from './bitdit-runtime.js?v=20260506-sana-browser-export-v1';

let loadedModelId = '';
let currentJob = null;
let bitdit = null;
const DEFAULT_IMAGE_MODEL_ID = 'agentkernel_lite_image_bitdit_hf_cifar_distilled_v1';

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function hashText(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function escapeXml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function wrapText(text, maxChars = 34, maxLines = 5) {
  const words = String(text || '').replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
  const lines = [];
  let line = '';
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (candidate.length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
    if (lines.length >= maxLines) break;
  }
  if (line && lines.length < maxLines) lines.push(line);
  return lines;
}

function hsl(h, s, l) {
  return `hsl(${Math.round(h % 360)} ${Math.round(s)}% ${Math.round(l)}%)`;
}

function buildPreviewSvg(prompt, options = {}) {
  const width = Number(options.width || 768);
  const height = Number(options.height || 768);
  const seed = Number(options.seed || hashText(prompt));
  const random = mulberry32(seed ^ hashText(prompt));
  const hue = Math.floor(random() * 360);
  const hue2 = hue + 55 + Math.floor(random() * 80);
  const hue3 = hue + 160 + Math.floor(random() * 80);
  const titleLines = wrapText(prompt, 36, 5);
  const shapes = Array.from({ length: 28 }, (_, index) => {
    const cx = Math.round(random() * width);
    const cy = Math.round(random() * height);
    const size = Math.round(22 + random() * 110);
    const opacity = (0.08 + random() * 0.22).toFixed(3);
    const color = hsl(index % 3 === 0 ? hue : index % 3 === 1 ? hue2 : hue3, 58 + random() * 24, 48 + random() * 28);
    if (index % 4 === 0) {
      return `<rect x="${cx - size / 2}" y="${cy - size / 2}" width="${size}" height="${size}" rx="${Math.round(size * 0.16)}" fill="${color}" opacity="${opacity}" transform="rotate(${Math.round(random() * 360)} ${cx} ${cy})"/>`;
    }
    return `<circle cx="${cx}" cy="${cy}" r="${Math.round(size / 2)}" fill="${color}" opacity="${opacity}"/>`;
  }).join('');
  const lines = titleLines.map((line, index) => (
    `<text x="52" y="${height - 176 + index * 30}" fill="#ffffff" fill-opacity="0.92" font-size="24" font-weight="700" font-family="Inter, system-ui, sans-serif">${escapeXml(line)}</text>`
  )).join('');
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(prompt)}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="${hsl(hue, 42, 18)}"/>
      <stop offset="52%" stop-color="${hsl(hue2, 48, 28)}"/>
      <stop offset="100%" stop-color="${hsl(hue3, 45, 16)}"/>
    </linearGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#000000" flood-opacity="0.28"/>
    </filter>
  </defs>
  <rect width="${width}" height="${height}" fill="url(#bg)"/>
  <g>${shapes}</g>
  <path d="M0 ${height * 0.18} C ${width * 0.22} ${height * 0.05}, ${width * 0.36} ${height * 0.34}, ${width * 0.56} ${height * 0.18} S ${width * 0.84} ${height * 0.02}, ${width} ${height * 0.14}" fill="none" stroke="#ffffff" stroke-opacity="0.24" stroke-width="3"/>
  <rect x="34" y="${height - 228}" width="${width - 68}" height="174" rx="18" fill="#070b10" fill-opacity="0.52" stroke="#ffffff" stroke-opacity="0.20" filter="url(#softShadow)"/>
  <text x="52" y="${height - 194}" fill="#ffffff" fill-opacity="0.62" font-size="13" font-weight="800" letter-spacing="1.5" font-family="Inter, system-ui, sans-serif">AGENTKERNEL LITE IMAGE PREVIEW</text>
  ${lines}
</svg>`;
}

self.addEventListener('message', async (event) => {
  const data = event.data || {};
  try {
    if (data.type === 'load') {
      loadedModelId = String(data.modelId || DEFAULT_IMAGE_MODEL_ID);
      self.postMessage({ type: 'status', message: `Loading ${loadedModelId}` });
      if (loadedModelId === 'agentkernel_lite_image_preview_v0') {
        await sleep(120);
        self.postMessage({ type: 'ready', modelId: loadedModelId, backend: 'preview' });
        return;
      }
      const manifestUrl = new URL(`../models/${loadedModelId}/manifest.json`, import.meta.url).href;
      const manifestResponse = await fetch(manifestUrl, { cache: 'no-store' });
      if (!manifestResponse.ok) throw new Error(`Image manifest failed: ${manifestResponse.status}`);
      const manifest = await manifestResponse.json();
      if (manifest.format === 'agentkernel-lite-image-dev-server') {
        bitdit = await ImageDevServerRuntime.fromManifest(manifest, manifestResponse.url);
      } else if (manifest.format === 'agentkernel-lite-image-sana-student-dev-snapshot') {
        bitdit = await SanaSnapshotRuntime.fromManifest(manifest, manifestResponse.url);
      } else if (manifest.format === 'agentkernel-lite-image-sana-browser') {
        bitdit = await SanaBrowserBundleRuntime.fromManifest(manifest, manifestResponse.url);
      } else if (manifest.format === 'agentkernel-lite-image-flux-packed-browser') {
        throw new Error('FLUX browser bundle export is present, but the FLUX WASM/WebGPU runtime is not wired yet.');
      } else if (manifest.format === 'agentkernel-lite-image-onnx-webgpu') {
        throw new Error('ONNX WebGPU image runtime manifest is present, but this worker does not load the ONNX image backend yet.');
      } else {
        bitdit = await BitDiTRuntime.fromManifest(manifest, manifestResponse.url);
      }
      self.postMessage({ type: 'ready', modelId: loadedModelId, backend: bitdit.backend });
      return;
    }
    if (data.type === 'cancel') {
      currentJob = null;
      return;
    }
    if (data.type === 'generate') {
      if (!loadedModelId) loadedModelId = DEFAULT_IMAGE_MODEL_ID;
      const id = String(data.id || `img_${Date.now()}`);
      currentJob = id;
      const started = performance.now();
      if (bitdit) {
        const prompt = String(data.prompt || '').trim();
        const seed = Number(data.options?.seed || hashText(prompt));
        const result = await bitdit.generate({
          prompt,
          seed,
          steps: Math.max(4, Number(data.options?.steps || 24)),
          onProgress(progress) {
            if (currentJob !== id) return;
            self.postMessage({
              type: 'progress',
              id,
              stage: `BitDiT ${progress.label || 'image'}`,
              step: progress.step,
              totalSteps: progress.total,
            });
          },
        });
        if (currentJob !== id) return;
        const outputSize = Math.max(256, Math.min(768, Number(data.options?.width || 512)));
        const imageBase64 = result.imageBase64 || await imageToPngBase64(result.image, {
          width: bitdit.config.image_size,
          height: bitdit.config.image_size,
          outputSize,
        });
        currentJob = null;
        self.postMessage({
          type: 'result',
          id,
          prompt,
          seed,
          imageBase64,
          mimeType: imageBase64 ? 'image/png' : 'image/svg+xml',
          svg: result.svg || imageToSvg(result.image, {
            width: bitdit.config.image_size,
            height: bitdit.config.image_size,
            scale: Math.max(1, Math.floor(outputSize / bitdit.config.image_size)),
            prompt,
            labelName: result.labelName,
          }),
          elapsedMs: performance.now() - started,
          metadata: {
            model: loadedModelId,
            backend: bitdit.backend,
            label: result.labelName,
            source_resolution: `${bitdit.config.image_size}x${bitdit.config.image_size}`,
            output_resolution: imageBase64 ? `${outputSize}x${outputSize}` : `${bitdit.config.image_size}x${bitdit.config.image_size}`,
            quality_tier: bitdit.config.quality_tier || (bitdit.config.image_size < 64 ? 'dev-low-resolution' : 'trained'),
            checkpoint: result.checkpoint || bitdit.config.checkpoint || '',
            training_step: result.trainingStep || bitdit.config.training_step || null,
          },
        });
        return;
      }
      const steps = Math.max(1, Number(data.options?.steps || 6));
      for (let step = 1; step <= steps; step += 1) {
        if (currentJob !== id) return;
        await sleep(95);
        self.postMessage({ type: 'progress', id, stage: 'preview synthesis', step, totalSteps: steps });
      }
      if (currentJob !== id) return;
      const prompt = String(data.prompt || '').trim();
      const seed = Number(data.options?.seed || hashText(prompt));
      const svg = buildPreviewSvg(prompt, { ...data.options, seed });
      currentJob = null;
      self.postMessage({
        type: 'result',
        id,
        prompt,
        seed,
        svg,
        elapsedMs: performance.now() - started,
        metadata: {
          model: loadedModelId,
          backend: 'preview-worker',
        },
      });
    }
  } catch (error) {
    self.postMessage({ type: 'error', id: data.id || currentJob || '', message: error.message || String(error) });
  }
});
