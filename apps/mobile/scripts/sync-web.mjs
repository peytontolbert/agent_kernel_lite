import { createReadStream, createWriteStream } from 'node:fs';
import { access, cp, mkdir, readFile, rm, stat, unlink, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const mobileRoot = resolve(here, '..');
const repoRoot = resolve(mobileRoot, '..', '..');
const sourceWeb = resolve(repoRoot, 'web');
const bundledApp = resolve(mobileRoot, 'www', 'app');
const bundledModelName = 'agentkernel_lite_100m_bitnet_12000';
const sourceModel = resolve(sourceWeb, 'models', bundledModelName);
const bundledExtraModelNames = ['vocos_mel_24khz_q4_v0'];
const packagedAssets = resolve(mobileRoot, 'packaged-assets');
const packagedPapers = resolve(packagedAssets, 'papers_50000.json');
const packagedVoice = resolve(packagedAssets, 'peyton_voice_q4');
const voiceAssetUrl = process.env.AGENT_KERNEL_LITE_VOICE_Q4_URL
  || 'https://github.com/peytontolbert/agent_kernel_lite/releases/download/voice-q4-v0/agent-kernel-lite-peyton-voice-q4-v0.tar';

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

await rm(bundledApp, { recursive: true, force: true });
await mkdir(bundledApp, { recursive: true });
await cp(sourceWeb, bundledApp, {
  recursive: true,
  filter: (source) => {
    const normalized = source.replaceAll('\\', '/');
    if (normalized.includes('/web/models/')) return false;
    if (normalized.endsWith('/web/app-release-manifest.example.json')) return false;
    if (normalized.endsWith('/web/extensions/image_generation.dev.json')) return false;
    if (normalized.endsWith('/web/js/bitdit-runtime.js')) return false;
    if (normalized.endsWith('/web/js/image-worker.js')) return false;
    return true;
  },
});

await cp(sourceModel, resolve(bundledApp, 'models', bundledModelName), { recursive: true });
for (const modelName of bundledExtraModelNames) {
  const modelPath = resolve(sourceWeb, 'models', modelName);
  if (await exists(modelPath)) {
    await cp(modelPath, resolve(bundledApp, 'models', modelName), { recursive: true });
  }
}

if (await exists(packagedVoice)) {
  await cp(packagedVoice, bundledApp, { recursive: true });
} else if (voiceAssetUrl) {
  const tmpTar = resolve(packagedAssets, 'peyton_voice_q4.tar');
  await mkdir(packagedAssets, { recursive: true });
  const response = await fetch(voiceAssetUrl, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download Peyton voice assets: HTTP ${response.status}`);
  }
  await pipeline(response.body, createWriteStream(tmpTar));
  await untar(tmpTar, bundledApp);
} else {
  console.warn('Peyton voice assets were not bundled; AGENT_KERNEL_LITE_VOICE_Q4_URL is empty.');
}

const f5VoiceModel = resolve(bundledApp, 'models', 'f5tts_peyton_q4_v0');
if (await exists(f5VoiceModel)) {
  await chunkQ4ModelBuffer(f5VoiceModel, {
    fileKey: 'q4',
    chunksKey: 'q4_chunks',
    nbytesKey: 'q4_nbytes',
    chunkDir: 'chunks',
    chunkPrefix: 'tensors.q4',
    chunkBytes: 8 * 1024 * 1024,
  });
}

if (await exists(packagedPapers)) {
  await mkdir(resolve(bundledApp, 'packed-data'), { recursive: true });
  await cp(packagedPapers, resolve(bundledApp, 'packed-data', 'papers_50000.json'));
} else {
  console.warn(`Missing packaged 50k paper pack: ${packagedPapers}`);
  console.warn('Run npm run prepare:assets before building native release artifacts.');
}

await writeFile(
  resolve(mobileRoot, 'www', 'app-source.json'),
  JSON.stringify({
    schema: 'agent_kernel_lite_mobile_app_source',
    generated_at: new Date().toISOString(),
    source: '../../web',
    bundled_app: './app/',
    excluded: [
      'web/models/* except web/models/agentkernel_lite_100m_bitnet_12000/',
      'web/app-release-manifest.example.json',
    ],
    bundled_model: `./app/models/${bundledModelName}/manifest.json`,
    bundled_voice: {
      speaker: 'Peyton',
      f5tts_q4: './app/models/f5tts_peyton_q4_v0/manifest.json',
      f5tts_q4_asset_layout: 'chunked-q4',
      vocos_q4: './app/models/vocos_mel_24khz_q4_v0/manifest.json',
      reference_wav: './app/voice/peyton/sample_0.wav',
      vocab: './app/voice/peyton/F5TTS_Base_vocab.txt',
    },
    bundled_native_tts: null,
    bundled_paper_pack: (await exists(packagedPapers)) ? './app/packed-data/papers_50000.json' : null,
    remote_asset_policy: 'Native builds bundle the default model and 50k paper pack. Hugging Face remains the upstream source for refreshed or larger packs.',
  }, null, 2) + '\n',
);

console.log(`Synced ${sourceWeb} -> ${bundledApp}`);

function untar(archive, destination) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn('tar', ['-xf', archive, '-C', destination], { stdio: 'inherit' });
    child.on('error', rejectPromise);
    child.on('exit', (code) => {
      if (code === 0) resolvePromise();
      else rejectPromise(new Error(`tar exited with ${code}`));
    });
  });
}

async function chunkQ4ModelBuffer(modelDir, options) {
  const manifestPath = resolve(modelDir, 'manifest.json');
  if (!(await exists(manifestPath))) return;
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const files = manifest.files || {};
  if (Array.isArray(files[options.chunksKey])) return;
  if (!files[options.fileKey]) return;

  const sourcePath = resolve(modelDir, files[options.fileKey]);
  if (!(await exists(sourcePath))) return;
  const sourceStat = await stat(sourcePath);
  if (sourceStat.size <= options.chunkBytes) return;

  const outputDir = resolve(modelDir, options.chunkDir);
  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });

  const chunks = [];
  for (let offset = 0, index = 0; offset < sourceStat.size; offset += options.chunkBytes, index += 1) {
    const end = Math.min(offset + options.chunkBytes, sourceStat.size) - 1;
    const relativePath = `${options.chunkDir}/${options.chunkPrefix}.${String(index).padStart(4, '0')}.bin`;
    await pipeline(createReadStream(sourcePath, { start: offset, end }), createWriteStream(resolve(modelDir, relativePath)));
    chunks.push(relativePath);
  }

  files[options.chunksKey] = chunks;
  files[options.nbytesKey] = sourceStat.size;
  delete files[options.fileKey];
  manifest.files = files;
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  await unlink(sourcePath);
  console.log(`Chunked ${sourcePath} into ${chunks.length} files of up to ${options.chunkBytes} bytes`);
}
