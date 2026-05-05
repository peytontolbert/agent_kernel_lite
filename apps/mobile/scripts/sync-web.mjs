import { access, cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const mobileRoot = resolve(here, '..');
const repoRoot = resolve(mobileRoot, '..', '..');
const sourceWeb = resolve(repoRoot, 'web');
const bundledApp = resolve(mobileRoot, 'www', 'app');
const bundledModelName = 'agentkernel_lite_100m_bitnet_12000';
const sourceModel = resolve(sourceWeb, 'models', bundledModelName);
const packagedAssets = resolve(mobileRoot, 'packaged-assets');
const packagedPapers = resolve(packagedAssets, 'papers_50000.json');

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
    bundled_paper_pack: (await exists(packagedPapers)) ? './app/packed-data/papers_50000.json' : null,
    remote_asset_policy: 'Native builds bundle the default model and 50k paper pack. Hugging Face remains the upstream source for refreshed or larger packs.',
  }, null, 2) + '\n',
);

console.log(`Synced ${sourceWeb} -> ${bundledApp}`);
