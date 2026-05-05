import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const mobileRoot = resolve(here, '..');
const repoRoot = resolve(mobileRoot, '..', '..');
const sourceWeb = resolve(repoRoot, 'web');
const bundledApp = resolve(mobileRoot, 'www', 'app');

await rm(bundledApp, { recursive: true, force: true });
await mkdir(bundledApp, { recursive: true });
await cp(sourceWeb, bundledApp, {
  recursive: true,
  filter: (source) => {
    const normalized = source.replaceAll('\\', '/');
    if (normalized.includes('/web/models/')) return false;
    if (normalized.endsWith('/web/extensions/image_generation.dev.json')) return false;
    if (normalized.endsWith('/web/js/bitdit-runtime.js')) return false;
    if (normalized.endsWith('/web/js/image-worker.js')) return false;
    return true;
  },
});

await writeFile(
  resolve(mobileRoot, 'www', 'app-source.json'),
  JSON.stringify({
    schema: 'agent_kernel_lite_mobile_app_source',
    generated_at: new Date().toISOString(),
    source: '../../web',
    bundled_app: './app/',
    excluded: ['web/models/'],
    model_and_paper_assets: 'downloaded from Hugging Face by the app runtime',
  }, null, 2) + '\n',
);

console.log(`Synced ${sourceWeb} -> ${bundledApp}`);
