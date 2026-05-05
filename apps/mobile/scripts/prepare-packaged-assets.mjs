import { createWriteStream } from 'node:fs';
import { mkdir, stat, unlink } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const mobileRoot = resolve(here, '..');
const assetsRoot = resolve(mobileRoot, 'packaged-assets');

const paperPackUrl = process.env.AGENT_KERNEL_LITE_PAPERS_50K_URL
  || 'https://huggingface.co/datasets/PeytonT/paper_universe_interactive/resolve/main/interactive/papers_50000.json';
const paperPackPath = resolve(assetsRoot, 'papers_50000.json');

async function download(url, target) {
  await mkdir(dirname(target), { recursive: true });
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${url}: HTTP ${response.status}`);
  }
  try {
    await pipeline(response.body, createWriteStream(target));
  } catch (error) {
    await unlink(target).catch(() => {});
    throw error;
  }
}

await download(paperPackUrl, paperPackPath);
const info = await stat(paperPackPath);
console.log(`Prepared ${paperPackPath} (${info.size} bytes)`);
