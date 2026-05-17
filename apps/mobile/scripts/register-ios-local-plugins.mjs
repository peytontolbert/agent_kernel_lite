import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const mobileRoot = resolve(here, '..');
const iosConfigPath = resolve(mobileRoot, 'ios', 'App', 'App', 'capacitor.config.json');
const config = JSON.parse(await readFile(iosConfigPath, 'utf8'));
const localPlugins = [];
config.packageClassList = [...localPlugins];
await writeFile(iosConfigPath, JSON.stringify(config, null, '\t') + '\n');
console.log(`Registered iOS local plugins: ${localPlugins.join(', ')}`);
