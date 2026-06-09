import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const root = '/data/agent_kernel_lite';
const args = process.argv.slice(2);
const full = args.includes('--full');
const positionals = args.filter((arg) => !arg.startsWith('--'));
const f5Bundle = positionals[0] || 'apps/mobile/packaged-assets/peyton_voice_q4/models/f5tts_peyton_q4_v0';
const vocosBundle = positionals[1] || 'apps/mobile/packaged-assets/peyton_voice_q4/models/vocos_mel_24khz_q4_v0';
const refWav = positionals[2] || 'apps/mobile/packaged-assets/peyton_voice_q4/voice/peyton/sample_0.wav';
const fixtureDir = positionals[3] || `${root}/tmp/f5tts_q4_forward_fixture`;
const hfReleaseDir = `${root}/artifacts/hf_releases/f5tts-4bit-distill`;
const hfQualitySample = `${hfReleaseDir}/samples/BF_fullq4_surface_v2_best_nfe8_cfg2_speed115.wav`;
const hfQualitySampleSha256 = 'ca0ffaa08b99f049ded6a85fb117a3ef95cd7994b4900860dd3bb3caf668ffa3';

function resolveRepoPath(item) {
  return path.isAbsolute(item) ? item : path.join(root, item);
}

function sha256File(file) {
  return createHash('sha256').update(readFileSync(file)).digest('hex');
}

function requireSameSha(label, left, right) {
  const leftSha = sha256File(left);
  const rightSha = sha256File(right);
  if (leftSha !== rightSha) {
    throw new Error(`${label} drifted from Hugging Face release: ${leftSha} != ${rightSha}`);
  }
}

const releaseMetadata = JSON.parse(readFileSync(`${hfReleaseDir}/release_metadata.json`, 'utf8'));
if (releaseMetadata.quality_reference_sample !== 'samples/BF_fullq4_surface_v2_best_nfe8_cfg2_speed115.wav') {
  throw new Error('Hugging Face release metadata no longer points at the expected Peyton quality sample');
}
if (sha256File(hfQualitySample) !== hfQualitySampleSha256) {
  throw new Error('Hugging Face Peyton quality sample checksum changed locally');
}
if (Number(releaseMetadata.recommended_probe_settings?.nfe_steps) !== 8
    || Number(releaseMetadata.recommended_probe_settings?.cfg_strength) !== 2.0
    || Number(releaseMetadata.recommended_probe_settings?.speed) !== 1.15) {
  throw new Error('Hugging Face release probe settings must remain nfe=8 cfg=2 speed=1.15');
}

const resolvedF5Bundle = resolveRepoPath(f5Bundle);
for (const file of ['manifest.json', 'tensors.q4.bin', 'tensors.fp16.bin']) {
  requireSameSha(`packaged F5 ${file}`, path.join(resolvedF5Bundle, file), path.join(hfReleaseDir, file));
}

for (const appFile of ['web/js/agent-kernel-app.js', 'apps/mobile/www/app/js/agent-kernel-app.js']) {
  const source = readFileSync(`${root}/${appFile}`, 'utf8');
  if (!/steps:\s*8\b/.test(source)) {
    throw new Error(`${appFile} must request the clarity-first F5TTS Q4 8-step path`);
  }
  if (/steps:\s*12\b/.test(source)) {
    throw new Error(`${appFile} still contains a 12-step Peyton voice override`);
  }
  if (!/cfgStrength:\s*2(?:\.0)?\b/.test(source)) {
    throw new Error(`${appFile} must request the coherent CFG2 F5TTS path`);
  }
  if (!/speed:\s*1\.15\b/.test(source)) {
    throw new Error(`${appFile} must request speed 1.15 for the current clarity preset`);
  }
}

for (const workerFile of ['web/js/tts-worker.js', 'apps/mobile/www/app/js/tts-worker.js']) {
  const source = readFileSync(`${root}/${workerFile}`, 'utf8');
  if (!/const DEFAULT_STEPS\s*=\s*8\b/.test(source)) {
    throw new Error(`${workerFile} must default Peyton F5TTS to 8 steps`);
  }
  if (!/const DEFAULT_CFG_STRENGTH\s*=\s*2(?:\.0)?\b/.test(source)) {
    throw new Error(`${workerFile} must default Peyton F5TTS to the coherent CFG2 path`);
  }
  if (!/const SPEECH_SPEED\s*=\s*1\.15\b/.test(source)) {
    throw new Error(`${workerFile} must default Peyton F5TTS to speed 1.15`);
  }
  if (/speed100|const SPEECH_SPEED\s*=\s*1\.0\b/.test(source)) {
    throw new Error(`${workerFile} still contains stale Peyton voice speed metadata/defaults`);
  }
  if (!/GENERATED_FRAMES_PER_TEXT_BYTE\s*=\s*6\.33\b/.test(source)) {
    throw new Error(`${workerFile} must use Hugging Face quality-sample calibrated generated-frame duration`);
  }
  if (/framesPerTextByte:\s*condSeqLen\s*\//.test(source)) {
    throw new Error(`${workerFile} regressed to conditioning length as generated duration`);
  }
  const defaultProbeBytes = 47;
  const defaultProbeFrames = Math.ceil(6.33 * defaultProbeBytes / 1.15);
  if (defaultProbeFrames < 250) {
    throw new Error(`${workerFile} default Peyton probe would truncate: ${defaultProbeFrames} frames`);
  }
  const hfProbeBytes = 105;
  const hfProbeFrames = Math.ceil(6.33 * hfProbeBytes / 1.15);
  if (Math.abs(hfProbeFrames - 578) > 2) {
    throw new Error(`${workerFile} no longer matches Hugging Face quality sample frame budget: ${hfProbeFrames}`);
  }
  if (!/(preferWasm:\s*true|graphKind:\s*['"]f5['"])/.test(source) || !/with fused WASM/.test(source)) {
    throw new Error(`${workerFile} must load Peyton F5TTS through the fused WASM bundle path`);
  }
}

for (const runtimeFile of [
  'model-stack/browser/bitnet/f5tts_q4_dit_runtime.js',
  'web/vendor/model-stack-bitnet/f5tts_q4_dit_runtime.js',
  'apps/mobile/www/app/vendor/model-stack-bitnet/f5tts_q4_dit_runtime.js',
]) {
  const source = readFileSync(`${root}/${runtimeFile}`, 'utf8');
  if (!/runF5SampleMel/.test(source) || !/JavaScript model fallback is disabled/.test(source)) {
    throw new Error(`${runtimeFile} must require the fused WASM F5 sample_mel path`);
  }
}

for (const wasmSourceFile of ['model-stack/browser/bitnet_wasm/src/lib.rs']) {
  const source = readFileSync(`${root}/${wasmSourceFile}`, 'utf8');
  const canonicalSource = readFileSync('/data/transformer_10/browser/bitnet_wasm/src/lib.rs', 'utf8');
  if (source !== canonicalSource) {
    throw new Error(`${wasmSourceFile} must be synced from /data/transformer_10/browser/bitnet_wasm/src/lib.rs`);
  }
  if (!/const F5_USE_I8ACT_Q4_LINEAR:\s*bool\s*=\s*false;/.test(source)
      || !/const F5_USE_TILED_I8ACT_Q4_LINEAR:\s*bool\s*=\s*false;/.test(source)
      || !/const F5_USE_Q4ACT_Q4_LINEAR:\s*bool\s*=\s*false;/.test(source)) {
    throw new Error(`${wasmSourceFile} must keep F5 activation quantization disabled for Peyton quality`);
  }
}

function runJson(label, args, options = {}) {
  const result = spawnSync(args[0], args.slice(1), {
    cwd: root,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 32,
  });
  if (result.status !== 0) {
    process.stderr.write(result.stdout || '');
    process.stderr.write(result.stderr || '');
    throw new Error(`${label} failed with status ${result.status}`);
  }
  const text = result.stdout.trim();
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start < 0 || end < start) {
    throw new Error(`${label} did not emit JSON`);
  }
  const parsed = JSON.parse(text.slice(start, end + 1));
  if (options.print) {
    console.log(JSON.stringify({ label, ...parsed }, null, 2));
  }
  return parsed;
}

function requireReport(reports, name) {
  const report = reports.find((item) => item.name === name);
  if (!report) throw new Error(`missing report: ${name}`);
  return report;
}

const parity = runJson('forward parity', [
  'node',
  'scripts/compare_f5tts_wasm_forward.mjs',
  f5Bundle,
  fixtureDir,
  '32',
]);
const output = requireReport(parity.reports, 'output');
if (output.mae > 0.13 || output.maxAbs > 5.6) {
  throw new Error(`forward parity drift: mae=${output.mae} maxAbs=${output.maxAbs}`);
}

const profile = runJson('forward profile', [
  'node',
  'scripts/profile_f5tts_wasm_forward.mjs',
  f5Bundle,
  '347',
]);
const inputEmbedding = requireReport(profile.reports, 'input_embedding');
if (profile.totalMs > 13000) {
  throw new Error(`forward profile too slow: totalMs=${profile.totalMs}`);
}
if (inputEmbedding.ms > 2300) {
  throw new Error(`input embedding too slow: ms=${inputEmbedding.ms}`);
}

const smoke1 = runJson('peyton 1-step smoke', [
  'node',
  'examples/18_f5tts_q4_peyton_ref_smoke/run.mjs',
  f5Bundle,
  vocosBundle,
  refWav,
  'This is Peyton.',
  '256',
  '91',
  '1',
  '0',
  "Hi, I'm recording this sample to create a ",
]);
if (!smoke1.finite || smoke1.audioSamples !== 23040 || smoke1.generationMs > 16000) {
  throw new Error(`1-step smoke failed: finite=${smoke1.finite} samples=${smoke1.audioSamples} generationMs=${smoke1.generationMs}`);
}
if (Math.abs(smoke1.checksum - 73.719445) > 0.001) {
  throw new Error(`1-step smoke checksum drift: checksum=${smoke1.checksum}`);
}

let smoke2 = null;
if (full) {
  smoke2 = runJson('peyton 2-step cfg smoke', [
    'node',
    'examples/18_f5tts_q4_peyton_ref_smoke/run.mjs',
    f5Bundle,
    vocosBundle,
    refWav,
    'This is Peyton.',
    '256',
    '91',
    '2',
    '2',
    "Hi, I'm recording this sample to create a ",
  ]);
  if (!smoke2.finite || smoke2.audioSamples !== 23040 || smoke2.generationMs > 45000) {
    throw new Error(`2-step smoke failed: finite=${smoke2.finite} samples=${smoke2.audioSamples} generationMs=${smoke2.generationMs}`);
  }
  if (Math.abs(smoke2.checksum - -590.153) > 0.001) {
    throw new Error(`2-step smoke checksum drift: checksum=${smoke2.checksum}`);
  }
}

console.log(JSON.stringify({
  ok: true,
  parity: {
    outputMae: output.mae,
    outputMaxAbs: output.maxAbs,
  },
  profile: {
    totalMs: profile.totalMs,
    inputEmbeddingMs: inputEmbedding.ms,
  },
  smoke1: {
    generationMs: smoke1.generationMs,
    decodeMs: smoke1.decodeMs,
    checksum: smoke1.checksum,
  },
  smoke2: smoke2 ? {
    generationMs: smoke2.generationMs,
    decodeMs: smoke2.decodeMs,
    checksum: smoke2.checksum,
  } : null,
}, null, 2));
