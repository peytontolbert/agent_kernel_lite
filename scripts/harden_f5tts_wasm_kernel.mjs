import { spawnSync } from 'node:child_process';

const root = '/data/agent_kernel_lite';
const args = process.argv.slice(2);
const full = args.includes('--full');
const positionals = args.filter((arg) => !arg.startsWith('--'));
const f5Bundle = positionals[0] || '/data/resumebot/checkpoints/f5tts_peyton_q4_v0';
const vocosBundle = positionals[1] || '/data/resumebot/checkpoints/vocos_mel_24khz_q4_v0';
const refWav = positionals[2] || 'apps/mobile/www/app/voice/peyton/sample_0.wav';

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
  `${root}/tmp/f5tts_q4_forward_fixture`,
  '32',
]);
const output = requireReport(parity.reports, 'output');
if (output.mae > 0.0006 || output.maxAbs > 0.004) {
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
if (Math.abs(smoke1.checksum - 72.669288) > 0.001) {
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
  if (Math.abs(smoke2.checksum - -520.945459) > 0.001) {
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
