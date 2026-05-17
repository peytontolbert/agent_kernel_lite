const SAMPLE_RATE = 24000;
const N_FFT = 1024;
const HOP = 256;
const BINS = N_FFT / 2 + 1;
const MEL_DIM = 100;

export function decodeWavMono(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (readAscii(view, 0, 4) !== "RIFF" || readAscii(view, 8, 4) !== "WAVE") {
    throw new Error("expected RIFF/WAVE wav");
  }

  let offset = 12;
  let format = null;
  let dataOffset = -1;
  let dataSize = 0;
  while (offset + 8 <= view.byteLength) {
    const id = readAscii(view, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const payload = offset + 8;
    if (id === "fmt ") {
      format = {
        audioFormat: view.getUint16(payload, true),
        channels: view.getUint16(payload + 2, true),
        sampleRate: view.getUint32(payload + 4, true),
        bitsPerSample: view.getUint16(payload + 14, true),
      };
    } else if (id === "data") {
      dataOffset = payload;
      dataSize = size;
    }
    offset = payload + size + (size & 1);
  }
  if (!format || dataOffset < 0) throw new Error("missing wav fmt/data chunk");
  if (format.sampleRate !== SAMPLE_RATE) {
    throw new Error(`expected ${SAMPLE_RATE} Hz wav, got ${format.sampleRate}; resampling is not implemented in this smoke`);
  }
  const frames = dataSize / (format.bitsPerSample / 8) / format.channels;
  const samples = new Float32Array(frames);
  for (let i = 0; i < frames; i += 1) {
    let sum = 0;
    for (let ch = 0; ch < format.channels; ch += 1) {
      const byteOffset = dataOffset + (i * format.channels + ch) * (format.bitsPerSample / 8);
      if (format.audioFormat === 1 && format.bitsPerSample === 16) {
        sum += view.getInt16(byteOffset, true) / 32768;
      } else if (format.audioFormat === 3 && format.bitsPerSample === 32) {
        sum += view.getFloat32(byteOffset, true);
      } else {
        throw new Error(`unsupported wav encoding format=${format.audioFormat} bits=${format.bitsPerSample}`);
      }
    }
    samples[i] = sum / format.channels;
  }
  return { samples, sampleRate: format.sampleRate };
}

export function vocosMelFromMono(samples, bundle, options = {}) {
  const maxFrames = options.maxFrames || Math.max(1, Math.floor(samples.length / HOP));
  const melFilter = bundle.denseF32Tensor("feature_extractor.mel_spec.mel_scale.fb");
  const window = bundle.denseF32Tensor("feature_extractor.mel_spec.spectrogram.window");
  const padded = reflectPad(samples, N_FFT / 2);
  const frames = Math.min(maxFrames, Math.max(1, Math.floor((padded.length - N_FFT) / HOP) + 1));
  const mel = new Float32Array(frames * MEL_DIM);
  const real = new Float32Array(BINS);
  const imag = new Float32Array(BINS);

  for (let frame = 0; frame < frames; frame += 1) {
    const start = frame * HOP;
    real.fill(0);
    imag.fill(0);
    for (let bin = 0; bin < BINS; bin += 1) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < N_FFT; n += 1) {
        const value = padded[start + n] * window[n];
        const angle = (-2 * Math.PI * bin * n) / N_FFT;
        re += value * Math.cos(angle);
        im += value * Math.sin(angle);
      }
      real[bin] = re;
      imag[bin] = im;
    }
    for (let melBin = 0; melBin < MEL_DIM; melBin += 1) {
      let energy = 0;
      for (let bin = 0; bin < BINS; bin += 1) {
        const magnitude = Math.sqrt(real[bin] * real[bin] + imag[bin] * imag[bin]);
        energy += magnitude * melFilter[bin * MEL_DIM + melBin];
      }
      mel[frame * MEL_DIM + melBin] = Math.log(Math.max(energy, 1e-5));
    }
  }
  return { mel, frames };
}

function reflectPad(samples, pad) {
  const out = new Float32Array(samples.length + pad * 2);
  for (let i = 0; i < pad; i += 1) {
    out[i] = samples[Math.min(samples.length - 1, pad - i)];
    out[out.length - 1 - i] = samples[Math.max(0, samples.length - 2 - i)];
  }
  out.set(samples, pad);
  return out;
}

function readAscii(view, offset, length) {
  let value = "";
  for (let i = 0; i < length; i += 1) value += String.fromCharCode(view.getUint8(offset + i));
  return value;
}
