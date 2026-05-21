const DIM = 1024;
const TEXT_DIM = 512;
const MEL_DIM = 100;
const HEADS = 16;
const HEAD_DIM = 64;
const DEPTH = 22;
const EPS = 1e-6;

export class F5TTSQ4DiTRuntime {
  constructor(bundle, options = {}) {
    this.bundle = bundle;
    this.dim = options.dim || DIM;
    this.textDim = options.textDim || TEXT_DIM;
    this.melDim = options.melDim || MEL_DIM;
    this.heads = options.heads || HEADS;
    this.headDim = options.headDim || HEAD_DIM;
    this.depth = options.depth || DEPTH;
  }

  prepareSession() {
    if (this.bundle.prepareF5Session) {
      this.bundle.prepareF5Session();
    }
  }

  forward({ x, cond, textIds, time = 0.5, dropAudioCond = false, dropText = false }) {
    if (this.bundle.runF5Forward) {
      return this.bundle.runF5Forward({ x, cond, textIds, time, dropAudioCond, dropText });
    }
    const seqLen = x.length / this.melDim;
    if (!Number.isInteger(seqLen) || seqLen <= 0) {
      throw new Error("x must be a flat Float32Array with shape [seqLen, 100]");
    }
    if (cond.length !== x.length) {
      throw new Error("cond must match x shape");
    }

    const t = this.timeEmbedding(time);
    const text = this.textEmbedding(textIds, seqLen, dropText);
    let hidden = this.inputEmbedding(x, cond, text, seqLen, dropAudioCond);

    for (let block = 0; block < this.depth; block += 1) {
      hidden = this.ditBlock(block, hidden, t, seqLen);
    }

    hidden = this.finalAdaNorm(hidden, t, seqLen);
    return this.linear("transformer.proj_out.weight", "transformer.proj_out.bias", hidden, seqLen);
  }

  sampleMel({
    condMel,
    condSeqLen,
    textIds,
    duration,
    steps = 8,
    cfgStrength = 2.0,
    swaySamplingCoef = -1.0,
    seed = 1337,
    onProgress = null,
  }) {
    if (duration < condSeqLen) {
      throw new Error("duration must be >= condSeqLen");
    }
    if (condMel.length !== condSeqLen * this.melDim) {
      throw new Error("condMel length must match condSeqLen * melDim");
    }
    if (this.bundle.runF5SampleMel) {
      return this.bundle.runF5SampleMel({
        condMel,
        condSeqLen,
        textIds,
        duration,
        steps,
        cfgStrength,
        swaySamplingCoef,
        seed,
        onProgress,
      });
    }
    throw new Error("F5TTS Q4 requires the fused WASM sample_mel kernel; JavaScript model fallback is disabled.");
  }

  timeEmbedding(time) {
    const freqDim = 256;
    const half = freqDim / 2;
    const emb = new Float32Array(freqDim);
    const factor = Math.log(10000) / (half - 1);
    for (let i = 0; i < half; i += 1) {
      const value = 1000 * time * Math.exp(i * -factor);
      emb[i] = Math.sin(value);
      emb[i + half] = Math.cos(value);
    }
    let out = this.linear("transformer.time_embed.time_mlp.0.weight", "transformer.time_embed.time_mlp.0.bias", emb, 1);
    siluInPlace(out);
    out = this.linear("transformer.time_embed.time_mlp.2.weight", "transformer.time_embed.time_mlp.2.bias", out, 1);
    return out;
  }

  textEmbedding(textIds, seqLen, dropText) {
    const weight = this.bundle.denseF32Tensor("transformer.text_embed.text_embed.weight");
    const out = new Float32Array(seqLen * this.textDim);
    for (let pos = 0; pos < seqLen; pos += 1) {
      const rawId = pos < textIds.length ? textIds[pos] : -1;
      const token = dropText ? 0 : Math.max(0, rawId + 1);
      const src = Math.min(token, weight.length / this.textDim - 1) * this.textDim;
      out.set(weight.subarray(src, src + this.textDim), pos * this.textDim);
    }
    addTextSinusoidalPos(out, seqLen, this.textDim);
    let hidden = out;
    for (let block = 0; block < 4; block += 1) {
      hidden = this.convNeXtTextBlock(block, hidden, seqLen);
    }
    return hidden;
  }

  convNeXtTextBlock(block, input, seqLen) {
    const prefix = `transformer.text_embed.text_blocks.${block}`;
    let x = depthwiseConv1dQ4(
      this.bundle,
      `${prefix}.dwconv.weight`,
      `${prefix}.dwconv.bias`,
      input,
      seqLen,
      this.textDim,
      7,
      3,
    );
    x = layerNormRows(x, this.bundle.denseF32Tensor(`${prefix}.norm.weight`), this.bundle.denseF32Tensor(`${prefix}.norm.bias`), seqLen, this.textDim);
    x = this.linear(`${prefix}.pwconv1.weight`, `${prefix}.pwconv1.bias`, x, seqLen);
    geluInPlace(x);
    applyGRN(
      x,
      seqLen,
      this.textDim * 2,
      this.bundle.denseF32Tensor(`${prefix}.grn.gamma`),
      this.bundle.denseF32Tensor(`${prefix}.grn.beta`),
    );
    x = this.linear(`${prefix}.pwconv2.weight`, `${prefix}.pwconv2.bias`, x, seqLen);
    addInPlace(x, input);
    return x;
  }

  inputEmbedding(x, cond, text, seqLen, dropAudioCond) {
    const joined = new Float32Array(seqLen * (this.melDim * 2 + this.textDim));
    for (let row = 0; row < seqLen; row += 1) {
      const dst = row * (this.melDim * 2 + this.textDim);
      joined.set(x.subarray(row * this.melDim, (row + 1) * this.melDim), dst);
      if (!dropAudioCond) {
        joined.set(cond.subarray(row * this.melDim, (row + 1) * this.melDim), dst + this.melDim);
      }
      joined.set(text.subarray(row * this.textDim, (row + 1) * this.textDim), dst + this.melDim * 2);
    }
    const projected = this.linear("transformer.input_embed.proj.weight", "transformer.input_embed.proj.bias", joined, seqLen);
    let pos = groupedConv1dQ4(
      this.bundle,
      "transformer.input_embed.conv_pos_embed.conv1d.0.weight",
      "transformer.input_embed.conv_pos_embed.conv1d.0.bias",
      projected,
      seqLen,
      this.dim,
      31,
      15,
      16,
    );
    mishInPlace(pos);
    pos = groupedConv1dQ4(
      this.bundle,
      "transformer.input_embed.conv_pos_embed.conv1d.2.weight",
      "transformer.input_embed.conv_pos_embed.conv1d.2.bias",
      pos,
      seqLen,
      this.dim,
      31,
      15,
      16,
    );
    mishInPlace(pos);
    addInPlace(pos, projected);
    return pos;
  }

  ditBlock(block, input, t, seqLen) {
    if (this.bundle.runF5DiTBlock) {
      return this.bundle.runF5DiTBlock(block, input, t, seqLen, this.dim, this.heads, this.headDim, EPS);
    }
    const prefix = `transformer.transformer_blocks.${block}`;
    const mod = this.linear(`${prefix}.attn_norm.linear.weight`, `${prefix}.attn_norm.linear.bias`, siluCopy(t), 1);
    const shiftMsa = mod.subarray(0, this.dim);
    const scaleMsa = mod.subarray(this.dim, this.dim * 2);
    const gateMsa = mod.subarray(this.dim * 2, this.dim * 3);
    const shiftMlp = mod.subarray(this.dim * 3, this.dim * 4);
    const scaleMlp = mod.subarray(this.dim * 4, this.dim * 5);
    const gateMlp = mod.subarray(this.dim * 5, this.dim * 6);

    let norm = this.bundle.runLayerNormAffine
      ? this.bundle.runLayerNormAffine(input, shiftMsa, scaleMsa, seqLen, this.dim, EPS)
      : layerNormAffineRows(input, shiftMsa, scaleMsa, seqLen, this.dim);

    let q;
    let k;
    let v;
    if (this.bundle.runQ4Linear3) {
      const qkv = this.bundle.runQ4Linear3(
        { weightName: `${prefix}.attn.to_q.weight`, biasName: `${prefix}.attn.to_q.bias` },
        { weightName: `${prefix}.attn.to_k.weight`, biasName: `${prefix}.attn.to_k.bias` },
        { weightName: `${prefix}.attn.to_v.weight`, biasName: `${prefix}.attn.to_v.bias` },
        norm,
        seqLen,
      );
      const part = seqLen * this.dim;
      q = qkv.subarray(0, part);
      k = qkv.subarray(part, part * 2);
      v = qkv.subarray(part * 2, part * 3);
    } else {
      q = this.linear(`${prefix}.attn.to_q.weight`, `${prefix}.attn.to_q.bias`, norm, seqLen);
      k = this.linear(`${prefix}.attn.to_k.weight`, `${prefix}.attn.to_k.bias`, norm, seqLen);
      v = this.linear(`${prefix}.attn.to_v.weight`, `${prefix}.attn.to_v.bias`, norm, seqLen);
    }
    applyRotary(q, k, seqLen, this.heads, this.headDim);
    let attn = this.bundle.runAttention
      ? this.bundle.runAttention(q, k, v, seqLen, seqLen, this.heads, this.headDim, false, 0)
      : attention(q, k, v, seqLen, this.heads, this.headDim);
    attn = this.linear(`${prefix}.attn.to_out.0.weight`, `${prefix}.attn.to_out.0.bias`, attn, seqLen);

    const x = this.bundle.runGatedAddRows
      ? this.bundle.runGatedAddRows(input, attn, gateMsa, seqLen, this.dim)
      : gatedAddRows(input, attn, gateMsa, seqLen, this.dim);

    norm = this.bundle.runLayerNormAffine
      ? this.bundle.runLayerNormAffine(x, shiftMlp, scaleMlp, seqLen, this.dim, EPS)
      : layerNormAffineRows(x, shiftMlp, scaleMlp, seqLen, this.dim);
    let ff = this.bundle.runQ4Mlp
      ? this.bundle.runQ4Mlp(
          { weightName: `${prefix}.ff.ff.0.0.weight`, biasName: `${prefix}.ff.ff.0.0.bias` },
          { weightName: `${prefix}.ff.ff.2.weight`, biasName: `${prefix}.ff.ff.2.bias` },
          norm,
          seqLen,
          "gelu",
        )
      : this.linear(`${prefix}.ff.ff.0.0.weight`, `${prefix}.ff.ff.0.0.bias`, norm, seqLen);
    if (!this.bundle.runQ4Mlp) {
      geluTanhInPlace(ff);
      ff = this.linear(`${prefix}.ff.ff.2.weight`, `${prefix}.ff.ff.2.bias`, ff, seqLen);
    }
    return this.bundle.runGatedAddRows
      ? this.bundle.runGatedAddRows(x, ff, gateMlp, seqLen, this.dim)
      : gatedAddRows(x, ff, gateMlp, seqLen, this.dim);
  }

  finalAdaNorm(input, t, seqLen) {
    const mod = this.linear("transformer.norm_out.linear.weight", "transformer.norm_out.linear.bias", siluCopy(t), 1);
    const scale = mod.subarray(0, this.dim);
    const shift = mod.subarray(this.dim, this.dim * 2);
    return this.bundle.runLayerNormAffine
      ? this.bundle.runLayerNormAffine(input, shift, scale, seqLen, this.dim, EPS)
      : layerNormAffineRows(input, shift, scale, seqLen, this.dim);
  }



  async sampleMelAsync({
    condMel,
    condSeqLen,
    textIds,
    duration,
    steps = 8,
    cfgStrength = 2.0,
    swaySamplingCoef = -1.0,
    seed = 1337,
    onProgress = null,
  }) {
    if (duration < condSeqLen) throw new Error("duration must be >= condSeqLen");
    if (condMel.length !== condSeqLen * this.melDim) throw new Error("condMel length must match condSeqLen * melDim");
    const cond = new Float32Array(duration * this.melDim);
    cond.set(condMel);
    let y = gaussianArray(duration * this.melDim, seed);
    const times = makeTimeGrid(steps, swaySamplingCoef);
    for (let step = 0; step < steps; step += 1) {
      const t = times[step];
      const dt = times[step + 1] - times[step];
      const pred = await this.forwardAsync({ x: y, cond, textIds, time: t, dropAudioCond: false, dropText: false });
      let flow = pred;
      if (cfgStrength >= 1e-5) {
        const nullPred = await this.forwardAsync({ x: y, cond, textIds, time: t, dropAudioCond: true, dropText: true });
        flow = pred.slice();
        for (let i = 0; i < flow.length; i += 1) flow[i] = pred[i] + (pred[i] - nullPred[i]) * cfgStrength;
      }
      for (let i = 0; i < y.length; i += 1) y[i] += dt * flow[i];
      if (typeof onProgress === "function") onProgress(step + 1, steps);
    }
    y.set(cond.subarray(0, condSeqLen * this.melDim), 0);
    return y;
  }

  async forwardAsync({ x, cond, textIds, time = 0.5, dropAudioCond = false, dropText = false }) {
    const seqLen = x.length / this.melDim;
    if (!Number.isInteger(seqLen) || seqLen <= 0) throw new Error("x must be a flat Float32Array with shape [seqLen, 100]");
    if (cond.length !== x.length) throw new Error("cond must match x shape");
    const t = await this.timeEmbeddingAsync(time);
    const text = await this.textEmbeddingAsync(textIds, seqLen, dropText);
    let hidden = await this.inputEmbeddingAsync(x, cond, text, seqLen, dropAudioCond);
    for (let block = 0; block < this.depth; block += 1) hidden = await this.ditBlockAsync(block, hidden, t, seqLen);
    hidden = await this.finalAdaNormAsync(hidden, t, seqLen);
    return this.linearAsync("transformer.proj_out.weight", "transformer.proj_out.bias", hidden, seqLen);
  }

  async timeEmbeddingAsync(time) {
    const freqDim = 256;
    const half = freqDim / 2;
    const emb = new Float32Array(freqDim);
    const factor = Math.log(10000) / (half - 1);
    for (let i = 0; i < half; i += 1) {
      const value = 1000 * time * Math.exp(i * -factor);
      emb[i] = Math.sin(value);
      emb[i + half] = Math.cos(value);
    }
    let out = await this.linearAsync("transformer.time_embed.time_mlp.0.weight", "transformer.time_embed.time_mlp.0.bias", emb, 1);
    siluInPlace(out);
    out = await this.linearAsync("transformer.time_embed.time_mlp.2.weight", "transformer.time_embed.time_mlp.2.bias", out, 1);
    return out;
  }

  async textEmbeddingAsync(textIds, seqLen, dropText) {
    const weight = this.bundle.denseF32Tensor("transformer.text_embed.text_embed.weight");
    const out = new Float32Array(seqLen * this.textDim);
    for (let pos = 0; pos < seqLen; pos += 1) {
      const rawId = pos < textIds.length ? textIds[pos] : -1;
      const token = dropText ? 0 : Math.max(0, rawId + 1);
      const src = Math.min(token, weight.length / this.textDim - 1) * this.textDim;
      out.set(weight.subarray(src, src + this.textDim), pos * this.textDim);
    }
    addTextSinusoidalPos(out, seqLen, this.textDim);
    let hidden = out;
    for (let block = 0; block < 4; block += 1) hidden = await this.convNeXtTextBlockAsync(block, hidden, seqLen);
    return hidden;
  }

  async convNeXtTextBlockAsync(block, input, seqLen) {
    const prefix = `transformer.text_embed.text_blocks.${block}`;
    let x = this.bundle.runQ4DepthwiseConv1dAsync
      ? await this.bundle.runQ4DepthwiseConv1dAsync(`${prefix}.dwconv.weight`, `${prefix}.dwconv.bias`, input, seqLen, this.textDim, 7, 3)
      : depthwiseConv1dQ4(this.bundle, `${prefix}.dwconv.weight`, `${prefix}.dwconv.bias`, input, seqLen, this.textDim, 7, 3);
    x = layerNormRows(x, this.bundle.denseF32Tensor(`${prefix}.norm.weight`), this.bundle.denseF32Tensor(`${prefix}.norm.bias`), seqLen, this.textDim);
    x = await this.linearAsync(`${prefix}.pwconv1.weight`, `${prefix}.pwconv1.bias`, x, seqLen);
    geluInPlace(x);
    applyGRN(x, seqLen, this.textDim * 2, this.bundle.denseF32Tensor(`${prefix}.grn.gamma`), this.bundle.denseF32Tensor(`${prefix}.grn.beta`));
    x = await this.linearAsync(`${prefix}.pwconv2.weight`, `${prefix}.pwconv2.bias`, x, seqLen);
    addInPlace(x, input);
    return x;
  }

  async inputEmbeddingAsync(x, cond, text, seqLen, dropAudioCond) {
    const joined = new Float32Array(seqLen * (this.melDim * 2 + this.textDim));
    for (let row = 0; row < seqLen; row += 1) {
      const dst = row * (this.melDim * 2 + this.textDim);
      joined.set(x.subarray(row * this.melDim, (row + 1) * this.melDim), dst);
      if (!dropAudioCond) joined.set(cond.subarray(row * this.melDim, (row + 1) * this.melDim), dst + this.melDim);
      joined.set(text.subarray(row * this.textDim, (row + 1) * this.textDim), dst + this.melDim * 2);
    }
    const projected = await this.linearAsync("transformer.input_embed.proj.weight", "transformer.input_embed.proj.bias", joined, seqLen);
    let pos = this.bundle.runQ4GroupedConv1dAsync
      ? await this.bundle.runQ4GroupedConv1dAsync("transformer.input_embed.conv_pos_embed.conv1d.0.weight", "transformer.input_embed.conv_pos_embed.conv1d.0.bias", projected, seqLen, this.dim, 31, 15, 16)
      : groupedConv1dQ4(this.bundle, "transformer.input_embed.conv_pos_embed.conv1d.0.weight", "transformer.input_embed.conv_pos_embed.conv1d.0.bias", projected, seqLen, this.dim, 31, 15, 16);
    mishInPlace(pos);
    pos = this.bundle.runQ4GroupedConv1dAsync
      ? await this.bundle.runQ4GroupedConv1dAsync("transformer.input_embed.conv_pos_embed.conv1d.2.weight", "transformer.input_embed.conv_pos_embed.conv1d.2.bias", pos, seqLen, this.dim, 31, 15, 16)
      : groupedConv1dQ4(this.bundle, "transformer.input_embed.conv_pos_embed.conv1d.2.weight", "transformer.input_embed.conv_pos_embed.conv1d.2.bias", pos, seqLen, this.dim, 31, 15, 16);
    mishInPlace(pos);
    addInPlace(pos, projected);
    return pos;
  }

  async ditBlockAsync(block, input, t, seqLen) {
    const prefix = `transformer.transformer_blocks.${block}`;
    const mod = await this.linearAsync(`${prefix}.attn_norm.linear.weight`, `${prefix}.attn_norm.linear.bias`, siluCopy(t), 1);
    const shiftMsa = mod.subarray(0, this.dim);
    const scaleMsa = mod.subarray(this.dim, this.dim * 2);
    const gateMsa = mod.subarray(this.dim * 2, this.dim * 3);
    const shiftMlp = mod.subarray(this.dim * 3, this.dim * 4);
    const scaleMlp = mod.subarray(this.dim * 4, this.dim * 5);
    const gateMlp = mod.subarray(this.dim * 5, this.dim * 6);
    let norm = this.bundle.runLayerNormAffineAsync
      ? await this.bundle.runLayerNormAffineAsync(input, shiftMsa, scaleMsa, seqLen, this.dim, EPS)
      : this.bundle.runLayerNormAffine ? this.bundle.runLayerNormAffine(input, shiftMsa, scaleMsa, seqLen, this.dim, EPS) : layerNormAffineRows(input, shiftMsa, scaleMsa, seqLen, this.dim);
    let q;
    let k;
    let v;
    if (this.bundle.runQ4Linear3Async) {
      const qkv = await this.bundle.runQ4Linear3Async(
        { weightName: `${prefix}.attn.to_q.weight`, biasName: `${prefix}.attn.to_q.bias` },
        { weightName: `${prefix}.attn.to_k.weight`, biasName: `${prefix}.attn.to_k.bias` },
        { weightName: `${prefix}.attn.to_v.weight`, biasName: `${prefix}.attn.to_v.bias` },
        norm,
        seqLen,
      );
      const part = seqLen * this.dim;
      q = qkv.subarray(0, part);
      k = qkv.subarray(part, part * 2);
      v = qkv.subarray(part * 2, part * 3);
    } else {
      q = await this.linearAsync(`${prefix}.attn.to_q.weight`, `${prefix}.attn.to_q.bias`, norm, seqLen);
      k = await this.linearAsync(`${prefix}.attn.to_k.weight`, `${prefix}.attn.to_k.bias`, norm, seqLen);
      v = await this.linearAsync(`${prefix}.attn.to_v.weight`, `${prefix}.attn.to_v.bias`, norm, seqLen);
    }
    let attn;
    if (this.bundle.runRotaryAttentionAsync) {
      attn = await this.bundle.runRotaryAttentionAsync(q, k, v, seqLen, this.heads, this.headDim);
    } else {
      applyRotary(q, k, seqLen, this.heads, this.headDim);
      attn = this.bundle.runAttention ? this.bundle.runAttention(q, k, v, seqLen, seqLen, this.heads, this.headDim, false, 0) : attention(q, k, v, seqLen, this.heads, this.headDim);
    }
    attn = await this.linearAsync(`${prefix}.attn.to_out.0.weight`, `${prefix}.attn.to_out.0.bias`, attn, seqLen);
    const x = this.bundle.runGatedAddRowsAsync
      ? await this.bundle.runGatedAddRowsAsync(input, attn, gateMsa, seqLen, this.dim)
      : this.bundle.runGatedAddRows ? this.bundle.runGatedAddRows(input, attn, gateMsa, seqLen, this.dim) : gatedAddRows(input, attn, gateMsa, seqLen, this.dim);
    norm = this.bundle.runLayerNormAffineAsync
      ? await this.bundle.runLayerNormAffineAsync(x, shiftMlp, scaleMlp, seqLen, this.dim, EPS)
      : this.bundle.runLayerNormAffine ? this.bundle.runLayerNormAffine(x, shiftMlp, scaleMlp, seqLen, this.dim, EPS) : layerNormAffineRows(x, shiftMlp, scaleMlp, seqLen, this.dim);
    let ff = this.bundle.runQ4MlpAsync
      ? await this.bundle.runQ4MlpAsync({ weightName: `${prefix}.ff.ff.0.0.weight`, biasName: `${prefix}.ff.ff.0.0.bias` }, { weightName: `${prefix}.ff.ff.2.weight`, biasName: `${prefix}.ff.ff.2.bias` }, norm, seqLen, "gelu")
      : await this.linearAsync(`${prefix}.ff.ff.0.0.weight`, `${prefix}.ff.ff.0.0.bias`, norm, seqLen);
    if (!this.bundle.runQ4MlpAsync) {
      geluTanhInPlace(ff);
      ff = await this.linearAsync(`${prefix}.ff.ff.2.weight`, `${prefix}.ff.ff.2.bias`, ff, seqLen);
    }
    return this.bundle.runGatedAddRowsAsync
      ? this.bundle.runGatedAddRowsAsync(x, ff, gateMlp, seqLen, this.dim)
      : this.bundle.runGatedAddRows ? this.bundle.runGatedAddRows(x, ff, gateMlp, seqLen, this.dim) : gatedAddRows(x, ff, gateMlp, seqLen, this.dim);
  }

  async finalAdaNormAsync(input, t, seqLen) {
    const mod = await this.linearAsync("transformer.norm_out.linear.weight", "transformer.norm_out.linear.bias", siluCopy(t), 1);
    const scale = mod.subarray(0, this.dim);
    const shift = mod.subarray(this.dim, this.dim * 2);
    return this.bundle.runLayerNormAffineAsync
      ? this.bundle.runLayerNormAffineAsync(input, shift, scale, seqLen, this.dim, EPS)
      : this.bundle.runLayerNormAffine ? this.bundle.runLayerNormAffine(input, shift, scale, seqLen, this.dim, EPS) : layerNormAffineRows(input, shift, scale, seqLen, this.dim);
  }

  linearAsync(weightName, biasName, input, rows) {
    return this.bundle.runQ4LinearAsync(weightName, input, rows, biasName);
  }

  linear(weightName, biasName, input, rows) {
    return this.bundle.runQ4Linear(weightName, input, rows, biasName);
  }
}

function addTextSinusoidalPos(x, seqLen, dim) {
  const half = dim / 2;
  for (let pos = 0; pos < seqLen; pos += 1) {
    const base = pos * dim;
    for (let i = 0; i < half; i += 1) {
      const inv = 1 / (10000 ** ((2 * i) / dim));
      const angle = pos * inv;
      x[base + i] += Math.cos(angle);
      x[base + i + half] += Math.sin(angle);
    }
  }
}

function makeTimeGrid(steps, swaySamplingCoef) {
  const times = new Float32Array(steps + 1);
  for (let i = 0; i <= steps; i += 1) {
    let t = i / steps;
    if (swaySamplingCoef !== null && swaySamplingCoef !== undefined) {
      t = t + swaySamplingCoef * (Math.cos((Math.PI / 2) * t) - 1 + t);
    }
    times[i] = t;
  }
  return times;
}

function gaussianArray(length, seed) {
  const out = new Float32Array(length);
  let index = 0;
  let state = seed >>> 0;
  function next() {
    state = (1664525 * state + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  }
  while (index < length) {
    const u1 = Math.max(next(), 1e-7);
    const u2 = next();
    const mag = Math.sqrt(-2 * Math.log(u1));
    out[index] = mag * Math.cos(2 * Math.PI * u2);
    if (index + 1 < length) out[index + 1] = mag * Math.sin(2 * Math.PI * u2);
    index += 2;
  }
  return out;
}

function f16ToF32(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x03ff;
  if (exp === 0) return sign * (frac ? 2 ** -14 * (frac / 1024) : 0);
  if (exp === 0x1f) return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

function q4At(packed, scales, rowSize, row, col) {
  const linear = row * rowSize + col;
  const byte = packed[linear >> 1];
  let nibble = (linear & 1) === 0 ? byte & 0x0f : byte >> 4;
  if (nibble >= 8) nibble -= 16;
  return nibble * f16ToF32(scales[row]);
}

function depthwiseConv1dQ4(bundle, weightName, biasName, input, seqLen, channels, kernel, padding) {
  const { entry, packedWeight, rowScalesF16 } = bundle.q4Tensor(weightName);
  const bias = bundle.denseF32Tensor(biasName);
  const rowSize = entry.shape.slice(1).reduce((acc, value) => acc * Number(value), 1);
  const out = new Float32Array(seqLen * channels);
  for (let pos = 0; pos < seqLen; pos += 1) {
    for (let ch = 0; ch < channels; ch += 1) {
      let sum = bias[ch] || 0;
      for (let k = 0; k < kernel; k += 1) {
        const srcPos = pos + k - padding;
        if (srcPos >= 0 && srcPos < seqLen) {
          sum += input[srcPos * channels + ch] * q4At(packedWeight, rowScalesF16, rowSize, ch, k);
        }
      }
      out[pos * channels + ch] = sum;
    }
  }
  return out;
}

function groupedConv1dQ4(bundle, weightName, biasName, input, seqLen, channels, kernel, padding, groups) {
  const { entry, packedWeight, rowScalesF16 } = bundle.q4Tensor(weightName);
  const bias = bundle.denseF32Tensor(biasName);
  const rowSize = entry.shape.slice(1).reduce((acc, value) => acc * Number(value), 1);
  const groupIn = channels / groups;
  const out = new Float32Array(seqLen * channels);
  for (let pos = 0; pos < seqLen; pos += 1) {
    for (let outCh = 0; outCh < channels; outCh += 1) {
      const group = Math.floor(outCh / groupIn);
      const inStart = group * groupIn;
      let sum = bias[outCh] || 0;
      for (let localIn = 0; localIn < groupIn; localIn += 1) {
        for (let k = 0; k < kernel; k += 1) {
          const srcPos = pos + k - padding;
          if (srcPos >= 0 && srcPos < seqLen) {
            const wCol = localIn * kernel + k;
            sum += input[srcPos * channels + inStart + localIn] * q4At(packedWeight, rowScalesF16, rowSize, outCh, wCol);
          }
        }
      }
      out[pos * channels + outCh] = sum;
    }
  }
  return out;
}

function layerNormRows(input, weight, bias, rows, cols) {
  const out = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * cols;
    let mean = 0;
    for (let col = 0; col < cols; col += 1) mean += input[offset + col];
    mean /= cols;
    let variance = 0;
    for (let col = 0; col < cols; col += 1) {
      const delta = input[offset + col] - mean;
      variance += delta * delta;
    }
    const inv = 1 / Math.sqrt(variance / cols + EPS);
    for (let col = 0; col < cols; col += 1) {
      const w = weight ? weight[col] : 1;
      const b = bias ? bias[col] : 0;
      out[offset + col] = (input[offset + col] - mean) * inv * w + b;
    }
  }
  return out;
}

function affineRowsInPlace(x, shift, scale, rows, cols) {
  for (let row = 0; row < rows; row += 1) {
    const offset = row * cols;
    for (let col = 0; col < cols; col += 1) {
      x[offset + col] = x[offset + col] * (1 + scale[col]) + shift[col];
    }
  }
}

function gatedAddRowsInPlace(dst, src, gate, rows, cols) {
  for (let row = 0; row < rows; row += 1) {
    const offset = row * cols;
    for (let col = 0; col < cols; col += 1) {
      dst[offset + col] += gate[col] * src[offset + col];
    }
  }
}

function gatedAddRows(input, src, gate, rows, cols) {
  const out = input.slice();
  gatedAddRowsInPlace(out, src, gate, rows, cols);
  return out;
}

function layerNormAffineRows(input, shift, scale, rows, cols) {
  const out = layerNormRows(input, null, null, rows, cols);
  affineRowsInPlace(out, shift, scale, rows, cols);
  return out;
}

function addInPlace(dst, src) {
  for (let i = 0; i < dst.length; i += 1) dst[i] += src[i];
}

function applyGRN(x, seqLen, dim, gamma, beta) {
  const gx = new Float32Array(dim);
  let mean = 0;
  for (let col = 0; col < dim; col += 1) {
    let sum = 0;
    for (let pos = 0; pos < seqLen; pos += 1) {
      const value = x[pos * dim + col];
      sum += value * value;
    }
    gx[col] = Math.sqrt(sum);
    mean += gx[col];
  }
  mean = mean / dim + 1e-6;
  for (let pos = 0; pos < seqLen; pos += 1) {
    const offset = pos * dim;
    for (let col = 0; col < dim; col += 1) {
      const value = x[offset + col];
      x[offset + col] = (gamma[col] || 0) * (value * (gx[col] / mean)) + (beta[col] || 0) + value;
    }
  }
}

function attention(q, k, v, seqLen, heads, headDim) {
  const dim = heads * headDim;
  const out = new Float32Array(seqLen * dim);
  const scale = 1 / Math.sqrt(headDim);
  const scores = new Float32Array(seqLen);
  for (let head = 0; head < heads; head += 1) {
    const headOffset = head * headDim;
    for (let qi = 0; qi < seqLen; qi += 1) {
      let maxScore = -Infinity;
      for (let kj = 0; kj < seqLen; kj += 1) {
        let score = 0;
        for (let d = 0; d < headDim; d += 1) {
          score += q[qi * dim + headOffset + d] * k[kj * dim + headOffset + d];
        }
        score *= scale;
        scores[kj] = score;
        if (score > maxScore) maxScore = score;
      }
      let denom = 0;
      for (let kj = 0; kj < seqLen; kj += 1) {
        const value = Math.exp(scores[kj] - maxScore);
        scores[kj] = value;
        denom += value;
      }
      for (let kj = 0; kj < seqLen; kj += 1) {
        const weight = scores[kj] / denom;
        for (let d = 0; d < headDim; d += 1) {
          out[qi * dim + headOffset + d] += weight * v[kj * dim + headOffset + d];
        }
      }
    }
  }
  return out;
}

function applyRotary(q, k, seqLen, heads, headDim) {
  const half = headDim / 2;
  const dim = heads * headDim;
  for (let pos = 0; pos < seqLen; pos += 1) {
    for (let head = 0; head < heads; head += 1) {
      const base = pos * dim + head * headDim;
      for (let i = 0; i < half; i += 1) {
        const angle = pos / (10000 ** ((2 * i) / headDim));
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        rotatePair(q, base + i, base + i + half, c, s);
        rotatePair(k, base + i, base + i + half, c, s);
      }
    }
  }
}

function rotatePair(x, left, right, c, s) {
  const a = x[left];
  const b = x[right];
  x[left] = a * c - b * s;
  x[right] = b * c + a * s;
}

function siluCopy(input) {
  const out = input.slice();
  siluInPlace(out);
  return out;
}

function siluInPlace(x) {
  for (let i = 0; i < x.length; i += 1) x[i] = x[i] / (1 + Math.exp(-x[i]));
}

function geluInPlace(x) {
  for (let i = 0; i < x.length; i += 1) {
    const value = x[i];
    x[i] = 0.5 * value * (1 + erf(value / Math.SQRT2));
  }
}

function geluTanhInPlace(x) {
  const coeff = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.length; i += 1) {
    const value = x[i];
    x[i] = 0.5 * value * (1 + Math.tanh(coeff * (value + 0.044715 * value ** 3)));
  }
}

function mishInPlace(x) {
  for (let i = 0; i < x.length; i += 1) {
    x[i] = x[i] * Math.tanh(Math.log1p(Math.exp(x[i])));
  }
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}
