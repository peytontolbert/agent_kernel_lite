use wasm_bindgen::prelude::*;

const HEADER_LEN: usize = 13;
const IDX_FORMAT_VERSION: usize = 0;
const IDX_TILE_N: usize = 1;
const IDX_TILE_K: usize = 2;
const IDX_LOGICAL_OUT: usize = 3;
const IDX_LOGICAL_IN: usize = 4;
const IDX_PADDED_IN: usize = 6;
const IDX_SCALE_GRANULARITY: usize = 7;
const IDX_SCALE_GROUP_SIZE: usize = 8;
const IDX_INTERLEAVE_MODE: usize = 9;
const IDX_SEGMENT_COUNT: usize = 11;
const OUT_TILE: usize = 8;

fn validate_header(layout_header: &[i32]) -> Result<(), JsValue> {
    if layout_header.len() < HEADER_LEN {
        return Err(JsValue::from_str("BitNet layout_header must contain at least 13 entries"));
    }
    if layout_header[IDX_FORMAT_VERSION] != 1
        || layout_header[IDX_TILE_N] != 16
        || layout_header[IDX_TILE_K] != 32
        || layout_header[IDX_INTERLEAVE_MODE] != 1
    {
        return Err(JsValue::from_str(
            "Unsupported BitNet WASM layout; expected v1 16x32 interleave mode 1",
        ));
    }
    Ok(())
}

fn resolve_scale(
    out_idx: usize,
    scale_values: &[f32],
    segment_offsets: &[i32],
    scale_granularity: usize,
    scale_group_size: usize,
    segment_count: usize,
) -> f32 {
    match scale_granularity {
        0 => scale_values.first().copied().unwrap_or(0.0),
        1 => {
            for seg in 0..segment_count {
                let start = segment_offsets.get(seg).copied().unwrap_or(0).max(0) as usize;
                let end = segment_offsets.get(seg + 1).copied().unwrap_or(0).max(0) as usize;
                if out_idx >= start && out_idx < end {
                    return scale_values.get(seg).copied().unwrap_or(0.0);
                }
            }
            0.0
        }
        2 => {
            if scale_group_size == 0 {
                0.0
            } else {
                scale_values.get(out_idx / scale_group_size).copied().unwrap_or(0.0)
            }
        }
        _ => 0.0,
    }
}

fn quant_max(bits: u32) -> f32 {
    if bits == 0 || bits >= 31 {
        return 127.0;
    }
    ((1u32 << (bits - 1)) - 1) as f32
}

fn input_value(
    input: &[f32],
    row: usize,
    col: usize,
    logical_in: usize,
    input_scales: &[f32],
    input_quant_mode: u32,
    input_quant_bits: u32,
    input_scale_rows: usize,
) -> f32 {
    let value = input[row * logical_in + col];
    if input_quant_mode == 0 {
        return value;
    }
    let scale_row = if input_scale_rows == 1 { 0 } else { row };
    let scale = input_scales.get(scale_row).copied().unwrap_or(1.0).max(1e-8);
    let qmax = quant_max(input_quant_bits);
    (value / scale).round().clamp(-qmax, qmax) * scale
}

fn decode_signed_ternary_code(code: u8) -> f32 {
    match code & 3 {
        0 => -1.0,
        2 => 1.0,
        _ => 0.0,
    }
}

fn dot_packed_row_quantized(
    input: &[f32],
    row: usize,
    logical_in: usize,
    packed_row: &[u8],
    input_scales: &[f32],
    input_quant_mode: u32,
    input_quant_bits: u32,
    input_scale_rows: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (packed_col, packed_byte) in packed_row.iter().enumerate() {
        let base_col = packed_col * 4;
        if base_col >= logical_in {
            break;
        }
        let byte = *packed_byte;
        for lane in 0..4 {
            let col = base_col + lane;
            if col >= logical_in {
                break;
            }
            let w = decode_signed_ternary_code(byte >> (lane * 2));
            if w != 0.0 {
                acc += input_value(
                    input,
                    row,
                    col,
                    logical_in,
                    input_scales,
                    input_quant_mode,
                    input_quant_bits,
                    input_scale_rows,
                ) * w;
            }
        }
    }
    acc
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_packed_row_noquant_simd(input_row: &[f32], packed_row: &[u8], logical_in: usize) -> f32 {
    use core::arch::wasm32::*;

    let packed_cols = logical_in / 4;
    let mut sum = f32x4_splat(0.0);
    for packed_col in 0..packed_cols {
        let byte = *packed_row.get_unchecked(packed_col);
        let weights = f32x4(
            decode_signed_ternary_code(byte),
            decode_signed_ternary_code(byte >> 2),
            decode_signed_ternary_code(byte >> 4),
            decode_signed_ternary_code(byte >> 6),
        );
        let x = v128_load(input_row.as_ptr().add(packed_col * 4) as *const v128);
        sum = f32x4_add(sum, f32x4_mul(x, weights));
    }

    let mut acc = f32x4_extract_lane::<0>(sum)
        + f32x4_extract_lane::<1>(sum)
        + f32x4_extract_lane::<2>(sum)
        + f32x4_extract_lane::<3>(sum);
    let base_col = packed_cols * 4;
    if base_col < logical_in {
        let byte = *packed_row.get_unchecked(packed_cols);
        for lane in 0..(logical_in - base_col) {
            acc += input_row[base_col + lane] * decode_signed_ternary_code(byte >> (lane * 2));
        }
    }
    acc
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn dot_packed_row_noquant(input_row: &[f32], packed_row: &[u8], logical_in: usize) -> f32 {
    unsafe { dot_packed_row_noquant_simd(input_row, packed_row, logical_in) }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn dot_packed_row_noquant(input_row: &[f32], packed_row: &[u8], logical_in: usize) -> f32 {
    let mut acc = 0.0f32;
    for (packed_col, packed_byte) in packed_row.iter().enumerate() {
        let base_col = packed_col * 4;
        if base_col >= logical_in {
            break;
        }
        let byte = *packed_byte;
        let remaining = (logical_in - base_col).min(4);
        if remaining > 0 {
            acc += input_row[base_col] * decode_signed_ternary_code(byte);
        }
        if remaining > 1 {
            acc += input_row[base_col + 1] * decode_signed_ternary_code(byte >> 2);
        }
        if remaining > 2 {
            acc += input_row[base_col + 2] * decode_signed_ternary_code(byte >> 4);
        }
        if remaining > 3 {
            acc += input_row[base_col + 3] * decode_signed_ternary_code(byte >> 6);
        }
    }
    acc
}

#[wasm_bindgen]
pub fn bitnet_linear_f32(
    input: &[f32],
    packed_weight: &[u8],
    scale_values: &[f32],
    segment_offsets: &[i32],
    bias_values: &[f32],
    layout_header: &[i32],
    input_scales: &[f32],
    rows: usize,
    input_quant_mode: u32,
    input_quant_bits: u32,
    input_scale_rows: usize,
) -> Result<Vec<f32>, JsValue> {
    validate_header(layout_header)?;
    bitnet_linear_impl(
        input,
        packed_weight,
        scale_values,
        segment_offsets,
        bias_values,
        layout_header,
        input_scales,
        rows,
        input_quant_mode,
        input_quant_bits,
        input_scale_rows,
    )
}

fn bitnet_linear_impl(
    input: &[f32],
    packed_weight: &[u8],
    scale_values: &[f32],
    segment_offsets: &[i32],
    bias_values: &[f32],
    layout_header: &[i32],
    input_scales: &[f32],
    rows: usize,
    input_quant_mode: u32,
    input_quant_bits: u32,
    input_scale_rows: usize,
) -> Result<Vec<f32>, JsValue> {
    let logical_out = layout_header[IDX_LOGICAL_OUT].max(0) as usize;
    let logical_in = layout_header[IDX_LOGICAL_IN].max(0) as usize;
    let padded_in = layout_header[IDX_PADDED_IN].max(0) as usize;
    let scale_granularity = layout_header[IDX_SCALE_GRANULARITY].max(0) as usize;
    let scale_group_size = layout_header[IDX_SCALE_GROUP_SIZE].max(0) as usize;
    let segment_count = layout_header[IDX_SEGMENT_COUNT].max(0) as usize;
    if logical_in == 0 || logical_out == 0 || rows == 0 {
        return Ok(Vec::new());
    }
    if input.len() != rows * logical_in {
        return Err(JsValue::from_str("BitNet input length does not match rows * logical_in"));
    }
    let row_stride_bytes = padded_in / 4;
    if packed_weight.len() < logical_out * row_stride_bytes {
        return Err(JsValue::from_str("BitNet packed_weight is shorter than layout requires"));
    }

    let mut output = vec![0.0f32; rows * logical_out];
    let packed_cols = logical_in.div_ceil(4);
    for row in 0..rows {
        let input_row = &input[row * logical_in..(row + 1) * logical_in];
        for out_base in (0..logical_out).step_by(OUT_TILE) {
            let out_end = (out_base + OUT_TILE).min(logical_out);
            for out_idx in out_base..out_end {
                let row_base = out_idx * row_stride_bytes;
                let packed_row = &packed_weight[row_base..row_base + packed_cols];
                let acc = if input_quant_mode == 0 {
                    dot_packed_row_noquant(input_row, packed_row, logical_in)
                } else {
                    dot_packed_row_quantized(
                        input,
                        row,
                        logical_in,
                        packed_row,
                        input_scales,
                        input_quant_mode,
                        input_quant_bits,
                        input_scale_rows,
                    )
                };
                let scale = resolve_scale(
                    out_idx,
                    scale_values,
                    segment_offsets,
                    scale_granularity,
                    scale_group_size,
                    segment_count,
                );
                let bias = bias_values.get(out_idx).copied().unwrap_or(0.0);
                output[row * logical_out + out_idx] = acc * scale + bias;
            }
        }
    }
    Ok(output)
}

#[derive(Clone)]
#[wasm_bindgen]
pub struct BitnetLinearHandle {
    bias_values: Vec<f32>,
    input_scales: Vec<f32>,
    pos_offsets: Vec<u32>,
    pos_indices: Vec<u16>,
    neg_offsets: Vec<u32>,
    neg_indices: Vec<u16>,
    row_scales: Vec<f32>,
    input_quant_mode: u32,
    input_quant_bits: u32,
    input_scale_rows: usize,
    logical_out: usize,
    logical_in: usize,
}

fn build_sparse_indices(
    packed_weight: &[u8],
    logical_out: usize,
    logical_in: usize,
    padded_in: usize,
) -> (Vec<u32>, Vec<u16>, Vec<u32>, Vec<u16>) {
    let row_stride_bytes = padded_in / 4;
    let packed_cols = logical_in.div_ceil(4);
    let mut pos_offsets = Vec::with_capacity(logical_out + 1);
    let mut neg_offsets = Vec::with_capacity(logical_out + 1);
    let mut pos_indices = Vec::new();
    let mut neg_indices = Vec::new();
    pos_offsets.push(0);
    neg_offsets.push(0);
    for out_idx in 0..logical_out {
        let row_base = out_idx * row_stride_bytes;
        let row = &packed_weight[row_base..row_base + packed_cols.min(packed_weight.len().saturating_sub(row_base))];
        for (packed_col, packed_byte) in row.iter().enumerate() {
            let base_col = packed_col * 4;
            for lane in 0..4 {
                let col = base_col + lane;
                if col >= logical_in {
                    break;
                }
                let code = (packed_byte >> (lane * 2)) & 3;
                if code == 0 {
                    neg_indices.push(col as u16);
                } else if code == 2 {
                    pos_indices.push(col as u16);
                }
            }
        }
        pos_offsets.push(pos_indices.len() as u32);
        neg_offsets.push(neg_indices.len() as u32);
    }
    (pos_offsets, pos_indices, neg_offsets, neg_indices)
}

fn build_row_scales(
    logical_out: usize,
    scale_values: &[f32],
    segment_offsets: &[i32],
    scale_granularity: usize,
    scale_group_size: usize,
    segment_count: usize,
) -> Vec<f32> {
    (0..logical_out)
        .map(|out_idx| {
            resolve_scale(
                out_idx,
                scale_values,
                segment_offsets,
                scale_granularity,
                scale_group_size,
                segment_count,
            )
        })
        .collect()
}

fn layer_norm_one_into(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    cols: usize,
    eps: f32,
    output: &mut Vec<f32>,
) -> Result<(), JsValue> {
    if cols == 0 {
        output.clear();
        return Ok(());
    }
    if input.len() < cols || weight.len() < cols {
        return Err(JsValue::from_str("layer norm one input shape mismatch"));
    }
    output.resize(cols, 0.0);
    let mean = input.iter().take(cols).copied().sum::<f32>() / cols as f32;
    let mut variance = 0.0f32;
    for value in input.iter().take(cols) {
        let delta = *value - mean;
        variance += delta * delta;
    }
    let inv = 1.0 / (variance / cols as f32 + eps).sqrt();
    for col in 0..cols {
        let b = bias.get(col).copied().unwrap_or(0.0);
        output[col] = (input[col] - mean) * inv * weight[col] + b;
    }
    Ok(())
}

fn add_in_place(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn to_head_major_cache(row_major: &[f32], kv_len: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let model_dim = n_heads * head_dim;
    let mut out = vec![0.0f32; kv_len * model_dim];
    for head in 0..n_heads {
        let head_out = head * kv_len * head_dim;
        let head_in = head * head_dim;
        for pos in 0..kv_len {
            let src = pos * model_dim + head_in;
            let dst = head_out + pos * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&row_major[src..src + head_dim]);
        }
    }
    out
}

#[inline(always)]
fn dot_scaled_64(a: &[f32], b: &[f32], scale: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    while i < 64 {
        sum += a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3]
            + a[i + 4] * b[i + 4]
            + a[i + 5] * b[i + 5]
            + a[i + 6] * b[i + 6]
            + a[i + 7] * b[i + 7];
        i += 8;
    }
    sum * scale
}

#[inline(always)]
fn add_weighted_64(output: &mut [f32], values: &[f32], weight: f32) {
    let mut i = 0usize;
    while i < 64 {
        output[i] += weight * values[i];
        output[i + 1] += weight * values[i + 1];
        output[i + 2] += weight * values[i + 2];
        output[i + 3] += weight * values[i + 3];
        output[i + 4] += weight * values[i + 4];
        output[i + 5] += weight * values[i + 5];
        output[i + 6] += weight * values[i + 6];
        output[i + 7] += weight * values[i + 7];
        i += 8;
    }
}

fn apply_rotary_one(q: &mut [f32], k: &mut [f32], position: usize, n_heads: usize, head_dim: usize, base_theta: f32) {
    if base_theta <= 0.0 || head_dim % 2 != 0 {
        return;
    }
    let half = head_dim / 2;
    for head in 0..n_heads {
        let base_offset = head * head_dim;
        for i in 0..half {
            let inv_freq = 1.0 / base_theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = position as f32 * inv_freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let left = base_offset + i;
            let right = base_offset + i + half;
            let q1 = q[left];
            let q2 = q[right];
            let k1 = k[left];
            let k2 = k[right];
            q[left] = q1 * cos - q2 * sin;
            q[right] = q2 * cos + q1 * sin;
            k[left] = k1 * cos - k2 * sin;
            k[right] = k2 * cos + k1 * sin;
        }
    }
}

#[wasm_bindgen]
pub struct DecoderLayerHandle {
    self_q: BitnetLinearHandle,
    self_k: BitnetLinearHandle,
    self_v: BitnetLinearHandle,
    self_o: BitnetLinearHandle,
    self_mlp_in: BitnetLinearHandle,
    self_mlp_out: BitnetLinearHandle,
    cross_q: BitnetLinearHandle,
    cross_k: BitnetLinearHandle,
    cross_v: BitnetLinearHandle,
    cross_o: BitnetLinearHandle,
    cross_mlp_in: BitnetLinearHandle,
    cross_mlp_out: BitnetLinearHandle,
    self_n1_weight: Vec<f32>,
    self_n1_bias: Vec<f32>,
    self_n2_weight: Vec<f32>,
    self_n2_bias: Vec<f32>,
    cross_n1_weight: Vec<f32>,
    cross_n1_bias: Vec<f32>,
    cross_n2_weight: Vec<f32>,
    cross_n2_bias: Vec<f32>,
    activation: String,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    rotary_base: f32,
    self_k_cache: Vec<f32>,
    self_v_cache: Vec<f32>,
    self_len: usize,
    cross_k_cache: Vec<f32>,
    cross_v_cache: Vec<f32>,
    cross_len: usize,
    self_scores: Vec<f32>,
    cross_scores: Vec<f32>,
    norm_buf: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    attn_buf: Vec<f32>,
    proj_buf: Vec<f32>,
}

#[wasm_bindgen]
impl DecoderLayerHandle {
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(constructor)]
    pub fn new(
        self_q: &BitnetLinearHandle,
        self_k: &BitnetLinearHandle,
        self_v: &BitnetLinearHandle,
        self_o: &BitnetLinearHandle,
        self_mlp_in: &BitnetLinearHandle,
        self_mlp_out: &BitnetLinearHandle,
        cross_q: &BitnetLinearHandle,
        cross_k: &BitnetLinearHandle,
        cross_v: &BitnetLinearHandle,
        cross_o: &BitnetLinearHandle,
        cross_mlp_in: &BitnetLinearHandle,
        cross_mlp_out: &BitnetLinearHandle,
        self_n1_weight: Vec<f32>,
        self_n1_bias: Vec<f32>,
        self_n2_weight: Vec<f32>,
        self_n2_bias: Vec<f32>,
        cross_n1_weight: Vec<f32>,
        cross_n1_bias: Vec<f32>,
        cross_n2_weight: Vec<f32>,
        cross_n2_bias: Vec<f32>,
        activation: String,
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        rotary_base: f32,
    ) -> DecoderLayerHandle {
        DecoderLayerHandle {
            self_q: self_q.clone(),
            self_k: self_k.clone(),
            self_v: self_v.clone(),
            self_o: self_o.clone(),
            self_mlp_in: self_mlp_in.clone(),
            self_mlp_out: self_mlp_out.clone(),
            cross_q: cross_q.clone(),
            cross_k: cross_k.clone(),
            cross_v: cross_v.clone(),
            cross_o: cross_o.clone(),
            cross_mlp_in: cross_mlp_in.clone(),
            cross_mlp_out: cross_mlp_out.clone(),
            self_n1_weight,
            self_n1_bias,
            self_n2_weight,
            self_n2_bias,
            cross_n1_weight,
            cross_n1_bias,
            cross_n2_weight,
            cross_n2_bias,
            activation,
            d_model,
            n_heads,
            head_dim,
            rotary_base,
            self_k_cache: Vec::new(),
            self_v_cache: Vec::new(),
            self_len: 0,
            cross_k_cache: Vec::new(),
            cross_v_cache: Vec::new(),
            cross_len: 0,
            self_scores: Vec::new(),
            cross_scores: Vec::new(),
            norm_buf: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            attn_buf: Vec::new(),
            proj_buf: Vec::new(),
        }
    }

    pub fn self_len(&self) -> usize {
        self.self_len
    }

    pub fn clone_cache(&self) -> DecoderLayerHandle {
        DecoderLayerHandle {
            self_q: self.self_q.clone(),
            self_k: self.self_k.clone(),
            self_v: self.self_v.clone(),
            self_o: self.self_o.clone(),
            self_mlp_in: self.self_mlp_in.clone(),
            self_mlp_out: self.self_mlp_out.clone(),
            cross_q: self.cross_q.clone(),
            cross_k: self.cross_k.clone(),
            cross_v: self.cross_v.clone(),
            cross_o: self.cross_o.clone(),
            cross_mlp_in: self.cross_mlp_in.clone(),
            cross_mlp_out: self.cross_mlp_out.clone(),
            self_n1_weight: self.self_n1_weight.clone(),
            self_n1_bias: self.self_n1_bias.clone(),
            self_n2_weight: self.self_n2_weight.clone(),
            self_n2_bias: self.self_n2_bias.clone(),
            cross_n1_weight: self.cross_n1_weight.clone(),
            cross_n1_bias: self.cross_n1_bias.clone(),
            cross_n2_weight: self.cross_n2_weight.clone(),
            cross_n2_bias: self.cross_n2_bias.clone(),
            activation: self.activation.clone(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            rotary_base: self.rotary_base,
            self_k_cache: self.self_k_cache.clone(),
            self_v_cache: self.self_v_cache.clone(),
            self_len: self.self_len,
            cross_k_cache: self.cross_k_cache.clone(),
            cross_v_cache: self.cross_v_cache.clone(),
            cross_len: self.cross_len,
            self_scores: self.self_scores.clone(),
            cross_scores: self.cross_scores.clone(),
            norm_buf: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            attn_buf: Vec::new(),
            proj_buf: Vec::new(),
        }
    }

    pub fn next(&mut self, input: &[f32], memory: &[f32], memory_len: usize) -> Result<Vec<f32>, JsValue> {
        if input.len() != self.d_model || memory.len() < memory_len * self.d_model {
            return Err(JsValue::from_str("DecoderLayerHandle next shape mismatch"));
        }

        layer_norm_one_into(input, &self.self_n1_weight, &self.self_n1_bias, self.d_model, 1e-5, &mut self.norm_buf)?;
        self.self_q.run_one_into(&self.norm_buf, &mut self.q_buf)?;
        self.self_k.run_one_into(&self.norm_buf, &mut self.k_buf)?;
        self.self_v.run_one_into(&self.norm_buf, &mut self.v_buf)?;
        apply_rotary_one(&mut self.q_buf, &mut self.k_buf, self.self_len, self.n_heads, self.head_dim, self.rotary_base);
        self.self_k_cache.extend_from_slice(&self.k_buf);
        self.self_v_cache.extend_from_slice(&self.v_buf);
        self.self_len += 1;
        attention_one_into(&self.q_buf, &self.self_k_cache, &self.self_v_cache, self.self_len, self.n_heads, self.head_dim, &mut self.self_scores, &mut self.attn_buf)?;
        self.self_o.run_one_into(&self.attn_buf, &mut self.proj_buf)?;
        let mut x = input.to_vec();
        add_in_place(&mut x, &self.proj_buf);

        layer_norm_one_into(&x, &self.self_n2_weight, &self.self_n2_bias, self.d_model, 1e-5, &mut self.norm_buf)?;
        let mlp = bitnet_mlp_f32(&self.self_mlp_in, &self.self_mlp_out, &self.norm_buf, 1, &self.activation)?;
        add_in_place(&mut x, &mlp);

        layer_norm_one_into(&x, &self.cross_n1_weight, &self.cross_n1_bias, self.d_model, 1e-5, &mut self.norm_buf)?;
        self.cross_q.run_one_into(&self.norm_buf, &mut self.q_buf)?;
        if self.cross_len != memory_len {
            let cross_k_row_major = self.cross_k.run_impl(memory, memory_len)?;
            let cross_v_row_major = self.cross_v.run_impl(memory, memory_len)?;
            self.cross_k_cache = to_head_major_cache(&cross_k_row_major, memory_len, self.n_heads, self.head_dim);
            self.cross_v_cache = to_head_major_cache(&cross_v_row_major, memory_len, self.n_heads, self.head_dim);
            self.cross_len = memory_len;
        }
        attention_one_head_major_into(&self.q_buf, &self.cross_k_cache, &self.cross_v_cache, self.cross_len, self.n_heads, self.head_dim, &mut self.cross_scores, &mut self.attn_buf)?;
        self.cross_o.run_one_into(&self.attn_buf, &mut self.proj_buf)?;
        add_in_place(&mut x, &self.proj_buf);

        layer_norm_one_into(&x, &self.cross_n2_weight, &self.cross_n2_bias, self.d_model, 1e-5, &mut self.norm_buf)?;
        let mlp = bitnet_mlp_f32(&self.cross_mlp_in, &self.cross_mlp_out, &self.norm_buf, 1, &self.activation)?;
        add_in_place(&mut x, &mlp);
        Ok(x)
    }
}

#[wasm_bindgen]
impl BitnetLinearHandle {
    #[wasm_bindgen(constructor)]
    pub fn new(
        packed_weight: Vec<u8>,
        scale_values: Vec<f32>,
        segment_offsets: Vec<i32>,
        bias_values: Vec<f32>,
        layout_header: Vec<i32>,
        input_scales: Vec<f32>,
        input_quant_mode: u32,
        input_quant_bits: u32,
        input_scale_rows: usize,
    ) -> Result<BitnetLinearHandle, JsValue> {
        validate_header(&layout_header)?;
        let logical_out = layout_header[IDX_LOGICAL_OUT].max(0) as usize;
        let logical_in = layout_header[IDX_LOGICAL_IN].max(0) as usize;
        let padded_in = layout_header[IDX_PADDED_IN].max(0) as usize;
        let scale_granularity = layout_header[IDX_SCALE_GRANULARITY].max(0) as usize;
        let scale_group_size = layout_header[IDX_SCALE_GROUP_SIZE].max(0) as usize;
        let segment_count = layout_header[IDX_SEGMENT_COUNT].max(0) as usize;
        let (pos_offsets, pos_indices, neg_offsets, neg_indices) =
            build_sparse_indices(&packed_weight, logical_out, logical_in, padded_in);
        let row_scales = build_row_scales(
            logical_out,
            &scale_values,
            &segment_offsets,
            scale_granularity,
            scale_group_size,
            segment_count,
        );
        Ok(BitnetLinearHandle {
            bias_values,
            input_scales,
            pos_offsets,
            pos_indices,
            neg_offsets,
            neg_indices,
            row_scales,
            input_quant_mode,
            input_quant_bits,
            input_scale_rows,
            logical_out,
            logical_in,
        })
    }

    pub fn run(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        self.run_impl(input, rows)
    }
}

impl BitnetLinearHandle {
    fn run_impl(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        if input.len() != rows * self.logical_in {
            return Err(JsValue::from_str("BitnetLinearHandle input shape mismatch"));
        }
        self.run_sparse_impl(input, rows)
    }

    fn run_one_into(&self, input: &[f32], output: &mut Vec<f32>) -> Result<(), JsValue> {
        if input.len() != self.logical_in {
            return Err(JsValue::from_str("BitnetLinearHandle one input shape mismatch"));
        }
        output.resize(self.logical_out, 0.0);
        if self.input_quant_mode == 0 {
            self.run_sparse_noquant_one_into(input, output);
            return Ok(());
        }
        let fallback = self.run_sparse_impl(input, 1)?;
        output.copy_from_slice(&fallback);
        Ok(())
    }

    fn run_sparse_impl(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        if self.input_quant_mode == 0 {
            return Ok(if rows == 1 {
                self.run_sparse_noquant_one(input)
            } else {
                self.run_sparse_noquant_rows(input, rows)
            });
        }
        let mut output = vec![0.0f32; rows * self.logical_out];
        for row in 0..rows {
            for out_idx in 0..self.logical_out {
                let mut acc = 0.0f32;
                let pos_start = self.pos_offsets[out_idx] as usize;
                let pos_end = self.pos_offsets[out_idx + 1] as usize;
                let neg_start = self.neg_offsets[out_idx] as usize;
                let neg_end = self.neg_offsets[out_idx + 1] as usize;
                for col in &self.pos_indices[pos_start..pos_end] {
                    acc += input_value(
                        input,
                        row,
                        *col as usize,
                        self.logical_in,
                        &self.input_scales,
                        self.input_quant_mode,
                        self.input_quant_bits,
                        self.input_scale_rows,
                    );
                }
                for col in &self.neg_indices[neg_start..neg_end] {
                    acc -= input_value(
                        input,
                        row,
                        *col as usize,
                        self.logical_in,
                        &self.input_scales,
                        self.input_quant_mode,
                        self.input_quant_bits,
                        self.input_scale_rows,
                    );
                }
                let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
                output[row * self.logical_out + out_idx] = acc * self.row_scales[out_idx] + bias;
            }
        }
        Ok(output)
    }

    fn run_sparse_noquant_one(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.logical_out];
        self.run_sparse_noquant_one_into(input, &mut output);
        output
    }

    fn run_sparse_noquant_one_into(&self, input: &[f32], output: &mut [f32]) {
        for out_idx in 0..self.logical_out {
            let mut acc = 0.0f32;
            let pos_start = self.pos_offsets[out_idx] as usize;
            let pos_end = self.pos_offsets[out_idx + 1] as usize;
            let neg_start = self.neg_offsets[out_idx] as usize;
            let neg_end = self.neg_offsets[out_idx + 1] as usize;
            for col in &self.pos_indices[pos_start..pos_end] {
                acc += input[*col as usize];
            }
            for col in &self.neg_indices[neg_start..neg_end] {
                acc -= input[*col as usize];
            }
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            output[out_idx] = acc * self.row_scales[out_idx] + bias;
        }
    }

    #[inline(always)]
    fn score_sparse_noquant_one(&self, input: &[f32], out_idx: usize) -> f32 {
        let mut acc = 0.0f32;
        let pos_start = self.pos_offsets[out_idx] as usize;
        let pos_end = self.pos_offsets[out_idx + 1] as usize;
        let neg_start = self.neg_offsets[out_idx] as usize;
        let neg_end = self.neg_offsets[out_idx + 1] as usize;
        for col in &self.pos_indices[pos_start..pos_end] {
            acc += input[*col as usize];
        }
        for col in &self.neg_indices[neg_start..neg_end] {
            acc -= input[*col as usize];
        }
        let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
        acc * self.row_scales[out_idx] + bias
    }

    fn run_sparse_noquant_rows(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; rows * self.logical_out];
        for row in 0..rows {
            let input_row = &input[row * self.logical_in..(row + 1) * self.logical_in];
            let output_offset = row * self.logical_out;
            for out_idx in 0..self.logical_out {
                let mut acc = 0.0f32;
                let pos_start = self.pos_offsets[out_idx] as usize;
                let pos_end = self.pos_offsets[out_idx + 1] as usize;
                let neg_start = self.neg_offsets[out_idx] as usize;
                let neg_end = self.neg_offsets[out_idx + 1] as usize;
                for col in &self.pos_indices[pos_start..pos_end] {
                    acc += input_row[*col as usize];
                }
                for col in &self.neg_indices[neg_start..neg_end] {
                    acc -= input_row[*col as usize];
                }
                let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
                output[output_offset + out_idx] = acc * self.row_scales[out_idx] + bias;
            }
        }
        output
    }
}

#[wasm_bindgen]
pub struct TokenSample {
    token_id: u32,
    probability: f32,
    top_probability: f32,
    rank: u32,
}

#[wasm_bindgen]
impl TokenSample {
    #[wasm_bindgen(getter)]
    pub fn token_id(&self) -> u32 {
        self.token_id
    }

    #[wasm_bindgen(getter)]
    pub fn probability(&self) -> f32 {
        self.probability
    }

    #[wasm_bindgen(getter)]
    pub fn top_probability(&self) -> f32 {
        self.top_probability
    }

    #[wasm_bindgen(getter)]
    pub fn rank(&self) -> u32 {
        self.rank
    }
}

#[wasm_bindgen]
pub fn bitnet_sample_token_f32(
    lm_head: &BitnetLinearHandle,
    hidden: &[f32],
    generated_ids: &[u32],
    blocked_ids: &[u32],
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    random_value: f32,
) -> Result<TokenSample, JsValue> {
    if hidden.len() != lm_head.logical_in {
        return Err(JsValue::from_str("bitnet_sample_token_f32 hidden shape mismatch"));
    }
    let vocab_len = lm_head.logical_out;
    let mut repeated_mask = vec![false; vocab_len];
    for token_id in generated_ids {
        let idx = *token_id as usize;
        if idx < vocab_len {
            repeated_mask[idx] = true;
        }
    }
    let mut blocked_mask = vec![false; vocab_len];
    for token_id in blocked_ids {
        let idx = *token_id as usize;
        if idx < vocab_len {
            blocked_mask[idx] = true;
        }
    }
    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(vocab_len);
    let penalty = repetition_penalty.max(1.0);
    if lm_head.input_quant_mode == 0 {
        for idx in 0..vocab_len {
            if blocked_mask[idx] {
                continue;
            }
            let mut value = lm_head.score_sparse_noquant_one(hidden, idx);
            if repeated_mask[idx] {
                value = if value >= 0.0 { value / penalty } else { value * penalty };
            }
            if value.is_finite() {
                candidates.push((idx as u32, value));
            }
        }
    } else {
        let logits = lm_head.run_sparse_impl(hidden, 1)?;
        for (idx, raw) in logits.iter().enumerate() {
            if blocked_mask[idx] {
                continue;
            }
            let mut value = *raw;
            if repeated_mask[idx] {
                value = if value >= 0.0 { value / penalty } else { value * penalty };
            }
            if value.is_finite() {
                candidates.push((idx as u32, value));
            }
        }
    }
    if candidates.is_empty() {
        return Ok(TokenSample { token_id: 2, probability: 1.0, top_probability: 1.0, rank: 1 });
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    if temperature <= 0.001 {
        return Ok(TokenSample {
            token_id: candidates[0].0,
            probability: 1.0,
            top_probability: 1.0,
            rank: 1,
        });
    }
    let temp = temperature.max(1e-4);
    let max_value = candidates[0].1 / temp;
    let mut total = 0.0f32;
    let mut weights = Vec::with_capacity(candidates.len());
    for (_, value) in &candidates {
        let weight = (*value / temp - max_value).exp();
        weights.push(weight);
        total += weight;
    }
    let target_p = top_p.clamp(0.01, 1.0);
    let mut kept_end = 0usize;
    let mut kept_total = 0.0f32;
    for weight in &weights {
        kept_total += *weight;
        kept_end += 1;
        if kept_total / total.max(1e-12) >= target_p {
            break;
        }
    }
    let top_probability = weights[0] / kept_total.max(1e-12);
    let mut sample = random_value.clamp(0.0, 0.99999994) * kept_total.max(1e-12);
    for idx in 0..kept_end {
        sample -= weights[idx];
        if sample <= 0.0 {
            return Ok(TokenSample {
                token_id: candidates[idx].0,
                probability: weights[idx] / kept_total.max(1e-12),
                top_probability,
                rank: (idx + 1) as u32,
            });
        }
    }
    Ok(TokenSample {
        token_id: candidates[0].0,
        probability: top_probability,
        top_probability,
        rank: 1,
    })
}

#[wasm_bindgen]
pub fn bitnet_linear2_f32(
    first: &BitnetLinearHandle,
    second: &BitnetLinearHandle,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>, JsValue> {
    let a = first.run_impl(input, rows)?;
    let b = second.run_impl(input, rows)?;
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(&a);
    out.extend_from_slice(&b);
    Ok(out)
}

#[wasm_bindgen]
pub fn bitnet_linear3_f32(
    first: &BitnetLinearHandle,
    second: &BitnetLinearHandle,
    third: &BitnetLinearHandle,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>, JsValue> {
    let a = first.run_impl(input, rows)?;
    let b = second.run_impl(input, rows)?;
    let c = third.run_impl(input, rows)?;
    let mut out = Vec::with_capacity(a.len() + b.len() + c.len());
    out.extend_from_slice(&a);
    out.extend_from_slice(&b);
    out.extend_from_slice(&c);
    Ok(out)
}

#[wasm_bindgen]
pub fn bitnet_mlp_f32(
    w_in: &BitnetLinearHandle,
    w_out: &BitnetLinearHandle,
    input: &[f32],
    rows: usize,
    activation: &str,
) -> Result<Vec<f32>, JsValue> {
    let hidden = w_in.run_impl(input, rows)?;
    let out_cols = w_out.logical_in;
    let activated = if hidden.len() == rows * out_cols * 2 {
        gated_activation_impl(&hidden, rows, out_cols, activation)?
    } else {
        activate_impl(&hidden, activation)
    };
    w_out.run_impl(&activated, rows)
}

#[wasm_bindgen]
pub fn layer_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Vec<f32>, JsValue> {
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    if input.len() < rows * cols || weight.len() < cols {
        return Err(JsValue::from_str("layer_norm_f32 input shape mismatch"));
    }
    let mut output = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let row_offset = row * cols;
        let row_values = &input[row_offset..row_offset + cols];
        let mean = row_values.iter().copied().sum::<f32>() / cols as f32;
        let mut variance = 0.0f32;
        for value in row_values {
            let delta = *value - mean;
            variance += delta * delta;
        }
        let inv = 1.0 / (variance / cols as f32 + eps).sqrt();
        for col in 0..cols {
            let b = bias.get(col).copied().unwrap_or(0.0);
            output[row_offset + col] = (input[row_offset + col] - mean) * inv * weight[col] + b;
        }
    }
    Ok(output)
}

#[wasm_bindgen]
pub fn attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    n_heads: usize,
    head_dim: usize,
    causal: bool,
    past_len: usize,
) -> Result<Vec<f32>, JsValue> {
    attention_impl(q, k, v, q_len, kv_len, n_heads, head_dim, causal, past_len)
}

fn attention_impl(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    n_heads: usize,
    head_dim: usize,
    causal: bool,
    past_len: usize,
) -> Result<Vec<f32>, JsValue> {
    if q_len == 0 || kv_len == 0 || n_heads == 0 || head_dim == 0 {
        return Ok(Vec::new());
    }
    let model_dim = n_heads * head_dim;
    if q.len() < q_len * model_dim || k.len() < kv_len * model_dim || v.len() < kv_len * model_dim {
        return Err(JsValue::from_str("attention_f32 input shape mismatch"));
    }
    let mut output = vec![0.0f32; q_len * model_dim];
    let mut scores = vec![0.0f32; kv_len];
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for head in 0..n_heads {
        for qi in 0..q_len {
            let mut max_score = f32::NEG_INFINITY;
            let q_base = qi * model_dim + head * head_dim;
            for kj in 0..kv_len {
                let mut score = if causal && kj > past_len + qi { -1.0e30 } else { 0.0 };
                if score > -1.0e20 {
                    let k_base = kj * model_dim + head * head_dim;
                    for dim in 0..head_dim {
                        score += q[q_base + dim] * k[k_base + dim] * scale;
                    }
                }
                scores[kj] = score;
                if score > max_score {
                    max_score = score;
                }
            }
            let mut denom = 0.0f32;
            for score in scores.iter_mut().take(kv_len) {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            let denom = denom.max(1.0e-20);
            let out_base = qi * model_dim + head * head_dim;
            for dim in 0..head_dim {
                let mut sum = 0.0f32;
                for kj in 0..kv_len {
                    let v_base = kj * model_dim + head * head_dim;
                    sum += (scores[kj] / denom) * v[v_base + dim];
                }
                output[out_base + dim] = sum;
            }
        }
    }
    Ok(output)
}

fn attention_one_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_len: usize,
    n_heads: usize,
    head_dim: usize,
    scores: &mut Vec<f32>,
    output: &mut Vec<f32>,
) -> Result<(), JsValue> {
    if kv_len == 0 || n_heads == 0 || head_dim == 0 {
        output.clear();
        return Ok(());
    }
    let model_dim = n_heads * head_dim;
    if q.len() < model_dim || k.len() < kv_len * model_dim || v.len() < kv_len * model_dim {
        return Err(JsValue::from_str("attention one input shape mismatch"));
    }
    scores.resize(kv_len, 0.0);
    output.resize(model_dim, 0.0);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for head in 0..n_heads {
        let q_base = head * head_dim;
        let mut max_score = f32::NEG_INFINITY;
        for kj in 0..kv_len {
            let k_base = kj * model_dim + head * head_dim;
            let score = if head_dim == 64 {
                dot_scaled_64(&q[q_base..q_base + 64], &k[k_base..k_base + 64], scale)
            } else {
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    score += q[q_base + dim] * k[k_base + dim] * scale;
                }
                score
            };
            scores[kj] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut denom = 0.0f32;
        for score in scores.iter_mut().take(kv_len) {
            *score = (*score - max_score).exp();
            denom += *score;
        }
        let denom = denom.max(1.0e-20);
        let out_base = head * head_dim;
        for dim in 0..head_dim {
            output[out_base + dim] = 0.0;
        }
        for kj in 0..kv_len {
            let weight = scores[kj] / denom;
            let v_base = kj * model_dim + head * head_dim;
            if head_dim == 64 {
                add_weighted_64(&mut output[out_base..out_base + 64], &v[v_base..v_base + 64], weight);
            } else {
                for dim in 0..head_dim {
                    output[out_base + dim] += weight * v[v_base + dim];
                }
            }
        }
    }
    Ok(())
}

fn attention_one_head_major_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_len: usize,
    n_heads: usize,
    head_dim: usize,
    scores: &mut Vec<f32>,
    output: &mut Vec<f32>,
) -> Result<(), JsValue> {
    if kv_len == 0 || n_heads == 0 || head_dim == 0 {
        output.clear();
        return Ok(());
    }
    let model_dim = n_heads * head_dim;
    if q.len() < model_dim || k.len() < kv_len * model_dim || v.len() < kv_len * model_dim {
        return Err(JsValue::from_str("head-major attention one input shape mismatch"));
    }
    scores.resize(kv_len, 0.0);
    output.resize(model_dim, 0.0);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for head in 0..n_heads {
        let q_base = head * head_dim;
        let cache_base = head * kv_len * head_dim;
        let mut max_score = f32::NEG_INFINITY;
        for kj in 0..kv_len {
            let k_base = cache_base + kj * head_dim;
            let score = if head_dim == 64 {
                dot_scaled_64(&q[q_base..q_base + 64], &k[k_base..k_base + 64], scale)
            } else {
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    score += q[q_base + dim] * k[k_base + dim] * scale;
                }
                score
            };
            scores[kj] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut denom = 0.0f32;
        for score in scores.iter_mut().take(kv_len) {
            *score = (*score - max_score).exp();
            denom += *score;
        }
        let denom = denom.max(1.0e-20);
        let out_base = head * head_dim;
        for dim in 0..head_dim {
            output[out_base + dim] = 0.0;
        }
        for kj in 0..kv_len {
            let weight = scores[kj] / denom;
            let v_base = cache_base + kj * head_dim;
            if head_dim == 64 {
                add_weighted_64(&mut output[out_base..out_base + 64], &v[v_base..v_base + 64], weight);
            } else {
                for dim in 0..head_dim {
                    output[out_base + dim] += weight * v[v_base + dim];
                }
            }
        }
    }
    Ok(())
}

#[wasm_bindgen]
pub struct AttentionKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
    n_heads: usize,
    head_dim: usize,
}

#[wasm_bindgen]
impl AttentionKvCache {
    #[wasm_bindgen(constructor)]
    pub fn new(n_heads: usize, head_dim: usize) -> AttentionKvCache {
        AttentionKvCache {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
            n_heads,
            head_dim,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.len = 0;
    }

    pub fn clone_cache(&self) -> AttentionKvCache {
        AttentionKvCache {
            k: self.k.clone(),
            v: self.v.clone(),
            len: self.len,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
        }
    }

    pub fn set_cross(&mut self, k: &[f32], v: &[f32], kv_len: usize) -> Result<(), JsValue> {
        let model_dim = self.n_heads * self.head_dim;
        if k.len() < kv_len * model_dim || v.len() < kv_len * model_dim {
            return Err(JsValue::from_str("AttentionKvCache set_cross shape mismatch"));
        }
        self.k.clear();
        self.v.clear();
        self.k.extend_from_slice(&k[..kv_len * model_dim]);
        self.v.extend_from_slice(&v[..kv_len * model_dim]);
        self.len = kv_len;
        Ok(())
    }

    pub fn append_self_attention(
        &mut self,
        q: &[f32],
        k_new: &[f32],
        v_new: &[f32],
        q_len: usize,
        causal: bool,
    ) -> Result<Vec<f32>, JsValue> {
        let model_dim = self.n_heads * self.head_dim;
        if q.len() < q_len * model_dim || k_new.len() < q_len * model_dim || v_new.len() < q_len * model_dim {
            return Err(JsValue::from_str("AttentionKvCache append_self_attention shape mismatch"));
        }
        let past_len = self.len;
        self.k.extend_from_slice(&k_new[..q_len * model_dim]);
        self.v.extend_from_slice(&v_new[..q_len * model_dim]);
        self.len += q_len;
        attention_impl(q, &self.k, &self.v, q_len, self.len, self.n_heads, self.head_dim, causal, past_len)
    }

    pub fn attention(&self, q: &[f32], q_len: usize, causal: bool, past_len: usize) -> Result<Vec<f32>, JsValue> {
        attention_impl(q, &self.k, &self.v, q_len, self.len, self.n_heads, self.head_dim, causal, past_len)
    }
}

fn gelu_scalar(value: f32) -> f32 {
    0.5 * value * (1.0 + (core::f32::consts::FRAC_2_SQRT_PI * (value + 0.044715 * value * value * value)).tanh())
}

fn silu_scalar(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn activate_impl(input: &[f32], activation: &str) -> Vec<f32> {
    let use_gelu = activation.eq_ignore_ascii_case("gelu");
    input
        .iter()
        .map(|value| if use_gelu { gelu_scalar(*value) } else { silu_scalar(*value) })
        .collect()
}

fn gated_activation_impl(
    input: &[f32],
    rows: usize,
    cols: usize,
    activation: &str,
) -> Result<Vec<f32>, JsValue> {
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    if input.len() < rows * cols * 2 {
        return Err(JsValue::from_str("gated_activation_f32 input shape mismatch"));
    }
    let gate_name = activation.to_ascii_lowercase();
    let use_gelu = gate_name == "geglu";
    let mut output = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let input_offset = row * cols * 2;
        let output_offset = row * cols;
        for col in 0..cols {
            let a = input[input_offset + col];
            let b = input[input_offset + cols + col];
            let activated = if use_gelu { gelu_scalar(a) } else { silu_scalar(a) };
            output[output_offset + col] = activated * b;
        }
    }
    Ok(output)
}

#[wasm_bindgen]
pub fn activate_f32(input: &[f32], activation: &str) -> Vec<f32> {
    activate_impl(input, activation)
}

#[wasm_bindgen]
pub fn gated_activation_f32(
    input: &[f32],
    rows: usize,
    cols: usize,
    activation: &str,
) -> Result<Vec<f32>, JsValue> {
    gated_activation_impl(input, rows, cols, activation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_tiny_linear() {
        let input = [2.0f32, 3.0, 5.0, 7.0];
        // Codes per row: [-1, 0, +1, 0].
        let mut packed_weight = [0u8; 8];
        packed_weight[0] = 0b01_10_01_00u8;
        let scales = [0.5f32];
        let offsets = [0i32, 1];
        let header = [1, 16, 32, 1, 4, 16, 32, 0, 1, 1, 0, 1, 0];
        let y = bitnet_linear_f32(
            &input,
            &packed_weight,
            &scales,
            &offsets,
            &[],
            &header,
            &[1.0],
            1,
            0,
            8,
            1,
        )
        .unwrap();
        assert_eq!(y.len(), 1);
        assert!((y[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn decodes_two_output_rows_with_tile_loop() {
        let input = [2.0f32, 3.0, 5.0, 7.0];
        let mut packed_weight = [0u8; 16];
        packed_weight[0] = 0b01_10_01_00u8; // -2 + 5 = 3
        packed_weight[8] = 0b10_00_10_01u8; // 0 + 3 - 5 + 7 = 5
        let scales = [1.0f32, 2.0];
        let offsets = [0i32, 1, 2];
        let header = [1, 16, 32, 2, 4, 16, 32, 1, 1, 1, 0, 2, 0];
        let y = bitnet_linear_f32(
            &input,
            &packed_weight,
            &scales,
            &offsets,
            &[0.0, 1.0],
            &header,
            &[1.0],
            1,
            0,
            8,
            1,
        )
        .unwrap();
        assert_eq!(y.len(), 2);
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 11.0).abs() < 1e-6);
    }
}
