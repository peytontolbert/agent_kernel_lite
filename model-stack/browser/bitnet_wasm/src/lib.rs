use std::collections::HashMap;
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Date, js_name = now)]
    fn date_now() -> f64;
}

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
const F5_USE_GROUPED_I8ACT_Q4_LINEAR: bool = false;
const F5_USE_I8ACT_Q4_LINEAR: bool = false;
const F5_USE_TILED_I8ACT_Q4_LINEAR: bool = false;
const F5_USE_Q4ACT_Q4_LINEAR: bool = false;
const F5_GROUPED_I8ACT_GROUP: usize = 64;

#[inline(always)]
fn f5_use_i8act_q4_linear() -> bool {
    F5_USE_GROUPED_I8ACT_Q4_LINEAR || F5_USE_I8ACT_Q4_LINEAR || F5_USE_TILED_I8ACT_Q4_LINEAR
}
static Q4_BYTE_DOT_TABLE: OnceLock<Box<[i16; 65536]>> = OnceLock::new();
static VOCOS_HANN_1024: OnceLock<Box<[f32]>> = OnceLock::new();
static VOCOS_BIT_REVERSE_1024: OnceLock<Box<[usize]>> = OnceLock::new();

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
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot_scaled_64_simd(a, b, scale) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
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
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_scaled_64_simd(a: &[f32], b: &[f32], scale: f32) -> f32 {
    use core::arch::wasm32::*;

    let mut sum = f32x4_splat(0.0);
    let mut i = 0usize;
    while i + 7 < 64 {
        let av = v128_load(a.as_ptr().add(i) as *const v128);
        let bv = v128_load(b.as_ptr().add(i) as *const v128);
        let av_next = v128_load(a.as_ptr().add(i + 4) as *const v128);
        let bv_next = v128_load(b.as_ptr().add(i + 4) as *const v128);
        sum = f32x4_add(f32x4_add(sum, f32x4_mul(av, bv)), f32x4_mul(av_next, bv_next));
        i += 8;
    }
    while i < 64 {
        let av = v128_load(a.as_ptr().add(i) as *const v128);
        let bv = v128_load(b.as_ptr().add(i) as *const v128);
        sum = f32x4_add(sum, f32x4_mul(av, bv));
        i += 4;
    }
    (f32x4_extract_lane::<0>(sum)
        + f32x4_extract_lane::<1>(sum)
        + f32x4_extract_lane::<2>(sum)
        + f32x4_extract_lane::<3>(sum))
        * scale
}

#[inline(always)]
fn add_weighted_64(output: &mut [f32], values: &[f32], weight: f32) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            add_weighted_64_simd(output, values, weight);
        }
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
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
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn add_weighted_64_simd(output: &mut [f32], values: &[f32], weight: f32) {
    use core::arch::wasm32::*;

    let w = f32x4_splat(weight);
    let mut i = 0usize;
    while i + 7 < 64 {
        let out = v128_load(output.as_ptr().add(i) as *const v128);
        let val = v128_load(values.as_ptr().add(i) as *const v128);
        let out_next = v128_load(output.as_ptr().add(i + 4) as *const v128);
        let val_next = v128_load(values.as_ptr().add(i + 4) as *const v128);
        v128_store(output.as_mut_ptr().add(i) as *mut v128, f32x4_add(out, f32x4_mul(w, val)));
        v128_store(output.as_mut_ptr().add(i + 4) as *mut v128, f32x4_add(out_next, f32x4_mul(w, val_next)));
        i += 8;
    }
    while i < 64 {
        let out = v128_load(output.as_ptr().add(i) as *const v128);
        let val = v128_load(values.as_ptr().add(i) as *const v128);
        v128_store(output.as_mut_ptr().add(i) as *mut v128, f32x4_add(out, f32x4_mul(w, val)));
        i += 4;
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
pub fn layer_norm_affine_f32(
    input: &[f32],
    shift: &[f32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Vec<f32>, JsValue> {
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let mut output = vec![0.0f32; rows * cols];
    layer_norm_affine_into(input, shift, scale, rows, cols, eps, &mut output)?;
    Ok(output)
}

fn layer_norm_affine_into(
    input: &[f32],
    shift: &[f32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    output: &mut [f32],
) -> Result<(), JsValue> {
    if input.len() < rows * cols || shift.len() < cols || scale.len() < cols || output.len() < rows * cols {
        return Err(JsValue::from_str("layer_norm_affine_f32 input shape mismatch"));
    }
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
            let normalized = (input[row_offset + col] - mean) * inv;
            output[row_offset + col] = normalized * (1.0 + scale[col]) + shift[col];
        }
    }
    Ok(())
}

#[wasm_bindgen]
pub fn gated_add_rows_f32(
    input: &[f32],
    src: &[f32],
    gate: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, JsValue> {
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let mut output = vec![0.0f32; rows * cols];
    gated_add_rows_into(input, src, gate, rows, cols, &mut output)?;
    Ok(output)
}

fn gated_add_rows_into(
    input: &[f32],
    src: &[f32],
    gate: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> Result<(), JsValue> {
    if input.len() < rows * cols || src.len() < rows * cols || gate.len() < cols || output.len() < rows * cols {
        return Err(JsValue::from_str("gated_add_rows_f32 input shape mismatch"));
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        for row in 0..rows {
            let offset = row * cols;
            let mut col = 0usize;
            while col + 3 < cols {
                let x = v128_load(input.as_ptr().add(offset + col) as *const v128);
                let y = v128_load(src.as_ptr().add(offset + col) as *const v128);
                let g = v128_load(gate.as_ptr().add(col) as *const v128);
                v128_store(
                    output.as_mut_ptr().add(offset + col) as *mut v128,
                    f32x4_add(x, f32x4_mul(g, y)),
                );
                col += 4;
            }
            while col < cols {
                *output.get_unchecked_mut(offset + col) =
                    *input.get_unchecked(offset + col) + *gate.get_unchecked(col) * *src.get_unchecked(offset + col);
                col += 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        for row in 0..rows {
            let offset = row * cols;
            for col in 0..cols {
                output[offset + col] = input[offset + col] + gate[col] * src[offset + col];
            }
        }
    }
    Ok(())
}

#[wasm_bindgen]
pub fn q4_conv1d_f32(
    input: &[f32],
    packed_weight: &[u8],
    row_scales_f16: &[u16],
    bias_values: &[f32],
    seq_len: usize,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    padding: usize,
) -> Result<Vec<f32>, JsValue> {
    if input.len() != seq_len * in_channels {
        return Err(JsValue::from_str("q4_conv1d_f32 input shape mismatch"));
    }
    let row_size = in_channels * kernel;
    let value_count = out_channels * row_size;
    if packed_weight.len() < value_count.div_ceil(2) || row_scales_f16.len() < out_channels {
        return Err(JsValue::from_str("q4_conv1d_f32 weight shape mismatch"));
    }
    let mut output = vec![0.0f32; seq_len * out_channels];
    for out_ch in 0..out_channels {
        let scale = f16_to_f32(row_scales_f16[out_ch]);
        let bias = bias_values.get(out_ch).copied().unwrap_or(0.0);
        let row_offset = out_ch * row_size;
        for pos in 0..seq_len {
            let mut sum = bias;
            let k_start = padding.saturating_sub(pos);
            let k_end = kernel.min(seq_len + padding - pos);
            for in_ch in 0..in_channels {
                let input_base = in_ch;
                let weight_base = row_offset + in_ch * kernel;
                for k in k_start..k_end {
                    let src_pos = pos + k - padding;
                    let linear = weight_base + k;
                    let packed = packed_weight[linear >> 1];
                    let nibble = if linear & 1 == 0 { packed & 0x0f } else { packed >> 4 };
                    let signed = if nibble >= 8 { nibble as i8 - 16 } else { nibble as i8 };
                    sum += input[src_pos * in_channels + input_base] * signed as f32 * scale;
                }
            }
            output[pos * out_channels + out_ch] = sum;
        }
    }
    Ok(output)
}

#[wasm_bindgen]
pub fn q4_depthwise_conv1d_f32(
    input: &[f32],
    packed_weight: &[u8],
    row_scales_f16: &[u16],
    bias_values: &[f32],
    seq_len: usize,
    channels: usize,
    kernel: usize,
    padding: usize,
) -> Result<Vec<f32>, JsValue> {
    if seq_len == 0 || channels == 0 || kernel == 0 {
        return Ok(Vec::new());
    }
    if input.len() < seq_len * channels {
        return Err(JsValue::from_str("q4_depthwise_conv1d_f32 input shape mismatch"));
    }
    let value_count = channels * kernel;
    if packed_weight.len() < value_count.div_ceil(2) || row_scales_f16.len() < channels {
        return Err(JsValue::from_str("q4_depthwise_conv1d_f32 weight shape mismatch"));
    }
    let mut output = vec![0.0f32; seq_len * channels];
    for pos in 0..seq_len {
        for ch in 0..channels {
            let mut sum = bias_values.get(ch).copied().unwrap_or(0.0);
            let scale = f16_to_f32(row_scales_f16[ch]);
            for k in 0..kernel {
                let src_pos = pos as isize + k as isize - padding as isize;
                if src_pos < 0 || src_pos >= seq_len as isize {
                    continue;
                }
                let weight_index = ch * kernel + k;
                let packed = packed_weight[weight_index >> 1];
                let nibble = if weight_index & 1 == 0 { packed & 0x0f } else { packed >> 4 };
                let q = nibble as i8;
                let q = if q >= 8 { q - 16 } else { q } as f32;
                sum += input[src_pos as usize * channels + ch] * q * scale;
            }
            output[pos * channels + ch] = sum;
        }
    }
    Ok(output)
}

#[wasm_bindgen]
pub fn q4_grouped_conv1d_f32(
    input: &[f32],
    packed_weight: &[u8],
    row_scales_f16: &[u16],
    bias_values: &[f32],
    seq_len: usize,
    channels: usize,
    kernel: usize,
    padding: usize,
    groups: usize,
) -> Result<Vec<f32>, JsValue> {
    if seq_len == 0 || channels == 0 || kernel == 0 || groups == 0 {
        return Ok(Vec::new());
    }
    if input.len() < seq_len * channels || channels % groups != 0 {
        return Err(JsValue::from_str("q4_grouped_conv1d_f32 input shape mismatch"));
    }
    let in_per_group = channels / groups;
    let row_size = in_per_group * kernel;
    let value_count = channels * row_size;
    if packed_weight.len() < value_count.div_ceil(2) || row_scales_f16.len() < channels {
        return Err(JsValue::from_str("q4_grouped_conv1d_f32 weight shape mismatch"));
    }
    let mut output = vec![0.0f32; seq_len * channels];
    for pos in 0..seq_len {
        for out_ch in 0..channels {
            let group = out_ch / in_per_group;
            let in_start = group * in_per_group;
            let scale = f16_to_f32(row_scales_f16[out_ch]);
            let mut sum = bias_values.get(out_ch).copied().unwrap_or(0.0);
            for k in 0..kernel {
                let src_pos = pos as isize + k as isize - padding as isize;
                if src_pos < 0 || src_pos >= seq_len as isize {
                    continue;
                }
                for local_in in 0..in_per_group {
                    let col = local_in * kernel + k;
                    let weight_index = out_ch * row_size + col;
                    let packed = packed_weight[weight_index >> 1];
                    let nibble = if weight_index & 1 == 0 { packed & 0x0f } else { packed >> 4 };
                    let q = nibble as i8;
                    let q = if q >= 8 { q - 16 } else { q } as f32;
                    sum += input[src_pos as usize * channels + in_start + local_in] * q * scale;
                }
            }
            output[pos * channels + out_ch] = sum;
        }
    }
    Ok(output)
}

#[wasm_bindgen]
pub fn vocos_istft_head_f32(stft_rows: &[f32], frames: usize) -> Result<Vec<f32>, JsValue> {
    const N_FFT: usize = 1024;
    const HOP: usize = 256;
    const BINS: usize = N_FFT / 2 + 1;
    const ROW: usize = N_FFT + 2;
    const LOG_100: f32 = 4.6051702;

    if frames == 0 {
        return Ok(Vec::new());
    }
    if stft_rows.len() < frames * ROW {
        return Err(JsValue::from_str("vocos_istft_head_f32 input shape mismatch"));
    }

    let hann = vocos_hann_1024();
    let bit_reverse = vocos_bit_reverse_1024();

    let padded_len = (frames - 1) * HOP + N_FFT;
    let mut audio = vec![0.0f32; padded_len];
    let mut envelope = vec![0.0f32; padded_len];
    let mut real = vec![0.0f32; N_FFT];
    let mut imag = vec![0.0f32; N_FFT];

    for frame in 0..frames {
        let row = frame * ROW;
        for bin in 0..BINS {
            let mag = stft_rows[row + bin].min(LOG_100).exp();
            let phase = stft_rows[row + BINS + bin];
            let re = mag * phase.cos();
            let im = mag * phase.sin();
            real[bin] = re;
            imag[bin] = if bin == 0 || bin == BINS - 1 { 0.0 } else { im };
            if bin > 0 && bin < BINS - 1 {
                real[N_FFT - bin] = re;
                imag[N_FFT - bin] = -im;
            }
        }

        inverse_fft_1024_in_place(&mut real, &mut imag, &bit_reverse);
        let offset = frame * HOP;
        for n in 0..N_FFT {
            audio[offset + n] += real[n] * hann[n];
            envelope[offset + n] += hann[n] * hann[n];
        }
    }

    for idx in 0..audio.len() {
        if envelope[idx] > 1.0e-11 {
            audio[idx] /= envelope[idx];
        }
    }
    let start = N_FFT / 2;
    let end = padded_len.saturating_sub(N_FFT / 2).max(start);
    Ok(audio[start..end].to_vec())
}

fn vocos_hann_1024() -> &'static [f32] {
    VOCOS_HANN_1024.get_or_init(|| {
        let mut hann = vec![0.0f32; 1024];
        for n in 0..1024 {
            let value = ((std::f32::consts::PI * n as f32) / 1024.0).sin();
            hann[n] = value * value;
        }
        hann.into_boxed_slice()
    })
}

fn vocos_bit_reverse_1024() -> &'static [usize] {
    VOCOS_BIT_REVERSE_1024.get_or_init(|| {
        let mut bit_reverse = vec![0usize; 1024];
        for i in 0..1024 {
            bit_reverse[i] = i.reverse_bits() >> (usize::BITS as usize - 10);
        }
        bit_reverse.into_boxed_slice()
    })
}

fn inverse_fft_1024_in_place(real: &mut [f32], imag: &mut [f32], bit_reverse: &[usize]) {
    const N_FFT: usize = 1024;
    for i in 0..N_FFT {
        let j = bit_reverse[i];
        if j > i {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= N_FFT {
        let half = len / 2;
        let angle = 2.0f32 * std::f32::consts::PI / len as f32;
        let step_re = angle.cos();
        let step_im = angle.sin();
        let mut start = 0usize;
        while start < N_FFT {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for j in 0..half {
                let even = start + j;
                let odd = even + half;
                let odd_re = real[odd] * w_re - imag[odd] * w_im;
                let odd_im = real[odd] * w_im + imag[odd] * w_re;
                real[odd] = real[even] - odd_re;
                imag[odd] = imag[even] - odd_im;
                real[even] += odd_re;
                imag[even] += odd_im;
                let next_re = w_re * step_re - w_im * step_im;
                w_im = w_re * step_im + w_im * step_re;
                w_re = next_re;
            }
            start += len;
        }
        len <<= 1;
    }
    let scale = 1.0f32 / N_FFT as f32;
    for idx in 0..N_FFT {
        real[idx] *= scale;
        imag[idx] *= scale;
    }
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
                    if head_dim == 64 {
                        score = dot_scaled_64(&q[q_base..q_base + 64], &k[k_base..k_base + 64], scale);
                    } else {
                        for dim in 0..head_dim {
                            score += q[q_base + dim] * k[k_base + dim] * scale;
                        }
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
            if head_dim == 64 {
                for kj in 0..kv_len {
                    let v_base = kj * model_dim + head * head_dim;
                    add_weighted_64(
                        &mut output[out_base..out_base + 64],
                        &v[v_base..v_base + 64],
                        scores[kj] / denom,
                    );
                }
            } else {
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
    }
    Ok(output)
}

fn attention_impl_kv_head_major(
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
        return Err(JsValue::from_str("attention_f32 head-major input shape mismatch"));
    }
    if head_dim != 64 {
        return attention_impl(q, k, v, q_len, kv_len, n_heads, head_dim, causal, past_len);
    }
    let k_head = to_head_major_cache(k, kv_len, n_heads, head_dim);
    let v_head = to_head_major_cache(v, kv_len, n_heads, head_dim);
    let mut output = vec![0.0f32; q_len * model_dim];
    let mut scores = vec![0.0f32; kv_len];
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    if !causal {
        for head in 0..n_heads {
            let head_base = head * kv_len * head_dim;
            for qi in 0..q_len {
                let mut max_score = f32::NEG_INFINITY;
                let q_base = qi * model_dim + head * head_dim;
                for kj in 0..kv_len {
                    let k_base = head_base + kj * head_dim;
                    let score = dot_scaled_64(&q[q_base..q_base + 64], &k_head[k_base..k_base + 64], scale);
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
                for kj in 0..kv_len {
                    let v_base = head_base + kj * head_dim;
                    add_weighted_64(
                        &mut output[out_base..out_base + 64],
                        &v_head[v_base..v_base + 64],
                        scores[kj] / denom,
                    );
                }
            }
        }
        return Ok(output);
    }
    for head in 0..n_heads {
        let head_base = head * kv_len * head_dim;
        for qi in 0..q_len {
            let mut max_score = f32::NEG_INFINITY;
            let q_base = qi * model_dim + head * head_dim;
            for kj in 0..kv_len {
                let mut score = if causal && kj > past_len + qi { -1.0e30 } else { 0.0 };
                if score > -1.0e20 {
                    let k_base = head_base + kj * head_dim;
                    score = dot_scaled_64(&q[q_base..q_base + 64], &k_head[k_base..k_base + 64], scale);
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
            for kj in 0..kv_len {
                let v_base = head_base + kj * head_dim;
                add_weighted_64(
                    &mut output[out_base..out_base + 64],
                    &v_head[v_base..v_base + 64],
                    scores[kj] / denom,
                );
            }
        }
    }
    Ok(output)
}

fn attention_impl_kv_already_head_major(
    q: &[f32],
    k_head: &[f32],
    v_head: &[f32],
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
    if q.len() < q_len * model_dim || k_head.len() < kv_len * model_dim || v_head.len() < kv_len * model_dim {
        return Err(JsValue::from_str("attention_f32 pretransposed input shape mismatch"));
    }
    if head_dim != 64 {
        return attention_impl(q, k_head, v_head, q_len, kv_len, n_heads, head_dim, causal, past_len);
    }
    let mut output = vec![0.0f32; q_len * model_dim];
    let mut scores = vec![0.0f32; kv_len];
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    if !causal {
        for head in 0..n_heads {
            let head_base = head * kv_len * head_dim;
            for qi in 0..q_len {
                let mut max_score = f32::NEG_INFINITY;
                let q_base = qi * model_dim + head * head_dim;
                for kj in 0..kv_len {
                    let k_base = head_base + kj * head_dim;
                    let score = dot_scaled_64(&q[q_base..q_base + 64], &k_head[k_base..k_base + 64], scale);
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
                for kj in 0..kv_len {
                    let v_base = head_base + kj * head_dim;
                    add_weighted_64(
                        &mut output[out_base..out_base + 64],
                        &v_head[v_base..v_base + 64],
                        scores[kj] / denom,
                    );
                }
            }
        }
        return Ok(output);
    }
    for head in 0..n_heads {
        let head_base = head * kv_len * head_dim;
        for qi in 0..q_len {
            let mut max_score = f32::NEG_INFINITY;
            let q_base = qi * model_dim + head * head_dim;
            for kj in 0..kv_len {
                let mut score = if causal && kj > past_len + qi { -1.0e30 } else { 0.0 };
                if score > -1.0e20 {
                    let k_base = head_base + kj * head_dim;
                    score = dot_scaled_64(&q[q_base..q_base + 64], &k_head[k_base..k_base + 64], scale);
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
            for kj in 0..kv_len {
                let v_base = head_base + kj * head_dim;
                add_weighted_64(
                    &mut output[out_base..out_base + 64],
                    &v_head[v_base..v_base + 64],
                    scores[kj] / denom,
                );
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
    if value > 10.0 {
        return value;
    }
    if value < -10.0 {
        return 0.0;
    }
    let coeff = (2.0f32 / core::f32::consts::PI).sqrt();
    0.5 * value * (1.0 + (coeff * (value + 0.044715 * value * value * value)).tanh())
}

fn silu_scalar(value: f32) -> f32 {
    if value > 20.0 {
        return value;
    }
    if value < -20.0 {
        return 0.0;
    }
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

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            let exp32 = ((e + 127) as u32) << 23;
            sign | exp32 | (mant << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let exp32 = ((exp - 15 + 127) as u32) << 23;
        sign | exp32 | (frac << 13)
    };
    f32::from_bits(out)
}

#[wasm_bindgen]
pub fn q4_symmetric_linear_f32(
    input: &[f32],
    packed_weight: &[u8],
    row_scales_f16: &[u16],
    bias_values: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, JsValue> {
    if rows == 0 || in_dim == 0 || out_dim == 0 {
        return Ok(Vec::new());
    }
    if input.len() != rows * in_dim {
        return Err(JsValue::from_str("Q4 input length does not match rows * in_dim"));
    }
    let packed_cols = in_dim.div_ceil(2);
    if packed_weight.len() < out_dim * packed_cols {
        return Err(JsValue::from_str("Q4 packed_weight is shorter than layout requires"));
    }
    if row_scales_f16.len() < out_dim {
        return Err(JsValue::from_str("Q4 row_scales_f16 length is shorter than out_dim"));
    }

    let mut output = vec![0.0f32; rows * out_dim];
    let even_in_dim = in_dim & !1;
    for out_idx in 0..out_dim {
        let weight_row = &packed_weight[out_idx * packed_cols..(out_idx + 1) * packed_cols];
        let scale = f16_to_f32(row_scales_f16[out_idx]);
        let bias = bias_values.get(out_idx).copied().unwrap_or(0.0);
        for row in 0..rows {
            let input_row = &input[row * in_dim..(row + 1) * in_dim];
            let mut acc = 0.0f32;
            let mut col = 0usize;
            let mut packed_col = 0usize;
            while col < even_in_dim {
                let packed = weight_row[packed_col];
                let lo = (packed & 0x0f) as i8;
                let hi = (packed >> 4) as i8;
                let lo = if lo >= 8 { lo - 16 } else { lo } as f32;
                let hi = if hi >= 8 { hi - 16 } else { hi } as f32;
                acc += input_row[col] * lo + input_row[col + 1] * hi;
                col += 2;
                packed_col += 1;
            }
            if col < in_dim {
                let lo = (weight_row[packed_col] & 0x0f) as i8;
                let lo = if lo >= 8 { lo - 16 } else { lo } as f32;
                acc += input_row[col] * lo;
            }
            output[row * out_dim + out_idx] = acc * scale + bias;
        }
    }
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot8_unpacked_i8_f32_simd(input: &[f32], row: usize, in_dim: usize, weight_row: &[i8]) -> [f32; 8] {
    use core::arch::wasm32::*;

    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut acc2 = f32x4_splat(0.0);
    let mut acc3 = f32x4_splat(0.0);
    let mut acc4 = f32x4_splat(0.0);
    let mut acc5 = f32x4_splat(0.0);
    let mut acc6 = f32x4_splat(0.0);
    let mut acc7 = f32x4_splat(0.0);
    let mut col = 0usize;
    while col + 3 < in_dim {
        let w = f32x4(
            *weight_row.get_unchecked(col) as f32,
            *weight_row.get_unchecked(col + 1) as f32,
            *weight_row.get_unchecked(col + 2) as f32,
            *weight_row.get_unchecked(col + 3) as f32,
        );
        acc0 = f32x4_add(acc0, f32x4_mul(v128_load(input.as_ptr().add(row * in_dim + col) as *const v128), w));
        acc1 = f32x4_add(acc1, f32x4_mul(v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128), w));
        acc2 = f32x4_add(acc2, f32x4_mul(v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128), w));
        acc3 = f32x4_add(acc3, f32x4_mul(v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128), w));
        acc4 = f32x4_add(acc4, f32x4_mul(v128_load(input.as_ptr().add((row + 4) * in_dim + col) as *const v128), w));
        acc5 = f32x4_add(acc5, f32x4_mul(v128_load(input.as_ptr().add((row + 5) * in_dim + col) as *const v128), w));
        acc6 = f32x4_add(acc6, f32x4_mul(v128_load(input.as_ptr().add((row + 6) * in_dim + col) as *const v128), w));
        acc7 = f32x4_add(acc7, f32x4_mul(v128_load(input.as_ptr().add((row + 7) * in_dim + col) as *const v128), w));
        col += 4;
    }
    let mut out = [
        f32x4_extract_lane::<0>(acc0) + f32x4_extract_lane::<1>(acc0) + f32x4_extract_lane::<2>(acc0) + f32x4_extract_lane::<3>(acc0),
        f32x4_extract_lane::<0>(acc1) + f32x4_extract_lane::<1>(acc1) + f32x4_extract_lane::<2>(acc1) + f32x4_extract_lane::<3>(acc1),
        f32x4_extract_lane::<0>(acc2) + f32x4_extract_lane::<1>(acc2) + f32x4_extract_lane::<2>(acc2) + f32x4_extract_lane::<3>(acc2),
        f32x4_extract_lane::<0>(acc3) + f32x4_extract_lane::<1>(acc3) + f32x4_extract_lane::<2>(acc3) + f32x4_extract_lane::<3>(acc3),
        f32x4_extract_lane::<0>(acc4) + f32x4_extract_lane::<1>(acc4) + f32x4_extract_lane::<2>(acc4) + f32x4_extract_lane::<3>(acc4),
        f32x4_extract_lane::<0>(acc5) + f32x4_extract_lane::<1>(acc5) + f32x4_extract_lane::<2>(acc5) + f32x4_extract_lane::<3>(acc5),
        f32x4_extract_lane::<0>(acc6) + f32x4_extract_lane::<1>(acc6) + f32x4_extract_lane::<2>(acc6) + f32x4_extract_lane::<3>(acc6),
        f32x4_extract_lane::<0>(acc7) + f32x4_extract_lane::<1>(acc7) + f32x4_extract_lane::<2>(acc7) + f32x4_extract_lane::<3>(acc7),
    ];
    while col < in_dim {
        let w = *weight_row.get_unchecked(col) as f32;
        for local_row in 0..8 {
            out[local_row] += *input.get_unchecked((row + local_row) * in_dim + col) * w;
        }
        col += 1;
    }
    out
}

fn dot8_unpacked_i8_f32(input: &[f32], row: usize, in_dim: usize, weight_row: &[i8]) -> [f32; 8] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot8_unpacked_i8_f32_simd(input, row, in_dim, weight_row) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut out = [0.0f32; 8];
        let mut col = 0usize;
        while col + 3 < in_dim {
            let w0 = weight_row[col] as f32;
            let w1 = weight_row[col + 1] as f32;
            let w2 = weight_row[col + 2] as f32;
            let w3 = weight_row[col + 3] as f32;
            for local_row in 0..8 {
                let base = (row + local_row) * in_dim + col;
                out[local_row] += input[base] * w0 + input[base + 1] * w1 + input[base + 2] * w2 + input[base + 3] * w3;
            }
            col += 4;
        }
        while col < in_dim {
            let w = weight_row[col] as f32;
            for local_row in 0..8 {
                out[local_row] += input[(row + local_row) * in_dim + col] * w;
            }
            col += 1;
        }
        out
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot8_unpacked_i8_f32_pair_simd(
    input: &[f32],
    row: usize,
    in_dim: usize,
    weight_a: &[i8],
    weight_b: &[i8],
) -> ([f32; 8], [f32; 8]) {
    use core::arch::wasm32::*;

    let mut a0 = f32x4_splat(0.0);
    let mut a1 = f32x4_splat(0.0);
    let mut a2 = f32x4_splat(0.0);
    let mut a3 = f32x4_splat(0.0);
    let mut a4 = f32x4_splat(0.0);
    let mut a5 = f32x4_splat(0.0);
    let mut a6 = f32x4_splat(0.0);
    let mut a7 = f32x4_splat(0.0);
    let mut b0 = f32x4_splat(0.0);
    let mut b1 = f32x4_splat(0.0);
    let mut b2 = f32x4_splat(0.0);
    let mut b3 = f32x4_splat(0.0);
    let mut b4 = f32x4_splat(0.0);
    let mut b5 = f32x4_splat(0.0);
    let mut b6 = f32x4_splat(0.0);
    let mut b7 = f32x4_splat(0.0);
    let mut col = 0usize;
    while col + 3 < in_dim {
        let wa = f32x4(
            *weight_a.get_unchecked(col) as f32,
            *weight_a.get_unchecked(col + 1) as f32,
            *weight_a.get_unchecked(col + 2) as f32,
            *weight_a.get_unchecked(col + 3) as f32,
        );
        let wb = f32x4(
            *weight_b.get_unchecked(col) as f32,
            *weight_b.get_unchecked(col + 1) as f32,
            *weight_b.get_unchecked(col + 2) as f32,
            *weight_b.get_unchecked(col + 3) as f32,
        );
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        let x4 = v128_load(input.as_ptr().add((row + 4) * in_dim + col) as *const v128);
        let x5 = v128_load(input.as_ptr().add((row + 5) * in_dim + col) as *const v128);
        let x6 = v128_load(input.as_ptr().add((row + 6) * in_dim + col) as *const v128);
        let x7 = v128_load(input.as_ptr().add((row + 7) * in_dim + col) as *const v128);
        a0 = f32x4_add(a0, f32x4_mul(x0, wa));
        a1 = f32x4_add(a1, f32x4_mul(x1, wa));
        a2 = f32x4_add(a2, f32x4_mul(x2, wa));
        a3 = f32x4_add(a3, f32x4_mul(x3, wa));
        a4 = f32x4_add(a4, f32x4_mul(x4, wa));
        a5 = f32x4_add(a5, f32x4_mul(x5, wa));
        a6 = f32x4_add(a6, f32x4_mul(x6, wa));
        a7 = f32x4_add(a7, f32x4_mul(x7, wa));
        b0 = f32x4_add(b0, f32x4_mul(x0, wb));
        b1 = f32x4_add(b1, f32x4_mul(x1, wb));
        b2 = f32x4_add(b2, f32x4_mul(x2, wb));
        b3 = f32x4_add(b3, f32x4_mul(x3, wb));
        b4 = f32x4_add(b4, f32x4_mul(x4, wb));
        b5 = f32x4_add(b5, f32x4_mul(x5, wb));
        b6 = f32x4_add(b6, f32x4_mul(x6, wb));
        b7 = f32x4_add(b7, f32x4_mul(x7, wb));
        col += 4;
    }
    let mut out_a = [
        sum_f32x4(a0), sum_f32x4(a1), sum_f32x4(a2), sum_f32x4(a3),
        sum_f32x4(a4), sum_f32x4(a5), sum_f32x4(a6), sum_f32x4(a7),
    ];
    let mut out_b = [
        sum_f32x4(b0), sum_f32x4(b1), sum_f32x4(b2), sum_f32x4(b3),
        sum_f32x4(b4), sum_f32x4(b5), sum_f32x4(b6), sum_f32x4(b7),
    ];
    while col < in_dim {
        let wa = *weight_a.get_unchecked(col) as f32;
        let wb = *weight_b.get_unchecked(col) as f32;
        for local_row in 0..8 {
            let x = *input.get_unchecked((row + local_row) * in_dim + col);
            out_a[local_row] += x * wa;
            out_b[local_row] += x * wb;
        }
        col += 1;
    }
    (out_a, out_b)
}

fn dot8_unpacked_i8_f32_pair(
    input: &[f32],
    row: usize,
    in_dim: usize,
    weight_a: &[i8],
    weight_b: &[i8],
) -> ([f32; 8], [f32; 8]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot8_unpacked_i8_f32_pair_simd(input, row, in_dim, weight_a, weight_b) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut out_a = [0.0f32; 8];
        let mut out_b = [0.0f32; 8];
        let mut col = 0usize;
        while col + 3 < in_dim {
            let wa0 = weight_a[col] as f32;
            let wa1 = weight_a[col + 1] as f32;
            let wa2 = weight_a[col + 2] as f32;
            let wa3 = weight_a[col + 3] as f32;
            let wb0 = weight_b[col] as f32;
            let wb1 = weight_b[col + 1] as f32;
            let wb2 = weight_b[col + 2] as f32;
            let wb3 = weight_b[col + 3] as f32;
            for local_row in 0..8 {
                let base = (row + local_row) * in_dim + col;
                let x0 = input[base];
                let x1 = input[base + 1];
                let x2 = input[base + 2];
                let x3 = input[base + 3];
                out_a[local_row] += x0 * wa0 + x1 * wa1 + x2 * wa2 + x3 * wa3;
                out_b[local_row] += x0 * wb0 + x1 * wb1 + x2 * wb2 + x3 * wb3;
            }
            col += 4;
        }
        while col < in_dim {
            let wa = weight_a[col] as f32;
            let wb = weight_b[col] as f32;
            for local_row in 0..8 {
                let x = input[(row + local_row) * in_dim + col];
                out_a[local_row] += x * wa;
                out_b[local_row] += x * wb;
            }
            col += 1;
        }
        (out_a, out_b)
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot4_unpacked_i8_f32_quad_simd(
    input: &[f32],
    row: usize,
    in_dim: usize,
    weight_a: &[i8],
    weight_b: &[i8],
    weight_c: &[i8],
    weight_d: &[i8],
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    use core::arch::wasm32::*;

    let mut a0 = f32x4_splat(0.0);
    let mut a1 = f32x4_splat(0.0);
    let mut a2 = f32x4_splat(0.0);
    let mut a3 = f32x4_splat(0.0);
    let mut b0 = f32x4_splat(0.0);
    let mut b1 = f32x4_splat(0.0);
    let mut b2 = f32x4_splat(0.0);
    let mut b3 = f32x4_splat(0.0);
    let mut c0 = f32x4_splat(0.0);
    let mut c1 = f32x4_splat(0.0);
    let mut c2 = f32x4_splat(0.0);
    let mut c3 = f32x4_splat(0.0);
    let mut d0 = f32x4_splat(0.0);
    let mut d1 = f32x4_splat(0.0);
    let mut d2 = f32x4_splat(0.0);
    let mut d3 = f32x4_splat(0.0);
    let mut col = 0usize;
    while col + 7 < in_dim {
        let wa = f32x4(
            *weight_a.get_unchecked(col) as f32,
            *weight_a.get_unchecked(col + 1) as f32,
            *weight_a.get_unchecked(col + 2) as f32,
            *weight_a.get_unchecked(col + 3) as f32,
        );
        let wb = f32x4(
            *weight_b.get_unchecked(col) as f32,
            *weight_b.get_unchecked(col + 1) as f32,
            *weight_b.get_unchecked(col + 2) as f32,
            *weight_b.get_unchecked(col + 3) as f32,
        );
        let wc = f32x4(
            *weight_c.get_unchecked(col) as f32,
            *weight_c.get_unchecked(col + 1) as f32,
            *weight_c.get_unchecked(col + 2) as f32,
            *weight_c.get_unchecked(col + 3) as f32,
        );
        let wd = f32x4(
            *weight_d.get_unchecked(col) as f32,
            *weight_d.get_unchecked(col + 1) as f32,
            *weight_d.get_unchecked(col + 2) as f32,
            *weight_d.get_unchecked(col + 3) as f32,
        );
        let wa_next = f32x4(
            *weight_a.get_unchecked(col + 4) as f32,
            *weight_a.get_unchecked(col + 5) as f32,
            *weight_a.get_unchecked(col + 6) as f32,
            *weight_a.get_unchecked(col + 7) as f32,
        );
        let wb_next = f32x4(
            *weight_b.get_unchecked(col + 4) as f32,
            *weight_b.get_unchecked(col + 5) as f32,
            *weight_b.get_unchecked(col + 6) as f32,
            *weight_b.get_unchecked(col + 7) as f32,
        );
        let wc_next = f32x4(
            *weight_c.get_unchecked(col + 4) as f32,
            *weight_c.get_unchecked(col + 5) as f32,
            *weight_c.get_unchecked(col + 6) as f32,
            *weight_c.get_unchecked(col + 7) as f32,
        );
        let wd_next = f32x4(
            *weight_d.get_unchecked(col + 4) as f32,
            *weight_d.get_unchecked(col + 5) as f32,
            *weight_d.get_unchecked(col + 6) as f32,
            *weight_d.get_unchecked(col + 7) as f32,
        );
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        let x0_next = v128_load(input.as_ptr().add(row * in_dim + col + 4) as *const v128);
        let x1_next = v128_load(input.as_ptr().add((row + 1) * in_dim + col + 4) as *const v128);
        let x2_next = v128_load(input.as_ptr().add((row + 2) * in_dim + col + 4) as *const v128);
        let x3_next = v128_load(input.as_ptr().add((row + 3) * in_dim + col + 4) as *const v128);
        a0 = f32x4_add(f32x4_add(a0, f32x4_mul(x0, wa)), f32x4_mul(x0_next, wa_next));
        a1 = f32x4_add(f32x4_add(a1, f32x4_mul(x1, wa)), f32x4_mul(x1_next, wa_next));
        a2 = f32x4_add(f32x4_add(a2, f32x4_mul(x2, wa)), f32x4_mul(x2_next, wa_next));
        a3 = f32x4_add(f32x4_add(a3, f32x4_mul(x3, wa)), f32x4_mul(x3_next, wa_next));
        b0 = f32x4_add(f32x4_add(b0, f32x4_mul(x0, wb)), f32x4_mul(x0_next, wb_next));
        b1 = f32x4_add(f32x4_add(b1, f32x4_mul(x1, wb)), f32x4_mul(x1_next, wb_next));
        b2 = f32x4_add(f32x4_add(b2, f32x4_mul(x2, wb)), f32x4_mul(x2_next, wb_next));
        b3 = f32x4_add(f32x4_add(b3, f32x4_mul(x3, wb)), f32x4_mul(x3_next, wb_next));
        c0 = f32x4_add(f32x4_add(c0, f32x4_mul(x0, wc)), f32x4_mul(x0_next, wc_next));
        c1 = f32x4_add(f32x4_add(c1, f32x4_mul(x1, wc)), f32x4_mul(x1_next, wc_next));
        c2 = f32x4_add(f32x4_add(c2, f32x4_mul(x2, wc)), f32x4_mul(x2_next, wc_next));
        c3 = f32x4_add(f32x4_add(c3, f32x4_mul(x3, wc)), f32x4_mul(x3_next, wc_next));
        d0 = f32x4_add(f32x4_add(d0, f32x4_mul(x0, wd)), f32x4_mul(x0_next, wd_next));
        d1 = f32x4_add(f32x4_add(d1, f32x4_mul(x1, wd)), f32x4_mul(x1_next, wd_next));
        d2 = f32x4_add(f32x4_add(d2, f32x4_mul(x2, wd)), f32x4_mul(x2_next, wd_next));
        d3 = f32x4_add(f32x4_add(d3, f32x4_mul(x3, wd)), f32x4_mul(x3_next, wd_next));
        col += 8;
    }
    while col + 3 < in_dim {
        let wa = f32x4(
            *weight_a.get_unchecked(col) as f32,
            *weight_a.get_unchecked(col + 1) as f32,
            *weight_a.get_unchecked(col + 2) as f32,
            *weight_a.get_unchecked(col + 3) as f32,
        );
        let wb = f32x4(
            *weight_b.get_unchecked(col) as f32,
            *weight_b.get_unchecked(col + 1) as f32,
            *weight_b.get_unchecked(col + 2) as f32,
            *weight_b.get_unchecked(col + 3) as f32,
        );
        let wc = f32x4(
            *weight_c.get_unchecked(col) as f32,
            *weight_c.get_unchecked(col + 1) as f32,
            *weight_c.get_unchecked(col + 2) as f32,
            *weight_c.get_unchecked(col + 3) as f32,
        );
        let wd = f32x4(
            *weight_d.get_unchecked(col) as f32,
            *weight_d.get_unchecked(col + 1) as f32,
            *weight_d.get_unchecked(col + 2) as f32,
            *weight_d.get_unchecked(col + 3) as f32,
        );
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        a0 = f32x4_add(a0, f32x4_mul(x0, wa));
        a1 = f32x4_add(a1, f32x4_mul(x1, wa));
        a2 = f32x4_add(a2, f32x4_mul(x2, wa));
        a3 = f32x4_add(a3, f32x4_mul(x3, wa));
        b0 = f32x4_add(b0, f32x4_mul(x0, wb));
        b1 = f32x4_add(b1, f32x4_mul(x1, wb));
        b2 = f32x4_add(b2, f32x4_mul(x2, wb));
        b3 = f32x4_add(b3, f32x4_mul(x3, wb));
        c0 = f32x4_add(c0, f32x4_mul(x0, wc));
        c1 = f32x4_add(c1, f32x4_mul(x1, wc));
        c2 = f32x4_add(c2, f32x4_mul(x2, wc));
        c3 = f32x4_add(c3, f32x4_mul(x3, wc));
        d0 = f32x4_add(d0, f32x4_mul(x0, wd));
        d1 = f32x4_add(d1, f32x4_mul(x1, wd));
        d2 = f32x4_add(d2, f32x4_mul(x2, wd));
        d3 = f32x4_add(d3, f32x4_mul(x3, wd));
        col += 4;
    }
    let mut out_a = [sum_f32x4(a0), sum_f32x4(a1), sum_f32x4(a2), sum_f32x4(a3)];
    let mut out_b = [sum_f32x4(b0), sum_f32x4(b1), sum_f32x4(b2), sum_f32x4(b3)];
    let mut out_c = [sum_f32x4(c0), sum_f32x4(c1), sum_f32x4(c2), sum_f32x4(c3)];
    let mut out_d = [sum_f32x4(d0), sum_f32x4(d1), sum_f32x4(d2), sum_f32x4(d3)];
    while col < in_dim {
        let wa = *weight_a.get_unchecked(col) as f32;
        let wb = *weight_b.get_unchecked(col) as f32;
        let wc = *weight_c.get_unchecked(col) as f32;
        let wd = *weight_d.get_unchecked(col) as f32;
        for local_row in 0..4 {
            let x = *input.get_unchecked((row + local_row) * in_dim + col);
            out_a[local_row] += x * wa;
            out_b[local_row] += x * wb;
            out_c[local_row] += x * wc;
            out_d[local_row] += x * wd;
        }
        col += 1;
    }
    (out_a, out_b, out_c, out_d)
}

fn dot4_unpacked_i8_f32_quad(
    input: &[f32],
    row: usize,
    in_dim: usize,
    weight_a: &[i8],
    weight_b: &[i8],
    weight_c: &[i8],
    weight_d: &[i8],
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot4_unpacked_i8_f32_quad_simd(input, row, in_dim, weight_a, weight_b, weight_c, weight_d) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut out_a = [0.0f32; 4];
        let mut out_b = [0.0f32; 4];
        let mut out_c = [0.0f32; 4];
        let mut out_d = [0.0f32; 4];
        for col in 0..in_dim {
            let wa = weight_a[col] as f32;
            let wb = weight_b[col] as f32;
            let wc = weight_c[col] as f32;
            let wd = weight_d[col] as f32;
            for local_row in 0..4 {
                let x = input[(row + local_row) * in_dim + col];
                out_a[local_row] += x * wa;
                out_b[local_row] += x * wb;
                out_c[local_row] += x * wc;
                out_d[local_row] += x * wd;
            }
        }
        (out_a, out_b, out_c, out_d)
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot4_unpacked_i8_f32_six_simd(
    input: &[f32],
    row: usize,
    in_dim: usize,
    weight_a0: &[i8],
    weight_b0: &[i8],
    weight_c0: &[i8],
    weight_a1: &[i8],
    weight_b1: &[i8],
    weight_c1: &[i8],
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    use core::arch::wasm32::*;

    let mut a0 = f32x4_splat(0.0);
    let mut a1 = f32x4_splat(0.0);
    let mut a2 = f32x4_splat(0.0);
    let mut a3 = f32x4_splat(0.0);
    let mut b0 = f32x4_splat(0.0);
    let mut b1 = f32x4_splat(0.0);
    let mut b2 = f32x4_splat(0.0);
    let mut b3 = f32x4_splat(0.0);
    let mut c0 = f32x4_splat(0.0);
    let mut c1 = f32x4_splat(0.0);
    let mut c2 = f32x4_splat(0.0);
    let mut c3 = f32x4_splat(0.0);
    let mut d0 = f32x4_splat(0.0);
    let mut d1 = f32x4_splat(0.0);
    let mut d2 = f32x4_splat(0.0);
    let mut d3 = f32x4_splat(0.0);
    let mut e0 = f32x4_splat(0.0);
    let mut e1 = f32x4_splat(0.0);
    let mut e2 = f32x4_splat(0.0);
    let mut e3 = f32x4_splat(0.0);
    let mut f0 = f32x4_splat(0.0);
    let mut f1 = f32x4_splat(0.0);
    let mut f2 = f32x4_splat(0.0);
    let mut f3 = f32x4_splat(0.0);
    let mut col = 0usize;
    while col + 7 < in_dim {
        let wa0 = f32x4(
            *weight_a0.get_unchecked(col) as f32,
            *weight_a0.get_unchecked(col + 1) as f32,
            *weight_a0.get_unchecked(col + 2) as f32,
            *weight_a0.get_unchecked(col + 3) as f32,
        );
        let wb0 = f32x4(
            *weight_b0.get_unchecked(col) as f32,
            *weight_b0.get_unchecked(col + 1) as f32,
            *weight_b0.get_unchecked(col + 2) as f32,
            *weight_b0.get_unchecked(col + 3) as f32,
        );
        let wc0 = f32x4(
            *weight_c0.get_unchecked(col) as f32,
            *weight_c0.get_unchecked(col + 1) as f32,
            *weight_c0.get_unchecked(col + 2) as f32,
            *weight_c0.get_unchecked(col + 3) as f32,
        );
        let wa1 = f32x4(
            *weight_a1.get_unchecked(col) as f32,
            *weight_a1.get_unchecked(col + 1) as f32,
            *weight_a1.get_unchecked(col + 2) as f32,
            *weight_a1.get_unchecked(col + 3) as f32,
        );
        let wb1 = f32x4(
            *weight_b1.get_unchecked(col) as f32,
            *weight_b1.get_unchecked(col + 1) as f32,
            *weight_b1.get_unchecked(col + 2) as f32,
            *weight_b1.get_unchecked(col + 3) as f32,
        );
        let wc1 = f32x4(
            *weight_c1.get_unchecked(col) as f32,
            *weight_c1.get_unchecked(col + 1) as f32,
            *weight_c1.get_unchecked(col + 2) as f32,
            *weight_c1.get_unchecked(col + 3) as f32,
        );
        let wa0_next = f32x4(
            *weight_a0.get_unchecked(col + 4) as f32,
            *weight_a0.get_unchecked(col + 5) as f32,
            *weight_a0.get_unchecked(col + 6) as f32,
            *weight_a0.get_unchecked(col + 7) as f32,
        );
        let wb0_next = f32x4(
            *weight_b0.get_unchecked(col + 4) as f32,
            *weight_b0.get_unchecked(col + 5) as f32,
            *weight_b0.get_unchecked(col + 6) as f32,
            *weight_b0.get_unchecked(col + 7) as f32,
        );
        let wc0_next = f32x4(
            *weight_c0.get_unchecked(col + 4) as f32,
            *weight_c0.get_unchecked(col + 5) as f32,
            *weight_c0.get_unchecked(col + 6) as f32,
            *weight_c0.get_unchecked(col + 7) as f32,
        );
        let wa1_next = f32x4(
            *weight_a1.get_unchecked(col + 4) as f32,
            *weight_a1.get_unchecked(col + 5) as f32,
            *weight_a1.get_unchecked(col + 6) as f32,
            *weight_a1.get_unchecked(col + 7) as f32,
        );
        let wb1_next = f32x4(
            *weight_b1.get_unchecked(col + 4) as f32,
            *weight_b1.get_unchecked(col + 5) as f32,
            *weight_b1.get_unchecked(col + 6) as f32,
            *weight_b1.get_unchecked(col + 7) as f32,
        );
        let wc1_next = f32x4(
            *weight_c1.get_unchecked(col + 4) as f32,
            *weight_c1.get_unchecked(col + 5) as f32,
            *weight_c1.get_unchecked(col + 6) as f32,
            *weight_c1.get_unchecked(col + 7) as f32,
        );
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        let x0_next = v128_load(input.as_ptr().add(row * in_dim + col + 4) as *const v128);
        let x1_next = v128_load(input.as_ptr().add((row + 1) * in_dim + col + 4) as *const v128);
        let x2_next = v128_load(input.as_ptr().add((row + 2) * in_dim + col + 4) as *const v128);
        let x3_next = v128_load(input.as_ptr().add((row + 3) * in_dim + col + 4) as *const v128);
        a0 = f32x4_add(f32x4_add(a0, f32x4_mul(x0, wa0)), f32x4_mul(x0_next, wa0_next));
        a1 = f32x4_add(f32x4_add(a1, f32x4_mul(x1, wa0)), f32x4_mul(x1_next, wa0_next));
        a2 = f32x4_add(f32x4_add(a2, f32x4_mul(x2, wa0)), f32x4_mul(x2_next, wa0_next));
        a3 = f32x4_add(f32x4_add(a3, f32x4_mul(x3, wa0)), f32x4_mul(x3_next, wa0_next));
        b0 = f32x4_add(f32x4_add(b0, f32x4_mul(x0, wb0)), f32x4_mul(x0_next, wb0_next));
        b1 = f32x4_add(f32x4_add(b1, f32x4_mul(x1, wb0)), f32x4_mul(x1_next, wb0_next));
        b2 = f32x4_add(f32x4_add(b2, f32x4_mul(x2, wb0)), f32x4_mul(x2_next, wb0_next));
        b3 = f32x4_add(f32x4_add(b3, f32x4_mul(x3, wb0)), f32x4_mul(x3_next, wb0_next));
        c0 = f32x4_add(f32x4_add(c0, f32x4_mul(x0, wc0)), f32x4_mul(x0_next, wc0_next));
        c1 = f32x4_add(f32x4_add(c1, f32x4_mul(x1, wc0)), f32x4_mul(x1_next, wc0_next));
        c2 = f32x4_add(f32x4_add(c2, f32x4_mul(x2, wc0)), f32x4_mul(x2_next, wc0_next));
        c3 = f32x4_add(f32x4_add(c3, f32x4_mul(x3, wc0)), f32x4_mul(x3_next, wc0_next));
        d0 = f32x4_add(f32x4_add(d0, f32x4_mul(x0, wa1)), f32x4_mul(x0_next, wa1_next));
        d1 = f32x4_add(f32x4_add(d1, f32x4_mul(x1, wa1)), f32x4_mul(x1_next, wa1_next));
        d2 = f32x4_add(f32x4_add(d2, f32x4_mul(x2, wa1)), f32x4_mul(x2_next, wa1_next));
        d3 = f32x4_add(f32x4_add(d3, f32x4_mul(x3, wa1)), f32x4_mul(x3_next, wa1_next));
        e0 = f32x4_add(f32x4_add(e0, f32x4_mul(x0, wb1)), f32x4_mul(x0_next, wb1_next));
        e1 = f32x4_add(f32x4_add(e1, f32x4_mul(x1, wb1)), f32x4_mul(x1_next, wb1_next));
        e2 = f32x4_add(f32x4_add(e2, f32x4_mul(x2, wb1)), f32x4_mul(x2_next, wb1_next));
        e3 = f32x4_add(f32x4_add(e3, f32x4_mul(x3, wb1)), f32x4_mul(x3_next, wb1_next));
        f0 = f32x4_add(f32x4_add(f0, f32x4_mul(x0, wc1)), f32x4_mul(x0_next, wc1_next));
        f1 = f32x4_add(f32x4_add(f1, f32x4_mul(x1, wc1)), f32x4_mul(x1_next, wc1_next));
        f2 = f32x4_add(f32x4_add(f2, f32x4_mul(x2, wc1)), f32x4_mul(x2_next, wc1_next));
        f3 = f32x4_add(f32x4_add(f3, f32x4_mul(x3, wc1)), f32x4_mul(x3_next, wc1_next));
        col += 8;
    }
    while col + 3 < in_dim {
        let wa0 = f32x4(
            *weight_a0.get_unchecked(col) as f32,
            *weight_a0.get_unchecked(col + 1) as f32,
            *weight_a0.get_unchecked(col + 2) as f32,
            *weight_a0.get_unchecked(col + 3) as f32,
        );
        let wb0 = f32x4(
            *weight_b0.get_unchecked(col) as f32,
            *weight_b0.get_unchecked(col + 1) as f32,
            *weight_b0.get_unchecked(col + 2) as f32,
            *weight_b0.get_unchecked(col + 3) as f32,
        );
        let wc0 = f32x4(
            *weight_c0.get_unchecked(col) as f32,
            *weight_c0.get_unchecked(col + 1) as f32,
            *weight_c0.get_unchecked(col + 2) as f32,
            *weight_c0.get_unchecked(col + 3) as f32,
        );
        let wa1 = f32x4(
            *weight_a1.get_unchecked(col) as f32,
            *weight_a1.get_unchecked(col + 1) as f32,
            *weight_a1.get_unchecked(col + 2) as f32,
            *weight_a1.get_unchecked(col + 3) as f32,
        );
        let wb1 = f32x4(
            *weight_b1.get_unchecked(col) as f32,
            *weight_b1.get_unchecked(col + 1) as f32,
            *weight_b1.get_unchecked(col + 2) as f32,
            *weight_b1.get_unchecked(col + 3) as f32,
        );
        let wc1 = f32x4(
            *weight_c1.get_unchecked(col) as f32,
            *weight_c1.get_unchecked(col + 1) as f32,
            *weight_c1.get_unchecked(col + 2) as f32,
            *weight_c1.get_unchecked(col + 3) as f32,
        );
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        a0 = f32x4_add(a0, f32x4_mul(x0, wa0));
        a1 = f32x4_add(a1, f32x4_mul(x1, wa0));
        a2 = f32x4_add(a2, f32x4_mul(x2, wa0));
        a3 = f32x4_add(a3, f32x4_mul(x3, wa0));
        b0 = f32x4_add(b0, f32x4_mul(x0, wb0));
        b1 = f32x4_add(b1, f32x4_mul(x1, wb0));
        b2 = f32x4_add(b2, f32x4_mul(x2, wb0));
        b3 = f32x4_add(b3, f32x4_mul(x3, wb0));
        c0 = f32x4_add(c0, f32x4_mul(x0, wc0));
        c1 = f32x4_add(c1, f32x4_mul(x1, wc0));
        c2 = f32x4_add(c2, f32x4_mul(x2, wc0));
        c3 = f32x4_add(c3, f32x4_mul(x3, wc0));
        d0 = f32x4_add(d0, f32x4_mul(x0, wa1));
        d1 = f32x4_add(d1, f32x4_mul(x1, wa1));
        d2 = f32x4_add(d2, f32x4_mul(x2, wa1));
        d3 = f32x4_add(d3, f32x4_mul(x3, wa1));
        e0 = f32x4_add(e0, f32x4_mul(x0, wb1));
        e1 = f32x4_add(e1, f32x4_mul(x1, wb1));
        e2 = f32x4_add(e2, f32x4_mul(x2, wb1));
        e3 = f32x4_add(e3, f32x4_mul(x3, wb1));
        f0 = f32x4_add(f0, f32x4_mul(x0, wc1));
        f1 = f32x4_add(f1, f32x4_mul(x1, wc1));
        f2 = f32x4_add(f2, f32x4_mul(x2, wc1));
        f3 = f32x4_add(f3, f32x4_mul(x3, wc1));
        col += 4;
    }
    let mut out_a0 = [sum_f32x4(a0), sum_f32x4(a1), sum_f32x4(a2), sum_f32x4(a3)];
    let mut out_b0 = [sum_f32x4(b0), sum_f32x4(b1), sum_f32x4(b2), sum_f32x4(b3)];
    let mut out_c0 = [sum_f32x4(c0), sum_f32x4(c1), sum_f32x4(c2), sum_f32x4(c3)];
    let mut out_a1 = [sum_f32x4(d0), sum_f32x4(d1), sum_f32x4(d2), sum_f32x4(d3)];
    let mut out_b1 = [sum_f32x4(e0), sum_f32x4(e1), sum_f32x4(e2), sum_f32x4(e3)];
    let mut out_c1 = [sum_f32x4(f0), sum_f32x4(f1), sum_f32x4(f2), sum_f32x4(f3)];
    while col < in_dim {
        let wa0 = *weight_a0.get_unchecked(col) as f32;
        let wb0 = *weight_b0.get_unchecked(col) as f32;
        let wc0 = *weight_c0.get_unchecked(col) as f32;
        let wa1 = *weight_a1.get_unchecked(col) as f32;
        let wb1 = *weight_b1.get_unchecked(col) as f32;
        let wc1 = *weight_c1.get_unchecked(col) as f32;
        for local_row in 0..4 {
            let x = *input.get_unchecked((row + local_row) * in_dim + col);
            out_a0[local_row] += x * wa0;
            out_b0[local_row] += x * wb0;
            out_c0[local_row] += x * wc0;
            out_a1[local_row] += x * wa1;
            out_b1[local_row] += x * wb1;
            out_c1[local_row] += x * wc1;
        }
        col += 1;
    }
    (out_a0, out_b0, out_c0, out_a1, out_b1, out_c1)
}

fn dot4_unpacked_i8_f32_six(
    input: &[f32],
    row: usize,
    in_dim: usize,
    weight_a0: &[i8],
    weight_b0: &[i8],
    weight_c0: &[i8],
    weight_a1: &[i8],
    weight_b1: &[i8],
    weight_c1: &[i8],
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot4_unpacked_i8_f32_six_simd(input, row, in_dim, weight_a0, weight_b0, weight_c0, weight_a1, weight_b1, weight_c1) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut out_a0 = [0.0f32; 4];
        let mut out_b0 = [0.0f32; 4];
        let mut out_c0 = [0.0f32; 4];
        let mut out_a1 = [0.0f32; 4];
        let mut out_b1 = [0.0f32; 4];
        let mut out_c1 = [0.0f32; 4];
        for col in 0..in_dim {
            let wa0 = weight_a0[col] as f32;
            let wb0 = weight_b0[col] as f32;
            let wc0 = weight_c0[col] as f32;
            let wa1 = weight_a1[col] as f32;
            let wb1 = weight_b1[col] as f32;
            let wc1 = weight_c1[col] as f32;
            for local_row in 0..4 {
                let x = input[(row + local_row) * in_dim + col];
                out_a0[local_row] += x * wa0;
                out_b0[local_row] += x * wb0;
                out_c0[local_row] += x * wc0;
                out_a1[local_row] += x * wa1;
                out_b1[local_row] += x * wb1;
                out_c1[local_row] += x * wc1;
            }
        }
        (out_a0, out_b0, out_c0, out_a1, out_b1, out_c1)
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot8_unpacked_i8_f32_triple_simd(
    input: &[f32],
    row: usize,
    in_dim: usize,
    weight_a: &[i8],
    weight_b: &[i8],
    weight_c: &[i8],
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    use core::arch::wasm32::*;

    let mut a0 = f32x4_splat(0.0);
    let mut a1 = f32x4_splat(0.0);
    let mut a2 = f32x4_splat(0.0);
    let mut a3 = f32x4_splat(0.0);
    let mut a4 = f32x4_splat(0.0);
    let mut a5 = f32x4_splat(0.0);
    let mut a6 = f32x4_splat(0.0);
    let mut a7 = f32x4_splat(0.0);
    let mut b0 = f32x4_splat(0.0);
    let mut b1 = f32x4_splat(0.0);
    let mut b2 = f32x4_splat(0.0);
    let mut b3 = f32x4_splat(0.0);
    let mut b4 = f32x4_splat(0.0);
    let mut b5 = f32x4_splat(0.0);
    let mut b6 = f32x4_splat(0.0);
    let mut b7 = f32x4_splat(0.0);
    let mut c0 = f32x4_splat(0.0);
    let mut c1 = f32x4_splat(0.0);
    let mut c2 = f32x4_splat(0.0);
    let mut c3 = f32x4_splat(0.0);
    let mut c4 = f32x4_splat(0.0);
    let mut c5 = f32x4_splat(0.0);
    let mut c6 = f32x4_splat(0.0);
    let mut c7 = f32x4_splat(0.0);
    let mut col = 0usize;
    while col + 7 < in_dim {
        let wa = f32x4(
            *weight_a.get_unchecked(col) as f32,
            *weight_a.get_unchecked(col + 1) as f32,
            *weight_a.get_unchecked(col + 2) as f32,
            *weight_a.get_unchecked(col + 3) as f32,
        );
        let wb = f32x4(
            *weight_b.get_unchecked(col) as f32,
            *weight_b.get_unchecked(col + 1) as f32,
            *weight_b.get_unchecked(col + 2) as f32,
            *weight_b.get_unchecked(col + 3) as f32,
        );
        let wc = f32x4(
            *weight_c.get_unchecked(col) as f32,
            *weight_c.get_unchecked(col + 1) as f32,
            *weight_c.get_unchecked(col + 2) as f32,
            *weight_c.get_unchecked(col + 3) as f32,
        );
        let wa_next = f32x4(
            *weight_a.get_unchecked(col + 4) as f32,
            *weight_a.get_unchecked(col + 5) as f32,
            *weight_a.get_unchecked(col + 6) as f32,
            *weight_a.get_unchecked(col + 7) as f32,
        );
        let wb_next = f32x4(
            *weight_b.get_unchecked(col + 4) as f32,
            *weight_b.get_unchecked(col + 5) as f32,
            *weight_b.get_unchecked(col + 6) as f32,
            *weight_b.get_unchecked(col + 7) as f32,
        );
        let wc_next = f32x4(
            *weight_c.get_unchecked(col + 4) as f32,
            *weight_c.get_unchecked(col + 5) as f32,
            *weight_c.get_unchecked(col + 6) as f32,
            *weight_c.get_unchecked(col + 7) as f32,
        );
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        let x4 = v128_load(input.as_ptr().add((row + 4) * in_dim + col) as *const v128);
        let x5 = v128_load(input.as_ptr().add((row + 5) * in_dim + col) as *const v128);
        let x6 = v128_load(input.as_ptr().add((row + 6) * in_dim + col) as *const v128);
        let x7 = v128_load(input.as_ptr().add((row + 7) * in_dim + col) as *const v128);
        let x0_next = v128_load(input.as_ptr().add(row * in_dim + col + 4) as *const v128);
        let x1_next = v128_load(input.as_ptr().add((row + 1) * in_dim + col + 4) as *const v128);
        let x2_next = v128_load(input.as_ptr().add((row + 2) * in_dim + col + 4) as *const v128);
        let x3_next = v128_load(input.as_ptr().add((row + 3) * in_dim + col + 4) as *const v128);
        let x4_next = v128_load(input.as_ptr().add((row + 4) * in_dim + col + 4) as *const v128);
        let x5_next = v128_load(input.as_ptr().add((row + 5) * in_dim + col + 4) as *const v128);
        let x6_next = v128_load(input.as_ptr().add((row + 6) * in_dim + col + 4) as *const v128);
        let x7_next = v128_load(input.as_ptr().add((row + 7) * in_dim + col + 4) as *const v128);
        a0 = f32x4_add(f32x4_add(a0, f32x4_mul(x0, wa)), f32x4_mul(x0_next, wa_next));
        a1 = f32x4_add(f32x4_add(a1, f32x4_mul(x1, wa)), f32x4_mul(x1_next, wa_next));
        a2 = f32x4_add(f32x4_add(a2, f32x4_mul(x2, wa)), f32x4_mul(x2_next, wa_next));
        a3 = f32x4_add(f32x4_add(a3, f32x4_mul(x3, wa)), f32x4_mul(x3_next, wa_next));
        a4 = f32x4_add(f32x4_add(a4, f32x4_mul(x4, wa)), f32x4_mul(x4_next, wa_next));
        a5 = f32x4_add(f32x4_add(a5, f32x4_mul(x5, wa)), f32x4_mul(x5_next, wa_next));
        a6 = f32x4_add(f32x4_add(a6, f32x4_mul(x6, wa)), f32x4_mul(x6_next, wa_next));
        a7 = f32x4_add(f32x4_add(a7, f32x4_mul(x7, wa)), f32x4_mul(x7_next, wa_next));
        b0 = f32x4_add(f32x4_add(b0, f32x4_mul(x0, wb)), f32x4_mul(x0_next, wb_next));
        b1 = f32x4_add(f32x4_add(b1, f32x4_mul(x1, wb)), f32x4_mul(x1_next, wb_next));
        b2 = f32x4_add(f32x4_add(b2, f32x4_mul(x2, wb)), f32x4_mul(x2_next, wb_next));
        b3 = f32x4_add(f32x4_add(b3, f32x4_mul(x3, wb)), f32x4_mul(x3_next, wb_next));
        b4 = f32x4_add(f32x4_add(b4, f32x4_mul(x4, wb)), f32x4_mul(x4_next, wb_next));
        b5 = f32x4_add(f32x4_add(b5, f32x4_mul(x5, wb)), f32x4_mul(x5_next, wb_next));
        b6 = f32x4_add(f32x4_add(b6, f32x4_mul(x6, wb)), f32x4_mul(x6_next, wb_next));
        b7 = f32x4_add(f32x4_add(b7, f32x4_mul(x7, wb)), f32x4_mul(x7_next, wb_next));
        c0 = f32x4_add(f32x4_add(c0, f32x4_mul(x0, wc)), f32x4_mul(x0_next, wc_next));
        c1 = f32x4_add(f32x4_add(c1, f32x4_mul(x1, wc)), f32x4_mul(x1_next, wc_next));
        c2 = f32x4_add(f32x4_add(c2, f32x4_mul(x2, wc)), f32x4_mul(x2_next, wc_next));
        c3 = f32x4_add(f32x4_add(c3, f32x4_mul(x3, wc)), f32x4_mul(x3_next, wc_next));
        c4 = f32x4_add(f32x4_add(c4, f32x4_mul(x4, wc)), f32x4_mul(x4_next, wc_next));
        c5 = f32x4_add(f32x4_add(c5, f32x4_mul(x5, wc)), f32x4_mul(x5_next, wc_next));
        c6 = f32x4_add(f32x4_add(c6, f32x4_mul(x6, wc)), f32x4_mul(x6_next, wc_next));
        c7 = f32x4_add(f32x4_add(c7, f32x4_mul(x7, wc)), f32x4_mul(x7_next, wc_next));
        col += 8;
    }
    while col + 3 < in_dim {
        let wa = f32x4(
            *weight_a.get_unchecked(col) as f32,
            *weight_a.get_unchecked(col + 1) as f32,
            *weight_a.get_unchecked(col + 2) as f32,
            *weight_a.get_unchecked(col + 3) as f32,
        );
        let wb = f32x4(
            *weight_b.get_unchecked(col) as f32,
            *weight_b.get_unchecked(col + 1) as f32,
            *weight_b.get_unchecked(col + 2) as f32,
            *weight_b.get_unchecked(col + 3) as f32,
        );
        let wc = f32x4(
            *weight_c.get_unchecked(col) as f32,
            *weight_c.get_unchecked(col + 1) as f32,
            *weight_c.get_unchecked(col + 2) as f32,
            *weight_c.get_unchecked(col + 3) as f32,
        );
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        let x4 = v128_load(input.as_ptr().add((row + 4) * in_dim + col) as *const v128);
        let x5 = v128_load(input.as_ptr().add((row + 5) * in_dim + col) as *const v128);
        let x6 = v128_load(input.as_ptr().add((row + 6) * in_dim + col) as *const v128);
        let x7 = v128_load(input.as_ptr().add((row + 7) * in_dim + col) as *const v128);
        a0 = f32x4_add(a0, f32x4_mul(x0, wa));
        a1 = f32x4_add(a1, f32x4_mul(x1, wa));
        a2 = f32x4_add(a2, f32x4_mul(x2, wa));
        a3 = f32x4_add(a3, f32x4_mul(x3, wa));
        a4 = f32x4_add(a4, f32x4_mul(x4, wa));
        a5 = f32x4_add(a5, f32x4_mul(x5, wa));
        a6 = f32x4_add(a6, f32x4_mul(x6, wa));
        a7 = f32x4_add(a7, f32x4_mul(x7, wa));
        b0 = f32x4_add(b0, f32x4_mul(x0, wb));
        b1 = f32x4_add(b1, f32x4_mul(x1, wb));
        b2 = f32x4_add(b2, f32x4_mul(x2, wb));
        b3 = f32x4_add(b3, f32x4_mul(x3, wb));
        b4 = f32x4_add(b4, f32x4_mul(x4, wb));
        b5 = f32x4_add(b5, f32x4_mul(x5, wb));
        b6 = f32x4_add(b6, f32x4_mul(x6, wb));
        b7 = f32x4_add(b7, f32x4_mul(x7, wb));
        c0 = f32x4_add(c0, f32x4_mul(x0, wc));
        c1 = f32x4_add(c1, f32x4_mul(x1, wc));
        c2 = f32x4_add(c2, f32x4_mul(x2, wc));
        c3 = f32x4_add(c3, f32x4_mul(x3, wc));
        c4 = f32x4_add(c4, f32x4_mul(x4, wc));
        c5 = f32x4_add(c5, f32x4_mul(x5, wc));
        c6 = f32x4_add(c6, f32x4_mul(x6, wc));
        c7 = f32x4_add(c7, f32x4_mul(x7, wc));
        col += 4;
    }
    let mut out_a = [
        sum_f32x4(a0), sum_f32x4(a1), sum_f32x4(a2), sum_f32x4(a3),
        sum_f32x4(a4), sum_f32x4(a5), sum_f32x4(a6), sum_f32x4(a7),
    ];
    let mut out_b = [
        sum_f32x4(b0), sum_f32x4(b1), sum_f32x4(b2), sum_f32x4(b3),
        sum_f32x4(b4), sum_f32x4(b5), sum_f32x4(b6), sum_f32x4(b7),
    ];
    let mut out_c = [
        sum_f32x4(c0), sum_f32x4(c1), sum_f32x4(c2), sum_f32x4(c3),
        sum_f32x4(c4), sum_f32x4(c5), sum_f32x4(c6), sum_f32x4(c7),
    ];
    while col < in_dim {
        let wa = *weight_a.get_unchecked(col) as f32;
        let wb = *weight_b.get_unchecked(col) as f32;
        let wc = *weight_c.get_unchecked(col) as f32;
        for local_row in 0..8 {
            let x = *input.get_unchecked((row + local_row) * in_dim + col);
            out_a[local_row] += x * wa;
            out_b[local_row] += x * wb;
            out_c[local_row] += x * wc;
        }
        col += 1;
    }
    (out_a, out_b, out_c)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn sum_f32x4(value: core::arch::wasm32::v128) -> f32 {
    use core::arch::wasm32::*;
    f32x4_extract_lane::<0>(value)
        + f32x4_extract_lane::<1>(value)
        + f32x4_extract_lane::<2>(value)
        + f32x4_extract_lane::<3>(value)
}

fn dot8_unpacked_i8_f32_triple(
    input: &[f32],
    row: usize,
    in_dim: usize,
    weight_a: &[i8],
    weight_b: &[i8],
    weight_c: &[i8],
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot8_unpacked_i8_f32_triple_simd(input, row, in_dim, weight_a, weight_b, weight_c) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut out_a = [0.0f32; 8];
        let mut out_b = [0.0f32; 8];
        let mut out_c = [0.0f32; 8];
        for col in 0..in_dim {
            let wa = weight_a[col] as f32;
            let wb = weight_b[col] as f32;
            let wc = weight_c[col] as f32;
            for local_row in 0..8 {
                let x = input[(row + local_row) * in_dim + col];
                out_a[local_row] += x * wa;
                out_b[local_row] += x * wb;
                out_c[local_row] += x * wc;
            }
        }
        (out_a, out_b, out_c)
    }
}

fn quantize_rows_i8(input: &[f32], rows: usize, cols: usize) -> Result<(Vec<i8>, Vec<f32>), JsValue> {
    if input.len() != rows * cols {
        return Err(JsValue::from_str("quantize_rows_i8 input shape mismatch"));
    }
    let mut quantized = vec![0i8; input.len()];
    let mut scales = vec![1.0f32; rows];
    for row in 0..rows {
        let offset = row * cols;
        let row_values = &input[offset..offset + cols];
        let mut max_abs = 0.0f32;
        for value in row_values {
            max_abs = max_abs.max(value.abs());
        }
        if max_abs <= 1.0e-8 {
            scales[row] = 1.0;
            continue;
        }
        let scale = max_abs / 127.0;
        let inv_scale = 1.0 / scale;
        scales[row] = scale;
        for col in 0..cols {
            let q = (input[offset + col] * inv_scale).round().clamp(-127.0, 127.0);
            quantized[offset + col] = q as i8;
        }
    }
    Ok((quantized, scales))
}

fn quantize_rows_i8_grouped(input: &[f32], rows: usize, cols: usize, group_size: usize) -> Result<(Vec<i8>, Vec<f32>, usize), JsValue> {
    if input.len() != rows * cols || group_size == 0 {
        return Err(JsValue::from_str("quantize_rows_i8_grouped input shape mismatch"));
    }
    let groups = cols.div_ceil(group_size);
    let mut quantized = vec![0i8; input.len()];
    let mut scales = vec![1.0f32; rows * groups];
    for row in 0..rows {
        let row_offset = row * cols;
        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(cols);
            let mut max_abs = 0.0f32;
            for col in start..end {
                max_abs = max_abs.max(input[row_offset + col].abs());
            }
            if max_abs <= 1.0e-8 {
                scales[row * groups + group] = 1.0;
                continue;
            }
            let scale = max_abs / 127.0;
            let inv_scale = 1.0 / scale;
            scales[row * groups + group] = scale;
            for col in start..end {
                let q = (input[row_offset + col] * inv_scale).round().clamp(-127.0, 127.0);
                quantized[row_offset + col] = q as i8;
            }
        }
    }
    Ok((quantized, scales, groups))
}

fn encode_i4(value: i8) -> u8 {
    (value as u8) & 0x0f
}

fn decode_i4(nibble: u8) -> i32 {
    let value = (nibble & 0x0f) as i8;
    if value >= 8 { (value - 16) as i32 } else { value as i32 }
}

fn q4_byte_dot_table() -> &'static [i16; 65536] {
    Q4_BYTE_DOT_TABLE.get_or_init(|| {
        let mut table = Box::new([0i16; 65536]);
        for a in 0..=255usize {
            let alo = decode_i4((a & 0x0f) as u8);
            let ahi = decode_i4((a >> 4) as u8);
            for w in 0..=255usize {
                let wlo = decode_i4((w & 0x0f) as u8);
                let whi = decode_i4((w >> 4) as u8);
                table[(a << 8) | w] = (alo * wlo + ahi * whi) as i16;
            }
        }
        table
    })
}

#[inline(always)]
fn q4_byte_dot(a: u8, w: u8) -> i32 {
    q4_byte_dot_table()[((a as usize) << 8) | w as usize] as i32
}

fn quantize_rows_i4_packed(input: &[f32], rows: usize, cols: usize) -> Result<(Vec<u8>, Vec<f32>, usize), JsValue> {
    if input.len() != rows * cols {
        return Err(JsValue::from_str("quantize_rows_i4_packed input shape mismatch"));
    }
    let packed_cols = cols.div_ceil(2);
    let mut packed = vec![0u8; rows * packed_cols];
    let mut scales = vec![1.0f32; rows];
    for row in 0..rows {
        let offset = row * cols;
        let row_values = &input[offset..offset + cols];
        let mut max_abs = 0.0f32;
        for value in row_values {
            max_abs = max_abs.max(value.abs());
        }
        if max_abs <= 1.0e-8 {
            scales[row] = 1.0;
            continue;
        }
        let scale = max_abs / 7.0;
        let inv_scale = 1.0 / scale;
        scales[row] = scale;
        for col in 0..cols {
            let q = (input[offset + col] * inv_scale).round().clamp(-8.0, 7.0) as i8;
            let dst = row * packed_cols + (col >> 1);
            if col & 1 == 0 {
                packed[dst] = (packed[dst] & 0xf0) | encode_i4(q);
            } else {
                packed[dst] = (packed[dst] & 0x0f) | (encode_i4(q) << 4);
            }
        }
    }
    Ok((packed, scales, packed_cols))
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot8_i8_i8_i32_simd(input: &[i8], row: usize, in_dim: usize, weight_row: &[i8]) -> [i32; 8] {
    use core::arch::wasm32::*;

    let mut acc0 = i32x4_splat(0);
    let mut acc1 = i32x4_splat(0);
    let mut acc2 = i32x4_splat(0);
    let mut acc3 = i32x4_splat(0);
    let mut acc4 = i32x4_splat(0);
    let mut acc5 = i32x4_splat(0);
    let mut acc6 = i32x4_splat(0);
    let mut acc7 = i32x4_splat(0);
    let mut col = 0usize;
    while col + 15 < in_dim {
        let w = v128_load(weight_row.as_ptr().add(col) as *const v128);
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        let x4 = v128_load(input.as_ptr().add((row + 4) * in_dim + col) as *const v128);
        let x5 = v128_load(input.as_ptr().add((row + 5) * in_dim + col) as *const v128);
        let x6 = v128_load(input.as_ptr().add((row + 6) * in_dim + col) as *const v128);
        let x7 = v128_load(input.as_ptr().add((row + 7) * in_dim + col) as *const v128);
        acc0 = i32x4_add(acc0, dot_i8x16_i32x4(x0, w));
        acc1 = i32x4_add(acc1, dot_i8x16_i32x4(x1, w));
        acc2 = i32x4_add(acc2, dot_i8x16_i32x4(x2, w));
        acc3 = i32x4_add(acc3, dot_i8x16_i32x4(x3, w));
        acc4 = i32x4_add(acc4, dot_i8x16_i32x4(x4, w));
        acc5 = i32x4_add(acc5, dot_i8x16_i32x4(x5, w));
        acc6 = i32x4_add(acc6, dot_i8x16_i32x4(x6, w));
        acc7 = i32x4_add(acc7, dot_i8x16_i32x4(x7, w));
        col += 16;
    }
    let mut out = [
        sum_i32x4(acc0),
        sum_i32x4(acc1),
        sum_i32x4(acc2),
        sum_i32x4(acc3),
        sum_i32x4(acc4),
        sum_i32x4(acc5),
        sum_i32x4(acc6),
        sum_i32x4(acc7),
    ];
    while col < in_dim {
        let w = *weight_row.get_unchecked(col) as i32;
        for local_row in 0..8 {
            out[local_row] += *input.get_unchecked((row + local_row) * in_dim + col) as i32 * w;
        }
        col += 1;
    }
    out
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn dot_i8x16_i32x4(a: core::arch::wasm32::v128, b: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    let lo = i16x8_extmul_low_i8x16(a, b);
    let hi = i16x8_extmul_high_i8x16(a, b);
    i32x4_add(i32x4_extadd_pairwise_i16x8(lo), i32x4_extadd_pairwise_i16x8(hi))
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn sum_i32x4(value: core::arch::wasm32::v128) -> i32 {
    use core::arch::wasm32::*;
    i32x4_extract_lane::<0>(value)
        + i32x4_extract_lane::<1>(value)
        + i32x4_extract_lane::<2>(value)
        + i32x4_extract_lane::<3>(value)
}

fn dot8_i8_i8_i32(input: &[i8], row: usize, in_dim: usize, weight_row: &[i8]) -> [i32; 8] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot8_i8_i8_i32_simd(input, row, in_dim, weight_row) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut out = [0i32; 8];
        for col in 0..in_dim {
            let w = weight_row[col] as i32;
            for local_row in 0..8 {
                out[local_row] += input[(row + local_row) * in_dim + col] as i32 * w;
            }
        }
        out
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot8_i8_i8_i32_pair_simd(
    input: &[i8],
    row: usize,
    in_dim: usize,
    weight_a: &[i8],
    weight_b: &[i8],
) -> ([i32; 8], [i32; 8]) {
    use core::arch::wasm32::*;

    let mut a0 = i32x4_splat(0);
    let mut a1 = i32x4_splat(0);
    let mut a2 = i32x4_splat(0);
    let mut a3 = i32x4_splat(0);
    let mut a4 = i32x4_splat(0);
    let mut a5 = i32x4_splat(0);
    let mut a6 = i32x4_splat(0);
    let mut a7 = i32x4_splat(0);
    let mut b0 = i32x4_splat(0);
    let mut b1 = i32x4_splat(0);
    let mut b2 = i32x4_splat(0);
    let mut b3 = i32x4_splat(0);
    let mut b4 = i32x4_splat(0);
    let mut b5 = i32x4_splat(0);
    let mut b6 = i32x4_splat(0);
    let mut b7 = i32x4_splat(0);
    let mut col = 0usize;
    while col + 15 < in_dim {
        let wa = v128_load(weight_a.as_ptr().add(col) as *const v128);
        let wb = v128_load(weight_b.as_ptr().add(col) as *const v128);
        let x0 = v128_load(input.as_ptr().add(row * in_dim + col) as *const v128);
        let x1 = v128_load(input.as_ptr().add((row + 1) * in_dim + col) as *const v128);
        let x2 = v128_load(input.as_ptr().add((row + 2) * in_dim + col) as *const v128);
        let x3 = v128_load(input.as_ptr().add((row + 3) * in_dim + col) as *const v128);
        let x4 = v128_load(input.as_ptr().add((row + 4) * in_dim + col) as *const v128);
        let x5 = v128_load(input.as_ptr().add((row + 5) * in_dim + col) as *const v128);
        let x6 = v128_load(input.as_ptr().add((row + 6) * in_dim + col) as *const v128);
        let x7 = v128_load(input.as_ptr().add((row + 7) * in_dim + col) as *const v128);
        a0 = i32x4_add(a0, dot_i8x16_i32x4(x0, wa));
        a1 = i32x4_add(a1, dot_i8x16_i32x4(x1, wa));
        a2 = i32x4_add(a2, dot_i8x16_i32x4(x2, wa));
        a3 = i32x4_add(a3, dot_i8x16_i32x4(x3, wa));
        a4 = i32x4_add(a4, dot_i8x16_i32x4(x4, wa));
        a5 = i32x4_add(a5, dot_i8x16_i32x4(x5, wa));
        a6 = i32x4_add(a6, dot_i8x16_i32x4(x6, wa));
        a7 = i32x4_add(a7, dot_i8x16_i32x4(x7, wa));
        b0 = i32x4_add(b0, dot_i8x16_i32x4(x0, wb));
        b1 = i32x4_add(b1, dot_i8x16_i32x4(x1, wb));
        b2 = i32x4_add(b2, dot_i8x16_i32x4(x2, wb));
        b3 = i32x4_add(b3, dot_i8x16_i32x4(x3, wb));
        b4 = i32x4_add(b4, dot_i8x16_i32x4(x4, wb));
        b5 = i32x4_add(b5, dot_i8x16_i32x4(x5, wb));
        b6 = i32x4_add(b6, dot_i8x16_i32x4(x6, wb));
        b7 = i32x4_add(b7, dot_i8x16_i32x4(x7, wb));
        col += 16;
    }
    let mut out_a = [sum_i32x4(a0), sum_i32x4(a1), sum_i32x4(a2), sum_i32x4(a3), sum_i32x4(a4), sum_i32x4(a5), sum_i32x4(a6), sum_i32x4(a7)];
    let mut out_b = [sum_i32x4(b0), sum_i32x4(b1), sum_i32x4(b2), sum_i32x4(b3), sum_i32x4(b4), sum_i32x4(b5), sum_i32x4(b6), sum_i32x4(b7)];
    while col < in_dim {
        let wa = *weight_a.get_unchecked(col) as i32;
        let wb = *weight_b.get_unchecked(col) as i32;
        for local_row in 0..8 {
            let x = *input.get_unchecked((row + local_row) * in_dim + col) as i32;
            out_a[local_row] += x * wa;
            out_b[local_row] += x * wb;
        }
        col += 1;
    }
    (out_a, out_b)
}

fn dot8_i8_i8_i32_pair(input: &[i8], row: usize, in_dim: usize, weight_a: &[i8], weight_b: &[i8]) -> ([i32; 8], [i32; 8]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot8_i8_i8_i32_pair_simd(input, row, in_dim, weight_a, weight_b) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut out_a = [0i32; 8];
        let mut out_b = [0i32; 8];
        for col in 0..in_dim {
            let wa = weight_a[col] as i32;
            let wb = weight_b[col] as i32;
            for local_row in 0..8 {
                let x = input[(row + local_row) * in_dim + col] as i32;
                out_a[local_row] += x * wa;
                out_b[local_row] += x * wb;
            }
        }
        (out_a, out_b)
    }
}

fn dot_i8_i8_i32(input_row: &[i8], weight_row: &[i8], cols: usize) -> i32 {
    let mut acc = 0i32;
    for col in 0..cols {
        acc += input_row[col] as i32 * weight_row[col] as i32;
    }
    acc
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_i8_i8_i32_quad_simd(
    input_row: &[i8],
    weight_a: &[i8],
    weight_b: &[i8],
    weight_c: &[i8],
    weight_d: &[i8],
    cols: usize,
) -> [i32; 4] {
    use core::arch::wasm32::*;

    let mut acc_a = i32x4_splat(0);
    let mut acc_b = i32x4_splat(0);
    let mut acc_c = i32x4_splat(0);
    let mut acc_d = i32x4_splat(0);
    let mut col = 0usize;
    while col + 31 < cols {
        let x0 = v128_load(input_row.as_ptr().add(col) as *const v128);
        let x1 = v128_load(input_row.as_ptr().add(col + 16) as *const v128);
        let wa0 = v128_load(weight_a.as_ptr().add(col) as *const v128);
        let wa1 = v128_load(weight_a.as_ptr().add(col + 16) as *const v128);
        let wb0 = v128_load(weight_b.as_ptr().add(col) as *const v128);
        let wb1 = v128_load(weight_b.as_ptr().add(col + 16) as *const v128);
        let wc0 = v128_load(weight_c.as_ptr().add(col) as *const v128);
        let wc1 = v128_load(weight_c.as_ptr().add(col + 16) as *const v128);
        let wd0 = v128_load(weight_d.as_ptr().add(col) as *const v128);
        let wd1 = v128_load(weight_d.as_ptr().add(col + 16) as *const v128);
        acc_a = i32x4_add(acc_a, i32x4_add(dot_i8x16_i32x4(x0, wa0), dot_i8x16_i32x4(x1, wa1)));
        acc_b = i32x4_add(acc_b, i32x4_add(dot_i8x16_i32x4(x0, wb0), dot_i8x16_i32x4(x1, wb1)));
        acc_c = i32x4_add(acc_c, i32x4_add(dot_i8x16_i32x4(x0, wc0), dot_i8x16_i32x4(x1, wc1)));
        acc_d = i32x4_add(acc_d, i32x4_add(dot_i8x16_i32x4(x0, wd0), dot_i8x16_i32x4(x1, wd1)));
        col += 32;
    }
    while col + 15 < cols {
        let x = v128_load(input_row.as_ptr().add(col) as *const v128);
        let wa = v128_load(weight_a.as_ptr().add(col) as *const v128);
        let wb = v128_load(weight_b.as_ptr().add(col) as *const v128);
        let wc = v128_load(weight_c.as_ptr().add(col) as *const v128);
        let wd = v128_load(weight_d.as_ptr().add(col) as *const v128);
        acc_a = i32x4_add(acc_a, dot_i8x16_i32x4(x, wa));
        acc_b = i32x4_add(acc_b, dot_i8x16_i32x4(x, wb));
        acc_c = i32x4_add(acc_c, dot_i8x16_i32x4(x, wc));
        acc_d = i32x4_add(acc_d, dot_i8x16_i32x4(x, wd));
        col += 16;
    }
    let mut out = [sum_i32x4(acc_a), sum_i32x4(acc_b), sum_i32x4(acc_c), sum_i32x4(acc_d)];
    while col < cols {
        let x = *input_row.get_unchecked(col) as i32;
        out[0] += x * *weight_a.get_unchecked(col) as i32;
        out[1] += x * *weight_b.get_unchecked(col) as i32;
        out[2] += x * *weight_c.get_unchecked(col) as i32;
        out[3] += x * *weight_d.get_unchecked(col) as i32;
        col += 1;
    }
    out
}

fn dot_i8_i8_i32_quad(
    input_row: &[i8],
    weight_a: &[i8],
    weight_b: &[i8],
    weight_c: &[i8],
    weight_d: &[i8],
    cols: usize,
) -> [i32; 4] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot_i8_i8_i32_quad_simd(input_row, weight_a, weight_b, weight_c, weight_d, cols) };
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut out = [0i32; 4];
        for col in 0..cols {
            let x = input_row[col] as i32;
            out[0] += x * weight_a[col] as i32;
            out[1] += x * weight_b[col] as i32;
            out[2] += x * weight_c[col] as i32;
            out[3] += x * weight_d[col] as i32;
        }
        out
    }
}

fn dot_q4_q4_packed_i32(input_row: &[u8], weight_row: &[u8], cols: usize) -> i32 {
    let packed_cols = cols.div_ceil(2);
    let mut acc = 0i32;
    let full_pairs = cols / 2;
    for packed_col in 0..full_pairs {
        acc += q4_byte_dot(input_row[packed_col], weight_row[packed_col]);
    }
    if cols & 1 != 0 {
        acc += decode_i4(input_row[packed_cols - 1] & 0x0f) * decode_i4(weight_row[packed_cols - 1] & 0x0f);
    }
    acc
}

#[wasm_bindgen]
pub struct Q4LinearHandle {
    packed_weight: Vec<u8>,
    unpacked_weight: Vec<i8>,
    row_scales: Vec<f32>,
    bias_values: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
    packed_cols: usize,
}

#[wasm_bindgen]
impl Q4LinearHandle {
    #[wasm_bindgen(constructor)]
    pub fn new(
        packed_weight: &[u8],
        row_scales_f16: &[u16],
        bias_values: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Q4LinearHandle, JsValue> {
        if in_dim == 0 || out_dim == 0 {
            return Err(JsValue::from_str("Q4LinearHandle dimensions must be non-zero"));
        }
        let packed_cols = in_dim.div_ceil(2);
        if packed_weight.len() < out_dim * packed_cols {
            return Err(JsValue::from_str("Q4LinearHandle packed_weight is shorter than layout requires"));
        }
        if row_scales_f16.len() < out_dim {
            return Err(JsValue::from_str("Q4LinearHandle row_scales_f16 length is shorter than out_dim"));
        }
        let row_scales = row_scales_f16
            .iter()
            .take(out_dim)
            .map(|bits| f16_to_f32(*bits))
            .collect();
        let mut unpacked_weight = vec![0i8; out_dim * in_dim];
        for out_idx in 0..out_dim {
            let packed_row = &packed_weight[out_idx * packed_cols..(out_idx + 1) * packed_cols];
            let dst = &mut unpacked_weight[out_idx * in_dim..(out_idx + 1) * in_dim];
            let mut col = 0usize;
            let mut packed_col = 0usize;
            while col + 1 < in_dim {
                let packed = packed_row[packed_col];
                let lo = (packed & 0x0f) as i8;
                let hi = (packed >> 4) as i8;
                dst[col] = if lo >= 8 { lo - 16 } else { lo };
                dst[col + 1] = if hi >= 8 { hi - 16 } else { hi };
                col += 2;
                packed_col += 1;
            }
            if col < in_dim {
                let lo = (packed_row[packed_col] & 0x0f) as i8;
                dst[col] = if lo >= 8 { lo - 16 } else { lo };
            }
        }
        Ok(Q4LinearHandle {
            packed_weight: packed_weight[..out_dim * packed_cols].to_vec(),
            unpacked_weight,
            row_scales,
            bias_values: bias_values.to_vec(),
            in_dim,
            out_dim,
            packed_cols,
        })
    }

    fn run_into(
        &self,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
        output_row_stride: usize,
        output_col_offset: usize,
    ) -> Result<(), JsValue> {
        if rows == 0 {
            return Ok(());
        }
        if input.len() != rows * self.in_dim {
            return Err(JsValue::from_str("Q4LinearHandle input length does not match rows * in_dim"));
        }
        if output.len() < rows * output_row_stride || output_col_offset + self.out_dim > output_row_stride {
            return Err(JsValue::from_str("Q4LinearHandle output shape mismatch"));
        }
        let mut out_idx = 0usize;
        while out_idx + 3 < self.out_dim {
            let weight_a = &self.unpacked_weight[out_idx * self.in_dim..(out_idx + 1) * self.in_dim];
            let weight_b = &self.unpacked_weight[(out_idx + 1) * self.in_dim..(out_idx + 2) * self.in_dim];
            let weight_c = &self.unpacked_weight[(out_idx + 2) * self.in_dim..(out_idx + 3) * self.in_dim];
            let weight_d = &self.unpacked_weight[(out_idx + 3) * self.in_dim..(out_idx + 4) * self.in_dim];
            let scale = self.row_scales[out_idx];
            let scale_b = self.row_scales[out_idx + 1];
            let scale_c = self.row_scales[out_idx + 2];
            let scale_d = self.row_scales[out_idx + 3];
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            let bias_b = self.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
            let bias_c = self.bias_values.get(out_idx + 2).copied().unwrap_or(0.0);
            let bias_d = self.bias_values.get(out_idx + 3).copied().unwrap_or(0.0);
            let mut row = 0usize;
            while row + 3 < rows {
                let (acc, acc_b, acc_c, acc_d) =
                    dot4_unpacked_i8_f32_quad(input, row, self.in_dim, weight_a, weight_b, weight_c, weight_d);
                for local_row in 0..4 {
                    let base = (row + local_row) * output_row_stride + output_col_offset + out_idx;
                    output[base] = acc[local_row] * scale + bias;
                    output[base + 1] = acc_b[local_row] * scale_b + bias_b;
                    output[base + 2] = acc_c[local_row] * scale_c + bias_c;
                    output[base + 3] = acc_d[local_row] * scale_d + bias_d;
                }
                row += 4;
            }
            while row < rows {
                let input_row = &input[row * self.in_dim..(row + 1) * self.in_dim];
                let mut acc = 0.0f32;
                let mut acc_b = 0.0f32;
                let mut acc_c = 0.0f32;
                let mut acc_d = 0.0f32;
                let mut col = 0usize;
                while col + 3 < self.in_dim {
                    acc += input_row[col] * weight_a[col] as f32
                        + input_row[col + 1] * weight_a[col + 1] as f32
                        + input_row[col + 2] * weight_a[col + 2] as f32
                        + input_row[col + 3] * weight_a[col + 3] as f32;
                    acc_b += input_row[col] * weight_b[col] as f32
                        + input_row[col + 1] * weight_b[col + 1] as f32
                        + input_row[col + 2] * weight_b[col + 2] as f32
                        + input_row[col + 3] * weight_b[col + 3] as f32;
                    acc_c += input_row[col] * weight_c[col] as f32
                        + input_row[col + 1] * weight_c[col + 1] as f32
                        + input_row[col + 2] * weight_c[col + 2] as f32
                        + input_row[col + 3] * weight_c[col + 3] as f32;
                    acc_d += input_row[col] * weight_d[col] as f32
                        + input_row[col + 1] * weight_d[col + 1] as f32
                        + input_row[col + 2] * weight_d[col + 2] as f32
                        + input_row[col + 3] * weight_d[col + 3] as f32;
                    col += 4;
                }
                while col < self.in_dim {
                    let x = input_row[col];
                    acc += x * weight_a[col] as f32;
                    acc_b += x * weight_b[col] as f32;
                    acc_c += x * weight_c[col] as f32;
                    acc_d += x * weight_d[col] as f32;
                    col += 1;
                }
                let base = row * output_row_stride + output_col_offset + out_idx;
                output[base] = acc * scale + bias;
                output[base + 1] = acc_b * scale_b + bias_b;
                output[base + 2] = acc_c * scale_c + bias_c;
                output[base + 3] = acc_d * scale_d + bias_d;
                row += 1;
            }
            out_idx += 4;
        }
        while out_idx + 1 < self.out_dim {
            let weight_a = &self.unpacked_weight[out_idx * self.in_dim..(out_idx + 1) * self.in_dim];
            let weight_b = &self.unpacked_weight[(out_idx + 1) * self.in_dim..(out_idx + 2) * self.in_dim];
            let scale = self.row_scales[out_idx];
            let scale_b = self.row_scales[out_idx + 1];
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            let bias_b = self.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
            let mut row = 0usize;
            while row + 7 < rows {
                let (acc, acc_b) = dot8_unpacked_i8_f32_pair(input, row, self.in_dim, weight_a, weight_b);
                output[row * output_row_stride + output_col_offset + out_idx] = acc[0] * scale + bias;
                output[row * output_row_stride + output_col_offset + out_idx + 1] = acc_b[0] * scale_b + bias_b;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx] = acc[1] * scale + bias;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx + 1] = acc_b[1] * scale_b + bias_b;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx] = acc[2] * scale + bias;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx + 1] = acc_b[2] * scale_b + bias_b;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx] = acc[3] * scale + bias;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx + 1] = acc_b[3] * scale_b + bias_b;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx] = acc[4] * scale + bias;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx + 1] = acc_b[4] * scale_b + bias_b;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx] = acc[5] * scale + bias;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx + 1] = acc_b[5] * scale_b + bias_b;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx] = acc[6] * scale + bias;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx + 1] = acc_b[6] * scale_b + bias_b;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx] = acc[7] * scale + bias;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx + 1] = acc_b[7] * scale_b + bias_b;
                row += 8;
            }
            while row < rows {
                let input_row = &input[row * self.in_dim..(row + 1) * self.in_dim];
                let mut acc = 0.0f32;
                let mut acc_b = 0.0f32;
                let mut col = 0usize;
                while col + 3 < self.in_dim {
                    acc += input_row[col] * weight_a[col] as f32
                        + input_row[col + 1] * weight_a[col + 1] as f32
                        + input_row[col + 2] * weight_a[col + 2] as f32
                        + input_row[col + 3] * weight_a[col + 3] as f32;
                    acc_b += input_row[col] * weight_b[col] as f32
                        + input_row[col + 1] * weight_b[col + 1] as f32
                        + input_row[col + 2] * weight_b[col + 2] as f32
                        + input_row[col + 3] * weight_b[col + 3] as f32;
                    col += 4;
                }
                while col < self.in_dim {
                    acc += input_row[col] * weight_a[col] as f32;
                    acc_b += input_row[col] * weight_b[col] as f32;
                    col += 1;
                }
                output[row * output_row_stride + output_col_offset + out_idx] = acc * scale + bias;
                output[row * output_row_stride + output_col_offset + out_idx + 1] = acc_b * scale_b + bias_b;
                row += 1;
            }
            out_idx += 2;
        }
        while out_idx < self.out_dim {
            let weight_row = &self.unpacked_weight[out_idx * self.in_dim..(out_idx + 1) * self.in_dim];
            let scale = self.row_scales[out_idx];
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            let mut row = 0usize;
            while row + 7 < rows {
                let acc = dot8_unpacked_i8_f32(input, row, self.in_dim, weight_row);
                output[row * output_row_stride + output_col_offset + out_idx] = acc[0] * scale + bias;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx] = acc[1] * scale + bias;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx] = acc[2] * scale + bias;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx] = acc[3] * scale + bias;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx] = acc[4] * scale + bias;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx] = acc[5] * scale + bias;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx] = acc[6] * scale + bias;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx] = acc[7] * scale + bias;
                row += 8;
            }
            while row < rows {
                let input_row = &input[row * self.in_dim..(row + 1) * self.in_dim];
                let mut acc = 0.0f32;
                let mut col = 0usize;
                while col + 3 < self.in_dim {
                    acc += input_row[col] * weight_row[col] as f32
                        + input_row[col + 1] * weight_row[col + 1] as f32
                        + input_row[col + 2] * weight_row[col + 2] as f32
                        + input_row[col + 3] * weight_row[col + 3] as f32;
                    col += 4;
                }
                while col < self.in_dim {
                    acc += input_row[col] * weight_row[col] as f32;
                    col += 1;
                }
                output[row * output_row_stride + output_col_offset + out_idx] = acc * scale + bias;
                row += 1;
            }
            out_idx += 1;
        }
        Ok(())
    }

    fn add_columns_into(
        &self,
        input: &[f32],
        rows: usize,
        weight_col_offset: usize,
        cols: usize,
        output: &mut [f32],
        output_row_stride: usize,
        output_col_offset: usize,
    ) -> Result<(), JsValue> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        if weight_col_offset + cols > self.in_dim || input.len() != rows * cols {
            return Err(JsValue::from_str("Q4LinearHandle partial-column input shape mismatch"));
        }
        if output.len() < rows * output_row_stride || output_col_offset + self.out_dim > output_row_stride {
            return Err(JsValue::from_str("Q4LinearHandle partial-column output shape mismatch"));
        }
        let mut out_idx = 0usize;
        while out_idx + 1 < self.out_dim {
            let row_start = out_idx * self.in_dim + weight_col_offset;
            let next_row_start = (out_idx + 1) * self.in_dim + weight_col_offset;
            let weight_a = &self.unpacked_weight[row_start..row_start + cols];
            let weight_b = &self.unpacked_weight[next_row_start..next_row_start + cols];
            let scale = self.row_scales[out_idx];
            let scale_b = self.row_scales[out_idx + 1];
            let mut row = 0usize;
            while row + 7 < rows {
                let (acc, acc_b) = dot8_unpacked_i8_f32_pair(input, row, cols, weight_a, weight_b);
                output[row * output_row_stride + output_col_offset + out_idx] += acc[0] * scale;
                output[row * output_row_stride + output_col_offset + out_idx + 1] += acc_b[0] * scale_b;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx] += acc[1] * scale;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx + 1] += acc_b[1] * scale_b;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx] += acc[2] * scale;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx + 1] += acc_b[2] * scale_b;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx] += acc[3] * scale;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx + 1] += acc_b[3] * scale_b;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx] += acc[4] * scale;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx + 1] += acc_b[4] * scale_b;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx] += acc[5] * scale;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx + 1] += acc_b[5] * scale_b;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx] += acc[6] * scale;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx + 1] += acc_b[6] * scale_b;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx] += acc[7] * scale;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx + 1] += acc_b[7] * scale_b;
                row += 8;
            }
            while row < rows {
                let input_row = &input[row * cols..(row + 1) * cols];
                let mut acc = 0.0f32;
                let mut acc_b = 0.0f32;
                for col in 0..cols {
                    acc += input_row[col] * weight_a[col] as f32;
                    acc_b += input_row[col] * weight_b[col] as f32;
                }
                output[row * output_row_stride + output_col_offset + out_idx] += acc * scale;
                output[row * output_row_stride + output_col_offset + out_idx + 1] += acc_b * scale_b;
                row += 1;
            }
            out_idx += 2;
        }
        while out_idx < self.out_dim {
            let row_start = out_idx * self.in_dim + weight_col_offset;
            let weight_row = &self.unpacked_weight[row_start..row_start + cols];
            let scale = self.row_scales[out_idx];
            let mut row = 0usize;
            while row + 7 < rows {
                let acc = dot8_unpacked_i8_f32(input, row, cols, weight_row);
                output[row * output_row_stride + output_col_offset + out_idx] += acc[0] * scale;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx] += acc[1] * scale;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx] += acc[2] * scale;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx] += acc[3] * scale;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx] += acc[4] * scale;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx] += acc[5] * scale;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx] += acc[6] * scale;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx] += acc[7] * scale;
                row += 8;
            }
            while row < rows {
                let input_row = &input[row * cols..(row + 1) * cols];
                let mut acc = 0.0f32;
                for col in 0..cols {
                    acc += input_row[col] * weight_row[col] as f32;
                }
                output[row * output_row_stride + output_col_offset + out_idx] += acc * scale;
                row += 1;
            }
            out_idx += 1;
        }
        Ok(())
    }

    fn run_impl(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let mut output = vec![0.0f32; rows * self.out_dim];
        self.run_into(input, rows, &mut output, self.out_dim, 0)?;
        Ok(output)
    }

    fn run_i8act_impl(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        if F5_USE_GROUPED_I8ACT_Q4_LINEAR {
            let (quantized, act_scales, groups) = quantize_rows_i8_grouped(input, rows, self.in_dim, F5_GROUPED_I8ACT_GROUP)?;
            return self.run_grouped_i8act_impl(&quantized, &act_scales, rows, groups, F5_GROUPED_I8ACT_GROUP);
        }
        let (quantized, act_scales) = quantize_rows_i8(input, rows, self.in_dim)?;
        if F5_USE_TILED_I8ACT_Q4_LINEAR {
            return self.run_quantized_i8act_quad_impl(&quantized, &act_scales, rows);
        }
        self.run_quantized_i8act_impl(&quantized, &act_scales, rows)
    }

    fn run_q4act_impl(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let (packed_input, act_scales, input_packed_cols) = quantize_rows_i4_packed(input, rows, self.in_dim)?;
        if input_packed_cols != self.packed_cols {
            return Err(JsValue::from_str("Q4LinearHandle q4 activation packed shape mismatch"));
        }
        let mut output = vec![0.0f32; rows * self.out_dim];
        for out_idx in 0..self.out_dim {
            let weight_row = &self.packed_weight[out_idx * self.packed_cols..(out_idx + 1) * self.packed_cols];
            let weight_scale = self.row_scales[out_idx];
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            for row in 0..rows {
                let input_row = &packed_input[row * input_packed_cols..(row + 1) * input_packed_cols];
                let dot = dot_q4_q4_packed_i32(input_row, weight_row, self.in_dim);
                output[row * self.out_dim + out_idx] = dot as f32 * act_scales[row] * weight_scale + bias;
            }
        }
        Ok(output)
    }

    fn run_quantized_i8act_impl(&self, quantized: &[i8], act_scales: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        let mut output = vec![0.0f32; rows * self.out_dim];
        if F5_USE_TILED_I8ACT_Q4_LINEAR {
            self.run_quantized_i8act_quad_into(quantized, act_scales, rows, &mut output, self.out_dim, 0)?;
        } else {
            self.run_quantized_i8act_into(quantized, act_scales, rows, &mut output, self.out_dim, 0)?;
        }
        Ok(output)
    }

    fn run_quantized_i8act_quad_impl(&self, quantized: &[i8], act_scales: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        let mut output = vec![0.0f32; rows * self.out_dim];
        self.run_quantized_i8act_quad_into(quantized, act_scales, rows, &mut output, self.out_dim, 0)?;
        Ok(output)
    }

    fn run_grouped_i8act_impl(&self, quantized: &[i8], act_scales: &[f32], rows: usize, groups: usize, group_size: usize) -> Result<Vec<f32>, JsValue> {
        let mut output = vec![0.0f32; rows * self.out_dim];
        self.run_grouped_i8act_into(quantized, act_scales, rows, groups, group_size, &mut output, self.out_dim, 0)?;
        Ok(output)
    }

    fn run_grouped_i8act_into(
        &self,
        quantized: &[i8],
        act_scales: &[f32],
        rows: usize,
        groups: usize,
        group_size: usize,
        output: &mut [f32],
        output_row_stride: usize,
        output_col_offset: usize,
    ) -> Result<(), JsValue> {
        if quantized.len() != rows * self.in_dim || act_scales.len() < rows * groups {
            return Err(JsValue::from_str("Q4LinearHandle grouped quantized input shape mismatch"));
        }
        if output.len() < rows * output_row_stride || output_col_offset + self.out_dim > output_row_stride {
            return Err(JsValue::from_str("Q4LinearHandle grouped quantized output shape mismatch"));
        }
        for out_idx in 0..self.out_dim {
            let weight_row = &self.unpacked_weight[out_idx * self.in_dim..(out_idx + 1) * self.in_dim];
            let weight_scale = self.row_scales[out_idx];
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            for row in 0..rows {
                let input_row = &quantized[row * self.in_dim..(row + 1) * self.in_dim];
                let mut acc = 0.0f32;
                for group in 0..groups {
                    let start = group * group_size;
                    let end = (start + group_size).min(self.in_dim);
                    let dot = dot_i8_i8_i32(&input_row[start..end], &weight_row[start..end], end - start);
                    acc += dot as f32 * act_scales[row * groups + group];
                }
                output[row * output_row_stride + output_col_offset + out_idx] = acc * weight_scale + bias;
            }
        }
        Ok(())
    }

    fn run_quantized_i8act_quad_into(
        &self,
        quantized: &[i8],
        act_scales: &[f32],
        rows: usize,
        output: &mut [f32],
        output_row_stride: usize,
        output_col_offset: usize,
    ) -> Result<(), JsValue> {
        if quantized.len() != rows * self.in_dim || act_scales.len() < rows {
            return Err(JsValue::from_str("Q4LinearHandle tiled quantized input shape mismatch"));
        }
        if output.len() < rows * output_row_stride || output_col_offset + self.out_dim > output_row_stride {
            return Err(JsValue::from_str("Q4LinearHandle tiled quantized output shape mismatch"));
        }
        let mut out_idx = 0usize;
        while out_idx + 3 < self.out_dim {
            let weight_a = &self.unpacked_weight[out_idx * self.in_dim..(out_idx + 1) * self.in_dim];
            let weight_b = &self.unpacked_weight[(out_idx + 1) * self.in_dim..(out_idx + 2) * self.in_dim];
            let weight_c = &self.unpacked_weight[(out_idx + 2) * self.in_dim..(out_idx + 3) * self.in_dim];
            let weight_d = &self.unpacked_weight[(out_idx + 3) * self.in_dim..(out_idx + 4) * self.in_dim];
            let scale_a = self.row_scales[out_idx];
            let scale_b = self.row_scales[out_idx + 1];
            let scale_c = self.row_scales[out_idx + 2];
            let scale_d = self.row_scales[out_idx + 3];
            let bias_a = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            let bias_b = self.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
            let bias_c = self.bias_values.get(out_idx + 2).copied().unwrap_or(0.0);
            let bias_d = self.bias_values.get(out_idx + 3).copied().unwrap_or(0.0);
            for row in 0..rows {
                let input_row = &quantized[row * self.in_dim..(row + 1) * self.in_dim];
                let dots = dot_i8_i8_i32_quad(input_row, weight_a, weight_b, weight_c, weight_d, self.in_dim);
                let act_scale = act_scales[row];
                let dst = row * output_row_stride + output_col_offset + out_idx;
                output[dst] = dots[0] as f32 * act_scale * scale_a + bias_a;
                output[dst + 1] = dots[1] as f32 * act_scale * scale_b + bias_b;
                output[dst + 2] = dots[2] as f32 * act_scale * scale_c + bias_c;
                output[dst + 3] = dots[3] as f32 * act_scale * scale_d + bias_d;
            }
            out_idx += 4;
        }
        while out_idx < self.out_dim {
            let weight_row = &self.unpacked_weight[out_idx * self.in_dim..(out_idx + 1) * self.in_dim];
            let weight_scale = self.row_scales[out_idx];
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            for row in 0..rows {
                let input_row = &quantized[row * self.in_dim..(row + 1) * self.in_dim];
                let dot = dot_i8_i8_i32(input_row, weight_row, self.in_dim);
                output[row * output_row_stride + output_col_offset + out_idx] = dot as f32 * act_scales[row] * weight_scale + bias;
            }
            out_idx += 1;
        }
        Ok(())
    }

    fn run_quantized_i8act_into(
        &self,
        quantized: &[i8],
        act_scales: &[f32],
        rows: usize,
        output: &mut [f32],
        output_row_stride: usize,
        output_col_offset: usize,
    ) -> Result<(), JsValue> {
        if quantized.len() != rows * self.in_dim || act_scales.len() < rows {
            return Err(JsValue::from_str("Q4LinearHandle quantized input shape mismatch"));
        }
        if output.len() < rows * output_row_stride || output_col_offset + self.out_dim > output_row_stride {
            return Err(JsValue::from_str("Q4LinearHandle quantized output shape mismatch"));
        }
        let mut out_idx = 0usize;
        while out_idx + 1 < self.out_dim {
            let weight_row = &self.unpacked_weight[out_idx * self.in_dim..(out_idx + 1) * self.in_dim];
            let next_weight_row = &self.unpacked_weight[(out_idx + 1) * self.in_dim..(out_idx + 2) * self.in_dim];
            let weight_scale = self.row_scales[out_idx];
            let next_weight_scale = self.row_scales[out_idx + 1];
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            let next_bias = self.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
            let mut row = 0usize;
            while row + 7 < rows {
                let (dot, next_dot) = dot8_i8_i8_i32_pair(&quantized, row, self.in_dim, weight_row, next_weight_row);
                output[row * output_row_stride + output_col_offset + out_idx] = dot[0] as f32 * act_scales[row] * weight_scale + bias;
                output[row * output_row_stride + output_col_offset + out_idx + 1] = next_dot[0] as f32 * act_scales[row] * next_weight_scale + next_bias;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx] = dot[1] as f32 * act_scales[row + 1] * weight_scale + bias;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx + 1] = next_dot[1] as f32 * act_scales[row + 1] * next_weight_scale + next_bias;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx] = dot[2] as f32 * act_scales[row + 2] * weight_scale + bias;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx + 1] = next_dot[2] as f32 * act_scales[row + 2] * next_weight_scale + next_bias;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx] = dot[3] as f32 * act_scales[row + 3] * weight_scale + bias;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx + 1] = next_dot[3] as f32 * act_scales[row + 3] * next_weight_scale + next_bias;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx] = dot[4] as f32 * act_scales[row + 4] * weight_scale + bias;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx + 1] = next_dot[4] as f32 * act_scales[row + 4] * next_weight_scale + next_bias;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx] = dot[5] as f32 * act_scales[row + 5] * weight_scale + bias;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx + 1] = next_dot[5] as f32 * act_scales[row + 5] * next_weight_scale + next_bias;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx] = dot[6] as f32 * act_scales[row + 6] * weight_scale + bias;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx + 1] = next_dot[6] as f32 * act_scales[row + 6] * next_weight_scale + next_bias;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx] = dot[7] as f32 * act_scales[row + 7] * weight_scale + bias;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx + 1] = next_dot[7] as f32 * act_scales[row + 7] * next_weight_scale + next_bias;
                row += 8;
            }
            while row < rows {
                let input_row = &quantized[row * self.in_dim..(row + 1) * self.in_dim];
                let dot = dot_i8_i8_i32(input_row, weight_row, self.in_dim);
                let next_dot = dot_i8_i8_i32(input_row, next_weight_row, self.in_dim);
                output[row * output_row_stride + output_col_offset + out_idx] = dot as f32 * act_scales[row] * weight_scale + bias;
                output[row * output_row_stride + output_col_offset + out_idx + 1] = next_dot as f32 * act_scales[row] * next_weight_scale + next_bias;
                row += 1;
            }
            out_idx += 2;
        }
        while out_idx < self.out_dim {
            let weight_row = &self.unpacked_weight[out_idx * self.in_dim..(out_idx + 1) * self.in_dim];
            let weight_scale = self.row_scales[out_idx];
            let bias = self.bias_values.get(out_idx).copied().unwrap_or(0.0);
            let mut row = 0usize;
            while row + 7 < rows {
                let dot = dot8_i8_i8_i32(&quantized, row, self.in_dim, weight_row);
                output[row * output_row_stride + output_col_offset + out_idx] = dot[0] as f32 * act_scales[row] * weight_scale + bias;
                output[(row + 1) * output_row_stride + output_col_offset + out_idx] = dot[1] as f32 * act_scales[row + 1] * weight_scale + bias;
                output[(row + 2) * output_row_stride + output_col_offset + out_idx] = dot[2] as f32 * act_scales[row + 2] * weight_scale + bias;
                output[(row + 3) * output_row_stride + output_col_offset + out_idx] = dot[3] as f32 * act_scales[row + 3] * weight_scale + bias;
                output[(row + 4) * output_row_stride + output_col_offset + out_idx] = dot[4] as f32 * act_scales[row + 4] * weight_scale + bias;
                output[(row + 5) * output_row_stride + output_col_offset + out_idx] = dot[5] as f32 * act_scales[row + 5] * weight_scale + bias;
                output[(row + 6) * output_row_stride + output_col_offset + out_idx] = dot[6] as f32 * act_scales[row + 6] * weight_scale + bias;
                output[(row + 7) * output_row_stride + output_col_offset + out_idx] = dot[7] as f32 * act_scales[row + 7] * weight_scale + bias;
                row += 8;
            }
            while row < rows {
                let input_row = &quantized[row * self.in_dim..(row + 1) * self.in_dim];
                let dot = dot_i8_i8_i32(input_row, weight_row, self.in_dim);
                output[row * output_row_stride + output_col_offset + out_idx] = dot as f32 * act_scales[row] * weight_scale + bias;
                row += 1;
            }
            out_idx += 1;
        }
        Ok(())
    }

    pub fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        self.run_impl(input, rows)
    }
}

#[wasm_bindgen]
pub fn q4_linear3_f32(
    first: &Q4LinearHandle,
    second: &Q4LinearHandle,
    third: &Q4LinearHandle,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>, JsValue> {
    if first.in_dim != second.in_dim || first.in_dim != third.in_dim {
        return Err(JsValue::from_str("q4_linear3_f32 input dimensions do not match"));
    }
    if first.out_dim != second.out_dim || first.out_dim != third.out_dim {
        return Err(JsValue::from_str("q4_linear3_f32 output dimensions do not match"));
    }
    let out_dim = first.out_dim;
    let mut out = vec![0.0f32; rows * out_dim * 3];
    let part = rows * out_dim;
    let mut out_idx = 0usize;
    while out_idx + 1 < out_dim {
        let weight_a = &first.unpacked_weight[out_idx * first.in_dim..(out_idx + 1) * first.in_dim];
        let weight_b = &second.unpacked_weight[out_idx * second.in_dim..(out_idx + 1) * second.in_dim];
        let weight_c = &third.unpacked_weight[out_idx * third.in_dim..(out_idx + 1) * third.in_dim];
        let weight_a_next = &first.unpacked_weight[(out_idx + 1) * first.in_dim..(out_idx + 2) * first.in_dim];
        let weight_b_next = &second.unpacked_weight[(out_idx + 1) * second.in_dim..(out_idx + 2) * second.in_dim];
        let weight_c_next = &third.unpacked_weight[(out_idx + 1) * third.in_dim..(out_idx + 2) * third.in_dim];
        let scale_a = first.row_scales[out_idx];
        let scale_b = second.row_scales[out_idx];
        let scale_c = third.row_scales[out_idx];
        let scale_a_next = first.row_scales[out_idx + 1];
        let scale_b_next = second.row_scales[out_idx + 1];
        let scale_c_next = third.row_scales[out_idx + 1];
        let bias_a = first.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_b = second.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_c = third.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_a_next = first.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
        let bias_b_next = second.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
        let bias_c_next = third.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
        let mut row = 0usize;
        while row + 3 < rows {
            let (dot_a, dot_b, dot_c, dot_a_next, dot_b_next, dot_c_next) = dot4_unpacked_i8_f32_six(
                input,
                row,
                first.in_dim,
                weight_a,
                weight_b,
                weight_c,
                weight_a_next,
                weight_b_next,
                weight_c_next,
            );
            for local_row in 0..4 {
                let dst = (row + local_row) * out_dim + out_idx;
                out[dst] = dot_a[local_row] * scale_a + bias_a;
                out[dst + 1] = dot_a_next[local_row] * scale_a_next + bias_a_next;
                out[part + dst] = dot_b[local_row] * scale_b + bias_b;
                out[part + dst + 1] = dot_b_next[local_row] * scale_b_next + bias_b_next;
                out[part * 2 + dst] = dot_c[local_row] * scale_c + bias_c;
                out[part * 2 + dst + 1] = dot_c_next[local_row] * scale_c_next + bias_c_next;
            }
            row += 4;
        }
        while row < rows {
            let input_row = &input[row * first.in_dim..(row + 1) * first.in_dim];
            let mut dot_a = 0.0f32;
            let mut dot_b = 0.0f32;
            let mut dot_c = 0.0f32;
            let mut dot_a_next = 0.0f32;
            let mut dot_b_next = 0.0f32;
            let mut dot_c_next = 0.0f32;
            for col in 0..first.in_dim {
                let value = input_row[col];
                dot_a += value * weight_a[col] as f32;
                dot_b += value * weight_b[col] as f32;
                dot_c += value * weight_c[col] as f32;
                dot_a_next += value * weight_a_next[col] as f32;
                dot_b_next += value * weight_b_next[col] as f32;
                dot_c_next += value * weight_c_next[col] as f32;
            }
            let dst = row * out_dim + out_idx;
            out[dst] = dot_a * scale_a + bias_a;
            out[dst + 1] = dot_a_next * scale_a_next + bias_a_next;
            out[part + dst] = dot_b * scale_b + bias_b;
            out[part + dst + 1] = dot_b_next * scale_b_next + bias_b_next;
            out[part * 2 + dst] = dot_c * scale_c + bias_c;
            out[part * 2 + dst + 1] = dot_c_next * scale_c_next + bias_c_next;
            row += 1;
        }
        out_idx += 2;
    }
    while out_idx < out_dim {
        let weight_a = &first.unpacked_weight[out_idx * first.in_dim..(out_idx + 1) * first.in_dim];
        let weight_b = &second.unpacked_weight[out_idx * second.in_dim..(out_idx + 1) * second.in_dim];
        let weight_c = &third.unpacked_weight[out_idx * third.in_dim..(out_idx + 1) * third.in_dim];
        let scale_a = first.row_scales[out_idx];
        let scale_b = second.row_scales[out_idx];
        let scale_c = third.row_scales[out_idx];
        let bias_a = first.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_b = second.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_c = third.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let mut row = 0usize;
        while row + 7 < rows {
            let (dot_a, dot_b, dot_c) = dot8_unpacked_i8_f32_triple(input, row, first.in_dim, weight_a, weight_b, weight_c);
            for local_row in 0..8 {
                let dst = (row + local_row) * out_dim + out_idx;
                out[dst] = dot_a[local_row] * scale_a + bias_a;
                out[part + dst] = dot_b[local_row] * scale_b + bias_b;
                out[part * 2 + dst] = dot_c[local_row] * scale_c + bias_c;
            }
            row += 8;
        }
        while row < rows {
            let input_row = &input[row * first.in_dim..(row + 1) * first.in_dim];
            let mut dot_a = 0.0f32;
            let mut dot_b = 0.0f32;
            let mut dot_c = 0.0f32;
            for col in 0..first.in_dim {
                let value = input_row[col];
                dot_a += value * weight_a[col] as f32;
                dot_b += value * weight_b[col] as f32;
                dot_c += value * weight_c[col] as f32;
            }
            let dst = row * out_dim + out_idx;
            out[dst] = dot_a * scale_a + bias_a;
            out[part + dst] = dot_b * scale_b + bias_b;
            out[part * 2 + dst] = dot_c * scale_c + bias_c;
            row += 1;
        }
        out_idx += 1;
    }
    Ok(out)
}

fn q4_linear3_f32_q_row_kv_head_major(
    first: &Q4LinearHandle,
    second: &Q4LinearHandle,
    third: &Q4LinearHandle,
    input: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), JsValue> {
    if first.in_dim != second.in_dim || first.in_dim != third.in_dim {
        return Err(JsValue::from_str("q4_linear3 head-major input dimensions do not match"));
    }
    if first.out_dim != second.out_dim || first.out_dim != third.out_dim || first.out_dim != heads * head_dim {
        return Err(JsValue::from_str("q4_linear3 head-major output dimensions do not match"));
    }
    let out_dim = first.out_dim;
    let mut q_out = vec![0.0f32; rows * out_dim];
    let mut k_head = vec![0.0f32; rows * out_dim];
    let mut v_head = vec![0.0f32; rows * out_dim];
    let mut out_idx = 0usize;
    while out_idx + 1 < out_dim {
        let weight_a = &first.unpacked_weight[out_idx * first.in_dim..(out_idx + 1) * first.in_dim];
        let weight_b = &second.unpacked_weight[out_idx * second.in_dim..(out_idx + 1) * second.in_dim];
        let weight_c = &third.unpacked_weight[out_idx * third.in_dim..(out_idx + 1) * third.in_dim];
        let weight_a_next = &first.unpacked_weight[(out_idx + 1) * first.in_dim..(out_idx + 2) * first.in_dim];
        let weight_b_next = &second.unpacked_weight[(out_idx + 1) * second.in_dim..(out_idx + 2) * second.in_dim];
        let weight_c_next = &third.unpacked_weight[(out_idx + 1) * third.in_dim..(out_idx + 2) * third.in_dim];
        let scale_a = first.row_scales[out_idx];
        let scale_b = second.row_scales[out_idx];
        let scale_c = third.row_scales[out_idx];
        let scale_a_next = first.row_scales[out_idx + 1];
        let scale_b_next = second.row_scales[out_idx + 1];
        let scale_c_next = third.row_scales[out_idx + 1];
        let bias_a = first.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_b = second.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_c = third.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_a_next = first.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
        let bias_b_next = second.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
        let bias_c_next = third.bias_values.get(out_idx + 1).copied().unwrap_or(0.0);
        let head = out_idx / head_dim;
        let lane = out_idx - head * head_dim;
        let head_offset = head * rows * head_dim;
        let mut row = 0usize;
        while row + 3 < rows {
            let (dot_a, dot_b, dot_c, dot_a_next, dot_b_next, dot_c_next) = dot4_unpacked_i8_f32_six(
                input,
                row,
                first.in_dim,
                weight_a,
                weight_b,
                weight_c,
                weight_a_next,
                weight_b_next,
                weight_c_next,
            );
            for local_row in 0..4 {
                let src_row = row + local_row;
                let q_dst = src_row * out_dim + out_idx;
                let kv_dst = head_offset + src_row * head_dim + lane;
                unsafe {
                    *q_out.get_unchecked_mut(q_dst) = dot_a[local_row] * scale_a + bias_a;
                    *q_out.get_unchecked_mut(q_dst + 1) = dot_a_next[local_row] * scale_a_next + bias_a_next;
                    *k_head.get_unchecked_mut(kv_dst) = dot_b[local_row] * scale_b + bias_b;
                    *k_head.get_unchecked_mut(kv_dst + 1) = dot_b_next[local_row] * scale_b_next + bias_b_next;
                    *v_head.get_unchecked_mut(kv_dst) = dot_c[local_row] * scale_c + bias_c;
                    *v_head.get_unchecked_mut(kv_dst + 1) = dot_c_next[local_row] * scale_c_next + bias_c_next;
                }
            }
            row += 4;
        }
        while row < rows {
            let input_row = &input[row * first.in_dim..(row + 1) * first.in_dim];
            let mut dot_a = 0.0f32;
            let mut dot_b = 0.0f32;
            let mut dot_c = 0.0f32;
            let mut dot_a_next = 0.0f32;
            let mut dot_b_next = 0.0f32;
            let mut dot_c_next = 0.0f32;
            for col in 0..first.in_dim {
                let value = input_row[col];
                dot_a += value * weight_a[col] as f32;
                dot_b += value * weight_b[col] as f32;
                dot_c += value * weight_c[col] as f32;
                dot_a_next += value * weight_a_next[col] as f32;
                dot_b_next += value * weight_b_next[col] as f32;
                dot_c_next += value * weight_c_next[col] as f32;
            }
            let q_dst = row * out_dim + out_idx;
            let kv_dst = head_offset + row * head_dim + lane;
            unsafe {
                *q_out.get_unchecked_mut(q_dst) = dot_a * scale_a + bias_a;
                *q_out.get_unchecked_mut(q_dst + 1) = dot_a_next * scale_a_next + bias_a_next;
                *k_head.get_unchecked_mut(kv_dst) = dot_b * scale_b + bias_b;
                *k_head.get_unchecked_mut(kv_dst + 1) = dot_b_next * scale_b_next + bias_b_next;
                *v_head.get_unchecked_mut(kv_dst) = dot_c * scale_c + bias_c;
                *v_head.get_unchecked_mut(kv_dst + 1) = dot_c_next * scale_c_next + bias_c_next;
            }
            row += 1;
        }
        out_idx += 2;
    }
    while out_idx < out_dim {
        let weight_a = &first.unpacked_weight[out_idx * first.in_dim..(out_idx + 1) * first.in_dim];
        let weight_b = &second.unpacked_weight[out_idx * second.in_dim..(out_idx + 1) * second.in_dim];
        let weight_c = &third.unpacked_weight[out_idx * third.in_dim..(out_idx + 1) * third.in_dim];
        let scale_a = first.row_scales[out_idx];
        let scale_b = second.row_scales[out_idx];
        let scale_c = third.row_scales[out_idx];
        let bias_a = first.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_b = second.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let bias_c = third.bias_values.get(out_idx).copied().unwrap_or(0.0);
        let head = out_idx / head_dim;
        let lane = out_idx - head * head_dim;
        let head_offset = head * rows * head_dim;
        let mut row = 0usize;
        while row + 7 < rows {
            let (dot_a, dot_b, dot_c) = dot8_unpacked_i8_f32_triple(input, row, first.in_dim, weight_a, weight_b, weight_c);
            for local_row in 0..8 {
                let src_row = row + local_row;
                let q_dst = src_row * out_dim + out_idx;
                let kv_dst = head_offset + src_row * head_dim + lane;
                unsafe {
                    *q_out.get_unchecked_mut(q_dst) = dot_a[local_row] * scale_a + bias_a;
                    *k_head.get_unchecked_mut(kv_dst) = dot_b[local_row] * scale_b + bias_b;
                    *v_head.get_unchecked_mut(kv_dst) = dot_c[local_row] * scale_c + bias_c;
                }
            }
            row += 8;
        }
        while row < rows {
            let input_row = &input[row * first.in_dim..(row + 1) * first.in_dim];
            let mut dot_a = 0.0f32;
            let mut dot_b = 0.0f32;
            let mut dot_c = 0.0f32;
            for col in 0..first.in_dim {
                let value = input_row[col];
                dot_a += value * weight_a[col] as f32;
                dot_b += value * weight_b[col] as f32;
                dot_c += value * weight_c[col] as f32;
            }
            let q_dst = row * out_dim + out_idx;
            let kv_dst = head_offset + row * head_dim + lane;
            unsafe {
                *q_out.get_unchecked_mut(q_dst) = dot_a * scale_a + bias_a;
                *k_head.get_unchecked_mut(kv_dst) = dot_b * scale_b + bias_b;
                *v_head.get_unchecked_mut(kv_dst) = dot_c * scale_c + bias_c;
            }
            row += 1;
        }
        out_idx += 1;
    }
    Ok((q_out, k_head, v_head))
}

fn q4_linear3_i8act_f32(
    first: &Q4LinearHandle,
    second: &Q4LinearHandle,
    third: &Q4LinearHandle,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>, JsValue> {
    if first.in_dim != second.in_dim || first.in_dim != third.in_dim {
        return Err(JsValue::from_str("q4_linear3_i8act_f32 input dimensions do not match"));
    }
    if first.out_dim != second.out_dim || first.out_dim != third.out_dim {
        return Err(JsValue::from_str("q4_linear3_i8act_f32 output dimensions do not match"));
    }
    let out_dim = first.out_dim;
    let mut out = vec![0.0f32; rows * out_dim * 3];
    let part = rows * out_dim;
    if F5_USE_GROUPED_I8ACT_Q4_LINEAR {
        let (quantized, act_scales, groups) = quantize_rows_i8_grouped(input, rows, first.in_dim, F5_GROUPED_I8ACT_GROUP)?;
        first.run_grouped_i8act_into(&quantized, &act_scales, rows, groups, F5_GROUPED_I8ACT_GROUP, &mut out[0..part], out_dim, 0)?;
        second.run_grouped_i8act_into(&quantized, &act_scales, rows, groups, F5_GROUPED_I8ACT_GROUP, &mut out[part..part * 2], out_dim, 0)?;
        third.run_grouped_i8act_into(&quantized, &act_scales, rows, groups, F5_GROUPED_I8ACT_GROUP, &mut out[part * 2..part * 3], out_dim, 0)?;
    } else {
        let (quantized, act_scales) = quantize_rows_i8(input, rows, first.in_dim)?;
        first.run_quantized_i8act_into(&quantized, &act_scales, rows, &mut out[0..part], out_dim, 0)?;
        second.run_quantized_i8act_into(&quantized, &act_scales, rows, &mut out[part..part * 2], out_dim, 0)?;
        third.run_quantized_i8act_into(&quantized, &act_scales, rows, &mut out[part * 2..part * 3], out_dim, 0)?;
    }
    Ok(out)
}

fn q4_linear3_q4act_f32(
    first: &Q4LinearHandle,
    second: &Q4LinearHandle,
    third: &Q4LinearHandle,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>, JsValue> {
    if first.in_dim != second.in_dim || first.in_dim != third.in_dim {
        return Err(JsValue::from_str("q4_linear3_q4act_f32 input dimensions do not match"));
    }
    if first.out_dim != second.out_dim || first.out_dim != third.out_dim {
        return Err(JsValue::from_str("q4_linear3_q4act_f32 output dimensions do not match"));
    }
    let a = first.run_q4act_impl(input, rows)?;
    let b = second.run_q4act_impl(input, rows)?;
    let c = third.run_q4act_impl(input, rows)?;
    let mut out = Vec::with_capacity(a.len() + b.len() + c.len());
    out.extend_from_slice(&a);
    out.extend_from_slice(&b);
    out.extend_from_slice(&c);
    Ok(out)
}

#[wasm_bindgen]
pub fn q4_mlp_f32(
    first: &Q4LinearHandle,
    second: &Q4LinearHandle,
    input: &[f32],
    rows: usize,
    activation: &str,
) -> Result<Vec<f32>, JsValue> {
    if first.out_dim != second.in_dim {
        return Err(JsValue::from_str("q4_mlp_f32 linear dimensions do not match"));
    }
    let mut hidden = first.run_impl(input, rows)?;
    let use_gelu = activation.eq_ignore_ascii_case("gelu");
    for value in &mut hidden {
        *value = if use_gelu { gelu_scalar(*value) } else { silu_scalar(*value) };
    }
    second.run_impl(&hidden, rows)
}

fn q4_mlp_gelu_f32(
    first: &Q4LinearHandle,
    second: &Q4LinearHandle,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>, JsValue> {
    if first.out_dim != second.in_dim {
        return Err(JsValue::from_str("q4_mlp_gelu_f32 linear dimensions do not match"));
    }
    let mut hidden = first.run_impl(input, rows)?;
    for value in &mut hidden {
        *value = gelu_scalar(*value);
    }
    second.run_impl(&hidden, rows)
}

fn q4_mlp_i8act_f32(
    first: &Q4LinearHandle,
    second: &Q4LinearHandle,
    input: &[f32],
    rows: usize,
    activation: &str,
) -> Result<Vec<f32>, JsValue> {
    if first.out_dim != second.in_dim {
        return Err(JsValue::from_str("q4_mlp_i8act_f32 linear dimensions do not match"));
    }
    let mut hidden = first.run_i8act_impl(input, rows)?;
    let use_gelu = activation.eq_ignore_ascii_case("gelu");
    for value in &mut hidden {
        *value = if use_gelu { gelu_scalar(*value) } else { silu_scalar(*value) };
    }
    second.run_i8act_impl(&hidden, rows)
}

fn q4_mlp_q4act_f32(
    first: &Q4LinearHandle,
    second: &Q4LinearHandle,
    input: &[f32],
    rows: usize,
    activation: &str,
) -> Result<Vec<f32>, JsValue> {
    if first.out_dim != second.in_dim {
        return Err(JsValue::from_str("q4_mlp_q4act_f32 linear dimensions do not match"));
    }
    let mut hidden = first.run_q4act_impl(input, rows)?;
    let use_gelu = activation.eq_ignore_ascii_case("gelu");
    for value in &mut hidden {
        *value = if use_gelu { gelu_scalar(*value) } else { silu_scalar(*value) };
    }
    second.run_q4act_impl(&hidden, rows)
}

#[wasm_bindgen]
pub fn f5_dit_block_f32(
    attn_norm: &Q4LinearHandle,
    to_q: &Q4LinearHandle,
    to_k: &Q4LinearHandle,
    to_v: &Q4LinearHandle,
    to_out: &Q4LinearHandle,
    ff_in: &Q4LinearHandle,
    ff_out: &Q4LinearHandle,
    input: &[f32],
    time_embedding: &[f32],
    seq_len: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>, JsValue> {
    if seq_len == 0 || dim == 0 || heads == 0 || head_dim == 0 {
        return Ok(Vec::new());
    }
    if input.len() != seq_len * dim || heads * head_dim != dim {
        return Err(JsValue::from_str("f5_dit_block_f32 input shape mismatch"));
    }
    let mut t = time_embedding.to_vec();
    for value in &mut t {
        *value = silu_scalar(*value);
    }
    f5_dit_block_silu_t(
        attn_norm,
        to_q,
        to_k,
        to_v,
        to_out,
        ff_in,
        ff_out,
        input,
        &t,
        seq_len,
        dim,
        heads,
        head_dim,
        eps,
    )
}

fn f5_dit_block_silu_t(
    attn_norm: &Q4LinearHandle,
    to_q: &Q4LinearHandle,
    to_k: &Q4LinearHandle,
    to_v: &Q4LinearHandle,
    to_out: &Q4LinearHandle,
    ff_in: &Q4LinearHandle,
    ff_out: &Q4LinearHandle,
    input: &[f32],
    silu_time_embedding: &[f32],
    seq_len: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>, JsValue> {
    let modulation = if F5_USE_Q4ACT_Q4_LINEAR {
        attn_norm.run_q4act_impl(silu_time_embedding, 1)?
    } else if f5_use_i8act_q4_linear() {
        attn_norm.run_i8act_impl(silu_time_embedding, 1)?
    } else {
        attn_norm.run_impl(silu_time_embedding, 1)?
    };
    if modulation.len() < dim * 6 {
        return Err(JsValue::from_str("f5_dit_block_f32 modulation shape mismatch"));
    }
    let shift_msa = &modulation[0..dim];
    let scale_msa = &modulation[dim..dim * 2];
    let gate_msa = &modulation[dim * 2..dim * 3];
    let shift_mlp = &modulation[dim * 3..dim * 4];
    let scale_mlp = &modulation[dim * 4..dim * 5];
    let gate_mlp = &modulation[dim * 5..dim * 6];

    let mut norm = vec![0.0f32; seq_len * dim];
    layer_norm_affine_into(input, shift_msa, scale_msa, seq_len, dim, eps, &mut norm)?;
    let mut qkv = if F5_USE_Q4ACT_Q4_LINEAR {
        q4_linear3_q4act_f32(to_q, to_k, to_v, &norm, seq_len)?
    } else if f5_use_i8act_q4_linear() {
        q4_linear3_i8act_f32(to_q, to_k, to_v, &norm, seq_len)?
    } else {
        q4_linear3_f32(to_q, to_k, to_v, &norm, seq_len)?
    };
    let part = seq_len * dim;
    {
        let (q, rest) = qkv.split_at_mut(part);
        let (k, _) = rest.split_at_mut(part);
        apply_rotary_f5(q, k, seq_len, heads, head_dim)?;
    }
    let attn = attention_impl_kv_head_major(&qkv[0..part], &qkv[part..part * 2], &qkv[part * 2..part * 3], seq_len, seq_len, heads, head_dim, false, 0)?;
    let attn = if F5_USE_Q4ACT_Q4_LINEAR {
        to_out.run_q4act_impl(&attn, seq_len)?
    } else if f5_use_i8act_q4_linear() {
        to_out.run_i8act_impl(&attn, seq_len)?
    } else {
        to_out.run_impl(&attn, seq_len)?
    };
    let mut x = vec![0.0f32; seq_len * dim];
    gated_add_rows_into(input, &attn, gate_msa, seq_len, dim, &mut x)?;

    layer_norm_affine_into(&x, shift_mlp, scale_mlp, seq_len, dim, eps, &mut norm)?;
    let ff = if F5_USE_Q4ACT_Q4_LINEAR {
        q4_mlp_q4act_f32(ff_in, ff_out, &norm, seq_len, "gelu")?
    } else if f5_use_i8act_q4_linear() {
        q4_mlp_i8act_f32(ff_in, ff_out, &norm, seq_len, "gelu")?
    } else {
        q4_mlp_gelu_f32(ff_in, ff_out, &norm, seq_len)?
    };
    let mut output = vec![0.0f32; seq_len * dim];
    gated_add_rows_into(&x, &ff, gate_mlp, seq_len, dim, &mut output)?;
    Ok(output)
}

fn f5_dit_block_silu_t_cached_rotary(
    attn_norm: &Q4LinearHandle,
    to_q: &Q4LinearHandle,
    to_k: &Q4LinearHandle,
    to_v: &Q4LinearHandle,
    to_out: &Q4LinearHandle,
    ff_in: &Q4LinearHandle,
    ff_out: &Q4LinearHandle,
    input: &[f32],
    silu_time_embedding: &[f32],
    seq_len: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    rotary_cos: &[f32],
    rotary_sin: &[f32],
) -> Result<Vec<f32>, JsValue> {
    let modulation = if F5_USE_Q4ACT_Q4_LINEAR {
        attn_norm.run_q4act_impl(silu_time_embedding, 1)?
    } else if f5_use_i8act_q4_linear() {
        attn_norm.run_i8act_impl(silu_time_embedding, 1)?
    } else {
        attn_norm.run_impl(silu_time_embedding, 1)?
    };
    if modulation.len() < dim * 6 {
        return Err(JsValue::from_str("f5_dit_block_f32 modulation shape mismatch"));
    }
    let shift_msa = &modulation[0..dim];
    let scale_msa = &modulation[dim..dim * 2];
    let gate_msa = &modulation[dim * 2..dim * 3];
    let shift_mlp = &modulation[dim * 3..dim * 4];
    let scale_mlp = &modulation[dim * 4..dim * 5];
    let gate_mlp = &modulation[dim * 5..dim * 6];

    let mut norm = vec![0.0f32; seq_len * dim];
    layer_norm_affine_into(input, shift_msa, scale_msa, seq_len, dim, eps, &mut norm)?;
    let attn = if F5_USE_Q4ACT_Q4_LINEAR {
        let mut qkv = q4_linear3_q4act_f32(to_q, to_k, to_v, &norm, seq_len)?;
        let part = seq_len * dim;
        {
            let (q, rest) = qkv.split_at_mut(part);
            let (k, _) = rest.split_at_mut(part);
            apply_rotary_f5_cached(q, k, seq_len, heads, head_dim, rotary_cos, rotary_sin)?;
        }
        attention_impl_kv_head_major(&qkv[0..part], &qkv[part..part * 2], &qkv[part * 2..part * 3], seq_len, seq_len, heads, head_dim, false, 0)?
    } else if f5_use_i8act_q4_linear() {
        let mut qkv = q4_linear3_i8act_f32(to_q, to_k, to_v, &norm, seq_len)?;
        let part = seq_len * dim;
        {
            let (q, rest) = qkv.split_at_mut(part);
            let (k, _) = rest.split_at_mut(part);
            apply_rotary_f5_cached(q, k, seq_len, heads, head_dim, rotary_cos, rotary_sin)?;
        }
        attention_impl_kv_head_major(&qkv[0..part], &qkv[part..part * 2], &qkv[part * 2..part * 3], seq_len, seq_len, heads, head_dim, false, 0)?
    } else {
        let (mut q, mut k_head, v_head) = q4_linear3_f32_q_row_kv_head_major(to_q, to_k, to_v, &norm, seq_len, heads, head_dim)?;
        apply_rotary_f5_cached_q_row_k_head(&mut q, &mut k_head, seq_len, heads, head_dim, rotary_cos, rotary_sin)?;
        attention_impl_kv_already_head_major(&q, &k_head, &v_head, seq_len, seq_len, heads, head_dim, false, 0)?
    };
    let attn = if F5_USE_Q4ACT_Q4_LINEAR {
        to_out.run_q4act_impl(&attn, seq_len)?
    } else if f5_use_i8act_q4_linear() {
        to_out.run_i8act_impl(&attn, seq_len)?
    } else {
        to_out.run_impl(&attn, seq_len)?
    };
    let mut x = vec![0.0f32; seq_len * dim];
    gated_add_rows_into(input, &attn, gate_msa, seq_len, dim, &mut x)?;

    layer_norm_affine_into(&x, shift_mlp, scale_mlp, seq_len, dim, eps, &mut norm)?;
    let ff = if F5_USE_Q4ACT_Q4_LINEAR {
        q4_mlp_q4act_f32(ff_in, ff_out, &norm, seq_len, "gelu")?
    } else if f5_use_i8act_q4_linear() {
        q4_mlp_i8act_f32(ff_in, ff_out, &norm, seq_len, "gelu")?
    } else {
        q4_mlp_gelu_f32(ff_in, ff_out, &norm, seq_len)?
    };
    let mut output = vec![0.0f32; seq_len * dim];
    gated_add_rows_into(&x, &ff, gate_mlp, seq_len, dim, &mut output)?;
    Ok(output)
}

fn f5_dit_block_modulation_cached_rotary(
    to_q: &Q4LinearHandle,
    to_k: &Q4LinearHandle,
    to_v: &Q4LinearHandle,
    to_out: &Q4LinearHandle,
    ff_in: &Q4LinearHandle,
    ff_out: &Q4LinearHandle,
    input: &[f32],
    modulation: &[f32],
    seq_len: usize,
    dim: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    rotary_cos: &[f32],
    rotary_sin: &[f32],
) -> Result<Vec<f32>, JsValue> {
    if modulation.len() < dim * 6 {
        return Err(JsValue::from_str("f5_dit_block_f32 modulation shape mismatch"));
    }
    let shift_msa = &modulation[0..dim];
    let scale_msa = &modulation[dim..dim * 2];
    let gate_msa = &modulation[dim * 2..dim * 3];
    let shift_mlp = &modulation[dim * 3..dim * 4];
    let scale_mlp = &modulation[dim * 4..dim * 5];
    let gate_mlp = &modulation[dim * 5..dim * 6];

    let mut norm = vec![0.0f32; seq_len * dim];
    layer_norm_affine_into(input, shift_msa, scale_msa, seq_len, dim, eps, &mut norm)?;
    let attn = if F5_USE_Q4ACT_Q4_LINEAR {
        let mut qkv = q4_linear3_q4act_f32(to_q, to_k, to_v, &norm, seq_len)?;
        let part = seq_len * dim;
        {
            let (q, rest) = qkv.split_at_mut(part);
            let (k, _) = rest.split_at_mut(part);
            apply_rotary_f5_cached(q, k, seq_len, heads, head_dim, rotary_cos, rotary_sin)?;
        }
        attention_impl_kv_head_major(&qkv[0..part], &qkv[part..part * 2], &qkv[part * 2..part * 3], seq_len, seq_len, heads, head_dim, false, 0)?
    } else if f5_use_i8act_q4_linear() {
        let mut qkv = q4_linear3_i8act_f32(to_q, to_k, to_v, &norm, seq_len)?;
        let part = seq_len * dim;
        {
            let (q, rest) = qkv.split_at_mut(part);
            let (k, _) = rest.split_at_mut(part);
            apply_rotary_f5_cached(q, k, seq_len, heads, head_dim, rotary_cos, rotary_sin)?;
        }
        attention_impl_kv_head_major(&qkv[0..part], &qkv[part..part * 2], &qkv[part * 2..part * 3], seq_len, seq_len, heads, head_dim, false, 0)?
    } else {
        let (mut q, mut k_head, v_head) = q4_linear3_f32_q_row_kv_head_major(to_q, to_k, to_v, &norm, seq_len, heads, head_dim)?;
        apply_rotary_f5_cached_q_row_k_head(&mut q, &mut k_head, seq_len, heads, head_dim, rotary_cos, rotary_sin)?;
        attention_impl_kv_already_head_major(&q, &k_head, &v_head, seq_len, seq_len, heads, head_dim, false, 0)?
    };
    let attn = if F5_USE_Q4ACT_Q4_LINEAR {
        to_out.run_q4act_impl(&attn, seq_len)?
    } else if f5_use_i8act_q4_linear() {
        to_out.run_i8act_impl(&attn, seq_len)?
    } else {
        to_out.run_impl(&attn, seq_len)?
    };
    let mut x = vec![0.0f32; seq_len * dim];
    gated_add_rows_into(input, &attn, gate_msa, seq_len, dim, &mut x)?;

    layer_norm_affine_into(&x, shift_mlp, scale_mlp, seq_len, dim, eps, &mut norm)?;
    let ff = if F5_USE_Q4ACT_Q4_LINEAR {
        q4_mlp_q4act_f32(ff_in, ff_out, &norm, seq_len, "gelu")?
    } else if f5_use_i8act_q4_linear() {
        q4_mlp_i8act_f32(ff_in, ff_out, &norm, seq_len, "gelu")?
    } else {
        q4_mlp_gelu_f32(ff_in, ff_out, &norm, seq_len)?
    };
    let mut output = vec![0.0f32; seq_len * dim];
    gated_add_rows_into(&x, &ff, gate_mlp, seq_len, dim, &mut output)?;
    Ok(output)
}

fn make_rotary_f5_cache(seq_len: usize, head_dim: usize) -> Result<(Vec<f32>, Vec<f32>), JsValue> {
    if head_dim % 2 != 0 {
        return Err(JsValue::from_str("make_rotary_f5_cache head_dim must be even"));
    }
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq_len * half];
    let mut sin = vec![0.0f32; seq_len * half];
    for pos in 0..seq_len {
        for idx in 0..half {
            let angle = pos as f32 / 10000.0f32.powf((2 * idx) as f32 / head_dim as f32);
            let offset = pos * half + idx;
            cos[offset] = angle.cos();
            sin[offset] = angle.sin();
        }
    }
    Ok((cos, sin))
}

fn apply_rotary_f5(
    q: &mut [f32],
    k: &mut [f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(), JsValue> {
    let dim = heads * head_dim;
    if q.len() < seq_len * dim || k.len() < seq_len * dim || head_dim % 2 != 0 {
        return Err(JsValue::from_str("apply_rotary_f5 input shape mismatch"));
    }
    let half = head_dim / 2;
    for pos in 0..seq_len {
        let base = pos * dim;
        for idx in 0..half {
            let angle = pos as f32 / 10000.0f32.powf((2 * idx) as f32 / head_dim as f32);
            let c = angle.cos();
            let s = angle.sin();
            let left = base + idx * 2;
            let right = left + 1;
            rotate_pair(q, left, right, c, s);
            rotate_pair(k, left, right, c, s);
        }
    }
    Ok(())
}

fn apply_rotary_f5_cached(
    q: &mut [f32],
    k: &mut [f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), JsValue> {
    let dim = heads * head_dim;
    if q.len() < seq_len * dim || k.len() < seq_len * dim || head_dim % 2 != 0 {
        return Err(JsValue::from_str("apply_rotary_f5_cached input shape mismatch"));
    }
    let half = head_dim / 2;
    if cos.len() < seq_len * half || sin.len() < seq_len * half {
        return Err(JsValue::from_str("apply_rotary_f5_cached cache shape mismatch"));
    }
    for pos in 0..seq_len {
        let base = pos * dim;
        let cache_base = pos * half;
        for idx in 0..half {
            let c = cos[cache_base + idx];
            let s = sin[cache_base + idx];
            let left = base + idx * 2;
            let right = left + 1;
            rotate_pair(q, left, right, c, s);
            rotate_pair(k, left, right, c, s);
        }
    }
    Ok(())
}

fn apply_rotary_f5_cached_q_row_k_head(
    q: &mut [f32],
    k_head: &mut [f32],
    seq_len: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), JsValue> {
    let dim = heads * head_dim;
    if q.len() < seq_len * dim || k_head.len() < seq_len * dim || head_dim % 2 != 0 {
        return Err(JsValue::from_str("apply_rotary_f5_cached_q_row_k_head input shape mismatch"));
    }
    let half = head_dim / 2;
    if cos.len() < seq_len * half || sin.len() < seq_len * half {
        return Err(JsValue::from_str("apply_rotary_f5_cached_q_row_k_head cache shape mismatch"));
    }
    for pos in 0..seq_len {
        let q_base = pos * dim;
        let k_base = pos * head_dim;
        let cache_base = pos * half;
        for idx in 0..half {
            let c = cos[cache_base + idx];
            let s = sin[cache_base + idx];
            let q_left = q_base + idx * 2;
            let q_right = q_left + 1;
            let k_left = k_base + idx * 2;
            let k_right = k_left + 1;
            rotate_pair(q, q_left, q_right, c, s);
            rotate_pair(k_head, k_left, k_right, c, s);
        }
    }
    Ok(())
}

fn rotate_pair(values: &mut [f32], left: usize, right: usize, c: f32, s: f32) {
    let a = values[left];
    let b = values[right];
    values[left] = a * c - b * s;
    values[right] = b * c + a * s;
}

#[wasm_bindgen]
pub struct F5Q4DiTSession {
    q4: HashMap<String, Q4LinearHandle>,
    q4_raw: HashMap<String, Q4RawTensor>,
    dense: HashMap<String, Vec<f32>>,
    dim: usize,
    text_dim: usize,
    mel_dim: usize,
    heads: usize,
    head_dim: usize,
    depth: usize,
}

#[wasm_bindgen]
impl F5Q4DiTSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> F5Q4DiTSession {
        F5Q4DiTSession {
            q4: HashMap::new(),
            q4_raw: HashMap::new(),
            dense: HashMap::new(),
            dim: 1024,
            text_dim: 512,
            mel_dim: 100,
            heads: 16,
            head_dim: 64,
            depth: 22,
        }
    }

    pub fn add_q4_tensor(
        &mut self,
        name: &str,
        packed_weight: &[u8],
        row_scales_f16: &[u16],
        bias_values: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), JsValue> {
        let is_conv = name.contains(".conv") || name.contains("dwconv");
        let has_padded_rows = packed_weight.len() >= out_dim * in_dim.div_ceil(2);
        if is_conv || !has_padded_rows {
            self.q4_raw.insert(name.to_string(), Q4RawTensor::new(packed_weight, row_scales_f16, bias_values, in_dim, out_dim)?);
        }
        if !is_conv && has_padded_rows {
            let handle = Q4LinearHandle::new(packed_weight, row_scales_f16, bias_values, in_dim, out_dim)?;
            self.q4.insert(name.to_string(), handle);
        }
        Ok(())
    }

    pub fn add_dense_f32(&mut self, name: &str, values: &[f32]) {
        self.dense.insert(name.to_string(), values.to_vec());
    }

    pub fn forward(
        &self,
        x: &[f32],
        cond: &[f32],
        text_ids: &[i32],
        time: f32,
        drop_audio_cond: bool,
        drop_text: bool,
    ) -> Result<Vec<f32>, JsValue> {
        self.forward_impl(x, cond, text_ids, time, drop_audio_cond, drop_text)
    }

    pub fn debug_time_embedding(&self, time: f32) -> Result<Vec<f32>, JsValue> {
        self.time_embedding(time)
    }

    pub fn debug_text_embedding(&self, text_ids: &[i32], seq_len: usize, drop_text: bool) -> Result<Vec<f32>, JsValue> {
        self.text_embedding(text_ids, seq_len, drop_text)
    }

    pub fn debug_input_embedding(
        &self,
        x: &[f32],
        cond: &[f32],
        text: &[f32],
        seq_len: usize,
        drop_audio_cond: bool,
    ) -> Result<Vec<f32>, JsValue> {
        self.input_embedding(x, cond, text, seq_len, drop_audio_cond)
    }

    pub fn debug_input_embedding_profile_json(
        &self,
        x: &[f32],
        cond: &[f32],
        text: &[f32],
        seq_len: usize,
        drop_audio_cond: bool,
    ) -> Result<String, JsValue> {
        if x.len() != seq_len * self.mel_dim || cond.len() != x.len() || text.len() != seq_len * self.text_dim {
            return Err(JsValue::from_str("F5 debug input embedding profile shape mismatch"));
        }
        let mut marks: Vec<(&str, f64)> = Vec::new();
        let started = date_now();

        let joined_dim = self.mel_dim * 2 + self.text_dim;
        let mut joined = vec![0.0f32; seq_len * joined_dim];
        for row in 0..seq_len {
            let dst = row * joined_dim;
            joined[dst..dst + self.mel_dim].copy_from_slice(&x[row * self.mel_dim..(row + 1) * self.mel_dim]);
            if !drop_audio_cond {
                joined[dst + self.mel_dim..dst + self.mel_dim * 2].copy_from_slice(&cond[row * self.mel_dim..(row + 1) * self.mel_dim]);
            }
            joined[dst + self.mel_dim * 2..dst + joined_dim].copy_from_slice(&text[row * self.text_dim..(row + 1) * self.text_dim]);
        }
        marks.push(("join", date_now()));

        let projected = self.linear("transformer.input_embed.proj.weight", &joined, seq_len)?;
        marks.push(("projection", date_now()));

        let mut pos = self.grouped_conv1d("transformer.input_embed.conv_pos_embed.conv1d.0.weight", &projected, seq_len, self.dim, 31, 15, 16)?;
        marks.push(("conv0", date_now()));
        for value in &mut pos {
            *value = mish_scalar(*value);
        }
        marks.push(("mish0", date_now()));

        pos = self.grouped_conv1d("transformer.input_embed.conv_pos_embed.conv1d.2.weight", &pos, seq_len, self.dim, 31, 15, 16)?;
        marks.push(("conv2", date_now()));
        for idx in 0..pos.len() {
            pos[idx] = mish_scalar(pos[idx]) + projected[idx];
        }
        marks.push(("mish2_residual", date_now()));

        let mut previous = started;
        let mut fields = Vec::new();
        for (name, at) in marks {
            fields.push(format!("\"{}\":{:.3}", name, at - previous));
            previous = at;
        }
        let checksum: f64 = pos.iter().take(4096).map(|value| *value as f64).sum();
        Ok(format!(
            "{{\"seqLen\":{},\"totalMs\":{:.3},\"checksum\":{:.6},\"timings\":{{{}}}}}",
            seq_len,
            previous - started,
            checksum,
            fields.join(",")
        ))
    }

    pub fn debug_dit_block(
        &self,
        block: usize,
        input: &[f32],
        time_embedding: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let (rotary_cos, rotary_sin) = make_rotary_f5_cache(seq_len, self.head_dim)?;
        self.dit_block_cached_rotary(block, input, time_embedding, seq_len, &rotary_cos, &rotary_sin)
    }

    pub fn debug_dit_block_profile_json(
        &self,
        block: usize,
        input: &[f32],
        time_embedding: &[f32],
        seq_len: usize,
    ) -> Result<String, JsValue> {
        if input.len() != seq_len * self.dim {
            return Err(JsValue::from_str("F5 debug block profile input shape mismatch"));
        }
        let prefix = format!("transformer.transformer_blocks.{}", block);
        let attn_norm = self.q4(&format!("{}.attn_norm.linear.weight", prefix))?;
        let to_q = self.q4(&format!("{}.attn.to_q.weight", prefix))?;
        let to_k = self.q4(&format!("{}.attn.to_k.weight", prefix))?;
        let to_v = self.q4(&format!("{}.attn.to_v.weight", prefix))?;
        let to_out = self.q4(&format!("{}.attn.to_out.0.weight", prefix))?;
        let ff_in = self.q4(&format!("{}.ff.ff.0.0.weight", prefix))?;
        let ff_out = self.q4(&format!("{}.ff.ff.2.weight", prefix))?;
        let mut marks: Vec<(&str, f64)> = Vec::new();
        let started = date_now();

        let (rotary_cos, rotary_sin) = make_rotary_f5_cache(seq_len, self.head_dim)?;
        marks.push(("rotary_cache", date_now()));

        let modulation = attn_norm.run_impl(time_embedding, 1)?;
        if modulation.len() < self.dim * 6 {
            return Err(JsValue::from_str("F5 debug block profile modulation shape mismatch"));
        }
        let shift_msa = &modulation[0..self.dim];
        let scale_msa = &modulation[self.dim..self.dim * 2];
        let gate_msa = &modulation[self.dim * 2..self.dim * 3];
        let shift_mlp = &modulation[self.dim * 3..self.dim * 4];
        let scale_mlp = &modulation[self.dim * 4..self.dim * 5];
        let gate_mlp = &modulation[self.dim * 5..self.dim * 6];
        marks.push(("modulation", date_now()));

        let mut norm = vec![0.0f32; seq_len * self.dim];
        layer_norm_affine_into(input, shift_msa, scale_msa, seq_len, self.dim, 1e-6, &mut norm)?;
        marks.push(("norm_msa", date_now()));

        let (mut q, mut k_head, v_head) = q4_linear3_f32_q_row_kv_head_major(to_q, to_k, to_v, &norm, seq_len, self.heads, self.head_dim)?;
        marks.push(("qkv", date_now()));

        apply_rotary_f5_cached_q_row_k_head(&mut q, &mut k_head, seq_len, self.heads, self.head_dim, &rotary_cos, &rotary_sin)?;
        marks.push(("rotary_apply", date_now()));

        let attn = attention_impl_kv_already_head_major(
            &q,
            &k_head,
            &v_head,
            seq_len,
            seq_len,
            self.heads,
            self.head_dim,
            false,
            0,
        )?;
        marks.push(("attention", date_now()));

        let attn = to_out.run_impl(&attn, seq_len)?;
        marks.push(("attn_out", date_now()));

        let mut x = vec![0.0f32; seq_len * self.dim];
        gated_add_rows_into(input, &attn, gate_msa, seq_len, self.dim, &mut x)?;
        marks.push(("gate_msa", date_now()));

        layer_norm_affine_into(&x, shift_mlp, scale_mlp, seq_len, self.dim, 1e-6, &mut norm)?;
        marks.push(("norm_mlp", date_now()));

        let mut hidden = ff_in.run_impl(&norm, seq_len)?;
        marks.push(("ff_in", date_now()));
        for value in &mut hidden {
            *value = gelu_scalar(*value);
        }
        marks.push(("gelu", date_now()));
        let ff = ff_out.run_impl(&hidden, seq_len)?;
        marks.push(("ff_out", date_now()));

        let mut output = vec![0.0f32; seq_len * self.dim];
        gated_add_rows_into(&x, &ff, gate_mlp, seq_len, self.dim, &mut output)?;
        marks.push(("gate_mlp", date_now()));

        let mut previous = started;
        let mut fields = Vec::new();
        for (name, at) in marks {
            fields.push(format!("\"{}\":{:.3}", name, at - previous));
            previous = at;
        }
        let checksum: f64 = output.iter().take(4096).map(|value| *value as f64).sum();
        Ok(format!(
            "{{\"block\":{},\"seqLen\":{},\"totalMs\":{:.3},\"checksum\":{:.6},\"timings\":{{{}}}}}",
            block,
            seq_len,
            previous - started,
            checksum,
            fields.join(",")
        ))
    }

    pub fn debug_final_ada_norm(&self, input: &[f32], time_embedding: &[f32], seq_len: usize) -> Result<Vec<f32>, JsValue> {
        self.final_ada_norm(input, time_embedding, seq_len)
    }

    pub fn sample_mel(
        &self,
        cond_mel: &[f32],
        cond_seq_len: usize,
        text_ids: &[i32],
        duration: usize,
        steps: usize,
        cfg_strength: f32,
        sway_sampling_coef: f32,
        seed: u32,
    ) -> Result<Vec<f32>, JsValue> {
        self.sample_mel_impl(cond_mel, cond_seq_len, text_ids, duration, steps, cfg_strength, sway_sampling_coef, seed, None)
    }

    pub fn sample_mel_with_progress(
        &self,
        cond_mel: &[f32],
        cond_seq_len: usize,
        text_ids: &[i32],
        duration: usize,
        steps: usize,
        cfg_strength: f32,
        sway_sampling_coef: f32,
        seed: u32,
        progress: &js_sys::Function,
    ) -> Result<Vec<f32>, JsValue> {
        self.sample_mel_impl(cond_mel, cond_seq_len, text_ids, duration, steps, cfg_strength, sway_sampling_coef, seed, Some(progress))
    }

    fn sample_mel_impl(
        &self,
        cond_mel: &[f32],
        cond_seq_len: usize,
        text_ids: &[i32],
        duration: usize,
        steps: usize,
        cfg_strength: f32,
        sway_sampling_coef: f32,
        seed: u32,
        progress: Option<&js_sys::Function>,
    ) -> Result<Vec<f32>, JsValue> {
        if duration < cond_seq_len || cond_mel.len() != cond_seq_len * self.mel_dim {
            return Err(JsValue::from_str("F5Q4DiTSession sample_mel shape mismatch"));
        }
        let mut cond = vec![0.0f32; duration * self.mel_dim];
        cond[..cond_mel.len()].copy_from_slice(cond_mel);
        let mut y = gaussian_vec(duration * self.mel_dim, seed);
        let times = make_f5_time_grid(steps, sway_sampling_coef);
        let text = self.text_embedding(text_ids, duration, false)?;
        let null_text = if cfg_strength.abs() >= 1e-5 {
            Some(self.text_embedding(text_ids, duration, true)?)
        } else {
            None
        };
        let projected_base = self.input_projection_base(&cond, &text, duration, false)?;
        let null_projected_base = if let Some(null_text) = null_text.as_ref() {
            Some(self.input_projection_base(&cond, null_text, duration, true)?)
        } else {
            None
        };
        let (rotary_cos, rotary_sin) = make_rotary_f5_cache(duration, self.head_dim)?;
        for step in 0..steps {
            let t = times[step];
            let dt = times[step + 1] - times[step];
            let t_emb = self.time_embedding(t)?;
            let mut t_silu = t_emb.clone();
            for value in &mut t_silu {
                *value = silu_scalar(*value);
            }
            let block_modulations = self.block_modulations(&t_silu)?;
            let final_modulation = self.linear("transformer.norm_out.linear.weight", &t_silu, 1)?;
            let x_projection = self.input_x_projection(&y, duration)?;
            if let Some(null_projected_base) = null_projected_base.as_ref() {
                let pred = self.forward_impl_with_projected_parts_cached_rotary_modulated(
                    &x_projection,
                    &projected_base,
                    &block_modulations,
                    &final_modulation,
                    &rotary_cos,
                    &rotary_sin,
                )?;
                let null_pred = self.forward_impl_with_projected_parts_cached_rotary_modulated(
                    &x_projection,
                    null_projected_base,
                    &block_modulations,
                    &final_modulation,
                    &rotary_cos,
                    &rotary_sin,
                )?;
                for idx in 0..y.len() {
                    let flow = pred[idx] + (pred[idx] - null_pred[idx]) * cfg_strength;
                    y[idx] += dt * flow;
                }
            } else {
                let pred = self.forward_impl_with_projected_parts_cached_rotary_modulated(
                    &x_projection,
                    &projected_base,
                    &block_modulations,
                    &final_modulation,
                    &rotary_cos,
                    &rotary_sin,
                )?;
                for idx in 0..y.len() {
                    y[idx] += dt * pred[idx];
                }
            }
            if let Some(progress) = progress {
                progress.call3(
                    &JsValue::NULL,
                    &JsValue::from_f64((step + 1) as f64),
                    &JsValue::from_f64(steps as f64),
                    &JsValue::from_f64(t as f64),
                )?;
            }
        }
        y[..cond_seq_len * self.mel_dim].copy_from_slice(&cond[..cond_seq_len * self.mel_dim]);
        Ok(y)
    }
}

impl F5Q4DiTSession {
    fn q4(&self, name: &str) -> Result<&Q4LinearHandle, JsValue> {
        self.q4.get(name).ok_or_else(|| JsValue::from_str(&format!("F5 q4 tensor not found: {}", name)))
    }

    fn q4_raw(&self, name: &str) -> Result<&Q4RawTensor, JsValue> {
        self.q4_raw.get(name).ok_or_else(|| JsValue::from_str(&format!("F5 raw q4 tensor not found: {}", name)))
    }

    fn dense(&self, name: &str) -> Result<&[f32], JsValue> {
        self.dense.get(name).map(|v| v.as_slice()).ok_or_else(|| JsValue::from_str(&format!("F5 dense tensor not found: {}", name)))
    }

    fn linear(&self, name: &str, input: &[f32], rows: usize) -> Result<Vec<f32>, JsValue> {
        let handle = self.q4(name)?;
        if F5_USE_Q4ACT_Q4_LINEAR {
            handle.run_q4act_impl(input, rows)
        } else if f5_use_i8act_q4_linear() {
            handle.run_i8act_impl(input, rows)
        } else {
            handle.run_impl(input, rows)
        }
    }

    fn forward_impl(
        &self,
        x: &[f32],
        cond: &[f32],
        text_ids: &[i32],
        time: f32,
        drop_audio_cond: bool,
        drop_text: bool,
    ) -> Result<Vec<f32>, JsValue> {
        if x.len() % self.mel_dim != 0 || cond.len() != x.len() {
            return Err(JsValue::from_str("F5Q4DiTSession forward shape mismatch"));
        }
        let seq_len = x.len() / self.mel_dim;
        let t = self.time_embedding(time)?;
        let mut t_silu = t.clone();
        for value in &mut t_silu {
            *value = silu_scalar(*value);
        }
        let text = self.text_embedding(text_ids, seq_len, drop_text)?;
        self.forward_impl_with_text(x, cond, &text, &t_silu, drop_audio_cond)
    }

    fn forward_impl_with_text(
        &self,
        x: &[f32],
        cond: &[f32],
        text: &[f32],
        t_silu: &[f32],
        drop_audio_cond: bool,
    ) -> Result<Vec<f32>, JsValue> {
        if x.len() % self.mel_dim != 0 || cond.len() != x.len() {
            return Err(JsValue::from_str("F5Q4DiTSession forward shape mismatch"));
        }
        let seq_len = x.len() / self.mel_dim;
        if text.len() != seq_len * self.text_dim {
            return Err(JsValue::from_str("F5Q4DiTSession text embedding shape mismatch"));
        }
        let mut hidden = self.input_embedding(x, cond, text, seq_len, drop_audio_cond)?;
        let (rotary_cos, rotary_sin) = make_rotary_f5_cache(seq_len, self.head_dim)?;
        for block in 0..self.depth {
            hidden = self.dit_block_cached_rotary(block, &hidden, &t_silu, seq_len, &rotary_cos, &rotary_sin)?;
        }
        hidden = self.final_ada_norm(&hidden, &t_silu, seq_len)?;
        self.linear("transformer.proj_out.weight", &hidden, seq_len)
    }

    fn forward_impl_with_projected_parts_cached_rotary_modulated(
        &self,
        x_projection: &[f32],
        projected_base: &[f32],
        block_modulations: &[Vec<f32>],
        final_modulation: &[f32],
        rotary_cos: &[f32],
        rotary_sin: &[f32],
    ) -> Result<Vec<f32>, JsValue> {
        if x_projection.len() % self.dim != 0 || projected_base.len() != x_projection.len() || block_modulations.len() != self.depth {
            return Err(JsValue::from_str("F5Q4DiTSession projected-part forward shape mismatch"));
        }
        let seq_len = x_projection.len() / self.dim;
        let mut hidden = self.input_embedding_from_projected_parts(x_projection, projected_base, seq_len)?;
        for (block, modulation) in block_modulations.iter().enumerate() {
            hidden = self.dit_block_cached_rotary_with_modulation(
                block,
                &hidden,
                modulation,
                seq_len,
                rotary_cos,
                rotary_sin,
            )?;
        }
        hidden = self.final_ada_norm_with_modulation(&hidden, final_modulation, seq_len)?;
        self.linear("transformer.proj_out.weight", &hidden, seq_len)
    }

    fn time_embedding(&self, time: f32) -> Result<Vec<f32>, JsValue> {
        let freq_dim = 256usize;
        let half = freq_dim / 2;
        let factor = 10000.0f32.ln() / (half as f32 - 1.0);
        let mut emb = vec![0.0f32; freq_dim];
        for idx in 0..half {
            let value = 1000.0 * time * (idx as f32 * -factor).exp();
            emb[idx] = value.sin();
            emb[idx + half] = value.cos();
        }
        let mut out = self.linear("transformer.time_embed.time_mlp.0.weight", &emb, 1)?;
        for value in &mut out {
            *value = silu_scalar(*value);
        }
        self.linear("transformer.time_embed.time_mlp.2.weight", &out, 1)
    }

    fn text_embedding(&self, text_ids: &[i32], seq_len: usize, drop_text: bool) -> Result<Vec<f32>, JsValue> {
        let weight = self.dense("transformer.text_embed.text_embed.weight")?;
        let vocab = weight.len() / self.text_dim;
        let mut out = vec![0.0f32; seq_len * self.text_dim];
        for pos in 0..seq_len {
            let raw_id = text_ids.get(pos).copied().unwrap_or(-1);
            let token = if drop_text { 0usize } else { (raw_id + 1).max(0) as usize }.min(vocab.saturating_sub(1));
            let src = token * self.text_dim;
            out[pos * self.text_dim..(pos + 1) * self.text_dim].copy_from_slice(&weight[src..src + self.text_dim]);
        }
        add_text_sinusoidal_pos(&mut out, seq_len, self.text_dim);
        let mut hidden = out;
        for block in 0..4 {
            hidden = self.convnext_text_block(block, &hidden, seq_len)?;
        }
        Ok(hidden)
    }

    fn convnext_text_block(&self, block: usize, input: &[f32], seq_len: usize) -> Result<Vec<f32>, JsValue> {
        let prefix = format!("transformer.text_embed.text_blocks.{}", block);
        let mut x = self.depthwise_conv1d(&format!("{}.dwconv.weight", prefix), input, seq_len, self.text_dim, 7, 3)?;
        x = layer_norm_weight_bias(&x, self.dense(&format!("{}.norm.weight", prefix))?, self.dense(&format!("{}.norm.bias", prefix))?, seq_len, self.text_dim, 1e-6)?;
        x = self.linear(&format!("{}.pwconv1.weight", prefix), &x, seq_len)?;
        for value in &mut x {
            *value = gelu_erf_scalar(*value);
        }
        apply_grn(&mut x, seq_len, self.text_dim * 2, self.dense(&format!("{}.grn.gamma", prefix))?, self.dense(&format!("{}.grn.beta", prefix))?);
        x = self.linear(&format!("{}.pwconv2.weight", prefix), &x, seq_len)?;
        for idx in 0..x.len() {
            x[idx] += input[idx];
        }
        Ok(x)
    }

    fn input_embedding(&self, x: &[f32], cond: &[f32], text: &[f32], seq_len: usize, drop_audio_cond: bool) -> Result<Vec<f32>, JsValue> {
        let joined_dim = self.mel_dim * 2 + self.text_dim;
        let mut joined = vec![0.0f32; seq_len * joined_dim];
        for row in 0..seq_len {
            let dst = row * joined_dim;
            joined[dst..dst + self.mel_dim].copy_from_slice(&x[row * self.mel_dim..(row + 1) * self.mel_dim]);
            if !drop_audio_cond {
                joined[dst + self.mel_dim..dst + self.mel_dim * 2].copy_from_slice(&cond[row * self.mel_dim..(row + 1) * self.mel_dim]);
            }
            joined[dst + self.mel_dim * 2..dst + joined_dim].copy_from_slice(&text[row * self.text_dim..(row + 1) * self.text_dim]);
        }
        let projected = self.linear("transformer.input_embed.proj.weight", &joined, seq_len)?;
        let mut pos = self.grouped_conv1d("transformer.input_embed.conv_pos_embed.conv1d.0.weight", &projected, seq_len, self.dim, 31, 15, 16)?;
        for value in &mut pos {
            *value = mish_scalar(*value);
        }
        pos = self.grouped_conv1d("transformer.input_embed.conv_pos_embed.conv1d.2.weight", &pos, seq_len, self.dim, 31, 15, 16)?;
        for idx in 0..pos.len() {
            pos[idx] = mish_scalar(pos[idx]) + projected[idx];
        }
        Ok(pos)
    }

    fn input_projection_base(&self, cond: &[f32], text: &[f32], seq_len: usize, drop_audio_cond: bool) -> Result<Vec<f32>, JsValue> {
        if cond.len() != seq_len * self.mel_dim || text.len() != seq_len * self.text_dim {
            return Err(JsValue::from_str("F5 input projection base shape mismatch"));
        }
        let handle = self.q4("transformer.input_embed.proj.weight")?;
        let mut projected = vec![0.0f32; seq_len * self.dim];
        for row in 0..seq_len {
            let dst = row * self.dim;
            for col in 0..self.dim {
                projected[dst + col] = handle.bias_values.get(col).copied().unwrap_or(0.0);
            }
        }
        if !drop_audio_cond {
            handle.add_columns_into(cond, seq_len, self.mel_dim, self.mel_dim, &mut projected, self.dim, 0)?;
        }
        handle.add_columns_into(text, seq_len, self.mel_dim * 2, self.text_dim, &mut projected, self.dim, 0)?;
        Ok(projected)
    }

    fn input_x_projection(&self, x: &[f32], seq_len: usize) -> Result<Vec<f32>, JsValue> {
        if x.len() != seq_len * self.mel_dim {
            return Err(JsValue::from_str("F5 input x projection shape mismatch"));
        }
        let handle = self.q4("transformer.input_embed.proj.weight")?;
        let mut projected = vec![0.0f32; seq_len * self.dim];
        handle.add_columns_into(x, seq_len, 0, self.mel_dim, &mut projected, self.dim, 0)?;
        Ok(projected)
    }

    fn input_embedding_from_projected_parts(&self, x_projection: &[f32], projected_base: &[f32], seq_len: usize) -> Result<Vec<f32>, JsValue> {
        if x_projection.len() != seq_len * self.dim || projected_base.len() != seq_len * self.dim {
            return Err(JsValue::from_str("F5 input projected parts shape mismatch"));
        }
        let mut projected = projected_base.to_vec();
        for idx in 0..projected.len() {
            projected[idx] += x_projection[idx];
        }
        self.input_embedding_from_projected(&projected, seq_len)
    }

    fn input_embedding_from_projected(&self, projected: &[f32], seq_len: usize) -> Result<Vec<f32>, JsValue> {
        if projected.len() != seq_len * self.dim {
            return Err(JsValue::from_str("F5 input projected shape mismatch"));
        }
        let mut pos = self.grouped_conv1d("transformer.input_embed.conv_pos_embed.conv1d.0.weight", &projected, seq_len, self.dim, 31, 15, 16)?;
        for value in &mut pos {
            *value = mish_scalar(*value);
        }
        pos = self.grouped_conv1d("transformer.input_embed.conv_pos_embed.conv1d.2.weight", &pos, seq_len, self.dim, 31, 15, 16)?;
        for idx in 0..pos.len() {
            pos[idx] = mish_scalar(pos[idx]) + projected[idx];
        }
        Ok(pos)
    }

    fn dit_block_cached_rotary(
        &self,
        block: usize,
        input: &[f32],
        t_silu: &[f32],
        seq_len: usize,
        rotary_cos: &[f32],
        rotary_sin: &[f32],
    ) -> Result<Vec<f32>, JsValue> {
        let prefix = format!("transformer.transformer_blocks.{}", block);
        f5_dit_block_silu_t_cached_rotary(
            self.q4(&format!("{}.attn_norm.linear.weight", prefix))?,
            self.q4(&format!("{}.attn.to_q.weight", prefix))?,
            self.q4(&format!("{}.attn.to_k.weight", prefix))?,
            self.q4(&format!("{}.attn.to_v.weight", prefix))?,
            self.q4(&format!("{}.attn.to_out.0.weight", prefix))?,
            self.q4(&format!("{}.ff.ff.0.0.weight", prefix))?,
            self.q4(&format!("{}.ff.ff.2.weight", prefix))?,
            input,
            t_silu,
            seq_len,
            self.dim,
            self.heads,
            self.head_dim,
            1e-6,
            rotary_cos,
            rotary_sin,
        )
    }

    fn block_modulation(&self, block: usize, t_silu: &[f32]) -> Result<Vec<f32>, JsValue> {
        let prefix = format!("transformer.transformer_blocks.{}", block);
        self.linear(&format!("{}.attn_norm.linear.weight", prefix), t_silu, 1)
    }

    fn block_modulations(&self, t_silu: &[f32]) -> Result<Vec<Vec<f32>>, JsValue> {
        let mut modulations = Vec::with_capacity(self.depth);
        for block in 0..self.depth {
            modulations.push(self.block_modulation(block, t_silu)?);
        }
        Ok(modulations)
    }

    fn dit_block_cached_rotary_with_modulation(
        &self,
        block: usize,
        input: &[f32],
        modulation: &[f32],
        seq_len: usize,
        rotary_cos: &[f32],
        rotary_sin: &[f32],
    ) -> Result<Vec<f32>, JsValue> {
        let prefix = format!("transformer.transformer_blocks.{}", block);
        f5_dit_block_modulation_cached_rotary(
            self.q4(&format!("{}.attn.to_q.weight", prefix))?,
            self.q4(&format!("{}.attn.to_k.weight", prefix))?,
            self.q4(&format!("{}.attn.to_v.weight", prefix))?,
            self.q4(&format!("{}.attn.to_out.0.weight", prefix))?,
            self.q4(&format!("{}.ff.ff.0.0.weight", prefix))?,
            self.q4(&format!("{}.ff.ff.2.weight", prefix))?,
            input,
            modulation,
            seq_len,
            self.dim,
            self.heads,
            self.head_dim,
            1e-6,
            rotary_cos,
            rotary_sin,
        )
    }

    fn final_ada_norm(&self, input: &[f32], t_silu: &[f32], seq_len: usize) -> Result<Vec<f32>, JsValue> {
        let modulation = self.linear("transformer.norm_out.linear.weight", t_silu, 1)?;
        self.final_ada_norm_with_modulation(input, &modulation, seq_len)
    }

    fn final_ada_norm_with_modulation(&self, input: &[f32], modulation: &[f32], seq_len: usize) -> Result<Vec<f32>, JsValue> {
        if modulation.len() < self.dim * 2 {
            return Err(JsValue::from_str("F5 final modulation shape mismatch"));
        }
        let scale = &modulation[0..self.dim];
        let shift = &modulation[self.dim..self.dim * 2];
        layer_norm_affine_f32(input, shift, scale, seq_len, self.dim, 1e-6)
    }

    fn depthwise_conv1d(&self, name: &str, input: &[f32], seq_len: usize, channels: usize, kernel: usize, padding: usize) -> Result<Vec<f32>, JsValue> {
        let handle = self.q4_raw(name)?;
        if seq_len == 0 || channels == 0 || kernel == 0 {
            return Ok(Vec::new());
        }
        if input.len() != seq_len * channels || handle.in_dim != kernel || handle.out_dim < channels {
            return Err(JsValue::from_str("F5 depthwise conv shape mismatch"));
        }
        let mut output = vec![0.0f32; seq_len * channels];
        for ch in 0..channels {
            let scale = handle.row_scales[ch];
            let weight_row = handle.unpacked_row(ch);
            let bias = handle.bias_values.get(ch).copied().unwrap_or(0.0);
            for pos in 0..seq_len {
                let mut sum = bias;
                let k_start = padding.saturating_sub(pos);
                let k_end = kernel.min(seq_len + padding - pos);
                for (k, &w) in weight_row.iter().enumerate().take(k_end).skip(k_start) {
                    let src_pos = pos + k - padding;
                    sum += input[src_pos * channels + ch] * w as f32 * scale;
                }
                output[pos * channels + ch] = sum;
            }
        }
        Ok(output)
    }

    fn grouped_conv1d(&self, name: &str, input: &[f32], seq_len: usize, channels: usize, kernel: usize, padding: usize, groups: usize) -> Result<Vec<f32>, JsValue> {
        let handle = self.q4_raw(name)?;
        if seq_len == 0 || channels == 0 || kernel == 0 || groups == 0 {
            return Ok(Vec::new());
        }
        if channels % groups != 0 {
            return Err(JsValue::from_str("F5 grouped conv channels must be divisible by groups"));
        }
        let group_in = channels / groups;
        if input.len() != seq_len * channels || handle.in_dim != group_in * kernel || handle.out_dim < channels {
            return Err(JsValue::from_str("F5 grouped conv shape mismatch"));
        }
        let mut output = vec![0.0f32; seq_len * channels];
        let mut out_ch = 0usize;
        while out_ch < channels {
            let group = out_ch / group_in;
            let in_start = group * group_in;
            let bias = handle.bias_values.get(out_ch).copied().unwrap_or(0.0);
            let scale = handle.row_scales[out_ch];
            let weight_row = handle.unpacked_row(out_ch);
            if out_ch + 7 < channels && (out_ch + 7) / group_in == group {
                let bias_b = handle.bias_values.get(out_ch + 1).copied().unwrap_or(0.0);
                let bias_c = handle.bias_values.get(out_ch + 2).copied().unwrap_or(0.0);
                let bias_d = handle.bias_values.get(out_ch + 3).copied().unwrap_or(0.0);
                let bias_e = handle.bias_values.get(out_ch + 4).copied().unwrap_or(0.0);
                let bias_f = handle.bias_values.get(out_ch + 5).copied().unwrap_or(0.0);
                let bias_g = handle.bias_values.get(out_ch + 6).copied().unwrap_or(0.0);
                let bias_h = handle.bias_values.get(out_ch + 7).copied().unwrap_or(0.0);
                let scale_b = handle.row_scales[out_ch + 1];
                let scale_c = handle.row_scales[out_ch + 2];
                let scale_d = handle.row_scales[out_ch + 3];
                let scale_e = handle.row_scales[out_ch + 4];
                let scale_f = handle.row_scales[out_ch + 5];
                let scale_g = handle.row_scales[out_ch + 6];
                let scale_h = handle.row_scales[out_ch + 7];
                let weight_b = handle.unpacked_row(out_ch + 1);
                let weight_c = handle.unpacked_row(out_ch + 2);
                let weight_d = handle.unpacked_row(out_ch + 3);
                let weight_e = handle.unpacked_row(out_ch + 4);
                let weight_f = handle.unpacked_row(out_ch + 5);
                let weight_g = handle.unpacked_row(out_ch + 6);
                let weight_h = handle.unpacked_row(out_ch + 7);
                for pos in 0..seq_len {
                    let mut sum = bias;
                    let mut sum_b = bias_b;
                    let mut sum_c = bias_c;
                    let mut sum_d = bias_d;
                    let mut sum_e = bias_e;
                    let mut sum_f = bias_f;
                    let mut sum_g = bias_g;
                    let mut sum_h = bias_h;
                    let k_start = padding.saturating_sub(pos);
                    let k_end = kernel.min(seq_len + padding - pos);
                    for k in k_start..k_end {
                        let src_pos = pos + k - padding;
                        let input_base = src_pos * channels + in_start;
                        for local_in in 0..group_in {
                            let col = local_in * kernel + k;
                            unsafe {
                                let value = *input.get_unchecked(input_base + local_in);
                                sum += value * *weight_row.get_unchecked(col) as f32 * scale;
                                sum_b += value * *weight_b.get_unchecked(col) as f32 * scale_b;
                                sum_c += value * *weight_c.get_unchecked(col) as f32 * scale_c;
                                sum_d += value * *weight_d.get_unchecked(col) as f32 * scale_d;
                                sum_e += value * *weight_e.get_unchecked(col) as f32 * scale_e;
                                sum_f += value * *weight_f.get_unchecked(col) as f32 * scale_f;
                                sum_g += value * *weight_g.get_unchecked(col) as f32 * scale_g;
                                sum_h += value * *weight_h.get_unchecked(col) as f32 * scale_h;
                            }
                        }
                    }
                    let dst = pos * channels + out_ch;
                    unsafe {
                        *output.get_unchecked_mut(dst) = sum;
                        *output.get_unchecked_mut(dst + 1) = sum_b;
                        *output.get_unchecked_mut(dst + 2) = sum_c;
                        *output.get_unchecked_mut(dst + 3) = sum_d;
                        *output.get_unchecked_mut(dst + 4) = sum_e;
                        *output.get_unchecked_mut(dst + 5) = sum_f;
                        *output.get_unchecked_mut(dst + 6) = sum_g;
                        *output.get_unchecked_mut(dst + 7) = sum_h;
                    }
                }
                out_ch += 8;
                continue;
            }
            if out_ch + 3 < channels && (out_ch + 3) / group_in == group {
                let bias_b = handle.bias_values.get(out_ch + 1).copied().unwrap_or(0.0);
                let bias_c = handle.bias_values.get(out_ch + 2).copied().unwrap_or(0.0);
                let bias_d = handle.bias_values.get(out_ch + 3).copied().unwrap_or(0.0);
                let scale_b = handle.row_scales[out_ch + 1];
                let scale_c = handle.row_scales[out_ch + 2];
                let scale_d = handle.row_scales[out_ch + 3];
                let weight_b = handle.unpacked_row(out_ch + 1);
                let weight_c = handle.unpacked_row(out_ch + 2);
                let weight_d = handle.unpacked_row(out_ch + 3);
                for pos in 0..seq_len {
                    let mut sum = bias;
                    let mut sum_b = bias_b;
                    let mut sum_c = bias_c;
                    let mut sum_d = bias_d;
                    let k_start = padding.saturating_sub(pos);
                    let k_end = kernel.min(seq_len + padding - pos);
                    for local_in in 0..group_in {
                        let input_base = in_start + local_in;
                        for k in k_start..k_end {
                            let src_pos = pos + k - padding;
                            let col = local_in * kernel + k;
                            unsafe {
                                let value = *input.get_unchecked(src_pos * channels + input_base);
                                sum += value * *weight_row.get_unchecked(col) as f32 * scale;
                                sum_b += value * *weight_b.get_unchecked(col) as f32 * scale_b;
                                sum_c += value * *weight_c.get_unchecked(col) as f32 * scale_c;
                                sum_d += value * *weight_d.get_unchecked(col) as f32 * scale_d;
                            }
                        }
                    }
                    let dst = pos * channels + out_ch;
                    unsafe {
                        *output.get_unchecked_mut(dst) = sum;
                        *output.get_unchecked_mut(dst + 1) = sum_b;
                        *output.get_unchecked_mut(dst + 2) = sum_c;
                        *output.get_unchecked_mut(dst + 3) = sum_d;
                    }
                }
                out_ch += 4;
                continue;
            }
            if out_ch + 1 < channels && (out_ch + 1) / group_in == group {
                let bias_b = handle.bias_values.get(out_ch + 1).copied().unwrap_or(0.0);
                let scale_b = handle.row_scales[out_ch + 1];
                let weight_b = handle.unpacked_row(out_ch + 1);
                for pos in 0..seq_len {
                    let mut sum = bias;
                    let mut sum_b = bias_b;
                    let k_start = padding.saturating_sub(pos);
                    let k_end = kernel.min(seq_len + padding - pos);
                    for local_in in 0..group_in {
                        let input_base = in_start + local_in;
                        for k in k_start..k_end {
                            let src_pos = pos + k - padding;
                            let col = local_in * kernel + k;
                            let value = input[src_pos * channels + input_base];
                            sum += value * weight_row[col] as f32 * scale;
                            sum_b += value * weight_b[col] as f32 * scale_b;
                        }
                    }
                    let dst = pos * channels + out_ch;
                    output[dst] = sum;
                    output[dst + 1] = sum_b;
                }
                out_ch += 2;
                continue;
            }
            for pos in 0..seq_len {
                let mut sum = bias;
                let k_start = padding.saturating_sub(pos);
                let k_end = kernel.min(seq_len + padding - pos);
                for local_in in 0..group_in {
                    let input_base = in_start + local_in;
                    for k in k_start..k_end {
                        let src_pos = pos + k - padding;
                        let col = local_in * kernel + k;
                        sum += input[src_pos * channels + input_base] * weight_row[col] as f32 * scale;
                    }
                }
                output[pos * channels + out_ch] = sum;
            }
            out_ch += 1;
        }
        Ok(output)
    }
}

struct Q4RawTensor {
    unpacked_weight: Vec<i8>,
    row_scales: Vec<f32>,
    bias_values: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Q4RawTensor {
    fn new(
        packed_weight: &[u8],
        row_scales_f16: &[u16],
        bias_values: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Q4RawTensor, JsValue> {
        if in_dim == 0 || out_dim == 0 {
            return Err(JsValue::from_str("Q4RawTensor dimensions must be non-zero"));
        }
        if packed_weight.len() < (out_dim * in_dim).div_ceil(2) || row_scales_f16.len() < out_dim {
            return Err(JsValue::from_str("Q4RawTensor shape mismatch"));
        }
        let mut unpacked_weight = vec![0i8; out_dim * in_dim];
        for row in 0..out_dim {
            let row_offset = row * in_dim;
            for col in 0..in_dim {
                let index = row_offset + col;
                let packed = packed_weight[index >> 1];
                let nibble = if index & 1 == 0 { packed & 0x0f } else { packed >> 4 };
                let q = nibble as i8;
                unpacked_weight[index] = if q >= 8 { q - 16 } else { q };
            }
        }
        Ok(Q4RawTensor {
            unpacked_weight,
            row_scales: row_scales_f16.iter().take(out_dim).map(|bits| f16_to_f32(*bits)).collect(),
            bias_values: bias_values.to_vec(),
            in_dim,
            out_dim,
        })
    }

    #[inline(always)]
    fn unpacked_row(&self, row: usize) -> &[i8] {
        &self.unpacked_weight[row * self.in_dim..(row + 1) * self.in_dim]
    }
}

fn gelu_erf_scalar(value: f32) -> f32 {
    if value > 6.0 {
        return value;
    }
    if value < -6.0 {
        return 0.0;
    }
    0.5 * value * (1.0 + erf_approx(value / core::f32::consts::SQRT_2))
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-ax * ax).exp();
    sign * y
}

fn mish_scalar(value: f32) -> f32 {
    let softplus = if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    };
    value * softplus.tanh()
}

fn layer_norm_weight_bias(input: &[f32], weight: &[f32], bias: &[f32], rows: usize, cols: usize, eps: f32) -> Result<Vec<f32>, JsValue> {
    if input.len() < rows * cols || weight.len() < cols || bias.len() < cols {
        return Err(JsValue::from_str("layer_norm_weight_bias shape mismatch"));
    }
    let mut output = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let offset = row * cols;
        let values = &input[offset..offset + cols];
        let mean = values.iter().copied().sum::<f32>() / cols as f32;
        let mut variance = 0.0f32;
        for value in values {
            let delta = *value - mean;
            variance += delta * delta;
        }
        let inv = 1.0 / (variance / cols as f32 + eps).sqrt();
        for col in 0..cols {
            output[offset + col] = (input[offset + col] - mean) * inv * weight[col] + bias[col];
        }
    }
    Ok(output)
}

fn add_text_sinusoidal_pos(x: &mut [f32], seq_len: usize, dim: usize) {
    let half = dim / 2;
    for pos in 0..seq_len {
        let base = pos * dim;
        for idx in 0..half {
            let inv = 1.0 / 10000.0f32.powf((2 * idx) as f32 / dim as f32);
            let angle = pos as f32 * inv;
            x[base + idx] += angle.cos();
            x[base + idx + half] += angle.sin();
        }
    }
}

fn apply_grn(x: &mut [f32], seq_len: usize, dim: usize, gamma: &[f32], beta: &[f32]) {
    let mut gx = vec![0.0f32; dim];
    let mut mean = 0.0f32;
    for col in 0..dim {
        let mut sum = 0.0f32;
        for pos in 0..seq_len {
            let value = x[pos * dim + col];
            sum += value * value;
        }
        gx[col] = sum.sqrt();
        mean += gx[col];
    }
    mean = mean / dim as f32 + 1e-6;
    for pos in 0..seq_len {
        let offset = pos * dim;
        for col in 0..dim {
            let value = x[offset + col];
            x[offset + col] = gamma.get(col).copied().unwrap_or(0.0) * (value * (gx[col] / mean)) + beta.get(col).copied().unwrap_or(0.0) + value;
        }
    }
}

fn make_f5_time_grid(steps: usize, sway_sampling_coef: f32) -> Vec<f32> {
    let mut times = vec![0.0f32; steps + 1];
    for idx in 0..=steps {
        let mut t = idx as f32 / steps as f32;
        t = t + sway_sampling_coef * (((core::f32::consts::PI / 2.0) * t).cos() - 1.0 + t);
        times[idx] = t;
    }
    times
}

fn gaussian_vec(length: usize, seed: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; length];
    let mut index = 0usize;
    let mut state = seed;
    while index < length {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u1 = ((state as f64 + 1.0) / 4294967297.0).max(1e-7) as f32;
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u2 = ((state as f64 + 1.0) / 4294967297.0) as f32;
        let mag = (-2.0 * u1.ln()).sqrt();
        out[index] = mag * (2.0 * core::f32::consts::PI * u2).cos();
        if index + 1 < length {
            out[index + 1] = mag * (2.0 * core::f32::consts::PI * u2).sin();
        }
        index += 2;
    }
    out
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

    #[test]
    fn decodes_q4_symmetric_linear() {
        let input = [2.0f32, 3.0, 5.0];
        // Row 0: [-1, 0, 7], row 1: [-8, 2, 1].
        let packed_weight = [
            0x0fu8, 0x07u8,
            0x28u8, 0x01u8,
        ];
        let scales = [0x3800u16, 0x4000u16]; // 0.5, 2.0
        let y = q4_symmetric_linear_f32(&input, &packed_weight, &scales, &[1.0], 1, 3, 2).unwrap();
        assert_eq!(y.len(), 2);
        assert!((y[0] - 17.5).abs() < 1e-6);
        assert!((y[1] + 10.0).abs() < 1e-6);
    }
}
