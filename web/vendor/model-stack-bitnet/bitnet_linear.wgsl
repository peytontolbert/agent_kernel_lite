struct BitNetLinearParams {
  rows: u32,
  in_features: u32,
  out_features: u32,
  padded_in_features: u32,
  scale_granularity: u32,
  scale_group_size: u32,
  segment_count: u32,
  has_bias: u32,
  input_quant_mode: u32,
  input_quant_bits: u32,
  input_scale_rows: u32,
  reserved0: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weight_words: array<u32>;
@group(0) @binding(2) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(3) var<storage, read> segment_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> bias_values: array<f32>;
@group(0) @binding(5) var<storage, read> input_scales: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<uniform> params: BitNetLinearParams;

fn decode_signed_ternary(packed_byte: u32, in_idx: u32) -> f32 {
  let shift = (in_idx & 3u) * 2u;
  let code = (packed_byte >> shift) & 3u;
  if (code == 0u) {
    return -1.0;
  }
  if (code == 2u) {
    return 1.0;
  }
  return 0.0;
}

fn load_packed_byte(out_idx: u32, in_idx: u32) -> u32 {
  let row_stride_bytes = params.padded_in_features / 4u;
  let byte_offset = out_idx * row_stride_bytes + (in_idx / 4u);
  let word = packed_weight_words[byte_offset / 4u];
  let byte_lane = byte_offset & 3u;
  return (word >> (byte_lane * 8u)) & 255u;
}

fn resolve_weight_scale(out_idx: u32) -> f32 {
  if (params.scale_granularity == 0u) {
    return weight_scales[0];
  }
  if (params.scale_granularity == 1u) {
    var seg = 0u;
    loop {
      if (seg >= params.segment_count) {
        break;
      }
      if (out_idx >= segment_offsets[seg] && out_idx < segment_offsets[seg + 1u]) {
        return weight_scales[seg];
      }
      seg = seg + 1u;
    }
    return 0.0;
  }
  if (params.scale_granularity == 2u) {
    return weight_scales[out_idx / params.scale_group_size];
  }
  return 0.0;
}

fn quant_max(bits: u32) -> f32 {
  return f32((1u << (bits - 1u)) - 1u);
}

fn input_value(row: u32, col: u32) -> f32 {
  let value = input[row * params.in_features + col];
  if (params.input_quant_mode == 0u) {
    return value;
  }

  let scale_row = select(row, 0u, params.input_scale_rows == 1u);
  let scale = max(input_scales[scale_row], 0.00000001);
  let qmax = quant_max(params.input_quant_bits);
  let code = clamp(round(value / scale), -qmax, qmax);
  return code * scale;
}

@compute @workgroup_size(8, 8, 1)
fn bitnet_linear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_idx = gid.x;
  let row = gid.y;
  if (out_idx >= params.out_features || row >= params.rows) {
    return;
  }

  var acc = 0.0;
  var col = 0u;
  loop {
    if (col >= params.in_features) {
      break;
    }
    let packed_byte = load_packed_byte(out_idx, col);
    let w = decode_signed_ternary(packed_byte, col);
    acc = acc + input_value(row, col) * w;
    col = col + 1u;
  }

  var y = acc * resolve_weight_scale(out_idx);
  if (params.has_bias != 0u) {
    y = y + bias_values[out_idx];
  }
  output[row * params.out_features + out_idx] = y;
}
