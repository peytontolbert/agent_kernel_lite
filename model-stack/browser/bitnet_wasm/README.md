# Model Stack BitNet WASM

This crate provides the browser WebAssembly fallback kernel for Model Stack's
packed BitNet linear format. It executes the same `bitnet_w2a8` packed tensor
layout used by the WebGPU runtime, so browser fallback keeps the compact BitNet
artifact instead of expanding to dense ONNX weights.

Build the SIMD package copied into browser bundles with:

```bash
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --target web --release --out-dir ../bitnet/pkg
```
