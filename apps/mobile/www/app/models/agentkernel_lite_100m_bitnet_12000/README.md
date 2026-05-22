---
library_name: model-stack
tags:
- agentkernel-lite
- bitnet
- webgpu
- encoder-decoder
---

# AgentKernel Lite Encoder-Decoder Browser BitNet

Self-contained browser BitNet export for the AgentKernel Lite chat model.

- Source bundle: `/data/agent_kernel_lite/artifacts/pocketpal_controller_100m_v289a_intent_boundary_low_lr_from_v288`
- Parameters before BitNet packing: `102488473`
- Final eval loss: `0.4050685465335846`
- Browser entrypoint: `manifest.json`
- Runtime: Model Stack browser BitNet WebGPU encoder-decoder with packed BitNet WASM fallback
- Tokenizer: AgentKernel byte-level BPE attached under `tokenizer/`

Web app route after uploading this directory to Hugging Face:

```text
?modelStackManifest=https://huggingface.co/<org>/<repo>/resolve/main/manifest.json
```

Serving notes: WebGPU is used when available; Safari or other no-WebGPU browsers use the packed BitNet WASM fallback. Large model files are fetched by the browser and cached by the app.
