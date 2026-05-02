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

- Source bundle: `/data/agentkernel/artifacts/agentkernel_lite_encdec/recommendation_answer_v11_exact_twopass_from_v9_train_00200_lr2e6`
- Parameters before BitNet packing: `113507328`
- Final eval loss: `0.0016785785555839539`
- Browser entrypoint: `manifest.json`
- Runtime: Model Stack browser BitNet WebGPU encoder-decoder with packed BitNet WASM fallback
- Tokenizer: AgentKernel byte-level BPE attached under `tokenizer/`

Web app route after uploading this directory to Hugging Face:

```text
?modelStackManifest=https://huggingface.co/<org>/<repo>/resolve/main/manifest.json
```

Serving notes: WebGPU is used when available; Safari or other no-WebGPU browsers use the packed BitNet WASM fallback. Large model files are fetched by the browser and cached by the app.
